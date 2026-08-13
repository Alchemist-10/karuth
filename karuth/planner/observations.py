"""Deterministic multimodal observation builder.

Turns image artifacts (embedded EXIF + synthetic sidecar metadata) into a list
of structured :class:`ImageObservation` objects — the "inspect multimodal
observations" stage of the dynamic investigation. Every observation carries:

- an observation id and artifact reference,
- a typed observation (person / vehicle / number_plate / gps_location /
  timestamp / camera_serial / visible_object / sensitive_content /
  activity_interpretation),
- a confidence (``None`` = "Not scored" — missing metadata is never 0),
- alternative readings for uncertain plate OCR,
- an evidence classification (Deterministic fact / Metadata fact / LLM visual
  interpretation / Candidate association / ...), and
- whether human review is required before the observation may be relied on.

Everything here is offline and deterministic; the LLM vision model is only a
*possible* source for ``activity_interpretation`` observations, which are
always labeled ``LLM visual interpretation`` and always require review.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

from karuth.config import SUPPORTED_IMAGE_SUFFIXES
from karuth.ingestion.exif_extractor import extract_image_metadata
from karuth.planner.extensions import (
    EVIDENCE_CLASSIFICATIONS,
    OBSERVATION_TYPES,
    ImageObservation,
)


def observation_id(artifact_ref: str, observation_type: str) -> str:
    material = f"{artifact_ref}|{observation_type}"
    return "OBS-" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:10].upper()


def _rel(root: Path, path: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def build_image_observations(
    case_folder: Path,
    image_paths: Iterable[Path] | None = None,
) -> list[ImageObservation]:
    """Build structured observations for the images in ``case_folder``.

    ``image_paths`` may be supplied to observe a subset (used by tests); by
    default every supported image in the case folder is observed.
    """
    root = Path(case_folder).resolve()
    if image_paths is None:
        image_paths = sorted(
            path
            for suffix in SUPPORTED_IMAGE_SUFFIXES
            for path in root.rglob(f"*{suffix}")
        )

    observations: list[ImageObservation] = []
    for image in image_paths:
        image = Path(image)
        try:
            metadata = extract_image_metadata(image)
        except Exception:  # noqa: BLE001 - corrupt files degrade to no observations
            metadata = {}
        artifact_ref = _rel(root, image) if root in image.resolve().parents else str(image)
        meta_conf = metadata.get("metadata_confidence")

        identity = metadata.get("synthetic_identity")
        if identity:
            observations.append(
                ImageObservation(
                    observation_id=observation_id(artifact_ref, "person"),
                    artifact_ref=artifact_ref,
                    image_path=str(image),
                    observation_type="person",
                    value=f"candidate subject (synthetic fixture: {identity})",
                    description=(
                        "A candidate subject is visible; the fixture identity label "
                        "must not be treated as real identification without human review."
                    ),
                    confidence=0.6,
                    evidence_classification="Candidate association",
                    model_used="synthetic_identity_metadata",
                    requires_human_review=True,
                )
            )

        visible_vehicle = metadata.get("visible_vehicle")
        if visible_vehicle and str(visible_vehicle).lower() not in ("", "none"):
            observations.append(
                ImageObservation(
                    observation_id=observation_id(artifact_ref, "vehicle"),
                    artifact_ref=artifact_ref,
                    image_path=str(image),
                    observation_type="vehicle",
                    value=str(visible_vehicle),
                    description="A vehicle is visible in the image.",
                    confidence=0.7,
                    evidence_classification="LLM visual interpretation",
                    model_used="synthetic_scene_metadata",
                    requires_human_review=True,
                )
            )

        plate = metadata.get("vehicle_plate")
        if plate:
            alternatives = list(metadata.get("plate_alternatives") or [])
            plate_conf = metadata.get("plate_confidence")
            confidence = plate_conf if plate_conf is not None else (meta_conf if meta_conf is not None else 0.8)
            observations.append(
                ImageObservation(
                    observation_id=observation_id(artifact_ref, "number_plate"),
                    artifact_ref=artifact_ref,
                    image_path=str(image),
                    observation_type="number_plate",
                    value=str(plate),
                    description=(
                        "Possible number plate reading; the reading may be ambiguous."
                    ),
                    confidence=round(float(confidence), 3),
                    alternatives=alternatives,
                    evidence_classification="Metadata fact",
                    model_used="synthetic_plate_metadata",
                    requires_human_review=bool(alternatives) or float(confidence) < 0.85,
                )
            )
        else:
            # No sidecar/EXIF plate: try local OCR when Tesseract is installed.
            # OCR candidates are always uncertain and require human review.
            try:
                from karuth.vision.ocr import read_plate_candidates

                ocr_candidates = read_plate_candidates(image)
            except Exception:  # noqa: BLE001 - OCR must never break the pipeline
                ocr_candidates = []
            if ocr_candidates:
                best = ocr_candidates[0]
                observations.append(
                    ImageObservation(
                        observation_id=observation_id(artifact_ref, "number_plate"),
                        artifact_ref=artifact_ref,
                        image_path=str(image),
                        observation_type="number_plate",
                        value=str(best["value"]),
                        description=(
                            "Plate text read from the photo by OCR; the reading is "
                            "uncertain and requires human confirmation."
                        ),
                        confidence=round(float(best["confidence"]), 3),
                        alternatives=list(best.get("alternatives") or []),
                        evidence_classification="Candidate association",
                        model_used="ocr (tesseract)",
                        requires_human_review=True,
                    )
                )

        gps = metadata.get("gps")
        if isinstance(gps, dict) and "lat" in gps and "lon" in gps:
            observations.append(
                ImageObservation(
                    observation_id=observation_id(artifact_ref, "gps_location"),
                    artifact_ref=artifact_ref,
                    image_path=str(image),
                    observation_type="gps_location",
                    value=f"{gps['lat']:.4f}, {gps['lon']:.4f}",
                    description=(
                        "GPS metadata describes the image/camera capture — it does "
                        "not by itself prove the subject was present at that location."
                    ),
                    confidence=meta_conf if meta_conf is not None else 0.95,
                    evidence_classification="Metadata fact",
                    model_used="exif_metadata",
                )
            )

        captured_at = metadata.get("captured_at")
        if captured_at:
            observations.append(
                ImageObservation(
                    observation_id=observation_id(artifact_ref, "timestamp"),
                    artifact_ref=artifact_ref,
                    image_path=str(image),
                    observation_type="timestamp",
                    value=str(captured_at),
                    description="Capture timestamp from image metadata.",
                    confidence=meta_conf if meta_conf is not None else 0.95,
                    evidence_classification="Metadata fact",
                    model_used="exif_metadata",
                )
            )

        camera_serial = metadata.get("camera_serial")
        if camera_serial:
            observations.append(
                ImageObservation(
                    observation_id=observation_id(artifact_ref, "camera_serial"),
                    artifact_ref=artifact_ref,
                    image_path=str(image),
                    observation_type="camera_serial",
                    value=str(camera_serial),
                    description="Camera serial recorded in image metadata.",
                    confidence=meta_conf if meta_conf is not None else 0.95,
                    evidence_classification="Metadata fact",
                    model_used="exif_metadata",
                )
            )

        landmark = metadata.get("landmark")
        if landmark:
            observations.append(
                ImageObservation(
                    observation_id=observation_id(artifact_ref, "visible_object"),
                    artifact_ref=artifact_ref,
                    image_path=str(image),
                    observation_type="visible_object",
                    value=str(landmark),
                    description="Distinctive landmark visible in the scene (location context).",
                    confidence=0.9,
                    evidence_classification="Metadata fact",
                    model_used="synthetic_scene_metadata",
                )
            )

        if metadata.get("sensitive_content") is True:
            observations.append(
                ImageObservation(
                    observation_id=observation_id(artifact_ref, "sensitive_content"),
                    artifact_ref=artifact_ref,
                    image_path=str(image),
                    observation_type="sensitive_content",
                    value="sensitive or traumatic content flagged",
                    description=(
                        "Sensitive content detected. Preview mode: blurred. "
                        "Original preserved securely; reveal requires investigator "
                        "authorization."
                    ),
                    confidence=None,
                    evidence_classification="Metadata fact",
                    model_used="synthetic_sensitivity_flag",
                    requires_human_review=True,
                )
            )

        interpretation = metadata.get("activity_interpretation")
        if interpretation:
            interpretation_conf = metadata.get("interpretation_confidence")
            observations.append(
                ImageObservation(
                    observation_id=observation_id(artifact_ref, "activity_interpretation"),
                    artifact_ref=artifact_ref,
                    image_path=str(image),
                    observation_type="activity_interpretation",
                    value=str(interpretation),
                    description=(
                        "LLM visual interpretation of what is visibly present. It is "
                        "not an extracted fact and does not imply intent or conduct; "
                        "investigator review is required."
                    ),
                    confidence=(
                        round(float(interpretation_conf), 3)
                        if interpretation_conf is not None
                        else 0.5
                    ),
                    evidence_classification="LLM visual interpretation",
                    model_used="llm_vision (assistance only)",
                    requires_human_review=True,
                )
            )

    return observations


def ambiguous_plate_observations(
    observations: list[ImageObservation],
) -> list[ImageObservation]:
    """Plate observations whose reading needs human confirmation."""
    return [
        observation
        for observation in observations
        if observation.observation_type == "number_plate"
        and observation.requires_human_confirmation
    ]


def sensitive_observations(
    observations: list[ImageObservation],
) -> list[ImageObservation]:
    return [
        observation
        for observation in observations
        if observation.observation_type == "sensitive_content"
    ]


def _observation_from_dict(data: dict) -> ImageObservation | None:
    """Coerce a dict back into an ImageObservation (UI/state tolerance)."""
    try:
        return ImageObservation.model_validate(data)
    except (TypeError, ValueError):
        return None


def normalize_observations(
    observations: list,
) -> list[ImageObservation]:
    """Accept pydantic models or dicts and return ImageObservation models."""
    normalized: list[ImageObservation] = []
    for item in observations:
        if isinstance(item, ImageObservation):
            normalized.append(item)
        elif isinstance(item, dict):
            parsed = _observation_from_dict(item)
            if parsed is not None:
                normalized.append(parsed)
    return normalized


__all__ = [
    "EVIDENCE_CLASSIFICATIONS",
    "OBSERVATION_TYPES",
    "ImageObservation",
    "ambiguous_plate_observations",
    "build_image_observations",
    "normalize_observations",
    "observation_id",
    "sensitive_observations",
]
