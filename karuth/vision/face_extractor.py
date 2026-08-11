"""Local face embedding extraction.

Two paths, both fully offline:

1. Synthetic metadata path (backward compatible): images tagged with a
   ``synthetic_identity`` EXIF sidecar (as produced by the synthetic case
   generator) get a deterministic 512-d embedding derived from the identity
   label. This keeps the demo case hermetic without real ML.

2. Real pixel path: real photos (no synthetic label) are processed locally
   with OpenCV DNN -- YuNet face detection + SFace recognition embeddings.
   The two small ONNX models are downloaded on first use into the model cache
   (``FACE_MODEL_DIR``, override with ``KARUTH_MODEL_DIR``). If cv2/DNN or the
   models are unavailable, extraction degrades gracefully to ``[]`` so the
   deterministic pipeline never breaks.
"""

from __future__ import annotations

import hashlib
import json
import random
import urllib.request
from pathlib import Path

from karuth.config import FACE_MODEL_DIR, SFACE_MODEL_URL, YUNET_MODEL_URL
from karuth.entities.schema import Entity
from karuth.ingestion.exif_extractor import extract_image_metadata

_EMBEDDING_DIMENSIONS = 512
_SFACE_DIMENSIONS = 128
_FACE_DETECTION_SCORE_THRESHOLD = 0.6


def _embedding_for_identity(identity: str) -> list[float]:
    """Deterministic 512-d embedding for the synthetic identity label path."""
    rng = random.Random(hashlib.sha256(identity.encode("utf-8")).hexdigest())
    vector = [rng.uniform(-1.0, 1.0) for _ in range(_EMBEDDING_DIMENSIONS)]
    norm = sum(value * value for value in vector) ** 0.5
    return [round(value / norm, 8) for value in vector]


def _synthetic_entities(image_path: Path, metadata: dict, identity: str) -> list[Entity]:
    """Entities for the synthetic-metadata path (unchanged behavior)."""
    embedding = _embedding_for_identity(str(identity))
    entity_id = hashlib.sha256(
        f"{image_path}|face_embedding|{identity}".encode("utf-8")
    ).hexdigest()[:24]
    return [
        Entity(
            entity_id=entity_id,
            type="face_embedding",
            value=embedding,
            source_file=str(image_path),
            confidence=0.99,
            metadata={
                "synthetic_identity": identity,
                "bounding_box": [16, 16, 96, 96],
                "embedding_dimensions": _EMBEDDING_DIMENSIONS,
                "raw_metadata": json.dumps(metadata, sort_keys=True),
            },
        )
    ]


def model_cache_dir() -> Path:
    """Directory where the YuNet/SFace ONNX models are cached."""
    path = Path(FACE_MODEL_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _download(url: str, dest: Path) -> bool:
    """Download one model file; returns True when the file is usable."""
    if dest.exists() and dest.stat().st_size > 10_000:
        return True
    try:
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)  # noqa: S310 - pinned local-model URLs
        tmp.replace(dest)
        return dest.stat().st_size > 10_000
    except OSError:
        return False


def ensure_face_models() -> tuple[Path, Path] | None:
    """Download (and verify) the YuNet detector and SFace recognizer.

    Returns ``(yunet_path, sface_path)`` when both load as ONNX models, or
    ``None`` when they cannot be obtained. The check is cheap (parse only,
    no image inference), so callers can probe on every extraction.
    """
    cache = model_cache_dir()
    yunet = cache / "face_detection_yunet_2023mar.onnx"
    sface = cache / "sface.onnx"

    if not (_download(YUNET_MODEL_URL, yunet) and _download(SFACE_MODEL_URL, sface)):
        return None

    try:
        import cv2  # lazy import: the synthetic path never needs it

        cv2.dnn.readNetFromONNX(str(yunet))
        cv2.dnn.readNetFromONNX(str(sface))
    except Exception:  # noqa: BLE001 - model corruption should degrade, not crash
        return None
    return yunet, sface


def _real_face_entities(image_path: Path, metadata: dict) -> list[Entity]:
    """Pixel-based face extraction via YuNet detection + SFace embeddings."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return []

    if not (hasattr(cv2, "FaceDetectorYN") and hasattr(cv2, "FaceRecognizerSF")):
        return []

    models = ensure_face_models()
    if models is None:
        return []
    yunet_path, sface_path = models

    try:
        data = np.fromfile(str(image_path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            return []
        height, width = image.shape[:2]

        detector = cv2.FaceDetectorYN.create(
            str(yunet_path),
            "",
            (width, height),
            _FACE_DETECTION_SCORE_THRESHOLD,
            0.3,
            5000,
        )
        detector.setInputSize((width, height))
        _ok, faces = detector.detect(image)
        if faces is None or len(faces) == 0:
            return []

        recognizer = cv2.FaceRecognizerSF.create(str(sface_path), "")
        entities: list[Entity] = []
        for index, face in enumerate(faces):
            score = float(face[14])  # YuNet row ends with the detection score
            aligned = recognizer.alignCrop(image, face)
            feature = recognizer.feature(aligned)
            embedding = [round(float(value), 8) for value in feature[0]]
            entity_id = hashlib.sha256(
                f"{image_path}|face_embedding|face_{index}".encode("utf-8")
            ).hexdigest()[:24]
            entities.append(
                Entity(
                    entity_id=entity_id,
                    type="face_embedding",
                    value=embedding,
                    source_file=str(image_path),
                    confidence=round(score, 3),
                    metadata={
                        "bounding_box": [round(float(v), 2) for v in face[:4]],
                        "face_model": "yunet+sface",
                        "embedding_dimensions": _SFACE_DIMENSIONS,
                        "raw_metadata": json.dumps(metadata, sort_keys=True),
                    },
                )
            )
        return entities
    except Exception:  # noqa: BLE001 - corrupt evidence must degrade, not crash
        return []


def extract_face_entities(image_path: Path) -> list[Entity]:
    """Extract face embeddings from an image.

    Images tagged with ``synthetic_identity`` use the deterministic synthetic
    path; all other readable images use the real YuNet+SFace pixel pipeline.
    Any failure (missing model, corrupt file, no cv2) degrades to ``[]``.
    """
    path = Path(image_path)
    try:
        metadata = extract_image_metadata(path)
    except Exception:  # noqa: BLE001 - corrupt evidence must degrade, not crash
        metadata = {}

    identity = metadata.get("synthetic_identity")
    if identity:
        return _synthetic_entities(path, metadata, str(identity))
    return _real_face_entities(path, metadata)
