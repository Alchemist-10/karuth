"""Deterministic contradiction hunting.

The contradiction hunter finds *inconsistent facts* across extracted entities
and case artifacts. It never invents contradictions: every result carries
deterministic evidence references (entity ids and/or artifact paths) and a
rule name. The LLM is not involved — these checks are pure, offline, and
reproducible.

Rules implemented today:

- ``impossible_movement``: the same vehicle plate has two sightings (image GPS
  or a chat message mentioning a known city) that are far apart within a short
  time window — an implausible travel timeline.
- ``inconsistent_identity_vehicle``: the same vehicle plate appears in images
  carrying different synthetic person identities.
- ``inconsistent_identity_camera``: the same camera serial appears in images
  carrying different synthetic person identities.
- ``metadata_mismatch``: embedded JPEG EXIF and the synthetic sidecar for the
  same image disagree on capture time or GPS.
- ``ambiguous_plate``: a number-plate reading has alternative OCR candidates
  that cannot be excluded without human confirmation.

Future rules (extension points) live behind the same ``hunt_contradictions``
interface so the planner can discover them without changing the loop.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timedelta
from typing import Any, Literal

from karuth.correlation.candidate_matcher import CorrelationResult
from karuth.entities.schema import Entity
from karuth.planner.hypothesis import (
    MEDIUM_CONTRADICTION_DELTA,
    STRONG_CONTRADICTION_DELTA,
    WEAK_CONTRADICTION_DELTA,
)

Severity = Literal["low", "medium", "high"]

_PLATE_RE = re.compile(r"\b[A-Z]{2}[A-Z0-9-]*\d{4}\b")

# Small city gazetteer used to turn chat mentions into geo sightings. Only
# used for deterministic contradiction checks; not a general extraction tool.
_GAZETTEER: dict[str, tuple[float, float]] = {
    "delhi": (28.6139, 77.2090),
    "kochi": (9.9312, 76.2673),
    "kaloor": (9.9884, 76.2734),
    "mumbai": (19.0760, 72.8777),
    "chennai": (13.0827, 80.2707),
    "bengaluru": (12.9716, 77.5946),
}

MAX_MOVEMENT_DELTA_MINUTES = 60
MIN_MOVEMENT_DISTANCE_KM = 100.0

CONTRADICTION_DELTA: dict[str, float] = {
    "high": STRONG_CONTRADICTION_DELTA,
    "medium": MEDIUM_CONTRADICTION_DELTA,
    "low": WEAK_CONTRADICTION_DELTA,
}


class Contradiction:
    """A deterministic conflict between two claims over shared evidence."""

    def __init__(
        self,
        *,
        rule: str,
        claim_a: str,
        claim_b: str,
        evidence_refs: list[str],
        severity: Severity = "medium",
        confidence: float = 0.8,
    ) -> None:
        self.rule = rule
        self.claim_a = claim_a
        self.claim_b = claim_b
        self.evidence_refs = evidence_refs
        self.severity = severity
        self.confidence = confidence
        self.resolution_status = "unresolved"
        # Deterministic id over the rule + refs so the same conflict is stable.
        self.id = _contradiction_id(rule, evidence_refs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "rule": self.rule,
            "claim_a": self.claim_a,
            "claim_b": self.claim_b,
            "evidence_refs": self.evidence_refs,
            "severity": self.severity,
            "confidence": self.confidence,
            "resolution_status": self.resolution_status,
        }


def _contradiction_id(rule: str, refs: list[str]) -> str:
    import hashlib

    material = "|".join([rule, *sorted(set(refs))])
    return "C-" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:8].upper()


def _haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def _timestamp(entity: Entity) -> datetime | None:
    for key in ("sent_at", "captured_at", "timestamp"):
        value = entity.metadata.get(key)
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                continue
    return None


# ---------------------------------------------------------------------------
# sighting extraction (deterministic)
# ---------------------------------------------------------------------------


def _message_sightings(entities: list[Entity]) -> list[dict[str, Any]]:
    """Sightings from chat messages: plate + city mentioned in the same body."""
    sightings: list[dict[str, Any]] = []
    for entity in entities:
        if entity.type != "message":
            continue
        body = str(entity.value)
        plates = _PLATE_RE.findall(body)
        if not plates:
            continue
        sent_at = _timestamp(entity)
        for city, coords in _GAZETTEER.items():
            if city.lower() in body.lower():
                for plate in plates:
                    sightings.append(
                        {
                            "plate": plate,
                            "time": sent_at,
                            "coords": coords,
                            "source": entity.source_file,
                            "refs": [entity.entity_id],
                            "label": f"{city} (chat message)",
                        }
                    )
    return sightings


def _image_sightings(entities: list[Entity]) -> list[dict[str, Any]]:
    """Sightings from images: plate + GPS location sharing a source file."""
    by_file: dict[str, list[Entity]] = {}
    for entity in entities:
        by_file.setdefault(entity.source_file, []).append(entity)

    sightings: list[dict[str, Any]] = []
    for source_file, group in by_file.items():
        plates = [e for e in group if e.type == "vehicle_plate"]
        locations = [e for e in group if e.type == "location"]
        if not plates or not locations:
            continue
        captured_at = _timestamp(plates[0])
        for plate_entity in plates:
            for loc_entity in locations:
                gps = loc_entity.value
                if not (isinstance(gps, dict) and "lat" in gps and "lon" in gps):
                    continue
                sightings.append(
                    {
                        "plate": str(plate_entity.value),
                        "time": captured_at,
                        "coords": (float(gps["lat"]), float(gps["lon"])),
                        "source": source_file,
                        "refs": [plate_entity.entity_id, loc_entity.entity_id],
                        "label": f"image GPS ({gps['lat']:.2f}, {gps['lon']:.2f})",
                    }
                )
    return sightings


def _identity_per_file(entities: list[Entity]) -> dict[str, set[str]]:
    """source_file -> set of synthetic identities appearing in its faces."""
    by_file: dict[str, set[str]] = {}
    for entity in entities:
        if entity.type != "face_embedding":
            continue
        identity = entity.metadata.get("synthetic_identity")
        if not identity:
            continue
        by_file.setdefault(entity.source_file, set()).add(str(identity))
    return by_file


def _same_value_identity_conflicts(
    entities: list[Entity],
    value_type: str,
    rule: str,
    label: str,
) -> list[Contradiction]:
    """Same identifier value on images tagged with different identities."""
    identities = _identity_per_file(entities)
    by_value: dict[str, list[tuple[str, str]]] = {}
    for entity in entities:
        if entity.type != value_type:
            continue
        value = str(entity.value)
        person = identities.get(entity.source_file)
        if person:
            by_value.setdefault(value, []).append((entity.source_file, ",".join(sorted(person))))

    conflicts: list[Contradiction] = []
    for value, entries in by_value.items():
        distinct_people = {person for _, person in entries if person}
        if len(distinct_people) < 2:
            continue
        refs = [f"{source}::{person}" for source, person in entries]
        conflicts.append(
            Contradiction(
                rule=rule,
                claim_a=f"{label} {value} is associated with identity {min(distinct_people)}",
                claim_b=f"{label} {value} is associated with identity {max(distinct_people)}",
                evidence_refs=refs,
                severity="medium",
                confidence=0.85,
            )
        )
    return conflicts


def _metadata_mismatch_contradictions(entities: list[Entity]) -> list[Contradiction]:
    """Embedded EXIF disagrees with the synthetic sidecar for one image.

    The extractor marks merged metadata ``provenance == "conflicting"``; this
    rule surfaces that as a real contradiction citing the affected artifacts.
    """
    seen: set[str] = set()
    findings: list[Contradiction] = []
    for entity in entities:
        if entity.metadata.get("exif_provenance") != "conflicting":
            continue
        source = entity.source_file
        if source in seen:
            continue
        seen.add(source)
        findings.append(
            Contradiction(
                rule="metadata_mismatch",
                claim_a=(
                    f"{entity.type} metadata for {source} comes from the synthetic "
                    "sidecar"
                ),
                claim_b="the embedded JPEG EXIF for the same image disagrees",
                evidence_refs=[entity.entity_id, source],
                severity="medium",
                confidence=0.7,
            )
        )
    return findings


def _ambiguous_plate_contradictions(
    observations: list[Any] | None,
) -> list[Contradiction]:
    """A plate reading with alternative OCR candidates is an unresolved ambiguity."""
    if not observations:
        return []
    from karuth.planner.extensions import ImageObservation
    from karuth.planner.observations import normalize_observations

    findings: list[Contradiction] = []
    for observation in normalize_observations(observations):
        if observation.observation_type != "number_plate":
            continue
        if not observation.requires_human_confirmation:
            continue
        alternatives = ", ".join(observation.alternatives)
        findings.append(
            Contradiction(
                rule="ambiguous_plate",
                claim_a=f"The number plate reading is {observation.value}",
                claim_b=(
                    f"alternative OCR readings ({alternatives}) cannot be excluded "
                    "without human confirmation"
                ),
                evidence_refs=[observation.artifact_ref, observation.observation_id],
                severity="low",
                confidence=0.6,
            )
        )
    return findings


def hunt_contradictions(
    entities: list[Entity],
    matches: list[CorrelationResult],
    *,
    max_movement_delta_minutes: int = MAX_MOVEMENT_DELTA_MINUTES,
    min_movement_distance_km: float = MIN_MOVEMENT_DISTANCE_KM,
    observations: list[Any] | None = None,
) -> list[Contradiction]:
    """Run every deterministic contradiction rule over extracted entities.

    Returns an empty list when no conflict exists — never an invented one.
    ``matches`` is accepted for interface stability; movement/identity rules
    operate on raw entities so the hunter works before or after correlation.
    ``observations`` (optional multimodal observations) enable the
    ``ambiguous_plate`` and ``metadata_mismatch`` rules.
    """
    findings: list[Contradiction] = []

    sightings = _message_sightings(entities) + _image_sightings(entities)
    by_plate: dict[str, list[dict[str, Any]]] = {}
    for sighting in sightings:
        if sighting["time"] is not None:
            by_plate.setdefault(sighting["plate"], []).append(sighting)

    for plate, group in by_plate.items():
        # Collapse the pairwise conflicts for one plate into its single most
        # implausible pair, so the report reads "one movement contradiction"
        # instead of N near-identical ones.
        worst: tuple[dict, dict, float, timedelta] | None = None
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                left, right = group[i], group[j]
                delta = abs(left["time"] - right["time"])
                if delta > timedelta(minutes=max_movement_delta_minutes):
                    continue
                distance = _haversine_km(left["coords"], right["coords"])
                if distance < min_movement_distance_km:
                    continue
                if worst is None or distance > worst[2]:
                    worst = (left, right, distance, delta)
        if worst is not None:
            left, right, distance, delta = worst
            refs = list(dict.fromkeys([*left["refs"], *right["refs"]]))
            findings.append(
                Contradiction(
                    rule="impossible_movement",
                    claim_a=(
                        f"Vehicle {plate} was sighted at {left['label']} at "
                        f"{left['time'].isoformat()}"
                    ),
                    claim_b=(
                        f"Vehicle {plate} was sighted at {right['label']} at "
                        f"{right['time'].isoformat()} ({distance:.0f} km apart in "
                        f"{delta.total_seconds() / 60:.0f} minutes)"
                    ),
                    evidence_refs=refs,
                    severity="high",
                    confidence=0.9,
                )
            )

    findings.extend(
        _same_value_identity_conflicts(
            entities,
            value_type="vehicle_plate",
            rule="inconsistent_identity_vehicle",
            label="Vehicle plate",
        )
    )
    findings.extend(
        _same_value_identity_conflicts(
            entities,
            value_type="camera_serial",
            rule="inconsistent_identity_camera",
            label="Camera serial",
        )
    )
    findings.extend(_metadata_mismatch_contradictions(entities))
    findings.extend(_ambiguous_plate_contradictions(observations))

    # Stable ordering so reports are reproducible.
    return sorted(findings, key=lambda c: (c.rule, c.id))
