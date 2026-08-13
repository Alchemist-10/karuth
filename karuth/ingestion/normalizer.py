"""Normalize extractor outputs into canonical entities."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable

from karuth.entities.schema import Entity

PHONE_RE = re.compile(r"\+?\d{10,15}")
# Compact (KL01CC2121 / KL01cc2121) and hyphenated (NM-JP-4242) plates.
PLATE_RE = re.compile(
    r"\b[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,3}\s?\d{3,4}\b"
    r"|\b[A-Z]{2}[- ][A-Z]{2}[- ]?\d{3,4}\b",
    re.IGNORECASE,
)


def normalize_plate(value: object) -> str:
    """Uppercase and strip whitespace (keep hyphens) for plate comparison."""
    return re.sub(r"\s+", "", str(value or "")).upper()


def entity_id(*parts: object) -> str:
    """Deterministic entity id shared by extraction and graph ingestion.

    The id is a truncated sha256 over the source material so the same value
    from the same file always produces the same node, no matter which layer
    (extraction, correlation sync, or graph ingestion) references it.
    """
    material = "|".join(str(part) for part in parts)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


_entity_id = entity_id  # backward-compatible alias for existing callers


def normalize_chat_carve(carve: dict[str, Any]) -> list[Entity]:
    source_file = str(carve["source_file"])
    entities: list[Entity] = []
    contact_names: dict[int, str] = {}
    for contact in carve["contacts"]:
        contact_id = int(contact["contact_id"])
        contact_names[contact_id] = str(contact["display_name"])
        for entity_type, value in (
            ("person_name", contact["display_name"]),
            ("phone", contact["phone"]),
            ("username", contact["username"]),
        ):
            entities.append(
                Entity(
                    entity_id=_entity_id(source_file, entity_type, value),
                    type=entity_type,
                    value=value,
                    source_file=source_file,
                    confidence=1.0,
                    metadata={"contact_id": contact_id},
                )
            )
    for message in carve["messages"]:
        body = str(message["body"])
        entities.append(
            Entity(
                entity_id=_entity_id(source_file, "message", message["message_id"]),
                type="message",
                value=body,
                source_file=source_file,
                confidence=1.0,
                metadata={
                    "message_id": message["message_id"],
                    "sent_at": message["sent_at"],
                    "contact_name": contact_names.get(int(message["contact_id"])),
                },
            )
        )
        for plate in PLATE_RE.findall(body):
            entities.append(
                Entity(
                    entity_id=_entity_id(source_file, "vehicle_plate", plate, message["message_id"]),
                    type="vehicle_plate",
                    value=plate,
                    source_file=source_file,
                    confidence=0.98,
                    metadata={"sent_at": message["sent_at"]},
                )
            )
    return entities


def _metadata_trail(metadata: dict[str, Any]) -> dict[str, Any]:
    """Carry EXIF provenance + confidence onto every derived entity so the
    contradiction hunter can detect embedded-vs-sidecar metadata conflicts."""
    trail: dict[str, Any] = {}
    if metadata.get("provenance"):
        trail["exif_provenance"] = metadata["provenance"]
    if metadata.get("metadata_confidence") is not None:
        trail["metadata_confidence"] = metadata["metadata_confidence"]
    if metadata.get("source_hash"):
        trail["source_hash"] = metadata["source_hash"]
    if metadata.get("plate_alternatives"):
        trail["plate_alternatives"] = metadata["plate_alternatives"]
    if metadata.get("plate_confidence") is not None:
        trail["plate_confidence"] = metadata["plate_confidence"]
    return trail


def normalize_image_metadata(metadata: dict[str, Any]) -> list[Entity]:
    source_file = str(metadata["source_file"])
    entities: list[Entity] = []
    trail = _metadata_trail(metadata)
    if "camera_serial" in metadata:
        entities.append(
            Entity(
                entity_id=_entity_id(source_file, "camera_serial", metadata["camera_serial"]),
                type="camera_serial",
                value=metadata["camera_serial"],
                source_file=source_file,
                confidence=1.0,
                metadata={"captured_at": metadata.get("captured_at"), **trail},
            )
        )
    if "vehicle_plate" in metadata:
        entities.append(
            Entity(
                entity_id=_entity_id(source_file, "vehicle_plate", metadata["vehicle_plate"]),
                type="vehicle_plate",
                value=metadata["vehicle_plate"],
                source_file=source_file,
                confidence=0.93,
                metadata={"captured_at": metadata.get("captured_at"), **trail},
            )
        )
    if "gps" in metadata:
        entities.append(
            Entity(
                entity_id=_entity_id(source_file, "location", metadata["gps"]),
                type="location",
                value=metadata["gps"],
                source_file=source_file,
                confidence=1.0,
                metadata={"captured_at": metadata.get("captured_at"), **trail},
            )
        )
    return entities


def normalize_many(groups: Iterable[list[Entity]]) -> list[Entity]:
    entities: list[Entity] = []
    for group in groups:
        entities.extend(group)
    return entities
