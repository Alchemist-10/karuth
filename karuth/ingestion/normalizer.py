"""Normalize extractor outputs into canonical entities."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable

from karuth.entities.schema import Entity

PHONE_RE = re.compile(r"\+?\d{10,15}")
PLATE_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}\b")


def _entity_id(*parts: object) -> str:
    material = "|".join(str(part) for part in parts)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


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


def normalize_image_metadata(metadata: dict[str, Any]) -> list[Entity]:
    source_file = str(metadata["source_file"])
    entities: list[Entity] = []
    if "camera_serial" in metadata:
        entities.append(
            Entity(
                entity_id=_entity_id(source_file, "camera_serial", metadata["camera_serial"]),
                type="camera_serial",
                value=metadata["camera_serial"],
                source_file=source_file,
                confidence=1.0,
                metadata={"captured_at": metadata.get("captured_at")},
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
                metadata={"captured_at": metadata.get("captured_at")},
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
                metadata={"captured_at": metadata.get("captured_at")},
            )
        )
    return entities


def normalize_many(groups: Iterable[list[Entity]]) -> list[Entity]:
    entities: list[Entity] = []
    for group in groups:
        entities.extend(group)
    return entities
