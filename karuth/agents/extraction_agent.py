"""Extraction agent wrapping Phase 2 extractors."""

from __future__ import annotations

from pathlib import Path

from karuth.config import SUPPORTED_IMAGE_SUFFIXES
from karuth.entities.schema import Entity
from karuth.ingestion.exif_extractor import extract_image_metadata
from karuth.ingestion.normalizer import normalize_chat_carve, normalize_image_metadata
from karuth.ingestion.sqlite_carver import carve_chat_db
from karuth.vision.face_extractor import extract_face_entities


def extract_case_entities(case_folder: Path) -> list[Entity]:
    entities: list[Entity] = []
    for db_path in sorted(case_folder.rglob("*.sqlite")):
        entities.extend(normalize_chat_carve(carve_chat_db(db_path)))
    for image_path in sorted(
        path
        for suffix in SUPPORTED_IMAGE_SUFFIXES
        for path in case_folder.rglob(f"*{suffix}")
    ):
        entities.extend(normalize_image_metadata(extract_image_metadata(image_path)))
        entities.extend(extract_face_entities(image_path))
    return entities
