from __future__ import annotations

from pathlib import Path

from scripts.generate_synthetic_case import generate_case
from karuth.correlation.candidate_matcher import match_entities
from karuth.entities.schema import Entity
from karuth.ingestion.exif_extractor import extract_image_metadata
from karuth.ingestion.normalizer import normalize_chat_carve, normalize_image_metadata
from karuth.ingestion.sqlite_carver import carve_chat_db
from karuth.vision.face_extractor import extract_face_entities


def _entities_from_case(case_dir: Path) -> list[Entity]:
    entities = normalize_chat_carve(carve_chat_db(case_dir / "chats" / "synthetic_whatsapp.sqlite"))
    for image_path in sorted((case_dir / "images").glob("*.jpg")):
        entities.extend(normalize_image_metadata(extract_image_metadata(image_path)))
        entities.extend(extract_face_entities(image_path))
    return entities


def test_deterministic_matches_find_plate_timestamp_and_face_cluster(tmp_path):
    case_dir = tmp_path / "synthetic_case"
    generate_case(case_dir)
    entities = _entities_from_case(case_dir)

    matches = match_entities(entities)

    assert any(match.rule == "exact_string" and "KL07CX4242" in match.explanation for match in matches)
    assert any(match.rule == "timestamp_proximity" for match in matches)
    face_matches = [match for match in matches if match.rule == "face_embedding_cosine"]
    assert face_matches
    assert all(match.metadata["person_cluster_id"].startswith("Person_Cluster_") for match in face_matches)


def test_fuzzy_plate_match_with_typo():
    entities = [
        Entity(
            entity_id="left",
            type="vehicle_plate",
            value="KL07CX4242",
            source_file="chat.sqlite",
            confidence=0.98,
        ),
        Entity(
            entity_id="right",
            type="vehicle_plate",
            value="KL07CX424Z",
            source_file="image.jpg",
            confidence=0.86,
        ),
    ]

    matches = match_entities(entities)

    assert any(match.rule == "fuzzy_string" for match in matches)


def test_different_synthetic_faces_stay_unclustered(tmp_path):
    case_dir = tmp_path / "synthetic_case"
    generate_case(case_dir)
    alpha = extract_face_entities(case_dir / "images" / "IMG_0003.jpg")[0]
    beta = extract_face_entities(case_dir / "images" / "IMG_0004.jpg")[0]

    matches = match_entities([alpha, beta])

    assert not [match for match in matches if match.rule == "face_embedding_cosine"]
