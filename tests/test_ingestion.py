from __future__ import annotations

from scripts.generate_synthetic_case import generate_case
from karuth.ingestion.exif_extractor import extract_image_metadata
from karuth.ingestion.normalizer import normalize_chat_carve, normalize_image_metadata
from karuth.ingestion.scanner import scan_case_folder
from karuth.ingestion.sqlite_carver import carve_chat_db
from karuth.ledger import LedgerStore
from karuth.vision.face_extractor import extract_face_entities


def test_scan_case_folder_hashes_files_and_writes_ledger(tmp_path):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "a.txt").write_text("alpha", encoding="utf-8")
    nested = case_dir / "nested"
    nested.mkdir()
    (nested / "b.txt").write_text("beta", encoding="utf-8")
    store = LedgerStore(tmp_path / "ledger.sqlite3")

    hashes = scan_case_folder("CASE-1", case_dir, store)

    entries = store.list_entries()
    assert len(hashes) == 2
    assert len(entries) == 2
    assert entries[0].entry_type == "evidence"
    assert store.verify()


def test_synthetic_case_generator_and_extractors(tmp_path):
    case_dir = tmp_path / "synthetic_case"

    generate_case(case_dir)

    chat_db = case_dir / "chats" / "synthetic_whatsapp.sqlite"
    image_path = case_dir / "images" / "IMG_0004.jpg"
    assert chat_db.exists()
    assert image_path.exists()
    assert (case_dir / "bank" / "transactions.csv").exists()

    chat_entities = normalize_chat_carve(carve_chat_db(chat_db))
    metadata = extract_image_metadata(image_path)
    image_entities = normalize_image_metadata(metadata)
    face_entities = extract_face_entities(image_path)

    assert any(entity.type == "phone" and entity.value == "+919900001001" for entity in chat_entities)
    assert any(entity.type == "vehicle_plate" and entity.value == "KL07CX4242" for entity in chat_entities)
    assert any(entity.type == "camera_serial" for entity in image_entities)
    assert face_entities[0].type == "face_embedding"
    assert len(face_entities[0].value) == 512
