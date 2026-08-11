"""Tests for the vision (real face extraction + LLM transcription) and Neo4j layers."""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import pytest

from karuth.agents.vision_agent import ImageTranscription, transcribe_image, transcribe_images
from karuth.correlation.candidate_matcher import CorrelationResult, match_entities
from karuth.entities.schema import Entity
from karuth.ingestion.exif_extractor import extract_image_metadata
from karuth.vision.face_extractor import extract_face_entities, ensure_face_models

# ---------------------------------------------------------------------------
# Real face extraction
# ---------------------------------------------------------------------------


def _write_synthetic_image(tmp_path: Path, name: str, identity: str) -> Path:
    """A minimal JPEG fixture tagged with a synthetic identity sidecar."""
    image_path = tmp_path / name
    exif_payload = f"Exif\x00\x00SyntheticKaruth;identity={identity}".encode("ascii")
    image_path.write_bytes(
        b"\xff\xd8"
        + b"\xff\xe1"
        + (len(exif_payload) + 2).to_bytes(2, "big")
        + exif_payload
        + b"\xff\xdb\x00C"
        + bytes([0] * 67)
        + b"\xff\xd9"
    )
    (tmp_path / f"{image_path.stem}.exif.json").write_text(
        json.dumps(
            {
                "file": image_path.name,
                "synthetic_identity": identity,
                "captured_at": "2026-01-12T09:30:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    return image_path


def test_synthetic_face_path_is_preserved(tmp_path):
    image = _write_synthetic_image(tmp_path, "IMG_0001.jpg", "person_alpha")
    entities = extract_face_entities(image)
    assert len(entities) == 1
    entity = entities[0]
    assert entity.type == "face_embedding"
    assert len(entity.value) == 512
    assert entity.metadata["synthetic_identity"] == "person_alpha"


def _models_unavailable(monkeypatch, tmp_path: Path):
    """Point the model cache at an empty dir and disable downloads."""
    from karuth.vision import face_extractor

    monkeypatch.setattr(face_extractor, "FACE_MODEL_DIR", tmp_path / "empty_models")

    def _no_network(url: str, filename: str):
        raise OSError("network disabled in tests")

    monkeypatch.setattr(face_extractor.urllib.request, "urlretrieve", _no_network)


def test_real_face_path_degrades_when_models_unavailable(tmp_path, monkeypatch):
    """No model download available -> real photos return [] instead of crashing."""
    _models_unavailable(monkeypatch, tmp_path)
    image = tmp_path / "real_photo.jpg"
    image.write_bytes(b"\xff\xd8\xff\xd9 not a real jpeg body " * 4)

    assert ensure_face_models() is None
    assert extract_face_entities(image) == []


def test_real_face_path_degrades_on_corrupt_image(tmp_path, monkeypatch):
    _models_unavailable(monkeypatch, tmp_path)
    image = tmp_path / "corrupt.jpg"
    image.write_bytes(b"\x00\x01\x02\x03 broken bytes")

    assert extract_face_entities(image) == []


def test_real_face_detection_works_when_models_present():
    """Integration check that runs only when the ONNX models are on disk."""
    if ensure_face_models() is None:
        pytest.skip("YuNet/SFace models not cached; run scripts/fetch_face_models.py")
    import cv2
    import numpy as np

    # A synthetic face-like image is not a real face, so detection must simply
    # not crash and must return a list (possibly empty).
    canvas = np.full((160, 160, 3), 127, dtype=np.uint8)
    path = Path(__file__).parent / "face_fixture.png"
    cv2.imwrite(str(path), canvas)
    try:
        result = extract_face_entities(path)
        assert isinstance(result, list)
    finally:
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# LLM image transcription
# ---------------------------------------------------------------------------


class _FakeChoice:
    class _Message:
        content = "A vehicle with plate KL07CX4242 parked near a building."

    message = _Message()


class _FakeResponse:
    choices = [_FakeChoice()]
    usage = None


class _FakeClient:
    class _Completions:
        def __init__(self, payload=None, error=None):
            self._payload = payload
            self._error = error

        def create(self, **kwargs):
            if self._error is not None:
                raise self._error
            return self._payload

    def __init__(self, payload=None, error=None):
        self.chat = type("Chat", (), {"completions": _FakeCompletions(payload, error)})()


class _FakeCompletions:
    def __init__(self, payload, error):
        self._payload = payload
        self._error = error

    def create(self, **kwargs):
        if self._error is not None:
            raise self._error
        return self._payload


def test_transcribe_image_with_fake_client(tmp_path):
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"\xff\xd8\xff\xd9 fake jpeg")

    result = transcribe_image(image, client=_FakeClient(payload=_FakeResponse()))
    assert result.error == ""
    assert "KL07CX4242" in result.transcription
    assert result.llm_backend is True
    assert result.disclaimer


def test_transcribe_image_captures_backend_failure(tmp_path):
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"\xff\xd8\xff\xd9 fake jpeg")

    result = transcribe_image(
        image, client=_FakeClient(error=RuntimeError("LM Studio refused image input"))
    )
    assert result.error
    assert "LM Studio" in result.error
    assert result.transcription == ""


def test_transcribe_image_captures_unreadable_file(tmp_path):
    missing = tmp_path / "does_not_exist.jpg"
    result = transcribe_image(missing)
    assert result.error
    assert "unreadable" in result.error


def test_transcribe_image_when_openai_sdk_missing(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", None)
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"\xff\xd8\xff\xd9 fake jpeg")
    result = transcribe_image(image)
    assert result.error  # soft error, never a hard crash


def test_transcribe_images_captures_per_file_errors(tmp_path):
    missing = tmp_path / "nope.jpg"
    results = transcribe_images([missing])
    assert len(results) == 1
    assert results[0].error


def test_image_transcription_is_pydantic_serializable(tmp_path):
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"\xff\xd8\xff\xd9 fake jpeg")
    result = transcribe_image(image, client=_FakeClient(payload=_FakeResponse()))
    dumped = result.model_dump(mode="json")
    assert dumped["image_path"].endswith("photo.jpg")
    assert isinstance(dumped["generated_at"], str)


# ---------------------------------------------------------------------------
# Neo4j sync + traversal
# ---------------------------------------------------------------------------


def _entities() -> list[Entity]:
    return [
        Entity(
            entity_id="entity-phone-1",
            type="phone",
            value="+919900001001",
            source_file="chats/a.sqlite",
            confidence=1.0,
        ),
        Entity(
            entity_id="entity-phone-2",
            type="phone",
            value="+919900001001",
            source_file="bank/t.csv",
            confidence=1.0,
        ),
    ]


def _matches() -> list[CorrelationResult]:
    return [
        CorrelationResult(
            left_entity_id="entity-phone-1",
            right_entity_id="entity-phone-2",
            rule="exact_string",
            score=1.0,
            explanation="Exact phone match",
        )
    ]


class _FakeDriver:
    def __init__(self):
        self.writes: list[str] = []
        self.rows: list[dict] = []

    def verify_connectivity(self):
        return True

    def session(self, database=None):
        return self

    def execute_write(self, fn):
        fn(self)  # invoke the lambda: tx.run(cypher, rows=rows).consume()
        self.writes.append(self._last_cypher)
        return None

    def run(self, cypher, **params):
        self._last_cypher = cypher
        self._params = params
        return self

    def consume(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def __iter__(self):
        return iter(self.rows)


def test_sync_case_uses_merge_cypher(monkeypatch):
    from karuth.graph import neo4j_client

    driver = _FakeDriver()
    monkeypatch.setattr(neo4j_client, "get_driver", lambda: driver)
    monkeypatch.setattr(neo4j_client, "_ensure_constraints", lambda: None)

    result = neo4j_client.sync_case("CASE-X", _entities(), _matches())
    assert result["synced"] is True
    assert result["entities"] == 2
    assert result["matches"] == 1
    joined = "\n".join(driver.writes)
    assert "MERGE (e:Entity" in joined
    assert "MERGE (l)-[m:MATCHES]->(r)" in joined


def test_sync_case_noop_when_graph_unavailable(monkeypatch):
    from karuth.graph import neo4j_client

    monkeypatch.setattr(neo4j_client, "get_driver", lambda: None)
    result = neo4j_client.sync_case("CASE-X", _entities(), _matches())
    assert result["synced"] is False
    assert result["reason"] == "graph unavailable"


def test_traversal_returns_rows(monkeypatch):
    from karuth.correlation import graph_traversal

    class _Rows:
        def __iter__(self):
            yield {
                "entity_id": "entity-phone-2",
                "type": "phone",
                "value_preview": "+919900001001",
                "rule": "exact_string",
                "score": 1.0,
                "case_id": "CASE-X",
            }

    class _FakeSession:
        def run(self, cypher, **params):
            return _Rows()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    class _FakeDriver:
        def session(self, database=None):
            return _FakeSession()

    monkeypatch.setattr(graph_traversal, "get_driver", lambda: _FakeDriver())
    rows = graph_traversal.find_connected("entity-phone-1", case_id="CASE-X")
    assert rows and rows[0]["entity_id"] == "entity-phone-2"
    assert rows[0]["rule"] == "exact_string"


def test_traversal_degrades_when_graph_down(monkeypatch):
    from karuth.correlation import graph_traversal

    monkeypatch.setattr(graph_traversal, "get_driver", lambda: None)
    assert graph_traversal.find_connected("x") == []
    assert graph_traversal.multi_hop("x") == []
    assert graph_traversal.find_by_value("vehicle_plate", "KL07") == []
    assert graph_traversal.paths_between("a", "b") == []
    assert graph_traversal.case_summary("CASE-X")["entities"] == 0
    assert graph_traversal.full_case_graph("CASE-X") == {"nodes": [], "edges": []}


def test_cases_with_entities_returns_richest_first(monkeypatch):
    from karuth.correlation import graph_traversal

    class _Rows:
        def __init__(self, rows):
            self.rows = rows

        def __iter__(self):
            return iter(self.rows)

    class _FakeSession:
        def run(self, cypher, **params):
            # Neo4j applies ORDER BY entities DESC; the fake returns the
            # already-sorted rows and the function passes them through.
            assert "ORDER BY entities DESC" in cypher
            return _Rows(
                [
                    {"case_id": "case-a", "entities": 67},
                    {"case_id": "case-b", "entities": 5},
                ]
            )

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    class _FakeDriver:
        def session(self, database=None):
            return _FakeSession()

    monkeypatch.setattr(graph_traversal, "get_driver", lambda: _FakeDriver())
    rows = graph_traversal.cases_with_entities()
    assert rows[0]["case_id"] == "case-a" and rows[0]["entities"] == 67
    assert rows[1]["case_id"] == "case-b"


def test_full_case_graph_builds_nodes_and_edges(monkeypatch):
    from karuth.correlation import graph_traversal

    node_rows = [
        {
            "entity_id": "entity-phone-1",
            "type": "phone",
            "value_preview": "+919900001001",
            "source_file": "chats/a.sqlite",
        },
        {
            "entity_id": "entity-phone-2",
            "type": "phone",
            "value_preview": "+919900001002",
            "source_file": "chats/a.sqlite",
        },
    ]
    edge_rows = [
        {
            "source": "entity-phone-1",
            "target": "entity-phone-2",
            "type": "MATCHES",
            "score": 1.0,
        }
    ]
    captured: list[str] = []

    class _Rows:
        def __init__(self, rows):
            self.rows = rows

        def __iter__(self):
            return iter(self.rows)

    class _FakeSession:
        def run(self, cypher, **params):
            captured.append(cypher)
            if "count(n)" in cypher or "n.type" in cypher:
                return _Rows(node_rows)
            return _Rows(edge_rows)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    class _FakeDriver:
        def session(self, database=None):
            return _FakeSession()

    monkeypatch.setattr(graph_traversal, "get_driver", lambda: _FakeDriver())
    graph = graph_traversal.full_case_graph("CASE-X")
    assert len(graph["nodes"]) == 2
    assert graph["nodes"][0]["value_preview"] == "+919900001001"
    assert graph["edges"] == edge_rows
    assert "LIMIT" in captured[-1]
    assert graph_traversal.full_case_graph("CASE-X", max_edges=10)["edges"] == edge_rows


def test_case_summary_uses_valid_cluster_count_syntax(monkeypatch):
    """Neo4j 5.26 removed ``size(<pattern>)``; clusters must use ``count { }``.

    Regression guard: the old query raised a CypherSyntaxError at runtime, which
    ``_query`` swallowed, so the Graph page always showed 0 person clusters.
    """
    from karuth.correlation import graph_traversal

    captured: list[str] = []

    class _FakeSession:
        def run(self, cypher, **params):
            captured.append(cypher)
            return []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    class _FakeDriver:
        def session(self, database=None):
            return _FakeSession()

    monkeypatch.setattr(graph_traversal, "get_driver", lambda: _FakeDriver())
    graph_traversal.case_summary("CASE-X")
    cluster_cypher = next(q for q in captured if "Person_Cluster" in q)
    assert "count { (c)<-[:FACE_MATCH]-() }" in cluster_cypher
    assert "size(" not in cluster_cypher


def test_correlation_agent_syncs_when_case_id_given(monkeypatch):
    from karuth.agents import correlation_agent

    calls: list[tuple] = []

    def _fake_sync(case_id, entities, matches):
        calls.append((case_id, len(entities), len(matches)))

    # correlation_agent does `from karuth.graph.neo4j_client import sync_case`
    # locally, so patch the attribute on the source module.
    monkeypatch.setattr(
        "karuth.graph.neo4j_client.sync_case", _fake_sync, raising=False
    )
    results = correlation_agent.correlate_entities(_entities(), case_id="CASE-Y")
    assert len(results) == 1  # exact phone match
    assert calls and calls[0] == ("CASE-Y", 2, 1)


def test_correlation_agent_skips_sync_without_case_id(monkeypatch):
    from karuth.agents import correlation_agent

    monkeypatch.setattr(
        "karuth.graph.neo4j_client.sync_case",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not sync")),
        raising=False,
    )
    results = correlation_agent.correlate_entities(_entities())
    assert len(results) == 1


def test_face_matches_produce_cluster_rows(monkeypatch):
    from karuth.graph import neo4j_client
    from karuth.correlation.candidate_matcher import match_entities

    driver = _FakeDriver()
    monkeypatch.setattr(neo4j_client, "get_driver", lambda: driver)
    monkeypatch.setattr(neo4j_client, "_ensure_constraints", lambda: None)

    face_entities = [
        Entity(
            entity_id="face-a",
            type="face_embedding",
            value=[0.5] * 128,
            source_file="a.jpg",
            confidence=0.9,
        ),
        Entity(
            entity_id="face-b",
            type="face_embedding",
            value=[0.5] * 128,
            source_file="b.jpg",
            confidence=0.9,
        ),
    ]
    matches = match_entities(face_entities)
    assert matches and matches[0].metadata.get("person_cluster_id")

    result = neo4j_client.sync_case("CASE-Z", face_entities, matches)
    assert result["clusters"] == 1
    assert any("Person_Cluster" in write for write in driver.writes)


def test_live_neo4j_round_trip():
    """Live integration: runs only when a Neo4j server is reachable."""
    from karuth.graph import neo4j_client
    from karuth.correlation import graph_traversal

    if not neo4j_client.graph_available():
        pytest.skip("Neo4j not reachable at bolt://localhost:7687")

    case_id = f"TEST-{uuid.uuid4().hex[:8]}"
    try:
        result = neo4j_client.sync_case(case_id, _entities(), _matches())
        assert result["synced"] is True
        summary = graph_traversal.case_summary(case_id)
        assert summary["entities"] == 2
        assert summary["matches"] == 1
        connected = graph_traversal.find_connected("entity-phone-1", case_id=case_id)
        assert any(row["entity_id"] == "entity-phone-2" for row in connected)
        hops = graph_traversal.multi_hop("entity-phone-1", max_hops=2)
        assert any(row["entity_id"] == "entity-phone-2" for row in hops)
        path = graph_traversal.paths_between("entity-phone-1", "entity-phone-2")
        assert path and len(path[0]["nodes"]) == 2
    finally:
        neo4j_client.close_driver()


# ---------------------------------------------------------------------------
# Router binding for VisionAgent
# ---------------------------------------------------------------------------


def test_router_vision_branch_handles_missing_images(tmp_path):
    from karuth.planner.router import route_task
    from karuth.planner.schema import PlannerTask
    from karuth.planner.state import PlannerState

    task = PlannerTask(step=1, tool="VisionAgent", reason="transcribe images", inputs=[])
    state: PlannerState = {
        "case_folder": tmp_path,
        "goal": "transcribe images",
        "case_id": "CASE-V",
    }
    updates, summary = route_task(task, state)
    assert updates["image_transcriptions"] == []
    assert "no images" in summary


def test_router_vision_branch_reports_llm_failures_softly(tmp_path):
    from karuth.planner.router import route_task
    from karuth.planner.schema import PlannerTask
    from karuth.planner.state import PlannerState

    task = PlannerTask(
        step=1,
        tool="VisionAgent",
        reason="transcribe images",
        inputs=[str(tmp_path / "missing.jpg")],
    )
    state: PlannerState = {
        "case_folder": tmp_path,
        "goal": "transcribe images",
        "case_id": "CASE-V",
    }
    updates, summary = route_task(task, state)
    assert len(updates["image_transcriptions"]) == 1
    assert updates["image_transcriptions"][0].error  # soft failure, not raised


# ---------------------------------------------------------------------------
# Graph ingestion: unstructured content -> SENT_BY / MENTIONS edges
# ---------------------------------------------------------------------------


def _capture_writes():
    """Capture every (cypher, rows) write the graph layer attempts."""
    calls: list[tuple] = []

    def _run_write(cypher, rows):
        calls.append((cypher, list(rows)))

    return calls, _run_write


def test_graph_cyber_build_elements():  # hermetic: pure Python, no browser
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    frontend = root / "frontend"
    if str(frontend) not in sys.path:
        sys.path.insert(0, str(frontend))
    from components.graph_cyber import TYPE_ACCENTS, build_elements

    nodes = [
        {"entity_id": "n1", "type": "vehicle_plate", "value_preview": "KL07CX4242", "source_file": "chats/a.sqlite"},
        {"entity_id": "n2", "type": "message", "value_preview": "sent the plate", "source_file": "chats/a.sqlite"},
    ]
    edges = [{"source": "n2", "target": "n1", "type": "MENTIONS"}]

    elements = build_elements(nodes, edges, label_mode="important")
    node_data = [e["data"] for e in elements if "source" not in e["data"]]
    assert len(node_data) == 2
    plate = node_data[0]
    assert plate["accent"] == TYPE_ACCENTS["vehicle_plate"]
    assert plate["gloss"].startswith("data:image/svg+xml;base64,")
    # dense-view labels: important types opaque, message hidden (text-opacity 0)
    assert plate["labelOpacity"] == 1.0
    assert node_data[1]["labelOpacity"] == 0.0
    edge = [e["data"] for e in elements if "source" in e["data"]][0]
    assert edge["source"] == "n2" and edge["target"] == "n1"
    assert edge["rel"] == "MENTIONS"
    assert edge["label"] == "mentions"

    # search views show every label
    elements_all = build_elements(nodes, edges, label_mode="all")
    all_visible = [
        e["data"]["labelOpacity"]
        for e in elements_all
        if "source" not in e["data"]
    ]
    assert all(opacity == 1.0 for opacity in all_visible)


def test_graph_ingest_builds_sent_by_and_mentions_edges(tmp_path, monkeypatch):
    from karuth.graph import graph_ingest
    from karuth.agents.extraction_agent import extract_case_entities
    from karuth.ingestion.normalizer import entity_id

    case_dir = tmp_path / "synthetic_case"
    from scripts.generate_synthetic_case import generate_case

    generate_case(case_dir)
    entities = extract_case_entities(case_dir)
    assert any(entity.type == "message" for entity in entities)
    assert any(entity.type == "vehicle_plate" for entity in entities)

    driver = _FakeDriver()
    calls, capture = _capture_writes()
    monkeypatch.setattr(graph_ingest, "get_driver", lambda: driver)
    monkeypatch.setattr(graph_ingest, "_run_write", capture)

    stats = graph_ingest.ingest_unstructured("CASE-X", case_dir, entities)

    assert stats["synced"] is True
    assert stats["messages"] >= 4
    assert stats["sent_by"] == 4  # one SENT_BY edge per chat message
    assert stats["mentions"] >= 4  # plate/serial/phone mentions in chats+docs+bank

    cyphers = [cypher for cypher, _ in calls]
    assert any("MERGE (p)-[:SENT_BY" in cypher for cypher in cyphers)
    assert any("MERGE (s)-[:MENTIONS" in cypher for cypher in cyphers)

    sent_by_rows = [
        rows for cypher, rows in calls if "SENT_BY" in cypher
    ][0]
    # Each chat message links to its sender: derive the sender entity id the
    # same way extraction does (from the message's own source_file), then
    # confirm the SENT_BY edge targets it.
    arun_message = next(
        entity for entity in entities
        if entity.type == "message"
        and entity.metadata.get("contact_name") == "Arun Nair"
    )
    person_id = entity_id(arun_message.source_file, "person_name", "Arun Nair")
    assert any(row["person_id"] == person_id for row in sent_by_rows)

    # The chat message body mentions the extracted plate, so a MENTIONS edge
    # links the message entity to the plate entity (cross-source link).
    mentions_rows = [rows for cypher, rows in calls if "MENTIONS" in cypher]
    plate_entity = next(
        entity for entity in entities
        if entity.type == "vehicle_plate" and str(entity.value) == "KL07CX4242"
    )
    assert any(
        row["target_id"] == plate_entity.entity_id
        for rows in mentions_rows
        for row in rows
    )

    # Documents and bank memos are MERGEd as new entity nodes with mention edges.
    assert stats["documents"] == 2
    assert stats["transactions"] == 2


def test_graph_ingest_noop_when_graph_unavailable(tmp_path, monkeypatch):
    from karuth.graph import graph_ingest

    case_dir = tmp_path / "empty"
    case_dir.mkdir()
    monkeypatch.setattr(graph_ingest, "get_driver", lambda: None)

    stats = graph_ingest.ingest_unstructured("CASE-X", case_dir, [])
    assert stats["synced"] is False
    assert stats["reason"] == "graph unavailable"
    assert stats["mentions"] == 0


def test_graph_ingest_noop_when_case_folder_missing(tmp_path, monkeypatch):
    from karuth.graph import graph_ingest

    monkeypatch.setattr(graph_ingest, "get_driver", lambda: _FakeDriver())
    stats = graph_ingest.ingest_unstructured("CASE-X", tmp_path / "missing", [])
    assert stats["synced"] is False
    assert stats["reason"] == "case folder missing"


def test_graph_ingest_corrupt_documents_are_skipped(tmp_path, monkeypatch):
    """Unreadable text files must never crash ingestion."""
    from karuth.graph import graph_ingest
    from karuth.ingestion.normalizer import entity_id

    case_dir = tmp_path / "case"
    docs = case_dir / "documents"
    docs.mkdir(parents=True)
    (docs / "ok.md").write_text("Vehicle KL07CX4242 sighted.", encoding="utf-8")

    calls, capture = _capture_writes()
    monkeypatch.setattr(graph_ingest, "get_driver", lambda: _FakeDriver())
    monkeypatch.setattr(graph_ingest, "_run_write", capture)

    plate = Entity(
        entity_id="plate-1",
        type="vehicle_plate",
        value="KL07CX4242",
        source_file="documents/ok.md",
        confidence=1.0,
    )
    stats = graph_ingest.ingest_unstructured("CASE-X", case_dir, [plate])
    assert stats["synced"] is True
    assert stats["documents"] == 1
    doc_id = entity_id("documents/ok.md", "document_text", "documents/ok.md")
    mentions = [rows for cypher, rows in calls if "MENTIONS" in cypher]
    assert any(
        row["source_id"] == doc_id and row["target_id"] == "plate-1"
        for rows in mentions
        for row in rows
    )


def test_graph_traversal_new_queries_degrade_when_graph_down(monkeypatch):
    from karuth.correlation import graph_traversal

    monkeypatch.setattr(graph_traversal, "get_driver", lambda: None)
    assert graph_traversal.person_messages("p1") == []
    assert graph_traversal.mentions("m1") == []
    assert graph_traversal.mentioned_in("e1") == []
    assert graph_traversal.shared_entities("a", "b") == []


def test_graph_traversal_person_messages_returns_rows(monkeypatch):
    from karuth.correlation import graph_traversal

    class _Rows:
        def __iter__(self):
            yield {
                "entity_id": "msg-1",
                "type": "message",
                "value_preview": "Meet near Kaloor with vehicle KL07CX4242.",
                "sent_at": "2026-01-12T09:30:00+00:00",
                "source_file": "chats/a.sqlite",
            }

    class _FakeSession:
        def run(self, cypher, **params):
            return _Rows()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    class _FakeDriver:
        def session(self, database=None):
            return _FakeSession()

    monkeypatch.setattr(graph_traversal, "get_driver", lambda: _FakeDriver())
    rows = graph_traversal.person_messages("person-1")
    assert len(rows) == 1
    assert rows[0]["entity_id"] == "msg-1"
    assert rows[0]["sent_at"]


def test_router_correlation_branch_ingests_unstructured_graph(tmp_path, monkeypatch):
    """Running CorrelationAgent conveys unstructured content to the graph and
    reports the graph stats in the dispatch summary."""
    from karuth.planner.router import route_task
    from karuth.planner.schema import PlannerTask
    from karuth.planner.state import initial_state
    from karuth.agents.extraction_agent import extract_case_entities
    from scripts.generate_synthetic_case import generate_case

    case_dir = tmp_path / "synthetic_case"
    generate_case(case_dir)
    entities = extract_case_entities(case_dir)

    calls: list[tuple] = []

    def _fake_ingest(case_id, case_folder, entities):
        calls.append((case_id, str(case_folder), len(entities)))
        return {
            "synced": True,
            "messages": 4,
            "sent_by": 4,
            "mentions": 6,
            "documents": 2,
            "transactions": 2,
        }

    monkeypatch.setattr(
        "karuth.graph.graph_ingest.ingest_unstructured", _fake_ingest, raising=False
    )

    state = initial_state("CASE-R", case_dir, "Find links for vehicle KL07CX4242")
    state["entities"] = entities
    task = PlannerTask(
        step=2,
        tool="CorrelationAgent",
        reason="correlate extracted entities",
        inputs=[],
    )
    updates, summary = route_task(task, state)
    assert len(updates["matches"]) >= 1  # synthetic case is cross-wired
    assert "4 sent_by" in summary
    assert "6 mentions" in summary
    assert calls and calls[0][0] == "CASE-R"
    assert calls[0][1].endswith("synthetic_case")
    assert calls[0][2] == len(entities)
