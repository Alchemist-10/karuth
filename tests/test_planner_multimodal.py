"""Multimodal Jesse Pinkman demo tests.

Covers: embedded/sidecar/missing/corrupted/conflicting EXIF, GPS + timestamp
normalization, synthetic face matching across three images, vehicle/plate
extraction, image observation serialization, alternative plate readings,
human confirmation gates, activity interpretation labeling, contradiction
detection, skeptic review after contradictions + targeting, contradiction-aware
finalization, sensitive-image blur, graph provenance + edge styles, final
report ordering, both demo presets, Streamlit state normalization (pydantic +
dict hypotheses), classic planner backward compatibility, ReportAgent gating,
and full audit-ledger verification.

All people and data are synthetic fixtures (jesse_pinkman is a fictional
case label; no real private individuals are used).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

from scripts.generate_synthetic_case import (
    JESSE_GOAL,
    JESSE_IDENTITY,
    JESSE_PLATE,
    generate_case,
    generate_jesse_case,
)
from karuth.agents.correlation_agent import correlate_entities
from karuth.agents.extraction_agent import extract_case_entities
from karuth.agents.hypothesis_agent import generate_hypotheses
from karuth.entities.schema import Entity
from karuth.ingestion.exif_extractor import extract_image_metadata
from karuth.ingestion.normalizer import normalize_image_metadata
from karuth.ledger import LedgerStore
from karuth.ledger.schema import LedgerEntryCreate
from karuth.planner.actions import ActionDecision, validate_action
from karuth.planner.contradictions import Contradiction, hunt_contradictions
from karuth.planner.hypothesis import Hypothesis
from karuth.planner.llm import DeterministicPlanner
from karuth.planner.observations import (
    ambiguous_plate_observations,
    build_image_observations,
)
from karuth.planner.router import route_task
from karuth.planner.schema import PlannerTask
from karuth.planner.skeptic import run_case_skeptic
from karuth.planner.state import initial_state
from karuth.planner.state_machine import run_planner_goal
from karuth.vision.face_extractor import extract_face_entities


def _jesse_case(tmp_path, *, adversarial: bool = False) -> Path:
    case_dir = tmp_path / ("adv" if adversarial else "clean")
    generate_jesse_case(case_dir, adversarial=adversarial)
    return case_dir


def _run_dynamic(case_dir: Path, ledger: LedgerStore, *, goal: str = JESSE_GOAL, max_actions: int | None = None):
    return run_planner_goal(
        "CASE-J",
        case_dir,
        goal,
        ledger,
        planner=DeterministicPlanner(),
        mode="dynamic",
        max_actions=max_actions,
    )


# ---------------------------------------------------------------------------
# EXIF extraction
# ---------------------------------------------------------------------------


def test_embedded_exif_parsing(tmp_path):
    case_dir = _jesse_case(tmp_path)
    img3 = case_dir / "images" / "IMG_003_vehicle.jpg"
    metadata = extract_image_metadata(img3)
    # clean variant merges embedded EXIF + sidecar consistently
    assert metadata["provenance"] in ("merged", "embedded", "sidecar")
    assert metadata["source_hash"]
    assert metadata["gps"] == {"lat": 35.0844, "lon": -106.6504}
    assert "captured_at" in metadata
    assert metadata.get("camera_make") == "KaruthSyntheticCam"
    assert metadata.get("camera_model") == "KaruthCam-1"
    assert metadata.get("orientation") == 1
    assert metadata["metadata_confidence"] is not None


def test_embedded_only_exif(tmp_path):
    """A JPEG with embedded EXIF and NO sidecar -> provenance embedded."""
    from karuth.planner.observations import build_image_observations

    case_dir = _jesse_case(tmp_path)
    img3 = case_dir / "images" / "IMG_003_vehicle.jpg"
    (img3.with_suffix(".exif.json")).unlink()
    metadata = extract_image_metadata(img3)
    assert metadata["provenance"] == "embedded"
    assert metadata["source"] == "embedded JPEG EXIF"
    assert metadata["gps"] == {"lat": 35.0844, "lon": -106.6504}
    assert metadata["metadata_confidence"] is not None


def test_sidecar_exif_parsing(tmp_path):
    """A fixture JPEG (no real EXIF) + sidecar -> provenance sidecar."""
    case_dir = _jesse_case(tmp_path)
    img1 = case_dir / "images" / "IMG_001_suspicious.jpg"
    metadata = extract_image_metadata(img1)
    assert metadata["provenance"] in ("sidecar", "merged")
    assert metadata["synthetic_identity"] == JESSE_IDENTITY
    assert metadata["metadata_confidence"] is not None


def test_missing_exif_is_not_zero_confidence(tmp_path):
    """No sidecar and no embedded EXIF -> unavailable, confidence None (not 0)."""
    image = tmp_path / "plain.jpg"
    image.write_bytes(b"\xff\xd8\xff\xd9 not a jpeg")
    metadata = extract_image_metadata(image)
    assert metadata["provenance"] == "unavailable"
    assert metadata["metadata_confidence"] is None  # never 0.0
    assert metadata["source_hash"]


def test_corrupted_exif_degrades(tmp_path):
    """A corrupt APP1 block must not crash parsing."""
    image = tmp_path / "corrupt.jpg"
    # APP1 declares a huge length, then the file ends -> PIL raises internally.
    image.write_bytes(b"\xff\xd8\xff\xe1\xff\xff\x00\x00junk\xff\xd9")
    metadata = extract_image_metadata(image)
    assert metadata["provenance"] in ("unavailable", "sidecar")
    assert metadata["metadata_confidence"] is None or metadata["metadata_confidence"] > 0


def test_conflicting_embedded_and_sidecar(tmp_path):
    """Adversarial IMG_003: embedded EXIF (NY / 09:48) conflicts with the
    sidecar (Albuquerque / 09:42); sidecar wins but the conflict is recorded."""
    case_dir = _jesse_case(tmp_path, adversarial=True)
    metadata = extract_image_metadata(case_dir / "images" / "IMG_003_vehicle.jpg")
    assert metadata["provenance"] == "conflicting"
    assert "gps" in metadata["conflicts"]
    assert "captured_at" in metadata["conflicts"]
    # sidecar wins for determinism
    assert metadata["gps"] == {"lat": 35.0844, "lon": -106.6504}
    assert metadata["metadata_confidence"] == 0.6


def test_gps_and_timestamp_normalization(tmp_path):
    case_dir = _jesse_case(tmp_path)
    metadata = extract_image_metadata(case_dir / "images" / "IMG_003_vehicle.jpg")
    assert abs(metadata["gps"]["lat"] - 35.0844) < 0.001
    assert abs(metadata["gps"]["lon"] - -106.6504) < 0.001
    assert metadata["captured_at"].startswith("2026-01-12T09:42")


def test_old_synthetic_case_metadata_backward_compatible(tmp_path):
    case_dir = tmp_path / "case"
    generate_case(case_dir)
    image_path = case_dir / "images" / "IMG_0004.jpg"
    metadata = extract_image_metadata(image_path)
    assert metadata["camera_serial"]
    assert metadata["vehicle_plate"]
    entities = normalize_image_metadata(metadata)
    assert any(e.type == "camera_serial" for e in entities)
    assert any(e.type == "vehicle_plate" for e in entities)


# ---------------------------------------------------------------------------
# synthetic face cluster + vehicle/plate extraction
# ---------------------------------------------------------------------------


def test_synthetic_face_matching_across_three_images(tmp_path):
    case_dir = _jesse_case(tmp_path)
    faces = []
    for image in sorted((case_dir / "images").glob("IMG_00[123]*.jpg")):
        extracted = extract_face_entities(image)
        assert extracted, f"no face entity for {image.name}"
        for entity in extracted:
            assert entity.metadata["synthetic_identity"] == JESSE_IDENTITY
            faces.append(entity)
    assert len(faces) == 3
    matches = correlate_entities(faces)
    face_matches = [m for m in matches if m.rule == "face_embedding_cosine"]
    assert len(face_matches) >= 2
    clusters = {m.metadata["person_cluster_id"] for m in face_matches}
    assert clusters, "face matches must form a person cluster"


def test_synthetic_vehicle_and_plate_extraction(tmp_path):
    case_dir = _jesse_case(tmp_path)
    entities = extract_case_entities(case_dir)
    plates = [e for e in entities if e.type == "vehicle_plate"]
    assert any(str(e.value) == JESSE_PLATE for e in plates)
    cameras = [e for e in entities if e.type == "camera_serial"]
    assert any(str(e.value) == "CAM-JESSE-001" for e in cameras)
    locations = [e for e in entities if e.type == "location"]
    assert locations


# ---------------------------------------------------------------------------
# image observations
# ---------------------------------------------------------------------------


def test_image_observation_serialization(tmp_path):
    case_dir = _jesse_case(tmp_path)
    observations = build_image_observations(case_dir)
    assert observations
    payload = json.loads(json.dumps([o.model_dump(mode="json") for o in observations]))
    assert payload[0]["observation_id"].startswith("OBS-")
    assert payload[0]["image_path"]
    assert "evidence_classification" in payload[0]
    assert "requires_human_review" in payload[0]


def test_observation_types_present(tmp_path):
    case_dir = _jesse_case(tmp_path)
    observations = build_image_observations(case_dir)
    types = {o.observation_type for o in observations}
    for expected in (
        "person",
        "number_plate",
        "gps_location",
        "timestamp",
        "camera_serial",
        "sensitive_content",
        "activity_interpretation",
    ):
        assert expected in types, f"missing observation type: {expected}"


def test_alternative_plate_readings(tmp_path):
    case_dir = _jesse_case(tmp_path, adversarial=True)
    observations = build_image_observations(case_dir)
    plates = ambiguous_plate_observations(observations)
    assert len(plates) == 1
    plate = plates[0]
    assert plate.value == JESSE_PLATE
    assert set(plate.alternatives) == {"NM-JP-424Z", "NM-JP-4247"}
    assert plate.requires_human_confirmation is True
    # clean variant has a confident reading with no alternatives
    clean = ambiguous_plate_observations(build_image_observations(_jesse_case(tmp_path)))
    assert clean == []


def test_activity_interpretation_labeling(tmp_path):
    case_dir = _jesse_case(tmp_path)
    observations = build_image_observations(case_dir)
    activity = [o for o in observations if o.observation_type == "activity_interpretation"]
    assert len(activity) == 1
    assert activity[0].evidence_classification == "LLM visual interpretation"
    assert activity[0].requires_human_review is True
    assert "transparent object" in activity[0].value
    assert "drug" not in activity[0].value.lower()  # no overclaiming


def test_sensitive_content_observation(tmp_path):
    case_dir = _jesse_case(tmp_path)
    observations = build_image_observations(case_dir)
    sensitive = [o for o in observations if o.observation_type == "sensitive_content"]
    assert len(sensitive) == 1
    assert sensitive[0].confidence is None  # "Not scored", not 0.0


# ---------------------------------------------------------------------------
# demo hypotheses
# ---------------------------------------------------------------------------


def test_demo_hypotheses_generated(tmp_path):
    case_dir = _jesse_case(tmp_path)
    entities = extract_case_entities(case_dir)
    matches = correlate_entities(entities, case_id="CASE-J")
    observations = build_image_observations(case_dir)
    hypotheses = generate_hypotheses(entities, matches, JESSE_GOAL, observations)
    statements = [h.statement for h in hypotheses]
    assert any("same candidate subject appears in Images 1, 2, and 3" in s for s in statements)
    assert any("associated with vehicle NM-JP-4242" in s for s in statements)
    assert any("places the candidate subject and vehicle" in s for s in statements)
    assert any("may show substance-related activity" in s for s in statements)
    assert any("same investigation event" in s for s in statements)
    for h in hypotheses:
        assert h.requires_human_review is True
        assert h.related_entity_ids or h.supporting_evidence  # evidence-linked


def test_demo_hypotheses_only_for_jesse_fixture(tmp_path):
    """The demo hypotheses must not leak into other synthetic cases."""
    case_dir = tmp_path / "case"
    generate_case(case_dir)
    entities = extract_case_entities(case_dir)
    matches = correlate_entities(entities, case_id="CASE-X")
    observations = build_image_observations(case_dir)
    hypotheses = generate_hypotheses(entities, matches, "Find links for vehicle KL07CX4242", observations)
    assert not any("jesse" in h.statement.lower() or "Images 1, 2, and 3" in h.statement for h in hypotheses)


# ---------------------------------------------------------------------------
# contradictions
# ---------------------------------------------------------------------------


def test_adversarial_contradictions_detected(tmp_path):
    case_dir = _jesse_case(tmp_path, adversarial=True)
    entities = extract_case_entities(case_dir)
    matches = correlate_entities(entities, case_id="CASE-J")
    observations = build_image_observations(case_dir)
    contradictions = hunt_contradictions(entities, matches, observations=observations)
    rules = {c.rule for c in contradictions}
    assert "impossible_movement" in rules
    assert "ambiguous_plate" in rules
    assert "metadata_mismatch" in rules
    assert "inconsistent_identity_vehicle" in rules
    assert "inconsistent_identity_camera" in rules
    for c in contradictions:
        assert c.evidence_refs, "contradictions must cite artifacts"


def test_clean_case_has_no_contradictions(tmp_path):
    case_dir = _jesse_case(tmp_path)
    entities = extract_case_entities(case_dir)
    matches = correlate_entities(entities, case_id="CASE-J")
    observations = build_image_observations(case_dir)
    contradictions = hunt_contradictions(entities, matches, observations=observations)
    assert contradictions == []


# ---------------------------------------------------------------------------
# skeptic review
# ---------------------------------------------------------------------------


def test_skeptic_responds_to_contradictions(tmp_path):
    case_dir = _jesse_case(tmp_path, adversarial=True)
    entities = extract_case_entities(case_dir)
    matches = correlate_entities(entities, case_id="CASE-J")
    observations = build_image_observations(case_dir)
    contradictions = [
        c.to_dict() for c in hunt_contradictions(entities, matches, observations=observations)
    ]
    review = run_case_skeptic(
        conclusion="The candidate subject is associated with vehicle NM-JP-4242",
        original_confidence=0.75,
        hypotheses=[Hypothesis(id="H-002", statement="vehicle link", confidence=0.75)],
        contradictions=contradictions,
        risk_tier="MED",
        hypothesis_id="H-002",
    )
    assert review.revised_confidence < review.original_confidence
    assert review.strongest_alternative
    assert "No contradictions were found" not in review.strongest_alternative
    assert review.requires_human_review is True


def test_skeptic_targets_specific_hypothesis(tmp_path):
    """The router must run the skeptic against the requested hypothesis id."""
    case_dir = _jesse_case(tmp_path, adversarial=True)
    entities = extract_case_entities(case_dir)
    matches = correlate_entities(entities, case_id="CASE-J")
    observations = build_image_observations(case_dir)
    hypotheses = generate_hypotheses(entities, matches, JESSE_GOAL, observations)
    contradictions = [
        c.to_dict() for c in hunt_contradictions(entities, matches, observations=observations)
    ]
    state = initial_state("CASE-J", case_dir, JESSE_GOAL)
    state["entities"] = entities
    state["matches"] = matches
    state["hypotheses"] = hypotheses
    state["contradictions"] = contradictions
    task = PlannerTask(step=1, tool="SkepticReview", reason="challenge the lead", inputs=["H-002"])
    updates, _ = route_task(task, state)
    assert updates["skeptic_review"]["hypothesis_id"] == "H-002"
    assert updates["skeptic_review"]["conclusion_under_review"] == next(
        h.statement for h in hypotheses if h.id == "H-002"
    )


# ---------------------------------------------------------------------------
# contradiction-aware finalization + demo presets
# ---------------------------------------------------------------------------


def test_adversarial_demo_preset(tmp_path):
    case_dir = _jesse_case(tmp_path, adversarial=True)
    ledger = LedgerStore(tmp_path / "ledger_adv.sqlite3")
    state = _run_dynamic(case_dir, ledger)
    assert state["stop_reason"] == "unresolved_contradictions"
    assert state["needs_human_review"] is True
    rules = {c["rule"] for c in state["contradictions"]}
    assert "impossible_movement" in rules and "metadata_mismatch" in rules
    # skeptic challenged the lead and lowered confidence
    skeptic = state["skeptic_review"]
    assert skeptic is not None
    assert skeptic["revised_confidence"] < skeptic["original_confidence"]
    # dynamic order: observation before correlation, re-rank after skeptic
    tools = [a.tool for a in state["actions_taken"] if a.tool]
    assert tools.index("ObservationAgent") < tools.index("CorrelationAgent")
    assert tools.index("SkepticReview") < tools.index("TriageAgent")
    # final report wording never overclaims
    report = state["final_result"]
    assert "HUMAN REVIEW REQUIRED" in report
    assert "does not establish identity, intent, ownership, or criminal activity" in report
    assert "candidate association" in report.lower()
    assert "sufficient_evidence" not in report
    # confidence drop is visible
    assert state["confidence_breakdown"]["vehicle_association"] is not None
    assert state["confidence_breakdown"]["vehicle_association"] < 0.6


def test_clean_demo_preset(tmp_path):
    case_dir = _jesse_case(tmp_path)
    ledger = LedgerStore(tmp_path / "ledger_clean.sqlite3")
    state = _run_dynamic(case_dir, ledger)
    assert state["stop_reason"] == "awaiting_human_confirmation"
    assert state["needs_human_review"] is True
    assert state["contradictions"] == []
    report = state["final_result"]
    assert "HUMAN REVIEW REQUIRED" in report
    assert "Confirm the face/subject association" in report
    assert "Synthetic fixture identity label: jesse_pinkman" in report


def test_sufficient_evidence_never_emitted_with_unconfirmed_identity(tmp_path):
    case_dir = _jesse_case(tmp_path)
    ledger = LedgerStore(tmp_path / "ledger.sqlite3")
    state = _run_dynamic(case_dir, ledger)
    assert state["stop_reason"] != "sufficient_evidence"
    assert "**Stop reason:** sufficient_evidence" not in state["final_result"]


# ---------------------------------------------------------------------------
# sensitive-image blur
# ---------------------------------------------------------------------------


def test_sensitive_image_blur_derivative(tmp_path):
    from karuth.vision.redaction import create_blurred_derivative, sha256_of

    image = tmp_path / "IMG_001_suspicious.jpg"
    shutil.copy2(_jesse_case(tmp_path) / "images" / "IMG_001_suspicious.jpg", image)
    original_bytes = image.read_bytes()
    original_hash = sha256_of(image)

    derivative = create_blurred_derivative(image, tmp_path / "derivatives")
    assert derivative["blurred"] is True
    assert derivative["sha256"] != derivative["original_sha256"]
    assert derivative["original_sha256"] == original_hash
    assert Path(derivative["path"]).exists()
    # the original evidence is never modified
    assert image.read_bytes() == original_bytes
    assert "blurred derivative of original" in derivative["provenance"]


def test_blur_degrades_for_corrupt_image(tmp_path):
    from karuth.vision.redaction import create_blurred_derivative

    image = tmp_path / "broken.jpg"
    image.write_bytes(b"not an image at all")
    derivative = create_blurred_derivative(image, tmp_path / "derivatives")
    assert derivative["blurred"] is False  # never raises


# ---------------------------------------------------------------------------
# provenance-aware graph
# ---------------------------------------------------------------------------


def _sample_graph(tmp_path):
    from karuth.planner.candidate_graph import build_candidate_graph

    case_dir = _jesse_case(tmp_path, adversarial=True)
    entities = extract_case_entities(case_dir)
    matches = correlate_entities(entities, case_id="CASE-J")
    observations = build_image_observations(case_dir)
    contradictions = [
        c.to_dict() for c in hunt_contradictions(entities, matches, observations=observations)
    ]
    return build_candidate_graph(
        case_id="CASE-J",
        entities=entities,
        matches=matches,
        observations=observations,
        contradictions=contradictions,
        ledger_entry_id="L-42",
    )


def test_graph_provenance(tmp_path):
    graph = _sample_graph(tmp_path)
    assert graph["nodes"], "candidate graph must have nodes"
    kinds = {n["kind"] for n in graph["nodes"]}
    assert "person" in kinds and "vehicle" in kinds and "location" in kinds
    assert graph["edges"], "candidate graph must have edges"
    for edge in graph["edges"]:
        assert "evidence_refs" in edge
        assert "confidence" in edge
        assert "provenance_type" in edge
        assert "review_status" in edge
        assert "source_artifact" in edge
        assert "ledger_entry_id" in edge


def test_graph_edge_styles_and_review_status(tmp_path):
    from karuth.planner.candidate_graph import EDGE_STYLES, edge_style

    graph = _sample_graph(tmp_path)
    styles = {edge["rel_type"]: edge["style"] for edge in graph["edges"]}
    assert styles.get("appears_in") == "solid blue"
    assert styles.get("candidate_associated_with") == "dotted amber"
    assert styles.get("gps_at") == "solid blue"
    assert styles.get("candidate_present_in_image_at") == "grey"
    assert styles.get("conflicts_with") == "broken red"
    # AI-proposed edges are never presented as confirmed
    for edge in graph["edges"]:
        if edge["provenance_type"] == "ai_proposed":
            assert edge["review_status"] != "confirmed"
    assert edge_style("human_confirmed") == "thick purple"
    assert edge_style("deterministic_fact") == "solid green"
    assert edge_style("unknown") == "grey"
    assert set(EDGE_STYLES) == {
        "deterministic_fact",
        "metadata_fact",
        "ai_proposed",
        "human_confirmed",
        "contradiction",
        "weak",
    }


# ---------------------------------------------------------------------------
# final report ordering
# ---------------------------------------------------------------------------


def test_final_report_ordering(tmp_path):
    from karuth.planner.evidence_report import build_evidence_report

    case_dir = _jesse_case(tmp_path, adversarial=True)
    entities = extract_case_entities(case_dir)
    matches = correlate_entities(entities, case_id="CASE-J")
    observations = build_image_observations(case_dir)
    hypotheses = generate_hypotheses(entities, matches, JESSE_GOAL, observations)
    contradictions = [
        c.to_dict() for c in hunt_contradictions(entities, matches, observations=observations)
    ]
    report = build_evidence_report(
        goal=JESSE_GOAL,
        matches=matches,
        entities=entities,
        hypotheses=hypotheses,
        contradictions=contradictions,
        skeptic=None,
        final_confidence=0.4,
        initial_confidence=0.7,
        stop_reason="unresolved_contradictions",
        image_observations=observations,
        needs_human_review=True,
    )
    sections = [
        "# Investigation status",
        "## Primary lead",
        "## Confirmed observations",
        "## AI interpretations",
        "## Contradictions and limitations",
        "## Recommended next action",
        "## Conclusion",
    ]
    positions = [report.index(section) for section in sections]
    assert positions == sorted(positions), "report sections must appear in order"
    assert "HUMAN REVIEW REQUIRED" in report


# ---------------------------------------------------------------------------
# ReportAgent gating in dynamic mode
# ---------------------------------------------------------------------------


def test_report_agent_rejected_in_dynamic_mode():
    state = {"hypotheses": [], "case_folder": "."}
    with pytest.raises(ValueError, match="ReportAgent cannot be selected in dynamic mode"):
        validate_action(
            ActionDecision(action_type="run_tool", tool="ReportAgent", rationale="report now"),
            state,
        )


def test_dynamic_run_never_executes_report_agent(tmp_path):
    case_dir = _jesse_case(tmp_path, adversarial=True)
    ledger = LedgerStore(tmp_path / "ledger.sqlite3")
    state = _run_dynamic(case_dir, ledger)
    tools = [a.tool for a in state["actions_taken"] if a.tool]
    assert "ReportAgent" not in tools
    # the final report comes from the finalize step only
    assert state["final_result"]


# ---------------------------------------------------------------------------
# Streamlit state normalization (pydantic + dict hypotheses)
# ---------------------------------------------------------------------------


def _import_view_helpers():
    root = Path(__file__).resolve().parent.parent
    frontend = root / "frontend"
    if str(frontend) not in sys.path:
        sys.path.insert(0, str(frontend))
    from components.investigation_view import (
        action_fields,
        as_dict,
        hypothesis_fields,
    )

    return action_fields, as_dict, hypothesis_fields


def test_streamlit_state_with_pydantic_hypotheses():
    _, _, hypothesis_fields = _import_view_helpers()
    h = Hypothesis(
        id="H-001",
        statement="The same candidate subject appears in Images 1, 2, and 3.",
        status="open",
        confidence=0.6,
        supporting_evidence=["face-a"],
        missing_evidence=["independent verification"],
        recommended_next_actions=["face_cluster_review"],
        requires_human_review=True,
    )
    fields = hypothesis_fields(h)
    assert fields["id"] == "H-001"
    assert fields["confidence"] == 0.6
    assert fields["supporting"] == ["face-a"]
    assert fields["review"] is True


def test_streamlit_state_with_dict_hypotheses():
    _, _, hypothesis_fields = _import_view_helpers()
    h = {
        "id": "H-002",
        "statement": "vehicle link",
        "status": "inconclusive",
        "confidence": 0.4,
        "supporting_evidence": [],
        "requires_human_review": False,
    }
    fields = hypothesis_fields(h)
    assert fields["id"] == "H-002"
    assert fields["status"] == "inconclusive"
    assert fields["confidence"] == 0.4
    assert fields["review"] is False


def test_action_label_normalization():
    action_fields, as_dict, _ = _import_view_helpers()
    decision = ActionDecision(
        action_type="challenge_hypothesis",
        tool="SkepticReview",
        target_hypothesis_id="H-002",
        rationale="challenge the lead",
        expected_information_gain=0.6,
    )
    fields = action_fields(decision)
    assert fields["action_type"] == "challenge_hypothesis"
    assert fields["target"] == "H-002"
    assert fields["gain"] == 0.6
    assert as_dict({"a": 1}) == {"a": 1}


# ---------------------------------------------------------------------------
# classic planner backward compatibility
# ---------------------------------------------------------------------------


def test_classic_planner_backward_compat(tmp_path):
    case_dir = _jesse_case(tmp_path, adversarial=True)
    ledger = LedgerStore(tmp_path / "ledger.sqlite3")
    state = run_planner_goal(
        "CASE-J", case_dir, JESSE_GOAL, ledger, planner=DeterministicPlanner(), mode="plan"
    )
    assert state["mode"] == "plan"
    assert state["plan"].tasks
    assert state.get("hypotheses", []) == []
    assert state.get("image_observations", []) == []
    assert ledger.verify()


# ---------------------------------------------------------------------------
# human confirmation gates + ledger
# ---------------------------------------------------------------------------


def test_human_confirmation_gates_recorded_in_ledger(tmp_path):
    case_dir = _jesse_case(tmp_path)
    ledger = LedgerStore(tmp_path / "ledger.sqlite3")
    state = _run_dynamic(case_dir, ledger)
    assert state["stop_reason"] == "awaiting_human_confirmation"
    assert state["needs_human_review"] is True

    # the investigator records decisions; the chain must stay verifiable
    ledger.append(
        LedgerEntryCreate(
            entry_type="review_action",
            case_id="CASE-J",
            actor="investigator",
            action="identity_link_confirmed",
            target="face_association",
            payload={"same_subject": True, "label": True},
        )
    )
    ledger.append(
        LedgerEntryCreate(
            entry_type="review_action",
            case_id="CASE-J",
            actor="investigator",
            action="plate_reading_confirmed",
            target="number_plate",
            payload={"value": "NM-JP-4242"},
        )
    )
    actions = [entry.action for entry in ledger.list_entries()]
    assert "identity_link_confirmed" in actions
    assert "plate_reading_confirmed" in actions
    assert ledger.verify()


def test_full_audit_ledger_verification(tmp_path):
    case_dir = _jesse_case(tmp_path, adversarial=True)
    ledger = LedgerStore(tmp_path / "ledger.sqlite3")
    state = _run_dynamic(case_dir, ledger)
    entries = ledger.list_entries()
    assert entries
    actions = {entry.action for entry in entries}
    for expected in (
        "hypothesis_created",
        "action_decided",
        "observation_recorded",
        "contradiction_found",
        "skeptic_reviewed",
        "dynamic_finalized",
        "planner_finished",
        "chain_verified",
    ):
        assert expected in actions, f"missing ledger action: {expected}"
    assert state["final_result"]
    assert ledger.verify()


# ---------------------------------------------------------------------------
# approval-gated mock MVD registry lookup (general, any plate)
# ---------------------------------------------------------------------------


def test_mock_mvd_lookup_returns_registered_owner():
    from karuth.external.mvd import MockMVDRegistry
    from karuth.planner.extensions import ExternalLookupRequest

    result = MockMVDRegistry().lookup(
        ExternalLookupRequest(lookup_kind="mvd_vehicle_owner", key="NM-JP-4242")
    )
    assert result.source_system == "MOCK_MVD_REGISTRY"
    assert result.key == "NM-JP-4242"
    fields = {f["field"]: f for f in result.findings}
    assert fields["registered_owner"]["value"] == "Dana R. Voss"
    assert fields["registered_owner"]["confidence"] == 0.99
    assert "does not prove who operated" in fields["registered_owner"]["note"]


def test_mock_mvd_lookup_unknown_plate_deterministic_fallback():
    """Any plate resolves deterministically to a clearly-fake fixture owner."""
    from karuth.external.mvd import MockMVDRegistry
    from karuth.planner.extensions import ExternalLookupRequest

    result = MockMVDRegistry().lookup(
        ExternalLookupRequest(lookup_kind="mvd_vehicle_owner", key="AP-01-X-9999")
    )
    assert result.key == "AP-01-X-9999"
    fields = {f["field"]: f for f in result.findings}
    assert fields["registered_owner"]["value"]
    # deterministic: same plate -> same owner on repeat calls
    again = MockMVDRegistry().lookup(
        ExternalLookupRequest(lookup_kind="mvd_vehicle_owner", key="AP-01-X-9999")
    )
    assert {
        f["field"]: f["value"] for f in again.findings
    } == {f["field"]: f["value"] for f in result.findings}


def test_mvd_lookup_blocked_without_approval(tmp_path):
    """The approval gate soft-fails: structured ToolResult error, no crash."""
    from karuth.planner.router import run_tool
    from karuth.planner.schema import PlannerTask
    from karuth.planner.state import initial_state

    case_dir = _jesse_case(tmp_path)
    state = initial_state("CASE-J", case_dir, JESSE_GOAL)  # registry_lookup_approved=False
    task = PlannerTask(step=1, tool="MvdRegistryLookup", reason="registry")
    result = run_tool(task, state)
    assert result.success is False
    assert "approval" in result.error.lower()
    assert result.state_updates.get("registry_results", []) == []


def test_mvd_lookup_approved_includes_rc_owner_in_report(tmp_path):
    """With approval, the dynamic loop runs the lookup and the report carries
    the RC owner as a clearly-labeled synthetic metadata fact."""
    case_dir = _jesse_case(tmp_path)  # clean preset
    ledger = LedgerStore(tmp_path / "ledger.sqlite3")
    state = run_planner_goal(
        "CASE-J",
        case_dir,
        JESSE_GOAL,
        ledger,
        planner=DeterministicPlanner(),
        mode="dynamic",
        registry_lookup_approved=True,
    )
    assert state.get("registry_results"), "registry lookup should have run"
    # the hard finalization rules still gate on the unconfirmed identity
    assert state["stop_reason"] == "awaiting_human_confirmation"
    assert state["needs_human_review"] is True
    report = state["final_result"]
    assert "registered owner" in report
    assert "MOCK_MVD_REGISTRY" in report or "mock MVD" in report
    assert "does not prove" in report
    actions = {entry.action for entry in ledger.list_entries()}
    assert "registry_lookup_approved" in actions
    assert "registry_lookup_result" in actions
    assert ledger.verify()


def test_registry_lookup_skipped_without_approval(tmp_path):
    """Existing flows are unchanged: no approval -> the registry never runs."""
    case_dir = _jesse_case(tmp_path, adversarial=True)
    ledger = LedgerStore(tmp_path / "ledger.sqlite3")
    state = _run_dynamic(case_dir, ledger)
    assert "MvdRegistryLookup" not in state.get("completed_steps", [])
    assert state["stop_reason"] == "unresolved_contradictions"
    assert state.get("registry_results") == []


# ---------------------------------------------------------------------------
# real photos with no machine-readable evidence (the "why nothing found" case)
# ---------------------------------------------------------------------------


def _photo_only_case(tmp_path) -> Path:
    """A case folder with a real-photo-like image: JPEG bytes, no EXIF, no
    sidecar — exactly what a screenshot/ChatGPT image looks like to Karuth."""
    case_dir = tmp_path / "photos"
    images = case_dir / "images"
    images.mkdir(parents=True)
    (images / "photo.jpg").write_bytes(b"\xff\xd8\xff\xe0 not really a jpeg but a file")
    return case_dir


def test_photo_only_case_stops_with_evidence_gap(tmp_path):
    """Images with no machine-readable content produce an explicit evidence
    gap in the report (and human review) instead of a confusing empty page."""
    case_dir = _photo_only_case(tmp_path)
    ledger = LedgerStore(tmp_path / "ledger.sqlite3")
    state = run_planner_goal(
        "CASE-P",
        case_dir,
        "Find the person connected to vehicle KL01CC2121 and tell me if any person is",
        ledger,
        planner=DeterministicPlanner(),
        mode="dynamic",
    )
    assert state["stop_reason"] == "no_information_gain"
    assert state["needs_human_review"] is True
    report = state["final_result"]
    assert "## Evidence gap" in report
    assert "KL01CC2121" in report
    assert "only in the investigator goal" in report
    assert "OCR" in report


def test_goal_plate_triggers_mvd_lookup_without_extracted_entities(tmp_path):
    """Approval + a plate named in the goal is enough to run the registry
    lookup even when the photos yield zero entities/observations. The owner is
    a clearly-labeled synthetic candidate, and the case still requires review."""
    case_dir = _photo_only_case(tmp_path)
    ledger = LedgerStore(tmp_path / "ledger.sqlite3")
    state = run_planner_goal(
        "CASE-P",
        case_dir,
        "Find the person connected to vehicle kl01cc2121 and tell me if any person is",
        ledger,
        planner=DeterministicPlanner(),
        mode="dynamic",
        registry_lookup_approved=True,
    )
    assert state.get("registry_results"), "registry lookup should have run from the goal plate"
    assert state["stop_reason"] == "awaiting_human_confirmation"
    assert state["needs_human_review"] is True
    report = state["final_result"]
    assert "KL01CC2121" in report  # compact lowercase goal plate normalized
    assert "registered owner" in report
    assert "does not prove" in report
    assert "## Evidence gap" in report  # photos still contributed nothing


def test_goal_plate_compact_lowercase_normalized():
    """The goal-plate regex accepts compact lowercase plates (KL01cc2121)."""
    from karuth.agents.hypothesis_agent import _goal_values

    values = _goal_values("Find the person connected to vehicle kl01cc2121")
    assert any(v == "KL01CC2121" for v in values)


def test_ocr_module_graceful_without_tesseract(tmp_path):
    """OCR reads must never crash or fabricate when Tesseract is absent."""
    from karuth.vision.ocr import ocr_available, read_plate_candidates

    image = tmp_path / "photo.jpg"
    image.write_bytes(b"\xff\xd8\xff\xe0 not a jpeg")
    candidates = read_plate_candidates(image)
    assert isinstance(candidates, list)
    # Either OCR is installed (real candidate dicts) or gracefully empty.
    if ocr_available():
        assert all("value" in c and "confidence" in c for c in candidates)
    else:
        assert candidates == []
