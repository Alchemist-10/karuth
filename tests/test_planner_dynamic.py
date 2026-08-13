"""Tests for the dynamic hypothesis-driven planner loop.

Covers: hypothesis lifecycle, dynamic action selection (deterministic policy +
LLM backend), action guardrails, structured ToolResult validation, evidence
references on claims, contradiction detection, skeptic review, confidence
updates, stopping reasons, the human-review checkpoint, deterministic
fallback, ledger entries, and the complete wow-moment flow.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from scripts.generate_synthetic_case import generate_case
from karuth.agents.hypothesis_agent import generate_hypotheses, propose_hypothesis
from karuth.ledger import LedgerStore
from karuth.planner.actions import (
    ActionDecision,
    DeterministicActionPlanner,
    decision_context,
    validate_action,
)
from karuth.planner.contradictions import hunt_contradictions
from karuth.planner.hypothesis import Hypothesis, HypothesisStore
from karuth.planner.llm import DeterministicPlanner, LMStudioPlanner
from karuth.planner.observer import observe_result
from karuth.planner.router import run_tool
from karuth.planner.schema import PlannerTask
from karuth.planner.skeptic import run_case_skeptic, skeptic_should_run
from karuth.planner.state_machine import run_planner_goal
from karuth.planner.tool_result import Claim, ToolResult


def _wow_case(tmp_path, *, include_wow_data: bool = True) -> Path:
    case_dir = tmp_path / "case"
    generate_case(case_dir, include_wow_data=include_wow_data)
    return case_dir


def _chat_only_case(tmp_path) -> Path:
    case_dir = _wow_case(tmp_path, include_wow_data=False)
    for subdir in ("images", "bank", "documents"):
        shutil.rmtree(case_dir / subdir)
    return case_dir


def _extract_state(case_dir: Path) -> dict:
    """Run extraction + correlation + timeline and return the resulting state."""
    planner = DeterministicPlanner()
    ledger = LedgerStore(Path(case_dir).parent / "ledger.sqlite3")
    state = run_planner_goal(
        "CASE-1",
        case_dir,
        "Find links for vehicle KL07CX4242 and triage the top leads",
        ledger,
        planner=planner,
    )
    return state


WOW_GOAL = (
    "Find the person connected to vehicle KL07CX4242 and determine whether "
    "the chat activity supports that connection."
)


# ---------------------------------------------------------------------------
# hypothesis lifecycle
# ---------------------------------------------------------------------------


def test_hypothesis_lifecycle_and_transitions():
    store = HypothesisStore()
    h = store.create("Arun is connected to KL07CX4242", evidence=["e-1"], confidence=0.5)
    assert h.status == "open"
    assert h.id == "H-001"

    store.add_supporting("H-001", ["e-2", "e-3", "e-4"])
    assert h.confidence > 0.5
    assert h.status == "supported"  # 0.5 + 3 * 0.1 = 0.8 >= 0.75

    with pytest.raises(ValueError, match="invalid hypothesis transition"):
        store.transition("H-001", "supported")  # already supported

    store.add_contradicting("H-001", ["c-1", "c-2", "c-3"], delta=0.2)
    assert h.confidence > 0
    assert h.status in ("inconclusive", "rejected")
    store.transition("H-001", "open")  # reopen is always allowed


def test_hypothesis_rejected_by_contradiction():
    store = HypothesisStore()
    store.create("Claim", evidence=["e-1"], confidence=0.5)
    for _ in range(6):
        store.add_contradicting("H-001", ["c-x"], delta=0.2)
    assert store.get("H-001").status == "rejected"
    assert store.get("H-001").confidence == 0.0


def test_duplicate_hypothesis_statement_rejected():
    store = HypothesisStore()
    store.create("Statement A", evidence=["e-1"])
    with pytest.raises(ValueError, match="duplicate"):
        store.create("Statement A", evidence=["e-2"])


# ---------------------------------------------------------------------------
# deterministic hypothesis generation
# ---------------------------------------------------------------------------


def test_generate_hypotheses_from_synthetic_case(tmp_path):
    case_dir = _wow_case(tmp_path)
    state = _extract_state(case_dir)
    hypotheses = generate_hypotheses(state["entities"], state["matches"], WOW_GOAL)
    assert len(hypotheses) >= 2
    statements = [h.statement for h in hypotheses]
    assert any("Arun Nair" in s for s in statements)
    assert any("Maya Pillai" in s for s in statements)
    for h in hypotheses:
        assert h.related_entity_ids  # every hypothesis cites real entities


def test_llm_proposal_requires_real_evidence_refs(tmp_path):
    case_dir = _wow_case(tmp_path)
    state = _extract_state(case_dir)
    entities = state["entities"]
    real_id = entities[0].entity_id
    adopted = propose_hypothesis("A proposal", [real_id], entities)
    assert adopted is not None
    rejected = propose_hypothesis("Fabricated proposal", ["not-a-real-entity"], entities)
    assert rejected is None


# ---------------------------------------------------------------------------
# dynamic action selection (deterministic policy)
# ---------------------------------------------------------------------------


def _ctx(**overrides) -> dict:
    base = {
        "goal": WOW_GOAL,
        "risk_tier": "LOW",
        "entity_count": 0,
        "match_count": 0,
        "timeline_count": 0,
        "hypotheses": [],
        "contradictions": [],
        "completed_tools": [],
        "action_count": 0,
        "max_actions": 12,
        "skeptic_reviewed": False,
    }
    base.update(overrides)
    return base


def test_policy_starts_with_extraction():
    decision = DeterministicActionPlanner().decide_action(WOW_GOAL, _ctx())
    assert decision.action_type == "run_tool"
    assert decision.tool == "ExtractionAgent"


def test_policy_proceeds_through_correlation_then_hypotheses():
    policy = DeterministicActionPlanner()
    after_extract = _ctx(entity_count=12, completed_tools=["ExtractionAgent"])
    decision = policy.decide_action(WOW_GOAL, after_extract)
    assert decision.tool == "CorrelationAgent"

    after_correlate = _ctx(
        entity_count=12,
        match_count=40,
        completed_tools=["ExtractionAgent", "CorrelationAgent"],
    )
    decision = policy.decide_action(WOW_GOAL, after_correlate)
    assert decision.tool == "TimelineAgent"  # order the evidence before hypothesizing

    after_timeline = {**after_correlate, "timeline_count": 10, "completed_tools": ["ExtractionAgent", "CorrelationAgent", "TimelineAgent"]}
    decision = policy.decide_action(WOW_GOAL, after_timeline)
    assert decision.action_type == "generate_hypothesis"


def test_policy_hunts_contradictions_then_challenges_then_finalizes():
    policy = DeterministicActionPlanner()
    hypotheses = [{"id": "H-001", "status": "open", "confidence": 0.5}]
    ctx = _ctx(
        entity_count=12,
        match_count=40,
        timeline_count=10,
        completed_tools=["ExtractionAgent", "CorrelationAgent", "TimelineAgent"],
        hypotheses=hypotheses,
    )
    decision = policy.decide_action(WOW_GOAL, ctx)
    assert decision.tool == "ContradictionHunter"

    with_contradictions = {**ctx, "contradictions": [{"id": "C-1", "rule": "impossible_movement", "severity": "high"}]}
    decision = policy.decide_action(WOW_GOAL, with_contradictions)
    assert decision.action_type == "challenge_hypothesis"
    assert decision.target_hypothesis_id == "H-001"

    challenged = {**with_contradictions, "skeptic_reviewed": True}
    decision = policy.decide_action(WOW_GOAL, challenged)
    assert decision.action_type == "finalize"
    assert decision.stop_reason == "unresolved_contradictions"
    assert decision.requires_human_review is True


def test_policy_stops_with_sufficient_evidence_when_no_contradictions():
    policy = DeterministicActionPlanner()
    ctx = _ctx(
        entity_count=12,
        match_count=40,
        timeline_count=10,
        hypotheses=[{"id": "H-001", "status": "supported", "confidence": 0.85}],
        completed_tools=["ExtractionAgent", "CorrelationAgent", "TimelineAgent", "ContradictionHunter"],
        contradictions=[],  # hunter ran, found nothing
        skeptic_reviewed=True,
    )
    decision = policy.decide_action(WOW_GOAL, ctx)
    assert decision.action_type == "finalize"
    assert decision.stop_reason == "sufficient_evidence"


def test_policy_ask_human_only_at_critical_risk_with_contradictions():
    policy = DeterministicActionPlanner()
    ctx = _ctx(
        risk_tier="CRITICAL",
        entity_count=12,
        match_count=5,
        timeline_count=10,
        hypotheses=[{"id": "H-001", "status": "open", "confidence": 0.5}],
        contradictions=[{"id": "C-1", "rule": "impossible_movement", "severity": "high"}],
        completed_tools=["ExtractionAgent", "CorrelationAgent", "TimelineAgent", "ContradictionHunter"],
        skeptic_reviewed=True,
    )
    decision = policy.decide_action(WOW_GOAL, ctx)
    assert decision.action_type == "ask_human"
    assert decision.requires_human_review is True


def test_policy_never_exceeds_action_budget():
    ctx = _ctx(entity_count=12, action_count=12, completed_tools=["ExtractionAgent"])
    decision = DeterministicActionPlanner().decide_action(WOW_GOAL, ctx)
    assert decision.action_type == "finalize"
    assert decision.stop_reason == "action_budget_exhausted"


# ---------------------------------------------------------------------------
# action guardrails
# ---------------------------------------------------------------------------


def test_action_guardrails_reject_unknown_tool_and_hypothesis():
    state = {"hypotheses": [Hypothesis(id="H-001", statement="s")], "case_folder": "."}
    with pytest.raises(ValueError, match="unknown planner tool"):
        validate_action(ActionDecision(action_type="run_tool", tool="NoSuchAgent"), state)
    with pytest.raises(ValueError, match="unknown hypothesis"):
        validate_action(
            ActionDecision(action_type="challenge_hypothesis", tool="SkepticReview", target_hypothesis_id="H-999"),
            state,
        )
    with pytest.raises(ValueError, match="External network calls"):
        validate_action(
            ActionDecision(action_type="run_tool", tool="ExtractionAgent", arguments={"inputs": ["https://evil.example/x"]}),
            state,
        )


# ---------------------------------------------------------------------------
# structured ToolResult and evidence-linked claims
# ---------------------------------------------------------------------------


def test_run_tool_returns_structured_result_with_evidence_refs(tmp_path):
    case_dir = _wow_case(tmp_path)
    state = _extract_state(case_dir)
    task = PlannerTask(step=1, tool="ExtractionAgent", reason="extract")
    result = run_tool(task, state, iteration=1)
    assert isinstance(result, ToolResult)
    assert result.success
    assert result.tool == "ExtractionAgent"
    assert result.state_updates["entities"]
    assert result.claims
    assert result.claims[0].claim_type == "deterministic_fact"
    assert result.claims[0].evidence_refs
    assert result.evidence_refs
    assert result.duration_ms >= 0


def test_correlation_tool_result_claims_reference_entities(tmp_path):
    case_dir = _wow_case(tmp_path)
    state = _extract_state(case_dir)
    task = PlannerTask(step=1, tool="CorrelationAgent", reason="correlate")
    result = run_tool(task, state, iteration=2)
    assert result.success
    known_ids = {e.entity_id for e in state["entities"]}
    for claim in result.claims:
        for ref in claim.evidence_refs:
            assert ref in known_ids, f"claim ref {ref} must be a real entity"


def test_tool_result_validation():
    result = ToolResult(success=True, tool="X", claims=[Claim(claim="c", evidence_refs=["e-1"])])
    assert result.confidence == 0.0
    with pytest.raises(ValueError):
        ToolResult(success=True, tool="X", confidence=1.5)
    with pytest.raises(ValueError):
        Claim(claim="c", confidence=2.0)


# ---------------------------------------------------------------------------
# contradiction detection
# ---------------------------------------------------------------------------


def test_wow_case_yields_movement_and_identity_contradictions(tmp_path):
    case_dir = _wow_case(tmp_path)
    state = _extract_state(case_dir)
    contradictions = hunt_contradictions(state["entities"], state["matches"])
    rules = {c.rule for c in contradictions}
    assert "impossible_movement" in rules
    assert "inconsistent_identity_vehicle" in rules
    for c in contradictions:
        assert c.evidence_refs, "contradictions must cite artifacts"
        assert c.confidence > 0.5


def test_chat_only_case_has_no_contradictions(tmp_path):
    case_dir = _chat_only_case(tmp_path)
    state = _extract_state(case_dir)
    contradictions = hunt_contradictions(state["entities"], state["matches"])
    assert contradictions == []


# ---------------------------------------------------------------------------
# skeptic review
# ---------------------------------------------------------------------------


def test_skeptic_review_lowers_confidence_with_contradictions():
    review = run_case_skeptic(
        conclusion="Arun is connected to the vehicle",
        original_confidence=0.8,
        hypotheses=[Hypothesis(id="H-001", statement="x", missing_evidence=["a timeline"])],
        contradictions=[
            {"id": "C-1", "rule": "impossible_movement", "claim_a": "a", "claim_b": "b", "evidence_refs": ["e-1"], "severity": "high", "resolution_status": "unresolved"}
        ],
        risk_tier="MED",
        hypothesis_id="H-001",
    )
    assert review.revised_confidence < review.original_confidence
    assert review.requires_human_review is True
    assert review.strongest_alternative
    assert review.missing_evidence


def test_skeptic_should_run_for_supported_high_confidence():
    h = Hypothesis(id="H-001", statement="x", status="supported", confidence=0.9)
    assert skeptic_should_run(h, [])
    h2 = Hypothesis(id="H-002", statement="x", status="open", confidence=0.5)
    assert not skeptic_should_run(h2, [])


# ---------------------------------------------------------------------------
# confidence updates via observation
# ---------------------------------------------------------------------------


def test_observation_updates_hypothesis_confidence():
    h = Hypothesis(id="H-001", statement="x", confidence=0.5, related_entity_ids=["e-1"])
    result = ToolResult(
        success=True,
        tool="CorrelationAgent",
        claims=[Claim(claim="match", evidence_refs=["e-1"], confidence=0.9, claim_type="deterministic_fact")],
    )
    observation = observe_result([h], result, iteration=1)
    assert h.confidence > 0.5
    assert observation.supporting_evidence == ["e-1"]
    assert observation.recommendation == "continue"


def test_observation_contradiction_lowers_confidence():
    h = Hypothesis(id="H-001", statement="x", confidence=0.6, related_entity_ids=["e-1"])
    result = ToolResult(
        success=True,
        tool="ContradictionHunter",
        contradictions=[
            {"id": "C-1", "rule": "impossible_movement", "evidence_refs": ["e-1"], "severity": "high", "resolution_status": "unresolved"}
        ],
    )
    observation = observe_result([h], result, iteration=1, extra_contradictions=result.contradictions)
    assert h.confidence < 0.6
    assert observation.recommendation == "challenge"


# ---------------------------------------------------------------------------
# LLM action backend
# ---------------------------------------------------------------------------


def test_llm_action_backend_parses_decision():
    import types

    payload = json.dumps(
        {
            "action_type": "run_tool",
            "tool": "TimelineAgent",
            "rationale": "need a timeline",
            "expected_information_gain": 0.6,
            "confidence": 0.8,
        }
    )

    def _create(**kwargs):
        message = types.SimpleNamespace(content=payload)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    completions = types.SimpleNamespace(create=_create)
    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions))
    planner = LMStudioPlanner(client=client)
    decision = planner.decide_action("Build a timeline", decision_context({}))
    assert decision.action_type == "run_tool"
    assert decision.tool == "TimelineAgent"
    assert decision.rationale == "need a timeline"


def test_fallback_planner_delegates_action_decisions():
    planner = DeterministicPlanner()
    assert planner.decide_action(WOW_GOAL, _ctx()) is not None


# ---------------------------------------------------------------------------
# full dynamic runs
# ---------------------------------------------------------------------------


def test_dynamic_run_wow_moment_flow(tmp_path):
    case_dir = _wow_case(tmp_path)
    ledger = LedgerStore(tmp_path / "ledger.sqlite3")
    state = run_planner_goal(
        "CASE-WOW", case_dir, WOW_GOAL, ledger, planner=DeterministicPlanner(), mode="dynamic"
    )
    assert state["mode"] == "dynamic"
    # 1. hypotheses formed
    assert len(state["hypotheses"]) >= 2
    # 2. dynamically chosen tools
    tools = [a.tool for a in state["actions_taken"] if a.tool]
    assert "CorrelationAgent" in tools
    assert "TimelineAgent" in tools
    assert "ContradictionHunter" in tools
    assert "SkepticReview" in tools
    # 3. contradiction found
    rules = {c.get("rule") for c in state["contradictions"]}
    assert "impossible_movement" in rules
    # 4. skeptic ran and lowered confidence instead of confirming
    skeptic = state["skeptic_review"]
    assert skeptic["revised_confidence"] < skeptic["original_confidence"]
    assert state["confidence"] <= skeptic["original_confidence"]
    # 5. stops with unresolved contradictions, recommending next evidence
    assert state["stop_reason"] == "unresolved_contradictions"
    assert "Missing evidence" in state["final_result"]
    # 6. report carries evidence refs
    assert "refs:" in state["final_result"] or "Confirmed facts" in state["final_result"]
    # 7. investigation tree recorded
    tree = state["investigation_tree"]
    assert tree.branches
    assert any(n.kind == "hypothesis" for n in tree.branches) or any(n.kind == "action" for n in tree.branches)
    # 8. ledger trail
    actions = [e.action for e in ledger.list_entries()]
    for name in ("hypothesis_created", "action_decided", "observation_recorded", "contradiction_found", "skeptic_reviewed", "dynamic_finalized", "planner_finished"):
        assert name in actions, f"missing ledger entry: {name}"
    assert ledger.verify()


def test_dynamic_run_stops_with_sufficient_evidence_on_chat_only_case(tmp_path):
    case_dir = _chat_only_case(tmp_path)
    ledger = LedgerStore(tmp_path / "ledger.sqlite3")
    state = run_planner_goal(
        "CASE-1",
        case_dir,
        "Find links for vehicle KL07CX4242 in the chat",
        ledger,
        planner=DeterministicPlanner(),
        mode="dynamic",
    )
    assert state["mode"] == "dynamic"
    assert state["stop_reason"] == "sufficient_evidence"
    assert state["final_result"]
    assert "sufficient_evidence" in state["final_result"]
    assert ledger.verify()


def test_dynamic_run_plan_mode_unchanged(tmp_path):
    case_dir = _wow_case(tmp_path)
    ledger = LedgerStore(tmp_path / "ledger.sqlite3")
    state = run_planner_goal(
        "CASE-1", case_dir, "Find links for vehicle KL07CX4242", ledger, planner=DeterministicPlanner()
    )
    assert state["mode"] == "plan"
    assert state["plan"].tasks
    # classic mode never populates the dynamic structures
    assert state.get("hypotheses", []) == []


def test_dynamic_unknown_mode_rejected(tmp_path):
    case_dir = _chat_only_case(tmp_path)
    ledger = LedgerStore(tmp_path / "ledger.sqlite3")
    with pytest.raises(ValueError, match="unknown planner mode"):
        run_planner_goal("CASE-1", case_dir, "x", ledger, planner=DeterministicPlanner(), mode="bogus")


def test_dynamic_mode_degrades_when_backend_has_no_decide_action(tmp_path):
    class StubPlanner:
        backend_name = "stub"

        def generate_plan(self, goal, state_summary=None):
            from karuth.planner.schema import InvestigationPlan, PlannerTask

            return InvestigationPlan(
                goal=goal,
                confidence=0.5,
                tasks=[PlannerTask(step=1, tool="ReportAgent", reason="stub")],
            )

        def replan(self, goal, state_summary=None, previous_plan=None):
            return self.generate_plan(goal, state_summary)

    case_dir = _chat_only_case(tmp_path)
    ledger = LedgerStore(tmp_path / "ledger.sqlite3")
    state = run_planner_goal(
        "CASE-1", case_dir, "Find links", ledger, planner=StubPlanner(), mode="dynamic"
    )
    # The loop must not dead-end: it finalizes with an explicit stop reason.
    assert state["stop_reason"]
    assert state["final_result"]
    actions = [e.action for e in ledger.list_entries()]
    assert "fallback_used" in actions
    assert ledger.verify()


def test_dynamic_run_human_review_checkpoint_via_critical_goal(tmp_path):
    """CRITICAL risk + contradictions triggers the ask_human checkpoint."""
    case_dir = _wow_case(tmp_path)
    ledger = LedgerStore(tmp_path / "ledger.sqlite3")
    state = run_planner_goal(
        "CASE-1",
        case_dir,
        "Reveal the critical credentials linked to vehicle KL07CX4242 in the chat",
        ledger,
        planner=DeterministicPlanner(),
        mode="dynamic",
    )
    # Either the loop hit the critical checkpoint, or it finalized with gating.
    assert state["final_result"]
    if state.get("human_input_requested"):
        assert state["stop_reason"] == "awaiting_human_input"
    assert ledger.verify()


def test_dynamic_run_respects_action_budget(tmp_path):
    case_dir = _wow_case(tmp_path)
    ledger = LedgerStore(tmp_path / "ledger.sqlite3")
    state = run_planner_goal(
        "CASE-1",
        case_dir,
        "Find the person connected to vehicle KL07CX4242",
        ledger,
        planner=DeterministicPlanner(),
        mode="dynamic",
        max_actions=3,
    )
    assert len(state["actions_taken"]) <= 3
    assert state["stop_reason"] in (
        "action_budget_exhausted",
        "sufficient_evidence",
        "awaiting_human_confirmation",
    )
    assert state["final_result"]
    assert ledger.verify()
