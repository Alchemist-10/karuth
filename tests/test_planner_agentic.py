"""Agentic planner behaviour tests: dynamic planning, routing, guardrails,
risk/reveal handling, replanning, and ledger integrity."""

from __future__ import annotations

import shutil

import pytest

from scripts.generate_synthetic_case import generate_case
from karuth.ledger import LedgerStore
from karuth.planner.llm import (
    DeterministicPlanner,
    FallbackPlanner,
    LMStudioPlanner,
    OpenRouterPlanner,
    build_planner,
)
from karuth.planner.schema import InvestigationPlan, PlannerTask
from karuth.planner.state_machine import run_planner_goal


def _chat_only_case(tmp_path):
    case_dir = tmp_path / "case"
    generate_case(case_dir)
    for subdir in ("images", "bank", "documents"):
        shutil.rmtree(case_dir / subdir)
    return case_dir


def _empty_case(tmp_path):
    case_dir = tmp_path / "empty"
    case_dir.mkdir()
    return case_dir


def _tool_set(state) -> set[str]:
    return {task.tool for task in state["plan"].tasks}


GOALS = {
    "network": "Find Suspect A in these videos, cross-reference with chat logs, and map their network.",
    "timeline": "Build a timeline for this missing person case.",
    "legal": "Generate the legal certificate for this case.",
    "sensitive": "Show me the high-risk images but keep sensitive content protected.",
}


def test_different_goals_produce_different_task_lists(tmp_path):
    case_dir = tmp_path / "synthetic_case"
    generate_case(case_dir)
    plans = {}
    for label, goal in GOALS.items():
        ledger = LedgerStore(tmp_path / f"ledger_{label}.sqlite3")
        state = run_planner_goal(
            "CASE-1", case_dir, goal, ledger, planner=DeterministicPlanner()
        )
        plans[label] = _tool_set(state)

    assert plans["network"] == {
        "ExtractionAgent",
        "CorrelationAgent",
        "TimelineAgent",
        "TriageAgent",
        "ReportAgent",
    }
    assert plans["timeline"] == {
        "ExtractionAgent",
        "TimelineAgent",
        "CorrelationAgent",
        "ReportAgent",
    }
    assert plans["legal"] == {"LegalActionAgent", "ReportAgent"}
    assert plans["sensitive"] == {
        "ExtractionAgent",
        "CorrelationAgent",
        "TriageAgent",
        "ReportAgent",
    }
    assert len({frozenset(tools) for tools in plans.values()}) == 4


def test_planner_stops_early_without_evidence(tmp_path):
    case_dir = _empty_case(tmp_path)
    ledger = LedgerStore(tmp_path / "ledger.sqlite3")

    state = run_planner_goal(
        "CASE-1", case_dir, "Investigate this case", ledger, planner=DeterministicPlanner()
    )

    assert state["entities"] == []
    assert state["stop_early"] is True
    assert any("stop_early" in note for note in state["notes"])
    assert state["final_result"]
    actions = [entry.action for entry in ledger.list_entries()]
    assert "planner_finished" in actions
    assert ledger.verify()


def test_planner_selects_legal_only_when_appropriate(tmp_path):
    case_dir = tmp_path / "synthetic_case"
    generate_case(case_dir)
    ledger = LedgerStore(tmp_path / "ledger.sqlite3")

    state = run_planner_goal(
        "CASE-1",
        case_dir,
        "Generate the legal certificate for this case.",
        ledger,
        planner=DeterministicPlanner(),
    )

    assert "LegalActionAgent" in _tool_set(state)
    assert "ReportAgent" in _tool_set(state)
    assert "ExtractionAgent" not in _tool_set(state)
    assert state["legal_output"]
    assert "Section 63" in state["legal_output"]
    assert "{{" not in state["legal_output"]
    assert ledger.verify()

    ledger2 = LedgerStore(tmp_path / "ledger2.sqlite3")
    state2 = run_planner_goal(
        "CASE-1",
        case_dir,
        "Find links for vehicle KL07CX4242",
        ledger2,
        planner=DeterministicPlanner(),
    )
    assert "LegalActionAgent" not in _tool_set(state2)


def test_planner_replans_when_correlation_finds_nothing(tmp_path):
    case_dir = _chat_only_case(tmp_path)
    ledger = LedgerStore(tmp_path / "ledger.sqlite3")

    state = run_planner_goal(
        "CASE-1",
        case_dir,
        "Cross-reference all chat contacts to find links.",
        ledger,
    )

    assert state["entities"]
    assert "TimelineAgent" in state["completed_steps"]
    actions = [entry.action for entry in ledger.list_entries()]
    assert "replan_triggered" in actions
    assert "replan_generated" in actions
    assert ledger.verify()


def test_planner_respects_med_risk_with_blurred_preview(tmp_path):
    case_dir = tmp_path / "synthetic_case"
    generate_case(case_dir)
    ledger = LedgerStore(tmp_path / "ledger.sqlite3")

    state = run_planner_goal(
        "CASE-1", case_dir, GOALS["sensitive"], ledger, planner=DeterministicPlanner()
    )

    assert state["plan"].safety_policy.risk_level == "MED"
    assert state["plan"].safety_policy.blur_by_default is True
    assert state["plan"].needs_human_review is True
    assert state["needs_human_review"] is True
    assert state["needs_authorization"] is False
    assert "SENSITIVE" in state["final_result"]
    assert "REDACTED" in state["final_result"]
    entry_types = {entry.entry_type for entry in ledger.list_entries()}
    assert "review_action" in entry_types
    assert ledger.verify()


def test_planner_critical_requires_dual_authorization(tmp_path):
    case_dir = tmp_path / "synthetic_case"
    generate_case(case_dir)
    ledger = LedgerStore(tmp_path / "ledger.sqlite3")

    state = run_planner_goal(
        "CASE-1",
        case_dir,
        "Reveal the critical credentials found in the chat.",
        ledger,
        planner=DeterministicPlanner(),
    )

    assert state["plan"].safety_policy.risk_level == "CRITICAL"
    assert state["plan"].safety_policy.dual_authorization_required is True
    assert state["needs_authorization"] is True
    assert "REVIEW_REQUIRED" in state["final_result"]
    assert any(entry.entry_type == "review_action" for entry in ledger.list_entries())
    assert ledger.verify()


def test_planner_critical_reveal_after_authorization(tmp_path):
    case_dir = tmp_path / "synthetic_case"
    generate_case(case_dir)
    ledger = LedgerStore(tmp_path / "ledger.sqlite3")

    state = run_planner_goal(
        "CASE-1",
        case_dir,
        "Reveal the critical credentials found in the chat.",
        ledger,
        planner=DeterministicPlanner(),
        reveal_authorized=True,
    )

    assert state["needs_authorization"] is False
    assert "REVIEW_REQUIRED" not in state["final_result"]
    decisions = [
        entry.payload.get("decision")
        for entry in ledger.list_entries()
        if entry.entry_type == "review_action"
    ]
    assert "approved_reveal" in decisions


def test_ledger_chain_remains_verifiable_across_runs(tmp_path):
    case_dir = tmp_path / "synthetic_case"
    generate_case(case_dir)
    ledger = LedgerStore(tmp_path / "ledger.sqlite3")

    for goal in GOALS.values():
        run_planner_goal(
            "CASE-1", case_dir, goal, ledger, planner=DeterministicPlanner()
        )
        assert ledger.verify()


def test_planner_default_backend_is_llm_first_with_fallback():
    planner = build_planner()
    assert isinstance(planner, FallbackPlanner)
    assert isinstance(planner.primary, OpenRouterPlanner)  # remote LLM default
    assert isinstance(planner.fallback, DeterministicPlanner)

    offline = build_planner("lmstudio")  # fully offline LLM option
    assert isinstance(offline.primary, LMStudioPlanner)


def test_deterministic_backend_is_explicitly_selectable():
    assert isinstance(build_planner("deterministic"), DeterministicPlanner)


def test_custom_planner_backend_is_injected(tmp_path):
    class StubPlanner:
        def generate_plan(self, goal, state_summary=None):
            return InvestigationPlan(
                goal=goal,
                confidence=0.5,
                tasks=[PlannerTask(step=1, tool="ReportAgent", reason="stub backend")],
            )

        def replan(self, goal, state_summary=None, previous_plan=None):
            return self.generate_plan(goal, state_summary)

    case_dir = tmp_path / "synthetic_case"
    generate_case(case_dir)
    ledger = LedgerStore(tmp_path / "ledger.sqlite3")

    state = run_planner_goal(
        "CASE-1",
        case_dir,
        "Anything",
        ledger,
        planner=StubPlanner(),
    )

    assert _tool_set(state) == {"ReportAgent"}
    assert "ExtractionAgent" not in state["completed_steps"]
    assert state["report"]
    assert ledger.verify()
