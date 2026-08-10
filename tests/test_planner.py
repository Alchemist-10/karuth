from __future__ import annotations

import pytest

from scripts.generate_synthetic_case import generate_case
from karuth.ledger import LedgerStore
from karuth.planner.guardrails import assert_plan_safe, ensure_inside_case, reject_external_url
from karuth.planner.llm import DeterministicPlanner
from karuth.planner.schema import InvestigationPlan, PlannerTask
from karuth.planner.state_machine import run_planner_goal


def test_planner_produces_dynamic_plan_and_audits(tmp_path):
    case_dir = tmp_path / "synthetic_case"
    generate_case(case_dir)
    ledger = LedgerStore(tmp_path / "ledger.sqlite3")

    state = run_planner_goal(
        "CASE-1",
        case_dir,
        "Find links for vehicle KL07CX4242 and triage the top leads",
        ledger,
        planner=DeterministicPlanner(),
    )

    assert state["entities"]
    assert state["matches"]
    assert state["ranked_matches"]
    assert state["final_result"]
    assert state["plan"].tasks
    assert "ExtractionAgent" in {task.tool for task in state["plan"].tasks}
    assert "CorrelationAgent" in {task.tool for task in state["plan"].tasks}

    actions = [entry.action for entry in ledger.list_entries()]
    assert "goal_received" in actions
    assert "plan_generated" in actions
    assert "task_dispatched" in actions
    assert "task_completed" in actions
    assert "planner_finished" in actions
    assert ledger.verify()


def test_guardrails_reject_out_of_scope_and_external_urls(tmp_path):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("nope", encoding="utf-8")

    with pytest.raises(ValueError):
        ensure_inside_case(case_dir, outside)

    with pytest.raises(ValueError):
        reject_external_url("https://example.com/data")


def test_assert_plan_safe_rejects_external_task_inputs(tmp_path):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    plan = InvestigationPlan(
        goal="safe goal",
        confidence=0.5,
        tasks=[
            PlannerTask(
                step=1,
                tool="ExtractionAgent",
                reason="test",
                inputs=["https://example.com/external.sqlite"],
            )
        ],
    )
    with pytest.raises(ValueError):
        assert_plan_safe(plan, case_dir)
