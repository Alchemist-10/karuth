"""Semantic planner tests: LM Studio backend, fallback, JSON validation,
dynamic execution, tool metadata routing, planner recovery, ledger logging,
replanning reasons, guardrails, and the enriched state."""

from __future__ import annotations

import json
import shutil
import types

import pytest

from scripts.generate_synthetic_case import generate_case
from karuth.ledger import LedgerStore
from karuth.planner.evidence import evidence_summary, inventory_case_evidence
from karuth.planner.llm import (
    DeterministicPlanner,
    FallbackPlanner,
    LMStudioPlanner,
    OpenRouterPlanner,
    PlannerLLM,
    build_planner,
)
from karuth.planner.schema import InvestigationPlan, PlannerTask
from karuth.planner.state_machine import run_planner_goal
from karuth.planner.tools import (
    PRODUCER_BY_OUTPUT,
    TOOL_REGISTRY,
    tool_metadata_text,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _plan_json(goal: str, tools: list[str], **overrides) -> str:
    tasks = [
        {
            "step": step,
            "tool": tool,
            "reason": f"goal requires {tool}",
            "inputs": [],
            "fields": {},
        }
        for step, tool in enumerate(tools, start=1)
    ]
    plan = {
        "goal": goal,
        "case_scope": "inside_case_folder_only",
        "confidence": 0.9,
        "needs_human_review": False,
        "safety_policy": {
            "risk_level": "LOW",
            "blur_by_default": False,
            "dual_authorization_required": False,
        },
        "tasks": tasks,
        "expected_output": ["test_output"],
        "stop_conditions": ["goal satisfied"],
        "reasoning_summary": "test plan",
    }
    plan.update(overrides)
    return json.dumps(plan)


class _FakeCompletions:
    def __init__(self, content: str):
        self._content = content
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = types.SimpleNamespace(content=self._content)
        choice = types.SimpleNamespace(message=message)
        usage = types.SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        return types.SimpleNamespace(choices=[choice], usage=usage)


class _FakeClient:
    def __init__(self, content: str):
        self.chat = types.SimpleNamespace(completions=_FakeCompletions(content))


class _SpyPlanner:
    """Records every state summary it sees, then delegates deterministically."""

    backend_name = "spy"

    def __init__(self) -> None:
        self.summaries: list[dict] = []
        self._deterministic = DeterministicPlanner()

    def generate_plan(self, goal, state_summary=None):
        self.summaries.append(state_summary)
        return self._deterministic.generate_plan(goal, state_summary)

    def replan(self, goal, state_summary=None, previous_plan=None):
        self.summaries.append(state_summary)
        return self._deterministic.replan(goal, state_summary, previous_plan)


class _FailingPlanner:
    """A backend that is never available."""

    backend_name = "failing"

    def generate_plan(self, goal, state_summary=None):
        raise RuntimeError("model unavailable")

    def replan(self, goal, state_summary=None, previous_plan=None):
        raise RuntimeError("model unavailable")


class _PartialPlanPlanner:
    """Planner that schedules only a subset (to exercise dependency skips)."""

    backend_name = "partial"

    def __init__(self, tools: list[str]) -> None:
        self._tools = tools
        self._deterministic = DeterministicPlanner()

    def generate_plan(self, goal, state_summary=None):
        return InvestigationPlan(
            goal=goal,
            confidence=0.5,
            tasks=[
                PlannerTask(step=step, tool=tool, reason="custom plan")
                for step, tool in enumerate(self._tools, start=1)
            ],
        )

    def replan(self, goal, state_summary=None, previous_plan=None):
        return self._deterministic.replan(goal, state_summary, previous_plan)


def _tool_set(state) -> set[str]:
    return {task.tool for task in state["plan"].tasks}


def _actions(ledger) -> list[str]:
    return [entry.action for entry in ledger.list_entries()]


def _chat_only_case(tmp_path):
    case_dir = tmp_path / "case"
    generate_case(case_dir)
    for subdir in ("images", "bank", "documents"):
        shutil.rmtree(case_dir / subdir)
    return case_dir


# ---------------------------------------------------------------------------
# evidence inventory
# ---------------------------------------------------------------------------


def test_evidence_inventory_lists_supported_kinds(tmp_path):
    case_dir = tmp_path / "synthetic_case"
    generate_case(case_dir)
    items = inventory_case_evidence(case_dir)
    kinds = {item.kind for item in items}
    assert kinds == {"manifest", "chats", "images", "bank", "documents"}
    chats = next(item for item in items if item.kind == "chats")
    assert chats.file_count >= 1
    summary = evidence_summary(items)
    assert "chats" in summary
    assert "images" in summary


def test_evidence_inventory_empty_folder(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert inventory_case_evidence(empty) == []
    assert "No evidence files" in evidence_summary([])


# ---------------------------------------------------------------------------
# LM Studio backend
# ---------------------------------------------------------------------------


def test_lmstudio_planner_builds_valid_plan_from_rest_endpoint():
    goal = "Find links for vehicle KL07CX4242"
    fake = _FakeClient(_plan_json(goal, ["ExtractionAgent", "CorrelationAgent", "ReportAgent"]))
    planner = LMStudioPlanner(
        client=fake,
        model="test-model",
        base_url="http://localhost:1234/v1",
        temperature=0.2,
        max_tokens=256,
        timeout=9.5,
    )

    plan = planner.generate_plan(
        goal,
        {"case_id": "C-1", "available_evidence": [{"kind": "chats", "file_count": 1}]},
    )

    assert isinstance(plan, InvestigationPlan)
    assert [task.tool for task in plan.tasks] == [
        "ExtractionAgent",
        "CorrelationAgent",
        "ReportAgent",
    ]
    assert planner.backend_name == "lmstudio"
    assert planner.last_usage == {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}

    kwargs = fake.chat.completions.calls[0]
    assert kwargs["model"] == "test-model"
    assert kwargs["temperature"] == 0.2
    assert kwargs["max_tokens"] == 256
    assert kwargs["timeout"] == 9.5
    messages = kwargs["messages"]
    system = messages[0]["content"]
    user = messages[1]["content"]
    assert "ExtractionAgent" in system
    assert "EVIDENCE INVENTORY OF THE CASE FOLDER" in system
    assert "chats: 1 file" in system
    assert goal in user
    assert "C-1" in user


def test_lmstudio_planner_repairs_fenced_json():
    goal = "Build a timeline"
    payload = _plan_json(goal, ["ExtractionAgent", "TimelineAgent", "ReportAgent"])
    planner = LMStudioPlanner(client=_FakeClient(f"Sure, here it is:\n```json\n{payload}\n```"))
    plan = planner.generate_plan(goal, {})
    assert [task.tool for task in plan.tasks] == [
        "ExtractionAgent",
        "TimelineAgent",
        "ReportAgent",
    ]


def test_openrouter_configuration_is_configurable_via_factory():
    planner = build_planner(
        "openrouter",
        fallback=False,
        model="openai/gpt-4o-mini",
        base_url="https://openrouter.ai/api/v1",
        temperature=0.2,
        max_tokens=400,
        timeout=30.0,
        api_key="test-key",
    )
    assert isinstance(planner, OpenRouterPlanner)
    assert planner._model == "openai/gpt-4o-mini"
    assert planner._base_url == "https://openrouter.ai/api/v1"
    assert planner._temperature == 0.2
    assert planner._max_tokens == 400
    assert planner._timeout == 30.0
    assert planner._api_key == "test-key"


def test_openrouter_refuses_without_api_key():
    import os

    planner = OpenRouterPlanner(api_key=None)
    assert planner._api_key == os.environ.get("OPENROUTER_API_KEY", "")
    if not planner._api_key:
        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
            planner._get_client()


def test_openrouter_falls_back_to_deterministic_on_failure(tmp_path):
    """OpenRouter errors (bad key, network, invalid JSON) degrade to the
    deterministic planner via FallbackPlanner, exactly like LM Studio."""
    case_dir = tmp_path / "synthetic_case"
    generate_case(case_dir)
    ledger = LedgerStore(tmp_path / "ledger.sqlite3")

    planner = FallbackPlanner(OpenRouterPlanner(client=_FakeClient("not json at all"), api_key="x"))
    state = run_planner_goal(
        "CASE-1",
        case_dir,
        "Find links for vehicle KL07CX4242",
        ledger,
        planner=planner,
    )
    assert state["final_result"]
    assert "fallback_used" in [entry.action for entry in ledger.list_entries()]
    assert ledger.verify()


def test_lmstudio_configuration_is_configurable_via_factory():
    planner = build_planner(
        "lmstudio",
        fallback=False,
        model="custom-model",
        base_url="http://localhost:9999/v1",
        temperature=0.3,
        max_tokens=500,
        timeout=12.5,
    )
    assert isinstance(planner, LMStudioPlanner)
    assert planner._model == "custom-model"
    assert planner._base_url == "http://localhost:9999/v1"
    assert planner._temperature == 0.3
    assert planner._max_tokens == 500
    assert planner._timeout == 12.5


def test_build_planner_factory_provider_abstraction():
    # The default backend is OpenRouter (remote LLM planning), wrapped in the
    # deterministic fallback so an unreachable/unauthorized LLM never dead-ends.
    default = build_planner()
    assert isinstance(default, FallbackPlanner)
    assert isinstance(default.primary, OpenRouterPlanner)
    assert isinstance(default.fallback, DeterministicPlanner)

    assert isinstance(build_planner("deterministic"), DeterministicPlanner)

    lmstudio = build_planner("lmstudio")
    assert isinstance(lmstudio, FallbackPlanner)
    assert isinstance(lmstudio.primary, LMStudioPlanner)
    assert isinstance(lmstudio.fallback, DeterministicPlanner)

    assert isinstance(build_planner("lmstudio", fallback=False), LMStudioPlanner)

    openrouter = build_planner("openrouter")
    assert isinstance(openrouter, FallbackPlanner)
    assert isinstance(openrouter.primary, OpenRouterPlanner)
    assert isinstance(openrouter.fallback, DeterministicPlanner)
    assert isinstance(build_planner("openrouter", fallback=False), OpenRouterPlanner)

    with pytest.raises(ValueError, match="Unknown planner backend"):
        build_planner("bogus")
    # Ollama was removed; only LM Studio, OpenRouter, and deterministic remain.
    with pytest.raises(ValueError, match="Unknown planner backend"):
        build_planner("ollama")


# ---------------------------------------------------------------------------
# fallback behaviour
# ---------------------------------------------------------------------------


def test_llm_failure_falls_back_to_deterministic_end_to_end(tmp_path):
    case_dir = tmp_path / "synthetic_case"
    generate_case(case_dir)
    ledger = LedgerStore(tmp_path / "ledger.sqlite3")

    planner = LMStudioPlanner(client=_FakeClient("this is not JSON at all"))
    state = run_planner_goal(
        "CASE-1",
        case_dir,
        "Find links for vehicle KL07CX4242",
        ledger,
        planner=planner,
    )

    assert "ExtractionAgent" in _tool_set(state)
    assert state["final_result"]
    actions = _actions(ledger)
    assert "fallback_used" in actions
    fallback = next(entry for entry in ledger.list_entries() if entry.action == "fallback_used")
    assert fallback.payload["primary"] == "lmstudio"
    assert fallback.payload["fallback"] == "DeterministicPlanner"
    assert ledger.verify()


def test_fallback_planner_wrapper_records_and_logs_fallback(tmp_path):
    case_dir = tmp_path / "synthetic_case"
    generate_case(case_dir)
    ledger = LedgerStore(tmp_path / "ledger.sqlite3")

    planner = FallbackPlanner(_FailingPlanner())
    state = run_planner_goal("CASE-1", case_dir, "Investigate the bank records", ledger, planner=planner)

    assert state["final_result"]
    assert "ExtractionAgent" in _tool_set(state)
    assert "fallback_used" in _actions(ledger)
    assert ledger.verify()

    direct = FallbackPlanner(_FailingPlanner())
    plan = direct.generate_plan("anything", {})
    assert isinstance(plan, InvestigationPlan)
    assert plan.tasks


def test_run_planner_goal_defaults_to_llm_first_and_falls_back(tmp_path, monkeypatch):
    """Default experience is LLM-backed; an unavailable LLM degrades to the
    deterministic planner and is recorded in the ledger."""
    case_dir = tmp_path / "synthetic_case"
    generate_case(case_dir)
    ledger = LedgerStore(tmp_path / "ledger.sqlite3")

    import karuth.planner.state_machine as sm

    def _failing_default(backend: str = "openrouter", **kwargs):
        assert backend == "openrouter"
        return FallbackPlanner(_FailingPlanner())

    monkeypatch.setattr(sm, "build_planner", _failing_default)

    state = sm.run_planner_goal(
        "CASE-1",
        case_dir,
        "Find links for vehicle KL07CX4242",
        ledger,
    )

    assert state["final_result"]
    assert "ExtractionAgent" in _tool_set(state)
    assert "fallback_used" in _actions(ledger)
    fallback = next(entry for entry in ledger.list_entries() if entry.action == "fallback_used")
    assert "model unavailable" in fallback.payload["error"]
    assert ledger.verify()


# ---------------------------------------------------------------------------
# dynamic execution
# ---------------------------------------------------------------------------


def test_dynamic_execution_runs_one_task_per_iteration(tmp_path):
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

    history = state["execution_history"]
    assert len(history) == 4
    assert [record["tool"] for record in history] == [
        "ExtractionAgent",
        "CorrelationAgent",
        "TriageAgent",
        "ReportAgent",
    ]
    assert all(record["status"] == "completed" for record in history)
    assert state["planner_iteration_count"] == 4
    assert len({record["tool"] for record in history}) == 4


def test_dependency_aware_selection_skips_unschedulable_tasks(tmp_path):
    case_dir = tmp_path / "synthetic_case"
    generate_case(case_dir)
    ledger = LedgerStore(tmp_path / "ledger.sqlite3")

    state = run_planner_goal(
        "CASE-1",
        case_dir,
        "Find links",
        ledger,
        planner=_PartialPlanPlanner(["CorrelationAgent", "ReportAgent"]),
    )

    actions = _actions(ledger)
    assert "task_skipped" in actions
    skipped = next(entry for entry in ledger.list_entries() if entry.action == "task_skipped")
    assert skipped.payload["tool"] == "CorrelationAgent"
    assert "cannot be produced" in skipped.payload["reason"]
    # replanning must have introduced the missing producer
    assert "ExtractionAgent" in state["completed_steps"]
    assert ledger.verify()


# ---------------------------------------------------------------------------
# recovery
# ---------------------------------------------------------------------------


def test_planner_recovers_from_corrupt_evidence(tmp_path):
    case_dir = tmp_path / "synthetic_case"
    generate_case(case_dir)
    (case_dir / "chats" / "synthetic_whatsapp.sqlite").write_bytes(b"this is not a sqlite db")
    ledger = LedgerStore(tmp_path / "ledger.sqlite3")

    state = run_planner_goal(
        "CASE-1", case_dir, "Investigate this case", ledger, planner=DeterministicPlanner()
    )

    assert state["final_result"]
    actions = _actions(ledger)
    assert "task_failed" in actions
    assert "planner_finished" in actions
    assert ledger.verify()


def test_planner_tolerates_empty_case_folder(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    ledger = LedgerStore(tmp_path / "ledger.sqlite3")

    state = run_planner_goal(
        "CASE-1", empty, "Investigate this case", ledger, planner=DeterministicPlanner()
    )

    assert state["entities"] == []
    assert state["stop_early"] is True
    assert state["final_result"]
    assert ledger.verify()


# ---------------------------------------------------------------------------
# replanning reasons
# ---------------------------------------------------------------------------


def test_replanning_records_explicit_reason(tmp_path):
    case_dir = _chat_only_case(tmp_path)
    ledger = LedgerStore(tmp_path / "ledger.sqlite3")

    state = run_planner_goal(
        "CASE-1",
        case_dir,
        "Cross-reference all chat contacts to find links.",
        ledger,
        planner=DeterministicPlanner(),
    )

    triggers = [
        entry for entry in ledger.list_entries() if entry.action == "replan_triggered"
    ]
    assert triggers
    assert triggers[0].payload["reason"] == "correlation returned no matches"
    assert any("replan: correlation returned no matches" in note for note in state["notes"])
    assert "TimelineAgent" in state["completed_steps"]
    assert ledger.verify()


def test_replanning_when_tool_fails_explains_reason(tmp_path):
    case_dir = tmp_path / "synthetic_case"
    generate_case(case_dir)
    (case_dir / "chats" / "synthetic_whatsapp.sqlite").write_bytes(b"broken")
    ledger = LedgerStore(tmp_path / "ledger.sqlite3")

    run_planner_goal(
        "CASE-1", case_dir, "Investigate this case", ledger, planner=DeterministicPlanner()
    )

    triggers = [
        entry for entry in ledger.list_entries() if entry.action == "replan_triggered"
    ]
    assert triggers
    assert "failed" in triggers[0].payload["reason"]
    assert ledger.verify()


# ---------------------------------------------------------------------------
# ledger coverage
# ---------------------------------------------------------------------------


def test_ledger_records_all_decision_events(tmp_path):
    case_dir = tmp_path / "synthetic_case"
    generate_case(case_dir)
    ledger = LedgerStore(tmp_path / "ledger.sqlite3")

    run_planner_goal(
        "CASE-1",
        case_dir,
        "Find links for vehicle KL07CX4242 and triage the top leads",
        ledger,
        planner=DeterministicPlanner(),
    )

    actions = set(_actions(ledger))
    for expected in (
        "goal_received",
        "planning_backend",
        "evidence_inventory",
        "plan_generated",
        "task_selected",
        "task_dispatched",
        "task_completed",
        "postcondition_checked",
        "goal_completed",
        "report_exported",
        "planner_finished",
        "chain_verified",
    ):
        assert expected in actions, f"missing ledger action: {expected}"
    entry_types = {entry.entry_type for entry in ledger.list_entries()}
    assert "agent_action" in entry_types
    assert "evidence" in entry_types
    assert ledger.verify()


# ---------------------------------------------------------------------------
# semantic planning context
# ---------------------------------------------------------------------------


def test_semantic_state_summary_reaches_planner(tmp_path):
    case_dir = _chat_only_case(tmp_path)
    ledger = LedgerStore(tmp_path / "ledger.sqlite3")
    spy = _SpyPlanner()

    run_planner_goal(
        "CASE-1",
        case_dir,
        "Cross-reference all chat contacts to find links.",
        ledger,
        planner=spy,
    )

    assert len(spy.summaries) >= 2  # initial plan + replan
    initial = spy.summaries[0]
    assert initial["case_id"] == "CASE-1"
    assert initial["completed_steps"] == []
    assert initial["entity_count"] == 0
    evidence = initial["available_evidence"]
    assert evidence
    kinds = {item["kind"] for item in evidence}
    assert "chats" in kinds

    replan_summary = spy.summaries[1]
    assert replan_summary["replan_reason"] == "correlation returned no matches"
    assert replan_summary["completed_steps"] == ["ExtractionAgent", "CorrelationAgent"]
    assert replan_summary["pending_tools"] == ["ReportAgent"]


def test_state_carries_richer_planner_fields(tmp_path):
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

    for key in (
        "available_evidence",
        "tool_history",
        "execution_history",
        "planner_thoughts",
        "confidence",
        "remaining_objectives",
        "planner_iteration_count",
        "token_usage",
        "model_used",
        "timing",
    ):
        assert key in state, f"missing state field: {key}"
    assert state["available_evidence"]
    assert state["tool_history"]
    assert state["execution_history"]
    assert state["planner_thoughts"]
    assert state["confidence"] > 0
    assert state["planner_iteration_count"] >= 1
    assert state["model_used"]
    assert "plan" in state["timing"]
    assert "dispatch" in state["timing"]


# ---------------------------------------------------------------------------
# guardrails against LLM plans
# ---------------------------------------------------------------------------


def test_guardrails_block_llm_external_url_in_plan(tmp_path):
    case_dir = tmp_path / "synthetic_case"
    generate_case(case_dir)
    ledger = LedgerStore(tmp_path / "ledger.sqlite3")

    evil = json.loads(_plan_json("Find links", ["ExtractionAgent"]))
    evil["tasks"][0]["inputs"] = ["https://evil.example.com/external.sqlite"]
    planner = LMStudioPlanner(client=_FakeClient(json.dumps(evil)))

    with pytest.raises(ValueError, match="External URL"):
        run_planner_goal("CASE-1", case_dir, "Find links", ledger, planner=planner)


def test_tool_registry_metadata_is_complete():
    assert set(TOOL_REGISTRY) == {
        "ExtractionAgent",
        "CorrelationAgent",
        "TimelineAgent",
        "TriageAgent",
        "ReportAgent",
        "LegalActionAgent",
        "VisionAgent",
    }
    for name, spec in TOOL_REGISTRY.items():
        assert spec.description
        assert spec.capabilities
        assert spec.produced_outputs
        # VisionAgent is the only LLM-dependent tool (needs LM Studio vision).
        assert spec.offline is (name != "VisionAgent")
    assert PRODUCER_BY_OUTPUT["entities"] == "ExtractionAgent"
    assert PRODUCER_BY_OUTPUT["matches"] == "CorrelationAgent"
    assert PRODUCER_BY_OUTPUT["report"] == "ReportAgent"
    assert PRODUCER_BY_OUTPUT["image_transcriptions"] == "VisionAgent"
    text = tool_metadata_text()
    for name in TOOL_REGISTRY:
        assert name in text
