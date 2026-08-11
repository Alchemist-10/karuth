"""LangGraph node functions for the agentic planner.

Each node is a pure-ish function over :class:`PlannerState` with injected
dependencies (the planning backend and the ledger auditor). The graph wiring in
``graph.py`` runs the reasoning loop:

    plan -> select -> dispatch -> check -> (select | replan -> select | finalize)

``plan_step`` generates a semantic plan, ``select_step`` picks exactly one
schedulable task (respecting tool dependencies), ``dispatch_step`` executes
that one task, ``check_step`` evaluates postconditions and explains why the
loop continues, replans, or finalizes, and ``replan_step`` revises the plan
with an explicit reason. The LLM is only ever used for planning decisions;
every tool execution stays deterministic.
"""

from __future__ import annotations

import time

from karuth.agents.report_agent import summarize_matches
from karuth.planner.audit import PlannerAuditor
from karuth.planner.evidence import EvidenceItem, inventory_case_evidence
from karuth.planner.guardrails import assert_plan_safe, reject_external_url
from karuth.planner.llm import DeterministicPlanner, PlannerLLM
from karuth.planner.postconditions import has_entities, has_matches, has_timeline
from karuth.planner.router import mask_report, route_task
from karuth.planner.schema import InvestigationPlan, PlannerTask
from karuth.planner.state import PlannerState
from karuth.planner.tools import PRODUCER_BY_OUTPUT, TOOL_REGISTRY


def _backend_name(planner: object) -> str:
    return getattr(planner, "backend_name", type(planner).__name__)


def _record_timing(state: PlannerState, key: str, seconds: float) -> dict[str, float]:
    timing = dict(state.get("timing") or {})
    timing[key] = round(float(timing.get(key, 0.0)) + seconds, 4)
    return timing


def _evidence_dicts(evidence: list[EvidenceItem]) -> list[dict]:
    return [item.model_dump(mode="json") for item in evidence]


def _summary(state: PlannerState) -> dict[str, object]:
    """Audit-safe state summary handed to the planning backend."""
    evidence = state.get("available_evidence", [])
    return {
        "case_id": state.get("case_id", ""),
        "entity_count": len(state.get("entities", [])),
        "match_count": len(state.get("matches", [])),
        "ranked_count": len(state.get("ranked_matches", [])),
        "timeline_count": len(state.get("timeline", [])),
        "has_report": bool(state.get("report")),
        "completed_steps": state.get("completed_steps", []),
        "pending_tools": [task.tool for task in state.get("pending_steps", [])],
        "available_evidence": [
            item.model_dump(mode="json") if hasattr(item, "model_dump") else item
            for item in evidence
        ],
        "confidence": state.get("confidence", 0.0),
        "planner_iteration_count": state.get("planner_iteration_count", 0),
        "replan_count": state.get("replan_count", 0),
        "replan_reason": state.get("replan_reason", ""),
        "failed_tools": [
            entry["tool"]
            for entry in state.get("execution_history", [])
            if entry.get("status") == "failed"
        ],
        "notes": state.get("notes", []),
    }


def _log_recorded_fallback(planner: object, auditor: PlannerAuditor) -> None:
    """Surface a fallback that a wrapper (e.g. FallbackPlanner) already handled."""
    last_error = getattr(planner, "last_error", None)
    if last_error:
        auditor.fallback_used(_backend_name(planner), str(last_error))
        setattr(planner, "last_error", None)


def _safe_generate(planner: PlannerLLM, goal: str, summary: dict, auditor: PlannerAuditor) -> InvestigationPlan:
    """Generate a plan, falling back to the deterministic planner on failure."""
    try:
        plan = planner.generate_plan(goal, summary)
    except Exception as exc:  # noqa: BLE001 - planner must degrade gracefully
        auditor.fallback_used(_backend_name(planner), str(exc))
        return DeterministicPlanner().generate_plan(goal, summary)
    _log_recorded_fallback(planner, auditor)
    return plan


def _safe_replan(
    planner: PlannerLLM,
    goal: str,
    summary: dict,
    previous_plan: InvestigationPlan,
    auditor: PlannerAuditor,
) -> InvestigationPlan:
    """Replan, falling back to the deterministic planner on failure."""
    try:
        plan = planner.replan(goal, summary, previous_plan)
    except Exception as exc:  # noqa: BLE001 - planner must degrade gracefully
        auditor.fallback_used(_backend_name(planner), str(exc))
        return DeterministicPlanner().replan(goal, summary, previous_plan)
    _log_recorded_fallback(planner, auditor)
    return plan


def plan_step(
    state: PlannerState, planner: PlannerLLM, auditor: PlannerAuditor
) -> dict[str, object]:
    """Interpret the goal, guard it, inventory evidence, and generate the plan."""
    goal = state["goal"]
    reject_external_url(goal)
    auditor.goal_received(goal)

    evidence = inventory_case_evidence(state["case_folder"])
    auditor.evidence_inventory(_evidence_dicts(evidence))

    summary = _summary(state)
    summary["available_evidence"] = _evidence_dicts(evidence)

    start = time.perf_counter()
    plan = _safe_generate(planner, goal, summary, auditor)
    elapsed = time.perf_counter() - start

    assert_plan_safe(plan, state["case_folder"])
    auditor.planning_backend(_backend_name(planner), getattr(planner, "_model", ""))
    auditor.plan_generated(plan)

    tool_list = [task.tool for task in plan.tasks]
    notes = list(state.get("notes", [])) + [f"plan generated: {tool_list}"]
    thoughts = [f"plan generated: {tool_list} (confidence {plan.confidence:.2f})"]
    return {
        "plan": plan,
        "pending_steps": list(plan.tasks),
        "remaining_objectives": list(tool_list),
        "completed_steps": [],
        "risk_tier": plan.safety_policy.risk_level,
        "needs_human_review": plan.needs_human_review,
        "replan_count": 0,
        "confidence": plan.confidence,
        "available_evidence": evidence,
        "planner_thoughts": thoughts,
        "token_usage": dict(getattr(planner, "last_usage", None) or {}),
        "model_used": getattr(planner, "_model", "") or _backend_name(planner),
        "timing": _record_timing(state, "plan", elapsed),
        "next_action": "dispatch",
        "notes": notes,
    }


def _order_tasks(tasks: list[PlannerTask], completed: set[str]) -> list[PlannerTask]:
    """Dependency-aware stable ordering: a task waits until its producers ran."""
    produced: set[str] = set()
    for tool in completed:
        spec = TOOL_REGISTRY.get(tool)
        if spec is not None:
            produced.update(spec.produced_keys)
    ordered: list[PlannerTask] = []
    remaining = list(tasks)
    while remaining:
        progressed = False
        for task in list(remaining):
            spec = TOOL_REGISTRY.get(task.tool)
            needed = set(spec.required_inputs) - produced if spec else set()
            if not needed:
                ordered.append(task)
                remaining.remove(task)
                if spec is not None:
                    produced.update(spec.produced_keys)
                progressed = True
                break
        if not progressed:
            ordered.extend(remaining)
            break
    return ordered


def _hard_blocked(task: PlannerTask, completed: set[str], pending: list[PlannerTask]) -> list[str]:
    """Return required inputs whose producer can never run (not completed, not scheduled)."""
    spec = TOOL_REGISTRY.get(task.tool)
    if spec is None:
        return []
    missing: list[str] = []
    for key in spec.required_inputs:
        producer = PRODUCER_BY_OUTPUT.get(key)
        if producer is None:
            continue
        if producer not in completed and not any(t.tool == producer for t in pending):
            missing.append(key)
    return missing


def select_step(state: PlannerState, auditor: PlannerAuditor) -> dict[str, object]:
    """Pick exactly one schedulable task, skipping tasks that can never run."""
    completed = set(state.get("completed_steps", []))
    pending = [task for task in state.get("pending_steps", []) if task.tool not in completed]
    ordered = _order_tasks(pending, completed)

    notes = list(state.get("notes", []))
    thoughts = list(state.get("planner_thoughts", []))
    selected: PlannerTask | None = None
    remaining: list[PlannerTask] = []
    for task in ordered:
        missing = _hard_blocked(task, completed, pending)
        if missing:
            reason = f"required inputs cannot be produced: {', '.join(missing)}"
            auditor.task_skipped(task.tool, reason)
            notes.append(f"{task.tool} skipped: {reason}")
            thoughts.append(f"skip {task.tool}: {reason}")
            continue
        selected = task
        break
    remaining = [task for task in ordered if task is not selected]

    updates: dict[str, object] = {
        "pending_steps": remaining,
        "remaining_objectives": [task.tool for task in remaining],
        "notes": notes,
        "planner_thoughts": thoughts,
    }
    if selected is None:
        updates["next_action"] = "replan"
        updates["needs_replan"] = True
        updates["replan_reason"] = "no selectable tasks remain"
    else:
        updates["current_task"] = selected
        updates["next_action"] = "dispatch"
        thoughts.append(f"select {selected.tool}")
        updates["planner_thoughts"] = thoughts
    return updates


def dispatch_step(state: PlannerState, auditor: PlannerAuditor) -> dict[str, object]:
    """Execute exactly one selected task through the deterministic router."""
    task = state.get("current_task")
    notes = list(state.get("notes", []))
    thoughts = list(state.get("planner_thoughts", []))
    if task is None:
        notes.append("dispatch called without a current task")
        return {
            "notes": notes,
            "last_dispatch_had_work": False,
            "next_action": "replan",
            "needs_replan": True,
            "replan_reason": "no task selected for dispatch",
        }

    auditor.task_selected(task.tool, task.reason)
    auditor.task_dispatched(task.tool, task.inputs)

    start = time.perf_counter()
    try:
        result, summary = route_task(task, state)
        status, error, outcome = "completed", None, summary
    except Exception as exc:  # noqa: BLE001 - planner must degrade gracefully
        auditor.task_failed(task.tool, str(exc))
        notes.append(f"{task.tool} failed: {exc}")
        status, error, outcome = "failed", str(exc), f"failed: {exc}"
    elapsed = time.perf_counter() - start

    updates: dict[str, object] = {}
    if status == "completed":
        for key, value in result.items():
            updates[key] = value
        completed = list(state.get("completed_steps", []))
        if task.tool not in completed:
            completed.append(task.tool)
        updates["completed_steps"] = completed
        auditor.task_completed(task.tool, outcome)
        thoughts.append(f"done {task.tool}: {outcome}")
    else:
        thoughts.append(f"failed {task.tool}: {error}")

    execution = list(state.get("execution_history", []))
    execution.append(
        {
            "tool": task.tool,
            "status": status,
            "summary": outcome,
            "duration_s": round(elapsed, 4),
            "error": error,
        }
    )
    tool_history = list(state.get("tool_history", []))
    tool_history.append(
        {"tool": task.tool, "status": status, "at_iteration": state.get("planner_iteration_count", 0)}
    )

    updates["execution_history"] = execution
    updates["tool_history"] = tool_history
    updates["last_dispatch_had_work"] = status == "completed"
    updates["planner_iteration_count"] = state.get("planner_iteration_count", 0) + 1
    updates["timing"] = _record_timing(state, "dispatch", elapsed)
    updates["notes"] = notes
    updates["planner_thoughts"] = thoughts

    if "entities" in updates:
        auditor.evidence_ingested(len(updates["entities"]))  # type: ignore[arg-type]
    return updates


def check_step(state: PlannerState, auditor: PlannerAuditor) -> dict[str, object]:
    """Evaluate postconditions and decide (with an explanation) what happens next."""
    entities = list(state.get("entities", []))
    matches = list(state.get("matches", []))
    timeline = list(state.get("timeline", []))
    report = state.get("report", "")
    completed = state.get("completed_steps", [])
    pending = state.get("pending_steps", [])
    notes = list(state.get("notes", []))
    thoughts = list(state.get("planner_thoughts", []))

    entities_ok = has_entities(entities)
    matches_ok = has_matches(matches)
    timeline_ok = has_timeline(timeline)
    report_ok = bool(report)
    for name, ok in (
        ("has_entities", entities_ok),
        ("has_matches", matches_ok),
        ("has_timeline", timeline_ok),
        ("has_report", report_ok),
    ):
        auditor.postcondition(name, ok)

    execution_history = state.get("execution_history", [])
    last = execution_history[-1] if execution_history else None
    last_tool = last["tool"] if last else None
    last_status = last["status"] if last else None

    next_action = "continue"
    reason = ""
    stop_early = False
    needs_replan = False

    if "ExtractionAgent" in completed and not entities_ok:
        stop_early = True
        reason = "no evidence entities extracted"
        next_action = "finalize"
    elif last_status == "failed":
        needs_replan = True
        reason = f"tool {last_tool} failed"
        next_action = "replan"
    elif last_tool == "CorrelationAgent" and last_status == "completed" and not matches_ok:
        needs_replan = True
        reason = "correlation returned no matches"
        next_action = "replan"
    elif not pending and last_status == "completed":
        if matches_ok or report_ok or state.get("legal_output"):
            reason = "planned tasks completed with sufficient output"
            next_action = "finalize"
        else:
            needs_replan = True
            reason = "plan exhausted without sufficient evidence"
            next_action = "replan"
    elif not pending:
        reason = "no pending tasks remain"
        next_action = "finalize"
    else:
        reason = f"continuing after {last_tool}" if last_tool else "beginning task loop"

    if stop_early:
        notes.append(f"stop_early: {reason}")
    else:
        notes.append(f"check: {reason}")
    thoughts.append(f"check: {reason}")

    return {
        "next_action": next_action,
        "replan_reason": reason if needs_replan else state.get("replan_reason", ""),
        "stop_early": stop_early,
        "needs_replan": needs_replan,
        "notes": notes,
        "planner_thoughts": thoughts,
    }


def replan_step(
    state: PlannerState, planner: PlannerLLM, auditor: PlannerAuditor
) -> dict[str, object]:
    """Revise the plan when the loop stalls, explaining why replanning occurred."""
    reason = state.get("replan_reason") or "first pass insufficient"
    auditor.replan_triggered(reason)

    notes = list(state.get("notes", [])) + [f"replan: {reason}"]
    thoughts = list(state.get("planner_thoughts", [])) + [f"replan: {reason}"]

    summary = _summary(state)
    summary["replan_reason"] = reason
    plan = _safe_replan(planner, state["goal"], summary, state.get("plan"), auditor)
    assert_plan_safe(plan, state["case_folder"])
    auditor.plan_generated(plan, replan=True)

    completed = set(state.get("completed_steps", []))
    pending = [task for task in plan.tasks if task.tool not in completed]
    tool_list = [task.tool for task in plan.tasks]

    needs_replan = True
    if not pending:
        needs_replan = False
        notes.append("replan produced no new steps; finalizing")
        thoughts.append("replan produced no new steps; finalizing")
    notes.append(f"replanned to: {tool_list}")
    thoughts.append(f"replanned to: {tool_list}")

    return {
        "plan": plan,
        "pending_steps": pending,
        "remaining_objectives": [task.tool for task in pending],
        "replan_count": state.get("replan_count", 0) + 1,
        "needs_replan": needs_replan,
        "next_action": "replan" if needs_replan else "finalize",
        "confidence": plan.confidence,
        "notes": notes,
        "planner_thoughts": thoughts,
    }


def finalize_step(state: PlannerState, auditor: PlannerAuditor) -> dict[str, object]:
    """Package the final result, apply risk gating, and close the audit trail."""
    plan = state.get("plan")
    risk_level = (
        plan.safety_policy.risk_level if plan is not None else state.get("risk_tier", "LOW")
    )
    needs_human_review = bool(state.get("needs_human_review")) or bool(
        plan is not None and plan.needs_human_review
    )

    report = state.get("report") or ""
    if not report:
        if state.get("ranked_matches"):
            report = summarize_matches(state["ranked_matches"])
        elif state.get("matches"):
            report = summarize_matches(state["matches"])
    if not report:
        report = (
            f"Investigation for goal: {state['goal']}. No actionable findings were produced."
        )

    masked = mask_report(report, risk_level)
    final_result = masked
    needs_authorization = False

    if risk_level == "CRITICAL":
        if state.get("reveal_authorized", False):
            auditor.review_decision("critical_content", "approved_reveal")
            auditor.authorization_granted("critical_content")
        else:
            needs_authorization = True
            auditor.review_request("critical_content", "CRITICAL", dual_required=True)
            auditor.critical_gated("critical_content")
            auditor.review_decision("critical_content", "pending_dual_authorization")
            final_result = (
                "REVIEW_REQUIRED: CRITICAL content is gated until dual authorization.\n\n"
                + masked
            )
    elif risk_level == "MED":
        auditor.review_request("sensitive_content", "MED", dual_required=False)
        auditor.review_decision("sensitive_content", "blurred_preview")
        final_result = "SENSITIVE: preview blurred/masked by default.\n\n" + masked

    auditor.goal_completed(
        {
            "goal": state["goal"],
            "risk_tier": risk_level,
            "completed_steps": state.get("completed_steps", []),
            "findings": bool(state.get("matches")),
            "confidence": state.get("confidence", 0.0),
        }
    )
    auditor.export(
        "report",
        "case_summary",
        {"risk_tier": risk_level, "needs_human_review": needs_human_review},
    )
    auditor.planner_finished(
        {
            "risk_tier": risk_level,
            "stop_early": bool(state.get("stop_early", False)),
            "replan_count": state.get("replan_count", 0),
            "completed_steps": state.get("completed_steps", []),
            "model_used": state.get("model_used", ""),
            "planner_iteration_count": state.get("planner_iteration_count", 0),
        }
    )
    auditor.chain_verified(auditor.verify())

    thoughts = list(state.get("planner_thoughts", [])) + [
        "finalize: stopped early" if state.get("stop_early") else "finalize: goal completed"
    ]
    return {
        "final_result": final_result,
        "needs_human_review": needs_human_review,
        "needs_authorization": needs_authorization,
        "risk_tier": risk_level,
        "notes": state.get("notes", []),
        "planner_thoughts": thoughts,
    }


__all__ = [
    "check_step",
    "dispatch_step",
    "finalize_step",
    "plan_step",
    "replan_step",
    "select_step",
]
