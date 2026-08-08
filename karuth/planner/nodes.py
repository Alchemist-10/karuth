"""LangGraph node functions for the agentic planner.

Each node is a pure-ish function over :class:`PlannerState` with injected
dependencies (the planning backend and the ledger auditor). The graph wiring in
``graph.py`` decides which path runs next.
"""

from __future__ import annotations

from karuth.agents.report_agent import summarize_matches
from karuth.planner.audit import PlannerAuditor
from karuth.planner.guardrails import assert_plan_safe, reject_external_url
from karuth.planner.llm import PlannerLLM
from karuth.planner.postconditions import has_entities, has_matches
from karuth.planner.router import mask_report, route_task
from karuth.planner.state import PlannerState


def _summary(state: PlannerState) -> dict[str, object]:
    return {
        "case_id": state.get("case_id", ""),
        "entity_count": len(state.get("entities", [])),
        "match_count": len(state.get("matches", [])),
        "ranked_count": len(state.get("ranked_matches", [])),
        "timeline_count": len(state.get("timeline", [])),
        "completed_steps": state.get("completed_steps", []),
        "notes": state.get("notes", []),
    }


def plan_step(
    state: PlannerState, planner: PlannerLLM, auditor: PlannerAuditor
) -> dict[str, object]:
    """Interpret the goal, guard it, and generate the initial plan."""
    goal = state["goal"]
    reject_external_url(goal)
    auditor.goal_received(goal)

    plan = planner.generate_plan(goal, _summary(state))
    assert_plan_safe(plan, state["case_folder"])
    auditor.plan_generated(plan)

    tool_list = [task.tool for task in plan.tasks]
    return {
        "plan": plan,
        "pending_steps": list(plan.tasks),
        "completed_steps": [],
        "risk_tier": plan.safety_policy.risk_level,
        "needs_human_review": plan.needs_human_review,
        "replan_count": 0,
        "notes": state.get("notes", []) + [f"plan generated: {tool_list}"],
    }


def dispatch_step(state: PlannerState, auditor: PlannerAuditor) -> dict[str, object]:
    """Run every pending task through the deterministic router."""
    pending = state.get("pending_steps", [])
    completed = list(state.get("completed_steps", []))
    notes = list(state.get("notes", []))
    updates: dict[str, object] = {}

    for task in pending:
        auditor.task_selected(task.tool, task.reason)
        auditor.task_dispatched(task.tool, task.inputs)
        merged = {**state, **updates}
        try:
            result, summary = route_task(task, merged)
            for key, value in result.items():
                updates[key] = value
            completed.append(task.tool)
            auditor.task_completed(task.tool, summary)
        except Exception as exc:  # noqa: BLE001 - planner must degrade gracefully
            auditor.task_failed(task.tool, str(exc))
            notes.append(f"{task.tool} failed: {exc}")

    if "entities" in updates:
        auditor.evidence_ingested(len(list(updates["entities"])))  # type: ignore[arg-type]

    updates["completed_steps"] = completed
    updates["pending_steps"] = []
    updates["last_dispatch_had_work"] = bool(pending)
    updates["notes"] = notes
    return updates


def check_step(state: PlannerState, auditor: PlannerAuditor) -> dict[str, object]:
    """Check postconditions and decide whether to stop or replan."""
    entities = list(state.get("entities", []))
    matches = list(state.get("matches", []))
    completed = state.get("completed_steps", [])
    notes = list(state.get("notes", []))

    entities_ok = has_entities(entities)
    matches_ok = has_matches(matches)
    auditor.postcondition("has_entities", entities_ok)
    auditor.postcondition("has_matches", matches_ok)

    stop_early = False
    needs_replan = False
    if "ExtractionAgent" in completed and not entities_ok:
        stop_early = True
        notes.append("stop_early: no evidence entities extracted")
    elif (
        "CorrelationAgent" in completed
        and not matches_ok
        and state.get("last_dispatch_had_work", False)
    ):
        needs_replan = True
        notes.append("replan: correlation returned no matches")

    return {"stop_early": stop_early, "needs_replan": needs_replan, "notes": notes}


def replan_step(
    state: PlannerState, planner: PlannerLLM, auditor: PlannerAuditor
) -> dict[str, object]:
    """Revise the plan when the first pass is insufficient."""
    reason = "first pass insufficient"
    auditor.replan_triggered(reason)

    plan = planner.replan(state["goal"], _summary(state), state.get("plan"))
    assert_plan_safe(plan, state["case_folder"])
    auditor.plan_generated(plan, replan=True)

    completed = state.get("completed_steps", [])
    pending = [task for task in plan.tasks if task.tool not in completed]
    notes = list(state.get("notes", []))
    tool_list = [task.tool for task in plan.tasks]

    needs_replan = True
    if not pending:
        needs_replan = False
        notes.append("replan produced no new steps; finalizing")
    notes.append(f"replanned to: {tool_list}")

    return {
        "plan": plan,
        "pending_steps": pending,
        "replan_count": state.get("replan_count", 0) + 1,
        "needs_replan": needs_replan,
        "notes": notes,
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
        }
    )
    auditor.chain_verified(auditor.verify())

    return {
        "final_result": final_result,
        "needs_human_review": needs_human_review,
        "needs_authorization": needs_authorization,
        "risk_tier": risk_level,
        "notes": state.get("notes", []),
    }


__all__ = [
    "check_step",
    "dispatch_step",
    "finalize_step",
    "plan_step",
    "replan_step",
]
