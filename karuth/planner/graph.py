"""LangGraph orchestration for the agentic planner.

Builds a stateful reasoning loop that plans, selects the next schedulable task,
executes exactly one task per dispatch, checks postconditions, optionally
replans with an explicit reason, and finalizes. The loop is capped by
``max_iterations`` so it can never spin forever:

    plan -> select -> dispatch -> check -> (select | replan -> select | finalize)
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from karuth.planner.audit import PlannerAuditor
from karuth.planner.llm import PlannerLLM
from karuth.planner.nodes import (
    check_step,
    dispatch_step,
    finalize_step,
    plan_step,
    replan_step,
    select_step,
)
from karuth.planner.state import PlannerState


def decide_after_select(state: PlannerState, max_replans: int, max_iterations: int) -> str:
    """Route from ``select``: dispatch the chosen task, or replan/finalize."""
    if state.get("planner_iteration_count", 0) >= max_iterations:
        return "finalize"
    action = state.get("next_action", "finalize")
    if action == "dispatch":
        return "dispatch"
    if action == "replan" and state.get("replan_count", 0) < max_replans:
        return "replan"
    return "finalize"


def decide_next(state: PlannerState, max_replans: int, max_iterations: int) -> str:
    """Conditional routing after the postcondition check."""
    if state.get("planner_iteration_count", 0) >= max_iterations:
        return "finalize"
    action = state.get("next_action", "finalize")
    if action in ("continue", "dispatch"):
        return "select"
    if action == "replan" and state.get("replan_count", 0) < max_replans:
        return "replan"
    return "finalize"


def build_planner_graph(
    planner: PlannerLLM,
    auditor: PlannerAuditor,
    *,
    max_replans: int = 2,
    max_iterations: int = 30,
):
    """Compile the agentic planner graph with injected dependencies."""
    graph = StateGraph(PlannerState)

    graph.add_node("plan", lambda state: plan_step(state, planner, auditor))
    graph.add_node("select", lambda state: select_step(state, auditor))
    graph.add_node("dispatch", lambda state: dispatch_step(state, auditor))
    graph.add_node("check", lambda state: check_step(state, auditor))
    graph.add_node("replan", lambda state: replan_step(state, planner, auditor))
    graph.add_node("finalize", lambda state: finalize_step(state, auditor))

    graph.set_entry_point("plan")
    graph.add_edge("plan", "select")
    graph.add_conditional_edges(
        "select",
        lambda state: decide_after_select(state, max_replans, max_iterations),
        {"dispatch": "dispatch", "replan": "replan", "finalize": "finalize"},
    )
    graph.add_edge("dispatch", "check")
    graph.add_conditional_edges(
        "check",
        lambda state: decide_next(state, max_replans, max_iterations),
        {"select": "select", "replan": "replan", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        "replan",
        lambda state: "select" if state.get("pending_steps") else "finalize",
        {"select": "select", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)

    return graph.compile()
