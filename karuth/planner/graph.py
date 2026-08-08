"""LangGraph orchestration for the agentic planner.

Builds a stateful workflow that plans, dispatches the selected tools, checks
postconditions, optionally replans, and finalizes. Different goals naturally
produce different paths because the plan decides which tools run.
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
)
from karuth.planner.state import PlannerState


def decide_next(state: PlannerState, max_replans: int) -> str:
    """Conditional routing after the postcondition check."""
    if state.get("stop_early"):
        return "finalize"
    if state.get("needs_replan") and state.get("replan_count", 0) < max_replans:
        return "replan"
    return "finalize"


def build_planner_graph(
    planner: PlannerLLM,
    auditor: PlannerAuditor,
    *,
    max_replans: int = 2,
):
    """Compile the agentic planner graph with injected dependencies."""
    graph = StateGraph(PlannerState)

    graph.add_node("plan", lambda state: plan_step(state, planner, auditor))
    graph.add_node("dispatch", lambda state: dispatch_step(state, auditor))
    graph.add_node("check", lambda state: check_step(state, auditor))
    graph.add_node("replan", lambda state: replan_step(state, planner, auditor))
    graph.add_node("finalize", lambda state: finalize_step(state, auditor))

    graph.set_entry_point("plan")
    graph.add_edge("plan", "dispatch")
    graph.add_edge("dispatch", "check")
    graph.add_conditional_edges(
        "check",
        lambda state: decide_next(state, max_replans),
        {"finalize": "finalize", "replan": "replan"},
    )
    graph.add_edge("replan", "dispatch")
    graph.add_edge("finalize", END)

    return graph.compile()
