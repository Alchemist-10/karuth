"""Agentic planner entry point built on LangGraph.

``run_planner_goal`` keeps the original signature so callers can switch from
the old fixed workflow to the agentic planner without changing their code. The
planning backend defaults to the offline deterministic planner and is swappable.
"""

from __future__ import annotations

from pathlib import Path

from karuth.ledger import LedgerStore
from karuth.planner.audit import PlannerAuditor
from karuth.planner.graph import build_planner_graph
from karuth.planner.llm import DeterministicPlanner, PlannerLLM
from karuth.planner.state import PlannerState, initial_state


def run_planner_goal(
    case_id: str,
    case_folder: Path,
    goal: str,
    ledger: LedgerStore,
    *,
    planner: PlannerLLM | None = None,
    max_replans: int = 2,
    reveal_authorized: bool = False,
) -> PlannerState:
    """Run the agentic planner over a goal and return the final planner state."""
    planner = planner or DeterministicPlanner()
    auditor = PlannerAuditor(ledger, case_id)
    graph = build_planner_graph(planner, auditor, max_replans=max_replans)
    state = initial_state(
        case_id,
        case_folder,
        goal,
        reveal_authorized=reveal_authorized,
    )
    return graph.invoke(state)
