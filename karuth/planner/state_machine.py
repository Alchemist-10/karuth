"""Minimal planner state machine.

LangGraph can replace this sequential runner in a deployment that installs it;
the node boundaries are kept explicit so that swap is mechanical.
"""

from __future__ import annotations

from pathlib import Path

from karuth.ledger import LedgerStore
from karuth.planner.nodes import (
    PlannerState,
    decompose_goal,
    dispatch_correlate,
    dispatch_extract,
    dispatch_triage,
)
from karuth.planner.postconditions import has_entities, has_matches


def run_planner_goal(case_id: str, case_folder: Path, goal: str, ledger: LedgerStore) -> PlannerState:
    state: PlannerState = {"case_folder": case_folder, "goal": goal}
    state = decompose_goal(state, ledger, case_id)
    state = dispatch_extract(state, ledger, case_id)
    if not has_entities(state.get("entities", [])):
        return state
    state = dispatch_correlate(state, ledger, case_id)
    if not has_matches(state.get("matches", [])):
        return state
    return dispatch_triage(state, ledger, case_id)
