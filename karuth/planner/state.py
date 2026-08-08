"""Extended planner state shared by LangGraph nodes.

The state tracks the dynamic execution plan, evidence, risk classification,
reveal authorization, and planner notes so downstream code can inspect exactly
what happened and why.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from karuth.correlation.candidate_matcher import CorrelationResult
from karuth.entities.schema import Entity
from karuth.planner.schema import InvestigationPlan, PlannerTask


class PlannerState(TypedDict, total=False):
    case_id: str
    case_folder: Path
    goal: str
    plan: InvestigationPlan
    entities: list[Entity]
    matches: list[CorrelationResult]
    ranked_matches: list[CorrelationResult]
    timeline: list[Entity]
    report: str
    legal_output: str
    notes: list[str]
    risk_tier: str
    needs_human_review: bool
    reveal_authorized: bool
    needs_authorization: bool
    completed_steps: list[str]
    pending_steps: list[PlannerTask]
    needs_replan: bool
    stop_early: bool
    last_dispatch_had_work: bool
    replan_count: int
    final_result: str
    plan_hash: str


def initial_state(
    case_id: str,
    case_folder: Path,
    goal: str,
    *,
    reveal_authorized: bool = False,
) -> PlannerState:
    """Build the starting planner state for a graph run."""
    return {
        "case_id": case_id,
        "case_folder": Path(case_folder).resolve(),
        "goal": goal,
        "notes": [],
        "risk_tier": "LOW",
        "needs_human_review": False,
        "reveal_authorized": reveal_authorized,
        "needs_authorization": False,
        "completed_steps": [],
        "pending_steps": [],
        "needs_replan": False,
        "stop_early": False,
        "last_dispatch_had_work": False,
        "replan_count": 0,
    }
