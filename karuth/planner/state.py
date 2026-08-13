"""Extended planner state shared by LangGraph nodes.

The state tracks the dynamic execution plan, evidence, risk classification,
reveal authorization, and planner notes so downstream code can inspect exactly
what happened and why. Part 7 fields — available evidence, tool/execution
history, planner thoughts, confidence, remaining objectives, iteration count,
token usage, model, and timing — make the reasoning loop inspectable and
auditable.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from karuth.correlation.candidate_matcher import CorrelationResult
from karuth.entities.schema import Entity
from karuth.planner.actions import ActionDecision
from karuth.planner.contradictions import Contradiction
from karuth.planner.evidence import EvidenceItem
from karuth.planner.hypothesis import Hypothesis
from karuth.planner.investigation import InvestigationTree
from karuth.planner.observer import Observation
from karuth.planner.schema import InvestigationPlan, PlannerTask
from karuth.planner.tool_result import ToolResult


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

    # ---- dynamic-execution loop (Part 4) ---------------------------------
    current_task: PlannerTask
    next_action: str
    replan_reason: str

    # ---- richer state (Part 7) -------------------------------------------
    available_evidence: list[EvidenceItem]
    tool_history: list[dict]
    execution_history: list[dict]
    planner_thoughts: list[str]
    confidence: float
    remaining_objectives: list[str]
    planner_iteration_count: int
    token_usage: dict
    model_used: str
    timing: dict

    # ---- dynamic hypothesis loop (agentic mode) ---------------------------
    mode: str
    hypotheses: list[Hypothesis]
    contradictions: list[Contradiction]
    observations: list[Observation]
    skeptic_review: dict
    current_action: ActionDecision
    current_result: ToolResult
    actions_taken: list[ActionDecision]
    tool_results: list[ToolResult]
    investigation_tree: InvestigationTree
    human_input_requested: bool
    stop_reason: str
    information_gain: float

    # ---- multimodal + safety (Jesse demo) ---------------------------------
    image_observations: list  # ImageObservation models
    sensitive_content_detected: bool
    identity_confirmed: bool  # investigator confirmation (UI checkbox)
    activity_reviewed: bool  # investigator reviewed the LLM interpretation
    rerank_pending: bool  # skeptic changed confidence -> re-rank leads
    post_skeptic_rank: bool  # TriageAgent already re-ranked after the skeptic
    confidence_breakdown: dict  # separate confidence values (None = Not scored)
    confidence_history: list[dict]  # breakdown snapshots per observation step
    # ---- external registry lookups (approval-gated, mock) ------------------
    registry_lookup_approved: bool  # investigator pre-approved the mock MVD lookup
    registry_results: list  # ExternalLookupResult dicts (synthetic fixtures)


def initial_state(
    case_id: str,
    case_folder: Path,
    goal: str,
    *,
    reveal_authorized: bool = False,
    registry_lookup_approved: bool = False,
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
        "next_action": "plan",
        "replan_reason": "",
        "available_evidence": [],
        "tool_history": [],
        "execution_history": [],
        "planner_thoughts": [],
        "confidence": 0.0,
        "remaining_objectives": [],
        "planner_iteration_count": 0,
        "token_usage": {},
        "model_used": "none",
        "timing": {},
        "mode": "plan",
        "hypotheses": [],
        "contradictions": [],
        "observations": [],
        "skeptic_review": None,
        "actions_taken": [],
        "tool_results": [],
        "investigation_tree": InvestigationTree(goal=goal),
        "human_input_requested": False,
        "stop_reason": "",
        "information_gain": 0.0,
        "image_observations": [],
        "sensitive_content_detected": False,
        "identity_confirmed": False,
        "activity_reviewed": False,
        "rerank_pending": False,
        "post_skeptic_rank": False,
        "confidence_breakdown": {},
        "confidence_history": [],
        "registry_lookup_approved": registry_lookup_approved,
        "registry_results": [],
    }
