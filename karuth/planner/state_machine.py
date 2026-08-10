"""Agentic planner entry point built on LangGraph.

``run_planner_goal`` keeps the original signature so callers can switch from
the old fixed workflow to the agentic planner without changing their code. The
planning backend defaults to the LLM (LM Studio) wrapped in a deterministic
fallback: the default experience is LLM-backed planning, and the offline
deterministic planner is used automatically whenever the LLM backend fails.
Pass ``planner=DeterministicPlanner()`` (or ``build_planner("deterministic")``)
to force the offline backend. Unexpected graph failures degrade into an
explicit final state with a ledger trail instead of crashing; guardrail/policy
rejections (ValueError) still propagate so callers see the rejection.
"""

from __future__ import annotations

from pathlib import Path

from karuth.config import PLANNER_MAX_ITERATIONS
from karuth.ledger import LedgerStore
from karuth.planner.audit import PlannerAuditor
from karuth.planner.graph import build_planner_graph
from karuth.planner.llm import PlannerLLM, build_planner
from karuth.planner.nodes import _backend_name
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
    max_iterations: int = PLANNER_MAX_ITERATIONS,
) -> PlannerState:
    """Run the agentic planner over a goal and return the final planner state."""
    planner = planner if planner is not None else build_planner()
    auditor = PlannerAuditor(ledger, case_id)
    graph = build_planner_graph(
        planner,
        auditor,
        max_replans=max_replans,
        max_iterations=max_iterations,
    )
    state = initial_state(
        case_id,
        case_folder,
        goal,
        reveal_authorized=reveal_authorized,
    )
    try:
        return graph.invoke(state)
    except ValueError:
        raise  # guardrail / policy rejections stay loud for the caller
    except Exception as exc:  # noqa: BLE001 - degrade instead of crashing
        auditor.fallback_used(_backend_name(planner), f"planner graph failed: {exc}")
        degraded = initial_state(
            case_id,
            case_folder,
            goal,
            reveal_authorized=reveal_authorized,
        )
        degraded["notes"] = [f"planner graph failed: {exc}"]
        degraded["final_result"] = (
            f"Investigation for goal: {goal}. "
            f"The planner terminated with an internal error: {exc}"
        )
        auditor.planner_finished(
            {"risk_tier": "LOW", "error": str(exc), "completed_steps": []}
        )
        return degraded
