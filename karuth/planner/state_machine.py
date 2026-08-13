"""Agentic planner entry point built on LangGraph.

``run_planner_goal`` keeps the original signature so callers can switch from
the old fixed workflow to the agentic planner without changing their code. The
planning backend defaults to the LLM (OpenRouter) wrapped in a deterministic
fallback: the default experience is LLM-backed planning, and the offline
deterministic planner is used automatically whenever the LLM backend fails.
Pass ``planner=DeterministicPlanner()`` (or ``build_planner("deterministic")``)
to force the offline backend, or ``build_planner("lmstudio")`` for the fully
offline LLM backend. Unexpected graph failures degrade into an explicit final
state with a ledger trail instead of crashing; guardrail/policy rejections
(ValueError) still propagate so callers see the rejection.
"""

from __future__ import annotations

from pathlib import Path

from karuth.config import DYNAMIC_MAX_ACTIONS, PLANNER_MAX_ITERATIONS
from karuth.ledger import LedgerStore
from karuth.planner.audit import PlannerAuditor
from karuth.planner.graph import build_dynamic_planner_graph, build_planner_graph
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
    registry_lookup_approved: bool = False,
    max_iterations: int = PLANNER_MAX_ITERATIONS,
    mode: str = "plan",
    max_actions: int | None = None,
) -> PlannerState:
    """Run the agentic planner over a goal and return the final planner state.

    ``mode="plan"`` (default) preserves the classic fixed-task loop;
    ``mode="dynamic"`` runs the hypothesis-driven loop (decide -> execute ->
    observe) with deterministic fallback for action selection.
    """
    if mode not in ("plan", "dynamic"):
        raise ValueError(f"unknown planner mode: {mode}")
    planner = planner if planner is not None else build_planner()
    auditor = PlannerAuditor(ledger, case_id)
    if registry_lookup_approved:
        auditor.registry_lookup_approved("mvd_vehicle_owner")
    effective_max_actions = max_actions if max_actions is not None else DYNAMIC_MAX_ACTIONS
    if mode == "dynamic":
        graph = build_dynamic_planner_graph(
            planner,
            auditor,
            max_actions=effective_max_actions,
        )
    else:
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
        registry_lookup_approved=registry_lookup_approved,
    )
    state["mode"] = mode
    # Each action costs ~3 super-steps (decide -> execute -> observe); size the
    # graph recursion budget off the action budget so the loop can never be
    # killed by LangGraph's default recursion limit mid-investigation.
    recursion_limit = max(64, effective_max_actions * 4 + 16)
    try:
        return graph.invoke(state, config={"recursion_limit": recursion_limit})
    except ValueError:
        raise  # guardrail / policy rejections stay loud for the caller
    except Exception as exc:  # noqa: BLE001 - degrade instead of crashing
        auditor.fallback_used(_backend_name(planner), f"planner graph failed: {exc}")
        degraded = initial_state(
            case_id,
            case_folder,
            goal,
            reveal_authorized=reveal_authorized,
            registry_lookup_approved=registry_lookup_approved,
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
