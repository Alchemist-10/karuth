"""Dynamic hypothesis-driven loop nodes.

These nodes make the planner genuinely agentic: after every tool execution the
loop *decides* its next action from the current investigation state (LLM
backend with a deterministic policy fallback), executes exactly one action
through the structured router, *observes* the outcome (updating hypothesis
confidence with deterministic evidence, recording contradictions), and decides
again — until it finalizes with an evidence-linked report.

The classic ``plan -> select -> dispatch -> check`` loop in ``nodes.py`` is
untouched; this module backs the optional ``mode="dynamic"`` graph only.
"""

from __future__ import annotations

from pathlib import Path

from karuth.planner.actions import (
    ActionDecision,
    DeterministicActionPlanner,
    EXECUTE_ACTIONS,
    decision_context,
    validate_action,
)
from karuth.planner.audit import PlannerAuditor
from karuth.planner.evidence_report import build_evidence_report
from karuth.planner.hypothesis import HypothesisStore, SUPPORT_THRESHOLD
from karuth.planner.investigation import InvestigationTree
from karuth.planner.llm import PlannerLLM
from karuth.planner.nodes import _backend_name, _log_recorded_fallback
from karuth.planner.observer import observe_result
from karuth.planner.router import mask_report, run_tool
from karuth.planner.schema import PlannerTask
from karuth.planner.state import PlannerState


def _safe_decide(
    planner: PlannerLLM, goal: str, ctx: dict, auditor: PlannerAuditor
) -> ActionDecision:
    """Ask the planner for the next action; always fall back deterministically."""
    decide = getattr(planner, "decide_action", None)
    if decide is None:
        auditor.fallback_used(
            _backend_name(planner), "planner backend has no decide_action"
        )
        return DeterministicActionPlanner().decide_action(goal, ctx)
    try:
        decision = decide(goal, ctx)
        _log_recorded_fallback(planner, auditor)
        return decision
    except Exception as exc:  # noqa: BLE001 - the loop must never dead-end
        auditor.fallback_used(_backend_name(planner), f"action decision failed: {exc}")
        return DeterministicActionPlanner().decide_action(goal, ctx)


def _route_action(decision: ActionDecision) -> str:
    if decision.action_type == "ask_human":
        return "human"
    if decision.action_type == "finalize":
        return "finalize"
    return "execute"


def dynamic_decide_step(
    state: PlannerState, planner: PlannerLLM, auditor: PlannerAuditor, *, max_actions: int
) -> dict[str, object]:
    """Pick exactly one next action, guard it, and record it in the tree."""
    ctx = decision_context(state)
    budget_exhausted = len(state.get("actions_taken", [])) >= max_actions
    if budget_exhausted:
        decision = ActionDecision(
            action_type="finalize",
            rationale="action budget exhausted",
            expected_information_gain=0.0,
            confidence=0.9,
            stop_reason="action_budget_exhausted",
        )
    else:
        decision = _safe_decide(planner, state["goal"], ctx, auditor)

    validate_action(decision, state)
    auditor.action_decided(decision.model_dump(mode="json"))

    iteration = state.get("planner_iteration_count", 0) + 1
    tree = state.get("investigation_tree") or InvestigationTree(goal=state["goal"])
    tree.record(
        "action",
        f"{decision.action_type}: {decision.tool or '—'}",
        detail=decision.rationale,
        iteration=iteration,
        parent=decision.target_hypothesis_id,
        confidence=decision.confidence,
    )
    # Only executable actions count against the budget; terminal decisions
    # (finalize / ask_human) are recorded in the tree but not the action log.
    actions = list(state.get("actions_taken", []))
    if decision.action_type in EXECUTE_ACTIONS:
        actions.append(decision)
    updates: dict[str, object] = {
        "current_action": decision,
        "actions_taken": actions,
        "investigation_tree": tree,
        "next_dynamic": _route_action(decision),
    }
    if decision.action_type in ("finalize", "ask_human") and decision.stop_reason:
        updates["stop_reason"] = decision.stop_reason
    if decision.action_type == "ask_human":
        updates["requires_human_review"] = True
    return updates


def _task_for(decision: ActionDecision, state: PlannerState) -> PlannerTask:
    """Map an action decision onto the deterministic router task shape."""
    step = state.get("planner_iteration_count", 0) + 1
    if decision.action_type == "generate_hypothesis":
        tool = "HypothesisAgent"
        proposal = decision.arguments.get("proposed_hypothesis")
        fields: dict = {"proposed_hypothesis": proposal} if isinstance(proposal, dict) else {}
        return PlannerTask(step=step, tool=tool, reason=decision.rationale, fields=fields)
    if decision.action_type == "challenge_hypothesis":
        return PlannerTask(
            step=step,
            tool="SkepticReview",
            reason=decision.rationale,
            inputs=[decision.target_hypothesis_id] if decision.target_hypothesis_id else [],
        )
    tool = decision.tool or "TimelineAgent"
    inputs = list(decision.arguments.get("inputs", [])) if decision.arguments.get("inputs") else []
    fields = {"target_hypothesis_id": decision.target_hypothesis_id} if decision.target_hypothesis_id else {}
    return PlannerTask(step=step, tool=tool, reason=decision.rationale, inputs=inputs, fields=fields)


def dynamic_execute_step(state: PlannerState, auditor: PlannerAuditor) -> dict[str, object]:
    """Execute the selected action through the structured router."""
    decision = state.get("current_action")
    notes = list(state.get("notes", []))
    if decision is None or decision.action_type not in EXECUTE_ACTIONS:
        notes.append("execute called without an executable action")
        return {"notes": notes, "next_dynamic": "finalize"}

    task = _task_for(decision, state)
    iteration = state.get("planner_iteration_count", 0) + 1
    auditor.task_selected(task.tool, decision.rationale)
    auditor.task_dispatched(task.tool, task.inputs)

    result = run_tool(task, state, iteration=iteration)
    if result.success:
        auditor.task_completed(task.tool, result.summary)
        if task.tool == "HypothesisAgent":
            for hypothesis in result.state_updates.get("hypotheses", []):
                auditor.hypothesis_created(
                    {
                        "id": hypothesis.id,
                        "statement": hypothesis.statement,
                        "status": hypothesis.status,
                        "confidence": hypothesis.confidence,
                        "evidence_refs": hypothesis.supporting_evidence[:20],
                        "created_by": hypothesis.created_by,
                    }
                )
        if task.tool == "ContradictionHunter":
            for contradiction in result.state_updates.get("contradictions", []):
                auditor.contradiction_found(contradiction)
        if task.tool == "MvdRegistryLookup":
            for registry in result.state_updates.get("registry_results", []):
                auditor.registry_lookup_result(registry)
    else:
        auditor.task_failed(task.tool, result.error)

    execution = list(state.get("execution_history", []))
    execution.append(
        {
            "tool": task.tool,
            "status": "completed" if result.success else "failed",
            "summary": result.summary,
            "duration_s": round(result.duration_ms / 1000.0, 4),
            "error": result.error,
            "action_type": decision.action_type,
            "target_hypothesis_id": decision.target_hypothesis_id,
        }
    )
    tool_history = list(state.get("tool_history", []))
    tool_history.append({"tool": task.tool, "status": "completed" if result.success else "failed", "at_iteration": iteration})

    updates: dict[str, object] = dict(result.state_updates)
    updates["current_result"] = result
    updates["tool_results"] = list(state.get("tool_results", [])) + [result]
    updates["planner_iteration_count"] = iteration
    updates["execution_history"] = execution
    updates["tool_history"] = tool_history
    updates["completed_steps"] = list(dict.fromkeys(state.get("completed_steps", []) + [task.tool]))
    updates["notes"] = notes
    updates["next_dynamic"] = "observe"

    # Multimodal safety flags are derived deterministically from observations.
    observations = list(result.state_updates.get("image_observations", state.get("image_observations", [])))
    if observations:
        from karuth.planner.observations import normalize_observations, sensitive_observations

        normalized = normalize_observations(observations)
        updates["sensitive_content_detected"] = bool(sensitive_observations(normalized))
        updates["image_observations"] = normalized

    # TriageAgent re-running after a skeptic clears the re-rank flag.
    if task.tool == "TriageAgent":
        updates["post_skeptic_rank"] = True
        updates["rerank_pending"] = False
    return updates


def _average_open_confidence(store: HypothesisStore) -> float:
    active = [h for h in store.all() if h.status in ("open", "inconclusive", "supported")]
    if not active:
        return 0.0
    return round(sum(h.confidence for h in active) / len(active), 3)


def dynamic_observe_step(state: PlannerState, auditor: PlannerAuditor) -> dict[str, object]:
    """Critic stage: update hypothesis confidence, record observations/tree."""
    result = state.get("current_result")
    iteration = state.get("planner_iteration_count", 0) + 1
    store = HypothesisStore(list(state.get("hypotheses", [])))

    observation = None
    if result is not None:
        observation = observe_result(
            store.as_state(),
            result,
            iteration=iteration,
            extra_contradictions=list(result.contradictions),
            # Once the skeptic has reviewed a lead, later observations may
            # record refs but must not silently re-raise its confidence.
            suppress_support=bool(state.get("skeptic_review")),
        )

    # The skeptic's revised confidence is authoritative for its target
    # hypothesis: a challenge that found contradictions lowers the lead.
    if result is not None and result.success and result.tool == "SkepticReview":
        review = result.state_updates.get("skeptic_review")
        if isinstance(review, dict):
            auditor.skeptic_reviewed(review)
            target_id = review.get("hypothesis_id", "")
            if target_id and any(h.id == target_id for h in store.all()):
                hypothesis = store.get(target_id)
                hypothesis.confidence = float(review.get("revised_confidence", hypothesis.confidence))
                if hypothesis.status == "supported" and hypothesis.confidence < SUPPORT_THRESHOLD:
                    try:
                        store.transition(target_id, "inconclusive")
                    except ValueError:
                        pass
                hypotheses = store.as_state()
            else:
                hypotheses = store.as_state()
        else:
            hypotheses = store.as_state()
    else:
        hypotheses = store.as_state()

    contradictions = list(state.get("contradictions", []))
    observations = list(state.get("observations", []))
    if observation is not None:
        auditor.observation_recorded(observation.to_dict())
        observations.append(observation)
        for update in observation.hypothesis_updates:
            if update.get("status") in ("supported", "rejected") or update.get("confidence", 0) <= 0.3:
                try:
                    store.transition(update["id"], update["status"])
                except (ValueError, KeyError):
                    pass
        hypotheses = store.as_state()

    confidence = _average_open_confidence(store)
    if result is not None and result.tool == "SkepticReview":
        review = result.state_updates.get("skeptic_review")
        if isinstance(review, dict):
            confidence = float(review.get("revised_confidence", confidence))

    # Deterministic information-gain proxy: how many new facts this action added.
    added = 0
    if result is not None:
        added += len(result.state_updates.get("entities", [])) + len(result.state_updates.get("matches", []))
        added += len(result.state_updates.get("hypotheses", [])) + len(result.state_updates.get("contradictions", []))
    gain = round(min(1.0, added / 20.0), 3)

    # A skeptic that lowered confidence invalidates the earlier lead ranking.
    rerank_pending = bool(state.get("rerank_pending"))
    if (
        result is not None
        and result.success
        and result.tool == "SkepticReview"
        and state.get("matches")
    ):
        review = result.state_updates.get("skeptic_review")
        if isinstance(review, dict) and float(review.get("revised_confidence", 0)) < float(
            review.get("original_confidence", 0)
        ):
            rerank_pending = True

    # Confidence breakdown snapshot for the UI (None fields = "Not scored").
    from karuth.planner.evidence_report import compute_confidence_breakdown

    breakdown = compute_confidence_breakdown(state)
    history = list(state.get("confidence_history", []))[-49:] + [breakdown]

    tree = state.get("investigation_tree") or InvestigationTree(goal=state["goal"])
    parent_id = tree.last("action").id if tree.last("action") else ""
    if observation is not None:
        tree.record(
            "observation",
            f"observed {result.tool if result else '—'}",
            detail=observation.explanation,
            refs=observation.supporting_evidence[:10],
            iteration=iteration,
            parent=parent_id,
            confidence=confidence,
        )
    # Record only contradictions discovered by this action (avoid duplicates).
    for contradiction in result.state_updates.get("contradictions", []) if result else []:
        tree.record(
            "contradiction",
            f"{contradiction.get('rule', '')}: {contradiction.get('claim_a', '')} vs {contradiction.get('claim_b', '')}",
            refs=list(contradiction.get("evidence_refs", []))[:10],
            iteration=iteration,
            parent=parent_id,
        )

    recommendation = observation.recommendation if observation else "continue"
    return {
        "hypotheses": hypotheses,
        "observations": observations,
        "contradictions": contradictions,
        "confidence": confidence,
        "information_gain": gain,
        "investigation_tree": tree,
        "rerank_pending": rerank_pending,
        "confidence_breakdown": breakdown,
        "confidence_history": history,
        "next_dynamic": "decide",
        "stop_reason": state.get("stop_reason", ""),
    }


def dynamic_human_step(state: PlannerState, auditor: PlannerAuditor) -> dict[str, object]:
    """Checkpoint: the loop requests investigator input before finalizing."""
    decision = state.get("current_action")
    note = decision.rationale if decision else "investigator input requested"
    auditor.human_input_requested(note)
    tree = state.get("investigation_tree") or InvestigationTree(goal=state["goal"])
    tree.record(
        "conclusion",
        "awaiting human input",
        detail=note,
        iteration=state.get("planner_iteration_count", 0) + 1,
    )
    return {
        "human_input_requested": True,
        "needs_human_review": True,
        "stop_reason": "awaiting_human_input",
        "investigation_tree": tree,
        "next_dynamic": "finalize",
    }


def _gate_result(report: str, risk_level: str, auditor: PlannerAuditor, state: PlannerState) -> tuple[str, bool]:
    """Apply the existing risk gating (MED blur / CRITICAL dual auth)."""
    masked = mask_report(report, risk_level)
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
            final = "REVIEW_REQUIRED: CRITICAL content is gated until dual authorization.\n\n" + masked
            return final, needs_authorization
    elif risk_level == "MED":
        auditor.review_request("sensitive_content", "MED", dual_required=False)
        auditor.review_decision("sensitive_content", "blurred_preview")
        return "SENSITIVE: preview blurred/masked by default.\n\n" + masked, needs_authorization
    return masked, needs_authorization


def _hard_finalization_rules(state: PlannerState) -> tuple[str, bool]:
    """Unsafe-finalization guards (spec item 8).

    Never emit ``sufficient_evidence`` while unresolved contradictions, an
    unconfirmed identity link, or an unreviewed activity interpretation remain,
    and never finalize sensitive content without reveal authorization.
    """
    from karuth.planner.observations import normalize_observations

    observations = normalize_observations(state.get("image_observations", []))
    contradictions = list(state.get("contradictions", []))
    sensitive_detected = any(
        observation.observation_type == "sensitive_content" for observation in observations
    )
    identity_pending = any(
        observation.observation_type == "person" and observation.requires_human_review
        for observation in observations
    )
    activity_pending = any(
        observation.observation_type == "activity_interpretation" for observation in observations
    )

    stop_reason = state.get("stop_reason") or ""
    requires_human_review = bool(state.get("needs_human_review"))

    # An explicit investigator checkpoint already stops safely; the hard rules
    # only guard against *unsafe* finalization (premature "sufficient_evidence").
    if stop_reason == "awaiting_human_input":
        return stop_reason, True

    if contradictions:
        stop_reason = "unresolved_contradictions"
        requires_human_review = True
    elif identity_pending and not state.get("identity_confirmed", False):
        stop_reason = "awaiting_human_confirmation"
        requires_human_review = True
    elif activity_pending and not state.get("activity_reviewed", False):
        stop_reason = "activity_interpretation_requires_review"
        requires_human_review = True
    elif sensitive_detected and not state.get("reveal_authorized", False):
        stop_reason = "sensitive_content_gated"
        requires_human_review = True
    elif (
        state.get("registry_results")
        and not state.get("entities")
        and not observations
    ):
        # An approved registry lookup returned a registered owner, but nothing
        # in the evidence connects anyone to the vehicle: the owner is a
        # candidate association that must not be relied on without review.
        stop_reason = "awaiting_human_confirmation"
        requires_human_review = True
    elif not observations and not state.get("entities") and _image_file_count(state) > 0:
        # Images were ingested but nothing machine-readable came out. The
        # investigator must decide next steps (sidecar, OCR, vision, or accept
        # the gap), so this is never silently "INVESTIGATION COMPLETE".
        stop_reason = "no_information_gain"
        requires_human_review = True
    return stop_reason, requires_human_review


def _image_file_count(state: PlannerState) -> int:
    """Number of supported image files inside the case folder."""
    from karuth.config import SUPPORTED_IMAGE_SUFFIXES

    case_folder = state.get("case_folder")
    if not case_folder:
        return 0
    return sum(
        1
        for suffix in SUPPORTED_IMAGE_SUFFIXES
        for _ in Path(case_folder).rglob(f"*{suffix}")
    )


def dynamic_finalize_step(state: PlannerState, auditor: PlannerAuditor) -> dict[str, object]:
    """Build the evidence-linked report, apply risk gating, close the audit trail."""
    from karuth.planner.evidence_report import compute_confidence_breakdown

    try:
        from karuth.vision.ocr import ocr_available

        _ocr = ocr_available()
    except Exception:  # noqa: BLE001 - OCR availability is a soft hint
        _ocr = False
    image_count = _image_file_count(state)

    plan = state.get("plan")
    risk_level = plan.safety_policy.risk_level if plan is not None else state.get("risk_tier", "LOW")
    initial_confidence = plan.confidence if plan is not None else float(state.get("confidence", 0.0) or 0.0)
    final_confidence = float(state.get("confidence", initial_confidence) or initial_confidence)

    stop_reason, requires_human_review = _hard_finalization_rules(state)

    hypotheses = list(state.get("hypotheses", []))
    breakdown = compute_confidence_breakdown(state)
    report = build_evidence_report(
        goal=state["goal"],
        matches=list(state.get("matches", [])),
        entities=list(state.get("entities", [])),
        hypotheses=hypotheses,
        contradictions=list(state.get("contradictions", [])),
        skeptic=state.get("skeptic_review"),
        final_confidence=final_confidence,
        initial_confidence=initial_confidence,
        stop_reason=stop_reason,
        image_observations=list(state.get("image_observations", [])),
        confidence_breakdown=breakdown,
        needs_human_review=requires_human_review,
        registry_results=list(state.get("registry_results", [])),
        image_count=image_count,
        ocr_available=_ocr,
    )

    final_result, needs_authorization = _gate_result(report, risk_level, auditor, state)

    tree = state.get("investigation_tree") or InvestigationTree(goal=state["goal"])
    tree.record(
        "conclusion",
        stop_reason,
        detail=f"final confidence {final_confidence:.2f}",
        iteration=state.get("planner_iteration_count", 0) + 1,
        parent=tree.last("action").id if tree.last("action") else "",
        confidence=final_confidence,
    )

    auditor.dynamic_finalized(
        {
            "goal": state["goal"],
            "risk_tier": risk_level,
            "stop_reason": stop_reason,
            "actions_taken": len(state.get("actions_taken", [])),
            "hypotheses": len(hypotheses),
            "contradictions": len(state.get("contradictions", [])),
            "confidence": final_confidence,
        }
    )
    auditor.goal_completed(
        {
            "goal": state["goal"],
            "risk_tier": risk_level,
            "completed_steps": state.get("completed_steps", []),
            "findings": bool(state.get("matches")),
            "confidence": final_confidence,
        }
    )
    auditor.export("report", "evidence_linked_case_summary", {"risk_tier": risk_level})
    auditor.planner_finished(
        {
            "risk_tier": risk_level,
            "mode": "dynamic",
            "stop_reason": stop_reason,
            "actions_taken": len(state.get("actions_taken", [])),
            "completed_steps": state.get("completed_steps", []),
            "model_used": state.get("model_used", ""),
            "planner_iteration_count": state.get("planner_iteration_count", 0),
        }
    )
    auditor.chain_verified(auditor.verify())

    return {
        "final_result": final_result,
        "needs_human_review": requires_human_review,
        "needs_authorization": needs_authorization,
        "risk_tier": risk_level,
        "stop_reason": stop_reason,
        "confidence_breakdown": breakdown,
        "investigation_tree": tree,
        "next_dynamic": "done",
    }


__all__ = [
    "dynamic_decide_step",
    "dynamic_execute_step",
    "dynamic_finalize_step",
    "dynamic_human_step",
    "dynamic_observe_step",
]
