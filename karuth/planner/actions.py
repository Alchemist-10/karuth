"""Dynamic action selection.

The dynamic loop decides *one* next action after every tool execution:

- ``run_tool`` — execute a deterministic tool
- ``investigate_hypothesis`` — run a tool against a specific hypothesis
- ``generate_hypothesis`` — propose new hypotheses from current evidence
- ``challenge_hypothesis`` — run the skeptic against a hypothesis
- ``ask_human`` — request investigator input (checkpoint)
- ``finalize`` — stop and produce the evidence-linked report

The LLM backend proposes the decision (JSON, strict schema); the deterministic
policy below is the always-available offline fallback and the source of truth
for tests. Every decision passes through :func:`validate_action` (tool
registry, argument scope, hypothesis existence, budget) before execution.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from karuth.config import DYNAMIC_MAX_ACTIONS
from karuth.planner.guardrails import ensure_inside_case, reject_external_url
from karuth.planner.tools import TOOL_REGISTRY, tool_spec

# Compact (KL01CC2121 / KL 01 CC 2121) and hyphenated (NM-JP-4242) plates.
_PLATE_IN_TEXT_RE = re.compile(
    r"\b[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,3}\s?\d{3,4}\b"
    r"|\b[A-Z]{2}[- ][A-Z]{2}[- ]?\d{3,4}\b",
    re.IGNORECASE,
)


def _plate_from_goal(goal: str) -> str:
    match = _PLATE_IN_TEXT_RE.search(goal or "")
    if not match:
        return ""
    return re.sub(r"\s+", "", match.group(0)).upper()

ActionType = Literal[
    "run_tool",
    "investigate_hypothesis",
    "generate_hypothesis",
    "challenge_hypothesis",
    "ask_human",
    "finalize",
]

VALID_ACTION_TYPES: set[str] = set(ActionType.__args__)  # type: ignore[attr-defined]

TOOL_BY_ACTION: dict[str, str] = {
    "generate_hypothesis": "HypothesisAgent",
    "challenge_hypothesis": "SkepticReview",
}

EXECUTE_ACTIONS: set[str] = {"run_tool", "investigate_hypothesis", "generate_hypothesis", "challenge_hypothesis"}


class ActionDecision(BaseModel):
    """One decision from the dynamic loop."""

    action_type: ActionType
    tool: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)
    target_hypothesis_id: str = ""
    rationale: str = Field(default="", description="audit-safe one-line rationale")
    expected_information_gain: float = Field(ge=0.0, le=1.0, default=0.0)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    stop_reason: str = ""
    requires_human_review: bool = False


# ---------------------------------------------------------------------------
# decision context (audit-safe, shared by LLM prompt and deterministic policy)
# ---------------------------------------------------------------------------


def decision_context(state: dict[str, Any]) -> dict[str, Any]:
    """Extract the audit-safe summary the action selector reasons over."""
    hypotheses = [
        {
            "id": h.id,
            "statement": h.statement,
            "status": h.status,
            "confidence": h.confidence,
            "related_entity_ids": h.related_entity_ids[:8],
        }
        for h in state.get("hypotheses", [])
    ]
    goal = state.get("goal", "")
    observations = list(state.get("image_observations", []))
    plate_observation_count = 0
    for observation in observations:
        data = observation if isinstance(observation, dict) else observation.model_dump(mode="json")
        if data.get("observation_type") == "number_plate" and data.get("value"):
            plate_observation_count += 1
    return {
        "goal": goal,
        "risk_tier": state.get("risk_tier", "LOW"),
        "entity_count": len(state.get("entities", [])),
        "match_count": len(state.get("matches", [])),
        "timeline_count": len(state.get("timeline", [])),
        "observation_count": len(observations),
        "image_count": _evidence_file_count(state, "images"),
        "goal_plate": _plate_from_goal(goal),
        "plate_observation_count": plate_observation_count,
        "registry_lookup_approved": bool(state.get("registry_lookup_approved", False)),
        "registry_results": len(state.get("registry_results", [])),
        "hypotheses": hypotheses,
        "contradictions": [
            {
                "id": c.get("id", ""),
                "rule": c.get("rule", ""),
                "severity": c.get("severity", "medium"),
                "claim_a": c.get("claim_a", ""),
                "claim_b": c.get("claim_b", ""),
                "evidence_refs": c.get("evidence_refs", [])[:8],
            }
            for c in state.get("contradictions", [])
        ],
        "completed_tools": list(state.get("completed_steps", [])),
        "action_count": len(state.get("actions_taken", [])),
        "max_actions": DYNAMIC_MAX_ACTIONS,
        "last_tool": state.get("current_result", {}).tool if state.get("current_result") else "",
        "last_summary": state.get("current_result", {}).summary if state.get("current_result") else "",
        "skeptic_reviewed": bool(state.get("skeptic_review")),
        "rerank_pending": bool(state.get("rerank_pending")),
        "confidence": state.get("confidence", 0.0),
        "information_gain": state.get("information_gain", 0.0),
    }


def _evidence_file_count(state: dict[str, Any], kind: str) -> int:
    """File count for an evidence kind from the planner's evidence inventory."""
    for item in state.get("available_evidence", []):
        data = item if isinstance(item, dict) else getattr(item, "model_dump", lambda: {})()
        if data.get("kind") == kind:
            return int(data.get("file_count", 0) or 0)
    return 0


# ---------------------------------------------------------------------------
# guardrails
# ---------------------------------------------------------------------------


def validate_action(decision: ActionDecision, state: dict[str, Any]) -> None:
    """Reject decisions that could violate scope, tools, or state invariants."""
    if decision.action_type not in VALID_ACTION_TYPES:
        raise ValueError(f"invalid action type: {decision.action_type}")

    if decision.action_type in EXECUTE_ACTIONS:
        tool = decision.tool or TOOL_BY_ACTION.get(decision.action_type, "")
        if not tool:
            raise ValueError(f"action {decision.action_type} requires a tool")
        if tool not in TOOL_REGISTRY:
            raise ValueError(f"unknown planner tool in action: {tool}")
        tool_spec(tool)
        # Dynamic mode never runs ReportAgent: the only report is the
        # evidence-linked one built by the finalize step, after contradiction
        # hunting and the skeptic review have run.
        if tool == "ReportAgent":
            raise ValueError(
                "ReportAgent cannot be selected in dynamic mode; the final "
                "report is generated only by the finalize step after "
                "contradiction and skeptic review."
            )

    if decision.action_type == "challenge_hypothesis" and not decision.target_hypothesis_id:
        raise ValueError("challenge_hypothesis requires a target_hypothesis_id")

    if decision.action_type == "ask_human":
        reject_external_url(decision.rationale)

    if decision.target_hypothesis_id:
        ids = {h.id for h in state.get("hypotheses", [])}
        if decision.target_hypothesis_id not in ids:
            raise ValueError(
                f"action targets unknown hypothesis: {decision.target_hypothesis_id}"
            )

    string_args: list[str] = []
    for key, value in (decision.arguments or {}).items():
        if isinstance(value, str):
            string_args.append(value)
        elif isinstance(value, list):
            string_args.extend(item for item in value if isinstance(item, str))
    for value in string_args:
        reject_external_url(value)
        if value.startswith(("http://", "https://", "ftp://", "file://")):
            raise ValueError(f"external URL in action argument: {value}")
        if value.startswith((".", "~")) or os.sep in value or Path(value).is_absolute():
            case_folder = Path(state.get("case_folder", "."))
            ensure_inside_case(case_folder, Path(value))


def action_under_budget(state: dict[str, Any]) -> bool:
    return len(state.get("actions_taken", [])) < DYNAMIC_MAX_ACTIONS


# ---------------------------------------------------------------------------
# deterministic next-action policy (offline fallback + testable source of truth)
# ---------------------------------------------------------------------------


class DeterministicActionPlanner:
    """Rule-based action selection; always available, fully offline."""

    backend_name = "deterministic_actions"

    def decide_action(self, goal: str, ctx: dict[str, Any]) -> ActionDecision:
        return _deterministic_decision(ctx)


def _run_tool(tool: str, rationale: str, gain: float = 0.5, **arguments: Any) -> ActionDecision:
    return ActionDecision(
        action_type="run_tool",
        tool=tool,
        arguments=arguments,
        rationale=rationale,
        expected_information_gain=gain,
        confidence=0.7,
    )


def _deterministic_decision(ctx: dict[str, Any]) -> ActionDecision:
    """Pick the next action from the decision context.

    Priority: establish facts (extract -> correlate -> timeline), then form
    hypotheses, then hunt contradictions, then challenge (skeptic), then stop
    with an explicit reason. The loop can never exceed ``max_actions``.
    """
    completed = set(ctx.get("completed_tools", []))
    hypotheses = ctx.get("hypotheses", [])
    contradictions = ctx.get("contradictions", [])
    action_count = int(ctx.get("action_count", 0))
    max_actions = int(ctx.get("max_actions", DYNAMIC_MAX_ACTIONS))

    if action_count >= max_actions:
        return ActionDecision(
            action_type="finalize",
            rationale="action budget exhausted",
            expected_information_gain=0.0,
            confidence=0.9,
            stop_reason="action_budget_exhausted",
        )

    if ctx.get("entity_count", 0) == 0 and "ExtractionAgent" not in completed:
        return _run_tool("ExtractionAgent", "no entities extracted yet; discover evidence first", 0.9)

    # Investigator-safe order: inspect what the images contribute before
    # correlating, so links are drawn over understood multimodal evidence.
    if (
        ctx.get("observation_count", 0) == 0
        and ctx.get("image_count", 0) > 0
        and "ObservationAgent" not in completed
    ):
        return _run_tool(
            "ObservationAgent",
            "images present; inspect the multimodal observations before correlating",
            0.8,
        )

    if ctx.get("match_count", 0) == 0 and "CorrelationAgent" not in completed:
        return _run_tool("CorrelationAgent", "entities exist but no links yet; correlate them", 0.9)

    if ctx.get("timeline_count", 0) == 0 and ctx.get("entity_count", 0) > 0 and "TimelineAgent" not in completed:
        return _run_tool("TimelineAgent", "order the evidence chronologically to assess the connection", 0.6)

    if (
        not hypotheses
        and "HypothesisAgent" not in completed
        and (ctx.get("match_count", 0) > 0 or ctx.get("entity_count", 0) > 0)
    ):
        return ActionDecision(
            action_type="generate_hypothesis",
            rationale="matches exist; propose competing hypotheses about who is connected to the key evidence",
            expected_information_gain=0.8,
            confidence=0.7,
        )

    if (
        not contradictions
        and "ContradictionHunter" not in completed
        and "CorrelationAgent" in completed
    ):
        return _run_tool(
            "ContradictionHunter",
            "probe the new matches for conflicting timestamps, identities, or movement",
            0.7,
        )

    open_hypotheses = [h for h in hypotheses if h.get("status") in ("open", "inconclusive")]
    if open_hypotheses and contradictions and not ctx.get("skeptic_reviewed", False):
        # Challenge the *lead*: prefer the hypothesis tied to the vehicle/key
        # evidence of the goal, then the highest-confidence open hypothesis.
        lead_candidates = [
            h
            for h in open_hypotheses
            if "vehicle" in str(h.get("statement", "")).lower()
        ] or open_hypotheses
        target = max(lead_candidates, key=lambda h: float(h.get("confidence", 0)))
        return ActionDecision(
            action_type="challenge_hypothesis",
            tool="SkepticReview",
            target_hypothesis_id=target["id"],
            rationale="contradictions exist; the skeptic must try to disprove the lead before finalizing",
            expected_information_gain=0.6,
            confidence=0.7,
        )

    supported = [h for h in hypotheses if h.get("status") == "supported"]
    if supported and not ctx.get("skeptic_reviewed", False):
        return ActionDecision(
            action_type="challenge_hypothesis",
            tool="SkepticReview",
            target_hypothesis_id=supported[0]["id"],
            rationale="a hypothesis is marked supported; adversarially review it before finalizing",
            expected_information_gain=0.5,
            confidence=0.8,
        )

    # A skeptic that changed confidence invalidates the earlier lead ranking:
    # re-rank the strongest leads before finalizing.
    if ctx.get("rerank_pending", False) and ctx.get("match_count", 0) > 0:
        return _run_tool(
            "TriageAgent",
            "skeptic review changed confidence or evidence state; re-rank the strongest leads",
            0.5,
        )

    # Approval-gated external lookup (general): triggers on any vehicle plate
    # that became central — a vehicle hypothesis, a plate observation from the
    # images (EXIF/OCR), or a plate the investigator named in the goal. Only
    # runs when the investigator pre-approved the mock registry, so existing
    # flows without approval are unchanged. Registered ownership is surfaced
    # as a metadata fact, never as proof of who operated the vehicle.
    if (
        ctx.get("registry_lookup_approved", False)
        and not ctx.get("registry_results")
        and "MvdRegistryLookup" not in completed
        and (
            _has_vehicle_hypothesis(ctx)
            or bool(ctx.get("goal_plate"))
            or ctx.get("plate_observation_count", 0) > 0
        )
    ):
        return _run_tool(
            "MvdRegistryLookup",
            "registry approval granted: retrieve the registered owner for the vehicle plate",
            0.5,
        )

    if contradictions and ctx.get("risk_tier", "LOW") == "CRITICAL":
        return ActionDecision(
            action_type="ask_human",
            rationale="critical-tier findings with unresolved contradictions require investigator adjudication",
            expected_information_gain=0.0,
            confidence=0.9,
            requires_human_review=True,
        )

    if contradictions:
        return ActionDecision(
            action_type="finalize",
            rationale="unresolved contradictions remain; the conclusion is withheld for human review",
            expected_information_gain=0.0,
            confidence=0.8,
            requires_human_review=True,
            stop_reason="unresolved_contradictions",
        )

    if hypotheses:
        return ActionDecision(
            action_type="finalize",
            rationale="sufficient deterministic evidence gathered; no contradictions found",
            expected_information_gain=0.0,
            confidence=0.9,
            stop_reason="sufficient_evidence",
        )

    return ActionDecision(
        action_type="finalize",
        rationale="no further tools can produce new information",
        expected_information_gain=0.0,
        confidence=0.8,
        stop_reason="no_information_gain",
    )


def _has_vehicle_hypothesis(ctx: dict[str, Any]) -> bool:
    """True when a hypothesis already ties someone to a vehicle or plate."""
    for hypothesis in ctx.get("hypotheses", []):
        statement = str(hypothesis.get("statement", "")).lower()
        if "vehicle" in statement or "plate" in statement:
            return True
    return False


def estimated_information_gain(tool: str, ctx: dict[str, Any]) -> float:
    """Deterministic proxy for how much new information a tool can add.

    Used by the LLM prompt as a hint and by tests to assert the policy
    avoids rerunning tools that would produce no new facts.
    """
    completed = set(ctx.get("completed_tools", []))
    if tool in completed:
        return 0.0
    if tool == "ExtractionAgent":
        return 0.9 if ctx.get("entity_count", 0) == 0 else 0.3
    if tool == "CorrelationAgent":
        return 0.9 if ctx.get("match_count", 0) == 0 else 0.2
    if tool == "TimelineAgent":
        return 0.6 if ctx.get("timeline_count", 0) == 0 else 0.0
    if tool == "ContradictionHunter":
        return 0.7 if not ctx.get("contradictions") else 0.1
    if tool == "SkepticReview":
        return 0.5
    return 0.4
