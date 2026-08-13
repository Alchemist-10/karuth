"""Observation and critic stage.

After every action the observer summarizes what changed, mechanically moves
hypothesis confidence with deterministic evidence (claims carrying
``deterministic_fact`` / ``contradiction`` types), records supporting and
contradicting references, notes missing evidence, and recommends the next
loop transition. Schema/provenance checks are deterministic; the LLM is never
required here.
"""

from __future__ import annotations

from typing import Any, Literal

from karuth.planner.hypothesis import (
    MEDIUM_CONTRADICTION_DELTA,
    SUPPORT_DELTA,
    WEAK_CONTRADICTION_DELTA,
    Hypothesis,
    HypothesisStore,
)
from karuth.planner.tool_result import Claim, ToolResult

Recommendation = Literal["continue", "replan", "challenge", "ask_investigator", "stop"]


class Observation:
    """Audit-safe summary of one action's effects on the investigation."""

    def __init__(
        self,
        *,
        iteration: int,
        tool: str,
        summary: str,
        hypothesis_updates: list[dict[str, Any]],
        supporting_evidence: list[str],
        contradictions: list[str],
        missing_evidence: list[str],
        recommendation: Recommendation,
        explanation: str,
    ) -> None:
        self.iteration = iteration
        self.tool = tool
        self.summary = summary
        self.hypothesis_updates = hypothesis_updates
        self.supporting_evidence = supporting_evidence
        self.contradictions = contradictions
        self.missing_evidence = missing_evidence
        self.recommendation = recommendation
        self.explanation = explanation

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "tool": self.tool,
            "summary": self.summary,
            "hypothesis_updates": self.hypothesis_updates,
            "supporting_evidence": self.supporting_evidence,
            "contradictions": self.contradictions,
            "missing_evidence": self.missing_evidence,
            "recommendation": self.recommendation,
            "explanation": self.explanation,
        }


def _contradiction_delta(severity: str) -> float:
    if severity == "high":
        return 0.20
    if severity == "medium":
        return MEDIUM_CONTRADICTION_DELTA
    return WEAK_CONTRADICTION_DELTA


def observe_result(
    hypotheses: list[Hypothesis],
    result: ToolResult,
    *,
    iteration: int,
    extra_contradictions: list[dict[str, Any]] | None = None,
    suppress_support: bool = False,
) -> Observation:
    """Turn one :class:`ToolResult` into an observation with confidence updates.

    Deterministic claims whose evidence refs overlap a hypothesis's related
    entity ids support it; contradiction claims (or contradictions carried by
    the tool result) subtract confidence. Returns the observation; the caller
    applies it to the store and state.

    ``suppress_support=True`` (post-skeptic iterations) records supporting
    references without raising confidence — once the skeptic has reviewed a
    lead, only human-confirmed evidence may raise it again.
    """
    store = HypothesisStore(hypotheses)
    updates: list[dict[str, Any]] = []
    supporting: list[str] = []
    contradiction_ids: list[str] = []
    missing: list[str] = []

    def related(h: Hypothesis, refs: list[str]) -> bool:
        return bool(set(refs) & set(h.related_entity_ids))

    for claim in result.claims:
        if claim.claim_type in ("deterministic_fact", "hypothesis_support"):
            for hypothesis in store.all():
                if related(hypothesis, claim.evidence_refs):
                    store.add_supporting(
                        hypothesis.id,
                        claim.evidence_refs,
                        delta=0.0 if suppress_support else max(SUPPORT_DELTA, round(claim.confidence * 0.1, 3)),
                    )
            supporting.extend(claim.evidence_refs)
        elif claim.claim_type == "contradiction":
            # Severity-aware contradiction handling happens below via
            # extra_contradictions; applying both would double-count.
            if extra_contradictions:
                continue
            contradiction_ids.extend(claim.evidence_refs)
            for hypothesis in store.all():
                if related(hypothesis, claim.evidence_refs):
                    store.add_contradicting(hypothesis.id, claim.evidence_refs)

    for raw in extra_contradictions or []:
        cid = str(raw.get("id", ""))
        if cid:
            contradiction_ids.append(cid)
        for hypothesis in store.all():
            if related(hypothesis, raw.get("evidence_refs", [])):
                store.add_contradicting(
                    hypothesis.id,
                    raw.get("evidence_refs", []),
                    delta=_contradiction_delta(str(raw.get("severity", "medium"))),
                )

    for hypothesis in store.all():
        if hypothesis.missing_evidence:
            missing.extend(hypothesis.missing_evidence)

    if not result.success:
        recommendation: Recommendation = "replan"
    elif contradiction_ids:
        recommendation = "challenge"
    elif all(h.status in ("supported", "rejected") for h in store.all()) and store.all():
        recommendation = "stop"
    else:
        recommendation = "continue"

    explanation = (
        f"After {result.tool}: {result.summary or 'no change'}. "
        f"Supporting refs: {len(supporting)}; contradictions: {len(contradiction_ids)}; "
        f"recommendation: {recommendation}."
    )

    for hypothesis in store.all():
        updates.append(
            {
                "id": hypothesis.id,
                "status": hypothesis.status,
                "confidence": hypothesis.confidence,
                "supporting": list(hypothesis.supporting_evidence),
                "contradicting": list(hypothesis.contradicting_evidence),
            }
        )

    return Observation(
        iteration=iteration,
        tool=result.tool,
        summary=result.summary or "",
        hypothesis_updates=updates,
        supporting_evidence=list(dict.fromkeys(supporting)),
        contradictions=list(dict.fromkeys(contradiction_ids)),
        missing_evidence=list(dict.fromkeys(missing)),
        recommendation=recommendation,
        explanation=explanation,
    )
