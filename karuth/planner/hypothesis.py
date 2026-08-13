"""Structured investigative hypotheses.

Hypotheses are the currency of the dynamic investigation loop: the agent
proposes them, deterministic evidence moves them through an explicit lifecycle
(``open -> supported | rejected | inconclusive``), and the skeptic stage can
reopen them. The LLM may *propose* a hypothesis statement, but only
deterministic evidence (artifact/entity references) may change its status or
confidence — the same rule that governs correlation matches elsewhere in the
platform.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

HypothesisStatus = Literal["open", "supported", "rejected", "inconclusive"]

# A hypothesis is "supported" when deterministic evidence raises confidence
# above this floor; "rejected" when it falls below the reject floor or the
# evidence contradicts it strongly.
SUPPORT_THRESHOLD = 0.75
REJECT_THRESHOLD = 0.25

# Confidence deltas applied per deterministic evidence reference.
SUPPORT_DELTA = 0.10
STRONG_CONTRADICTION_DELTA = 0.20
MEDIUM_CONTRADICTION_DELTA = 0.12
WEAK_CONTRADICTION_DELTA = 0.05


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Hypothesis(BaseModel):
    """A testable claim the investigation is tracking."""

    id: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    status: HypothesisStatus = "open"
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    supporting_evidence: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    recommended_next_actions: list[str] = Field(default_factory=list)
    created_by: str = "agent"
    created_at: str = Field(default_factory=_utc_now)
    updated_at: str = Field(default_factory=_utc_now)
    iteration: int = 0
    # Entity/artifact ids this hypothesis depends on, so deterministic claims
    # and contradictions can be matched against it mechanically.
    related_entity_ids: list[str] = Field(default_factory=list)
    # Artifact (file) references behind the hypothesis, for the report/UI.
    related_artifact_ids: list[str] = Field(default_factory=list)
    # Human review is required before this hypothesis may be relied on.
    requires_human_review: bool = False


_VALID_TRANSITIONS: dict[str, set[str]] = {
    "open": {"supported", "rejected", "inconclusive"},
    "inconclusive": {"open", "supported", "rejected"},
    "supported": {"open", "inconclusive", "rejected"},
    "rejected": {"open"},
}


class HypothesisStore:
    """Owns the hypothesis set and enforces evidence-driven transitions."""

    def __init__(self, hypotheses: list[Hypothesis] | None = None) -> None:
        self._by_id: dict[str, Hypothesis] = {}
        for hypothesis in hypotheses or []:
            self._by_id[hypothesis.id] = hypothesis

    def create(
        self,
        statement: str,
        *,
        evidence: list[str] | None = None,
        confidence: float = 0.5,
        created_by: str = "agent",
        related_entity_ids: list[str] | None = None,
        related_artifact_ids: list[str] | None = None,
        recommended_next_actions: list[str] | None = None,
        requires_human_review: bool = False,
        iteration: int = 0,
    ) -> Hypothesis:
        if statement.strip() in {h.statement for h in self._by_id.values()}:
            raise ValueError(f"duplicate hypothesis statement: {statement}")
        hypothesis = Hypothesis(
            id=f"H-{len(self._by_id) + 1:03d}",
            statement=statement,
            confidence=max(0.0, min(1.0, confidence)),
            supporting_evidence=list(evidence or []),
            related_entity_ids=list(related_entity_ids or []),
            related_artifact_ids=list(related_artifact_ids or []),
            recommended_next_actions=list(recommended_next_actions or []),
            requires_human_review=requires_human_review,
            created_by=created_by,
            iteration=iteration,
        )
        self._by_id[hypothesis.id] = hypothesis
        return hypothesis

    def get(self, hypothesis_id: str) -> Hypothesis:
        try:
            return self._by_id[hypothesis_id]
        except KeyError as exc:
            raise ValueError(f"unknown hypothesis: {hypothesis_id}") from exc

    def all(self) -> list[Hypothesis]:
        return list(self._by_id.values())

    def as_state(self) -> list[Hypothesis]:
        return self.all()

    def _touch(self, hypothesis: Hypothesis) -> Hypothesis:
        hypothesis.updated_at = _utc_now()
        hypothesis.iteration += 1
        return hypothesis

    def transition(self, hypothesis_id: str, status: HypothesisStatus) -> Hypothesis:
        """Move a hypothesis to a new status, enforcing the lifecycle."""
        hypothesis = self.get(hypothesis_id)
        if status not in _VALID_TRANSITIONS.get(hypothesis.status, set()):
            raise ValueError(
                f"invalid hypothesis transition: {hypothesis.status} -> {status}"
            )
        hypothesis.status = status
        self._touch(hypothesis)
        return hypothesis

    def add_supporting(
        self, hypothesis_id: str, refs: list[str], *, delta: float = SUPPORT_DELTA
    ) -> Hypothesis:
        hypothesis = self.get(hypothesis_id)
        for ref in refs:
            if ref not in hypothesis.supporting_evidence:
                hypothesis.supporting_evidence.append(ref)
        # Contradictions cap the upside: once a hypothesis has contradicting
        # evidence, new support may be recorded but not applied until the
        # conflict is resolved.
        if not hypothesis.contradicting_evidence:
            hypothesis.confidence = round(
                min(1.0, hypothesis.confidence + delta * len(refs)), 3
            )
        if (
            hypothesis.status != "supported"
            and hypothesis.confidence >= SUPPORT_THRESHOLD
            and hypothesis.supporting_evidence
            # Candidate associations (identity, activity, person-vehicle) are
            # never auto-promoted to "supported" — that requires the human
            # confirmation checkpoint.
            and not hypothesis.requires_human_review
        ):
            hypothesis.status = "supported"
        self._touch(hypothesis)
        return hypothesis

    def add_contradicting(
        self, hypothesis_id: str, refs: list[str], *, delta: float = MEDIUM_CONTRADICTION_DELTA
    ) -> Hypothesis:
        hypothesis = self.get(hypothesis_id)
        for ref in refs:
            if ref not in hypothesis.contradicting_evidence:
                hypothesis.contradicting_evidence.append(ref)
        hypothesis.confidence = round(
            max(0.0, hypothesis.confidence - delta * len(refs)), 3
        )
        if hypothesis.status == "open" and hypothesis.contradicting_evidence:
            if hypothesis.confidence <= REJECT_THRESHOLD:
                hypothesis.status = "rejected"
            else:
                hypothesis.status = "inconclusive"
        elif hypothesis.status in ("supported", "inconclusive") and hypothesis.confidence <= REJECT_THRESHOLD:
            hypothesis.status = "rejected"
        self._touch(hypothesis)
        return hypothesis

    def add_missing(self, hypothesis_id: str, items: list[str]) -> Hypothesis:
        hypothesis = self.get(hypothesis_id)
        for item in items:
            if item not in hypothesis.missing_evidence:
                hypothesis.missing_evidence.append(item)
        self._touch(hypothesis)
        return hypothesis
