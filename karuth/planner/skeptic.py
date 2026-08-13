"""Adversarial skeptic review.

Before finalization the skeptic tries to *disprove* the main conclusion: it
gathers whatever contradicts it, lowers confidence rather than confirming the
lead, names the strongest alternative explanation, and states exactly what
evidence is still needed. The core is deterministic (contradictions and
confidence adjustment); the LLM is never used here so offline runs behave
identically. The review itself is a labeled interpretation, never a fact.
"""

from __future__ import annotations

from typing import Any

from karuth.config import SKEPTIC_CONFIDENCE_THRESHOLD
from karuth.planner.hypothesis import Hypothesis

# Confidence haircut applied when contradictions remain unresolved.
_UNCERTAINTY_FACTOR = 0.6
_REVIEWED_FACTOR = 0.85

_ALTERNATIVE_TEMPLATES: dict[str, str] = {
    "impossible_movement": (
        "The vehicle's sightings imply an implausible travel timeline; the plate "
        "may be cloned, re-registered, or the sighting metadata may be wrong, so "
        "the identity link is weaker than the raw matches suggest."
    ),
    "inconsistent_identity_vehicle": (
        "The same vehicle plate is associated with more than one person identity; "
        "the plate may be shared, cloned, or the imagery misattributed."
    ),
    "inconsistent_identity_camera": (
        "The same camera serial is associated with more than one person identity; "
        "the serial may be reused or the provenance mislabeled."
    ),
}


class SkepticReview:
    """Deterministic skeptic output for the main conclusion of a case."""

    def __init__(
        self,
        *,
        hypothesis_id: str,
        conclusion_under_review: str,
        strongest_alternative: str,
        weakening_evidence: list[str],
        missing_evidence: list[str],
        original_confidence: float,
        revised_confidence: float,
        requires_human_review: bool,
        rationale: str,
    ) -> None:
        self.hypothesis_id = hypothesis_id
        self.conclusion_under_review = conclusion_under_review
        self.strongest_alternative = strongest_alternative
        self.weakening_evidence = weakening_evidence
        self.missing_evidence = missing_evidence
        self.original_confidence = original_confidence
        self.revised_confidence = revised_confidence
        self.requires_human_review = requires_human_review
        self.rationale = rationale

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "conclusion_under_review": self.conclusion_under_review,
            "strongest_alternative": self.strongest_alternative,
            "weakening_evidence": self.weakening_evidence,
            "missing_evidence": self.missing_evidence,
            "original_confidence": self.original_confidence,
            "revised_confidence": self.revised_confidence,
            "requires_human_review": self.requires_human_review,
            "rationale": self.rationale,
        }


def _as_dict(contradiction: Any) -> dict[str, Any]:
    if isinstance(contradiction, dict):
        return contradiction
    return {
        "id": contradiction.id,
        "rule": contradiction.rule,
        "claim_a": contradiction.claim_a,
        "claim_b": contradiction.claim_b,
        "evidence_refs": contradiction.evidence_refs,
        "severity": contradiction.severity,
        "confidence": contradiction.confidence,
        "resolution_status": contradiction.resolution_status,
    }


def _contradiction_text(contradiction: dict[str, Any]) -> str:
    rule = contradiction.get("rule", "")
    if rule in _ALTERNATIVE_TEMPLATES:
        return _ALTERNATIVE_TEMPLATES[rule]
    return (
        f"{contradiction.get('claim_a')} conflicts with {contradiction.get('claim_b')}; "
        "the conclusion should not be confirmed without resolving this."
    )


def run_case_skeptic(
    *,
    conclusion: str,
    original_confidence: float,
    hypotheses: list[Hypothesis],
    contradictions: list[Any],
    risk_tier: str,
    hypothesis_id: str = "",
) -> SkepticReview:
    """Review the main conclusion against every known contradiction.

    ``hypotheses`` are used only for missing-evidence recommendations; the
    confidence adjustment is driven by unresolved contradictions alone.
    """
    normalized = [_as_dict(c) for c in contradictions]
    unresolved = [c for c in normalized if c.get("resolution_status") == "unresolved"]
    weakening: list[str] = []
    for c in unresolved:
        weakening.extend(c.get("evidence_refs", []))

    missing: list[str] = []
    for h in hypotheses:
        for item in h.missing_evidence:
            if item not in missing:
                missing.append(item)
    if unresolved:
        missing.append("independent verification of the conflicting sightings")

    if unresolved:
        revised = round(original_confidence * _UNCERTAINTY_FACTOR, 3)
        alternative = _contradiction_text(
            sorted(unresolved, key=lambda c: str(c.get("severity", "low")))[-1]
        )
    else:
        revised = round(original_confidence * _REVIEWED_FACTOR, 3)
        alternative = (
            "No contradictions were found; the remaining risk is weak or missing "
            "evidence rather than conflicting evidence."
        )

    requires_human_review = risk_tier in ("MED", "CRITICAL") or bool(unresolved)

    rules = sorted({c.get("rule", "") for c in unresolved})
    rationale = (
        f"Skeptic review of '{conclusion}': {len(unresolved)} unresolved "
        f"contradiction(s) ({', '.join(rules) or 'none'}) adjusted confidence "
        f"{original_confidence:.2f} -> {revised:.2f}. Human review required: "
        f"{requires_human_review}."
    )

    return SkepticReview(
        hypothesis_id=hypothesis_id,
        conclusion_under_review=conclusion,
        strongest_alternative=alternative,
        weakening_evidence=weakening,
        missing_evidence=missing,
        original_confidence=original_confidence,
        revised_confidence=revised,
        requires_human_review=requires_human_review,
        rationale=rationale,
    )


def skeptic_should_run(hypothesis: Hypothesis, contradictions: list[Any]) -> bool:
    """Trigger conditions for the skeptic stage (item 6 of the spec)."""
    if hypothesis.status == "supported" and hypothesis.confidence >= SKEPTIC_CONFIDENCE_THRESHOLD:
        return True
    return bool(contradictions)
