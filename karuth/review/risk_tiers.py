"""LOW / MED / CRITICAL risk tier rules."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

RiskTier = Literal["LOW", "MED", "CRITICAL"]


class RiskAssessment(BaseModel):
    tier: RiskTier
    reason: str


def classify_content(entity_type: str, value: object | None = None) -> RiskAssessment:
    # Assumption: face embeddings and phone numbers are sensitive PII requiring
    # review before reveal; explicit high-harm tags require dual authorization.
    if entity_type in {"face_embedding", "phone"}:
        return RiskAssessment(tier="MED", reason="PII requires default blurring")
    if entity_type in {"critical_image", "high_harm_content"}:
        return RiskAssessment(tier="CRITICAL", reason="Critical reveal requires dual authorization")
    return RiskAssessment(tier="LOW", reason="No sensitive marker detected")
