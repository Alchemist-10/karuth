"""Dual-authorization data model for CRITICAL reveals."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field, model_validator


class RevealAuthorization(BaseModel):
    case_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    requested_by: str = Field(min_length=1)
    approved_by: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def requires_distinct_approver(self) -> "RevealAuthorization":
        if self.requested_by == self.approved_by:
            raise ValueError("CRITICAL reveals require two distinct authorized users")
        return self


def authorize_critical_reveal(
    *,
    case_id: str,
    target_id: str,
    requested_by: str,
    approved_by: str,
    reason: str,
) -> RevealAuthorization:
    return RevealAuthorization(
        case_id=case_id,
        target_id=target_id,
        requested_by=requested_by,
        approved_by=approved_by,
        reason=reason,
    )
