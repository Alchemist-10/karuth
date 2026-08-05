"""Pydantic models for the unified audit ledger."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

LedgerEntryType = Literal["evidence", "agent_action", "review_action", "export"]


class LedgerEntryCreate(BaseModel):
    """Input needed before hash-chain fields are assigned."""

    entry_type: LedgerEntryType
    case_id: str = Field(min_length=1)
    actor: str = Field(default="system", min_length=1)
    action: str = Field(min_length=1)
    target: str = Field(min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class LedgerEntry(LedgerEntryCreate):
    """A persisted append-only ledger entry."""

    model_config = ConfigDict(frozen=True)

    sequence: int = Field(ge=1)
    previous_hash: str
    entry_hash: str
