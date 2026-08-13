"""Structured tool results.

Every planner tool execution in the dynamic loop returns a :class:`ToolResult`:
a success flag, the deterministic state updates it produced, a concise summary,
and — critically — a list of :class:`Claim` objects. Claims are the only place
evidence references may be *created*: ``claim_type="deterministic_fact"`` is
reserved for output derived from deterministic tool state (entities, matches,
timelines, contradictions). Free-form LLM text may appear only as
``"interpretation"`` claims and can never populate evidence references.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ClaimType = Literal[
    "deterministic_fact",
    "interpretation",
    "hypothesis_support",
    "contradiction",
]


class Claim(BaseModel):
    """One evidence-linked statement produced by a tool execution."""

    claim: str = Field(min_length=1)
    evidence_refs: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    claim_type: ClaimType = "deterministic_fact"


class ToolResult(BaseModel):
    """Structured outcome of one planner tool execution."""

    success: bool = True
    tool: str = ""
    state_updates: dict[str, Any] = Field(default_factory=dict)
    summary: str = ""
    claims: list[Claim] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    limitations: list[str] = Field(default_factory=list)
    suggested_follow_ups: list[str] = Field(default_factory=list)
    contradictions: list[dict[str, Any]] = Field(default_factory=list)
    duration_ms: int = 0
    error: str = ""
    iteration: int = 0
