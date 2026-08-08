"""Strict JSON-friendly models for the agentic planner output.

These models mirror the plan JSON the planner backend returns. Everything stays
deterministic and inspectable; the LLM is only allowed to influence the task
selection, never the evidence.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ToolName = Literal[
    "ExtractionAgent",
    "CorrelationAgent",
    "TimelineAgent",
    "TriageAgent",
    "ReportAgent",
    "LegalActionAgent",
]

RiskLevel = Literal["LOW", "MED", "CRITICAL"]


class SafetyPolicy(BaseModel):
    risk_level: RiskLevel = "LOW"
    blur_by_default: bool = False
    dual_authorization_required: bool = False


class PlannerTask(BaseModel):
    step: int = Field(ge=1)
    tool: ToolName
    reason: str = Field(min_length=1)
    inputs: list[str] = Field(default_factory=list)
    fields: dict[str, Any] = Field(default_factory=dict)


class InvestigationPlan(BaseModel):
    goal: str = Field(min_length=1)
    case_scope: str = "inside_case_folder_only"
    confidence: float = Field(ge=0.0, le=1.0)
    needs_human_review: bool = False
    safety_policy: SafetyPolicy = Field(default_factory=SafetyPolicy)
    tasks: list[PlannerTask]
    expected_output: list[str] = Field(default_factory=list)
    stop_conditions: list[str] = Field(default_factory=list)
