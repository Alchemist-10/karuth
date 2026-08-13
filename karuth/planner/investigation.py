"""Investigation tree.

A structured, audit-safe record of the autonomous loop: goal, hypotheses,
actions, observations, contradictions, abandoned paths, and conclusions. The
UI renders this tree directly, and every entry stays concise and
reproducible (no chain-of-thought).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

BranchKind = Literal[
    "hypothesis",
    "action",
    "observation",
    "contradiction",
    "conclusion",
    "abandoned",
]


class InvestigationNode(BaseModel):
    id: str = ""
    kind: BranchKind
    label: str = Field(min_length=1)
    detail: str = ""
    refs: list[str] = Field(default_factory=list)
    iteration: int = 0
    parent: str = ""  # hypothesis id, action id, or "" for root-level nodes
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class InvestigationTree(BaseModel):
    goal: str = ""
    branches: list[InvestigationNode] = Field(default_factory=list)

    def record(
        self,
        kind: BranchKind,
        label: str,
        *,
        detail: str = "",
        refs: list[str] | None = None,
        iteration: int = 0,
        parent: str = "",
        confidence: float | None = None,
    ) -> InvestigationNode:
        node = InvestigationNode(
            id=f"{kind[:3].upper()}-{len(self.branches) + 1}",
            kind=kind,
            label=label,
            detail=detail,
            refs=refs or [],
            iteration=iteration,
            parent=parent,
            confidence=confidence,
        )
        self.branches.append(node)
        return node

    def last(self, kind: BranchKind) -> InvestigationNode | None:
        for node in reversed(self.branches):
            if node.kind == kind:
                return node
        return None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
