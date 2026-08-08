"""Offline and case-scope guardrails.

The planner must never touch external URLs, network endpoints, or anything
outside the selected case folder. These checks are mandatory before plan
generation and before every dispatch.
"""

from __future__ import annotations

import os
from pathlib import Path

from karuth.planner.schema import InvestigationPlan

_EXTERNAL_SCHEMES = ("http://", "https://", "ftp://", "file://", "ws://", "wss://")
_LOCALHOST_PREFIXES = (
    "http://localhost",
    "http://127.0.0.1",
    "https://localhost",
    "https://127.0.0.1",
)


def ensure_inside_case(case_folder: Path, target: Path) -> Path:
    case_root = case_folder.resolve()
    resolved = target.resolve()
    if resolved != case_root and case_root not in resolved.parents:
        raise ValueError(f"Target is outside case folder: {target}")
    return resolved


def reject_external_url(value: str) -> None:
    lowered = value.strip().lower()
    if lowered.startswith(_EXTERNAL_SCHEMES) and not lowered.startswith(_LOCALHOST_PREFIXES):
        raise ValueError("External network calls are not allowed")


def _is_path_like(value: str) -> bool:
    if Path(value).is_absolute():
        return True
    if os.sep in value:
        return True
    alt_sep = getattr(os, "altsep", None)
    if alt_sep and alt_sep in value:
        return True
    return value.startswith((".", "~"))


def assert_plan_safe(plan: InvestigationPlan, case_folder: Path) -> None:
    """Reject plans that reference external URLs or files outside the case."""
    reject_external_url(plan.goal)
    for task in plan.tasks:
        for item in task.inputs:
            if item.startswith(_EXTERNAL_SCHEMES):
                raise ValueError(f"External URL in planner task input: {item}")
            if _is_path_like(item):
                ensure_inside_case(case_folder, Path(item))
