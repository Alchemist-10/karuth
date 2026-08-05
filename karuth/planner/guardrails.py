"""Offline and case-scope guardrails."""

from __future__ import annotations

from pathlib import Path


def ensure_inside_case(case_folder: Path, target: Path) -> Path:
    case_root = case_folder.resolve()
    resolved = target.resolve()
    if resolved != case_root and case_root not in resolved.parents:
        raise ValueError(f"Target is outside case folder: {target}")
    return resolved


def reject_external_url(value: str) -> None:
    if value.startswith(("http://", "https://")) and not value.startswith(
        ("http://localhost", "http://127.0.0.1", "https://localhost", "https://127.0.0.1")
    ):
        raise ValueError("External network calls are not allowed")
