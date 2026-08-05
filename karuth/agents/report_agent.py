"""Case report generation interface.

Production narration may use a local LLM adapter, but tests can use this
deterministic summary generator.
"""

from __future__ import annotations

from karuth.correlation.candidate_matcher import CorrelationResult


def summarize_matches(matches: list[CorrelationResult]) -> str:
    lines = ["# Karuth Case Summary", ""]
    for match in matches:
        lines.append(f"- {match.rule}: {match.explanation} (score={match.score:.3f})")
    return "\n".join(lines)
