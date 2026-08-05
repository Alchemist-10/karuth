"""Triage leads by confidence and goal relevance."""

from __future__ import annotations

from karuth.correlation.candidate_matcher import CorrelationResult


def rank_leads(matches: list[CorrelationResult], goal: str = "") -> list[CorrelationResult]:
    goal_upper = goal.upper()

    def score(match: CorrelationResult) -> tuple[float, int]:
        relevance = 1 if any(token in match.explanation.upper() for token in goal_upper.split()) else 0
        return (match.score, relevance)

    return sorted(matches, key=score, reverse=True)
