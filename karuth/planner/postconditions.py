"""Planner postcondition checks."""

from __future__ import annotations

from karuth.correlation.candidate_matcher import CorrelationResult
from karuth.entities.schema import Entity


def has_entities(entities: list[Entity]) -> bool:
    return bool(entities)


def has_matches(matches: list[CorrelationResult]) -> bool:
    return bool(matches)


def has_timeline(timeline: list[Entity]) -> bool:
    return bool(timeline)


def has_ranked_matches(ranked_matches: list[CorrelationResult]) -> bool:
    return bool(ranked_matches)
