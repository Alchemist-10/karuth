"""Correlation agent wrapping deterministic candidate matching."""

from __future__ import annotations

from karuth.correlation.candidate_matcher import CorrelationResult, match_entities
from karuth.entities.schema import Entity


def correlate_entities(entities: list[Entity]) -> list[CorrelationResult]:
    return match_entities(entities)
