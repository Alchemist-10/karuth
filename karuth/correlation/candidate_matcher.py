"""Deterministic candidate matching.

LLMs are deliberately excluded from this module. They may narrate returned
matches later, but every match here comes from deterministic rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from itertools import combinations
from typing import Any

from karuth.config import (
    DEFAULT_TIMESTAMP_WINDOW_MINUTES,
    FACE_SIMILARITY_THRESHOLD,
    FUZZY_MATCH_THRESHOLD,
)
from karuth.entities.schema import Entity


@dataclass(frozen=True)
class CorrelationResult:
    left_entity_id: str
    right_entity_id: str
    rule: str
    score: float
    explanation: str
    metadata: dict[str, Any] = field(default_factory=dict)


EXACT_TYPES = {"phone", "username", "camera_serial", "vehicle_plate"}
FUZZY_TYPES = {"person_name", "vehicle_plate"}


def _similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left.upper(), right.upper()).ratio()


def _timestamp(entity: Entity) -> datetime | None:
    for key in ("sent_at", "captured_at", "timestamp"):
        value = entity.metadata.get(key)
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                continue
    return None


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def match_entities(
    entities: list[Entity],
    *,
    timestamp_window_minutes: int = DEFAULT_TIMESTAMP_WINDOW_MINUTES,
    fuzzy_threshold: float = FUZZY_MATCH_THRESHOLD,
    face_threshold: float = FACE_SIMILARITY_THRESHOLD,
) -> list[CorrelationResult]:
    results: list[CorrelationResult] = []
    for left, right in combinations(entities, 2):
        if left.source_file == right.source_file and left.type != "face_embedding":
            continue

        if left.type == right.type and left.type in EXACT_TYPES and left.value == right.value:
            results.append(
                CorrelationResult(
                    left.entity_id,
                    right.entity_id,
                    "exact_string",
                    1.0,
                    f"Exact {left.type} match on {left.value}",
                )
            )
            continue

        if left.type == right.type and left.type in FUZZY_TYPES:
            score = _similarity(str(left.value), str(right.value))
            if score >= fuzzy_threshold and str(left.value) != str(right.value):
                results.append(
                    CorrelationResult(
                        left.entity_id,
                        right.entity_id,
                        "fuzzy_string",
                        score,
                        f"Fuzzy {left.type} match",
                    )
                )
                continue

        left_time = _timestamp(left)
        right_time = _timestamp(right)
        if left_time and right_time:
            delta = abs(left_time - right_time)
            if delta <= timedelta(minutes=timestamp_window_minutes):
                score = 1.0 - (delta / timedelta(minutes=timestamp_window_minutes))
                results.append(
                    CorrelationResult(
                        left.entity_id,
                        right.entity_id,
                        "timestamp_proximity",
                        round(score, 6),
                        f"Timestamps are within {timestamp_window_minutes} minutes",
                        {"delta_seconds": delta.total_seconds()},
                    )
                )
                continue

        if left.type == right.type == "face_embedding":
            score = cosine_similarity(list(left.value), list(right.value))
            if score >= face_threshold:
                cluster_id = _cluster_id(left.entity_id, right.entity_id)
                results.append(
                    CorrelationResult(
                        left.entity_id,
                        right.entity_id,
                        "face_embedding_cosine",
                        round(score, 6),
                        f"Face embeddings matched into {cluster_id}",
                        {"person_cluster_id": cluster_id},
                    )
                )
    return results


def _cluster_id(left_id: str, right_id: str) -> str:
    first = min(left_id, right_id)[:6].upper()
    return f"Person_Cluster_{first}"
