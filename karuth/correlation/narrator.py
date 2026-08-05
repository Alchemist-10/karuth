"""Narrate deterministic matches without inventing new matches."""

from __future__ import annotations

from typing import Protocol

from karuth.correlation.candidate_matcher import CorrelationResult


class MatchNarrator(Protocol):
    def narrate(self, match: CorrelationResult) -> str:
        """Explain a provided deterministic match."""


class TemplateNarrator:
    """Offline-safe narrator used when no local LLM adapter is supplied."""

    def narrate(self, match: CorrelationResult) -> str:
        if match.rule == "face_embedding_cosine":
            cluster = match.metadata.get("person_cluster_id", "Person_Cluster_UNKNOWN")
            return (
                f"Face evidence matched at {match.score * 100:.1f}% geometric "
                f"similarity and was assigned to {cluster}."
            )
        return match.explanation
