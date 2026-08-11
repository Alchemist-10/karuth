"""Correlation agent wrapping deterministic candidate matching."""

from __future__ import annotations

from karuth.correlation.candidate_matcher import CorrelationResult, match_entities
from karuth.entities.schema import Entity


def correlate_entities(
    entities: list[Entity],
    *,
    case_id: str | None = None,
    sync_graph: bool = True,
) -> list[CorrelationResult]:
    """Deterministically correlate extracted entities.

    Matching itself is pure and unchanged. When ``case_id`` is provided and
    ``sync_graph`` is enabled, the entities and matches are pushed into Neo4j
    as best-effort enrichment afterwards -- a down or missing graph never
    changes the returned matches.
    """
    results = match_entities(entities)
    if sync_graph and case_id:
        try:
            from karuth.graph.neo4j_client import sync_case

            sync_case(case_id, entities, results)
        except Exception:  # noqa: BLE001 - graph is optional enrichment
            pass
    return results
