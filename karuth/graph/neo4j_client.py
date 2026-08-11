"""Best-effort Neo4j synchronization for the deterministic correlation layer.

Entities become ``Entity`` nodes, correlation matches become ``MATCHES``
edges, and person clusters become ``Person_Cluster`` nodes linked by
``FACE_MATCH``. Everything is best-effort enrichment:

* the ``neo4j`` driver is lazy-imported (the pipeline works without it),
* a down server, bad credentials, or a failed query is logged and skipped,
* the deterministic in-memory/SQLite chain never depends on the graph.

The graph answers multi-hop questions ("everything linked to this vehicle")
that pairwise matching cannot, without changing what the deterministic
matcher produces.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from karuth.config import (
    NEO4J_DATABASE,
    NEO4J_ENABLED,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
)

logger = logging.getLogger(__name__)

_driver: Any = None
_driver_error: str | None = None
_driver_lock = threading.Lock()

# A server that accepts TCP but never answers the Bolt handshake (e.g. a
# half-dead process after a restart) would otherwise block every connect
# forever. The connect+verify runs in a daemon thread and is abandoned after
# this wall-clock budget, leaving the graph disabled for the process.
NEO4J_CONNECT_BUDGET_SECONDS = 8.0


def _value_preview(value: Any, limit: int = 120) -> str:
    """Short, query-safe preview of an entity value (embeddings included)."""
    text = str(value)
    if len(text) > limit:
        text = text[:limit] + "…"
    return text


def get_driver():
    """Return the cached Neo4j driver, or ``None`` when unavailable.

    The connect and connectivity check run in a daemon thread abandoned after
    :data:`NEO4J_CONNECT_BUDGET_SECONDS`, so a half-dead server can never
    block the deterministic pipeline (best-effort enrichment only).
    """
    global _driver, _driver_error
    if _driver is not None:
        return _driver
    if not NEO4J_ENABLED or _driver_error is not None:
        return None
    with _driver_lock:
        if _driver is not None:
            return _driver
        if not NEO4J_ENABLED or _driver_error is not None:
            return None
        try:
            from neo4j import GraphDatabase  # lazy import: graph is optional
        except ImportError as exc:
            _driver_error = f"neo4j driver not installed: {exc}"
            logger.warning("Neo4j sync disabled: %s", _driver_error)
            return None

        outcome: dict[str, Any] = {}

        def _connect() -> None:
            try:
                driver = GraphDatabase.driver(
                    NEO4J_URI,
                    auth=(NEO4J_USER, NEO4J_PASSWORD),
                    connection_timeout=5,
                )
                driver.verify_connectivity()
                outcome["driver"] = driver
            except Exception as exc:  # noqa: BLE001 - best-effort graph enrichment
                outcome["error"] = exc

        worker = threading.Thread(target=_connect, daemon=True)
        worker.start()
        worker.join(timeout=NEO4J_CONNECT_BUDGET_SECONDS)
        if worker.is_alive():
            _driver_error = "Neo4j connect timed out (server not answering)"
            logger.warning("Neo4j sync disabled: %s", _driver_error)
            return None
        if "error" in outcome:
            _driver_error = f"{outcome['error']}"
            logger.warning("Neo4j sync disabled: %s", _driver_error)
            return None
        _driver = outcome.get("driver")
        return _driver


def graph_available() -> bool:
    """True when the driver is connected and the server answered a ping."""
    return get_driver() is not None


def close_driver() -> None:
    """Close the cached driver (used by tests and shutdown)."""
    global _driver, _driver_error
    if _driver is not None:
        try:
            _driver.close()
        except Exception:  # noqa: BLE001
            pass
    _driver = None
    _driver_error = None


def _run_write(cypher: str, rows: list[dict[str, Any]]) -> None:
    driver = get_driver()
    if driver is None or not rows:
        return
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            session.execute_write(
                lambda tx: tx.run(cypher, rows=rows).consume()
            )
    except Exception as exc:  # noqa: BLE001 - best-effort graph enrichment
        logger.warning("Neo4j write failed (continuing in-memory): %s", exc)


def _ensure_constraints() -> None:
    driver = get_driver()
    if driver is None:
        return
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            session.execute_write(
                lambda tx: tx.run(
                    "CREATE CONSTRAINT entity_id IF NOT EXISTS "
                    "FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE"
                ).consume()
            )
            session.execute_write(
                lambda tx: tx.run(
                    "CREATE CONSTRAINT cluster_id IF NOT EXISTS "
                    "FOR (c:Person_Cluster) REQUIRE c.cluster_id IS UNIQUE"
                ).consume()
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Neo4j constraint setup failed: %s", exc)


def sync_case(
    case_id: str,
    entities: list[Any],
    matches: list[Any],
) -> dict[str, Any]:
    """Push entities and correlation matches into Neo4j (best-effort).

    Returns ``{"synced": bool, "entities": n, "matches": n, "clusters": n}``.
    When the graph is unavailable the call is a cheap no-op and the pipeline
    continues exactly as before.
    """
    if not graph_available():
        return {"synced": False, "reason": "graph unavailable", "entities": 0, "matches": 0, "clusters": 0}

    _ensure_constraints()

    entity_rows = [
        {
            "entity_id": entity.entity_id,
            "type": entity.type,
            "value_preview": _value_preview(entity.value),
            "source_file": entity.source_file,
            "case_id": case_id,
            # Timestamp/contact metadata (e.g. chat message entities) so the
            # graph can order events and link people without re-reading files.
            "sent_at": entity.metadata.get("sent_at"),
            "contact_name": entity.metadata.get("contact_name"),
        }
        for entity in entities
    ]
    _run_write(
        "UNWIND $rows AS row "
        "MERGE (e:Entity {entity_id: row.entity_id}) "
        "SET e.type = row.type, e.value_preview = row.value_preview, "
        "    e.source_file = row.source_file, e.case_id = row.case_id, "
        "    e.sent_at = row.sent_at, e.contact_name = row.contact_name",
        entity_rows,
    )

    match_rows = [
        {
            "left": match.left_entity_id,
            "right": match.right_entity_id,
            "rule": match.rule,
            "score": float(match.score),
            "explanation": match.explanation,
            "case_id": case_id,
        }
        for match in matches
    ]
    _run_write(
        "UNWIND $rows AS row "
        "MATCH (l:Entity {entity_id: row.left}) "
        "MATCH (r:Entity {entity_id: row.right}) "
        "MERGE (l)-[m:MATCHES]->(r) "
        "SET m.rule = row.rule, m.score = row.score, "
        "    m.explanation = row.explanation, m.case_id = row.case_id",
        match_rows,
    )

    cluster_rows = [
        {
            "left": match.left_entity_id,
            "right": match.right_entity_id,
            "cluster_id": match.metadata["person_cluster_id"],
            "case_id": case_id,
        }
        for match in matches
        if match.metadata.get("person_cluster_id")
    ]
    _run_write(
        "UNWIND $rows AS row "
        "MERGE (c:Person_Cluster {cluster_id: row.cluster_id}) "
        "SET c.case_id = row.case_id "
        "WITH row, c "
        "MATCH (l:Entity {entity_id: row.left}) "
        "MATCH (r:Entity {entity_id: row.right}) "
        "MERGE (l)-[:FACE_MATCH {case_id: row.case_id}]->(c) "
        "MERGE (r)-[:FACE_MATCH {case_id: row.case_id}]->(c)",
        cluster_rows,
    )

    return {
        "synced": True,
        "entities": len(entity_rows),
        "matches": len(match_rows),
        "clusters": len(cluster_rows),
    }
