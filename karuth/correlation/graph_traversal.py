"""Cypher traversal queries over the synced case graph.

These answer multi-hop questions ("everything linked to this vehicle through
two hops") that pairwise in-memory matching cannot. Every function degrades
gracefully to ``[]`` when Neo4j is down, so the deterministic pipeline and
callers never depend on the graph being present.
"""

from __future__ import annotations

from typing import Any

from karuth.config import NEO4J_DATABASE
from karuth.graph.neo4j_client import get_driver


def _query(cypher: str, **params: Any) -> list[dict[str, Any]]:
    """Run a read query; return ``[]`` on any graph unavailability."""
    driver = get_driver()
    if driver is None:
        return []
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(cypher, **params)
            return [dict(record) for record in result]
    except Exception:  # noqa: BLE001 - graph is best-effort enrichment only
        return []


def find_connected(entity_id: str, case_id: str | None = None) -> list[dict[str, Any]]:
    """Direct neighbors of an entity, with the edge that links them."""
    cypher = (
        "MATCH (e:Entity {entity_id: $entity_id})-[m:MATCHES]-(n:Entity) "
        "RETURN n.entity_id AS entity_id, n.type AS type, "
        "       n.value_preview AS value_preview, m.rule AS rule, "
        "       m.score AS score, m.case_id AS case_id "
        "ORDER BY m.score DESC"
    )
    rows = _query(cypher, entity_id=entity_id)
    if case_id:
        rows = [row for row in rows if row.get("case_id") in (case_id, None)]
    return rows


def multi_hop(start_id: str, max_hops: int = 3, case_id: str | None = None) -> list[dict[str, Any]]:
    """Every entity reachable from ``start_id`` within ``max_hops`` MATCHES hops."""
    # Neo4j rejects parameterized variable-length bounds, so the validated int
    # is inlined (coerced to int; any non-integer raises before reaching Neo4j).
    hops = min(max(int(max_hops), 1), 10)
    cypher = (
        "MATCH (e:Entity {entity_id: $start_id}) "
        f"MATCH path = (e)-[:MATCHES*1..{hops}]-(n:Entity) "
        "RETURN n.entity_id AS entity_id, n.type AS type, "
        "       n.value_preview AS value_preview, length(path) AS hops "
        "ORDER BY hops, n.entity_id"
    )
    rows = _query(cypher, start_id=start_id)
    if case_id:
        rows = [row for row in rows if row.get("entity_id") != start_id]
    return rows


def find_by_value(entity_type: str, value: str) -> list[dict[str, Any]]:
    """Entities of a type whose value preview contains ``value`` (e.g. a plate)."""
    return _query(
        "MATCH (n:Entity {type: $type}) "
        "WHERE toLower(n.value_preview) CONTAINS toLower($value) "
        "RETURN n.entity_id AS entity_id, n.type AS type, "
        "       n.value_preview AS value_preview, n.source_file AS source_file "
        "ORDER BY n.entity_id",
        type=entity_type,
        value=value,
    )


def paths_between(
    left_id: str, right_id: str, max_hops: int = 4
) -> list[dict[str, Any]]:
    """Shortest MATCHES path between two entities (list of node id/type pairs)."""
    hops = min(max(int(max_hops), 1), 10)
    rows = _query(
        "MATCH path = shortestPath("
        "(a:Entity {entity_id: $left_id})-[:MATCHES*.."
        f"{hops}]-"
        "(b:Entity {entity_id: $right_id})) "
        "RETURN [node IN nodes(path) | {entity_id: node.entity_id, "
        "                              type: node.type}] AS nodes",
        left_id=left_id,
        right_id=right_id,
    )
    return rows


def person_messages(person_entity_id: str) -> list[dict[str, Any]]:
    """Messages sent by a person entity (via ``SENT_BY`` edges), time-ordered."""
    return _query(
        "MATCH (p:Entity {entity_id: $person_id})-[:SENT_BY]->(m:Entity) "
        "RETURN m.entity_id AS entity_id, m.type AS type, "
        "       m.value_preview AS value_preview, m.sent_at AS sent_at, "
        "       m.source_file AS source_file "
        "ORDER BY m.sent_at",
        person_id=person_entity_id,
    )


def mentions(source_entity_id: str) -> list[dict[str, Any]]:
    """Entities mentioned by a message/document/transaction entity."""
    return _query(
        "MATCH (s:Entity {entity_id: $source_id})-[:MENTIONS]->(t:Entity) "
        "RETURN t.entity_id AS entity_id, t.type AS type, "
        "       t.value_preview AS value_preview "
        "ORDER BY t.type, t.entity_id",
        source_id=source_entity_id,
    )


def mentioned_in(target_entity_id: str) -> list[dict[str, Any]]:
    """Every message/document/transaction that mentions a target entity."""
    return _query(
        "MATCH (s:Entity)-[:MENTIONS]->(t:Entity {entity_id: $target_id}) "
        "RETURN s.entity_id AS entity_id, s.type AS type, "
        "       s.value_preview AS value_preview, s.sent_at AS sent_at, "
        "       s.source_file AS source_file "
        "ORDER BY s.sent_at",
        target_id=target_entity_id,
    )


def shared_entities(person_a: str, person_b: str) -> list[dict[str, Any]]:
    """Entities mentioned in messages sent by *both* persons (common ground)."""
    return _query(
        "MATCH (:Entity {entity_id: $a_id})-[:SENT_BY]->(:Entity)-[:MENTIONS]->(x:Entity) "
        "MATCH (:Entity {entity_id: $b_id})-[:SENT_BY]->(:Entity)-[:MENTIONS]->(x:Entity) "
        "RETURN DISTINCT x.entity_id AS entity_id, x.type AS type, "
        "       x.value_preview AS value_preview "
        "ORDER BY x.type, x.entity_id",
        a_id=person_a,
        b_id=person_b,
    )


def full_case_graph(
    case_id: str, max_edges: int = 600
) -> dict[str, Any]:
    """All entity nodes and typed edges for a case, for whole-graph rendering.

    Returns ``{"nodes": [...], "edges": [...]}`` where nodes carry
    ``entity_id/type/value_preview/source_file`` and edges carry
    ``source/target/type`` (MATCHES, MENTIONS, SENT_BY, FACE_MATCH). The edge
    count is capped so a huge case stays renderable; edges are returned in
    stable order so the most-connected entities survive the cap.
    """
    nodes = _query(
        "MATCH (n:Entity {case_id: $case_id}) "
        "RETURN n.entity_id AS entity_id, n.type AS type, "
        "       n.value_preview AS value_preview, n.source_file AS source_file "
        "ORDER BY n.type, n.entity_id",
        case_id=case_id,
    )
    cap = min(max(int(max_edges), 10), 5000)
    edges = _query(
        "MATCH (a:Entity {case_id: $case_id})-[r]-(b:Entity) "
        "WHERE r.case_id = $case_id "
        "RETURN a.entity_id AS source, b.entity_id AS target, "
        "       type(r) AS type, coalesce(r.score, 0.0) AS score "
        "ORDER BY r.score DESC, type(r), source, target "
        f"LIMIT {cap}",
        case_id=case_id,
    )
    return {"nodes": nodes, "edges": edges}


def cases_with_entities() -> list[dict[str, Any]]:
    """Case ids that have entity nodes in the graph, richest first.

    Lets the UI default to a case that actually has data instead of the first
    folder alphabetically.
    """
    return _query(
        "MATCH (n:Entity) WHERE n.case_id IS NOT NULL "
        "RETURN n.case_id AS case_id, count(n) AS entities "
        "ORDER BY entities DESC"
    )


def case_summary(case_id: str) -> dict[str, Any]:
    """Node/edge counts and person clusters for a case (demo-friendly)."""
    counts = _query(
        "MATCH (n:Entity {case_id: $case_id}) RETURN count(n) AS entities",
        case_id=case_id,
    )
    edges = _query(
        "MATCH (:Entity {case_id: $case_id})-[m:MATCHES]->() "
        "RETURN count(m) AS matches",
        case_id=case_id,
    )
    clusters = _query(
        "MATCH (c:Person_Cluster {case_id: $case_id}) "
        "RETURN c.cluster_id AS cluster_id, "
        "       count { (c)<-[:FACE_MATCH]-() } AS members "
        "ORDER BY members DESC",
        case_id=case_id,
    )
    return {
        "available": bool(counts),
        "entities": counts[0]["entities"] if counts else 0,
        "matches": edges[0]["matches"] if edges else 0,
        "clusters": clusters,
    }
