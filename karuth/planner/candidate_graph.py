"""Provenance-aware candidate relationship graph.

A deterministic view of the investigation as a graph: a candidate person,
the images they appear in, the vehicle they may be associated with, the
location from image metadata, and the contradictions that weaken the lead.

Every edge carries its full evidence trail — evidence references, confidence,
provenance type, review status, source artifact, and ledger entry id — and is
styled by provenance so the UI can *visually* distinguish:

- solid green:   deterministic fact
- solid blue:    metadata fact
- dotted amber:  AI-proposed association (never presented as confirmed)
- thick purple:  human-confirmed relationship
- broken red:    contradiction
- grey:          weak or unresolved association

The graph is pure Python (works without Neo4j) and is rendered by the planner
page; an AI-proposed edge is never marked ``review_status == confirmed``.
"""

from __future__ import annotations

from typing import Any

from karuth.planner.observations import normalize_observations

EDGE_STYLES: dict[str, str] = {
    "deterministic_fact": "solid green",
    "metadata_fact": "solid blue",
    "ai_proposed": "dotted amber",
    "human_confirmed": "thick purple",
    "contradiction": "broken red",
    "weak": "grey",
}


def edge_style(provenance_type: str) -> str:
    """Visual style for an edge's provenance type (grey fallback)."""
    return EDGE_STYLES.get(provenance_type, "grey")


def _person_label(observations: list) -> str:
    for observation in normalize_observations(observations):
        if observation.observation_type == "person":
            value = observation.value
            if "synthetic fixture: " in value:
                identity = value.split("synthetic fixture: ", 1)[1].rstrip(")")
                return f"Candidate {identity.replace('_', ' ').title()}"
            return "Candidate subject"
    return "Candidate subject"


def _edge(
    *,
    source: str,
    target: str,
    rel_type: str,
    provenance_type: str,
    confidence: float | None,
    evidence_refs: list[str],
    source_artifact: str,
    review_status: str,
    ledger_entry_id: str = "",
) -> dict[str, Any]:
    return {
        "source": source,
        "target": target,
        "rel_type": rel_type,
        "style": edge_style(provenance_type),
        "provenance_type": provenance_type,
        "confidence": confidence,
        "evidence_refs": evidence_refs,
        "review_status": review_status,
        "source_artifact": source_artifact,
        "ledger_entry_id": ledger_entry_id,
    }


def build_candidate_graph(
    *,
    case_id: str = "",
    entities: list | None = None,
    matches: list | None = None,
    observations: list | None = None,
    hypotheses: list | None = None,
    contradictions: list | None = None,
    ledger_entry_id: str = "",
) -> dict[str, Any]:
    """Build ``{"nodes": [...], "edges": [...]}`` for the investigation.

    Nodes: candidate person, image artifacts, vehicle, location. Edges:

    - person -> image        ``appears_in``                 (metadata fact)
    - person -> vehicle      ``candidate_associated_with``  (AI proposed)
    - vehicle -> location    ``gps_at``                     (metadata fact)
    - person -> location     ``candidate_present_in_image_at`` (weak)
    - relevant node -> conflict ``conflicts_with``          (contradiction)
    """
    observations = normalize_observations(observations or [])
    contradictions = contradictions or []

    person_obs = [o for o in observations if o.observation_type == "person"]
    plate_obs = [o for o in observations if o.observation_type == "number_plate"]
    gps_obs = [o for o in observations if o.observation_type == "gps_location"]
    image_refs = sorted({o.artifact_ref for o in observations})

    person_id = "person:candidate"
    vehicle_id = f"vehicle:{plate_obs[0].value}" if plate_obs else ""
    location_id = f"location:{gps_obs[0].value}" if gps_obs else ""

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    node_ids: set[str] = set()

    def add_node(node_id: str, label: str, kind: str) -> None:
        if node_id in node_ids:
            return
        node_ids.add(node_id)
        nodes.append({"id": node_id, "label": label, "kind": kind})

    if person_obs:
        add_node(person_id, _person_label(observations), "person")
    for ref in image_refs:
        add_node(f"image:{ref}", f"Image {ref.split('/')[-1]}", "image")
    if plate_obs:
        add_node(vehicle_id, f"Vehicle {plate_obs[0].value}", "vehicle")
    if gps_obs:
        add_node(location_id, f"Location ({gps_obs[0].value})", "location")

    for observation in person_obs:
        image_node = f"image:{observation.artifact_ref}"
        if image_node in node_ids:
            edges.append(
                _edge(
                    source=person_id,
                    target=image_node,
                    rel_type="appears_in",
                    provenance_type="metadata_fact",
                    confidence=observation.confidence,
                    evidence_refs=[observation.observation_id, observation.artifact_ref],
                    source_artifact=observation.artifact_ref,
                    review_status="pending_human_review",
                    ledger_entry_id=ledger_entry_id,
                )
            )

    if person_obs and vehicle_id:
        vehicle_h = None
        for h in hypotheses or []:
            statement = str(h.statement if not isinstance(h, dict) else h.get("statement", "")).lower()
            if "vehicle" in statement:
                vehicle_h = h
                break
        confidence = float(getattr(vehicle_h, "confidence", 0.5)) if vehicle_h is not None else 0.5
        edges.append(
            _edge(
                source=person_id,
                target=vehicle_id,
                rel_type="candidate_associated_with",
                provenance_type="ai_proposed",
                confidence=confidence,
                evidence_refs=[],
                source_artifact=",".join(o.artifact_ref for o in person_obs),
                review_status="pending_human_review",
                ledger_entry_id=ledger_entry_id,
            )
        )

    if vehicle_id and location_id:
        edges.append(
            _edge(
                source=vehicle_id,
                target=location_id,
                rel_type="gps_at",
                provenance_type="metadata_fact",
                confidence=gps_obs[0].confidence,
                evidence_refs=[gps_obs[0].observation_id, gps_obs[0].artifact_ref],
                source_artifact=gps_obs[0].artifact_ref,
                review_status="confirmed",
                ledger_entry_id=ledger_entry_id,
            )
        )

    if person_obs and location_id:
        edges.append(
            _edge(
                source=person_id,
                target=location_id,
                rel_type="candidate_present_in_image_at",
                provenance_type="weak",
                confidence=None,
                evidence_refs=[gps_obs[0].artifact_ref],
                source_artifact=gps_obs[0].artifact_ref,
                review_status="pending_human_review",
                ledger_entry_id=ledger_entry_id,
            )
        )

    for contradiction in contradictions:
        rule = contradiction.get("rule", "") if isinstance(contradiction, dict) else getattr(contradiction, "rule", "")
        claim_a = contradiction.get("claim_a", "") if isinstance(contradiction, dict) else getattr(contradiction, "claim_a", "")
        refs = list(contradiction.get("evidence_refs", [])) if isinstance(contradiction, dict) else list(getattr(contradiction, "evidence_refs", []))
        conflict_id = f"conflict:{contradiction.get('id', rule)}" if isinstance(contradiction, dict) else f"conflict:{getattr(contradiction, 'id', rule)}"
        add_node(conflict_id, f"Contradiction: {rule}", "contradiction")
        source_node = vehicle_id if "vehicle" in claim_a.lower() or "plate" in claim_a.lower() else person_id
        if not source_node:
            source_node = vehicle_id or person_id
        if source_node and conflict_id != source_node:
            edges.append(
                _edge(
                    source=source_node,
                    target=conflict_id,
                    rel_type="conflicts_with",
                    provenance_type="contradiction",
                    confidence=None,
                    evidence_refs=refs,
                    source_artifact=",".join(refs[:4]),
                    review_status="unresolved",
                    ledger_entry_id=ledger_entry_id,
                )
            )

    return {"nodes": nodes, "edges": edges}


def render_tree_lines(graph: dict[str, Any], *, indent: str = "  ") -> list[str]:
    """Render the candidate graph as an indented investigation tree."""
    nodes = {node["id"]: node for node in graph["nodes"]}
    lines: list[str] = []
    person_node = next((node for node in graph["nodes"] if node["kind"] == "person"), None)
    if person_node is None:
        return ["(no candidate relationship graph to render)"]

    def label(node_id: str) -> str:
        node = nodes.get(node_id, {})
        return node.get("label", node_id)

    lines.append(label(person_node["id"]))
    outgoing: dict[str, list[dict]] = {}
    for edge in graph["edges"]:
        outgoing.setdefault(edge["source"], []).append(edge)

    def _review(edge: dict) -> str:
        status = edge.get("review_status", "")
        if status == "confirmed":
            return "human-confirmed"
        if status == "unresolved":
            return "unresolved"
        return "pending human review"

    for index, edge in enumerate(outgoing.get(person_node["id"], [])):
        if index == len(outgoing.get(person_node["id"], [])) - 1:
            branch = "└──"
        else:
            branch = "├──"
        conf = f"conf {edge['confidence']:.2f}" if edge.get("confidence") is not None else "Not scored"
        lines.append(
            f"{branch} {edge['rel_type']} -> {label(edge['target'])} "
            f"({edge['style']}, {conf}, {_review(edge)})"
        )
        for child_index, child in enumerate(outgoing.get(edge["target"], [])):
            if child_index == len(outgoing.get(edge["target"], [])) - 1:
                child_branch = "    └──"
            else:
                child_branch = "    ├──"
            lines.append(
                f"{child_branch} {child['rel_type']} -> {label(child['target'])} "
                f"({child['style']})"
            )
    return lines
