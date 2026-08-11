"""Streamlit graph page: interactive Neo4j case graph built by the pipeline.

Replaces the static DOT renderer with a force-directed, dark-themed, interactive
graph (see ``components/graph_cyber.py``): hover a node to highlight its
neighborhood, click one to inspect it, and re-layout on demand. Two views:

* **whole case** — every entity and typed edge for the case, colored by entity
  type and sized by connectivity, so the investigation structure is visible
  at a glance;
* **entity search** — type a value (plate, phone, serial...) and see that
  entity centered with the sources that mention it and the entities it
  correlates with.

Everything degrades gracefully when Neo4j is not running -- the graph is
optional enrichment on top of the deterministic pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = ROOT / "frontend"

for path in (ROOT, FRONTEND_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from components.graph_cyber import (
    TYPE_ACCENTS,
    build_elements,
    render_cyber_graph,
)
from components.theme import apply_theme
from karuth.config import DEFAULT_DATA_DIR
from karuth.correlation import graph_traversal
from karuth.graph.neo4j_client import graph_available

st.set_page_config(page_title="Graph", page_icon="🕸️", layout="wide")
apply_theme()
st.title("Case Graph · Neo4j")
st.caption(
    "Force-directed view of entities, correlation matches, person clusters, and "
    "mentions mined from chat / document / bank text. Hover to highlight a node's "
    "neighborhood, click a node to inspect it."
)

if not graph_available():
    st.warning(
        "Neo4j is not reachable. The graph is optional enrichment — the "
        "deterministic pipeline still runs fully in-memory/SQLite. "
        "Start Neo4j (e.g. `bin\\neo4j.bat console` in `~/karuth-neo4j`) to see the graph."
    )
    st.stop()

available_cases = sorted(
    path.name
    for path in DEFAULT_DATA_DIR.iterdir()
    if path.is_dir()
    and not path.name.endswith(".sqlite3")
    and any(
        (path / sub).is_dir()
        for sub in ("chats", "images", "documents", "bank", "audio")
    )
)
default_case = "demo-case" if "demo-case" in available_cases else (available_cases[0] if available_cases else "demo-case")

with st.sidebar:
    st.subheader("Case")
    if available_cases:
        # Default to a case that actually has graph data (richest first),
        # preferring the demo case when present -- not the first folder
        # alphabetically, which may never have been run through the pipeline.
        data_counts = {
            row["case_id"]: row["entities"]
            for row in graph_traversal.cases_with_entities()
        }
        ordered_cases = sorted(
            available_cases, key=lambda name: (-data_counts.get(name, 0), name)
        )
        default_idx = ordered_cases.index("demo-case") if "demo-case" in ordered_cases else 0
        case_id = st.selectbox("Case ID", ordered_cases, index=default_idx)
    else:
        case_id = st.text_input("Case ID", value="demo-case")
    show_edge_labels = st.toggle("Edge labels on hover", value=True, help="Show relation labels when hovering an edge.")

summary = graph_traversal.case_summary(case_id)

c1, c2, c3 = st.columns(3)
c1.metric("Entity nodes", summary["entities"])
c2.metric("Correlation matches", summary["matches"])
c3.metric("Person clusters", len(summary["clusters"]))

# ---- whole-case overview -------------------------------------------------------
st.divider()
st.subheader("Whole case")
st.caption(
    "Every entity and typed edge for the case. Nodes are colored by entity type, "
    "sized by connectivity. Use **re-layout** if the initial placement is busy."
)
full = graph_traversal.full_case_graph(case_id)
if full["nodes"]:
    # Size by degree (connectivity): count edges touching each node.
    degrees: dict[str, float] = {}
    for edge in full["edges"]:
        for key in (edge["source"], edge["target"]):
            degrees[key] = degrees.get(key, 0) + 1
    max_degree = max(degrees.values()) if degrees else 1
    sizes = {
        node_id: 24.0 + 30.0 * (degree / max_degree) ** 0.6
        for node_id, degree in degrees.items()
    }
    elements = build_elements(
        full["nodes"], full["edges"], label_mode="important", sizes=sizes
    )
    render_cyber_graph(
        elements,
        height=620,
        count_text=f"{len(full['nodes'])} nodes · {len(full['edges'])} edges",
        edge_labels_on_hover=show_edge_labels,
    )
else:
    st.info(
        f"No entities found for case `{case_id}` in Neo4j — the graph only "
        "holds what the deterministic pipeline synced. Run the planner on this "
        "case (Planner page) or pick a case that has been run already."
    )

# ---- entity search -------------------------------------------------------------
st.divider()
st.subheader("Find an entity")
search = st.text_input(
    "Search an entity value (plate, phone, serial...)",
    placeholder="e.g. KL07CX4242",
)
if search.strip():
    grouped_rows: dict[tuple[str, str], dict[str, object]] = {}
    for entity_type in ("vehicle_plate", "phone", "camera_serial", "person_name", "username", "location"):
        rows = graph_traversal.find_by_value(entity_type, search.strip())
        for row in rows:
            key = (str(row["type"]), str(row["value_preview"]))
            bucket = grouped_rows.setdefault(
                key,
                {
                    "representative": row,
                    "entity_ids": [],
                    "source_files": [],
                },
            )
            bucket["entity_ids"].append(row["entity_id"])
            bucket["source_files"].append(row["source_file"])

    if grouped_rows:
        for _, bucket in grouped_rows.items():
            representative = bucket["representative"]
            entity_id = representative["entity_id"]
            mentions = graph_traversal.mentioned_in(entity_id)
            correlated = graph_traversal.find_connected(entity_id, case_id=case_id)
            sent_by = graph_traversal.person_messages(entity_id) if representative["type"] == "person_name" else []
            unique_sources = sorted(set(bucket["source_files"]))

            st.subheader(f"{representative['value_preview']}  ·  {representative['type']}")
            st.caption(
                f"{len(bucket['entity_ids'])} graph node(s) · {len(unique_sources)} source file(s) · "
                f"{len(mentions)} mentions · {len(correlated)} correlated neighbors"
            )

            # Build a centered neighborhood graph: center + mention sources on
            # the left + correlated neighbors on the right (all typed edges).
            center_node = {
                "entity_id": entity_id,
                "type": representative["type"],
                "value_preview": representative["value_preview"],
                "source_file": next(iter(unique_sources), ""),
            }
            nodes = [center_node]
            edges: list[dict[str, object]] = []
            for row in mentions:
                nodes.append(
                    {
                        "entity_id": row["entity_id"],
                        "type": row["type"],
                        "value_preview": str(row.get("value_preview") or row.get("source_file") or row["type"]),
                        "source_file": row.get("source_file", ""),
                    }
                )
                edges.append({"source": row["entity_id"], "target": entity_id, "type": "MENTIONS"})
            for row in correlated:
                nodes.append(
                    {
                        "entity_id": row["entity_id"],
                        "type": row["type"],
                        "value_preview": row["value_preview"],
                        "source_file": "",
                    }
                )
                edges.append(
                    {
                        "source": entity_id,
                        "target": row["entity_id"],
                        "type": "MATCHES",
                    }
                )
            for row in sent_by:
                nodes.append(
                    {
                        "entity_id": row["entity_id"],
                        "type": row["type"],
                        "value_preview": str(row.get("value_preview") or row.get("source_file") or row["type"]),
                        "source_file": row.get("source_file", ""),
                    }
                )
                edges.append({"source": entity_id, "target": row["entity_id"], "type": "SENT_BY"})

            elements = build_elements(
                nodes,
                edges,
                label_mode="all",
                sizes={entity_id: 52.0},
            )
            render_cyber_graph(
                elements,
                center_id=entity_id,
                height=520,
                count_text="entity neighborhood",
                edge_labels_on_hover=show_edge_labels,
            )

            with st.expander("Sources and matches"):
                st.write("Source files")
                st.write(unique_sources)
                st.write("Mention rows")
                st.dataframe(mentions, use_container_width=True)
                st.write("Correlation neighbors")
                st.dataframe(correlated, use_container_width=True)
    else:
        st.info("No entities matched that value in the selected case.")

# ---- person clusters -----------------------------------------------------------
st.divider()
st.subheader("Person clusters")
if summary["clusters"]:
    st.dataframe(
        [
            {"cluster_id": cluster["cluster_id"], "members": cluster["members"]}
            for cluster in summary["clusters"]
        ],
        use_container_width=True,
    )
else:
    st.info("No person clusters in this case yet (face-embedding matching).")

st.divider()
with st.expander("Legend — entity types"):
    for entity_type, accent in sorted(TYPE_ACCENTS.items()):
        st.markdown(
            f"<span style='display:inline-block;width:10px;height:10px;border-radius:50%;"
            f"border:2px solid {accent};margin-right:8px'></span>{entity_type}",
            unsafe_allow_html=True,
        )
