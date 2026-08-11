"""Interactive "cyber" graph widget for the Streamlit Neo4j case views.

Replaces the flat DOT renderer with a force-directed, dark-themed, interactive
graph built on Cytoscape.js. Both libraries are vendored under
``frontend/static/`` and inlined into the component, so the whole widget works
air-gapped (no CDN at runtime) -- important for a forensics tool.

Features
--------
* fcose force-directed layout with a settling animation
* neon accents per entity type, glossy radial-gradient node fills
* hover a node to highlight its neighborhood and dim everything else
* click a node to inspect it in a floating detail panel
* fit / zoom controls and a live legend
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Iterable

from streamlit.components.v1 import html as _html

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

# ---- entity type -> neon accent -------------------------------------------------
TYPE_ACCENTS: dict[str, str] = {
    "person_name": "#22d3ee",      # cyan
    "phone": "#e879f9",            # magenta
    "vehicle_plate": "#fbbf24",    # amber
    "camera_serial": "#fb923c",    # orange
    "username": "#a78bfa",         # violet
    "location": "#4ade80",         # green
    "message": "#60a5fa",          # blue
    "document_text": "#2dd4bf",    # teal
    "transaction": "#f472b6",      # pink
    "image_tag": "#a3e635",        # lime
    "image": "#a3e635",
    "face": "#a3e635",
    "cluster": "#f472b6",
}
DEFAULT_ACCENT = "#94a3b8"  # slate

# ---- relation type -> edge color ------------------------------------------------
REL_ACCENTS: dict[str, str] = {
    "MATCHES": "#38bdf8",    # sky
    "MENTIONS": "#a78bfa",   # violet
    "SENT_BY": "#34d399",    # emerald
    "FACE_MATCH": "#f472b6",  # pink
}
DEFAULT_REL_ACCENT = "#64748b"

# Types that get labels even in the dense whole-case view.
_IMPORTANT_TYPES = {
    "person_name",
    "phone",
    "vehicle_plate",
    "camera_serial",
    "username",
    "location",
    "transaction",
}


def _load_vendored(name: str) -> str:
    """Read a vendored JS file; guard against `</script>` breaking the HTML."""
    data = (_STATIC_DIR / name).read_text(encoding="utf-8")
    return data.replace("</script", "<\\/script").replace("<!--", "<\\!--")


def _gloss_data_uri(accent: str) -> str:
    """Radial-gradient SVG (as a data URI) drawn inside each node for a glossy orb."""
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64">'
        f'<defs><radialGradient id="g" cx="32%" cy="28%" r="85%">'
        f'<stop offset="0%" stop-color="{accent}" stop-opacity="0.95"/>'
        f'<stop offset="55%" stop-color="{accent}" stop-opacity="0.28"/>'
        f'<stop offset="100%" stop-color="{accent}" stop-opacity="0"/>'
        f'</radialGradient></defs>'
        f'<rect width="64" height="64" fill="url(#g)"/></svg>'
    )
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def _trim(value: object, limit: int = 28) -> str:
    text = str(value or "").replace("\\", "/")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def build_elements(
    nodes: Iterable[dict[str, Any]],
    edges: Iterable[dict[str, Any]],
    *,
    label_mode: str = "all",
    sizes: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Turn traversal rows into Cytoscape ``elements`` JSON.

    ``nodes`` rows need ``entity_id`` / ``type`` / ``value_preview`` and may
    carry ``source_file``. ``edges`` rows need ``source`` / ``target`` / ``type``
    and may carry ``score``. ``label_mode`` is ``"all"`` (search views) or
    ``"important"`` (dense whole-case view).
    """
    elements: list[dict[str, Any]] = []
    sizes = sizes or {}

    for row in nodes:
        node_type = str(row.get("type") or "entity")
        accent = TYPE_ACCENTS.get(node_type, DEFAULT_ACCENT)
        label_visible = label_mode == "all" or node_type in _IMPORTANT_TYPES
        node_id = str(row.get("entity_id") or row.get("id") or "")
        if not node_id:
            continue
        detail = str(row.get("source_file") or "")
        elements.append(
            {
                "data": {
                    "id": node_id,
                    "label": _trim(row.get("value_preview") or row.get("label") or node_type),
                    "type": node_type,
                    "accent": accent,
                    "gloss": _gloss_data_uri(accent),
                    "detail": detail,
                    "labelOpacity": 1.0 if label_visible else 0.0,
                    "size": sizes.get(node_id, 38.0),
                }
            }
        )

    for index, row in enumerate(edges):
        source = str(row.get("source") or "")
        target = str(row.get("target") or "")
        if not source or not target:
            continue
        rel = str(row.get("type") or "MATCHES")
        elements.append(
            {
                "data": {
                    "id": f"e{index}",
                    "source": source,
                    "target": target,
                    "rel": rel,
                    "edgeColor": REL_ACCENTS.get(rel, DEFAULT_REL_ACCENT),
                    "label": rel.lower().replace("_", " "),
                }
            }
        )

    return elements


_CYTO = _load_vendored("cytoscape.min.js")
_LAYOUT_BASE = _load_vendored("layout-base.js")
_COSE_BASE = _load_vendored("cose-base.js")
_FCOse = _load_vendored("cytoscape-fcose.min.js")

_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  html, body {{ margin: 0; padding: 0; height: 100%; background: #05070f; overflow: hidden; }}
  #cy {{
    width: 100%; height: {height}px;
    background:
      radial-gradient(ellipse at 50% 40%, rgba(23,37,84,0.55), transparent 70%),
      linear-gradient(rgba(56,189,248,0.045) 1px, transparent 1px),
      linear-gradient(90deg, rgba(56,189,248,0.045) 1px, transparent 1px),
      #05070f;
    background-size: 100% 100%, 42px 42px, 42px 42px, 100% 100%;
    filter: drop-shadow(0 0 6px rgba(56,189,248,0.18));
    font-family: "Segoe UI", Inter, system-ui, sans-serif;
  }}
  #panel {{
    position: absolute; top: 14px; right: 14px; width: 250px; max-height: 40%;
    overflow: auto; background: rgba(8,13,28,0.92); border: 1px solid rgba(56,189,248,0.35);
    border-radius: 10px; padding: 10px 12px; color: #cbd5e1; font-size: 12px;
    box-shadow: 0 0 18px rgba(56,189,248,0.15); display: none; backdrop-filter: blur(4px);
  }}
  #panel h4 {{ margin: 0 0 6px; color: #e2e8f0; font-size: 13px; letter-spacing: 0.5px; }}
  #panel .kv {{ margin: 3px 0; word-break: break-all; }}
  #panel .k {{ color: #64748b; }}
  #toolbar {{
    position: absolute; left: 14px; bottom: 14px; display: flex; gap: 8px; align-items: center;
  }}
  #toolbar button {{
    background: rgba(8,13,28,0.85); color: #7dd3fc; border: 1px solid rgba(56,189,248,0.4);
    border-radius: 8px; padding: 6px 12px; font-size: 12px; cursor: pointer;
    letter-spacing: 0.5px; font-family: inherit;
  }}
  #toolbar button:hover {{ background: rgba(56,189,248,0.18); }}
  #legend {{
    position: absolute; right: 14px; bottom: 14px; max-width: 230px;
    background: rgba(8,13,28,0.85); border: 1px solid rgba(56,189,248,0.25);
    border-radius: 10px; padding: 8px 10px; color: #94a3b8; font-size: 11px;
    display: {legend_display};
  }}
  #legend .row {{ display: flex; align-items: center; gap: 6px; margin: 2px 0; }}
  #legend .dot {{ width: 9px; height: 9px; border-radius: 50%; border: 1.5px solid; display: inline-block; }}
  #legend .rel {{ width: 14px; height: 2px; display: inline-block; }}
  #count {{
    position: absolute; left: 14px; top: 14px; color: #475569; font-size: 11px;
    letter-spacing: 1px; text-transform: uppercase;
  }}
</style>
</head>
<body>
<div id="count">{count_text}</div>
<div id="cy"></div>
<div id="panel"></div>
<div id="toolbar">
  <button onclick="cy.fit(undefined, 40)">⛶ fit</button>
  <button onclick="cy.layout({{name:'fcose', randomize:true, animate:true, animationDuration:900, fit:true, padding:40}}).run()">⟳ re-layout</button>
</div>
<div id="legend">{legend_html}</div>
<script>{cytoscape_js}</script>
<script>{layout_base_js}</script>
<script>{cose_base_js}</script>
<script>{fcose_js}</script>
<script>
  const ELEMENTS = {elements_json};
  const cy = cytoscape({{
    container: document.getElementById('cy'),
    elements: ELEMENTS,
    style: [
      {{
        selector: 'node',
        style: {{
          'background-color': '#0b1220',
          'background-image': 'data(gloss)',
          'background-fit': 'cover',
          'border-width': 2,
          'border-color': 'data(accent)',
          'width': 'data(size)', 'height': 'data(size)',
          'color': '#cbd5e1',
          'font-size': 10,
          'text-valign': 'bottom',
          'text-halign': 'center',
          'text-margin-y': 5,
          'text-wrap': 'wrap',
          'text-max-width': 96,
          'text-opacity': 'data(labelOpacity)',
          'label': 'data(label)',
          'overlay-opacity': 0,
        }}
      }},
      {{
        selector: 'node.important',
        style: {{ 'border-width': 3, 'font-size': 11 }}
      }},
      {{
        selector: 'edge',
        style: {{
          'curve-style': 'bezier',
          'width': 1.6,
          'line-color': 'data(edgeColor)',
          'target-arrow-color': 'data(edgeColor)',
          'target-arrow-shape': 'triangle',
          'arrow-scale': 0.75,
          'opacity': 0.55,
          'label': 'data(label)',
          'font-size': 8,
          'color': 'data(edgeColor)',
          'text-background-color': '#05070f',
          'text-background-opacity': 0.75,
          'text-background-padding': 2,
          'text-rotation': 'autorotate',
          'text-opacity': 0,
        }}
      }},
      {{ selector: 'edge.showlabels', style: {{ 'text-opacity': 1 }} }},
      {{ selector: 'node.dim, edge.dim', style: {{ 'opacity': 0.10 }} }},
      {{ selector: 'node.lit', style: {{ 'border-width': 4, 'opacity': 1 }} }},
      {{ selector: 'edge.lit', style: {{ 'opacity': 1, 'width': 2.6 }} }},
      {{ selector: 'node.selected', style: {{ 'border-width': 5, 'overlay-opacity': 0.25 }} }}
    ],
    layout: {{
      name: 'fcose',
      quality: 'default',
      randomize: true,
      animate: true,
      animationDuration: 1100,
      fit: true,
      padding: 40,
      nodeRepulsion: 9000,
      idealEdgeLength: 90,
    }},
    wheelSensitivity: 0.2,
    minZoom: 0.15, maxZoom: 3,
  }});

  // ---- neighborhood highlight on hover ------------------------------------
  function setHover(node) {{
    if (!node) {{
      cy.elements().removeClass('dim lit');
      return;
    }}
    const nb = node.closedNeighborhood();
    cy.elements().addClass('dim');
    nb.removeClass('dim').addClass('lit');
  }}

  cy.on('mouseover', 'node', (e) => setHover(e.target));
  cy.on('mouseout',  'node', () => setHover(null));
  cy.on('tap', 'node', (e) => {{
    const n = e.target;
    cy.elements().removeClass('selected');
    n.addClass('selected');
    const panel = document.getElementById('panel');
    panel.style.display = 'block';
    const data = n.data();
    const degree = n.degree();
    panel.innerHTML =
      '<h4 style="color:' + data.accent + '">' + data.label + '</h4>' +
      '<div class="kv"><span class="k">type&nbsp;&nbsp;</span>' + data.type + '</div>' +
      '<div class="kv"><span class="k">links&nbsp;&nbsp;</span>' + degree + ' connection' + (degree === 1 ? '' : 's') + '</div>' +
      (data.detail ? '<div class="kv"><span class="k">source&nbsp;&nbsp;</span>' + data.detail + '</div>' : '') +
      '<div class="kv"><span class="k">id&nbsp;&nbsp;</span><code style="color:#7dd3fc;font-size:10px">' + data.id + '</code></div>';
    cy.animate({{ center: {{ eles: n }}, duration: 350 }});
  }});
  cy.on('tap', (e) => {{
    if (e.target === cy) {{ cy.elements().removeClass('selected'); document.getElementById('panel').style.display = 'none'; }}
  }});

  // Show edge labels on hover (toggleable; keeps dense views readable).
  if ({edge_labels}) {{
    cy.on('mouseover', 'edge', (e) => e.target.addClass('showlabels'));
    cy.on('mouseout',  'edge', (e) => e.target.removeClass('showlabels'));
  }}

  window.addEventListener('resize', () => cy.resize());
</script>
</body>
</html>
"""


def _legend_html(accent_names: dict[str, str], rel_names: dict[str, str]) -> str:
    rows: list[str] = []
    for accent, name in accent_names.items():
        rows.append(
            f'<div class="row"><span class="dot" style="border-color:{accent}"></span>{name}</div>'
        )
    for color, name in rel_names.items():
        rows.append(
            f'<div class="row"><span class="rel" style="background:{color};box-shadow:0 0 6px {color}"></span>{name}</div>'
        )
    return "".join(rows)


def render_cyber_graph(
    elements: list[dict[str, Any]],
    *,
    center_id: str | None = None,
    height: int = 560,
    legend_nodes: list[str] | None = None,
    legend_edges: list[str] | None = None,
    count_text: str = "",
    edge_labels_on_hover: bool = True,
) -> None:
    """Render the interactive graph via a vendored, air-gapped HTML component."""
    if legend_nodes is None:
        legend_nodes = list(TYPE_ACCENTS)
    if legend_edges is None:
        legend_edges = ["MATCHES", "MENTIONS", "SENT_BY", "FACE_MATCH"]

    # Order the legend by how many nodes use each accent, most common first.
    accent_counts: dict[str, int] = {}
    for element in elements:
        if "source" not in element["data"]:
            accent = element["data"].get("accent", DEFAULT_ACCENT)
            accent_counts[accent] = accent_counts.get(accent, 0) + 1
    ordered = sorted(accent_counts.items(), key=lambda item: -item[1])
    accent_names = {
        accent: next(
            (t for t, a in TYPE_ACCENTS.items() if a == accent),
            "entity",
        )
        for accent, _ in ordered
    }
    rel_names = {REL_ACCENTS[rel]: rel for rel in legend_edges if rel in REL_ACCENTS}

    html_doc = _HTML_TEMPLATE.format(
        height=height,
        cytoscape_js=_CYTO,
        layout_base_js=_LAYOUT_BASE,
        cose_base_js=_COSE_BASE,
        fcose_js=_FCOse,
        elements_json=json.dumps(elements).replace("</script", "<\\/script"),
        legend_html=_legend_html(accent_names, rel_names),
        legend_display="block",
        count_text=count_text,
        edge_labels="true" if edge_labels_on_hover else "false",
    )
    _html(html_doc, height=height + 60)
