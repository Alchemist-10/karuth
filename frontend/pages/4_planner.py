"""Streamlit planner page: multimodal, investigator-first investigation view.

The page runs the planner in-process (dynamic hypothesis-driven mode by
default; classic fixed-plan mode still available) and presents the result the
way an investigator would read it:

1. Investigation status
2. Primary lead
3. Evidence chain (candidate relationship graph)
4. Confidence summary (separate scores, "Not scored" instead of 0.00)
5. Investigation timeline
6. Hypotheses (readable blocks, not hidden objects)
7. Contradictions
8. Skeptic review
9. Sensitive-image controls (blurred preview, reveal requires authorization)
10. Human review checkpoints (recorded to the audit ledger)
11. Full evidence + audit ledger in collapsed sections

Raw JSON and raw ledger rows are hidden behind expanders — they are never the
primary experience. Two synthetic demo presets (clean and adversarial Jesse
Pinkman) are one click away; the adversarial preset is the pitch "wow moment".
"""

from __future__ import annotations

import hashlib
import html
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = ROOT / "frontend"
for _path in (ROOT, FRONTEND_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from components.investigation_view import (  # noqa: E402
    ACTION_LABELS,
    CONFIDENCE_LABELS,
    action_fields,
    as_dict,
    as_list,
    hypothesis_fields,
)
from components.theme import apply_theme
from karuth.config import DEFAULT_DATA_DIR
from karuth.ledger import LedgerStore
from karuth.ledger.schema import LedgerEntryCreate
from karuth.planner.llm import build_planner
from karuth.planner.state_machine import run_planner_goal

from scripts.generate_synthetic_case import JESSE_GOAL, generate_jesse_case  # noqa: E402

_TREE_ICONS = {
    "hypothesis": "🔬",
    "action": "⚡",
    "observation": "👁",
    "contradiction": "⚔",
    "conclusion": "🏁",
    "abandoned": "🚫",
}


# ---------------------------------------------------------------------------
# rendering helpers
# ---------------------------------------------------------------------------


def _render_tree(branches: list) -> None:
    """Render the investigation tree as an indented, parent-linked list."""
    if not branches:
        return
    children: dict[str, list] = {}
    for node in branches:
        children.setdefault(node.get("parent", ""), []).append(node)

    def walk(node_list: list, depth: int) -> None:
        for node in node_list:
            indent = "&nbsp;&nbsp;" * depth
            icon = _TREE_ICONS.get(node.get("kind", ""), "•")
            conf = (
                f" · conf {node['confidence']:.2f}"
                if node.get("confidence") is not None
                else ""
            )
            st.markdown(
                f"{indent}- {icon} **{html.escape(node.get('kind', ''))}**: "
                f"{html.escape(node.get('label', ''))}{conf}",
                unsafe_allow_html=True,
            )
            if node.get("detail"):
                st.markdown(
                    f"{indent}&nbsp;&nbsp;&nbsp;_ {html.escape(node['detail'])}_",
                    unsafe_allow_html=True,
                )
            walk(children.get(node.get("id", ""), []), depth + 1)

    walk(children.get("", []), 0)


def _render_evidence_chain(state: dict) -> None:
    """Evidence chain: candidate relationship graph as a readable tree."""
    from karuth.planner.candidate_graph import build_candidate_graph, render_tree_lines

    graph = build_candidate_graph(
        case_id=state.get("case_id", ""),
        entities=as_list(state.get("entities")),
        matches=as_list(state.get("matches")),
        observations=as_list(state.get("image_observations")),
        hypotheses=as_list(state.get("hypotheses")),
        contradictions=as_list(state.get("contradictions")),
        ledger_entry_id="",
    )
    st.subheader("Evidence chain")
    st.code("\n".join(render_tree_lines(graph)))
    st.caption(
        "Edge styles: solid green = deterministic fact · solid blue = metadata fact · "
        "dotted amber = AI-proposed (never confirmed) · thick purple = human-confirmed · "
        "broken red = contradiction · grey = weak/unresolved."
    )
    if graph["edges"]:
        with st.expander("Edge provenance (evidence refs, confidence, review status)"):
            rows = []
            for edge in graph["edges"]:
                rows.append(
                    {
                        "relationship": f"{edge['rel_type']} -> {edge['target']}",
                        "style": edge["style"],
                        "confidence": edge["confidence"] if edge["confidence"] is not None else "Not scored",
                        "review": edge["review_status"],
                        "evidence_refs": ", ".join(edge["evidence_refs"][:6]),
                        "source_artifact": edge["source_artifact"],
                    }
                )
            st.dataframe(rows, use_container_width=True)


def _render_confidence_summary(state: dict) -> None:
    breakdown = state.get("confidence_breakdown") or {}
    history = as_list(state.get("confidence_history"))
    st.subheader("Confidence summary")
    if not breakdown:
        st.caption("No confidence scores computed for this run.")
        return
    cols = st.columns(3)
    for index, (key, label) in enumerate(CONFIDENCE_LABELS.items()):
        value = breakdown.get(key)
        if value is None:
            text = "Not scored"
        else:
            text = f"{float(value):.2f}"
            first = next(
                (h.get(key) for h in history if h.get(key) is not None),
                None,
            )
            if first is not None and float(first) != float(value) and state.get("contradictions"):
                text = f"{float(first):.2f} -> {float(value):.2f} after contradiction review"
        cols[index % 3].metric(label, text)


def _render_timeline(state: dict) -> None:
    timeline = as_list(state.get("timeline"))
    if not timeline:
        return
    st.subheader("Investigation timeline")
    rows = []
    for entity in timeline[:12]:
        data = as_dict(entity)
        timestamp = (
            data.get("metadata", {}).get("captured_at")
            or data.get("metadata", {}).get("sent_at")
            or "unknown"
        )
        rows.append(
            {
                "time": str(timestamp),
                "type": data.get("type", ""),
                "value": str(data.get("value", ""))[:80],
                "source": data.get("source_file", ""),
            }
        )
    st.dataframe(rows, use_container_width=True)


def _render_hypotheses(state: dict) -> None:
    hypotheses = as_list(state.get("hypotheses"))
    if not hypotheses:
        return
    st.subheader("Hypotheses")
    for h in hypotheses:
        fields = hypothesis_fields(h)
        status_icon = {
            "supported": "✅",
            "rejected": "❌",
            "inconclusive": "⚠",
            "open": "🔬",
        }.get(fields["status"], "•")
        with st.expander(
            f"{status_icon} {fields['id']} · {fields['statement']} "
            f"({fields['status']})",
            expanded=False,
        ):
            conf = float(fields["confidence"]) if fields["confidence"] is not None else 0.0
            st.progress(min(1.0, max(0.0, conf)), text=f"Confidence {conf:.2f}")
            if fields["review"]:
                st.caption("Requires human confirmation before it may be relied on.")
            if fields["supporting"]:
                st.markdown(f"**Supporting evidence:** {', '.join(fields['supporting'][:10])}")
            if fields["contradicting"]:
                st.markdown(f"**Contradicting evidence:** {', '.join(fields['contradicting'][:10])}")
            if fields["missing"]:
                st.markdown(f"**Missing evidence:** {'; '.join(fields['missing'][:6])}")
            if fields["next"]:
                st.markdown(f"**Recommended next action:** {', '.join(fields['next'])}")


def _render_contradictions(state: dict) -> None:
    contradictions = as_list(state.get("contradictions"))
    if not contradictions:
        return
    st.subheader("Contradictions")
    for c in contradictions:
        severity = c.get("severity", "medium")
        icon = {"high": "🔴", "medium": "🟠", "low": "🟡"}.get(severity, "•")
        st.markdown(
            f"- {icon} **{c.get('rule', '')}** — {c.get('claim_a', '')} "
            f"**vs** {c.get('claim_b', '')}"
        )
        if c.get("evidence_refs"):
            st.caption(f"Evidence refs: {', '.join(c.get('evidence_refs', [])[:6])}")


def _render_skeptic(state: dict) -> None:
    skeptic = state.get("skeptic_review")
    if not skeptic:
        return
    skeptic = as_dict(skeptic)
    st.subheader("Skeptic review (adversarial challenge)")
    st.markdown(f"**Conclusion under review:** {skeptic.get('conclusion_under_review', '')}")
    st.markdown(f"**Strongest alternative:** {skeptic.get('strongest_alternative', '')}")
    original = float(skeptic.get("original_confidence", 0) or 0)
    revised = float(skeptic.get("revised_confidence", 0) or 0)
    st.markdown(f"**Confidence:** {original:.2f} → {revised:.2f}")
    st.caption(f"Human review required: {bool(skeptic.get('requires_human_review'))}")
    missing = skeptic.get("missing_evidence") or []
    if missing:
        st.markdown(f"**Evidence still needed:** {'; '.join(missing[:6])}")


def _render_sensitive_images(state: dict) -> None:
    from karuth.planner.observations import normalize_observations
    from karuth.vision.redaction import create_blurred_derivative

    observations = normalize_observations(as_list(state.get("image_observations")))
    sensitive = [o for o in observations if o.observation_type == "sensitive_content"]
    if not sensitive:
        return
    reveal_authorized = bool(state.get("reveal_authorized"))
    st.subheader("Sensitive-image controls")
    st.warning(
        "**Sensitive content detected.** Preview mode: blurred. Original: preserved "
        "securely. Reveal: investigator authorization required."
    )
    derivative_dir = DEFAULT_DATA_DIR / "_redacted_derivatives"
    for observation in sensitive:
        image_path = Path(observation.image_path)
        if not image_path.exists():
            continue
        reveal = st.checkbox(
            f"Reveal {image_path.name} (authorization required)",
            key=f"reveal_{observation.observation_id}",
            disabled=not reveal_authorized,
        )
        if reveal and reveal_authorized:
            st.image(str(image_path), caption=f"Original — reveal authorized ({image_path.name})")
        else:
            derivative = create_blurred_derivative(image_path, derivative_dir)
            st.image(
                derivative["path"],
                caption=f"Blurred preview ({image_path.name}) — {derivative['provenance']}",
            )
            st.caption(
                f"Original sha256 {derivative['original_sha256'][:16]}… · "
                f"derivative sha256 {derivative['sha256'][:16]}…"
            )


def _render_human_checkpoints(state: dict, ledger: LedgerStore, case_id: str) -> None:
    from karuth.planner.observations import ambiguous_plate_observations, normalize_observations

    observations = normalize_observations(as_list(state.get("image_observations")))
    person_pending = any(o.observation_type == "person" for o in observations)
    activity_pending = any(o.observation_type == "activity_interpretation" for o in observations)
    plates = ambiguous_plate_observations(observations)

    if not (person_pending or activity_pending or plates):
        return

    st.subheader("Human review checkpoints")
    if person_pending:
        st.markdown("Karuth found a candidate face match across the images.")
        identity_ok = st.checkbox("The images show the same candidate subject", key="ck_identity")
        label_ok = st.checkbox(
            "The subject may be labeled 'Jesse Pinkman' for this synthetic case",
            key="ck_label",
        )
    else:
        identity_ok = label_ok = False
    if activity_pending:
        activity_ok = st.checkbox(
            "I reviewed the LLM activity interpretation for Image 1",
            key="ck_activity",
        )
    else:
        activity_ok = False
    plate_choices: dict[str, str] = {}
    for plate in plates:
        options = [plate.value] + list(plate.alternatives)
        plate_choices[plate.observation_id] = st.radio(
            f"Plate reading is ambiguous. Please confirm the correct value before continuing. ({plate.artifact_ref})",
            options,
            key=f"ck_plate_{plate.observation_id}",
        )

    if st.button("Record decisions in the audit ledger"):
        recorded = 0
        if person_pending:
            ledger.append(
                LedgerEntryCreate(
                    entry_type="review_action",
                    case_id=case_id,
                    actor="investigator",
                    action="identity_link_confirmed",
                    target="face_association",
                    payload={"same_subject": identity_ok, "label": label_ok},
                )
            )
            recorded += 1
        if activity_pending:
            ledger.append(
                LedgerEntryCreate(
                    entry_type="review_action",
                    case_id=case_id,
                    actor="investigator",
                    action="activity_interpretation_reviewed",
                    target="activity_interpretation",
                    payload={"reviewed": activity_ok},
                )
            )
            recorded += 1
        for observation_id, choice in plate_choices.items():
            ledger.append(
                LedgerEntryCreate(
                    entry_type="review_action",
                    case_id=case_id,
                    actor="investigator",
                    action="plate_reading_confirmed",
                    target="number_plate",
                    payload={"observation_id": observation_id, "value": choice},
                )
            )
            recorded += 1
        st.success(f"{recorded} decision(s) recorded. Ledger chain verified: {ledger.verify()}")


def _render_leads(state: dict) -> None:
    """Top 3-5 leads by default; the full evidence set stays collapsed."""
    matches = as_list(state.get("matches"))
    if not matches:
        return
    st.subheader("Primary leads")
    ordered = sorted(matches, key=lambda m: getattr(m, "score", 0) or 0, reverse=True)
    for match in ordered[:5]:
        data = as_dict(match)
        st.markdown(
            f"- **{data.get('explanation', '')}** *(score {float(data.get('score', 0)):.2f}; "
            f"{data.get('rule', '')})*"
        )
    if len(ordered) > 5:
        with st.expander("Full evidence set"):
            st.dataframe(
                [
                    {
                        "rule": m.get("rule"),
                        "score": round(float(m.get("score", 0)), 3),
                        "explanation": m.get("explanation"),
                        "left": m.get("left_entity_id"),
                        "right": m.get("right_entity_id"),
                    }
                    for m in (as_dict(x) for x in ordered)
                ],
                use_container_width=True,
            )


def _render_dynamic_view(
    state: dict, backend_used: str, ledger: LedgerStore, case_id: str
) -> None:
    """The investigator-first dynamic investigation view (ordered sections)."""
    observations = as_list(state.get("image_observations"))
    observations = [
        o if isinstance(o, dict) else as_dict(o) for o in observations
    ]

    # 1. Investigation status
    st.subheader("Investigation status")
    stop_reason = state.get("stop_reason") or "n/a"
    if state.get("needs_human_review"):
        st.error(f"**HUMAN REVIEW REQUIRED** — stop reason: {stop_reason}")
    else:
        st.success(f"Investigation complete — stop reason: {stop_reason}")
    st.caption(
        f"Backend used: {backend_used} · "
        f"Fallback mode: {'yes' if backend_used == 'deterministic' else 'no'} · "
        f"Actions executed: {len(as_list(state.get('actions_taken')))}"
    )

    # 2. Primary lead
    from karuth.planner.evidence_report import _primary_lead

    st.subheader("Primary lead")
    st.markdown(
        _primary_lead(
            as_list(state.get("hypotheses")),
            as_list(state.get("matches")),
        )
    )

    # 3. Evidence chain
    _render_evidence_chain(state)

    # 4. Confidence summary
    _render_confidence_summary(state)

    # 5. Timeline
    _render_timeline(state)

    # 6. Hypotheses
    _render_hypotheses(state)

    # 7. Contradictions
    _render_contradictions(state)

    # 8. Skeptic review
    _render_skeptic(state)

    # 9. Registry (mock MVD)
    _render_registry(state)

    # 10. Sensitive-image controls
    _render_sensitive_images(state)

    # 11. Recommended next action
    recommended = _recommended_next_action(state)
    if recommended:
        st.subheader("Recommended next action")
        for item in recommended:
            st.markdown(f"- {item}")

    # 12. Human checkpoints + collapsed evidence/ledger
    _render_human_checkpoints(state, ledger, case_id)

    st.divider()
    _render_leads(state)

    _render_evidence_gap(state)

    with st.expander("Full evidence set (entities, observations, action history)"):
        if observations:
            st.markdown("**Image observations**")
            st.dataframe(
                [
                    {
                        "id": o.get("observation_id"),
                        "type": o.get("observation_type"),
                        "value": o.get("value"),
                        "confidence": (
                            "Not scored"
                            if o.get("confidence") is None
                            else f"{o.get('confidence'):.2f}"
                        ),
                        "classification": o.get("evidence_classification"),
                        "review": o.get("requires_human_review"),
                        "alternatives": ", ".join(o.get("alternatives") or []),
                    }
                    for o in observations
                ],
                use_container_width=True,
            )
        actions = as_list(state.get("actions_taken"))
        if actions:
            st.markdown("**Action history**")
            for action in actions:
                fields = action_fields(action)
                label = ACTION_LABELS.get(fields["action_type"], fields["action_type"])
                tool = f" · {fields['tool']}" if fields["tool"] else ""
                target = f" → {fields['target']}" if fields["target"] else ""
                st.markdown(
                    f"- **{label}**{tool}{target} — {fields['rationale']} "
                    f"_({fields['gain']:.2f} expected info gain)_"
                )

    _render_evidence_sources(state)

    with st.expander("Audit ledger"):
        rows = [
            {
                "seq": entry.sequence,
                "action": entry.action,
                "actor": entry.actor,
                "payload": entry.payload,
            }
            for entry in ledger.list_entries(case_id)
        ]
        st.json(rows)
        st.caption(f"Ledger chain verified: {'✓' if ledger.verify() else '✗'}")


def _render_registry(state: dict) -> None:
    """Approved mock MVD registry findings (RC owner), clearly labeled synthetic."""
    results = as_list(state.get("registry_results"))
    if not results:
        return
    st.subheader("Registry (mock MVD)")
    for result in results:
        data = as_dict(result)
        plate = data.get("key", "")
        st.markdown(
            f"**RC record for {plate}** — `{data.get('source_system', '')}` "
            "(synthetic fixture, not real data)"
        )
        st.dataframe(
            [
                {
                    "field": finding.get("field"),
                    "value": finding.get("value"),
                    "confidence": (
                        "Not scored"
                        if finding.get("confidence") is None
                        else f"{float(finding.get('confidence')):.2f}"
                    ),
                    "note": finding.get("note", ""),
                }
                for finding in data.get("findings", [])
            ],
            use_container_width=True,
        )
    st.caption(
        "Registered ownership is a registry fact; it does not prove who "
        "operated the vehicle."
    )


def _render_evidence_gap(state: dict) -> None:
    """Explain *why* nothing was found when images were ingested but no
    machine-readable observations came out. Honest gap, not silence."""
    if state.get("entities") or state.get("image_observations"):
        return
    case_folder = Path(state.get("case_folder", "."))
    images = sorted(
        path
        for suffix in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
        for path in case_folder.rglob(suffix)
    )
    if not images:
        return
    from karuth.vision.ocr import ocr_available

    try:
        ocr = ocr_available()
    except Exception:  # noqa: BLE001 - soft hint only
        ocr = False
    st.info(
        f"**No machine-readable evidence was extracted from the {len(images)} "
        "image(s).** Karuth does not guess: a photo only contributes what is "
        "machine-readable in it (EXIF GPS/camera/timestamp metadata, plate "
        "text via OCR, or a metadata sidecar)."
    )
    if ocr:
        st.caption("OCR engine: installed — but no plate-like text was found in these images.")
    else:
        st.caption(
            "OCR engine: not installed — plate text inside the photos could not "
            "be read. Install Tesseract OCR (and `pip install pytesseract`) to "
            "read plates from real photos automatically, or add an "
            "`.exif.json` metadata sidecar next to each image (see README)."
        )
    plate_in_goal = _plate_in_goal(state.get("goal", ""))
    if plate_in_goal:
        st.caption(
            f"The plate **{plate_in_goal}** appears only in your question, not in "
            "the evidence files — that is why no lead was formed. If you approve "
            "the mock MVD registry lookup in the sidebar, Karuth can still fetch "
            "the (synthetic) registered owner for that plate."
        )


def _plate_in_goal(goal: str) -> str:
    import re

    match = re.search(
        r"\b[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,3}\s?\d{3,4}\b"
        r"|\b[A-Z]{2}[- ][A-Z]{2}[- ]?\d{3,4}\b",
        goal or "",
        re.IGNORECASE,
    )
    return re.sub(r"\s+", "", match.group(0)).upper() if match else ""


def _recommended_next_action(state: dict) -> list[str]:
    """Readable next steps from the report + missing evidence."""
    from karuth.planner.observations import normalize_observations

    observations = normalize_observations(as_list(state.get("image_observations")))
    items: list[str] = []
    for observation in observations:
        if observation.observation_type == "activity_interpretation":
            items.append("Review the LLM activity interpretation before accepting it.")
        if observation.observation_type == "person" and observation.requires_human_review:
            items.append("Confirm the face/subject association across the images.")
        if observation.observation_type == "number_plate" and observation.requires_human_confirmation:
            items.append("Confirm the ambiguous plate reading before relying on it.")
    if any(o.observation_type == "sensitive_content" for o in observations):
        items.append("Sensitive imagery is blurred; reveal only with authorization.")
    missing: list[str] = []
    for h in as_list(state.get("hypotheses")):
        data = as_dict(h)
        missing.extend(as_list(data.get("missing_evidence")))
    for item in missing:
        if item not in items:
            items.append(item)
    return list(dict.fromkeys(items))


def _render_evidence_sources(state: dict) -> None:
    """Source artifacts behind the findings, with on-demand hashes."""
    entities = as_list(state.get("entities"))
    rows: list[dict] = []
    seen: set[str] = set()
    # Fall back to the images on disk when nothing was extracted, so the
    # investigator can still see which files were ingested and their hashes.
    if entities:
        for entity in entities:
            data = as_dict(entity)
            source = data.get("source_file", "")
            if not source or source in seen:
                continue
            seen.add(source)
            rows.append((source, None))
    else:
        case_folder = Path(state.get("case_folder", "."))
        for suffix in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
            for path in sorted(case_folder.rglob(suffix)):
                if str(path) not in seen:
                    seen.add(str(path))
                    rows.append((str(path), None))
    if not rows:
        return
    with st.expander("Evidence sources & hashes"):
        table: list[dict] = []
        for source, _ in rows:
            path = Path(source)
            try:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                size = path.stat().st_size
            except OSError:
                digest, size = "unreadable", None
            table.append(
                {
                    "source_file": source,
                    "sha256": digest[:16] if digest != "unreadable" else digest,
                    "bytes": size,
                }
            )
        st.dataframe(table, use_container_width=True)


def _thinking_html(text: str, *, live: bool, model: str = "") -> str:
    """Grok-style thinking box; ``live=True`` adds the animated dots + cursor."""
    safe = html.escape(text).replace("\n", "<br>")
    model_tag = f'<span class="karuth-model"> · {html.escape(model)}</span>' if model else ""
    dots = '<span class="karuth-dots"><span>.</span><span>.</span><span>.</span></span>' if live else ""
    cursor = '<span class="karuth-cursor">▍</span>' if live else ""
    return (
        '<div class="karuth-thinking">'
        f'<div class="karuth-thinking-head">🧠 Thinking{model_tag}{dots}</div>'
        f'<div class="karuth-thinking-body">{safe}{cursor}</div>'
        "</div>"
    )


def _generate_demo_case(case_folder: Path, *, include_wow_data: bool = False) -> None:
    from scripts.generate_synthetic_case import generate_case

    generate_case(case_folder, include_wow_data=include_wow_data)


WOW_GOAL = (
    "Find the person connected to vehicle KL07CX4242 and determine whether "
    "the chat activity supports that connection."
)


st.set_page_config(
    page_title="Planner",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()
# The sidebar holds all case controls; make sure it is never mistaken for
# missing content (Streamlit can collapse it to an icon on narrow windows).
st.sidebar.caption("All case controls live in this sidebar (left).")
st.title("Agentic Investigation Planner")
st.caption(
    "Type an investigator goal; Karuth extracts evidence, inspects multimodal "
    "observations, forms competing hypotheses, hunts contradictions, challenges "
    "its own conclusion, and asks you before it relies on any candidate "
    "association. The final report separates observed facts from AI "
    "interpretations."
)

with st.sidebar:
    st.header("Case")
    case_id = st.text_input("Case ID", value="case-demo")
    case_folder = st.text_input(
        "Case folder",
        value=str(DEFAULT_DATA_DIR / case_id),
        help="Folder containing the evidence (chats/, images/, bank/, ...).",
    )
    uploaded_images = st.file_uploader(
        "Upload case images (saved into images/)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Drop your real images here; they are written into the case folder and processed by the pipeline on the next run (EXIF is parsed directly).",
    )
    if uploaded_images:
        image_dir = Path(case_folder) / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        saved: list[str] = []
        for uploaded in uploaded_images:
            target = image_dir / Path(uploaded.name).name
            target.write_bytes(uploaded.getvalue())
            saved.append(target.name)
        st.success(f"Saved {len(saved)} image(s) to {image_dir}: {', '.join(saved)}")
    if st.button("Generate synthetic demo case", use_container_width=True):
        with st.spinner("Generating synthetic evidence..."):
            _generate_demo_case(Path(case_folder))
        st.success(f"Demo case written to {case_folder}")
    if st.button("🧪 Jesse demo (clean evidence)", use_container_width=True):
        with st.spinner("Generating the Jesse Pinkman synthetic case (clean variant)..."):
            generate_jesse_case(Path(case_folder), adversarial=False)
        st.session_state["goal_input"] = JESSE_GOAL
        st.session_state["planner_mode"] = "dynamic"
        st.rerun()
    if st.button("⚔️ Jesse demo (adversarial — wow moment)", use_container_width=True):
        with st.spinner("Generating the adversarial Jesse Pinkman case (planted contradictions)..."):
            generate_jesse_case(Path(case_folder), adversarial=True)
        st.session_state["goal_input"] = JESSE_GOAL
        st.session_state["planner_mode"] = "dynamic"
        st.rerun()

    st.divider()
    st.header("Execution mode")
    mode = st.radio(
        "Mode",
        options=["dynamic", "plan"],
        index=0,
        key="planner_mode",
        format_func=lambda m: {
            "dynamic": "Dynamic — hypothesis-driven (decides the next action after each step)",
            "plan": "Classic — fixed task plan",
        }[m],
        help="Dynamic mode runs the agentic loop: decide → execute → observe → ... "
        "with competing hypotheses, contradiction hunting, and a skeptic review "
        "before finalizing. Classic mode keeps the fixed-task pipeline.",
    )

    st.divider()
    st.header("Planner backend")
    backend = st.radio(
        "Backend",
        options=["openrouter", "lmstudio", "deterministic"],
        index=0,
        format_func=lambda b: {
            "openrouter": "OpenRouter (default; remote LLM, sends case state to a third party)",
            "lmstudio": "LM Studio (offline LLM, falls back to deterministic)",
            "deterministic": "Deterministic (offline keyword)",
        }[b],
        help="OpenRouter (default) sends case state summaries to a third-party service; only use it for testing, never real casework. LM Studio keeps evidence on this machine. Deterministic is fully offline. All backends fall back to deterministic when the LLM is unreachable.",
    )
    reveal_authorized = st.checkbox(
        "Reveal authorized (sensitive content may be revealed)",
        value=False,
    )
    registry_lookup_approved = st.checkbox(
        "Approve mock MVD registry lookup (registered owner)",
        value=False,
        help="Approves an approval-gated, clearly synthetic mock of an external vehicle registry (RC owner). Works for any plate in the case; registered ownership never proves who operated the vehicle.",
    )

goal = st.text_area(
    "Investigator goal",
    value="Find the person connected to vehicle NM-JP-4242 and determine whether the chat activity supports that connection.",
    height=90,
    key="goal_input",
)

if st.button("Run planner", type="primary", use_container_width=True):
    if not goal.strip():
        st.error("Enter a goal first.")
        st.stop()

    case_path = Path(case_folder)
    case_path.mkdir(parents=True, exist_ok=True)
    ledger = LedgerStore(DEFAULT_DATA_DIR / f"{case_id}.sqlite3")

    thought_buffer: list[str] = []
    thought_placeholder = st.empty()

    def _on_thought(token: str) -> None:
        thought_buffer.append(token)
        thought_placeholder.markdown(
            _thinking_html("".join(thought_buffer), live=True, model=model_name),
            unsafe_allow_html=True,
        )

    planner = build_planner(backend, thoughts_callback=_on_thought)
    model_name = getattr(planner, "_model", "") or ""
    with st.spinner("Planning and executing (LLM planning with deterministic tools)..."):
        try:
            state = run_planner_goal(
                case_id,
                case_path,
                goal.strip(),
                ledger,
                planner=planner,
                reveal_authorized=reveal_authorized,
                registry_lookup_approved=registry_lookup_approved,
                mode=mode,
            )
        except ValueError as exc:
            st.error(f"Goal rejected by guardrails: {exc}")
            st.stop()

    thought_text = "".join(thought_buffer) or "".join(getattr(planner, "last_thoughts", []) or [])
    if thought_text.strip():
        thought_placeholder.markdown(
            _thinking_html(thought_text, live=False, model=model_name),
            unsafe_allow_html=True,
        )
        st.caption(
            "Reasoning is displayed for transparency and is never written to the ledger — "
            "the ledger keeps only concise, audit-safe summaries."
        )
    else:
        thought_placeholder.empty()

    backend_used = getattr(planner, "last_backend", backend)

    # The final evidence-linked report is always generated last by the
    # dynamic_finalize_step; show it first, then the structured view.
    st.subheader("Final report")
    st.markdown(state.get("final_result") or "_no result_")

    if state.get("mode") == "dynamic" or state.get("hypotheses"):
        _render_dynamic_view(state, backend_used, ledger, case_id)
    else:
        st.divider()
        _render_leads(state)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Risk tier", state.get("risk_tier", "LOW"))
    c2.metric("Overall confidence", f"{state.get('confidence', 0.0):.2f}")
    c3.metric("Backend used", backend_used)
    c4.metric("Iterations", state.get("planner_iteration_count", 0))
    c5.metric("Ledger entries", len(ledger.list_entries(case_id)))
    c6.metric("Ledger verified", "✓" if ledger.verify() else "✗")

    st.subheader("Plan")
    plan = state.get("plan")
    if plan is not None:
        st.dataframe(
            [
                {
                    "step": task.step,
                    "tool": task.tool,
                    "reason": task.reason,
                }
                for task in plan.tasks
            ],
            use_container_width=True,
        )
        st.caption(
            f"Review required: {state.get('needs_human_review', False)} · "
            f"Authorization required: {state.get('needs_authorization', False)}"
        )
    else:
        st.info("No plan was produced.")

    with st.expander("Execution history (raw)"):
        st.json(state.get("execution_history", []))
    with st.expander("Planner thoughts"):
        st.write(state.get("planner_thoughts", []))
    with st.expander("Notes / audit trail"):
        st.write(state.get("notes", []))
    with st.expander("Ledger entries (raw)"):
        rows = [
            {
                "seq": entry.sequence,
                "action": entry.action,
                "actor": entry.actor,
                "payload": entry.payload,
            }
            for entry in ledger.list_entries(case_id)
        ]
        st.json(rows)

    st.divider()
    st.caption(
        "Same flow headlessly:  POST /planner/goal  with "
        '{"case_id": "...", "goal": "..."} on the FastAPI app.'
    )
