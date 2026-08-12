"""Streamlit planner page: enter an investigator prompt and get an LLM-backed
plan and result from the agentic planner.

The page runs the planner in-process: the goal is sent to the default LLM
backend (OpenRouter by default; LM Studio is the fully offline option) and
automatically falls back to the deterministic planner when the LLM is
unreachable or unauthenticated. Guardrails, review authorization, and the
hash-chained ledger are enforced by the planner. The same flow is available
headlessly via ``POST /planner/goal``.
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

import streamlit as st

from components.theme import apply_theme
from karuth.config import DEFAULT_DATA_DIR
from karuth.ledger import LedgerStore
from karuth.planner.llm import build_planner
from karuth.planner.state_machine import run_planner_goal

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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


def _generate_demo_case(case_folder: Path) -> None:
    from scripts.generate_synthetic_case import generate_case

    generate_case(case_folder)


st.set_page_config(page_title="Planner", page_icon="🧭", layout="wide")
apply_theme()
st.title("Agentic Investigation Planner")
st.caption(
    "Type an investigator goal; the planner reasons over the case evidence and schedules "
    "deterministic tools one at a time, then reports the result."
)

with st.sidebar:
    st.header("Case")
    case_id = st.text_input("Case ID", value="case-demo")
    case_folder = st.text_input(
        "Case folder",
        value=str(DEFAULT_DATA_DIR / case_id),
        help="Folder containing the evidence (chats/, images/, bank/, ...).",
    )
    if st.button("Generate synthetic demo case", use_container_width=True):
        with st.spinner("Generating synthetic evidence..."):
            _generate_demo_case(Path(case_folder))
        st.success(f"Demo case written to {case_folder}")

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
        "Reveal authorized (CRITICAL dual authorization granted)",
        value=False,
    )

goal = st.text_area(
    "Investigator goal",
    value="Find links for vehicle KL07CX4242 and triage the top leads",
    height=90,
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

    st.subheader("Result")
    st.markdown(state.get("final_result") or "_no result_")

    backend_used = getattr(planner, "last_backend", backend)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Risk tier", state.get("risk_tier", "LOW"))
    c2.metric("Confidence", f"{state.get('confidence', 0.0):.2f}")
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

    with st.expander("Execution history"):
        st.json(state.get("execution_history", []))
    with st.expander("Planner thoughts"):
        st.write(state.get("planner_thoughts", []))
    with st.expander("Notes / audit trail"):
        st.write(state.get("notes", []))
    with st.expander("Ledger entries"):
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
