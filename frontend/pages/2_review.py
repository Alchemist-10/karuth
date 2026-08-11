"""Streamlit review page — placeholder with a themed status card."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = ROOT / "frontend"
for path in (ROOT, FRONTEND_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from components.theme import page_header

page_header("Human Review", "🔎", "Human-in-the-loop gates: review authorization and sensitive-content unblur.")

st.info(
    "**Planned:** approve/reject planner authorization requests, confirm identity links "
    "(e.g. face ↔ MVD owner), and unblur sensitive imagery under dual authorization."
)
with st.expander("What this page will do"):
    st.markdown(
        "- Queue of pending authorization requests from the planner (external lookups, reveal gates)\n"
        "- Identity-link confirmation: reviewer confirms a proposed match before the graph edge is committed\n"
        "- Sensitive-content review: blur-by-default images with dual-authorization unblur\n"
        "- Every decision logged to the ledger with actor + reasoning"
    )
