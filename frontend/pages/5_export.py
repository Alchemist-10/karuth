"""Streamlit export page — placeholder with a themed status card."""

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

page_header("Export", "📤", "Package case artifacts and generate court-style documents.")

st.info(
    "**Planned:** export the case report as PDF, render the Section 63 BSA legal certificate, "
    "and package an evidence binder with the ledger hash chain."
)
with st.expander("What this page will do"):
    st.markdown(
        "- Case report → PDF (via the already-declared `reportlab` dependency)\n"
        "- Section 63 BSA legal certificate template\n"
        "- Evidence binder: files + hashes + ledger verification report\n"
        "- Deterministic export — no LLM involvement in what gets shipped"
    )
