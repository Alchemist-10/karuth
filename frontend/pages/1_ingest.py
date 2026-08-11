"""Streamlit ingest page — placeholder with a themed status card."""

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

page_header("Evidence Ingest", "📥", "Bring evidence into a case folder for the deterministic pipeline.")

st.info(
    "**Planned:** drop chats (SQLite), images, documents, bank exports, and audio into a "
    "case folder with hashing + ledger chain-of-custody registration."
)
with st.expander("What this page will do"):
    st.markdown(
        "- Upload or import evidence files into `.karuth-data/<case_id>/`\n"
        "- Hash every file and register it in the ledger (SHA-256 chain)\n"
        "- Write the case `manifest.json` so extraction knows what to process\n"
        "- Show per-file status: hashed ✓ · parsed ✓ · queued for extraction"
    )
