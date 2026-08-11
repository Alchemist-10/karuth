"""Streamlit entrypoint stub for Karuth."""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from components.theme import apply_theme

st.set_page_config(page_title="Karuth", layout="wide")
apply_theme()
st.title("Karuth")
st.caption("Offline forensic investigation workspace")
