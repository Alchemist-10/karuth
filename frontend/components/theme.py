"""Shared "cyber" theme for the Streamlit app.

Injecting one block of CSS on every page keeps the whole workspace looking
like the Neo4j graph widget: a deep navy canvas with neon cyan/violet glows,
monospace headings, glowing metric cards, and styled controls. Works
air-gapped -- no fonts or assets are fetched; the stack uses system
monospace/sans families and the vendored graph widget.

Usage
-----
Every page calls ``apply_theme()`` right after ``st.set_page_config(...)``::

    st.set_page_config(page_title="...", layout="wide")
    apply_theme()

``page_header()`` is a convenience that does set_page_config + theme + a
styled title/caption block for pages with little custom chrome.
"""

from __future__ import annotations

import streamlit as st

_CYBER_CSS = """
/* ===== Karuth cyber theme ========================================= */

/* ---- base canvas: deep navy + radial neon glows ------------------ */
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
  background:
    radial-gradient(1100px 700px at 88% -8%, rgba(34, 211, 238, 0.09), transparent 55%),
    radial-gradient(900px 650px at -8% 108%, rgba(168, 85, 247, 0.08), transparent 55%),
    #05070f !important;
  color: #dbe4f0;
}

[data-testid="stHeader"] {
  background: rgba(5, 7, 15, 0.8) !important;
  backdrop-filter: blur(6px);
}

/* ---- hide boilerplate chrome ------------------------------------- */
#MainMenu, footer, [data-testid="stDecoration"], [data-testid="stToolbar"] {
  display: none !important;
}

/* ---- typography --------------------------------------------------- */
h1, h2, h3, h4, [data-testid="stTitle"] h1 {
  font-family: ui-monospace, "Cascadia Code", "JetBrains Mono", Consolas,
               "Courier New", monospace;
  letter-spacing: 0.4px;
}
h1 { color: #e8f4ff !important; text-shadow: 0 0 20px rgba(56, 189, 248, 0.35); }
h2 { color: #cfe9ff !important; }
h3, h4 { color: #9fc4e8 !important; }
p { color: #c6d4e6; }
[data-testid="stCaptionContainer"] { color: #7c93b0 !important; }

/* ---- sidebar ------------------------------------------------------- */
[data-testid="stSidebar"] {
  background: #070b16 !important;
  border-right: 1px solid rgba(56, 189, 248, 0.16);
}
[data-testid="stSidebar"] * { color: #c6d4e6; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #9fd8ff !important; }

/* ---- metric cards -------------------------------------------------- */
[data-testid="stMetric"] {
  background: linear-gradient(180deg, rgba(13, 19, 34, 0.92), rgba(8, 13, 28, 0.92));
  border: 1px solid rgba(56, 189, 248, 0.22);
  border-radius: 10px;
  padding: 10px 14px;
  box-shadow: 0 0 18px rgba(56, 189, 248, 0.06);
}
[data-testid="stMetricValue"] {
  color: #7dd3fc !important;
  font-family: ui-monospace, Consolas, "Courier New", monospace;
  text-shadow: 0 0 12px rgba(34, 211, 238, 0.35);
}
[data-testid="stMetricLabel"] { color: #7c93b0 !important; }

/* ---- buttons -------------------------------------------------------- */
.stButton > button,
button[data-testid="baseButton-secondary"] {
  background: rgba(13, 19, 34, 0.9);
  border: 1px solid rgba(56, 189, 248, 0.35);
  color: #bae6fd;
  border-radius: 8px;
  transition: all 0.15s ease;
}
.stButton > button:hover,
button[data-testid="baseButton-secondary"]:hover {
  border-color: #38bdf8;
  box-shadow: 0 0 14px rgba(56, 189, 248, 0.35);
  color: #e0f2fe;
}
button[data-testid="baseButton-primary"] {
  background: linear-gradient(90deg, #0ea5e9, #22d3ee) !important;
  border: none !important;
  color: #04121a !important;
  font-weight: 650;
}
button[data-testid="baseButton-primary"]:hover {
  box-shadow: 0 0 18px rgba(34, 211, 238, 0.55);
}

/* ---- inputs --------------------------------------------------------- */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stNumberInput"] input {
  background: #0b1220 !important;
  border: 1px solid rgba(100, 116, 139, 0.35) !important;
  color: #dbe4f0 !important;
  border-radius: 8px;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus,
[data-testid="stNumberInput"] input:focus {
  border-color: #38bdf8 !important;
  box-shadow: 0 0 0 1px rgba(56, 189, 248, 0.4), 0 0 12px rgba(56, 189, 248, 0.2) !important;
}
[data-testid="stTextInput"] label,
[data-testid="stTextArea"] label { color: #9fc4e8 !important; }

/* ---- selectbox / radio / checkbox ----------------------------------- */
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
  background: #0b1220 !important;
  border-color: rgba(100, 116, 139, 0.35) !important;
  color: #dbe4f0 !important;
  border-radius: 8px;
}
[data-testid="stSelectbox"] [data-baseweb="select"] * { color: #dbe4f0; }
[data-testid="stRadio"] label { color: #c6d4e6 !important; }
[data-testid="stRadio"] label:has(input:checked) { color: #7dd3fc !important; }
[data-testid="stCheckbox"] label span { color: #c6d4e6; }
[data-testid="stCheckbox"] label:has(input:checked) span { color: #7dd3fc !important; }

/* ---- expanders -------------------------------------------------------- */
[data-testid="stExpander"] {
  background: rgba(13, 19, 34, 0.55);
  border: 1px solid rgba(56, 189, 248, 0.14);
  border-radius: 10px;
}
[data-testid="stExpander"] summary { color: #9fc4e8; }
[data-testid="stExpander"] summary:hover { color: #7dd3fc; }

/* ---- alerts ------------------------------------------------------------ */
[data-testid="stAlert"] {
  background: rgba(13, 19, 34, 0.85) !important;
  border-radius: 10px;
  border: 1px solid rgba(56, 189, 248, 0.18);
}

/* ---- code / json -------------------------------------------------------- */
[data-testid="stCode"] pre,
[data-testid="stJson"] pre {
  background: #070b16 !important;
  border: 1px solid rgba(56, 189, 248, 0.15);
  border-radius: 10px;
}
[data-testid="stCode"] code { color: #9fd8ff; }

/* ---- dataframes ---------------------------------------------------------- */
[data-testid="stDataFrame"] {
  border: 1px solid rgba(56, 189, 248, 0.15);
  border-radius: 10px;
  overflow: hidden;
}

/* ---- divider -------------------------------------------------------------- */
hr {
  border: none !important;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(56, 189, 248, 0.45), transparent) !important;
}

/* ---- tabs ------------------------------------------------------------------ */
[data-testid="stTabs"] [data-baseweb="tab-list"] button { color: #7c93b0; }
[data-testid="stTabs"] [data-baseweb="tab-list"] button[aria-selected="true"] { color: #7dd3fc; }
[data-baseweb="tab-highlight"] { background-color: #22d3ee; }

/* ---- scrollbars -------------------------------------------------------------- */
::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(56, 189, 248, 0.22); border-radius: 5px; }
::-webkit-scrollbar-thumb:hover { background: rgba(56, 189, 248, 0.4); }

/* ---- spinner ----------------------------------------------------------------- */
[data-testid="stSpinner"] { color: #7dd3fc; }
"""


def apply_theme() -> None:
    """Inject the cyber theme CSS into the current page.

    Idempotent and cheap -- safe to call from every page (and from app.py).
    Must be called after ``st.set_page_config``.
    """
    st.markdown(f"<style>{_CYBER_CSS}</style>", unsafe_allow_html=True)


def page_header(title: str, icon: str, caption: str) -> None:
    """set_page_config + theme + styled title block for pages with light chrome."""
    st.set_page_config(page_title=title, page_icon=icon, layout="wide")
    apply_theme()
    st.markdown(
        f"<h1 style='margin-bottom:0.2rem'>{icon}&nbsp;{title}</h1>",
        unsafe_allow_html=True,
    )
    st.caption(caption)
