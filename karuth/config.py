"""Central configuration for Karuth.

Assumption: thresholds start conservative and reviewable; later phases can tune
them against synthetic and locally approved validation data.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_DATA_DIR = Path(".karuth-data")
DEFAULT_LEDGER_DB = DEFAULT_DATA_DIR / "ledger.sqlite3"

REASONING_MODEL = "llama3:8b"
VISION_MODEL = "llava"

DEFAULT_TIMESTAMP_WINDOW_MINUTES = 15
FACE_SIMILARITY_THRESHOLD = 0.72
FUZZY_MATCH_THRESHOLD = 0.86

LOW_RISK = "LOW"
MED_RISK = "MED"
CRITICAL_RISK = "CRITICAL"
