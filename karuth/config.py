"""Central configuration for Karuth.

Assumption: thresholds start conservative and reviewable; later phases can tune
them against synthetic and locally approved validation data.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_DATA_DIR = Path(".karuth-data")
DEFAULT_LEDGER_DB = DEFAULT_DATA_DIR / "ledger.sqlite3"

VISION_MODEL = "llava"

# LM Studio planning backend (OpenAI-compatible REST endpoint). The planner
# only talks to this endpoint over HTTP; LM Studio manages model execution.
LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
LMSTUDIO_MODEL = "local-model"
LMSTUDIO_TEMPERATURE = 0.0
LMSTUDIO_MAX_TOKENS = 1024
LMSTUDIO_TIMEOUT = 60.0

# Planner graph loop safety net: how many single-task iterations may run
# before the planner is forced to finalize.
PLANNER_MAX_ITERATIONS = 30

DEFAULT_TIMESTAMP_WINDOW_MINUTES = 15
FACE_SIMILARITY_THRESHOLD = 0.72
FUZZY_MATCH_THRESHOLD = 0.86

LOW_RISK = "LOW"
MED_RISK = "MED"
CRITICAL_RISK = "CRITICAL"
