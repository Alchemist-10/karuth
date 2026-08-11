"""Central configuration for Karuth.

Every setting is env-overridable (``KARUTH_*`` / ``OPENROUTER_*``) and a
``.env`` file next to the project root is loaded automatically (real
environment variables always win over ``.env`` values). The ``.env.example``
file is the committed template; secrets such as ``OPENROUTER_API_KEY`` belong
only in your local git-ignored ``.env``.

Assumption: thresholds start conservative and reviewable; later phases can tune
them against synthetic and locally approved validation data.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    # Load the project-root .env regardless of the process cwd, so the app
    # finds it whether launched from the repo root, Streamlit, uvicorn, or
    # an IDE. Real environment variables always win over .env values.
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    _DOTENV_LOADED = True
except ImportError:
    _DOTENV_LOADED = False

DEFAULT_DATA_DIR = Path(os.environ.get("KARUTH_DATA_DIR", ".karuth-data"))
DEFAULT_LEDGER_DB = Path(
    os.environ.get("KARUTH_LEDGER_DB", str(DEFAULT_DATA_DIR / "ledger.sqlite3"))
)

VISION_MODEL = os.environ.get("KARUTH_VISION_MODEL", "")

# Image suffixes the extraction layer can process with the local OpenCV DNN
# pipeline (JPEG/PNG/WebP/BMP decode cleanly with cv2.imdecode).
SUPPORTED_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

# LM Studio planning backend (OpenAI-compatible REST endpoint). The planner
# only talks to this endpoint over HTTP; LM Studio manages model execution.
# All settings are env-overridable (KARUTH_LMSTUDIO_*) so the model, server,
# or limits can change without touching planner code. An empty LMSTUDIO_MODEL
# means "auto-discover the first loaded chat-capable model on the server".
LMSTUDIO_BASE_URL = os.environ.get("KARUTH_LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
LMSTUDIO_MODEL = os.environ.get("KARUTH_LMSTUDIO_MODEL", "")
LMSTUDIO_TEMPERATURE = float(os.environ.get("KARUTH_LMSTUDIO_TEMPERATURE", "0.0"))
LMSTUDIO_MAX_TOKENS = int(os.environ.get("KARUTH_LMSTUDIO_MAX_TOKENS", "1024"))
LMSTUDIO_TIMEOUT = float(os.environ.get("KARUTH_LMSTUDIO_TIMEOUT", "60.0"))

# OpenRouter backend (OpenAI-compatible REST endpoint). Opt-in only: LM Studio
# remains the offline default. The API key is read from the OPENROUTER_API_KEY
# environment variable and is never stored in code or the ledger.
OPENROUTER_BASE_URL = os.environ.get("KARUTH_OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.environ.get("KARUTH_OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
OPENROUTER_TEMPERATURE = float(os.environ.get("KARUTH_OPENROUTER_TEMPERATURE", "0.0"))
OPENROUTER_MAX_TOKENS = int(os.environ.get("KARUTH_OPENROUTER_MAX_TOKENS", "2048"))
OPENROUTER_TIMEOUT = float(os.environ.get("KARUTH_OPENROUTER_TIMEOUT", "120.0"))

# Planner graph loop safety net: how many single-task iterations may run
# before the planner is forced to finalize.
PLANNER_MAX_ITERATIONS = 30
# Real face recognition (OpenCV Zoo models, downloaded on first use into the
# model cache). YuNet detects faces; SFace produces 128-d recognition
# embeddings. Override the cache dir with KARUTH_MODEL_DIR.
FACE_MODEL_DIR = Path(os.environ.get("KARUTH_MODEL_DIR", str(Path.home() / ".cache" / "karuth" / "models")))
YUNET_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
SFACE_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx"
)

# Neo4j graph layer. Best-effort enrichment only: the deterministic pipeline
# runs fully in-memory/SQLite when the server or driver is unavailable.
NEO4J_URI = os.environ.get("KARUTH_NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("KARUTH_NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("KARUTH_NEO4J_PASSWORD", "karuth-local-only")
NEO4J_DATABASE = "neo4j"
NEO4J_ENABLED = os.environ.get("KARUTH_NEO4J_ENABLED", "1") == "1"

DEFAULT_TIMESTAMP_WINDOW_MINUTES = 15
FACE_SIMILARITY_THRESHOLD = 0.72
FUZZY_MATCH_THRESHOLD = 0.86

LOW_RISK = "LOW"
MED_RISK = "MED"
CRITICAL_RISK = "CRITICAL"
