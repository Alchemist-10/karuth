# Karuth

Karuth is an offline-first, agentic AI platform skeleton for digital forensics workflows. It is designed around a single tamper-evident audit ledger shared by evidence provenance and agent reasoning provenance.

This repository currently implements Phase 1:

- Full project scaffold.
- SQLite-backed hash-chained ledger.
- Synthetic case generator for local testing and demos.

## Safety and Data Rules

- Development and tests use synthetic evidence only.
- The platform is intended to run air-gapped. Network access is limited to local services such as LM Studio, Neo4j, and ChromaDB.
- Legal documents are generated from templates and field mappings only.
- Correlation matching must be deterministic. LLMs may narrate already-found matches but never create matches.

## Quick Start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
```

Generate a synthetic case folder:

```powershell
python scripts/generate_synthetic_case.py --output tests/fixtures/synthetic_case/generated
```

## Local Services

`docker-compose.yml` defines local-only Neo4j, ChromaDB, and LM Studio services for later phases. Phase 1 tests do not require Docker or network access.

## Agentic Planner

The planner (`karuth/planner/`) turns an investigator goal into a reasoning loop that schedules the existing deterministic agents (extraction, correlation, timeline, triage, report, legal, vision) one task at a time. The LLM is used only for goal interpretation, tool selection, ordering, replanning, and stopping decisions — it never touches evidence, matching, or guardrails.

Planning backends are swappable through `build_planner(backend, **kwargs)`:

- `"openrouter"` (default): LLM-backed planning against OpenRouter (`https://openrouter.ai/api/v1`), an OpenAI-compatible REST endpoint whose models run on remote GPUs — fast, with no local model execution. **It sends the case state summary to a third-party service, so it is for testing only, never real casework.** Set the `OPENROUTER_API_KEY` environment variable; model/server/limits come from `karuth/config.py` (`KARUTH_OPENROUTER_*`) or constructor kwargs. The default model is the free tier `nvidia/nemotron-3-super-120b-a12b:free`.
- `"lmstudio"`: fully offline LLM-backed planning against a local LM Studio server at `http://localhost:1234/v1` (OpenAI-compatible REST endpoint) using the OpenAI Python SDK. Configure `base_url`, `model`, `temperature`, `max_tokens`, and `timeout` via `karuth/config.py` or constructor kwargs — no planner code changes needed. Use this when evidence must never leave the machine.
- `"deterministic"`: offline keyword-based planner, always available.

LLM backends are wrapped in `FallbackPlanner` by default, so a timeout, unavailable model, or invalid JSON response automatically falls back to the deterministic planner and is recorded in the ledger. `run_planner_goal` and `build_planner()` therefore give you the LLM-backed experience by default, degrading gracefully when no LLM server is running.

LLM backends stream completions and capture the model's reasoning tokens (`last_thoughts`, forwarded to a `thoughts_callback` when configured), so the planner page can render the thinking live, xAI-style. Reasoning is **display-only** — it is never written to the ledger, which keeps only concise, audit-safe explanations.

```python
from karuth.planner.llm import build_planner

planner = build_planner()                       # OpenRouter default (set OPENROUTER_API_KEY)
planner = build_planner("lmstudio")             # fully offline LLM backend
planner = build_planner("deterministic")        # offline keyword backend
```

## Configuration

All settings are read from environment variables and a git-ignored `.env` file is loaded automatically (real environment variables win over `.env`). Copy `.env.example` to `.env` and fill in what you need:

- `OPENROUTER_API_KEY` — **required for the default OpenRouter backend** (get one at https://openrouter.ai/keys). Without it, planning falls back to the deterministic backend.
- `KARUTH_OPENROUTER_*` — base URL, model, temperature, max tokens, timeout for the OpenRouter backend (default model is the free tier `nvidia/nemotron-3-super-120b-a12b:free`).
- `KARUTH_LMSTUDIO_*` — the same knobs for the fully offline LM Studio backend (`http://localhost:1234/v1` by default).
- `KARUTH_VISION_MODEL` — LM Studio vision model used for image transcription (empty = auto-discover).
- `KARUTH_NEO4J_URI` / `KARUTH_NEO4J_USER` / `KARUTH_NEO4J_PASSWORD` / `KARUTH_NEO4J_ENABLED` — Neo4j graph enrichment (best-effort; the pipeline runs without it).
- `KARUTH_DATA_DIR` / `KARUTH_LEDGER_DB` — where case folders and the ledger database live.
- `KARUTH_MODEL_DIR` — cache for the face-recognition ONNX models (downloaded once by `scripts/fetch_face_models.py`).

## User entry points

### FastAPI

```powershell
uvicorn karuth.api.main:app --reload
```

```bash
curl -X POST http://localhost:8000/planner/goal \
  -H "Content-Type: application/json" \
  -d '{"case_id": "case-demo", "goal": "Find links for vehicle KL07CX4242 and triage the top leads"}'
```

`GET /planner/backends` lists the available planning backends. The `backend` field (`openrouter` default, `lmstudio`, `deterministic`) selects the planner; LLM backends fall back to the deterministic planner automatically. Guardrail rejections return `400`.

### Streamlit UI

```powershell
streamlit run frontend/app.py
```

Open the **Planner** page, enter an investigator prompt, and run it. The page generates the case folder, runs the agentic planner in-process, and shows the plan, execution history, result, and the hash-chained ledger. Open the **Graph** page to explore the case's Neo4j graph interactively.

The Graph page renders two force-directed views (dark, neon-accented, hover-to-highlight, click-to-inspect): a **whole-case** graph of every entity and typed edge (sized by connectivity, colored by entity type) and an **entity search** view that centers a searched value (plate, phone, serial...) with the sources that mention it and the entities it correlates with. The renderer is Cytoscape.js, vendored under `frontend/static/` so the UI stays fully offline/air-gapped.
