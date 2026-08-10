# Karuth

Karuth is an offline-first, agentic AI platform skeleton for digital forensics workflows. It is designed around a single tamper-evident audit ledger shared by evidence provenance and agent reasoning provenance.

This repository currently implements Phase 1:

- Full project scaffold.
- SQLite-backed hash-chained ledger.
- Synthetic case generator for local testing and demos.

## Safety and Data Rules

- Development and tests use synthetic evidence only.
- The platform is intended to run air-gapped. Network access is limited to local services such as Ollama, Neo4j, and ChromaDB.
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

`docker-compose.yml` defines local-only Neo4j, ChromaDB, and Ollama services for later phases. Phase 1 tests do not require Docker or network access.

## Agentic Planner

The planner (`karuth/planner/`) turns an investigator goal into a reasoning loop that schedules the existing deterministic agents (extraction, correlation, timeline, triage, report, legal) one task at a time. The LLM is used only for goal interpretation, tool selection, ordering, replanning, and stopping decisions — it never touches evidence, matching, or guardrails.

Planning backends are swappable through `build_planner(backend, **kwargs)`:

- `"lmstudio"` (default): LLM-backed planning against a local LM Studio server at `http://localhost:1234/v1` (OpenAI-compatible REST endpoint) using the OpenAI Python SDK. Configure `base_url`, `model`, `temperature`, `max_tokens`, and `timeout` via `karuth/config.py` or constructor kwargs — no planner code changes needed.
- `"deterministic"`: offline keyword-based planner, always available.

LLM backends are wrapped in `FallbackPlanner` by default, so a timeout, unavailable model, or invalid JSON response automatically falls back to the deterministic planner and is recorded in the ledger. `run_planner_goal` and `build_planner()` therefore give you the LLM-backed experience by default, degrading gracefully when no LLM server is running.

```python
from karuth.planner.llm import build_planner

planner = build_planner("lmstudio", model="your-model", base_url="http://localhost:1234/v1")
```

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

`GET /planner/backends` lists the available planning backends. The `backend` field (`lmstudio` default, `deterministic`) selects the planner; the LLM backend falls back to the deterministic planner automatically. Guardrail rejections return `400`.

### Streamlit UI

```powershell
streamlit run frontend/app.py
```

Open the **Planner** page, enter an investigator prompt, and run it. The page generates the case folder, runs the agentic planner in-process, and shows the plan, execution history, result, and the hash-chained ledger.
