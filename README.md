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
