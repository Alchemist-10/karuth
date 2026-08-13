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

## Dynamic Hypothesis Loop (agentic mode)

The classic planner executes a fixed task list. The **dynamic mode** is the
hypothesis-driven loop: after *every* tool execution the planner decides its
single next action from the current investigation state — extract → correlate
→ timeline → form competing hypotheses → hunt contradictions → challenge with
a skeptic → finalize.

```text
plan (scaffold) -> decide -> execute -> observe -> decide -> ... -> finalize
```

The decision step picks one of:

- `run_tool` — execute a deterministic tool
- `investigate_hypothesis` — run a tool against a specific hypothesis
- `generate_hypothesis` — propose competing hypotheses from the evidence
- `challenge_hypothesis` — run the skeptic against a hypothesis
- `ask_human` — checkpoint: request investigator input (critical contradictions)
- `finalize` — stop and produce the evidence-linked report

Selection uses the goal, known facts, open hypotheses, evidence quality,
previous results, contradictions, available tools, expected information gain,
tool cost, and risk. Every decision is validated by the action guardrails
(known tool, in-scope arguments, existing hypothesis, budget) before it runs.

```python
state = run_planner_goal(
    case_id, case_folder, goal, ledger,
    planner=DeterministicPlanner(),  # or build_planner("openrouter") / "lmstudio"
    mode="dynamic",
)
```

`mode="plan"` (the default) keeps the classic fixed-task loop; existing callers
are unchanged. In dynamic mode the LLM backend also proposes each next action
(streaming, strict JSON); the deterministic policy is the automatic offline
fallback, so dynamic mode works fully air-gapped.

### Deterministic vs LLM responsibilities (unchanged guarantees)

| Concern | Owner |
|---|---|
| Evidence extraction, hashing, normalization, matching, timestamps, provenance, contradictions | **Deterministic modules only** |
| Hypothesis *status and confidence changes* | **Deterministic** (evidence-driven transitions) |
| Hypotheses *proposals*, action selection, interpretations, recommendations | LLM (optional; deterministic fallback exists) |
| Skeptic confidence adjustment | Deterministic |
| Final report facts | Deterministic claims with evidence references |

The LLM can never invent evidence, create correlation matches, mark a
hypothesis supported/rejected, or bypass guardrails. Every claim in the
evidence-linked report carries `evidence_refs` (entity/artifact ids);
unsupported narrative is labeled *interpretation*.

### Hypothesis lifecycle

`open → supported | rejected | inconclusive`, reopened by the skeptic.
Supporting deterministic evidence raises confidence; contradictions lower it
(high/medium/low severity adjust by 0.20/0.12/0.05). A hypothesis is supported
at confidence ≥ 0.75 with supporting evidence, rejected at ≤ 0.25 or under
strong contradiction, inconclusive when contradicted but not refuted.

### Contradictions and the skeptic

The contradiction hunter (`karuth/planner/contradictions.py`) is deterministic
and rule-based: impossible movement timelines (same plate in two distant
sightings within the window), inconsistent identities on a shared plate or
camera serial. Every contradiction cites the artifacts behind it.

Before finalizing, the skeptic tries to *disprove* the main conclusion: it
names the strongest alternative explanation, lowers confidence when
contradictions remain (never blindly confirming the lead), lists the evidence
still needed, and marks the case for human review. It runs when a hypothesis is
supported at high confidence, when contradictions are unresolved, or on
request.

### Example investigator prompt

> Find the person connected to vehicle KL07CX4242 and determine whether the
> chat activity supports that connection.

### Demo (the “wow moment”)

In the **Planner** page, press **“🧪 Run hypothesis demo (wow moment)”**. It
generates the synthetic case with a planted contradiction (a distant-city
sighting of the same plate minutes apart from the images), then runs dynamic
mode. Watch the agent:

1. extract entities from the chat + images;
2. form two competing hypotheses (which contact is connected to the vehicle);
3. choose correlation and timeline tools dynamically;
4. find the movement and identity contradictions;
5. launch the skeptic review and **lower** confidence instead of confirming;
6. stop with `unresolved_uncertainty` and tell you the exact evidence it still
   needs (independent verification of the conflicting sightings);
7. produce a report where every major claim is evidence-linked, with the
   investigation tree, hypotheses, contradictions, and confidence changes
   visible in the UI.

The experience is: *“Karuth investigated, challenged itself, and showed me
exactly what it still needs.”*

The pitch demo is now the **multimodal Jesse Pinkman case** (below), which
runs the same loop over images, EXIF, plates, and face candidates.

### Privacy and offline guarantees

Dynamic mode never sends raw evidence anywhere. The action-selection LLM
receives only the audit-safe investigation context (counts, hypothesis
statements, contradiction summaries). OpenRouter is for testing only; LM
Studio and deterministic mode are fully offline. All decisions, hypotheses,
observations, contradictions, and the final decision are appended to the
hash-chained ledger.

### Configuration knobs

`KARUTH_DYNAMIC_MAX_ACTIONS`, `KARUTH_MAX_HYPOTHESES`,
`KARUTH_SKEPTIC_CONFIDENCE_THRESHOLD`, `KARUTH_INFO_GAIN_THRESHOLD`,
`KARUTH_HUMAN_REVIEW_THRESHOLD` (see `.env.example`).

### Future forensic workflows (extension points)

`karuth/planner/extensions.py` defines the contracts for future capabilities
that plug into the same dynamic action-selection system without a planner
rewrite: multimodal observations (images/OCR candidates with uncertainty),
external registry lookups (approval-gated, e.g. an MVD vehicle registry),
approval-gated action types, candidate identity relationships, sensitive-image
redaction artifacts, and provenance-aware graph edges.

## Multimodal Jesse Demo (pitch demo)

The main demo is a fully synthetic investigation of a **fictional** subject
(“Jesse Pinkman” is a synthetic case label only — no real person, no real
data). The case contains three real JPEG images with embedded EXIF:

1. **Image 1 — suspicious activity.** A person handling a small transparent
   object near a table. The vision model may describe only what is visibly
   present (*“The image appears to show a person handling a small transparent
   object near a table”*) and must never conclude “the person is taking
drugs”. This is always classified **LLM visual interpretation — requires
investigator review**, and the image stays blurred until authorization.
2. **Image 2 — location bridge.** The same candidate subject near a
   recognizable landmark, linking person → place/time without being a
duplicate of image 3.
3. **Image 3 — person, vehicle, plate, EXIF.** The subject beside a vehicle
   with a synthetic plate (`NM-JP-4242`), plus embedded EXIF capture time, GPS
   coordinates, camera serial, and a synthetic identity fixture
   (`jesse_pinkman`) for repeatable face-cluster testing.

Generate it with:

```python
from scripts.generate_synthetic_case import generate_jesse_case, JESSE_GOAL
generate_jesse_case(case_folder)                    # clean preset
generate_jesse_case(case_folder, adversarial=True)  # wow-moment preset
```

Or in the Streamlit **Planner** page press **“🧪 Jesse demo (clean
evidence)”** or **“⚔️ Jesse demo (adversarial — wow moment)”**; both
regenerate the case and run dynamic mode with `JESSE_GOAL`.

### The two presets

- **Clean evidence:** images 1–3 as above. Expected result: *candidate
  subject-vehicle-location association found; human confirmation required.*
- **Adversarial evidence (the pitch “wow moment”):** adds a distant-city
  sighting of the same plate minutes apart (impossible movement), an
  embedded-vs-sidecar EXIF GPS conflict, an alternate plate OCR reading, and a
  second identity on the same plate/camera. Expected result: *candidate lead
  discovered → contradiction found → confidence reduced → skeptic challenge
  completed → final conclusion withheld → human review requested.*

### The investigation story

With the adversarial preset, dynamic mode:

1. extracts entities and **inspects multimodal observations** via the
   `ObservationAgent` (person, vehicle, plate with OCR alternatives, GPS,
   timestamp, camera serial, sensitive content, activity interpretation);
2. builds the timeline and **forms competing hypotheses** (same candidate
   subject in images 1–3; subject ↔ vehicle; image 3 places subject and
   vehicle near a location; image 1 may show substance-related activity;
   activity + location evidence refer to the same event);
3. **hunts contradictions** deterministically (impossible movement, GPS
   conflict, ambiguous plate, metadata mismatch, multiple identities on one
   plate/camera serial);
4. runs the **skeptic against a specific hypothesis** (the vehicle lead),
   which *lowers* confidence (e.g. `0.55 → 0.33`) and names the strongest
   alternative explanation;
5. **re-ranks the leads** after the skeptic changes confidence or evidence
   state;
6. hits a **human-confirmation checkpoint** (face association, plate reading,
   activity interpretation, sensitive reveal — all gated) and stops with
   `awaiting_human_confirmation`, `unresolved_contradictions`,
   `activity_interpretation_requires_review`, or `sensitive_content_gated` —
   never `sufficient_evidence` while any of those remain open;
7. writes the **evidence-linked report** with separate sections for confirmed
   observations, AI interpretations, contradictions/limitations, recommended
   next action, and a conclusion that explicitly does *not* establish
   identity, intent, ownership, or criminal activity.

### Evidence rules

Every claim is classified as one of: `Deterministic fact`, `Metadata fact`,
`LLM visual interpretation`, `Candidate association`, `Human-confirmed
relationship`, `Contradiction`, `Unresolved possibility`. The LLM may describe
visible content, transcribe candidate plate text, propose hypotheses, and
recommend actions — it must never invent a plate/location/timestamp/person,
treat face similarity or GPS as proof of identity or presence, infer intent
from appearance, or create confirmed graph edges. Graph edges carry a style by
provenance (solid green = deterministic fact, solid blue = metadata fact,
dotted amber = AI-proposed association, thick purple = human-confirmed, broken
red = contradiction, grey = weak/unresolved) and are never presented as
confirmed unless human-confirmed.

### EXIF handling

`karuth/ingestion/exif_extractor.py` parses **embedded JPEG EXIF** (Pillow) in
addition to the `.exif.json` sidecars, records metadata provenance
(`embedded JPEG EXIF` / `synthetic sidecar` / `unavailable`), reports a
metadata-extraction confidence, and surfaces conflicts when embedded and
sidecar metadata disagree. Missing metadata shows **“Location metadata:
unavailable”** — never a misleading `0.00`.

### Sensitive images

The UI preserves originals unchanged, renders **blurred display derivatives**
(hash + provenance recorded), shows a warning before any reveal, and requires
“Reveal authorized” (an authorization ledger entry) to show the original.
Blurred derivatives are display-only and are never treated as evidence.

### Using your own images and data (no demo required)

The demo generator is only a convenience — the planner is **general**. Point it
at any case folder containing your own evidence and it processes whatever is
there:

- drop real JPEG/PNG images into `images/` (embedded EXIF is parsed directly;
  an optional `<name>.exif.json` sidecar adds fields the photo lacks);
- add chat databases under `chats/`, documents under `documents/`, bank
  exports under `bank/`;
- run `streamlit run frontend/app.py` → Planner → set the case folder → run.

The pipeline extracts, inspects multimodal observations, correlates, builds
the timeline, forms hypotheses, hunts contradictions, and runs the skeptic on
*whatever evidence exists* — nothing is tied to the synthetic fixture.

**Important — what Karuth can and cannot read from a photo.** Karuth only
extracts *machine-readable* evidence: embedded EXIF (GPS, capture time,
camera make/model), metadata sidecars, and chat databases. It does **not**
guess plate text or people out of a photo's pixels unless an OCR engine is
installed (optional `pytesseract` + the Tesseract binary — see
`karuth/vision/ocr.py`). Screenshots and AI-generated images usually carry no
EXIF GPS/camera data, so a bare photo may yield zero observations. When that
happens the report shows an explicit **Evidence gap** section (and the UI
explains it): it names the plate from your goal as *only in the question, not
in the evidence*, and tells you how to proceed — install OCR, add a sidecar,
or use a vision model. Karuth never invents evidence from a photo. The
investigator can also name the plate directly in the goal; with the mock MVD
lookup approved, Karuth will fetch the registered owner for that plate even
when the photo contributed no machine-readable data.

### Approval-gated MVD registry lookup (mock)

`karuth/external/mvd.py` provides `MockMVDRegistry`, a clearly synthetic
stand-in for an external vehicle registry (RC owner lookup). It implements the
`ExternalLookupTool` interface, so a real registry client can replace it
without planner changes. It is **general**: it works for any plate — known
demo plates resolve to fictional fixtures, and any other plate resolves to a
deterministic, clearly-fake owner.

How it behaves:

- **Approval-gated.** Tick “Approve mock MVD registry lookup (registered
owner)” in the Planner sidebar. Only then will the dynamic loop schedule
`MvdRegistryLookup` (after hypotheses/contradictions/skeptic, before
finalizing). Without approval the tool refuses (soft-fail) and the flow is
unchanged.
- **Discovered through the same dynamic action-selection system.** The tool is
registered in `TOOL_REGISTRY`, described to the LLM backend, and chosen by the
deterministic policy whenever a vehicle plate became central to the
investigation.
- **Never overclaims.** The RC owner appears in the report as a *synthetic
metadata fact* — “registered owner is not proof of who operated the vehicle”
is stated alongside it. Approval and results are recorded in the audit ledger
(`registry_lookup_approved`, `registry_lookup_result`).

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

Open the **Planner** page, enter an investigator prompt, and run it. The page generates the case folder, runs the agentic planner in-process, and presents the investigation the way an investigator would read it: investigation status, primary lead, evidence chain, confidence summary, timeline, hypotheses, contradictions, skeptic review, sensitive-image controls, and recommended next action — with raw JSON and ledger rows collapsed behind expanders. One-click **Jesse demo** buttons (clean and adversarial) regenerate the multimodal demo case and run it in dynamic mode. Open the **Graph** page to explore the case's Neo4j graph interactively.

The Graph page renders two force-directed views (dark, neon-accented, hover-to-highlight, click-to-inspect): a **whole-case** graph of every entity and typed edge (sized by connectivity, colored by entity type) and an **entity search** view that centers a searched value (plate, phone, serial...) with the sources that mention it and the entities it correlates with. The renderer is Cytoscape.js, vendored under `frontend/static/` so the UI stays fully offline/air-gapped.
