"""Planner API routes: run an investigator prompt through the agentic planner.

``POST /planner/goal`` accepts a free-text prompt and returns the LLM-backed
plan and final result. The default backend is LM Studio (LLM planning) wrapped
in a deterministic fallback, so a prompt always produces a plan even when the
LLM server is unreachable. Guardrails, review authorization, and the hash-
chained ledger are enforced by the planner itself; this layer only translates
requests and responses.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from karuth.config import DEFAULT_DATA_DIR, PLANNER_MAX_ITERATIONS
from karuth.ledger import LedgerStore
from karuth.planner.llm import build_planner
from karuth.planner.state_machine import run_planner_goal

router = APIRouter(prefix="/planner", tags=["planner"])

VALID_BACKENDS = ("lmstudio", "deterministic")
DEFAULT_BACKEND = "lmstudio"


class PlannerGoalRequest(BaseModel):
    case_id: str = Field(
        min_length=1,
        description="Case identifier; becomes the case folder and ledger name.",
    )
    goal: str = Field(min_length=1, description="Free-text investigator goal / prompt.")
    backend: str = Field(
        default=DEFAULT_BACKEND,
        description="Planner backend: 'lmstudio' (default, LLM + fallback) or 'deterministic'.",
    )
    reveal_authorized: bool = Field(
        default=False,
        description="True when dual authorization for CRITICAL content was already granted.",
    )
    max_iterations: int | None = Field(
        default=None,
        description="Safety cap on reasoning-loop iterations (defaults to PLANNER_MAX_ITERATIONS).",
    )


class PlannerGoalResponse(BaseModel):
    case_id: str
    goal: str
    backend: str
    backend_used: str
    model_used: str
    final_result: str
    risk_tier: str
    confidence: float
    needs_human_review: bool
    needs_authorization: bool
    plan: dict
    completed_steps: list[str]
    execution_history: list[dict]
    planner_thoughts: list[str]
    notes: list[str]
    ledger_entries: int
    ledger_verified: bool


@router.get("/backends")
def list_backends() -> dict[str, object]:
    """List the available planning backends and which one is the default."""
    return {
        "backends": list(VALID_BACKENDS),
        "default": DEFAULT_BACKEND,
        "note": "LLM backends automatically fall back to the deterministic planner on failure.",
    }


@router.post("/goal", response_model=PlannerGoalResponse)
def planner_goal(request: PlannerGoalRequest) -> PlannerGoalResponse:
    """Run an investigator prompt through the agentic planner and return the result."""
    if request.backend not in VALID_BACKENDS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown backend '{request.backend}'. Choose from {VALID_BACKENDS}.",
        )
    if any(ch in request.case_id for ch in ("/", "\\", "..", ":")):
        raise HTTPException(
            status_code=400,
            detail="case_id must be a plain identifier (no path separators).",
        )

    case_folder = DEFAULT_DATA_DIR / request.case_id
    case_folder.mkdir(parents=True, exist_ok=True)
    ledger = LedgerStore(DEFAULT_DATA_DIR / f"{request.case_id}.sqlite3")

    planner = build_planner(request.backend)  # LLM backends get the deterministic fallback
    try:
        state = run_planner_goal(
            request.case_id,
            case_folder,
            request.goal,
            ledger,
            planner=planner,
            reveal_authorized=request.reveal_authorized,
            max_iterations=request.max_iterations or PLANNER_MAX_ITERATIONS,
        )
    except ValueError as exc:
        # Guardrail / policy rejection (external URL, out-of-scope path, unsafe plan).
        raise HTTPException(status_code=400, detail=f"Goal rejected: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - surface unexpected planner failures
        raise HTTPException(status_code=500, detail=f"Planner failed: {exc}") from exc

    plan = state.get("plan")
    return PlannerGoalResponse(
        case_id=request.case_id,
        goal=request.goal,
        backend=request.backend,
        backend_used=str(getattr(planner, "last_backend", None) or request.backend),
        model_used=str(state.get("model_used") or ""),
        final_result=str(state.get("final_result") or ""),
        risk_tier=str(state.get("risk_tier") or "LOW"),
        confidence=float(state.get("confidence") or 0.0),
        needs_human_review=bool(state.get("needs_human_review")),
        needs_authorization=bool(state.get("needs_authorization")),
        plan=plan.model_dump(mode="json") if plan is not None else {},
        completed_steps=list(state.get("completed_steps") or []),
        execution_history=list(state.get("execution_history") or []),
        planner_thoughts=list(state.get("planner_thoughts") or []),
        notes=list(state.get("notes") or []),
        ledger_entries=len(ledger.list_entries(request.case_id)),
        ledger_verified=ledger.verify(),
    )
