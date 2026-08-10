"""Tool metadata registry for the agentic planner.

Every deterministic tool the planner can schedule is described here: what it
does, what it needs, what it produces, which evidence files it can read, its
cost, whether it works offline, and its review posture. The LLM planning
backend reasons over these specs instead of static keyword mappings, and the
graph's task-selection step uses ``required_inputs`` / produced outputs to
respect dependencies and skip tasks that could never produce useful output.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ToolName = Literal[
    "ExtractionAgent",
    "CorrelationAgent",
    "TimelineAgent",
    "TriageAgent",
    "ReportAgent",
    "LegalActionAgent",
]

CostLevel = Literal["low", "medium", "high"]


class ToolOutput(BaseModel):
    """One state key a tool writes, with a human-readable label."""

    key: str
    label: str
    description: str = ""


class ToolSpec(BaseModel):
    name: ToolName
    description: str = Field(min_length=1)
    capabilities: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    required_inputs: list[str] = Field(default_factory=list)
    produced_outputs: list[ToolOutput] = Field(default_factory=list)
    supported_evidence: list[str] = Field(default_factory=list)
    cost: CostLevel = "low"
    offline: bool = True
    risk: str = "LOW"
    review_required: bool = False

    @property
    def produced_keys(self) -> list[str]:
        return [output.key for output in self.produced_outputs]


TOOL_REGISTRY: dict[str, ToolSpec] = {
    "ExtractionAgent": ToolSpec(
        name="ExtractionAgent",
        description=(
            "Scans the case folder and deterministically extracts entities from "
            "supported evidence files (SQLite chat databases and JPEG images with "
            "EXIF/synthetic face metadata). This is the discovery tool: it must run "
            "before correlation, timeline, or triage on cases with raw evidence."
        ),
        capabilities=[
            "extract phone, username, person_name, message, vehicle_plate, "
            "camera_serial, location, and face_embedding entities",
            "carve SQLite chat databases",
            "read JPEG EXIF metadata and synthetic face embeddings",
        ],
        limitations=[
            "does not read videos or audio files",
            "does not interpret document prose beyond tagged metadata",
            "cannot fabricate entities: only what exists in the files",
        ],
        required_inputs=[],
        produced_outputs=[
            ToolOutput(key="entities", label="extracted entities", description="canonical entity list")
        ],
        supported_evidence=["chats", "images"],
        cost="medium",
        offline=True,
        risk="LOW",
        review_required=False,
    ),
    "CorrelationAgent": ToolSpec(
        name="CorrelationAgent",
        description=(
            "Deterministically correlates extracted entities across sources "
            "(exact string, fuzzy string, timestamp proximity, face-embedding "
            "cosine) into match results. Never invents relationships."
        ),
        capabilities=[
            "exact cross-source matches on phone/username/camera_serial/vehicle_plate",
            "fuzzy name/plate matching",
            "timestamp-proximity linking",
            "face-embedding cosine clustering",
        ],
        limitations=[
            "requires extracted entities",
            "produces no matches when overlapping signals are absent",
        ],
        required_inputs=["entities"],
        produced_outputs=[
            ToolOutput(key="matches", label="correlation matches", description="deterministic match results")
        ],
        supported_evidence=[],
        cost="medium",
        offline=True,
        risk="LOW",
        review_required=True,
    ),
    "TimelineAgent": ToolSpec(
        name="TimelineAgent",
        description=(
            "Orders extracted entities chronologically by their timestamps "
            "(sent_at, captured_at, or extraction time) into a case timeline."
        ),
        capabilities=["chronological ordering of entities", "sequence and timing analysis"],
        limitations=["requires extracted entities with usable timestamps"],
        required_inputs=["entities"],
        produced_outputs=[
            ToolOutput(key="timeline", label="chronological timeline", description="entities ordered by time")
        ],
        supported_evidence=[],
        cost="low",
        offline=True,
        risk="LOW",
        review_required=True,
    ),
    "TriageAgent": ToolSpec(
        name="TriageAgent",
        description=(
            "Ranks correlation matches by score and goal-keyword relevance to "
            "surface the top leads for the investigator."
        ),
        capabilities=["score-based lead ranking", "goal-relevance re-ranking"],
        limitations=["requires correlation matches", "ranking is deterministic"],
        required_inputs=["matches"],
        produced_outputs=[
            ToolOutput(key="ranked_matches", label="ranked leads", description="matches sorted by score/relevance")
        ],
        supported_evidence=[],
        cost="low",
        offline=True,
        risk="LOW",
        review_required=True,
    ),
    "ReportAgent": ToolSpec(
        name="ReportAgent",
        description=(
            "Produces a deterministic investigator-readable case summary from the "
            "correlation matches (or an explicit no-findings summary when there "
            "are none)."
        ),
        capabilities=["case summary generation", "findings narration"],
        limitations=["summarizes only deterministic results; never fabricates findings"],
        required_inputs=[],
        produced_outputs=[
            ToolOutput(key="report", label="case summary", description="markdown case report")
        ],
        supported_evidence=[],
        cost="low",
        offline=True,
        risk="LOW",
        review_required=True,
    ),
    "LegalActionAgent": ToolSpec(
        name="LegalActionAgent",
        description=(
            "Renders a legal template document (Section 63 BSA certificate or "
            "Section 94 BNSS notice) from the plan-provided fields. Requires the "
            "plan to supply the template name and field values."
        ),
        capabilities=["render section_63_bsa_certificate", "render section_94_bnss_notice"],
        limitations=["cannot draft novel legal text; fills the selected template only"],
        required_inputs=[],
        produced_outputs=[
            ToolOutput(key="legal_output", label="rendered legal document", description="template-filled document")
        ],
        supported_evidence=[],
        cost="high",
        offline=True,
        risk="LOW",
        review_required=False,
    ),
}

# State key -> tool that produces it. Used to detect unsatisfiable dependencies
# during task selection (a dependent task is skipped when its producer can
# never run, so the planner never wastes a dispatch).
PRODUCER_BY_OUTPUT: dict[str, str] = {
    output.key: name
    for name, spec in TOOL_REGISTRY.items()
    for output in spec.produced_outputs
}


def tool_spec(name: str) -> ToolSpec:
    """Look up a tool spec, raising a clear error for unknown tools."""
    try:
        return TOOL_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown planner tool: {name}") from exc


def tool_metadata_text() -> str:
    """Render the registry as a compact text block for the LLM system prompt."""
    lines: list[str] = []
    for name, spec in TOOL_REGISTRY.items():
        lines.append(f"- {name}: {spec.description}")
        lines.append(f"    capabilities: {', '.join(spec.capabilities)}")
        lines.append(f"    limitations: {', '.join(spec.limitations) or 'none'}")
        lines.append(f"    requires_state_inputs: {', '.join(spec.required_inputs) or 'none'}")
        lines.append(f"    produces: {', '.join(output.label for output in spec.produced_outputs)}")
        lines.append(
            f"    reads_evidence_kinds: {', '.join(spec.supported_evidence) or 'none (operates on extracted state)'}"
        )
        lines.append(f"    deterministic_and_offline: {spec.offline}")
        lines.append(f"    output_requires_review: {spec.review_required}")
        lines.append(f"    cost: {spec.cost}")
    return "\n".join(lines)
