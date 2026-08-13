"""Extension points for future forensic workflows.

These models are *contracts, not implementations*: they define how future
multimodal and external capabilities plug into the existing deterministic
pipeline and the dynamic action-selection loop. Nothing here is wired into
execution yet — the point is that adding a new evidence type or tool later
requires no planner rewrite.

Future capabilities anticipated by this module:

- image observations with OCR candidates and uncertainty
- human-approved external registry lookups (e.g. an MVD vehicle registry)
- approval-gated actions inside the dynamic loop
- candidate identity relationships (person A may be person B)
- sensitive-image redaction artifacts
- provenance-aware graph edges
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class MultimodalObservation(BaseModel):
    """An observation extracted from a non-text artifact (image, video, audio)."""

    observation_id: str = Field(min_length=1)
    artifact_ref: str = Field(min_length=1)  # file path inside the case folder
    modality: Literal["image", "video", "audio"] = "image"
    kind: str = Field(min_length=1)  # e.g. "ocr_text", "object", "face_candidate"
    candidates: list[dict] = Field(
        default_factory=list,
        description="each candidate carries value + uncertainty",
    )
    uncertainty: float = Field(ge=0.0, le=1.0, default=0.5)
    provenance: str = Field(default="", description="deterministic extractor name + params")


# Observation types supported by the multimodal observation layer. Each maps to
# a distinct evidence classification in the report.
OBSERVATION_TYPES = (
    "person",
    "vehicle",
    "number_plate",
    "gps_location",
    "timestamp",
    "camera_serial",
    "visible_object",
    "sensitive_content",
    "activity_interpretation",
)

EVIDENCE_CLASSIFICATIONS = (
    "Deterministic fact",
    "Metadata fact",
    "LLM visual interpretation",
    "Candidate association",
    "Human-confirmed relationship",
    "Contradiction",
    "Unresolved possibility",
)


class ImageObservation(BaseModel):
    """One structured observation from an image artifact.

    This is the multimodal counterpart of :class:`MultimodalObservation`: a
    typed, evidence-classified observation that may carry uncertain alternative
    readings (e.g. plate OCR) and always states whether human review is
    required before it may be relied on.
    """

    observation_id: str = Field(min_length=1)
    artifact_ref: str = Field(min_length=1)  # case-relative path (forward slashes)
    image_path: str = Field(min_length=1)  # absolute/on-disk path
    observation_type: str = Field(min_length=1)  # one of OBSERVATION_TYPES
    value: str = Field(default="", description="extracted value or description")
    description: str = Field(default="", description="human-readable explanation")
    # ``None`` means "Not scored" (missing metadata must never read as 0.0).
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    alternatives: list[str] = Field(
        default_factory=list, description="uncertain alternative readings (OCR)"
    )
    evidence_classification: str = "Metadata fact"
    model_used: str = Field(default="", description="extractor/model that produced this")
    requires_human_review: bool = False

    @property
    def requires_human_confirmation(self) -> bool:
        """Plate readings with alternatives (or low confidence) need confirmation."""
        return bool(self.alternatives) or (self.confidence is not None and self.confidence < 0.85)


class Candidate(BaseModel):
    """One uncertain value from a multimodal observation."""

    value: str
    confidence: float = Field(ge=0.0, le=1.0)
    source: str = ""


class ExternalLookupRequest(BaseModel):
    """A request for a human-approved external registry lookup."""

    lookup_kind: str = Field(min_length=1)  # e.g. "mvd_vehicle_owner"
    key: str = Field(min_length=1)  # e.g. a vehicle plate
    requester_note: str = ""
    requires_approval: bool = True


class ExternalLookupResult(BaseModel):
    """Result of an approved external lookup, with provenance."""

    lookup_kind: str
    key: str
    findings: list[dict] = Field(default_factory=list)
    source_system: str = Field(min_length=1)
    retrieved_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    approved_by: str = ""


@runtime_checkable
class ExternalLookupTool(Protocol):
    """Interface future registry-lookup tools implement."""

    def lookup(self, request: ExternalLookupRequest) -> ExternalLookupResult: ...

    @property
    def requires_approval(self) -> bool: ...


class ApprovalGatedAction(BaseModel):
    """An action the dynamic loop may propose but may not run without approval."""

    action_ref: str = Field(min_length=1)  # links to an ActionDecision
    description: str = Field(min_length=1)
    required_approval: Literal["none", "single", "dual"] = "single"
    approved_by: list[str] = Field(default_factory=list)
    status: Literal["pending", "approved", "denied"] = "pending"


class CandidateRelationship(BaseModel):
    """A candidate (not yet confirmed) identity or entity relationship."""

    relationship_id: str = Field(min_length=1)
    relation: str = Field(min_length=1)  # e.g. "same_person", "owns"
    source_refs: list[str] = Field(default_factory=list)
    target_refs: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    provenance: str = Field(min_length=1)


class SensitiveArtifact(BaseModel):
    """An artifact flagged for sensitive-content handling and redaction."""

    artifact_ref: str = Field(min_length=1)
    sensitivity: Literal["low", "medium", "high", "critical"] = "medium"
    redaction_status: Literal["none", "blurred", "redacted", "cleared"] = "none"
    redacted_by: str = ""
    reason: str = ""


class ProvenanceEdge(BaseModel):
    """A graph edge that always carries its evidence trail."""

    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    rel_type: str = Field(min_length=1)  # e.g. "MATCHES", "MENTIONS", "SENT_BY"
    evidence_refs: list[str] = Field(default_factory=list)
    created_by: str = Field(min_length=1)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    case_id: str = ""


# Registry so future planners can discover extension tooling through the same
# dynamic action-selection system (a tool spec per capability).
EXTENSION_TOOL_NAMES: dict[str, str] = {
    "multimodal_observation": "ObservationAgent",
    "external_lookup": "ExternalLookupAgent",
    "approval_gated_action": "ApprovalAgent",
    "candidate_relationship": "RelationshipAgent",
    "sensitive_artifact": "SensitiveReviewAgent",
    "provenance_edge": "GraphSyncAgent",
}
