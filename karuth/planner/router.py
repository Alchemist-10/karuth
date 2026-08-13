"""Route abstract planner tasks to concrete deterministic modules.

The planner decides *which* tools to run; this router binds each abstract task
to the existing deterministic evidence-processing modules. No LLM reasoning
happens here.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from karuth.agents.action_agent import fill_legal_document
from karuth.agents.correlation_agent import correlate_entities
from karuth.agents.extraction_agent import extract_case_entities
from karuth.agents.report_agent import summarize_matches
from karuth.agents.timeline_agent import order_chronologically
from karuth.agents.triage_agent import rank_leads
from karuth.agents.vision_agent import transcribe_images
from karuth.config import SUPPORTED_IMAGE_SUFFIXES
from karuth.planner.contradictions import Contradiction, hunt_contradictions
from karuth.planner.schema import PlannerTask
from karuth.planner.skeptic import run_case_skeptic
from karuth.planner.state import PlannerState
from karuth.planner.tool_result import Claim, ToolResult
from karuth.planner.tools import TOOL_REGISTRY

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "legal" / "templates"

# Cap how many images the LLM transcribes in one dispatch so a large image
# folder cannot blow the token budget.
MAX_VISION_IMAGES_PER_DISPATCH = 5

_PHONE_RE = re.compile(r"\+?\d{10,15}")
_FACE_RE = re.compile(r"Person_Cluster_[A-Z0-9_]+")

# Compact (KL01CC2121 / KL 01 CC 2121) and hyphenated (NM-JP-4242) plates.
_PLATE_IN_TEXT_RE = re.compile(
    r"\b[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,3}\s?\d{3,4}\b"
    r"|\b[A-Z]{2}[- ][A-Z]{2}[- ]?\d{3,4}\b",
    re.IGNORECASE,
)


def _normalize_plate(value: str) -> str:
    """Uppercase and strip whitespace (keep hyphens)."""
    return re.sub(r"\s+", "", str(value or "")).upper()


def route_task(task: PlannerTask, state: PlannerState) -> tuple[dict[str, object], str]:
    """Execute one planner task and return (state updates, human summary)."""
    tool = task.tool
    case_folder = state["case_folder"]

    if tool == "ExtractionAgent":
        entities = extract_case_entities(case_folder)
        return {"entities": entities}, f"extracted {len(entities)} entities"

    if tool == "CorrelationAgent":
        matches = correlate_entities(
            list(state.get("entities", [])),
            case_id=state.get("case_id"),
        )
        # Best-effort: convey unstructured case content (chat messages,
        # document prose, bank memos) into the Neo4j graph as typed edges over
        # the synced entities. A down graph never changes the matches.
        try:
            from karuth.graph.graph_ingest import ingest_unstructured

            graph_stats = ingest_unstructured(
                state.get("case_id", ""),
                Path(case_folder),
                list(state.get("entities", [])),
            )
        except Exception:  # noqa: BLE001 - graph is optional enrichment
            graph_stats = {}
        detail = f"found {len(matches)} matches"
        if graph_stats.get("synced"):
            detail += (
                f" (graph: {graph_stats.get('sent_by', 0)} sent_by, "
                f"{graph_stats.get('mentions', 0)} mentions)"
            )
        return {"matches": matches}, detail

    if tool == "ObservationAgent":
        from karuth.planner.observations import build_image_observations

        observations = build_image_observations(Path(case_folder))
        return (
            {"image_observations": observations},
            f"recorded {len(observations)} multimodal observations",
        )

    if tool == "TimelineAgent":
        timeline = order_chronologically(list(state.get("entities", [])))
        return {"timeline": timeline}, f"ordered {len(timeline)} entities"

    if tool == "TriageAgent":
        ranked = rank_leads(list(state.get("matches", [])), state.get("goal", ""))
        return {"ranked_matches": ranked}, f"ranked {len(ranked)} leads"

    if tool == "ReportAgent":
        report = summarize_matches(list(state.get("matches", [])))
        return {"report": report}, "generated case summary"

    if tool == "VisionAgent":
        if task.inputs:
            image_paths = [Path(path) for path in task.inputs[:MAX_VISION_IMAGES_PER_DISPATCH]]
        else:
            image_paths = sorted(
                path
                for suffix in SUPPORTED_IMAGE_SUFFIXES
                for path in Path(case_folder).rglob(f"*{suffix}")
            )[:MAX_VISION_IMAGES_PER_DISPATCH]
        transcriptions = transcribe_images(image_paths)
        failed = sum(1 for item in transcriptions if item.error)
        if not transcriptions:
            return {"image_transcriptions": []}, "no images found to transcribe"
        detail = f"transcribed {len(transcriptions)} images"
        if failed:
            detail += f" ({failed} failed: LLM vision backend unavailable?)"
        return {"image_transcriptions": transcriptions}, detail

    if tool == "LegalActionAgent":
        template_name = task.inputs[0] if task.inputs else ""
        document = fill_legal_document(TEMPLATE_DIR, template_name, task.fields)
        return {"legal_output": document}, f"rendered {template_name}"

    if tool == "HypothesisAgent":
        # Lazy import: this agent depends on the planner package, which can
        # still be mid-initialization when the router module loads.
        from karuth.agents.hypothesis_agent import generate_hypotheses, propose_hypothesis

        entities = list(state.get("entities", []))
        matches = list(state.get("matches", []))
        observations = list(state.get("image_observations", []))
        hypotheses = generate_hypotheses(entities, matches, state.get("goal", ""), observations)
        # An LLM-proposed hypothesis may ride along in fields, but only when
        # its evidence references resolve to real entities.
        proposal = task.fields.get("proposed_hypothesis")
        if isinstance(proposal, dict):
            adopted = propose_hypothesis(
                str(proposal.get("statement", "")),
                [str(ref) for ref in proposal.get("evidence_refs", [])],
                entities,
            )
            if adopted is not None and adopted.statement not in {h.statement for h in hypotheses}:
                hypotheses.append(adopted)
        return {"hypotheses": hypotheses}, f"generated {len(hypotheses)} hypotheses"

    if tool == "ContradictionHunter":
        findings = hunt_contradictions(
            list(state.get("entities", [])),
            list(state.get("matches", [])),
            observations=list(state.get("image_observations", [])),
        )
        # JSON-safe dicts: the dynamic loop, ledger, and UI all read dicts.
        return {
            "contradictions": [finding.to_dict() for finding in findings]
        }, f"found {len(findings)} contradictions"

    if tool == "MvdRegistryLookup":
        from karuth.external.mvd import MockMVDRegistry
        from karuth.planner.extensions import ExternalLookupRequest

        if not state.get("registry_lookup_approved", False):
            raise ValueError(
                "MvdRegistryLookup requires registry_lookup_approved=True "
                "(investigator approval of the external registry lookup)"
            )
        plate = _registry_plate_for(task, state)
        result = MockMVDRegistry().lookup(
            ExternalLookupRequest(
                lookup_kind="mvd_vehicle_owner",
                key=plate,
                requester_note="investigator",
            )
        )
        return (
            {"registry_results": [result.model_dump(mode="json")]},
            f"mock MVD registry: RC record for {plate} retrieved (synthetic fixture)",
        )

    if tool == "SkepticReview":
        hypotheses = list(state.get("hypotheses", []))
        contradictions = list(state.get("contradictions", []))
        target = task.inputs[0] if task.inputs else ""
        target_h = next((h for h in hypotheses if h.id == target), None)
        if target_h is None:
            candidates = [h for h in hypotheses if h.status in ("open", "supported", "inconclusive")]
            target_h = max(candidates, key=lambda h: h.confidence, default=None)
        conclusion = target_h.statement if target_h else "case conclusions"
        original = target_h.confidence if target_h else float(state.get("confidence", 0.0) or 0.0)
        review = run_case_skeptic(
            conclusion=conclusion,
            original_confidence=original,
            hypotheses=hypotheses,
            contradictions=contradictions,
            risk_tier=state.get("risk_tier", "LOW"),
            hypothesis_id=target_h.id if target_h else "",
        )
        return (
            {"skeptic_review": review.to_dict()},
            f"skeptic review: confidence {review.original_confidence:.2f} -> {review.revised_confidence:.2f}",
        )

    raise ValueError(f"Unsupported planner tool: {tool}")


# ---------------------------------------------------------------------------
# structured tool execution (dynamic loop)
# ---------------------------------------------------------------------------


def _registry_plate_for(task: PlannerTask, state: PlannerState) -> str:
    """Pick the plate to look up: task input, entity, plate observation, goal.

    The goal fallback accepts compact Indian-format plates (``KL01CC2121``,
    ``KL 01 CC 2121``, lowercase ``kl01cc2121``) as well as the hyphenated
    demo format (``NM-JP-4242``), so an approved lookup works even when the
    photo had no machine-readable plate text.
    """
    if task.inputs and task.inputs[0].strip():
        return _normalize_plate(task.inputs[0])
    for entity in list(state.get("entities", [])):
        if entity.type == "vehicle_plate":
            return _normalize_plate(entity.value)
    for observation in list(state.get("image_observations", [])):
        data = observation if isinstance(observation, dict) else observation.model_dump(mode="json")
        if data.get("observation_type") == "number_plate" and data.get("value"):
            return _normalize_plate(data["value"])
    goal_match = _PLATE_IN_TEXT_RE.search(str(state.get("goal", "")))
    if goal_match:
        return _normalize_plate(goal_match.group(0))
    raise ValueError("MvdRegistryLookup has no plate to look up")


def _follow_ups(tool: str, updates: dict) -> list[str]:
    """Deterministic suggested next steps per tool."""
    if tool == "ExtractionAgent":
        return ["observation_review", "correlation", "timeline_analysis"]
    if tool == "ObservationAgent":
        return ["correlation", "timeline_analysis"]
    if tool == "CorrelationAgent":
        return ["contradiction_hunting", "timeline_analysis"]
    if tool == "TimelineAgent":
        return ["hypothesis_generation", "contradiction_hunting"]
    if tool == "HypothesisAgent":
        return ["contradiction_hunting", "challenge_review"]
    if tool == "ContradictionHunter":
        return ["skeptic_review"]
    if tool == "SkepticReview":
        return ["re_rank_leads", "finalize_report"]
    if tool == "TriageAgent":
        return ["report_generation"]
    if tool == "MvdRegistryLookup":
        return ["finalize_report"]
    return []


def _claims_for(tool: str, updates: dict, state: PlannerState) -> list[Claim]:
    """Derive evidence-linked claims from deterministic tool state only.

    Claims reference real entity/artifact ids; the LLM is never the source of
    a claim here.
    """
    if tool == "ExtractionAgent":
        entities = list(updates.get("entities", []))
        files = len({entity.source_file for entity in entities})
        return [
            Claim(
                claim=f"Extracted {len(entities)} entities from {files} source file(s)",
                evidence_refs=[entity.entity_id for entity in entities[:50]],
                confidence=0.98,
            )
        ]
    if tool == "CorrelationAgent":
        from karuth.planner.evidence_report import aggregate_match_claims

        claims, _ = aggregate_match_claims(
            list(updates.get("matches", [])),
            list(state.get("entities", [])),
        )
        return claims
    if tool == "ObservationAgent":
        from karuth.planner.observations import normalize_observations

        observations = normalize_observations(updates.get("image_observations", []))
        return [
            Claim(
                claim=(
                    f"{o.observation_type.replace('_', ' ')}: {o.value} "
                    f"({o.artifact_ref}) — {o.evidence_classification}"
                ),
                evidence_refs=[o.observation_id, o.artifact_ref],
                confidence=o.confidence if o.confidence is not None else 0.5,
                claim_type="deterministic_fact"
                if o.evidence_classification in ("Deterministic fact", "Metadata fact")
                else "interpretation",
            )
            for o in observations
        ]
    if tool == "TimelineAgent":
        timeline = list(updates.get("timeline", []))
        return [
            Claim(
                claim=f"Timeline built: {len(timeline)} events ordered chronologically",
                evidence_refs=[entity.entity_id for entity in timeline[:50]],
                confidence=0.95,
            )
        ]
    if tool == "TriageAgent":
        ranked = list(updates.get("ranked_matches", []))
        refs: list[str] = []
        for match in ranked[:50]:
            refs.extend([match.left_entity_id, match.right_entity_id])
        return [
            Claim(
                claim=f"Ranked {len(ranked)} leads by score and goal relevance",
                evidence_refs=list(dict.fromkeys(refs)),
                confidence=0.9,
            )
        ]
    if tool == "ReportAgent":
        return [Claim(claim="Generated deterministic case summary", confidence=0.9)]
    if tool == "HypothesisAgent":
        return [
            Claim(
                claim=f"Hypothesis {h.id} ({h.status}): {h.statement}",
                evidence_refs=list(h.related_entity_ids),
                confidence=h.confidence,
                claim_type="interpretation",
            )
            for h in updates.get("hypotheses", [])
        ]
    if tool == "ContradictionHunter":
        return [
            Claim(
                claim=f"Contradiction {c.get('id')} ({c.get('rule')}): {c.get('claim_a')} vs {c.get('claim_b')}",
                evidence_refs=list(c.get("evidence_refs", [])),
                confidence=c.get("confidence", 0.8),
                claim_type="contradiction",
            )
            for c in updates.get("contradictions", [])
        ]
    if tool == "SkepticReview":
        review = updates.get("skeptic_review")
        if not review:
            return []
        return [
            Claim(
                claim=f"Skeptic review: confidence {review.get('original_confidence', 0):.2f} -> "
                f"{review.get('revised_confidence', 0):.2f}",
                evidence_refs=list(review.get("weakening_evidence", [])),
                confidence=round(1.0 - float(review.get("revised_confidence", 0.5)), 3),
                claim_type="interpretation",
            )
        ]
    if tool == "VisionAgent":
        return [
            Claim(
                claim=f"LLM vision transcription for {item.image_path} (assistance only, not evidence)",
                claim_type="interpretation",
                confidence=0.5,
            )
            for item in updates.get("image_transcriptions", [])
        ]
    if tool == "MvdRegistryLookup":
        plate_refs = {
            e.value.upper(): e.entity_id
            for e in list(state.get("entities", []))
            if e.type == "vehicle_plate"
        }
        claims: list[Claim] = []
        for result in list(updates.get("registry_results", [])):
            plate = str(result.get("key", "")).upper()
            ref = plate_refs.get(plate, plate)
            for finding in result.get("findings", []):
                claims.append(
                    Claim(
                        claim=(
                            f"{finding.get('field', 'field').replace('_', ' ')}: "
                            f"{finding.get('value', '')} "
                            f"(mock MVD registry - synthetic fixture; registered "
                            f"owner is not proof of who operated the vehicle)"
                        ),
                        evidence_refs=[ref],
                        confidence=float(finding.get("confidence", 0.5)),
                        claim_type="deterministic_fact",
                    )
                )
        return claims
    return []


def run_tool(task: PlannerTask, state: PlannerState, *, iteration: int = 0) -> ToolResult:
    """Execute one planner task and return a structured :class:`ToolResult`.

    Wraps the existing deterministic :func:`route_task` so every tool keeps
    working while the dynamic loop receives claims, evidence references,
    limitations, and follow-ups. Free-form LLM text never becomes a claim:
    claims are derived only from deterministic tool state.
    """
    start = time.perf_counter()
    try:
        updates, summary = route_task(task, state)
        success, error = True, ""
    except Exception as exc:  # noqa: BLE001 - structured failures, not crashes
        success, error = False, str(exc)
        updates, summary = {}, f"failed: {exc}"
    duration_ms = int((time.perf_counter() - start) * 1000)

    claims = _claims_for(task.tool, updates, state) if success else []
    refs = [ref for claim in claims for ref in claim.evidence_refs]
    spec = TOOL_REGISTRY.get(task.tool)
    contradictions = [
        item.to_dict() if isinstance(item, Contradiction) else item
        for item in updates.get("contradictions", [])
    ]
    return ToolResult(
        success=success,
        tool=task.tool,
        state_updates=updates,
        summary=summary,
        claims=claims,
        evidence_refs=list(dict.fromkeys(refs)),
        confidence=max((claim.confidence for claim in claims), default=0.0),
        limitations=list(spec.limitations) if spec else [],
        suggested_follow_ups=_follow_ups(task.tool, updates),
        contradictions=contradictions,
        duration_ms=duration_ms,
        error=error,
        iteration=iteration,
    )


def mask_report(report: str, risk_level: str) -> str:
    """Mask sensitive values for MED/CRITICAL default-blurred previews."""
    if risk_level == "LOW":
        return report
    masked = _PHONE_RE.sub("[REDACTED PHONE]", report)
    masked = _FACE_RE.sub("[REDACTED FACE CLUSTER]", masked)
    return masked
