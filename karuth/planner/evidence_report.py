"""Evidence-linked report generation (investigator-first format).

The final report separates *confirmed observations* (deterministic facts and
metadata facts) from *AI interpretations* (hypotheses, LLM activity readings,
skeptic alternatives), lists contradictions and limitations explicitly, and
ends with a careful conclusion that never overclaims identity, intent,
ownership, or criminal activity. Matches are aggregated by (rule, value) so a
value seen in N sources becomes one claim with the full source list, and every
important claim carries ``refs:`` (evidence references).
"""

from __future__ import annotations

import re
from collections import defaultdict
from statistics import mean

from karuth.correlation.candidate_matcher import CorrelationResult
from karuth.entities.schema import Entity
from karuth.planner.observations import normalize_observations
from karuth.planner.tool_result import Claim

_PLATE_VALUE_RE = re.compile(r"match on (.+)$")

# Stop reasons that mean the investigation cannot conclude without human input.
HUMAN_REVIEW_STOP_REASONS = {
    "unresolved_contradictions",
    "awaiting_human_confirmation",
    "activity_interpretation_requires_review",
    "sensitive_content_gated",
    "awaiting_human_input",
}

# Standard limitations every report states — the exact overclaiming traps the
# system must never fall into.
STANDARD_LIMITATIONS = [
    "GPS metadata describes the image/camera capture and may not prove the subject was present.",
    "The registered or visible vehicle does not prove ownership or who operated it.",
    "The activity interpretation does not prove intent or criminal conduct.",
    "Face or identity similarity is a candidate association, not definitive identification.",
]


def _match_value(match: CorrelationResult) -> str:
    """Deterministic value label for a match result (for aggregation)."""
    if match.rule == "face_embedding_cosine":
        return match.metadata.get("person_cluster_id", "face_cluster")
    if match.rule == "exact_string":
        found = _PLATE_VALUE_RE.search(match.explanation)
        return found.group(1).strip() if found else match.explanation
    return match.explanation


def aggregate_match_claims(
    matches: list[CorrelationResult],
    entities: list[Entity],
    *,
    top_n: int = 10,
) -> tuple[list[Claim], int]:
    """Collapse pairwise matches into evidence-linked claims per (rule, value).

    Returns ``(claims, suppressed_count)`` where ``suppressed_count`` is the
    number of raw pairwise matches folded away (weak timestamp links included
    in the count but never emitted as individual claims).
    """
    by_id: dict[str, Entity] = {e.entity_id: e for e in entities}
    groups: dict[tuple[str, str], list[CorrelationResult]] = defaultdict(list)
    weak_count = 0
    for match in matches:
        if match.rule == "timestamp_proximity":
            weak_count += 1
            continue
        groups[(match.rule, _match_value(match))].append(match)

    claims: list[Claim] = []
    for (rule, value), group in sorted(groups.items(), key=lambda kv: -max(m.score for m in kv[1]))[:top_n]:
        refs: list[str] = []
        sources: list[str] = []
        for match in group:
            for entity_id in (match.left_entity_id, match.right_entity_id):
                if entity_id not in refs:
                    refs.append(entity_id)
                entity = by_id.get(entity_id)
                if entity is not None and entity.source_file not in sources:
                    sources.append(entity.source_file)
        max_score = max(m.score for m in group)
        if rule == "face_embedding_cosine":
            claim = (
                f"Face evidence forms cluster {value}: {len(group) + 1} face(s) "
                f"across {len(sources)} source(s): {', '.join(sources)}"
            )
        else:
            claim = (
                f"{rule.replace('_', ' ')}: {value} appears in {len(refs)} "
                f"entity record(s) across {len(sources)} source(s): "
                f"{', '.join(sources)}"
            )
        claims.append(
            Claim(
                claim=claim,
                evidence_refs=refs,
                confidence=round(max_score, 3),
                claim_type="deterministic_fact",
            )
        )
    return claims, weak_count


def _metadata_fact_claims(observations: list) -> list[Claim]:
    """Metadata facts from multimodal observations (GPS, timestamp, serial, plate)."""
    claims: list[Claim] = []
    for observation in normalize_observations(observations):
        if observation.observation_type not in (
            "gps_location",
            "timestamp",
            "camera_serial",
            "number_plate",
            "visible_object",
        ):
            continue
        confidence = observation.confidence if observation.confidence is not None else 0.9
        claims.append(
            Claim(
                claim=(
                    f"{observation.observation_type.replace('_', ' ')}: "
                    f"{observation.value} ({observation.artifact_ref})"
                ),
                evidence_refs=[observation.artifact_ref, observation.observation_id],
                confidence=round(float(confidence), 3),
                claim_type="deterministic_fact",
            )
        )
    return claims


def _h_get(h, key: str, default=None):
    """Read a field from a pydantic hypothesis or a plain dict."""
    if isinstance(h, dict):
        return h.get(key, default)
    return getattr(h, key, default)


def _interpretation_claims(
    hypotheses: list,
    observations: list,
    skeptic: dict | None,
) -> list[Claim]:
    claims: list[Claim] = []
    for h in hypotheses:
        status = _h_get(h, "status", "open") or "open"
        conf = float(_h_get(h, "confidence", 0) or 0)
        hypothesis_id = _h_get(h, "id", "H-?")
        statement = _h_get(h, "statement", "")
        refs = list(_h_get(h, "related_entity_ids", []) or [])
        claims.append(
            Claim(
                claim=f"Hypothesis {hypothesis_id} ({status}, conf {conf:.2f}): {statement}",
                evidence_refs=refs,
                confidence=conf,
                claim_type="interpretation",
            )
        )
    for observation in normalize_observations(observations):
        if observation.observation_type == "activity_interpretation":
            claims.append(
                Claim(
                    claim=(
                        f"LLM visual interpretation ({observation.artifact_ref}): "
                        f"{observation.value} - requires investigator review"
                    ),
                    evidence_refs=[observation.artifact_ref, observation.observation_id],
                    confidence=observation.confidence if observation.confidence is not None else 0.5,
                    claim_type="interpretation",
                )
            )
    if skeptic:
        claims.append(
            Claim(
                claim=f"Skeptic alternative: {skeptic.get('strongest_alternative', '')}",
                evidence_refs=list(skeptic.get("weakening_evidence", [])),
                confidence=round(1.0 - skeptic.get("revised_confidence", 0.5), 3),
                claim_type="interpretation",
            )
        )
    return claims


def compute_confidence_breakdown(state: dict) -> dict:
    """Expose the separate confidence values; ``None`` means "Not scored"."""
    observations = normalize_observations(state.get("image_observations", []))
    hypotheses = state.get("hypotheses", [])
    entities = state.get("entities", [])
    matches = state.get("matches", [])
    breakdown: dict = {
        "action_selection": None,
        "exif_metadata": None,
        "face_association": None,
        "vehicle_association": None,
        "activity_interpretation": None,
        "overall": None,
    }

    exif_scores = [
        float(observation.confidence)
        for observation in observations
        if observation.observation_type in ("gps_location", "timestamp", "camera_serial", "number_plate")
        and observation.confidence is not None
    ]
    if exif_scores:
        breakdown["exif_metadata"] = round(mean(exif_scores), 3)

    def _h(statement: str) -> float | None:
        for h in hypotheses:
            text = str(_h_get(h, "statement", "")).lower()
            if statement in text:
                conf = _h_get(h, "confidence")
                return round(float(conf), 3) if conf is not None else None
        return None

    face_h = _h("same candidate subject") or _h("images 1, 2, and 3")
    if face_h is None:
        face_scores = [
            float(o.confidence)
            for o in observations
            if o.observation_type == "person" and o.confidence is not None
        ]
        if face_scores:
            face_h = round(max(face_scores), 3)
    breakdown["face_association"] = face_h

    vehicle_h = _h("vehicle")
    if vehicle_h is not None:
        breakdown["vehicle_association"] = vehicle_h
    else:
        plate_ids = {e.entity_id for e in entities if e.type == "vehicle_plate"}
        scores = [
            m.score
            for m in matches
            if m.left_entity_id in plate_ids or m.right_entity_id in plate_ids
        ]
        if scores:
            breakdown["vehicle_association"] = round(max(scores), 3)

    activity_scores = [
        float(o.confidence)
        for o in observations
        if o.observation_type == "activity_interpretation" and o.confidence is not None
    ]
    if activity_scores:
        breakdown["activity_interpretation"] = round(max(activity_scores), 3)

    overall = state.get("confidence")
    if overall is not None:
        breakdown["overall"] = round(float(overall), 3)
    action = state.get("current_action")
    if action is not None:
        breakdown["action_selection"] = round(float(getattr(action, "confidence", 0) or 0), 3)
    return breakdown


def _primary_lead(hypotheses: list, matches: list) -> str:
    best = max(
        (h for h in hypotheses if (_h_get(h, "status", "") or "") != "rejected"),
        key=lambda h: float(_h_get(h, "confidence", 0) or 0),
        default=None,
    )
    if best is not None:
        statement = _h_get(best, "statement", "")
        return f"{statement} - candidate association, human confirmation required before it may be relied on."
    if matches:
        top = max(matches, key=lambda m: m.score)
        return f"{top.explanation} (score {top.score:.2f}) — candidate association."
    return "No lead identified yet."


def _synthetic_identity_label(observations: list) -> str:
    labels: set[str] = set()
    for observation in normalize_observations(observations):
        if observation.observation_type != "person":
            continue
        value = observation.value
        marker = "synthetic fixture: "
        if marker in value:
            labels.add(value.split(marker, 1)[1].rstrip(")"))
    return ", ".join(sorted(labels))


def _goal_plate(goal: str) -> str:
    """Compact or hyphenated plate found in the goal text (normalized)."""
    match = re.search(
        r"\b[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,3}\s?\d{3,4}\b"
        r"|\b[A-Z]{2}[- ][A-Z]{2}[- ]?\d{3,4}\b",
        goal or "",
        re.IGNORECASE,
    )
    return re.sub(r"\s+", "", match.group(0)).upper() if match else ""


def build_evidence_report(
    *,
    goal: str,
    matches: list[CorrelationResult],
    entities: list[Entity],
    hypotheses: list,
    contradictions: list[dict],
    skeptic: dict | None,
    final_confidence: float,
    initial_confidence: float,
    stop_reason: str = "",
    image_observations: list | None = None,
    confidence_breakdown: dict | None = None,
    needs_human_review: bool = False,
    registry_results: list | None = None,
    image_count: int = 0,
    ocr_available: bool = False,
) -> str:
    """Render the evidence-linked markdown report for the dynamic loop."""
    observations = image_observations or []
    fact_claims, weak_count = aggregate_match_claims(matches, entities)
    metadata_claims = _metadata_fact_claims(observations)
    interpretation_claims = _interpretation_claims(hypotheses, observations, skeptic)

    missing: list[str] = []
    for h in hypotheses:
        for item in _h_get(h, "missing_evidence", []) or []:
            if item not in missing:
                missing.append(item)
    if skeptic:
        for item in skeptic.get("missing_evidence", []):
            if item not in missing:
                missing.append(item)

    review_required = needs_human_review or stop_reason in HUMAN_REVIEW_STOP_REASONS or bool(contradictions)
    status = "HUMAN REVIEW REQUIRED" if review_required else "INVESTIGATION COMPLETE"
    identity_label = _synthetic_identity_label(observations)

    lines: list[str] = [
        "# Investigation status",
        "",
        f"**{status}**",
        "",
        f"**Goal:** {goal}",
        f"**Stop reason:** {stop_reason or 'goal completed'}",
        f"**Confidence:** {initial_confidence:.2f} -> **{final_confidence:.2f}**",
        "",
    ]

    lines.append("## Primary lead")
    lines.append(f"- {_primary_lead(hypotheses, matches)}")
    if identity_label:
        lines.append(f"- Synthetic fixture identity label: {identity_label} (demo-only; not real identification)")
    lines.append("")

    # Approved external-registry (mock MVD) findings are metadata facts too:
    # registered owner is a registry fact, never proof of who operated the
    # vehicle. Clearly labeled synthetic so it cannot be mistaken for real data.
    registry_claims: list[Claim] = []
    for result in registry_results or []:
        plate = str(result.get("key", "")).upper()
        for finding in result.get("findings", []):
            registry_claims.append(
                Claim(
                    claim=(
                        f"{finding.get('field', 'field').replace('_', ' ')}: "
                        f"{finding.get('value', '')} (mock MVD registry - "
                        f"synthetic fixture for {plate}; registered owner is not "
                        f"proof of who operated the vehicle)"
                    ),
                    evidence_refs=[plate],
                    confidence=float(finding.get("confidence", 0.5)),
                    claim_type="deterministic_fact",
                )
            )

    lines.append("## Confirmed observations")
    observed = fact_claims + metadata_claims + registry_claims
    if observed:
        for claim in observed:
            refs = ", ".join(claim.evidence_refs[:6])
            more = f" (+{len(claim.evidence_refs) - 6} refs)" if len(claim.evidence_refs) > 6 else ""
            lines.append(f"- {claim.claim}  *(conf {claim.confidence:.2f}; refs: {refs}{more})*")
        if weak_count:
            lines.append(
                f"- *{weak_count} weak timestamp-proximity links within the window "
                f"were folded into this summary rather than listed individually.*"
            )
    else:
        lines.append("- No deterministic findings yet.")
    lines.append("")

    # Honest evidence-gap note: images were ingested but nothing machine-
    # readable came out of the images themselves (no EXIF GPS/camera data, no
    # OCR engine, plate only in the goal). Registry claims are separate: they
    # describe a synthetic external record, not the images. Karuth never
    # pretends a photo of a plate is evidence of one.
    if image_count and not fact_claims and not metadata_claims and not entities:
        plate_in_goal = _goal_plate(goal)
        lines.append("## Evidence gap")
        lines.append(
            f"- {image_count} image(s) were ingested, but no machine-readable "
            "observations were extracted from them."
        )
        if plate_in_goal:
            lines.append(
                f"- The plate **{plate_in_goal}** appears only in the investigator "
                "goal, not in the evidence files: Karuth does not invent "
                "evidence from a photo's pixels."
            )
        if ocr_available:
            lines.append(
                "- OCR found no plate-like text in the photos (or the images "
                "carry no EXIF GPS/camera metadata)."
            )
        else:
            lines.append(
                "- No OCR engine is installed, so plate text in the photos could "
                "not be read automatically. Options: install Tesseract OCR, add "
                "an .exif.json metadata sidecar per image, or use a vision model."
            )
        lines.append("")

    lines.append("## AI interpretations")
    if interpretation_claims:
        for claim in interpretation_claims:
            lines.append(f"- {claim.claim}")
    else:
        lines.append("- None.")
    lines.append("")

    lines.append("## Contradictions and limitations")
    if contradictions:
        for c in contradictions:
            lines.append(
                f"- **[{c.get('severity', 'medium')}] {c.get('rule', '')}** - "
                f"{c.get('claim_a', '')} vs {c.get('claim_b', '')} "
                f"*(refs: {', '.join(c.get('evidence_refs', [])[:4])})*"
            )
    for limitation in STANDARD_LIMITATIONS:
        lines.append(f"- {limitation}")
    if registry_results:
        lines.append(
            "- Registry findings are synthetic mock data; they describe the "
            "registration record and do not connect anyone to this case."
        )
    if skeptic and skeptic.get("requires_human_review"):
        lines.append("- Human review required before the conclusion is relied on.")
    lines.append("")

    lines.append("## Recommended next action")
    checkpoints: list[str] = []
    seen_checkpoints: set[str] = set()
    for observation in normalize_observations(observations):
        if observation.observation_type == "activity_interpretation":
            checkpoints.append("Review the activity interpretation for Image 1 (LLM visual interpretation).")
        elif observation.observation_type == "person" and observation.requires_human_review:
            checkpoints.append("Confirm the face/subject association across the images before naming anyone.")
        elif observation.observation_type == "number_plate" and observation.requires_human_confirmation:
            checkpoints.append("Confirm the ambiguous plate reading before relying on it.")
    if any(o.observation_type == "sensitive_content" for o in normalize_observations(observations)):
        checkpoints.append("Sensitive imagery is blurred; reveal only with investigator authorization.")
    for item in checkpoints:
        if item not in seen_checkpoints:
            seen_checkpoints.add(item)
            lines.append(f"- {item}")
    lines.append("- **Missing evidence:** " + ("; ".join(missing[:6]) if missing else "none explicitly required beyond human confirmation"))
    lines.append("")

    lines.append("## Conclusion")
    lines.append(
        "The evidence supports further investigation but does not establish "
        "identity, intent, ownership, or criminal activity."
    )
    lines.append("")
    return "\n".join(lines)
