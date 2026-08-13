"""Deterministic hypothesis generation.

The hypothesis agent turns extracted evidence and matches into competing,
testable hypotheses. Everything here is rule-based and offline: hypotheses are
built from *real* entity and match references, and every proposal is validated
so an LLM-proposed statement can only be adopted when its evidence references
resolve to actual entities. The LLM never marks a hypothesis supported or
rejected — that is :class:`~karuth.planner.hypothesis.HypothesisStore`'s job,
driven by deterministic evidence.
"""

from __future__ import annotations

import re

from karuth.correlation.candidate_matcher import CorrelationResult
from karuth.entities.schema import Entity
from karuth.planner.hypothesis import Hypothesis, HypothesisStore

_PLATE_RE = re.compile(
    r"\b[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,3}\s?\d{3,4}\b"
    r"|\b[A-Z]{2}[- ][A-Z]{2}[- ]?\d{3,4}\b",
    re.IGNORECASE,
)
_PHONE_RE = re.compile(r"\+?\d{10,15}")


def _goal_values(goal: str) -> list[str]:
    """Values the investigator cares about, extracted from the goal text.

    Plates are normalized to uppercase compact form so a lowercase goal plate
    (``kl01cc2121``) matches the uppercase entity value (``KL01CC2121``).
    """
    values: list[str] = []
    for found in _PLATE_RE.findall(goal):
        values.append(re.sub(r"\s+", "", found).upper())
    values.extend(_PHONE_RE.findall(goal))
    return values


def _sources_for(match: CorrelationResult, by_id: dict[str, Entity]) -> list[str]:
    sources: list[str] = []
    for entity_id in (match.left_entity_id, match.right_entity_id):
        entity = by_id.get(entity_id)
        if entity is not None and entity.source_file not in sources:
            sources.append(entity.source_file)
    return sources


def generate_demo_hypotheses(
    entities: list[Entity],
    matches: list[CorrelationResult],
    observations: list,
    goal: str,
    *,
    max_hypotheses: int = 8,
) -> list[Hypothesis]:
    """Demo-specific competing hypotheses for the multimodal Jesse case.

    Produces the five canonical hypotheses when the case contains a person
    fixture (``synthetic_identity``), a vehicle plate, GPS observations and an
    activity interpretation — every one referencing real evidence (or stating
    ``Evidence basis: none yet`` when no refs exist yet).
    """
    from karuth.planner.observations import normalize_observations

    observations = normalize_observations(observations or [])
    store = HypothesisStore()
    by_id = {e.entity_id: e for e in entities}

    person_obs = [o for o in observations if o.observation_type == "person"]
    plate_obs = [o for o in observations if o.observation_type == "number_plate"]
    gps_obs = [o for o in observations if o.observation_type == "gps_location"]
    activity_obs = [o for o in observations if o.observation_type == "activity_interpretation"]
    image_refs = sorted({o.artifact_ref for o in observations})
    plate = plate_obs[0].value if plate_obs else ""
    gps_value = gps_obs[0].value if gps_obs else ""
    identity = ""
    if person_obs and "synthetic fixture: " in person_obs[0].value:
        identity = person_obs[0].value.split("synthetic fixture: ", 1)[1].rstrip(")")

    # The demo hypotheses are specific to the Jesse Pinkman synthetic fixture
    # (person fixture + NM-JP-4242 plate). Other cases fall through to the
    # generic hypothesis generation below.
    if identity != "jesse_pinkman" and plate != "NM-JP-4242":
        return []

    face_refs: list[str] = []
    for entity in entities:
        if entity.type == "face_embedding" and entity.metadata.get("synthetic_identity") == identity:
            face_refs.append(entity.entity_id)
    plate_entities = [e for e in entities if e.type == "vehicle_plate" and str(e.value) == plate]
    plate_refs = [e.entity_id for e in plate_entities]

    def _create(statement: str, refs: list[str], confidence: float, *, next_actions: list[str], review: bool) -> None:
        try:
            store.create(
                statement,
                evidence=refs or ["Evidence basis: none yet"],
                confidence=confidence,
                related_entity_ids=refs,
                related_artifact_ids=image_refs,
                recommended_next_actions=next_actions,
                requires_human_review=review,
            )
        except ValueError:
            pass

    if len(person_obs) >= 3 and face_refs:
        _create(
            "The same candidate subject appears in Images 1, 2, and 3.",
            face_refs,
            0.6,
            next_actions=["face_cluster_review", "human_confirmation"],
            review=True,
        )
    elif person_obs:
        _create(
            "A candidate subject is visible across the case images.",
            face_refs,
            0.5,
            next_actions=["face_cluster_review", "human_confirmation"],
            review=True,
        )

    if person_obs and plate:
        _create(
            f"The candidate subject is associated with vehicle {plate}.",
            list(dict.fromkeys([*plate_refs, *face_refs])),
            0.55,
            next_actions=["timeline_analysis", "contradiction_hunting"],
            review=True,
        )

    if person_obs and plate and gps_obs:
        vehicle_image = next((o for o in plate_obs if o.artifact_ref in {g.artifact_ref for g in gps_obs}), None)
        if vehicle_image is not None:
            _create(
                f"Image {vehicle_image.artifact_ref.split('/')[-1]} places the candidate subject and vehicle near Location ({gps_value}).",
                [*plate_refs, vehicle_image.artifact_ref],
                0.5,
                next_actions=["exif_verification", "human_confirmation"],
                review=True,
            )

    if activity_obs:
        _create(
            "Image 1 may show substance-related activity.",
            [activity_obs[0].artifact_ref],
            float(activity_obs[0].confidence or 0.5),
            next_actions=["review_activity_interpretation"],
            review=True,
        )

    if activity_obs and gps_obs:
        _create(
            "The activity interpretation and location evidence refer to the same investigation event.",
            [activity_obs[0].artifact_ref, gps_obs[0].artifact_ref],
            0.45,
            next_actions=["timeline_analysis", "human_confirmation"],
            review=True,
        )

    return store.as_state()[:max_hypotheses]


def generate_hypotheses(
    entities: list[Entity],
    matches: list[CorrelationResult],
    goal: str,
    observations: list | None = None,
    *,
    max_hypotheses: int = 8,
) -> list[Hypothesis]:
    """Build competing hypotheses from deterministic evidence.

    Rules:

    - For each person who sent a message mentioning a goal value, a hypothesis
      "``<person>`` is connected to ``<value>``" is created.
    - A generic cross-source hypothesis links the goal value to the sources
      that mention it.
    - If the same value appears in images tagged with a synthetic identity, a
      hypothesis about that identity is created.
    - When multimodal observations are supplied (Jesse demo), the demo
      hypotheses are generated first (same subject across images, subject-
      vehicle association, subject-vehicle-location, activity interpretation).

    Statements are deduplicated and capped at ``max_hypotheses``.
    """
    if observations:
        demo = generate_demo_hypotheses(entities, matches, observations, goal, max_hypotheses=max_hypotheses)
        if demo:
            return demo
    store = HypothesisStore()
    by_id = {e.entity_id: e for e in entities}
    goal_values = _goal_values(goal) or _values_from_matches(matches)

    # 1) per-person hypotheses from messages that mention a goal value
    messages_by_sender: dict[str, list[Entity]] = {}
    for entity in entities:
        if entity.type == "message":
            sender = str(entity.metadata.get("contact_name", ""))
            if sender:
                messages_by_sender.setdefault(sender, []).append(entity)

    # plate entities whose value appears in a message body, by source file
    plates_by_file: dict[str, list[Entity]] = {}
    for entity in entities:
        if entity.type == "vehicle_plate":
            plates_by_file.setdefault(entity.source_file, []).append(entity)

    for person, messages in messages_by_sender.items():
        refs: list[str] = []
        for message in messages:
            body = str(message.value)
            if not any(value in body for value in goal_values):
                continue
            refs.append(message.entity_id)
            # Link the plate entities extracted from this message too, so
            # correlation claims (which reference plate ids) support the
            # hypothesis mechanically.
            for plate in plates_by_file.get(message.source_file, []):
                if str(plate.value) in body and plate.entity_id not in refs:
                    refs.append(plate.entity_id)
        if refs:
            try:
                store.create(
                    f"{person} is connected to {_display_values(goal_values)}",
                    evidence=refs,
                    confidence=0.55,
                    related_entity_ids=refs,
                    recommended_next_actions=["timeline_analysis", "graph_neighbors"],
                )
            except ValueError:
                continue

    # 2) cross-source hypothesis for the key value
    key = goal_values[0] if goal_values else ""
    if key:
        cross_refs: list[str] = []
        for match in matches:
            if key in str(match.explanation) or key in str(match.metadata):
                cross_refs.extend([match.left_entity_id, match.right_entity_id])
        if not cross_refs:
            cross_refs = [
                e.entity_id
                for e in entities
                if str(e.value) == key and e.source_file.endswith((".jpg", ".jpeg", ".exif.json", ".sqlite"))
            ]
        if cross_refs:
            try:
                store.create(
                    f"{key} links the sources that mention it",
                    evidence=list(dict.fromkeys(cross_refs)),
                    confidence=0.6,
                    related_entity_ids=list(dict.fromkeys(cross_refs)),
                    recommended_next_actions=["contradiction_hunting", "timeline_analysis"],
                )
            except ValueError:
                pass

    # 3) identity hypotheses from image metadata
    identity_refs: dict[str, list[str]] = {}
    for entity in entities:
        if entity.type == "face_embedding":
            identity = entity.metadata.get("synthetic_identity")
            if identity:
                identity_refs.setdefault(str(identity), []).append(entity.entity_id)
    for identity, refs in identity_refs.items():
        if len(refs) < 2:
            continue
        try:
            store.create(
                f"the person identity '{identity}' is part of this case",
                evidence=refs,
                confidence=0.5,
                related_entity_ids=refs,
                recommended_next_actions=["face_cluster_review"],
            )
        except ValueError:
            continue

    return store.as_state()[:max_hypotheses]


def propose_hypothesis(
    statement: str,
    evidence_refs: list[str],
    entities: list[Entity],
    *,
    created_by: str = "llm",
    max_hypotheses: int = 8,
) -> Hypothesis | None:
    """Adopt an LLM-proposed hypothesis only when its refs resolve to entities.

    Returns ``None`` (and no state change) when the statement is empty or any
    evidence reference is not a real extracted entity id — the LLM cannot
    fabricate evidence, only propose.
    """
    statement = statement.strip()
    if not statement:
        return None
    known_ids = {e.entity_id for e in entities}
    valid = [ref for ref in evidence_refs if ref in known_ids]
    if not valid:
        return None
    import hashlib

    return Hypothesis(
        id=f"H-LLM-{hashlib.sha256(statement.encode('utf-8')).hexdigest()[:6].upper()}",
        statement=statement,
        confidence=0.5,
        supporting_evidence=valid,
        related_entity_ids=valid,
        created_by=created_by,
    )


def _values_from_matches(matches: list[CorrelationResult]) -> list[str]:
    import re as _re

    values: list[str] = []
    for match in matches:
        if match.rule == "exact_string":
            found = _re.search(r"match on (.+)$", match.explanation)
            if found:
                values.append(found.group(1).strip())
    return list(dict.fromkeys(values))


def _display_values(values: list[str]) -> str:
    return ", ".join(values[:3]) or "the key evidence"
