"""Pure helpers shared by the planner page and its tests.

These functions contain no Streamlit calls, so they can be imported and tested
in isolation. They normalize pydantic models or plain dicts into the plain
dicts the investigation view renders, and map internal action labels to
investigator-facing text (spec item 10).
"""

from __future__ import annotations

from typing import Any

# Investigator-facing labels for internal action types.
ACTION_LABELS = {
    "run_tool": "Investigate",
    "investigate_hypothesis": "Investigate hypothesis",
    "generate_hypothesis": "Generate hypothesis",
    "challenge_hypothesis": "Challenge hypothesis",
    "ask_human": "Ask investigator",
    "finalize": "Finalize",
}

CONFIDENCE_LABELS = {
    "action_selection": "Action selection",
    "exif_metadata": "EXIF metadata",
    "face_association": "Face association",
    "vehicle_association": "Vehicle association",
    "activity_interpretation": "Activity interpretation",
    "overall": "Overall case",
}


def as_dict(value: Any) -> dict:
    """Coerce a pydantic model or dict into a plain dict (UI tolerance)."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return {}


def as_list(values: Any) -> list:
    return list(values or [])


def hypothesis_fields(h: Any) -> dict:
    """Flatten a Hypothesis (model or dict) into display fields."""
    data = as_dict(h)
    return {
        "id": data.get("id", "H-?"),
        "statement": data.get("statement", ""),
        "status": data.get("status", "open"),
        "confidence": data.get("confidence"),
        "supporting": as_list(data.get("supporting_evidence")),
        "contradicting": as_list(data.get("contradicting_evidence")),
        "missing": as_list(data.get("missing_evidence")),
        "next": as_list(data.get("recommended_next_actions")),
        "review": bool(data.get("requires_human_review")),
        "artifacts": as_list(data.get("related_artifact_ids")),
    }


def action_fields(a: Any) -> dict:
    """Flatten an ActionDecision (model or dict) into display fields."""
    data = as_dict(a)
    return {
        "action_type": data.get("action_type", ""),
        "tool": data.get("tool", ""),
        "target": data.get("target_hypothesis_id", ""),
        "rationale": data.get("rationale", ""),
        "gain": data.get("expected_information_gain"),
        "confidence": data.get("confidence"),
    }


def human_readable_action(action_type: str) -> str:
    """Internal ``action_type`` -> investigator-facing label."""
    return ACTION_LABELS.get(action_type, action_type.replace("_", " "))
