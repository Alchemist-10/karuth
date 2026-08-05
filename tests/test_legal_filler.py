from __future__ import annotations

import pytest

from karuth.agents.action_agent import fill_legal_document
from karuth.legal.filler import render_section_94_notice
from karuth.review.auth import authorize_critical_reveal
from karuth.review.risk_tiers import classify_content


def test_legal_filler_uses_template_fields_only():
    rendered = render_section_94_notice(
        __import__("pathlib").Path("karuth/legal/templates"),
        {
            "case_id": "CASE-1",
            "recipient_name": "Synthetic Telecom Ltd",
            "requested_records": "Synthetic subscriber records",
            "officer_name": "Inspector Demo",
        },
    )

    assert "CASE-1" in rendered
    assert "{{" not in rendered


def test_action_agent_rejects_unknown_template():
    with pytest.raises(ValueError):
        fill_legal_document(__import__("pathlib").Path("karuth/legal/templates"), "freeform", {})


def test_critical_reveal_requires_distinct_authorizers():
    authorization = authorize_critical_reveal(
        case_id="CASE-1",
        target_id="face-1",
        requested_by="investigator-a",
        approved_by="supervisor-b",
        reason="Synthetic review check",
    )

    assert authorization.requested_by != authorization.approved_by
    with pytest.raises(ValueError):
        authorize_critical_reveal(
            case_id="CASE-1",
            target_id="face-1",
            requested_by="investigator-a",
            approved_by="investigator-a",
            reason="same person",
        )


def test_risk_tiers_blur_faces_by_default():
    assert classify_content("face_embedding").tier == "MED"
    assert classify_content("high_harm_content").tier == "CRITICAL"
