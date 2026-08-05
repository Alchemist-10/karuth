from __future__ import annotations

import pytest

from scripts.generate_synthetic_case import generate_case
from karuth.ledger import LedgerStore
from karuth.planner.guardrails import ensure_inside_case, reject_external_url
from karuth.planner.state_machine import run_planner_goal


def test_planner_extracts_correlates_triages_and_audits(tmp_path):
    case_dir = tmp_path / "synthetic_case"
    generate_case(case_dir)
    ledger = LedgerStore(tmp_path / "ledger.sqlite3")

    state = run_planner_goal(
        "CASE-1",
        case_dir,
        "Find links for vehicle KL07CX4242",
        ledger,
    )

    assert state["entities"]
    assert state["matches"]
    assert state["ranked_matches"]
    assert [entry.action for entry in ledger.list_entries()] == [
        "decompose_goal",
        "extract_entities",
        "match_entities",
        "rank_leads",
    ]
    assert ledger.verify()


def test_guardrails_reject_out_of_scope_and_external_urls(tmp_path):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("nope", encoding="utf-8")

    with pytest.raises(ValueError):
        ensure_inside_case(case_dir, outside)

    with pytest.raises(ValueError):
        reject_external_url("https://example.com/data")
