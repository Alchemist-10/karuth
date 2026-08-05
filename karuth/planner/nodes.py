"""Planner nodes for the minimal offline state machine."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from karuth.agents.correlation_agent import correlate_entities
from karuth.agents.extraction_agent import extract_case_entities
from karuth.agents.triage_agent import rank_leads
from karuth.correlation.candidate_matcher import CorrelationResult
from karuth.entities.schema import Entity
from karuth.ledger import LedgerEntryCreate, LedgerStore
from karuth.planner.guardrails import ensure_inside_case, reject_external_url


class PlannerState(TypedDict, total=False):
    case_folder: Path
    goal: str
    entities: list[Entity]
    matches: list[CorrelationResult]
    ranked_matches: list[CorrelationResult]


def decompose_goal(state: PlannerState, ledger: LedgerStore, case_id: str) -> PlannerState:
    reject_external_url(state["goal"])
    ledger.append(
        LedgerEntryCreate(
            entry_type="agent_action",
            case_id=case_id,
            actor="planner",
            action="decompose_goal",
            target="goal",
            payload={"goal": state["goal"], "steps": ["extract", "correlate", "triage"]},
        )
    )
    return state


def dispatch_extract(state: PlannerState, ledger: LedgerStore, case_id: str) -> PlannerState:
    case_folder = ensure_inside_case(state["case_folder"], state["case_folder"])
    entities = extract_case_entities(case_folder)
    state["entities"] = entities
    ledger.append(
        LedgerEntryCreate(
            entry_type="agent_action",
            case_id=case_id,
            actor="extraction_agent",
            action="extract_entities",
            target=str(case_folder),
            payload={"entity_count": len(entities)},
        )
    )
    return state


def dispatch_correlate(state: PlannerState, ledger: LedgerStore, case_id: str) -> PlannerState:
    matches = correlate_entities(state.get("entities", []))
    state["matches"] = matches
    ledger.append(
        LedgerEntryCreate(
            entry_type="agent_action",
            case_id=case_id,
            actor="correlation_agent",
            action="match_entities",
            target="entities",
            payload={"match_count": len(matches)},
        )
    )
    return state


def dispatch_triage(state: PlannerState, ledger: LedgerStore, case_id: str) -> PlannerState:
    ranked = rank_leads(state.get("matches", []), state.get("goal", ""))
    state["ranked_matches"] = ranked
    ledger.append(
        LedgerEntryCreate(
            entry_type="agent_action",
            case_id=case_id,
            actor="triage_agent",
            action="rank_leads",
            target="matches",
            payload={"ranked_count": len(ranked)},
        )
    )
    return state
