"""Ledger-backed audit trail for planner decisions.

Every meaningful planner transition is appended to the hash-chained ledger so
the whole investigation is auditable and replayable. Entry types follow the
existing schema: ``agent_action`` for planner steps, ``review_action`` for
reveal/authorization decisions, ``export`` for report/export actions, and
``evidence`` for ingestion events the planner triggers.
"""

from __future__ import annotations

from typing import Any

from karuth.ledger import LedgerEntry, LedgerEntryCreate, LedgerStore
from karuth.planner.schema import InvestigationPlan


# Entry types used by the dynamic hypothesis loop. Keeping them under
# ``agent_action`` (like the rest of the planner) preserves the ledger schema.
_HYPOTHESIS_ENTRY = "agent_action"
_ACTION_ENTRY = "agent_action"
_OBSERVATION_ENTRY = "agent_action"
_CONTRADICTION_ENTRY = "agent_action"
_SKEPTIC_ENTRY = "agent_action"


class PlannerAuditor:
    def __init__(self, ledger: LedgerStore, case_id: str, actor: str = "planner") -> None:
        self._ledger = ledger
        self._case_id = case_id
        self._actor = actor

    def _append(
        self,
        entry_type: str,
        action: str,
        target: str,
        payload: dict[str, Any],
    ) -> LedgerEntry:
        return self._ledger.append(
            LedgerEntryCreate(
                entry_type=entry_type,
                case_id=self._case_id,
                actor=self._actor,
                action=action,
                target=target,
                payload=payload,
            )
        )

    # ---- agent_action -----------------------------------------------------
    def goal_received(self, goal: str) -> LedgerEntry:
        return self._append("agent_action", "goal_received", "goal", {"goal": goal})

    def planning_backend(self, backend: str, model: str = "") -> LedgerEntry:
        return self._append(
            "agent_action", "planning_backend", "plan", {"backend": backend, "model": model}
        )

    def evidence_inventory(self, evidence: list[dict]) -> LedgerEntry:
        return self._append(
            "agent_action",
            "evidence_inventory",
            "case_evidence",
            {"evidence": evidence},
        )

    def task_skipped(self, tool: str, reason: str) -> LedgerEntry:
        return self._append(
            "agent_action", "task_skipped", "task", {"tool": tool, "reason": reason}
        )

    def fallback_used(self, backend: str, error: str) -> LedgerEntry:
        return self._append(
            "agent_action",
            "fallback_used",
            "plan",
            {"fallback": "DeterministicPlanner", "primary": backend, "error": error},
        )

    def goal_completed(self, summary: dict) -> LedgerEntry:
        return self._append("agent_action", "goal_completed", "goal", summary)

    def plan_generated(self, plan: InvestigationPlan, *, replan: bool = False) -> LedgerEntry:
        return self._append(
            "agent_action",
            "replan_generated" if replan else "plan_generated",
            "plan",
            {
                "tool_list": [task.tool for task in plan.tasks],
                "rationale": {task.tool: task.reason for task in plan.tasks},
                "risk_level": plan.safety_policy.risk_level,
                "plan": plan.model_dump(mode="json"),
            },
        )

    def task_selected(self, tool: str, reason: str) -> LedgerEntry:
        return self._append(
            "agent_action", "task_selected", "task", {"tool": tool, "reason": reason}
        )

    def task_dispatched(self, tool: str, inputs: list[str]) -> LedgerEntry:
        return self._append(
            "agent_action", "task_dispatched", "task", {"tool": tool, "inputs": inputs}
        )

    def task_completed(self, tool: str, summary: str) -> LedgerEntry:
        return self._append(
            "agent_action", "task_completed", "task", {"tool": tool, "summary": summary}
        )

    def task_failed(self, tool: str, error: str) -> LedgerEntry:
        return self._append(
            "agent_action", "task_failed", "task", {"tool": tool, "error": error}
        )

    def postcondition(self, name: str, ok: bool) -> LedgerEntry:
        return self._append(
            "agent_action", "postcondition_checked", "postcondition", {"name": name, "ok": ok}
        )

    def replan_triggered(self, reason: str) -> LedgerEntry:
        return self._append(
            "agent_action", "replan_triggered", "plan", {"reason": reason}
        )

    def chain_verified(self, ok: bool) -> LedgerEntry:
        return self._append(
            "agent_action", "chain_verified", "ledger", {"ok": ok}
        )

    # ---- dynamic hypothesis loop -----------------------------------------
    def hypothesis_created(self, summary: dict) -> LedgerEntry:
        return self._append(
            _HYPOTHESIS_ENTRY,
            "hypothesis_created",
            "hypothesis",
            summary,
        )

    def hypothesis_transitioned(self, hypothesis_id: str, old: str, new: str) -> LedgerEntry:
        return self._append(
            _HYPOTHESIS_ENTRY,
            "hypothesis_transitioned",
            "hypothesis",
            {"hypothesis_id": hypothesis_id, "from": old, "to": new},
        )

    def action_decided(self, decision: dict) -> LedgerEntry:
        return self._append(
            _ACTION_ENTRY,
            "action_decided",
            "action",
            decision,
        )

    def observation_recorded(self, observation: dict) -> LedgerEntry:
        return self._append(
            _OBSERVATION_ENTRY,
            "observation_recorded",
            "observation",
            observation,
        )

    def contradiction_found(self, contradiction: dict) -> LedgerEntry:
        return self._append(
            _CONTRADICTION_ENTRY,
            "contradiction_found",
            "contradiction",
            contradiction,
        )

    def skeptic_reviewed(self, review: dict) -> LedgerEntry:
        return self._append(
            _SKEPTIC_ENTRY,
            "skeptic_reviewed",
            "skeptic_review",
            review,
        )

    def dynamic_finalized(self, summary: dict) -> LedgerEntry:
        return self._append(
            _ACTION_ENTRY,
            "dynamic_finalized",
            "plan",
            summary,
        )

    def human_input_requested(self, note: str) -> LedgerEntry:
        return self._append(
            _ACTION_ENTRY,
            "human_input_requested",
            "checkpoint",
            {"note": note},
        )

    def planner_finished(self, summary: dict[str, Any]) -> LedgerEntry:
        return self._append(
            "agent_action", "planner_finished", "plan", summary
        )

    # ---- evidence ---------------------------------------------------------
    def evidence_ingested(self, entity_count: int) -> LedgerEntry:
        return self._append(
            "evidence", "evidence_ingested", "case_evidence", {"entity_count": entity_count}
        )

    # ---- review_action ----------------------------------------------------
    def review_request(self, target: str, tier: str, dual_required: bool) -> LedgerEntry:
        return self._append(
            "review_action",
            "reveal_requested",
            target,
            {"tier": tier, "dual_authorization_required": dual_required},
        )

    def critical_gated(self, target: str) -> LedgerEntry:
        return self._append("review_action", "critical_content_gated", target, {"gated": True})

    def review_decision(self, target: str, decision: str, note: str = "") -> LedgerEntry:
        return self._append(
            "review_action",
            "review_decision",
            target,
            {"decision": decision, "note": note},
        )

    def authorization_granted(self, target: str) -> LedgerEntry:
        return self._append(
            "review_action", "authorization_granted", target, {"granted": True}
        )

    # ---- external registry lookups ---------------------------------------
    def registry_lookup_approved(self, lookup_kind: str) -> LedgerEntry:
        return self._append(
            "review_action",
            "registry_lookup_approved",
            lookup_kind,
            {"approved": True, "mode": "synthetic_mock"},
        )

    def registry_lookup_result(self, result: dict) -> LedgerEntry:
        return self._append(
            "agent_action",
            "registry_lookup_result",
            "external_lookup",
            {
                "lookup_kind": result.get("lookup_kind"),
                "key": result.get("key"),
                "source_system": result.get("source_system"),
                "retrieved_at": result.get("retrieved_at"),
                "findings": len(result.get("findings", [])),
            },
        )

    # ---- export -----------------------------------------------------------
    def export(self, kind: str, target: str, summary: dict[str, Any]) -> LedgerEntry:
        return self._append(
            "export", f"{kind}_exported", target, summary
        )

    # ---- integrity --------------------------------------------------------
    def verify(self) -> bool:
        return self._ledger.verify()
