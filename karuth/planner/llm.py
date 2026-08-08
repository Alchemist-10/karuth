"""Replaceable LLM planning backend.

The planner uses the backend only for goal interpretation, task selection, and
replanning. Evidence extraction and correlation stay deterministic. The default
backend is offline-safe (keyword based); an optional Ollama backend can be
swapped in when `langchain-ollama` is installed.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from karuth.config import REASONING_MODEL
from karuth.planner.schema import InvestigationPlan, PlannerTask, SafetyPolicy

EVIDENCE_KEYWORDS = (
    "video",
    "chat",
    "image",
    "photo",
    "phone",
    "transaction",
    "bank",
    "face",
    "document",
    "account",
    "message",
    "contact",
    "gps",
    "location",
)
CORRELATE_KEYWORDS = (
    "correlat",
    "cross-reference",
    "cross reference",
    "link",
    "links",
    "linkage",
    "relationship",
    "associate",
    "connection",
    "connect",
    "alias",
    "device-to-person",
    "mapping",
    "network",
    "who else",
    "linked",
)
TIMELINE_KEYWORDS = (
    "timeline",
    "sequence",
    "chronolog",
    "when",
    "before",
    "after",
    "movement",
    "schedule",
    "order of events",
    "map their network",
    "activity",
)
TRIAGE_KEYWORDS = (
    "triage",
    "top lead",
    "top leads",
    "priority",
    "most important",
    "rank",
    "suspect",
    "high-risk",
    "high risk",
    "most relevant",
    "prioritize",
)
REPORT_KEYWORDS = (
    "report",
    "summary",
    "brief",
    "findings",
    "conclusion",
    "overview",
    "briefing",
    "case brief",
)
LEGAL_KEYWORDS = (
    "legal",
    "certificate",
    "notice",
    "section 63",
    "section 94",
    "bsa",
    "bnss",
    "summons",
)
SENSITIVE_KEYWORDS = (
    "sensitive",
    "high-risk",
    "high risk",
    "explicit",
    "blur",
    "reveal",
    "protected",
    "nsfw",
    "adult",
)
CRITICAL_KEYWORDS = ("critical", "credential", "password", "dual", "extreme")

TOOL_REASONS: dict[str, str] = {
    "ExtractionAgent": "goal depends on case evidence; extraction must run first",
    "CorrelationAgent": "goal mentions linking/relationships; deterministic correlation may find matches",
    "TimelineAgent": "goal asks about sequence or timing; ordering entities chronologically",
    "TriageAgent": "goal asks for priority or top leads; rank matches by score and relevance",
    "ReportAgent": "produce an investigator-ready case summary",
    "LegalActionAgent": "goal requests a legal document; fill the requested template",
}


class PlannerLLM(Protocol):
    """Contract for any planner backend (deterministic or LLM)."""

    def generate_plan(self, goal: str, state_summary: dict[str, Any]) -> InvestigationPlan:
        """Produce a structured plan for a goal."""

    def replan(
        self,
        goal: str,
        state_summary: dict[str, Any],
        previous_plan: InvestigationPlan | None = None,
    ) -> InvestigationPlan:
        """Produce a revised plan when the first pass is insufficient."""


def _hit(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def _legal_inputs(goal: str) -> tuple[str, list[str], dict[str, Any]]:
    if "notice" in goal or "section 94" in goal:
        return "section_94_bnss_notice", ["section_94_bnss_notice"], {
            "recipient_name": "Pending Recipient Input",
            "requested_records": "Pending Records Request",
        }
    return "section_63_bsa_certificate", ["section_63_bsa_certificate"], {
        "source_description": "Synthetic evidence bundle",
        "evidence_hash": "PENDING",
    }


def _build_plan(
    goal: str,
    tools: list[str],
    state_summary: dict[str, Any],
    *,
    risk_level: str = "LOW",
    blur_by_default: bool = False,
    dual_authorization_required: bool = False,
    confidence: float = 0.7,
) -> InvestigationPlan:
    ordered = list(dict.fromkeys(tools))
    tasks: list[PlannerTask] = []
    for step, tool in enumerate(ordered, start=1):
        inputs: list[str] = []
        fields: dict[str, Any] = {}
        if tool == "LegalActionAgent":
            _, inputs, extra = _legal_inputs(goal)
            fields = {
                "case_id": state_summary.get("case_id", "UNKNOWN"),
                "officer_name": "Pending Officer Input",
                **extra,
            }
        tasks.append(
            PlannerTask(
                step=step,
                tool=tool,
                reason=TOOL_REASONS[tool],
                inputs=inputs,
                fields=fields,
            )
        )

    expected_output = ["investigation_plan_json", "ledger_audit_entries"]
    if "ExtractionAgent" in ordered:
        expected_output.append("extracted_entities")
    if "CorrelationAgent" in ordered:
        expected_output.append("correlation_matches")
    if "TimelineAgent" in ordered:
        expected_output.append("chronological_timeline")
    if "TriageAgent" in ordered:
        expected_output.append("ranked_leads")
    if "ReportAgent" in ordered:
        expected_output.append("case_summary")
    if "LegalActionAgent" in ordered:
        expected_output.append("rendered_legal_document")

    needs_human_review = risk_level in ("MED", "CRITICAL")
    return InvestigationPlan(
        goal=goal,
        case_scope="inside_case_folder_only",
        confidence=confidence,
        needs_human_review=needs_human_review,
        safety_policy=SafetyPolicy(
            risk_level=risk_level,
            blur_by_default=blur_by_default,
            dual_authorization_required=dual_authorization_required,
        ),
        tasks=tasks,
        expected_output=expected_output,
        stop_conditions=[
            "no evidence entities extracted",
            "correlation returns no matches",
            "plan exhausted without sufficient evidence",
        ],
    )


class DeterministicPlanner:
    """Offline, explainable, keyword-based planner.

    Default backend. Keeps the system fully air-gapped while still producing
    goal-dependent plans. Swappable for an LLM backend through `PlannerLLM`.
    """

    def generate_plan(self, goal: str, state_summary: dict[str, Any] | None = None) -> InvestigationPlan:
        state_summary = state_summary or {}
        text = goal.lower()

        if _hit(text, LEGAL_KEYWORDS):
            return _build_plan(
                goal,
                ["LegalActionAgent", "ReportAgent"],
                state_summary,
                confidence=0.95,
            )

        tools = ["ExtractionAgent"]
        if _hit(text, TIMELINE_KEYWORDS):
            tools.append("TimelineAgent")
        if _hit(text, CORRELATE_KEYWORDS):
            tools.append("CorrelationAgent")
        if _hit(text, TRIAGE_KEYWORDS):
            tools.append("TriageAgent")
        if _hit(text, REPORT_KEYWORDS):
            tools.append("ReportAgent")

        risk_level = "LOW"
        blur = False
        dual = False
        if _hit(text, CRITICAL_KEYWORDS):
            risk_level, blur, dual = "CRITICAL", True, True
        elif _hit(text, SENSITIVE_KEYWORDS):
            risk_level, blur = "MED", True
            if "CorrelationAgent" not in tools:
                tools.append("CorrelationAgent")
            if "TriageAgent" not in tools:
                tools.append("TriageAgent")

        if len(tools) == 1:
            tools.extend(["CorrelationAgent", "TriageAgent", "ReportAgent"])
        else:
            if "ReportAgent" not in tools:
                tools.append("ReportAgent")
            if "CorrelationAgent" not in tools and (
                "TimelineAgent" in tools or "TriageAgent" in tools
            ):
                tools.append("CorrelationAgent")

        confidence = 0.8 if risk_level != "LOW" else 0.7
        return _build_plan(
            goal,
            tools,
            state_summary,
            risk_level=risk_level,
            blur_by_default=blur,
            dual_authorization_required=dual,
            confidence=confidence,
        )

    def replan(
        self,
        goal: str,
        state_summary: dict[str, Any] | None = None,
        previous_plan: InvestigationPlan | None = None,
    ) -> InvestigationPlan:
        state_summary = state_summary or {}
        current = [task.tool for task in previous_plan.tasks] if previous_plan else []
        tools = list(current)

        if state_summary.get("match_count", 0) == 0:
            if "TimelineAgent" not in tools:
                tools.append("TimelineAgent")
            if "CorrelationAgent" not in tools:
                tools.append("CorrelationAgent")
        if "ReportAgent" not in tools:
            tools.append("ReportAgent")

        if tools == current:
            tools = ["ReportAgent"]

        risk_level = "LOW"
        blur = False
        dual = False
        if previous_plan is not None:
            risk_level = previous_plan.safety_policy.risk_level
            blur = previous_plan.safety_policy.blur_by_default
            dual = previous_plan.safety_policy.dual_authorization_required
        return _build_plan(
            goal,
            tools,
            state_summary,
            risk_level=risk_level,
            blur_by_default=blur,
            dual_authorization_required=dual,
            confidence=0.7,
        )


_SYSTEM_PROMPT = (
    "You are the planning backend for Karuth, an offline, case-scoped "
    "investigation intelligence system. Interpret the investigator goal and "
    "select the minimal ordered tool list from: ExtractionAgent, "
    "CorrelationAgent, TimelineAgent, TriageAgent, ReportAgent, "
    "LegalActionAgent. Never invent evidence, never propose external URLs or "
    "network calls, never fabricate matches. Only produce the JSON plan.\n"
    "Safety policy: LOW is normal; MED requires blur_by_default true; CRITICAL "
    "requires dual_authorization_required true. Set needs_human_review true for "
    "MED or CRITICAL."
)


class OllamaPlanner:
    """Ollama-backed planner backend (requires ``langchain-ollama``).

    Lazily imports the integration so the package stays importable offline.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
    ) -> None:
        self._model = model or REASONING_MODEL
        self._base_url = base_url
        self._temperature = temperature
        self._llm: Any = None

    def _get_llm(self) -> Any:
        if self._llm is None:
            try:
                from langchain_ollama import ChatOllama
            except ImportError as exc:
                raise RuntimeError(
                    "OllamaPlanner requires 'langchain-ollama'. "
                    "Install it or use DeterministicPlanner for offline runs."
                ) from exc
            self._llm = ChatOllama(
                model=self._model,
                base_url=self._base_url,
                temperature=self._temperature,
            )
        return self._llm

    def generate_plan(self, goal: str, state_summary: dict[str, Any] | None = None) -> InvestigationPlan:
        return self._ask(goal, state_summary or {}, replan=False)

    def replan(
        self,
        goal: str,
        state_summary: dict[str, Any] | None = None,
        previous_plan: InvestigationPlan | None = None,
    ) -> InvestigationPlan:
        return self._ask(goal, state_summary or {}, replan=True, previous_plan=previous_plan)

    def _ask(
        self,
        goal: str,
        state_summary: dict[str, Any],
        *,
        replan: bool,
        previous_plan: InvestigationPlan | None = None,
    ) -> InvestigationPlan:
        from langchain_core.output_parsers import JsonOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        parser = JsonOutputParser()
        human = "goal: {goal}\ncase state summary: {state}\n"
        if replan:
            human += "previous plan (JSON): {previous_plan}\nReturn a REVISED minimal plan."
        else:
            human += "Return ONLY a minimal JSON plan."
        prompt = ChatPromptTemplate.from_messages(
            [("system", _SYSTEM_PROMPT), ("human", human)]
        )
        chain = prompt | self._get_llm() | parser
        raw = chain.invoke(
            {
                "goal": goal,
                "state": json.dumps(state_summary, sort_keys=True),
                "previous_plan": (
                    previous_plan.model_dump(mode="json") if previous_plan else None
                ),
            }
        )
        return InvestigationPlan.model_validate(raw)


def build_planner(backend: str = "deterministic", **kwargs: Any) -> PlannerLLM:
    """Factory for the planning backend; defaults to the offline backend."""
    if backend == "deterministic":
        return DeterministicPlanner()
    if backend == "ollama":
        return OllamaPlanner(**kwargs)
    raise ValueError(f"Unknown planner backend: {backend}")
