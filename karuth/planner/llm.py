"""Replaceable planning backends for the agentic planner.

The planner uses a backend only for goal interpretation, task selection, and
replanning. Evidence extraction, correlation, triage, reporting, and legal
rendering stay deterministic — the LLM is never allowed to invent evidence,
modify extracted entities, perform matching, or bypass guardrails.

Backends
--------
- ``LMStudioPlanner``: the default. Talks to a local LM Studio server at an
  OpenAI-compatible REST endpoint (``http://localhost:1234/v1``) via the
  OpenAI Python SDK. LM Studio manages model execution; the planner never
  loads GGUF weights or uses transformers directly.
- ``DeterministicPlanner``: offline, explainable, keyword-based fallback,
  always available.
- ``FallbackPlanner``: wraps an LLM backend and falls back to
  ``DeterministicPlanner`` on any failure (timeout, invalid JSON, unavailable
  model, network error), so the agentic default never dead-ends.

New providers subclass ``_BaseChatPlanner``, implement ``_complete``, and are
registered in :func:`build_planner`.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from karuth.config import (
    LMSTUDIO_BASE_URL,
    LMSTUDIO_MAX_TOKENS,
    LMSTUDIO_MODEL,
    LMSTUDIO_TEMPERATURE,
    LMSTUDIO_TIMEOUT,
)
from karuth.planner.evidence import EvidenceItem, evidence_summary
from karuth.planner.schema import InvestigationPlan, PlannerTask, SafetyPolicy
from karuth.planner.tools import TOOL_REGISTRY, tool_metadata_text

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


def _hit(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


class PlannerLLM(Protocol):
    """Contract for any planner backend (deterministic or LLM).

    Backends may additionally expose ``backend_name`` (str) and
    ``is_available()`` (bool) as optional attributes; the planner reads them
    defensively with ``getattr`` so existing custom backends keep working.
    """

    def generate_plan(self, goal: str, state_summary: dict[str, Any]) -> InvestigationPlan:
        """Produce a structured plan for a goal."""

    def replan(
        self,
        goal: str,
        state_summary: dict[str, Any],
        previous_plan: InvestigationPlan | None = None,
    ) -> InvestigationPlan:
        """Produce a revised plan when the first pass is insufficient."""


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


def _reason_for(tool: str) -> str:
    """Audit-safe one-line reason for scheduling a tool, from its metadata."""
    spec = TOOL_REGISTRY.get(tool)
    if spec is None:
        return f"goal requires {tool}"
    return spec.description.split(".")[0].strip() + "."


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
                reason=_reason_for(tool),
                inputs=inputs,
                fields=fields,
            )
        )

    expected_output = ["investigation_plan_json", "ledger_audit_entries"]
    for tool in ordered:
        spec = TOOL_REGISTRY.get(tool)
        if spec is not None:
            expected_output.extend(output.label for output in spec.produced_outputs)

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
        reasoning_summary=(
            f"Selected {len(ordered)} deterministic tools ({', '.join(ordered)}) to satisfy "
            f"the goal; confidence {confidence:.2f}."
        ),
    )


class DeterministicPlanner:
    """Offline, explainable, keyword-based planner.

    Fallback backend: always available, keeps the system fully air-gapped
    while still producing goal-dependent plans. Used automatically whenever
    an LLM backend fails, or on demand via ``build_planner("deterministic")``.
    """

    backend_name = "deterministic"

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
            if "ExtractionAgent" not in tools:
                tools.insert(0, "ExtractionAgent")
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

    def is_available(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Semantic (LLM) planning backends
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
You are the planning backend for Karuth, an offline, case-scoped investigation \
intelligence system. You interpret the investigator's goal and return a STRICT \
JSON InvestigationPlan. You NEVER invent evidence, NEVER modify extracted \
entities, NEVER perform matching yourself, NEVER fabricate results, NEVER \
bypass guardrails or review authorization, and NEVER propose external URLs or \
network calls. You ONLY decide which deterministic tools to run, in what \
order, and when to stop.

AVAILABLE DETERMINISTIC TOOLS (all run locally; you only schedule them):
{tool_metadata}

EVIDENCE INVENTORY OF THE CASE FOLDER:
{evidence}

RULES:
- Schedule tools that can actually produce useful output for THIS case and \
THIS goal. Do not schedule a tool whose required state inputs cannot be \
produced, and do not schedule ExtractionAgent when the case folder holds no \
supported evidence files (chats/images).
- ExtractionAgent must precede CorrelationAgent, TimelineAgent, and \
TriageAgent. CorrelationAgent must precede TriageAgent and ReportAgent.
- Outputs from CorrelationAgent, TimelineAgent, TriageAgent, and ReportAgent \
may contain PII and require human review; set needs_human_review true for MED \
or CRITICAL risk.
- Safety policy: LOW is normal; MED requires blur_by_default true; CRITICAL \
requires dual_authorization_required true and needs_human_review true.
- If the goal is a legal document request, schedule LegalActionAgent with the \
correct template input ("section_94_bnss_notice" for notices / section 94, \
otherwise "section_63_bsa_certificate") plus ReportAgent. No evidence tools.
- Keep the tool list minimal and ordered. Never schedule a tool twice.

Return ONLY a JSON object matching this schema (no prose, no markdown fences):
{{
  "goal": "<the investigator goal verbatim>",
  "case_scope": "inside_case_folder_only",
  "confidence": <0.0 to 1.0>,
  "needs_human_review": <bool>,
  "safety_policy": {{
    "risk_level": "LOW" | "MED" | "CRITICAL",
    "blur_by_default": <bool>,
    "dual_authorization_required": <bool>
  }},
  "tasks": [
    {{
      "step": <int starting at 1>,
      "tool": "ExtractionAgent" | "CorrelationAgent" | "TimelineAgent" | \
"TriageAgent" | "ReportAgent" | "LegalActionAgent",
      "reason": "<audit-safe one-line reason>",
      "inputs": [],
      "fields": {{}}
    }}
  ],
  "expected_output": ["<labels of outputs each task produces>"],
  "stop_conditions": ["<conditions under which the investigation can stop>"],
  "reasoning_summary": "<concise, audit-safe explanation of the plan; never chain-of-thought>"
}}"""


def _build_messages(
    goal: str,
    state_summary: dict[str, Any],
    *,
    replan: bool,
    previous_plan: InvestigationPlan | None = None,
) -> list[dict[str, str]]:
    evidence = state_summary.get("available_evidence") or []
    evidence_text = _evidence_text(evidence)
    system = _SYSTEM_PROMPT_TEMPLATE.format(
        tool_metadata=tool_metadata_text(),
        evidence=evidence_text,
    )
    serializable = json.dumps(_json_safe(state_summary), sort_keys=True, default=str)
    user = f"Investigator goal: {goal}\n\nCurrent case state summary:\n{serializable}\n\n"
    if replan:
        previous = (
            json.dumps(previous_plan.model_dump(mode="json"), sort_keys=True, default=str)
            if previous_plan is not None
            else "none"
        )
        user += (
            "Previous plan (JSON):\n" + previous + "\n\n"
            "The previous plan was insufficient. Return a REVISED minimal "
            "InvestigationPlan JSON that fixes the problem, still respecting "
            "the rules above."
        )
    else:
        user += "Return ONLY the strict InvestigationPlan JSON."
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _evidence_text(evidence: Any) -> str:
    """Render the evidence inventory (dicts or EvidenceItems) for the prompt."""
    items: list[EvidenceItem] = []
    for item in evidence or []:
        if isinstance(item, EvidenceItem):
            items.append(item)
        elif isinstance(item, dict):
            try:
                items.append(EvidenceItem(**item))
            except (TypeError, ValueError):
                continue
    return evidence_summary(items)


def _json_safe(value: Any) -> Any:
    """Convert pydantic models/objects into JSON-serializable structures."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_json(raw: str) -> str:
    """Pull a JSON object out of an LLM response (fences, prose, etc.)."""
    text = raw.strip()
    if not text:
        raise ValueError("LLM returned an empty response")
    fenced = _FENCE_RE.findall(text)
    if fenced:
        text = fenced[0].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("LLM response contains no JSON object")
    return text[start : end + 1]


def _parse_plan_json(raw: str, goal: str) -> InvestigationPlan:
    """Strictly validate an LLM plan response into an InvestigationPlan."""
    payload = json.loads(_extract_json(raw))
    if not isinstance(payload, dict):
        raise ValueError("LLM plan JSON is not an object")
    if not payload.get("goal"):
        payload["goal"] = goal
    return InvestigationPlan.model_validate(payload)


class _BaseChatPlanner:
    """Shared plumbing for chat-completion planner backends.

    Subclasses implement :meth:`_complete` (turn a message list into the raw
    completion text) and set ``backend_name``. Plan/replan validation, the
    semantic prompt, and JSON repair live here so future providers stay thin.
    """

    backend_name = "chat"

    def __init__(self, model: str | None = None, temperature: float = 0.0) -> None:
        self._model = model or LMSTUDIO_MODEL
        self._temperature = temperature
        self.last_usage: dict[str, Any] | None = None

    def generate_plan(self, goal: str, state_summary: dict[str, Any] | None = None) -> InvestigationPlan:
        return self._ask(goal, state_summary or {}, replan=False)

    def replan(
        self,
        goal: str,
        state_summary: dict[str, Any] | None = None,
        previous_plan: InvestigationPlan | None = None,
    ) -> InvestigationPlan:
        return self._ask(goal, state_summary or {}, replan=True, previous_plan=previous_plan)

    def is_available(self) -> bool:
        return True

    def _ask(
        self,
        goal: str,
        state_summary: dict[str, Any],
        *,
        replan: bool,
        previous_plan: InvestigationPlan | None = None,
    ) -> InvestigationPlan:
        messages = _build_messages(goal, state_summary, replan=replan, previous_plan=previous_plan)
        raw = self._complete(messages)
        return _parse_plan_json(raw, goal)

    def _complete(self, messages: list[dict[str, str]]) -> str:
        raise NotImplementedError


class LMStudioPlanner(_BaseChatPlanner):
    """LM Studio-backed planner via the OpenAI-compatible REST endpoint.

    Communicates with ``base_url`` (default ``http://localhost:1234/v1``) using
    the OpenAI Python SDK. LM Studio manages model execution; this class never
    loads weights or uses transformers. All settings are configurable through
    constructor kwargs (defaults come from ``karuth.config``), so switching
    models or servers requires no planner code changes.
    """

    backend_name = "lmstudio"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        api_key: str = "lm-studio",
        client: Any = None,
    ) -> None:
        super().__init__(
            model=model or LMSTUDIO_MODEL,
            temperature=LMSTUDIO_TEMPERATURE if temperature is None else temperature,
        )
        self._base_url = base_url or LMSTUDIO_BASE_URL
        self._max_tokens = max_tokens or LMSTUDIO_MAX_TOKENS
        self._timeout = LMSTUDIO_TIMEOUT if timeout is None else timeout
        self._api_key = api_key
        self._client = client  # injectable for tests; avoids importing openai

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self._base_url,
                api_key=self._api_key,
                timeout=self._timeout,
            )
        return self._client

    def _complete(self, messages: list[dict[str, str]]) -> str:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            timeout=self._timeout,
        )
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.last_usage = {
                key: getattr(usage, key)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                if hasattr(usage, key)
            }
        content = response.choices[0].message.content
        return content if isinstance(content, str) else json.dumps(content)


class FallbackPlanner:
    """Wrap a primary planner; delegate to a deterministic planner on failure.

    Catches timeouts, invalid JSON, unavailable models, and any other planner
    exception, records the fallback, and produces a deterministic plan so the
    investigation never dead-ends because the LLM backend is down.
    """

    backend_name = "fallback"

    def __init__(self, primary: PlannerLLM, fallback: PlannerLLM | None = None) -> None:
        self._primary = primary
        self._fallback = fallback or DeterministicPlanner()
        self.last_error: str | None = None
        self.last_backend: str | None = None  # which backend actually produced the last plan

    @property
    def primary(self) -> PlannerLLM:
        return self._primary

    @property
    def fallback(self) -> PlannerLLM:
        return self._fallback

    @property
    def _model(self) -> str:
        return getattr(self._primary, "_model", "")

    @property
    def last_usage(self) -> dict[str, Any] | None:
        return getattr(self._primary, "last_usage", None)

    def generate_plan(self, goal: str, state_summary: dict[str, Any] | None = None) -> InvestigationPlan:
        try:
            plan = self._primary.generate_plan(goal, state_summary)
            self.last_backend = getattr(self._primary, "backend_name", "primary")
            return plan
        except Exception as exc:
            self.last_error = str(exc)
            self.last_backend = getattr(self._fallback, "backend_name", "deterministic")
            return self._fallback.generate_plan(goal, state_summary)

    def replan(
        self,
        goal: str,
        state_summary: dict[str, Any] | None = None,
        previous_plan: InvestigationPlan | None = None,
    ) -> InvestigationPlan:
        try:
            plan = self._primary.replan(goal, state_summary, previous_plan)
            self.last_backend = getattr(self._primary, "backend_name", "primary")
            return plan
        except Exception as exc:
            self.last_error = str(exc)
            self.last_backend = getattr(self._fallback, "backend_name", "deterministic")
            return self._fallback.replan(goal, state_summary, previous_plan)

    def is_available(self) -> bool:
        return True


def build_planner(backend: str = "lmstudio", *, fallback: bool = True, **kwargs: Any) -> PlannerLLM:
    """Factory for the planning backend; defaults to the LLM backend.

    The default experience is LLM-backed planning (LM Studio by default);
    ``fallback=True`` (default) wraps LLM backends in :class:`FallbackPlanner`
    so a timeout, unavailable model, or invalid JSON response degrades to the
    deterministic planner automatically. Pass ``backend="deterministic"`` to
    force the offline keyword-based planner.

    ``backend`` may be ``"lmstudio"`` or ``"deterministic"``.
    """
    if backend == "deterministic":
        return DeterministicPlanner()
    if backend == "lmstudio":
        return FallbackPlanner(LMStudioPlanner(**kwargs)) if fallback else LMStudioPlanner(**kwargs)
    raise ValueError(f"Unknown planner backend: {backend}")
