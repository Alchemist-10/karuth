"""Replaceable planning backends for the agentic planner.

The planner uses a backend only for goal interpretation, task selection, and
replanning. Evidence extraction, correlation, triage, reporting, and legal
rendering stay deterministic — the LLM is never allowed to invent evidence,
modify extracted entities, perform matching, or bypass guardrails.

Backends
--------
- ``OpenRouterPlanner``: the default. Talks to OpenRouter's OpenAI-compatible
  REST endpoint (``https://openrouter.ai/api/v1``) via the OpenAI Python SDK,
  which runs models on remote GPUs (fast, no local model execution). This is
  opt-in-by-default: the API key comes from the ``OPENROUTER_API_KEY``
  environment variable, the default model is a free-tier model, and anything
  sent to OpenRouter is the case *state summary* (never raw evidence files
  beyond extracted values).
- ``LMStudioPlanner``: fully offline alternative. Talks to a local LM Studio
  server at an OpenAI-compatible REST endpoint (``http://localhost:1234/v1``)
  via the OpenAI Python SDK. LM Studio manages model execution; the planner
  never loads GGUF weights or uses transformers directly. Use this when
  evidence must never leave the machine.
- ``DeterministicPlanner``: offline, explainable, keyword-based fallback,
  always available.
- ``FallbackPlanner``: wraps an LLM backend and falls back to
  ``DeterministicPlanner`` on any failure (timeout, invalid JSON, unavailable
  model, network error), so the agentic default never dead-ends.

New providers subclass ``_BaseChatPlanner``, implement ``_complete``, and are
registered in :func:`build_planner`.

LLM backends stream completions and capture the model's reasoning tokens into
``last_thoughts`` (forwarding them to a ``thoughts_callback`` when one is
configured) so the UI can render the thinking live, xAI-style. Reasoning is
**display-only** -- it is never written to the ledger, which keeps only
concise, audit-safe explanations.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Iterable
from typing import Any, Protocol

from karuth.config import (
    LMSTUDIO_BASE_URL,
    LMSTUDIO_MAX_TOKENS,
    LMSTUDIO_MODEL,
    LMSTUDIO_TEMPERATURE,
    LMSTUDIO_TIMEOUT,
    OPENROUTER_BASE_URL,
    OPENROUTER_MAX_TOKENS,
    OPENROUTER_MODEL,
    OPENROUTER_TEMPERATURE,
    OPENROUTER_TIMEOUT,
)
from karuth.planner.actions import ActionDecision, DeterministicActionPlanner
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

    def decide_action(self, goal: str, context: dict[str, Any]) -> ActionDecision:
        """Pick the single next action in the dynamic loop (optional).

        Backends that do not implement dynamic decisions (e.g. minimal test
        stubs) may omit this; the graph falls back to the deterministic
        policy via :class:`~karuth.planner.actions.DeterministicActionPlanner`.
        """


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

    def decide_action(self, goal: str, context: dict[str, Any]) -> ActionDecision:
        """Offline rule-based next-action selection (always available)."""
        return DeterministicActionPlanner().decide_action(goal, context)

    def replan(
        self,
        goal: str,
        state_summary: dict[str, Any] | None = None,
        previous_plan: InvestigationPlan | None = None,
    ) -> InvestigationPlan:
        state_summary = state_summary or {}
        current = [task.tool for task in previous_plan.tasks] if previous_plan else []
        tools = list(current)

        # Drop tools that already failed once; re-scheduling them only repeats
        # the failure. The plan converges on the remaining tools and finalizes.
        failed = set(state_summary.get("failed_tools", []) or [])
        if failed:
            tools = [tool for tool in tools if tool not in failed]
        if not tools:
            tools = ["ReportAgent"]

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
- VisionAgent (LLM image transcription) is optional and only useful when the \
case folder contains images AND the LM Studio vision model is reachable; its \
output is LLM-generated and requires review. Never treat its output as \
extracted evidence.
- Never re-schedule a tool listed under failed_tools in the case state summary.
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
"TriageAgent" | "ReportAgent" | "LegalActionAgent" | "VisionAgent",
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


# ---------------------------------------------------------------------------
# Dynamic action selection (one next action after each tool execution)
# ---------------------------------------------------------------------------

_ACTION_SYSTEM_PROMPT_TEMPLATE = """\
You are the dynamic action-selection backend for Karuth, an offline, \
case-scoped investigation system. After every tool execution you choose \
exactly ONE next action from the current investigation context. You never \
invent evidence, never create correlation matches, never modify extracted \
entities, never fabricate results, and never bypass guardrails or review \
authorization. You only decide *what to do next*.

AVAILABLE DETERMINISTIC TOOLS (all run locally; you only schedule them):
{tools}

CURRENT INVESTIGATION CONTEXT (JSON):
{context}

RULES (investigator-safe order — you may not violate these):
- The safe order is: extract evidence -> inspect multimodal observations \
(ObservationAgent when images exist) -> correlate entities -> build timeline \
-> generate competing hypotheses -> hunt contradictions -> challenge a \
specific hypothesis (SkepticReview) -> re-rank the strongest leads \
(TriageAgent) -> ask for human confirmation when required -> finalize.
- NEVER select ReportAgent. The final evidence-linked report is generated \
only by the finalize step, after contradiction and skeptic review.
- ``run_tool`` executes a deterministic tool listed above whose required \
state inputs already exist (ExtractionAgent needs no inputs; ObservationAgent \
needs images; CorrelationAgent needs extracted entities; TimelineAgent needs \
entities; ContradictionHunter needs entities and matches; SkepticReview needs \
hypotheses; TriageAgent needs matches).
- ``generate_hypothesis`` proposes competing hypotheses; never fabricate \
evidence references (``proposed_hypothesis.evidence_refs`` must be real ids \
from the context).
- ``investigate_hypothesis`` runs a tool against ``target_hypothesis_id``.
- ``challenge_hypothesis`` runs the skeptic (SkepticReview) to try to \
disprove ``target_hypothesis_id``; it must always target a real hypothesis id \
and must only be chosen after ContradictionHunter has run when contradictions \
exist.
- After a skeptic review that changes confidence, re-run TriageAgent to \
re-rank the strongest leads before finalizing.
- ``MvdRegistryLookup`` is approval-gated: schedule it only when the context \
shows ``registry_lookup_approved=true``; otherwise never select it (the \
router will refuse). It returns synthetic fixture data from a mock registry \
and registered ownership never proves who operated the vehicle.
- ``ask_human`` requests investigator input when contradictions are \
unresolved at CRITICAL risk, or when review authorization is required.
- ``finalize`` stops; set ``stop_reason`` to one of: sufficient_evidence, \
unresolved_contradictions, unresolved_uncertainty, awaiting_human_confirmation, \
activity_interpretation_requires_review, sensitive_content_gated, \
action_budget_exhausted, no_information_gain. Never emit \
sufficient_evidence while unresolved contradictions, an unconfirmed identity, \
or an unreviewed activity interpretation remain.
- Prefer tools that add information; never rerun a completed tool that would \
produce no new facts. Set ``expected_information_gain`` accordingly (0.0-1.0).
- ``rationale`` must be audit-safe and concise (never chain-of-thought).
- Never propose external URLs or anything outside the case folder.

Return ONLY a JSON object matching this schema (no prose, no markdown fences):
{{
  "action_type": "run_tool" | "investigate_hypothesis" | "generate_hypothesis" | \
"challenge_hypothesis" | "ask_human" | "finalize",
  "tool": "<tool name or empty>",
  "arguments": {{}},
  "target_hypothesis_id": "<id or empty>",
  "rationale": "<audit-safe one-line rationale>",
  "expected_information_gain": <0.0 to 1.0>,
  "confidence": <0.0 to 1.0>,
  "stop_reason": "<when finalize>",
  "requires_human_review": <bool>
}}"""


def _build_action_messages(goal: str, context: dict[str, Any]) -> list[dict[str, str]]:
    system = _ACTION_SYSTEM_PROMPT_TEMPLATE.format(
        tools=tool_metadata_text(),
        context=json.dumps(_json_safe(context), sort_keys=True, default=str),
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Investigator goal: {goal}\nChoose the single next action."},
    ]


def _parse_action_json(raw: str, goal: str) -> ActionDecision:
    """Strictly validate an LLM action decision into an ActionDecision."""
    payload = json.loads(_extract_json(raw))
    if not isinstance(payload, dict):
        raise ValueError("LLM action JSON is not an object")
    return ActionDecision.model_validate(payload)


# ---------------------------------------------------------------------------
# Streaming completion helpers (live "thinking" capture)
# ---------------------------------------------------------------------------


def _delta_reasoning(delta: Any) -> str:
    """Extract reasoning text from a streamed delta chunk.

    OpenAI-style models emit ``delta.reasoning``; DeepSeek/Qwen/OpenRouter
    reasoning models emit ``delta.reasoning_content``. Returns ``""`` when the
    chunk carries no reasoning, so plain models simply produce no thoughts.
    """
    for attr in ("reasoning_content", "reasoning"):
        value = getattr(delta, attr, None)
        if isinstance(value, str) and value:
            return value
    return ""


def _message_reasoning(message: Any) -> str:
    """Extract reasoning text from a non-streamed message, if the model emits it."""
    for attr in ("reasoning_content", "reasoning"):
        value = getattr(message, attr, None)
        if isinstance(value, str) and value:
            return value
    return ""


def _usage_dict(usage: Any) -> dict[str, int] | None:
    if usage is None:
        return None
    captured = {
        key: getattr(usage, key)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        if hasattr(usage, key)
    }
    return captured or None


def _is_stream_response(response: Any) -> bool:
    """True when ``create()`` returned a streamed (iterable) response."""
    return isinstance(response, Iterable) and not isinstance(
        response, (str, bytes, dict, list, tuple)
    )


def _finish_non_streaming_response(
    response: Any,
    *,
    on_thought: Callable[[str], None] | None = None,
) -> tuple[str, list[str], dict[str, int] | None]:
    """Read (content, thoughts, usage) from a non-streamed response."""
    try:
        message = response.choices[0].message
    except (AttributeError, IndexError, TypeError):
        return "", [], None
    content = message.content if isinstance(message.content, str) else json.dumps(message.content)
    reasoning = _message_reasoning(message)
    thoughts = [reasoning] if reasoning else []
    if on_thought is not None and reasoning:
        on_thought(reasoning)
    return content, thoughts, _usage_dict(getattr(response, "usage", None))


def _consume_stream(
    response: Iterable[Any],
    *,
    on_thought: Callable[[str], None] | None = None,
) -> tuple[str, list[str], dict[str, int] | None]:
    """Fold a streamed response into (content, thoughts, usage).

    Reasoning tokens are appended to ``thoughts`` and forwarded to
    ``on_thought`` as they arrive, so the UI can render the model's thinking
    live. Content tokens accumulate into the final completion text.
    """
    content: list[str] = []
    thoughts: list[str] = []
    usage: dict[str, int] | None = None
    for chunk in response:
        chunk_usage = getattr(chunk, "usage", None)
        if chunk_usage is not None:
            usage = _usage_dict(chunk_usage)
        choices = getattr(chunk, "choices", None)
        if not choices:
            continue
        delta = choices[0].delta
        if delta is None:
            continue
        reasoning = _delta_reasoning(delta)
        if reasoning:
            thoughts.append(reasoning)
            if on_thought is not None:
                on_thought(reasoning)
        part = getattr(delta, "content", None)
        if isinstance(part, str) and part:
            content.append(part)
    return "".join(content), thoughts, usage


def _stream_completion(
    *,
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: float,
    extra_headers: dict[str, str] | None = None,
    on_thought: Callable[[str], None] | None = None,
) -> tuple[str, list[str], dict[str, int] | None]:
    """Call ``chat.completions.create``, returning (content, thoughts, usage).

    Requests ``stream=True`` so reasoning tokens can be surfaced live. Any
    server or injected client that cannot stream falls back to a plain
    non-streaming call, so existing fakes and older backends keep working;
    a streamed response whose ``choices`` are missing is treated as a usage
    tick. Raises only when both streaming and non-streaming fail.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout,
        "stream": True,
    }
    if extra_headers:
        kwargs["extra_headers"] = extra_headers

    # Try progressively safer call modes: streaming with usage (modern servers),
    # plain streaming (older servers), then non-streaming (no streaming support).
    response = None
    attempts = (
        {**kwargs, "stream_options": {"include_usage": True}},
        kwargs,
        {key: value for key, value in kwargs.items() if key != "stream"},
    )
    for attempt in attempts:
        try:
            response = client.chat.completions.create(**attempt)
            break
        except Exception:
            continue
    if response is None:
        raise RuntimeError("chat completion unavailable (all call modes failed)")

    if _is_stream_response(response):
        return _consume_stream(response, on_thought=on_thought)
    return _finish_non_streaming_response(response, on_thought=on_thought)


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
        self.last_thoughts: list[str] = []  # reasoning tokens from the last call (display only)
        self._thoughts_callback: Callable[[str], None] | None = None

    def _resolve_model(self) -> str:
        """Return the model to chat with; subclasses may auto-discover it."""
        return self._model or ""

    def _chat_complete(self, messages: list[dict[str, str]]) -> str:
        """Stream a completion and capture reasoning tokens for the UI.

        Returns the final completion text unchanged; reasoning tokens are
        accumulated in ``self.last_thoughts`` and, when a ``thoughts_callback``
        was configured, forwarded as they arrive so the UI can render the
        model's thinking live (xAI-style). The reasoning is display-only and
        is never written to the ledger.
        """
        content, thoughts, usage = _stream_completion(
            client=self._get_client(),
            model=self._resolve_model(),
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            timeout=self._timeout,
            extra_headers=getattr(self, "_extra_headers", None),
            on_thought=self._thoughts_callback,
        )
        self.last_usage = usage
        self.last_thoughts = list(thoughts)
        return content

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

    def decide_action(self, goal: str, context: dict[str, Any]) -> ActionDecision:
        """Ask the LLM for the single next dynamic action (strict JSON)."""
        messages = _build_action_messages(goal, context)
        raw = self._complete(messages)
        return _parse_action_json(raw, goal)

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
        plan = _parse_plan_json(raw, goal)
        if replan:
            # Safety net: never re-schedule a tool that already failed this run.
            failed = set(state_summary.get("failed_tools", []) or [])
            kept = [task for task in plan.tasks if task.tool not in failed]
            if kept and len(kept) < len(plan.tasks):
                plan.tasks = kept
        return plan

    def _complete(self, messages: list[dict[str, str]]) -> str:
        raise NotImplementedError


_EMBEDDING_HINTS = ("embed", "embedding", "bge-", "e5-", "nomic")


def _discover_chat_model(client: Any, preferred: str | None = None) -> str | None:
    """Pick the first loaded chat-capable model from ``GET /v1/models``.

    LM Studio reports every loaded model (embeddings included). Embedding-only
    models cannot chat, so they are skipped. A preferred id that is actually
    loaded wins; otherwise the first non-embedding model is returned. Returns
    ``None`` when the server is unreachable or only embedding models are
    loaded, so callers can fall back to their configured model name.
    """
    try:
        models = client.models.list()
    except Exception:
        return None
    loaded = []
    for model in models:
        model_id = getattr(model, "id", None)
        if model_id and not any(hint in model_id.lower() for hint in _EMBEDDING_HINTS):
            loaded.append(model_id)
    if not loaded:
        return None
    if preferred and preferred in loaded:
        return preferred
    return loaded[0]


class LMStudioPlanner(_BaseChatPlanner):
    """LM Studio-backed planner via the OpenAI-compatible REST endpoint.

    Communicates with ``base_url`` (default ``http://localhost:1234/v1``) using
    the OpenAI Python SDK. LM Studio manages model execution; this class never
    loads weights or uses transformers. All settings are configurable through
    constructor kwargs (defaults come from ``karuth.config``), so switching
    models or servers requires no planner code changes.

    When no explicit model is configured (``LMSTUDIO_MODEL`` empty or
    ``model=None``), the first loaded chat-capable model is auto-discovered
    from ``GET /v1/models`` on first use, so a fresh LM Studio install with a
    different model still works without editing config.
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
        thoughts_callback: Callable[[str], None] | None = None,
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
        self._resolved_model: str | None = None
        self._thoughts_callback = thoughts_callback

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self._base_url,
                api_key=self._api_key,
                timeout=self._timeout,
                max_retries=0,  # a dead server must fail fast, not retry
            )
        return self._client

    def _resolve_model(self) -> str:
        """Return the model to use, auto-discovering when none is configured."""
        if self._resolved_model:
            return self._resolved_model
        configured = self._model or ""
        if configured:
            return configured
        discovered = _discover_chat_model(self._get_client())
        if discovered:
            self._resolved_model = discovered
            return discovered
        return ""

    def _complete(self, messages: list[dict[str, str]]) -> str:
        return self._chat_complete(messages)


class OpenRouterPlanner(_BaseChatPlanner):
    """OpenRouter-backed planner via its OpenAI-compatible REST endpoint.

    OPT-IN ONLY: this backend sends the case state summary (extracted entity
    values, match counts, evidence descriptions) to a third-party service. It
    must never be used for real casework where evidence must stay on the
    machine; LM Studio remains the offline default. The API key is read from
    the ``OPENROUTER_API_KEY`` environment variable (never stored in code or
    the ledger). Constructor kwargs override config defaults, so switching
    models or servers needs no code changes.
    """

    backend_name = "openrouter"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        api_key: str | None = None,
        client: Any = None,
        thoughts_callback: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(
            model=model or OPENROUTER_MODEL,
            temperature=OPENROUTER_TEMPERATURE if temperature is None else temperature,
        )
        self._base_url = base_url or OPENROUTER_BASE_URL
        self._max_tokens = max_tokens or OPENROUTER_MAX_TOKENS
        self._timeout = OPENROUTER_TIMEOUT if timeout is None else timeout
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self._client = client  # injectable for tests; avoids importing openai
        self._thoughts_callback = thoughts_callback
        self._extra_headers = {"HTTP-Referer": "https://karuth.local", "X-Title": "Karuth"}

    def _get_client(self) -> Any:
        if self._client is None:
            if not self._api_key:
                raise RuntimeError(
                    "OPENROUTER_API_KEY is not set; refusing to call OpenRouter without a key"
                )
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self._base_url,
                api_key=self._api_key,
                timeout=self._timeout,
                max_retries=0,  # fail fast; FallbackPlanner handles errors
            )
        return self._client

    def _complete(self, messages: list[dict[str, str]]) -> str:
        return self._chat_complete(messages)


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

    @property
    def last_thoughts(self) -> list[str]:
        """Reasoning tokens captured from the primary LLM backend (display only)."""
        return list(getattr(self._primary, "last_thoughts", []) or [])

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

    def decide_action(self, goal: str, context: dict[str, Any]) -> ActionDecision:
        """Pick the next action, falling back to the deterministic policy."""
        try:
            decision = self._primary.decide_action(goal, context)
            self.last_backend = getattr(self._primary, "backend_name", "primary")
            return decision
        except Exception as exc:
            self.last_error = str(exc)
            self.last_backend = getattr(self._fallback, "backend_name", "deterministic")
            return self._fallback.decide_action(goal, context)

    def is_available(self) -> bool:
        return True


def build_planner(backend: str = "openrouter", *, fallback: bool = True, **kwargs: Any) -> PlannerLLM:
    """Factory for the planning backend; defaults to the OpenRouter LLM.

    The default experience is LLM-backed planning (OpenRouter by default,
    which runs models remotely so local planning is fast); ``fallback=True``
    (default) wraps LLM backends in :class:`FallbackPlanner` so a timeout,
    unavailable model, or invalid JSON response degrades to the deterministic
    planner automatically. Pass ``backend="deterministic"`` to force the
    offline keyword-based planner, or ``backend="lmstudio"`` for the fully
    offline LLM backend.

    ``backend`` may be ``"openrouter"`` (default), ``"lmstudio"`` (offline
    LLM), or ``"deterministic"``.

    Extra ``kwargs`` (e.g. ``model``, ``base_url``, ``thoughts_callback``) are
    passed through to the backend constructor. ``thoughts_callback`` receives
    reasoning tokens as they stream in, letting a UI render the model's
    thinking live; the tokens are also captured on ``last_thoughts``.
    """
    if backend == "deterministic":
        return DeterministicPlanner()
    if backend == "lmstudio":
        return FallbackPlanner(LMStudioPlanner(**kwargs)) if fallback else LMStudioPlanner(**kwargs)
    if backend == "openrouter":
        return FallbackPlanner(OpenRouterPlanner(**kwargs)) if fallback else OpenRouterPlanner(**kwargs)
    raise ValueError(f"Unknown planner backend: {backend}")
