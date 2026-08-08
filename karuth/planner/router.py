"""Route abstract planner tasks to concrete deterministic modules.

The planner decides *which* tools to run; this router binds each abstract task
to the existing deterministic evidence-processing modules. No LLM reasoning
happens here.
"""

from __future__ import annotations

import re
from pathlib import Path

from karuth.agents.action_agent import fill_legal_document
from karuth.agents.correlation_agent import correlate_entities
from karuth.agents.extraction_agent import extract_case_entities
from karuth.agents.report_agent import summarize_matches
from karuth.agents.timeline_agent import order_chronologically
from karuth.agents.triage_agent import rank_leads
from karuth.planner.schema import PlannerTask
from karuth.planner.state import PlannerState

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "legal" / "templates"

_PHONE_RE = re.compile(r"\+?\d{10,15}")
_FACE_RE = re.compile(r"Person_Cluster_[A-Z0-9_]+")


def route_task(task: PlannerTask, state: PlannerState) -> tuple[dict[str, object], str]:
    """Execute one planner task and return (state updates, human summary)."""
    tool = task.tool
    case_folder = state["case_folder"]

    if tool == "ExtractionAgent":
        entities = extract_case_entities(case_folder)
        return {"entities": entities}, f"extracted {len(entities)} entities"

    if tool == "CorrelationAgent":
        matches = correlate_entities(list(state.get("entities", [])))
        return {"matches": matches}, f"found {len(matches)} matches"

    if tool == "TimelineAgent":
        timeline = order_chronologically(list(state.get("entities", [])))
        return {"timeline": timeline}, f"ordered {len(timeline)} entities"

    if tool == "TriageAgent":
        ranked = rank_leads(list(state.get("matches", [])), state.get("goal", ""))
        return {"ranked_matches": ranked}, f"ranked {len(ranked)} leads"

    if tool == "ReportAgent":
        report = summarize_matches(list(state.get("matches", [])))
        return {"report": report}, "generated case summary"

    if tool == "LegalActionAgent":
        template_name = task.inputs[0] if task.inputs else ""
        document = fill_legal_document(TEMPLATE_DIR, template_name, task.fields)
        return {"legal_output": document}, f"rendered {template_name}"

    raise ValueError(f"Unsupported planner tool: {tool}")


def mask_report(report: str, risk_level: str) -> str:
    """Mask sensitive values for MED/CRITICAL default-blurred previews."""
    if risk_level == "LOW":
        return report
    masked = _PHONE_RE.sub("[REDACTED PHONE]", report)
    masked = _FACE_RE.sub("[REDACTED FACE CLUSTER]", masked)
    return masked
