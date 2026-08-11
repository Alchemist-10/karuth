"""API tests for the agentic planner routes (``/planner/*``).

Covers the backends listing, running a prompt through the LLM-first default
(which degrades to the deterministic fallback), the explicit deterministic
backend, and guardrail rejections.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from scripts.generate_synthetic_case import generate_case
from karuth.api.main import app
from karuth.api.routes.planner import DEFAULT_DATA_DIR
from karuth.planner.llm import FallbackPlanner


class _FailingPlanner:
    """A backend that is never available, to exercise the fallback path."""

    backend_name = "failing"

    def generate_plan(self, goal, state_summary=None):
        raise RuntimeError("model unavailable")

    def replan(self, goal, state_summary=None, previous_plan=None):
        raise RuntimeError("model unavailable")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr("karuth.api.routes.planner.DEFAULT_DATA_DIR", tmp_path)
    return TestClient(app)


def _seed_case(tmp_path: object) -> str:
    case_folder = DEFAULT_DATA_DIR / "CASE-API"
    generate_case(case_folder)
    return "CASE-API"


def test_backends_listing(client):
    resp = client.get("/planner/backends")
    assert resp.status_code == 200
    body = resp.json()
    assert body["default"] == "openrouter"
    assert set(body["backends"]) == {"lmstudio", "openrouter", "deterministic"}


def test_planner_goal_runs_deterministic_backend(client, tmp_path):
    case_id = _seed_case(tmp_path)
    resp = client.post(
        "/planner/goal",
        json={
            "case_id": case_id,
            "goal": "Find links for vehicle KL07CX4242 and triage the top leads",
            "backend": "deterministic",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["backend"] == "deterministic"
    assert body["backend_used"] == "deterministic"
    assert body["final_result"]
    assert body["plan"]["tasks"]
    tools = {task["tool"] for task in body["plan"]["tasks"]}
    assert "ExtractionAgent" in tools
    assert body["completed_steps"]
    assert body["execution_history"]
    assert body["ledger_entries"] > 0
    assert body["ledger_verified"] is True


def test_planner_goal_default_is_llm_first_and_falls_back(client, tmp_path, monkeypatch):
    case_id = _seed_case(tmp_path)
    monkeypatch.setattr(
        "karuth.api.routes.planner.build_planner",
        lambda backend="openrouter", **kwargs: FallbackPlanner(_FailingPlanner()),
    )
    resp = client.post(
        "/planner/goal",
        json={"case_id": case_id, "goal": "Find links for vehicle KL07CX4242"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["backend"] == "openrouter"  # the requested (default) backend
    assert body["backend_used"] == "deterministic"  # produced by the fallback
    assert body["final_result"]
    assert "ExtractionAgent" in {task["tool"] for task in body["plan"]["tasks"]}


def test_planner_goal_rejects_unknown_backend(client):
    resp = client.post(
        "/planner/goal",
        json={"case_id": "CASE-1", "goal": "anything", "backend": "bogus"},
    )
    assert resp.status_code == 400
    assert "Unknown backend" in resp.json()["detail"]


def test_planner_goal_rejects_unsafe_case_id(client):
    resp = client.post(
        "/planner/goal",
        json={"case_id": "../escape", "goal": "anything"},
    )
    assert resp.status_code == 400
    assert "plain identifier" in resp.json()["detail"]


def test_planner_goal_guardrail_rejects_external_url(client, tmp_path):
    _seed_case(tmp_path)
    resp = client.post(
        "/planner/goal",
        json={"case_id": "CASE-API", "goal": "https://example.com/exfil"},
    )
    assert resp.status_code == 400
    assert "rejected" in resp.json()["detail"]


def test_planner_goal_requires_goal_and_case_id(client):
    resp = client.post("/planner/goal", json={"case_id": "CASE-1"})
    assert resp.status_code == 422
    resp = client.post("/planner/goal", json={"goal": "anything"})
    assert resp.status_code == 422
