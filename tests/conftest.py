"""Shared pytest fixtures.

The suite is hermetic: no test may wait on (or depend on) live services.
Neo4j sync is disabled and LM Studio is blocked, so any code path that
accidentally reaches out fails fast instead of blocking for minutes on a real
model or a half-dead server. Tests that exercise the graph or the LLM
explicitly use mocks/fakes and their own configuration.
"""

from __future__ import annotations

import os

import pytest

# Disable best-effort Neo4j sync globally: deterministic correlation must not
# block on a graph server during tests (see karuth/graph/neo4j_client.py).
os.environ.setdefault("KARUTH_NEO4J_ENABLED", "0")

# Point LM Studio at an unreachable port as a second line of defence. The
# autouse fixture below makes any live client construction fail instantly, so
# default-backend paths fall back to DeterministicPlanner fast instead of
# auto-discovering a real loaded model and generating plans on CPU for
# minutes.
os.environ.setdefault("KARUTH_LMSTUDIO_BASE_URL", "http://127.0.0.1:59999/v1")
os.environ.setdefault("KARUTH_LMSTUDIO_MODEL", "")
os.environ.setdefault("KARUTH_VISION_MODEL", "")


@pytest.fixture(autouse=True)
def _block_live_llm(monkeypatch):
    """Make any attempt to construct a live OpenAI/LM Studio client fail fast.

    Tests that intentionally exercise the LLM inject fake clients, so this
    fixture only affects code paths that would otherwise hit the network.
    Failing fast (instead of retrying a dead server) is what keeps the suite
    fast and deterministic.
    """

    class _BlockedOpenAI:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("live LLM disabled in tests")

    monkeypatch.setattr("openai.OpenAI", _BlockedOpenAI, raising=False)
