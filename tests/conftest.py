"""Shared fixtures and e2e skip logic for token_analysis tests."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import pytest

# ── Known trace IDs for e2e tests ──────────────────────────────────────────
RECRUITER_TRACE_ID = "deca41fd2e57cc3fec2b438b73979c31"
COLOMBIA_TRACE_ID = "973d582dd28b3e2085ae9729beabf37d"
BRAZIL_TRACE_ID = "8c2194b0bf69154862e28c0212e130b9"


# ── Auto-skip e2e tests unless TOKEN_ANALYSIS_E2E=1 ───────────────────────


def pytest_collection_modifyitems(config, items):
    """Auto-skip @pytest.mark.e2e tests unless TOKEN_ANALYSIS_E2E=1."""
    if os.environ.get("TOKEN_ANALYSIS_E2E", "").strip() == "1":
        return  # Run everything
    skip_e2e = pytest.mark.skip(reason="Set TOKEN_ANALYSIS_E2E=1 to run e2e tests")
    for item in items:
        if "e2e" in item.keywords:
            item.add_marker(skip_e2e)


# ── Live ClickHouse client (session scope) ─────────────────────────────────


@pytest.fixture(scope="session")
def ch_client():
    """Live ClickHouse client — smoke-tests with SELECT 1."""
    from token_analysis import get_client

    client = get_client()
    # Smoke test
    result = client.query("SELECT 1")
    assert result.result_rows[0][0] == 1
    return client


# ── Mock span attribute dicts ──────────────────────────────────────────────


@pytest.fixture()
def openai_span_attrs() -> dict[str, str]:
    """Mock span attributes from an OpenAI-style span with reported tokens.

    4 prompt messages: system, user, assistant with tool_call, tool response.
    Tokens: input=1500, output=350, cache_read=200.
    """
    return {
        "gen_ai.request.model": "azure/gpt-4o",
        "gen_ai.response.model": "gpt-4o-2024-05-13",
        "gen_ai.usage.input_tokens": "1500",
        "gen_ai.usage.output_tokens": "350",
        "gen_ai.usage.cache_read_input_tokens": "200",
        "gen_ai.agent.name": "test_agent",
        "gen_ai.completion.0.content": "Sure, I can help with that.",
        "gen_ai.completion.0.finish_reason": "stop",
        # Message 0: system
        "gen_ai.prompt.0.role": "system",
        "gen_ai.prompt.0.content": "You are a helpful assistant.",
        # Message 1: user
        "gen_ai.prompt.1.role": "user",
        "gen_ai.prompt.1.content": "What is the weather in SF?",
        # Message 2: assistant with tool call
        "gen_ai.prompt.2.role": "assistant",
        "gen_ai.prompt.2.content": "",
        "gen_ai.prompt.2.tool_calls.0.name": "get_weather",
        "gen_ai.prompt.2.tool_calls.0.arguments": '{"location": "San Francisco"}',
        # Message 3: tool response
        "gen_ai.prompt.3.role": "tool",
        "gen_ai.prompt.3.content": '{"temperature": 65, "condition": "foggy"}',
        "gen_ai.prompt.3.tool_call_id": "call_abc123",
    }


@pytest.fixture()
def traceloop_span_attrs() -> dict[str, str]:
    """Mock span attributes from a traceloop-instrumented span with no token counts.

    Has traceloop.entity.input JSON but no gen_ai.usage.* attributes.
    """
    return {
        "gen_ai.request.model": "azure/gpt-4o",
        "gen_ai.response.model": "gpt-4o-2024-05-13",
        "traceloop.entity.input": '{"messages": [{"role": "system", "content": "You are a coffee expert."}, {"role": "user", "content": "Tell me about Brazilian coffee farms."}]}',
        "traceloop.entity.output": '{"content": "Brazilian coffee farms produce a wide variety of beans."}',
        "ioa_observe.entity.name": "coffee_agent",
        "gen_ai.completion.0.content": "Brazilian coffee farms produce a wide variety of beans.",
        "gen_ai.completion.0.finish_reason": "stop",
        # Prompt messages (shorter than traceloop.entity.input)
        "gen_ai.prompt.0.role": "system",
        "gen_ai.prompt.0.content": "You are a coffee expert.",
        "gen_ai.prompt.1.role": "user",
        "gen_ai.prompt.1.content": "Tell me about Brazilian coffee farms.",
    }


# ── Span row factory ──────────────────────────────────────────────────────


@pytest.fixture()
def span_row_factory():
    """Factory for mock span row dicts as returned by queries module."""

    def _make(
        *,
        trace_id: str = "abc123",
        span_id: str = "span001",
        parent_span_id: str = "",
        service_name: str = "test_service",
        span_name: str = "openai.chat",
        duration: int = 500_000_000,  # 500ms in ns
        timestamp: datetime | None = None,
        span_attributes: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if timestamp is None:
            timestamp = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        if span_attributes is None:
            span_attributes = {}
        return {
            "trace_id": trace_id,
            "span_id": span_id,
            "parent_span_id": parent_span_id,
            "service_name": service_name,
            "span_name": span_name,
            "duration": duration,
            "timestamp": timestamp,
            "span_attributes": span_attributes,
        }

    return _make


# ── Pre-built LLMCallTokens ──────────────────────────────────────────────


@pytest.fixture()
def mock_llm_call():
    """Pre-built LLMCallTokens instance for cost/breakdown tests."""
    from token_analysis.models import LLMCallTokens

    return LLMCallTokens(
        span_id="span_001",
        trace_id="trace_abc",
        span_name="openai.chat",
        service_name="lungo.recruiter_supervisor",
        agent_name="lungo.recruiter_supervisor",
        model="azure/gpt-4o",
        response_model="gpt-4o-2024-05-13",
        timestamp_ns=1_700_000_000_000_000_000,
        duration_ms=500.0,
        input_tokens=1500,
        output_tokens=350,
        cache_read_input_tokens=200,
        reasoning_tokens=0,
        total_tokens=1850,
        tokens_estimated=False,
        context_window_size=128_000,
        context_utilization=0.0117,
        finish_reason="stop",
    )
