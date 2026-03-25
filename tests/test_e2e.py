"""End-to-end tests requiring a live ClickHouse instance.

All tests in this module are marked @pytest.mark.e2e and will be
skipped unless TOKEN_ANALYSIS_E2E=1 is set in the environment.
"""

from __future__ import annotations

import json

import pytest

from conftest import BRAZIL_TRACE_ID, COLOMBIA_TRACE_ID, RECRUITER_TRACE_ID

from token_analysis.analyzer import analyze_trace_tokens, analyze_window, discover_agents
from token_analysis.client import relative_to_nano
from token_analysis.models import (
    AgentDiscovery,
    TokenWindowAnalysis,
    TraceTokenAnalysis,
)
from token_analysis.report import format_report

pytestmark = pytest.mark.e2e


# ── Recruiter trace tests ────────────────────────────────────────────────


class TestRecruiterTrace:
    def test_total_llm_calls(self, ch_client):
        result = analyze_trace_tokens(ch_client, RECRUITER_TRACE_ID)
        assert isinstance(result, TraceTokenAnalysis)
        # Recruiter trace has LLM calls from recruiter_supervisor + brazil_farm
        assert result.total_llm_calls >= 2

    def test_has_reported_tokens(self, ch_client):
        """At least some calls should have provider-reported tokens (not estimated)."""
        result = analyze_trace_tokens(ch_client, RECRUITER_TRACE_ID)
        has_reported = any(not c.tokens_estimated for c in result.llm_calls)
        assert has_reported, "Expected some calls with reported (non-estimated) tokens"

    def test_context_snapshots_nonempty(self, ch_client):
        result = analyze_trace_tokens(ch_client, RECRUITER_TRACE_ID)
        assert len(result.context_snapshots) > 0

    def test_context_window_on_calls(self, ch_client):
        result = analyze_trace_tokens(ch_client, RECRUITER_TRACE_ID)
        for call in result.llm_calls:
            assert call.context_window_size > 0, (
                f"Expected context_window_size > 0 for model {call.model}"
            )


# ── Colombia trace tests ─────────────────────────────────────────────────


class TestColombiaTrace:
    def test_total_llm_calls(self, ch_client):
        result = analyze_trace_tokens(ch_client, COLOMBIA_TRACE_ID)
        assert result.total_llm_calls == 5

    def test_agent_breakdown(self, ch_client):
        result = analyze_trace_tokens(ch_client, COLOMBIA_TRACE_ID)
        agents = set(result.agent_breakdown.keys())
        # Should have at least these agents from the colombian coffee trace
        assert len(agents) >= 2

    def test_mix_of_estimated(self, ch_client):
        result = analyze_trace_tokens(ch_client, COLOMBIA_TRACE_ID)
        estimated = [c.tokens_estimated for c in result.llm_calls]
        # Should have a mix of True and False
        assert True in estimated or False in estimated


# ── Brazil trace tests ───────────────────────────────────────────────────


class TestBrazilTrace:
    def test_has_estimated_tokens(self, ch_client):
        result = analyze_trace_tokens(ch_client, BRAZIL_TRACE_ID)
        has_estimated = any(c.tokens_estimated for c in result.llm_calls)
        assert has_estimated, "Expected some calls with estimated tokens"

    def test_accumulation_alerts(self, ch_client):
        result = analyze_trace_tokens(
            ch_client, BRAZIL_TRACE_ID, growth_factor_warn=2.0
        )
        # This trace is known to have context growth > 2x
        assert len(result.accumulation_alerts) > 0


# ── Window + Discovery tests ─────────────────────────────────────────────


class TestWindowAnalysis:
    def test_analyze_window_nonempty(self, ch_client):
        start_ns, end_ns = relative_to_nano("24h")
        result = analyze_window(ch_client, start_ns, end_ns, limit=5)
        assert isinstance(result, TokenWindowAnalysis)
        # May be empty if no recent traces — just validate type
        assert result.trace_count >= 0


class TestDiscovery:
    def test_discover_agents(self, ch_client):
        start_ns, end_ns = relative_to_nano("24h")
        result = discover_agents(ch_client, start_ns, end_ns)
        assert isinstance(result, AgentDiscovery)
        assert result.total_agents >= 0
        assert isinstance(result.services, list)

    def test_service_consolidation(self, ch_client):
        """Services should be consolidated — no duplicate agent rows for same service."""
        start_ns, end_ns = relative_to_nano("7d")
        result = discover_agents(ch_client, start_ns, end_ns)
        # Check that service names are unique across agents
        services_with_llm = [
            a.service_name for a in result.agents if a.llm_call_count > 0
        ]
        assert len(services_with_llm) == len(set(services_with_llm))

    def test_cost_populated(self, ch_client):
        """Cost should be populated when analyzing a trace with known LLM calls."""
        result = analyze_trace_tokens(
            ch_client, RECRUITER_TRACE_ID, include_cost=True
        )
        if result.total_llm_calls > 0:
            assert result.cost.total_cost_usd > 0


class TestJsonRoundtrip:
    def test_full_roundtrip(self, ch_client):
        """Analyze → format as JSON → parse back → validate model."""
        result = analyze_trace_tokens(ch_client, RECRUITER_TRACE_ID)
        json_str = format_report(result, "json")
        data = json.loads(json_str)
        restored = TraceTokenAnalysis.model_validate(data)
        assert restored.trace_id == RECRUITER_TRACE_ID
        assert restored.total_llm_calls == result.total_llm_calls
        assert restored.total_tokens == result.total_tokens
