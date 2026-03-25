"""Tests for analyzer.py — internal helpers and mocked integration tests."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from token_analysis.analyzer import (
    _build_agent_breakdown,
    _collect_prompt_content,
    _estimate_tokens_from_text,
    _extract_service_chain,
    _parse_llm_call,
    _resolve_agent_identity,
    _to_int,
    _aggregate_window,
    analyze_trace_tokens,
    analyze_window,
)
from token_analysis.models import (
    LLMCallTokens,
    TraceTokenAnalysis,
)


# ── _resolve_agent_identity tests ─────────────────────────────────────────


class TestResolveAgentIdentity:
    def test_gen_ai_agent_name_first(self):
        attrs = {
            "gen_ai.agent.name": "primary_agent",
            "ioa_observe.entity.name": "secondary_agent",
        }
        assert _resolve_agent_identity(attrs, "service") == "primary_agent"

    def test_ioa_observe_fallback(self):
        attrs = {"ioa_observe.entity.name": "ioa_agent"}
        assert _resolve_agent_identity(attrs, "service") == "ioa_agent"

    def test_service_name_fallback(self):
        attrs = {}
        assert _resolve_agent_identity(attrs, "my_service") == "my_service"

    def test_whitespace_stripping(self):
        attrs = {"gen_ai.agent.name": "  agent_name  "}
        assert _resolve_agent_identity(attrs, "svc") == "agent_name"

    def test_empty_gen_ai_uses_ioa(self):
        attrs = {"gen_ai.agent.name": "", "ioa_observe.entity.name": "fallback"}
        assert _resolve_agent_identity(attrs, "svc") == "fallback"

    def test_whitespace_only_gen_ai_uses_ioa(self):
        attrs = {"gen_ai.agent.name": "   ", "ioa_observe.entity.name": "fb"}
        assert _resolve_agent_identity(attrs, "svc") == "fb"


# ── _to_int tests ─────────────────────────────────────────────────────────


class TestToInt:
    def test_normal(self):
        assert _to_int("42") == 42

    def test_empty(self):
        assert _to_int("") == 0

    def test_none(self):
        assert _to_int(None) == 0

    def test_non_numeric(self):
        assert _to_int("abc") == 0

    def test_float_string(self):
        assert _to_int("3.14") == 0  # int() can't parse floats

    def test_zero_string(self):
        assert _to_int("0") == 0


# ── _estimate_tokens_from_text tests ──────────────────────────────────────


class TestEstimateTokensFromText:
    def test_basic(self):
        text = "a" * 400  # 400 chars -> ~100 tokens
        assert _estimate_tokens_from_text(text) == 100

    def test_empty(self):
        assert _estimate_tokens_from_text("") == 0

    def test_short(self):
        assert _estimate_tokens_from_text("hi") == 1  # max(1, 2//4=0) = 1

    def test_single_char(self):
        assert _estimate_tokens_from_text("x") == 1


# ── _collect_prompt_content tests ─────────────────────────────────────────


class TestCollectPromptContent:
    def test_basic(self):
        attrs = {
            "gen_ai.prompt.0.role": "system",
            "gen_ai.prompt.0.content": "Be helpful.",
            "gen_ai.prompt.1.role": "user",
            "gen_ai.prompt.1.content": "Hello",
        }
        result = _collect_prompt_content(attrs)
        assert "Be helpful." in result
        assert "Hello" in result

    def test_with_tool_calls(self):
        attrs = {
            "gen_ai.prompt.0.role": "assistant",
            "gen_ai.prompt.0.content": "",
            "gen_ai.prompt.0.tool_calls.0.name": "get_weather",
            "gen_ai.prompt.0.tool_calls.0.arguments": '{"city": "SF"}',
        }
        result = _collect_prompt_content(attrs)
        assert "get_weather" in result
        assert '{"city": "SF"}' in result

    def test_empty(self):
        result = _collect_prompt_content({})
        assert result == ""

    def test_role_without_content(self):
        attrs = {"gen_ai.prompt.0.role": "user"}
        result = _collect_prompt_content(attrs)
        assert result == ""


# ── _parse_llm_call tests ────────────────────────────────────────────────


class TestParseLlmCall:
    def test_reported_tokens(self, openai_span_attrs, span_row_factory):
        row = span_row_factory(span_attributes=openai_span_attrs)
        call = _parse_llm_call(row, "trace1")
        assert call.input_tokens == 1500
        assert call.output_tokens == 350
        assert call.cache_read_input_tokens == 200
        assert call.tokens_estimated is False
        assert call.total_tokens == 1500 + 350
        assert call.agent_name == "test_agent"
        assert call.model == "azure/gpt-4o"
        assert call.finish_reason == "stop"

    def test_estimated_tokens_traceloop(self, traceloop_span_attrs, span_row_factory):
        row = span_row_factory(span_attributes=traceloop_span_attrs)
        call = _parse_llm_call(row, "trace1")
        assert call.tokens_estimated is True
        assert call.input_tokens > 0
        assert call.output_tokens > 0
        # Agent should resolve to ioa_observe.entity.name
        assert call.agent_name == "coffee_agent"

    def test_largest_source_selected(self, span_row_factory):
        """When both prompt content and traceloop.entity.input exist, use the larger one."""
        attrs = {
            "gen_ai.request.model": "gpt-4o",
            # Short prompt content
            "gen_ai.prompt.0.role": "user",
            "gen_ai.prompt.0.content": "Short",
            # Much longer traceloop content
            "traceloop.entity.input": "A" * 1000,
            "traceloop.entity.output": "B" * 200,
        }
        row = span_row_factory(span_attributes=attrs)
        call = _parse_llm_call(row, "trace1")
        assert call.tokens_estimated is True
        # Should use traceloop (1000 chars -> ~250 tokens), not prompt (5 chars -> 1 token)
        assert call.input_tokens == 250

    def test_context_window_fields(self, openai_span_attrs, span_row_factory):
        row = span_row_factory(span_attributes=openai_span_attrs)
        call = _parse_llm_call(row, "trace1")
        assert call.context_window_size == 128_000
        assert call.context_utilization > 0

    def test_finish_reason_fallback(self, span_row_factory):
        attrs = {
            "gen_ai.request.model": "gpt-4o",
            "gen_ai.usage.input_tokens": "100",
            "gen_ai.usage.output_tokens": "50",
            "llm.response.finish_reason": "length",
        }
        row = span_row_factory(span_attributes=attrs)
        call = _parse_llm_call(row, "trace1")
        assert call.finish_reason == "length"

    def test_unknown_model_context_window(self, span_row_factory):
        attrs = {
            "gen_ai.request.model": "unknown-model",
            "gen_ai.usage.input_tokens": "100",
            "gen_ai.usage.output_tokens": "50",
        }
        row = span_row_factory(span_attributes=attrs)
        call = _parse_llm_call(row, "trace1")
        assert call.context_window_size == 0
        assert call.context_utilization == 0.0


# ── _build_agent_breakdown tests ─────────────────────────────────────────


class TestBuildAgentBreakdown:
    def test_single_agent(self):
        calls = [
            LLMCallTokens(
                span_id="s1",
                trace_id="t1",
                agent_name="agent_a",
                service_name="svc_a",
                model="gpt-4o",
                input_tokens=1000,
                output_tokens=500,
                cache_read_input_tokens=200,
                total_tokens=1500,
                context_window_size=128_000,
                context_utilization=0.0078,
            ),
        ]
        result = _build_agent_breakdown(calls)
        assert "agent_a" in result
        ab = result["agent_a"]
        assert ab.llm_call_count == 1
        assert ab.input_tokens == 1000
        assert ab.output_tokens == 500
        assert ab.avg_input_per_call == 1000.0
        assert ab.cache_hit_ratio == 0.2
        assert ab.context_window_size == 128_000

    def test_multiple_agents(self):
        calls = [
            LLMCallTokens(
                span_id="s1", trace_id="t1",
                agent_name="agent_a", model="gpt-4o",
                input_tokens=1000, output_tokens=500, total_tokens=1500,
            ),
            LLMCallTokens(
                span_id="s2", trace_id="t1",
                agent_name="agent_b", model="gpt-4o",
                input_tokens=2000, output_tokens=800, total_tokens=2800,
            ),
        ]
        result = _build_agent_breakdown(calls)
        assert len(result) == 2
        assert result["agent_a"].input_tokens == 1000
        assert result["agent_b"].input_tokens == 2000

    def test_averages(self):
        calls = [
            LLMCallTokens(
                span_id="s1", trace_id="t1",
                agent_name="agent_a", model="gpt-4o",
                input_tokens=1000, output_tokens=400, total_tokens=1400,
            ),
            LLMCallTokens(
                span_id="s2", trace_id="t1",
                agent_name="agent_a", model="gpt-4o",
                input_tokens=3000, output_tokens=600, total_tokens=3600,
            ),
        ]
        result = _build_agent_breakdown(calls)
        ab = result["agent_a"]
        assert ab.llm_call_count == 2
        assert ab.avg_input_per_call == 2000.0
        assert ab.avg_output_per_call == 500.0

    def test_context_utilization(self):
        calls = [
            LLMCallTokens(
                span_id="s1", trace_id="t1",
                agent_name="agent_a", model="gpt-4o",
                input_tokens=1000, output_tokens=400, total_tokens=1400,
                context_window_size=128_000, context_utilization=0.008,
            ),
            LLMCallTokens(
                span_id="s2", trace_id="t1",
                agent_name="agent_a", model="gpt-4o",
                input_tokens=64000, output_tokens=600, total_tokens=64600,
                context_window_size=128_000, context_utilization=0.5,
            ),
        ]
        result = _build_agent_breakdown(calls)
        ab = result["agent_a"]
        assert ab.max_context_utilization == 0.5
        assert ab.avg_context_utilization > 0

    def test_models_used(self):
        calls = [
            LLMCallTokens(
                span_id="s1", trace_id="t1",
                agent_name="agent_a", model="gpt-4o",
                input_tokens=100, output_tokens=50, total_tokens=150,
            ),
            LLMCallTokens(
                span_id="s2", trace_id="t1",
                agent_name="agent_a", model="claude-3-5-sonnet",
                input_tokens=100, output_tokens=50, total_tokens=150,
            ),
        ]
        result = _build_agent_breakdown(calls)
        ab = result["agent_a"]
        assert "gpt-4o" in ab.models_used
        assert "claude-3-5-sonnet" in ab.models_used


# ── _extract_service_chain tests ──────────────────────────────────────────


class TestExtractServiceChain:
    def test_ordered_unique(self):
        spans = [
            {"service_name": "svc_a"},
            {"service_name": "svc_b"},
            {"service_name": "svc_a"},
            {"service_name": "svc_c"},
        ]
        chain = _extract_service_chain(spans)
        assert chain == ["svc_a", "svc_b", "svc_c"]

    def test_empty(self):
        assert _extract_service_chain([]) == []


# ── _aggregate_window tests ──────────────────────────────────────────────


class TestAggregateWindow:
    def _from_nano(self, ns):
        return f"2024-01-01T00:00:00.000Z"

    def test_empty_traces(self):
        result = _aggregate_window([], 0, 1, self._from_nano)
        assert result.trace_count == 0
        assert result.total_llm_calls == 0

    def test_single_trace(self):
        trace = TraceTokenAnalysis(
            trace_id="t1",
            total_llm_calls=2,
            total_input_tokens=1000,
            total_output_tokens=500,
            total_tokens=1500,
        )
        result = _aggregate_window([trace], 0, 1, self._from_nano)
        assert result.trace_count == 1
        assert result.total_llm_calls == 2
        assert result.total_tokens == 1500
        # Single trace: percentiles equal the single value
        assert result.p50_tokens_per_trace == 1500.0

    def test_multiple_traces(self):
        t1 = TraceTokenAnalysis(
            trace_id="t1",
            total_llm_calls=2,
            total_input_tokens=1000,
            total_output_tokens=500,
            total_tokens=1500,
        )
        t2 = TraceTokenAnalysis(
            trace_id="t2",
            total_llm_calls=3,
            total_input_tokens=2000,
            total_output_tokens=800,
            total_tokens=2800,
        )
        result = _aggregate_window([t1, t2], 0, 1, self._from_nano)
        assert result.trace_count == 2
        assert result.total_llm_calls == 5
        assert result.total_tokens == 4300

    def test_cache_ratio(self):
        trace = TraceTokenAnalysis(
            trace_id="t1",
            total_input_tokens=1000,
            total_cache_read_tokens=300,
            total_tokens=1000,
        )
        result = _aggregate_window([trace], 0, 1, self._from_nano)
        assert result.overall_cache_hit_ratio == 0.3


# ── Mocked integration tests ──────────────────────────────────────────────


class TestAnalyzeTraceTokensMocked:
    @patch("token_analysis.analyzer.queries")
    def test_returns_valid_analysis(self, mock_queries, openai_span_attrs, span_row_factory):
        # Mock all queries
        mock_queries.get_all_trace_spans.return_value = [
            span_row_factory(span_id="root", duration=1_000_000_000),
        ]
        mock_queries.get_deduplicated_llm_calls.return_value = [
            span_row_factory(span_id="llm1", span_attributes=openai_span_attrs),
        ]
        mock_queries.get_context_window_spans.return_value = [
            span_row_factory(span_id="llm1", span_attributes=openai_span_attrs),
        ]

        client = MagicMock()
        result = analyze_trace_tokens(client, "trace_abc")

        assert isinstance(result, TraceTokenAnalysis)
        assert result.trace_id == "trace_abc"
        assert result.total_llm_calls == 1
        assert result.total_input_tokens == 1500
        assert result.total_output_tokens == 350
        assert len(result.llm_calls) == 1
        assert len(result.context_snapshots) == 1

    @patch("token_analysis.analyzer.queries")
    def test_cost_populated(self, mock_queries, openai_span_attrs, span_row_factory):
        mock_queries.get_all_trace_spans.return_value = [
            span_row_factory(span_id="root"),
        ]
        mock_queries.get_deduplicated_llm_calls.return_value = [
            span_row_factory(span_id="llm1", span_attributes=openai_span_attrs),
        ]
        mock_queries.get_context_window_spans.return_value = []

        client = MagicMock()
        result = analyze_trace_tokens(client, "trace_abc", include_cost=True)

        assert result.cost.total_cost_usd > 0

    @patch("token_analysis.analyzer.queries")
    def test_empty_trace(self, mock_queries, span_row_factory):
        mock_queries.get_all_trace_spans.return_value = []
        mock_queries.get_deduplicated_llm_calls.return_value = []
        mock_queries.get_context_window_spans.return_value = []

        client = MagicMock()
        result = analyze_trace_tokens(client, "empty_trace")

        assert result.total_llm_calls == 0
        assert result.total_tokens == 0


class TestAnalyzeWindowMocked:
    @patch("token_analysis.analyzer.queries")
    def test_returns_valid_analysis(self, mock_queries, openai_span_attrs, span_row_factory):
        mock_queries.get_trace_ids.return_value = ["trace1"]
        mock_queries.get_all_trace_spans.return_value = [
            span_row_factory(span_id="root"),
        ]
        mock_queries.get_deduplicated_llm_calls.return_value = [
            span_row_factory(span_id="llm1", span_attributes=openai_span_attrs),
        ]
        mock_queries.get_context_window_spans.return_value = []

        client = MagicMock()
        result = analyze_window(client, 0, 1_000_000_000_000_000_000)

        assert result.trace_count == 1
        assert result.total_llm_calls == 1
        assert len(result.traces) == 1
