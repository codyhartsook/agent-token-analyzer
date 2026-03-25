"""Tests for cost.py — pricing lookup and cost estimation."""

from __future__ import annotations

from token_analysis.cost import (
    DEFAULT_PRICING,
    _find_pricing,
    estimate_call_cost,
    estimate_cost,
)
from token_analysis.models import LLMCallTokens, ModelPricing


# ── _find_pricing tests ──────────────────────────────────────────────────


class TestFindPricing:
    def test_gpt4o_match(self):
        result = _find_pricing("gpt-4o-2024-05-13", DEFAULT_PRICING)
        assert result is not None
        assert result.input_per_1m == 2.50

    def test_specificity_mini_over_base(self):
        """gpt-4o-mini should match the more specific pattern."""
        result = _find_pricing("gpt-4o-mini", DEFAULT_PRICING)
        assert result is not None
        assert result.model_pattern == "*gpt-4o-mini*"
        assert result.input_per_1m == 0.15

    def test_claude_35_sonnet(self):
        result = _find_pricing("claude-3-5-sonnet-20241022", DEFAULT_PRICING)
        assert result is not None
        assert result.input_per_1m == 3.00

    def test_no_match(self):
        result = _find_pricing("totally-unknown", DEFAULT_PRICING)
        assert result is None

    def test_case_insensitive(self):
        result = _find_pricing("GPT-4O", DEFAULT_PRICING)
        assert result is not None


# ── estimate_call_cost tests ─────────────────────────────────────────────


class TestEstimateCallCost:
    def test_basic(self):
        call = LLMCallTokens(
            span_id="s1",
            trace_id="t1",
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
        )
        cost = estimate_call_cost(call)
        # Input: 1000 * 2.50 / 1M = 0.0025
        # Output: 500 * 10.00 / 1M = 0.005
        assert cost > 0
        assert abs(cost - 0.0075) < 0.0001

    def test_with_cache(self):
        call = LLMCallTokens(
            span_id="s1",
            trace_id="t1",
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
            cache_read_input_tokens=400,
        )
        cost = estimate_call_cost(call)
        # Billable: 600 * 2.50 / 1M = 0.0015
        # Cached: 400 * 1.25 / 1M = 0.0005
        # Output: 500 * 10.00 / 1M = 0.005
        expected = 0.0015 + 0.0005 + 0.005
        assert abs(cost - expected) < 0.0001

    def test_with_reasoning(self):
        call = LLMCallTokens(
            span_id="s1",
            trace_id="t1",
            model="o1",
            input_tokens=1000,
            output_tokens=500,
            reasoning_tokens=200,
        )
        cost = estimate_call_cost(call)
        # Input: 1000 * 15.00 / 1M = 0.015
        # Output: 500 * 60.00 / 1M = 0.03
        # Reasoning: 200 * 60.00 / 1M = 0.012
        expected = 0.015 + 0.03 + 0.012
        assert abs(cost - expected) < 0.0001

    def test_no_match_returns_zero(self):
        call = LLMCallTokens(
            span_id="s1",
            trace_id="t1",
            model="unknown-model",
            input_tokens=1000,
            output_tokens=500,
        )
        cost = estimate_call_cost(call)
        assert cost == 0.0

    def test_response_model_fallback(self):
        call = LLMCallTokens(
            span_id="s1",
            trace_id="t1",
            model="",
            response_model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
        )
        cost = estimate_call_cost(call)
        assert cost > 0


# ── estimate_cost tests ──────────────────────────────────────────────────


class TestEstimateCost:
    def test_aggregate(self):
        calls = [
            LLMCallTokens(
                span_id="s1",
                trace_id="t1",
                model="gpt-4o",
                agent_name="agent_a",
                input_tokens=1000,
                output_tokens=500,
            ),
            LLMCallTokens(
                span_id="s2",
                trace_id="t1",
                model="gpt-4o",
                agent_name="agent_b",
                input_tokens=2000,
                output_tokens=1000,
            ),
        ]
        result = estimate_cost(calls)
        assert result.total_cost_usd > 0
        assert result.input_cost_usd > 0
        assert result.output_cost_usd > 0

    def test_per_model_map(self):
        calls = [
            LLMCallTokens(
                span_id="s1",
                trace_id="t1",
                model="gpt-4o",
                input_tokens=1000,
                output_tokens=500,
            ),
        ]
        result = estimate_cost(calls)
        assert "gpt-4o" in result.per_model
        assert result.per_model["gpt-4o"] > 0

    def test_per_agent_map(self):
        calls = [
            LLMCallTokens(
                span_id="s1",
                trace_id="t1",
                model="gpt-4o",
                agent_name="my_agent",
                input_tokens=1000,
                output_tokens=500,
            ),
        ]
        result = estimate_cost(calls)
        assert "my_agent" in result.per_agent

    def test_empty_list(self):
        result = estimate_cost([])
        assert result.total_cost_usd == 0.0
        assert result.calls_without_pricing == 0

    def test_mixed_known_unknown(self):
        calls = [
            LLMCallTokens(
                span_id="s1",
                trace_id="t1",
                model="gpt-4o",
                input_tokens=1000,
                output_tokens=500,
            ),
            LLMCallTokens(
                span_id="s2",
                trace_id="t1",
                model="unknown-model",
                input_tokens=1000,
                output_tokens=500,
            ),
        ]
        result = estimate_cost(calls)
        assert result.calls_without_pricing == 1
        assert result.total_cost_usd > 0  # First call still counted
