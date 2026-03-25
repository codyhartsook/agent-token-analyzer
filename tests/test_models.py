"""Tests for Pydantic models — computed fields, serialization, model_dump."""

from __future__ import annotations

import json

from token_analysis.models import (
    AgentTokenBreakdown,
    ContextWindowSnapshot,
    CostEstimate,
    LLMCallTokens,
    PromptMessage,
    TraceTokenAnalysis,
)


class TestLLMCallTokensCacheHitRatio:
    def test_correct_ratio(self):
        call = LLMCallTokens(
            span_id="s1",
            trace_id="t1",
            input_tokens=1000,
            cache_read_input_tokens=300,
        )
        assert call.cache_hit_ratio == 0.3

    def test_zero_input_returns_zero(self):
        call = LLMCallTokens(span_id="s1", trace_id="t1", input_tokens=0)
        assert call.cache_hit_ratio == 0.0

    def test_full_cache(self):
        call = LLMCallTokens(
            span_id="s1",
            trace_id="t1",
            input_tokens=500,
            cache_read_input_tokens=500,
        )
        assert call.cache_hit_ratio == 1.0


class TestLLMCallTokensBillableInput:
    def test_basic(self):
        call = LLMCallTokens(
            span_id="s1",
            trace_id="t1",
            input_tokens=1000,
            cache_read_input_tokens=300,
        )
        assert call.billable_input_tokens == 700

    def test_never_negative(self):
        call = LLMCallTokens(
            span_id="s1",
            trace_id="t1",
            input_tokens=100,
            cache_read_input_tokens=200,
        )
        assert call.billable_input_tokens == 0

    def test_zero_cache(self):
        call = LLMCallTokens(
            span_id="s1",
            trace_id="t1",
            input_tokens=500,
            cache_read_input_tokens=0,
        )
        assert call.billable_input_tokens == 500


class TestContextWindowSnapshotHasToolResults:
    def test_true_when_tool_role_present(self):
        snap = ContextWindowSnapshot(
            span_id="s1",
            call_index=0,
            messages_by_role={"system": 1, "user": 1, "tool": 2},
        )
        assert snap.has_tool_results is True

    def test_false_when_no_tool_role(self):
        snap = ContextWindowSnapshot(
            span_id="s1",
            call_index=0,
            messages_by_role={"system": 1, "user": 1, "assistant": 1},
        )
        assert snap.has_tool_results is False

    def test_false_when_empty(self):
        snap = ContextWindowSnapshot(span_id="s1", call_index=0)
        assert snap.has_tool_results is False


class TestSerialization:
    def test_llm_call_json_roundtrip(self, mock_llm_call):
        data = json.loads(mock_llm_call.model_dump_json())
        assert data["span_id"] == "span_001"
        assert data["input_tokens"] == 1500
        assert data["cache_hit_ratio"] == 0.1333
        assert data["billable_input_tokens"] == 1300

    def test_llm_call_computed_fields_in_dump(self, mock_llm_call):
        d = mock_llm_call.model_dump()
        assert "cache_hit_ratio" in d
        assert "billable_input_tokens" in d

    def test_trace_token_analysis_roundtrip(self):
        t = TraceTokenAnalysis(
            trace_id="t1",
            total_spans=10,
            total_llm_calls=3,
            total_input_tokens=5000,
            total_output_tokens=1000,
            total_tokens=6000,
        )
        data = json.loads(t.model_dump_json())
        assert data["trace_id"] == "t1"
        assert data["total_tokens"] == 6000
        # Validate nested models are empty lists by default
        assert data["llm_calls"] == []
        assert data["context_snapshots"] == []

    def test_context_snapshot_roundtrip(self):
        snap = ContextWindowSnapshot(
            span_id="s1",
            call_index=0,
            agent_name="test",
            messages=[
                PromptMessage(index=0, role="system", content="Hello"),
            ],
            messages_by_role={"system": 1},
            total_messages=1,
            total_content_chars=5,
        )
        data = json.loads(snap.model_dump_json())
        assert data["total_messages"] == 1
        assert data["has_tool_results"] is False
        assert len(data["messages"]) == 1
        assert data["messages"][0]["role"] == "system"
