"""Tests for context.py — reconstruction, growth, accumulation detection."""

from __future__ import annotations

from token_analysis.context import (
    compute_context_growth,
    detect_accumulation,
    reconstruct_context,
)
from token_analysis.models import (
    ContextWindowSnapshot,
    LLMCallTokens,
    PromptMessage,
)


# ── reconstruct_context tests ──────────────────────────────────────────────


class TestReconstructContext:
    def test_basic_two_messages(self):
        attrs = {
            "gen_ai.prompt.0.role": "system",
            "gen_ai.prompt.0.content": "You are helpful.",
            "gen_ai.prompt.1.role": "user",
            "gen_ai.prompt.1.content": "Hello!",
        }
        snap = reconstruct_context(attrs, span_id="s1", call_index=0)
        assert snap.total_messages == 2
        assert snap.messages[0].role == "system"
        assert snap.messages[1].role == "user"
        assert snap.messages_by_role == {"system": 1, "user": 1}

    def test_tool_calls(self):
        attrs = {
            "gen_ai.prompt.0.role": "assistant",
            "gen_ai.prompt.0.content": "",
            "gen_ai.prompt.0.tool_calls.0.name": "get_weather",
            "gen_ai.prompt.0.tool_calls.0.arguments": '{"city": "SF"}',
            "gen_ai.prompt.0.tool_calls.1.name": "get_time",
            "gen_ai.prompt.0.tool_calls.1.arguments": '{"tz": "PST"}',
        }
        snap = reconstruct_context(attrs, span_id="s1", call_index=0)
        assert len(snap.messages) == 1
        assert len(snap.messages[0].tool_calls) == 2
        assert snap.messages[0].tool_calls[0].name == "get_weather"
        assert snap.messages[0].tool_calls[1].name == "get_time"
        assert snap.total_tool_calls == 2

    def test_tool_response(self):
        attrs = {
            "gen_ai.prompt.0.role": "tool",
            "gen_ai.prompt.0.content": '{"temp": 65}',
            "gen_ai.prompt.0.tool_call_id": "call_123",
        }
        snap = reconstruct_context(attrs, span_id="s1", call_index=0)
        assert snap.messages[0].role == "tool"
        assert snap.messages[0].tool_call_id == "call_123"

    def test_completion_extraction(self):
        attrs = {
            "gen_ai.prompt.0.role": "user",
            "gen_ai.prompt.0.content": "Hi",
            "gen_ai.completion.0.content": "Hello there!",
            "gen_ai.completion.0.finish_reason": "stop",
        }
        snap = reconstruct_context(attrs, span_id="s1", call_index=0)
        assert snap.completion_content == "Hello there!"
        assert snap.completion_finish_reason == "stop"

    def test_metrics(self):
        attrs = {
            "gen_ai.prompt.0.role": "system",
            "gen_ai.prompt.0.content": "Be helpful.",
            "gen_ai.prompt.1.role": "user",
            "gen_ai.prompt.1.content": "How are you?",
        }
        snap = reconstruct_context(attrs, span_id="s1", call_index=0)
        assert snap.total_content_chars == len("Be helpful.") + len("How are you?")
        assert snap.total_tool_calls == 0

    def test_empty_attrs(self):
        snap = reconstruct_context({}, span_id="s1", call_index=0)
        assert snap.total_messages == 0
        assert snap.messages == []
        assert snap.completion_content == ""

    def test_agent_name_and_model(self):
        attrs = {
            "gen_ai.prompt.0.role": "user",
            "gen_ai.prompt.0.content": "Test",
        }
        snap = reconstruct_context(
            attrs,
            span_id="s1",
            call_index=5,
            agent_name="my_agent",
            model="gpt-4o",
            timestamp_ns=12345,
        )
        assert snap.agent_name == "my_agent"
        assert snap.model == "gpt-4o"
        assert snap.call_index == 5
        assert snap.timestamp_ns == 12345

    def test_full_fixture(self, openai_span_attrs):
        snap = reconstruct_context(
            openai_span_attrs, span_id="s1", call_index=0, agent_name="test"
        )
        assert snap.total_messages == 4
        assert snap.messages_by_role["system"] == 1
        assert snap.messages_by_role["user"] == 1
        assert snap.messages_by_role["assistant"] == 1
        assert snap.messages_by_role["tool"] == 1
        assert snap.total_tool_calls == 1
        assert snap.completion_content == "Sure, I can help with that."
        assert snap.completion_finish_reason == "stop"


# ── compute_context_growth tests ──────────────────────────────────────────


class TestComputeContextGrowth:
    def _make_snap(self, call_index, span_id, total_messages, total_chars, msgs_by_role=None):
        return ContextWindowSnapshot(
            span_id=span_id,
            call_index=call_index,
            total_messages=total_messages,
            total_content_chars=total_chars,
            messages_by_role=msgs_by_role or {},
        )

    def test_basic_deltas(self):
        s1 = self._make_snap(0, "sp1", 3, 100, {"system": 1, "user": 1, "assistant": 1})
        s2 = self._make_snap(1, "sp2", 5, 250, {"system": 1, "user": 2, "assistant": 2})
        steps = compute_context_growth([s1, s2])
        assert len(steps) == 1
        assert steps[0].message_count_delta == 2
        assert steps[0].content_chars_delta == 150
        assert steps[0].growth_pct == 150.0

    def test_with_token_map(self):
        s1 = self._make_snap(0, "sp1", 3, 100)
        s2 = self._make_snap(1, "sp2", 5, 200)
        calls = [
            LLMCallTokens(span_id="sp1", trace_id="t1", input_tokens=500),
            LLMCallTokens(span_id="sp2", trace_id="t1", input_tokens=800),
        ]
        steps = compute_context_growth([s1, s2], calls)
        assert steps[0].input_tokens_delta == 300

    def test_single_snapshot(self):
        s1 = self._make_snap(0, "sp1", 3, 100)
        steps = compute_context_growth([s1])
        assert len(steps) == 0

    def test_zero_base_chars(self):
        s1 = self._make_snap(0, "sp1", 0, 0)
        s2 = self._make_snap(1, "sp2", 3, 100)
        steps = compute_context_growth([s1, s2])
        assert steps[0].growth_pct == 0.0  # Division by zero guard

    def test_new_roles(self):
        s1 = self._make_snap(0, "sp1", 2, 100, {"system": 1, "user": 1})
        s2 = self._make_snap(1, "sp2", 4, 200, {"system": 1, "user": 1, "assistant": 1, "tool": 1})
        steps = compute_context_growth([s1, s2])
        assert steps[0].new_roles == {"assistant": 1, "tool": 1}

    def test_three_snapshots(self):
        s1 = self._make_snap(0, "sp1", 2, 100)
        s2 = self._make_snap(1, "sp2", 4, 200)
        s3 = self._make_snap(2, "sp3", 6, 400)
        steps = compute_context_growth([s1, s2, s3])
        assert len(steps) == 2
        assert steps[0].from_call_index == 0
        assert steps[0].to_call_index == 1
        assert steps[1].from_call_index == 1
        assert steps[1].to_call_index == 2


# ── detect_accumulation tests ─────────────────────────────────────────────


class TestDetectAccumulation:
    def _make_calls(self, agent_name, input_tokens_seq):
        return [
            LLMCallTokens(
                span_id=f"sp{i}",
                trace_id="t1",
                agent_name=agent_name,
                input_tokens=t,
            )
            for i, t in enumerate(input_tokens_seq)
        ]

    def test_no_growth(self):
        calls = self._make_calls("agent_a", [500, 550])
        alerts = detect_accumulation("t1", calls, [])
        assert len(alerts) == 0

    def test_warning_at_2x(self):
        calls = self._make_calls("agent_a", [500, 1100])
        alerts = detect_accumulation("t1", calls, [])
        assert len(alerts) == 1
        assert alerts[0].severity == "warning"
        assert alerts[0].growth_factor == 2.2

    def test_critical_at_5x(self):
        calls = self._make_calls("agent_a", [100, 600])
        alerts = detect_accumulation("t1", calls, [])
        assert len(alerts) == 1
        assert alerts[0].severity == "critical"
        assert alerts[0].growth_factor == 6.0

    def test_single_call(self):
        calls = self._make_calls("agent_a", [500])
        alerts = detect_accumulation("t1", calls, [])
        assert len(alerts) == 0

    def test_zero_initial(self):
        calls = self._make_calls("agent_a", [0, 1000])
        alerts = detect_accumulation("t1", calls, [])
        assert len(alerts) == 0  # Division by zero guard

    def test_groups_by_agent(self):
        calls_a = self._make_calls("agent_a", [100, 300])
        calls_b = self._make_calls("agent_b", [100, 150])
        all_calls = calls_a + calls_b
        alerts = detect_accumulation("t1", all_calls, [])
        # agent_a grew 3x (warning), agent_b grew 1.5x (no alert)
        assert len(alerts) == 1
        assert alerts[0].agent_name == "agent_a"

    def test_custom_thresholds(self):
        calls = self._make_calls("agent_a", [100, 160])
        # Default threshold is 2.0, so 1.6x should not trigger
        alerts = detect_accumulation("t1", calls, [])
        assert len(alerts) == 0
        # Lower threshold to 1.5
        alerts = detect_accumulation(
            "t1", calls, [], growth_factor_warn=1.5, growth_factor_critical=3.0
        )
        assert len(alerts) == 1
        assert alerts[0].severity == "warning"
