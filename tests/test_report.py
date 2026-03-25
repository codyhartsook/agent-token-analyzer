"""Tests for report.py — terminal/JSON/CSV formatting, write_reports."""

from __future__ import annotations

import csv
import io
import json

from token_analysis.models import (
    AgentDiscovery,
    AgentTokenBreakdown,
    ContextAccumulationAlert,
    ContextWindowSnapshot,
    CostEstimate,
    DiscoveredAgent,
    LLMCallTokens,
    PromptMessage,
    TokenWindowAnalysis,
    TraceTokenAnalysis,
)
from token_analysis.report import format_report, write_reports


# ── Helper to build test models ───────────────────────────────────────────


def _make_trace_analysis() -> TraceTokenAnalysis:
    return TraceTokenAnalysis(
        trace_id="abc123",
        total_spans=10,
        total_llm_calls=2,
        total_duration_ms=5000.0,
        service_chain=["svc_a", "svc_b"],
        total_input_tokens=3000,
        total_output_tokens=800,
        total_cache_read_tokens=500,
        total_tokens=3800,
        overall_cache_hit_ratio=0.1667,
        max_context_utilization=0.0234,
        llm_calls=[
            LLMCallTokens(
                span_id="s1",
                trace_id="abc123",
                agent_name="agent_a",
                model="gpt-4o",
                input_tokens=2000,
                output_tokens=500,
                cache_read_input_tokens=300,
                total_tokens=2500,
                duration_ms=3000.0,
                context_window_size=128_000,
                context_utilization=0.0156,
                finish_reason="stop",
            ),
            LLMCallTokens(
                span_id="s2",
                trace_id="abc123",
                agent_name="agent_b",
                model="gpt-4o",
                input_tokens=1000,
                output_tokens=300,
                cache_read_input_tokens=200,
                total_tokens=1300,
                duration_ms=2000.0,
                tokens_estimated=True,
                context_window_size=128_000,
                context_utilization=0.0078,
                finish_reason="stop",
            ),
        ],
        context_snapshots=[
            ContextWindowSnapshot(
                span_id="s1",
                call_index=0,
                agent_name="agent_a",
                model="gpt-4o",
                total_messages=3,
                messages_by_role={"system": 1, "user": 1, "assistant": 1},
                total_content_chars=500,
            ),
        ],
        agent_breakdown={
            "agent_a": AgentTokenBreakdown(
                agent_name="agent_a",
                service_name="svc_a",
                llm_call_count=1,
                input_tokens=2000,
                output_tokens=500,
                total_tokens=2500,
                context_window_size=128_000,
                max_context_utilization=0.0156,
            ),
            "agent_b": AgentTokenBreakdown(
                agent_name="agent_b",
                service_name="svc_b",
                llm_call_count=1,
                input_tokens=1000,
                output_tokens=300,
                total_tokens=1300,
            ),
        },
        cost=CostEstimate(total_cost_usd=0.025),
    )


def _make_window_analysis() -> TokenWindowAnalysis:
    trace = _make_trace_analysis()
    return TokenWindowAnalysis(
        start_time="2024-01-01T00:00:00.000Z",
        end_time="2024-01-01T01:00:00.000Z",
        trace_count=1,
        total_llm_calls=2,
        total_input_tokens=3000,
        total_output_tokens=800,
        total_tokens=3800,
        p50_tokens_per_trace=3800.0,
        p95_tokens_per_trace=3800.0,
        p99_tokens_per_trace=3800.0,
        p50_input_per_call=1500.0,
        p95_input_per_call=2000.0,
        p50_context_utilization=0.0117,
        p95_context_utilization=0.0156,
        max_context_utilization=0.0156,
        traces=[trace],
        agent_breakdown=trace.agent_breakdown,
    )


def _make_discovery() -> AgentDiscovery:
    return AgentDiscovery(
        start_time="2024-01-01T00:00:00.000Z",
        end_time="2024-01-01T01:00:00.000Z",
        total_agents=2,
        total_services=2,
        total_traces=5,
        agents=[
            DiscoveredAgent(
                agent_name="agent_a",
                service_name="svc_a",
                trace_count=3,
                llm_call_count=5,
                total_input_tokens=10000,
                total_output_tokens=3000,
                total_tokens=13000,
                models_used=["gpt-4o"],
                context_window_size=128_000,
                max_context_utilization=0.08,
                sub_agents=["sub_tool_1"],
            ),
            DiscoveredAgent(
                agent_name="agent_b",
                service_name="svc_b",
                trace_count=2,
                llm_call_count=0,
                total_input_tokens=0,
                total_output_tokens=0,
                total_tokens=0,
                span_kinds=["agent"],
            ),
        ],
        services=["svc_a", "svc_b"],
    )


# ── Terminal format tests ────────────────────────────────────────────────


class TestTerminalFormatTrace:
    def test_contains_sections(self):
        report = format_report(_make_trace_analysis(), "terminal")
        assert "Token & Context Analysis" in report
        assert "Overview" in report
        assert "Token Usage" in report
        assert "Per-LLM-Call Breakdown" in report
        assert "Context Window Reconstruction" in report
        assert "Per-Agent Token Breakdown" in report

    def test_trace_id_present(self):
        report = format_report(_make_trace_analysis(), "terminal")
        assert "abc123" in report

    def test_estimated_marker(self):
        report = format_report(_make_trace_analysis(), "terminal")
        assert "~" in report  # Estimated marker

    def test_cost_section(self):
        report = format_report(_make_trace_analysis(), "terminal")
        assert "Cost Estimate" in report


class TestTerminalFormatWindow:
    def test_contains_percentiles(self):
        report = format_report(_make_window_analysis(), "terminal")
        assert "Percentiles" in report
        assert "p50" in report.lower() or "Tokens/trace" in report

    def test_contains_agent_breakdown(self):
        report = format_report(_make_window_analysis(), "terminal")
        assert "Per-Agent Token Breakdown" in report


class TestTerminalFormatDiscovery:
    def test_contains_agent_table(self):
        report = format_report(_make_discovery(), "terminal")
        assert "Discovered Agents" in report
        assert "agent_a" in report
        assert "agent_b" in report

    def test_shows_sub_agents(self):
        report = format_report(_make_discovery(), "terminal")
        assert "sub_tool_1" in report

    def test_shows_context_window(self):
        report = format_report(_make_discovery(), "terminal")
        assert "128K" in report


# ── JSON format tests ────────────────────────────────────────────────────


class TestJsonFormat:
    def test_valid_json_trace(self):
        report = format_report(_make_trace_analysis(), "json")
        data = json.loads(report)
        assert data["trace_id"] == "abc123"

    def test_roundtrip_trace(self):
        report = format_report(_make_trace_analysis(), "json")
        data = json.loads(report)
        restored = TraceTokenAnalysis.model_validate(data)
        assert restored.trace_id == "abc123"
        assert restored.total_tokens == 3800

    def test_valid_json_window(self):
        report = format_report(_make_window_analysis(), "json")
        data = json.loads(report)
        assert data["trace_count"] == 1

    def test_valid_json_discovery(self):
        report = format_report(_make_discovery(), "json")
        data = json.loads(report)
        assert data["total_agents"] == 2


# ── CSV format tests ─────────────────────────────────────────────────────


class TestCsvFormat:
    def test_correct_headers(self):
        report = format_report(_make_trace_analysis(), "csv")
        reader = csv.reader(io.StringIO(report))
        headers = next(reader)
        assert "trace_id" in headers
        assert "input_tokens" in headers
        assert "context_window_size" in headers
        assert "context_utilization" in headers

    def test_correct_row_count(self):
        report = format_report(_make_trace_analysis(), "csv")
        reader = csv.reader(io.StringIO(report))
        rows = list(reader)
        # 1 header + 2 data rows (2 LLM calls)
        assert len(rows) == 3

    def test_discovery_csv_falls_back_to_json(self):
        report = format_report(_make_discovery(), "csv")
        # Discovery CSV falls back to JSON
        data = json.loads(report)
        assert data["total_agents"] == 2


# ── write_reports tests ──────────────────────────────────────────────────


class TestWriteReports:
    def test_creates_files(self, tmp_path):
        analysis = _make_trace_analysis()
        files = write_reports(analysis, tmp_path)
        assert len(files) == 3
        assert any(f.name == "analysis.json" for f in files)
        assert any(f.name == "llm_calls.csv" for f in files)
        assert any(f.name == "agent_breakdown.csv" for f in files)

    def test_json_valid(self, tmp_path):
        analysis = _make_trace_analysis()
        write_reports(analysis, tmp_path)
        json_file = tmp_path / "analysis.json"
        data = json.loads(json_file.read_text())
        assert data["trace_id"] == "abc123"

    def test_csv_headers(self, tmp_path):
        analysis = _make_trace_analysis()
        write_reports(analysis, tmp_path)
        csv_file = tmp_path / "llm_calls.csv"
        reader = csv.reader(io.StringIO(csv_file.read_text()))
        headers = next(reader)
        assert "trace_id" in headers
        assert "context_window_size" in headers

    def test_agent_csv_headers(self, tmp_path):
        analysis = _make_trace_analysis()
        write_reports(analysis, tmp_path)
        csv_file = tmp_path / "agent_breakdown.csv"
        reader = csv.reader(io.StringIO(csv_file.read_text()))
        headers = next(reader)
        assert "agent_name" in headers
        assert "max_context_utilization" in headers

    def test_creates_directory(self, tmp_path):
        out_dir = tmp_path / "nested" / "output"
        analysis = _make_trace_analysis()
        files = write_reports(analysis, out_dir)
        assert out_dir.is_dir()
        assert len(files) == 3
