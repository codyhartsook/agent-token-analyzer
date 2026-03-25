"""Output rendering for terminal (ANSI), JSON, and CSV formats.

The ``format_report()`` function is the main entry point — returns a
string for terminal/JSON or writes files for CSV.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from .models import AgentDiscovery, TokenWindowAnalysis, TraceTokenAnalysis


# ── ANSI helpers ────────────────────────────────────────────────────────────

_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_CYAN = "\033[36m"
_RESET = "\033[0m"


def _header(title: str) -> str:
    return f"\n{_BOLD}{_CYAN}── {title} ──{_RESET}\n"


def _kv(key: str, value: str, indent: int = 2) -> str:
    pad = " " * indent
    return f"{pad}{_DIM}{key:<36}{_RESET}{value}"


def _comma(n: int | float) -> str:
    """Format number with comma separators."""
    if isinstance(n, float):
        return f"{n:,.1f}"
    return f"{n:,}"


def _pct(ratio: float) -> str:
    return f"{ratio * 100:.1f}%"


def _usd(amount: float) -> str:
    return f"${amount:.4f}"


def _ctx_color(ratio: float) -> str:
    """Color-code a context utilization ratio: green < 70%, yellow 70-90%, red 90%+."""
    if ratio >= 0.90:
        return f"{_RED}{_pct(ratio)}{_RESET}"
    if ratio >= 0.70:
        return f"{_YELLOW}{_pct(ratio)}{_RESET}"
    return f"{_GREEN}{_pct(ratio)}{_RESET}"


def _ctx_win_str(size: int) -> str:
    """Format a context window size as compact string (e.g. '128K', '2M')."""
    if size == 0:
        return "—"
    if size >= 1_000_000:
        return f"{size / 1_000_000:.0f}M"
    return f"{size // 1_000}K"


# ── Terminal report (single trace) ──────────────────────────────────────────


def _trace_report(t: TraceTokenAnalysis, *, verbose: bool = False) -> str:
    lines: list[str] = []

    lines.append(f"\n{_BOLD}╔{'═' * 62}╗{_RESET}")
    lines.append(f"{_BOLD}║  Token & Context Analysis{' ' * 36}║{_RESET}")
    lines.append(f"{_BOLD}╚{'═' * 62}╝{_RESET}")

    # Overview
    lines.append(_header("Overview"))
    lines.append(_kv("Trace ID:", t.trace_id))
    lines.append(_kv("Total spans:", str(t.total_spans)))
    lines.append(_kv("LLM calls (deduplicated):", str(t.total_llm_calls)))
    lines.append(_kv("Duration:", f"{_comma(t.total_duration_ms)} ms"))
    if t.service_chain:
        lines.append(_kv("Services:", " → ".join(t.service_chain)))

    # Token usage
    lines.append(_header("Token Usage (Deduplicated)"))
    lines.append(_kv("Input tokens:", _comma(t.total_input_tokens)))
    lines.append(_kv("Output tokens:", _comma(t.total_output_tokens)))
    lines.append(_kv("Cache read tokens:", _comma(t.total_cache_read_tokens)))
    lines.append(_kv("Reasoning tokens:", _comma(t.total_reasoning_tokens)))
    lines.append(_kv("Total tokens:", _comma(t.total_tokens)))
    lines.append(_kv("Cache hit ratio:", _pct(t.overall_cache_hit_ratio)))
    if t.max_context_utilization > 0:
        lines.append(
            _kv("Peak context utilization:", _ctx_color(t.max_context_utilization))
        )

    # Per-LLM-Call breakdown
    if t.llm_calls:
        has_any_estimated = any(c.tokens_estimated for c in t.llm_calls)
        lines.append(_header("Per-LLM-Call Breakdown"))
        # Header row
        hdr = (
            f"  {'#':<3} {'Agent':<28} {'Model':<14} "
            f"{'In Tok':>8} {'Out Tok':>8} {'Cache':>7} "
            f"{'Ctx %':>6} "
            f"{'Duration':>12}"
        )
        lines.append(f"{_DIM}{hdr}{_RESET}")
        lines.append(f"  {'─' * 98}")
        for i, c in enumerate(t.llm_calls, 1):
            agent = (c.agent_name or c.service_name)[:27]
            model = (c.model or c.response_model)[:13]
            est = "~" if c.tokens_estimated else ""
            if c.context_window_size > 0:
                ctx_str = _ctx_color(c.context_utilization)
            else:
                ctx_str = f"{'—':>6}"
            lines.append(
                f"  {i:<3} {agent:<28} {model:<14} "
                f"{est}{_comma(c.input_tokens):>7} {est}{_comma(c.output_tokens):>7} "
                f"{_comma(c.cache_read_input_tokens):>7} "
                f"{ctx_str:>6} "
                f"{_comma(c.duration_ms):>10} ms"
            )
        if has_any_estimated:
            lines.append(
                f"  {_DIM}~ = estimated from prompt/completion content "
                f"(~4 chars/token){_RESET}"
            )

    # Context window reconstruction
    if t.context_snapshots:
        lines.append(_header("Context Window Reconstruction"))
        for snap in t.context_snapshots:
            agent = snap.agent_name or "unknown"
            roles_str = ", ".join(
                f"{role}: {count}"
                for role, count in sorted(snap.messages_by_role.items())
            )
            lines.append(
                f"  Call #{snap.call_index + 1} ({agent}, {snap.model}):"
            )
            lines.append(
                f"    Messages: {snap.total_messages} ({roles_str})"
            )
            lines.append(
                f"    Content size: {_comma(snap.total_content_chars)} chars"
            )
            if snap.total_tool_calls > 0:
                lines.append(
                    f"    Tool calls in context: {snap.total_tool_calls}"
                )
            lines.append("")

    # Context growth
    if t.context_growth:
        lines.append(_header("Context Growth"))
        for step in t.context_growth:
            sign_msg = f"+{step.message_count_delta}" if step.message_count_delta >= 0 else str(step.message_count_delta)
            sign_char = f"+{_comma(step.content_chars_delta)}" if step.content_chars_delta >= 0 else _comma(step.content_chars_delta)
            sign_tok = f"+{_comma(step.input_tokens_delta)}" if step.input_tokens_delta >= 0 else _comma(step.input_tokens_delta)
            lines.append(
                f"  Call {step.from_call_index + 1} → {step.to_call_index + 1}:  "
                f"{sign_msg} messages, {sign_char} chars ({step.growth_pct}%), "
                f"{sign_tok} input tokens"
            )

    # Accumulation alerts
    if t.accumulation_alerts:
        lines.append(_header("⚠ Context Accumulation Alerts"))
        for alert in t.accumulation_alerts:
            color = _RED if alert.severity == "critical" else _YELLOW
            lines.append(
                f"  {color}[{alert.severity.upper()}]{_RESET} "
                f"Agent '{alert.agent_name}': context grew {alert.growth_factor}x "
                f"over {alert.call_count} calls "
                f"({_comma(alert.initial_input_tokens)} → "
                f"{_comma(alert.final_input_tokens)} input tokens)"
            )

    # Per-agent breakdown
    if t.agent_breakdown:
        lines.append(_header("Per-Agent Token Breakdown"))
        hdr = (
            f"  {'Agent':<28} {'LLM Calls':>10} {'In Tok':>10} "
            f"{'Out Tok':>10} {'Cache':>10} {'Cache %':>8} {'Max Ctx':>8}"
        )
        lines.append(f"{_DIM}{hdr}{_RESET}")
        lines.append(f"  {'─' * 90}")
        for ab in sorted(
            t.agent_breakdown.values(),
            key=lambda x: x.total_tokens,
            reverse=True,
        ):
            ctx_str = _pct(ab.max_context_utilization) if ab.context_window_size > 0 else "  —"
            lines.append(
                f"  {ab.agent_name:<28} {ab.llm_call_count:>10} "
                f"{_comma(ab.input_tokens):>10} "
                f"{_comma(ab.output_tokens):>10} "
                f"{_comma(ab.cache_read_tokens):>10} "
                f"{_pct(ab.cache_hit_ratio):>8} "
                f"{ctx_str:>8}"
            )

    # Cost estimate
    if t.cost.total_cost_usd > 0:
        lines.append(_header("Cost Estimate"))
        lines.append(_kv("Total estimated cost:", _usd(t.cost.total_cost_usd)))
        lines.append(_kv("Input cost:", _usd(t.cost.input_cost_usd)))
        lines.append(_kv("Output cost:", _usd(t.cost.output_cost_usd)))
        if t.cost.cached_input_cost_usd > 0:
            lines.append(
                _kv("Cached input cost:", _usd(t.cost.cached_input_cost_usd))
            )
        if t.cost.reasoning_cost_usd > 0:
            lines.append(
                _kv("Reasoning cost:", _usd(t.cost.reasoning_cost_usd))
            )
        if t.cost.calls_without_pricing > 0:
            lines.append(
                _kv(
                    "Calls without pricing:",
                    str(t.cost.calls_without_pricing),
                )
            )

    return "\n".join(lines)


# ── Terminal report (window) ────────────────────────────────────────────────


def _window_report(w: TokenWindowAnalysis, *, verbose: bool = False) -> str:
    lines: list[str] = []

    lines.append(f"\n{_BOLD}╔{'═' * 62}╗{_RESET}")
    lines.append(f"{_BOLD}║  Token & Context Window Analysis{' ' * 29}║{_RESET}")
    lines.append(f"{_BOLD}╚{'═' * 62}╝{_RESET}")

    # Overview
    lines.append(_header("Window Overview"))
    lines.append(_kv("Time range:", f"{w.start_time} → {w.end_time}"))
    lines.append(_kv("Traces analyzed:", str(w.trace_count)))
    lines.append(_kv("Total LLM calls:", str(w.total_llm_calls)))

    # Aggregate tokens
    lines.append(_header("Aggregate Token Usage"))
    lines.append(_kv("Input tokens:", _comma(w.total_input_tokens)))
    lines.append(_kv("Output tokens:", _comma(w.total_output_tokens)))
    lines.append(_kv("Cache read tokens:", _comma(w.total_cache_read_tokens)))
    lines.append(_kv("Reasoning tokens:", _comma(w.total_reasoning_tokens)))
    lines.append(_kv("Total tokens:", _comma(w.total_tokens)))
    lines.append(_kv("Cache hit ratio:", _pct(w.overall_cache_hit_ratio)))

    # Percentiles
    lines.append(_header("Percentiles"))
    lines.append(
        _kv("Tokens/trace (p50/p95/p99):",
            f"{_comma(w.p50_tokens_per_trace)} / "
            f"{_comma(w.p95_tokens_per_trace)} / "
            f"{_comma(w.p99_tokens_per_trace)}")
    )
    lines.append(
        _kv("Input tokens/call (p50/p95):",
            f"{_comma(w.p50_input_per_call)} / {_comma(w.p95_input_per_call)}")
    )
    lines.append(
        _kv("Context messages (p50/p95):",
            f"{_comma(w.p50_context_messages)} / {_comma(w.p95_context_messages)}")
    )
    if w.max_context_utilization > 0:
        lines.append(
            _kv("Context utilization (p50/p95/max):",
                f"{_pct(w.p50_context_utilization)} / "
                f"{_pct(w.p95_context_utilization)} / "
                f"{_ctx_color(w.max_context_utilization)}")
        )

    # Per-agent breakdown
    if w.agent_breakdown:
        lines.append(_header("Per-Agent Token Breakdown"))
        hdr = (
            f"  {'Agent':<28} {'LLM Calls':>10} {'In Tok':>10} "
            f"{'Out Tok':>10} {'Avg In/Call':>12} {'Cache %':>8} {'Max Ctx':>8}"
        )
        lines.append(f"{_DIM}{hdr}{_RESET}")
        lines.append(f"  {'─' * 92}")
        for ab in sorted(
            w.agent_breakdown.values(),
            key=lambda x: x.total_tokens,
            reverse=True,
        ):
            ctx_str = _pct(ab.max_context_utilization) if ab.context_window_size > 0 else "  —"
            lines.append(
                f"  {ab.agent_name:<28} {ab.llm_call_count:>10} "
                f"{_comma(ab.input_tokens):>10} "
                f"{_comma(ab.output_tokens):>10} "
                f"{_comma(ab.avg_input_per_call):>12} "
                f"{_pct(ab.cache_hit_ratio):>8} "
                f"{ctx_str:>8}"
            )

    # Accumulation alerts
    if w.accumulation_alerts:
        lines.append(_header(f"⚠ Context Accumulation Alerts ({len(w.accumulation_alerts)})"))
        for alert in w.accumulation_alerts:
            color = _RED if alert.severity == "critical" else _YELLOW
            lines.append(
                f"  {color}[{alert.severity.upper()}]{_RESET} "
                f"Trace {alert.trace_id[:12]}... agent '{alert.agent_name}': "
                f"context grew {alert.growth_factor}x over {alert.call_count} calls"
            )

    # Cost
    if w.cost.total_cost_usd > 0:
        lines.append(_header("Aggregate Cost Estimate"))
        lines.append(_kv("Total estimated cost:", _usd(w.cost.total_cost_usd)))
        lines.append(_kv("Input cost:", _usd(w.cost.input_cost_usd)))
        lines.append(_kv("Output cost:", _usd(w.cost.output_cost_usd)))
        if w.cost.per_agent:
            lines.append("")
            lines.append(f"  {_DIM}Per-agent cost:{_RESET}")
            for agent, cost in sorted(
                w.cost.per_agent.items(), key=lambda x: x[1], reverse=True
            ):
                lines.append(f"    {agent:<28} {_usd(cost)}")

    # Per-trace summaries (verbose)
    if verbose and w.traces:
        lines.append(_header(f"Per-Trace Details ({len(w.traces)} traces)"))
        for t in w.traces:
            lines.append(
                f"  {t.trace_id[:16]}...  "
                f"LLM calls: {t.total_llm_calls}  "
                f"Tokens: {_comma(t.total_tokens)}  "
                f"Duration: {_comma(t.total_duration_ms)} ms"
            )

    return "\n".join(lines)


# ── JSON format ─────────────────────────────────────────────────────────────


def _json_report(analysis: TraceTokenAnalysis | TokenWindowAnalysis) -> str:
    """Serialize analysis to JSON."""
    return analysis.model_dump_json(indent=2)


# ── CSV format ──────────────────────────────────────────────────────────────


def _csv_llm_calls(
    analysis: TraceTokenAnalysis | TokenWindowAnalysis,
) -> str:
    """Generate CSV for LLM call breakdown."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "trace_id", "span_id", "agent_name", "model", "input_tokens",
        "output_tokens", "cache_read_tokens", "reasoning_tokens",
        "total_tokens", "cache_hit_ratio", "context_window_size",
        "context_utilization", "duration_ms", "finish_reason",
    ])

    if isinstance(analysis, TraceTokenAnalysis):
        traces = [analysis]
    else:
        traces = analysis.traces

    for t in traces:
        for c in t.llm_calls:
            writer.writerow([
                c.trace_id, c.span_id, c.agent_name, c.model,
                c.input_tokens, c.output_tokens, c.cache_read_input_tokens,
                c.reasoning_tokens, c.total_tokens, c.cache_hit_ratio,
                c.context_window_size, c.context_utilization,
                round(c.duration_ms, 1), c.finish_reason,
            ])

    return buf.getvalue()


def _csv_agent_breakdown(
    analysis: TraceTokenAnalysis | TokenWindowAnalysis,
) -> str:
    """Generate CSV for per-agent breakdown."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "agent_name", "service_name", "llm_call_count", "input_tokens",
        "output_tokens", "cache_read_tokens", "total_tokens",
        "avg_input_per_call", "cache_hit_ratio", "context_window_size",
        "max_context_utilization", "estimated_cost_usd",
    ])

    breakdown = analysis.agent_breakdown
    for ab in sorted(
        breakdown.values(), key=lambda x: x.total_tokens, reverse=True
    ):
        writer.writerow([
            ab.agent_name, ab.service_name, ab.llm_call_count,
            ab.input_tokens, ab.output_tokens, ab.cache_read_tokens,
            ab.total_tokens, round(ab.avg_input_per_call, 1),
            round(ab.cache_hit_ratio, 4), ab.context_window_size,
            round(ab.max_context_utilization, 4),
            round(ab.estimated_cost_usd, 6),
        ])

    return buf.getvalue()


# ── Terminal report (discovery) ──────────────────────────────────────────────


def _discovery_report(d: AgentDiscovery) -> str:
    lines: list[str] = []

    lines.append(f"\n{_BOLD}╔{'═' * 62}╗{_RESET}")
    lines.append(f"{_BOLD}║  Agent Discovery{' ' * 45}║{_RESET}")
    lines.append(f"{_BOLD}╚{'═' * 62}╝{_RESET}")

    lines.append(_header("Overview"))
    lines.append(_kv("Time range:", f"{d.start_time} → {d.end_time}"))
    lines.append(_kv("Total traces:", str(d.total_traces)))
    lines.append(_kv("Distinct agents:", str(d.total_agents)))
    lines.append(_kv("Distinct services:", str(d.total_services)))

    if d.services:
        lines.append(_header("Services"))
        for svc in d.services:
            lines.append(f"  • {svc}")

    if d.agents:
        has_any_estimated = any(a.tokens_estimated for a in d.agents)
        lines.append(_header("Discovered Agents"))
        # Table header
        hdr = (
            f"  {'Agent':<30} {'Service':<24} "
            f"{'Traces':>7} {'LLM':>5} "
            f"{'In Tok':>9} {'Out Tok':>9} "
            f"{'Ctx Win':>8} {'Peak':>6} "
            f"{'Models'}"
        )
        lines.append(f"{_DIM}{hdr}{_RESET}")
        lines.append(f"  {'─' * 118}")

        for a in d.agents:
            models_str = ", ".join(a.models_used) if a.models_used else "—"
            agent_display = a.agent_name[:29]
            svc_display = a.service_name[:23]
            est = "~" if a.tokens_estimated else ""
            ctx_win = _ctx_win_str(a.context_window_size)
            peak_ctx = _pct(a.max_context_utilization) if a.max_context_utilization > 0 else "—"
            lines.append(
                f"  {agent_display:<30} {svc_display:<24} "
                f"{a.trace_count:>7} {a.llm_call_count:>5} "
                f"{est}{_comma(a.total_input_tokens):>8} "
                f"{est}{_comma(a.total_output_tokens):>8} "
                f"{ctx_win:>8} {peak_ctx:>6} "
                f"{models_str}"
            )

            # Show sub-agents, span kinds, and activity window on subsequent lines
            extras: list[str] = []
            if a.sub_agents:
                extras.append(f"agents: {', '.join(a.sub_agents)}")
            if a.span_kinds:
                extras.append(f"kinds: {', '.join(a.span_kinds)}")
            if a.first_seen and a.last_seen and a.first_seen != a.last_seen:
                extras.append(f"active: {a.first_seen[:19]} → {a.last_seen[:19]}")
            elif a.first_seen:
                extras.append(f"seen: {a.first_seen[:19]}")
            if extras:
                lines.append(f"  {'':<30} {_DIM}{' | '.join(extras)}{_RESET}")

        if has_any_estimated:
            lines.append(
                f"  {_DIM}~ = estimated from prompt/completion content "
                f"(~4 chars/token){_RESET}"
            )

    # Usage hint
    lines.append("")
    lines.append(
        f"  {_DIM}Use --agent <name> with --last/--start/--end "
        f"to filter analysis to a specific agent.{_RESET}"
    )
    lines.append(
        f"  {_DIM}Example: uv run python -m token_analysis "
        f"--last 1h --agent recruiter_supervisor --cost{_RESET}"
    )

    return "\n".join(lines)


# ── Public API ──────────────────────────────────────────────────────────────


def format_report(
    analysis: TraceTokenAnalysis | TokenWindowAnalysis | AgentDiscovery,
    fmt: str = "terminal",
    *,
    verbose: bool = False,
) -> str:
    """Format analysis results.

    Args:
        analysis: The analysis result to format.
        fmt: One of ``"terminal"``, ``"json"``, ``"csv"``.
        verbose: Include per-trace details (terminal only).

    Returns:
        Formatted string.
    """
    if fmt == "json":
        return _json_report(analysis)
    if fmt == "csv":
        if isinstance(analysis, AgentDiscovery):
            return _json_report(analysis)  # CSV doesn't apply to discovery
        return _csv_llm_calls(analysis)
    # Terminal
    if isinstance(analysis, AgentDiscovery):
        return _discovery_report(analysis)
    if isinstance(analysis, TraceTokenAnalysis):
        return _trace_report(analysis, verbose=verbose)
    return _window_report(analysis, verbose=verbose)


def write_reports(
    analysis: TraceTokenAnalysis | TokenWindowAnalysis,
    output_dir: str | Path,
    *,
    verbose: bool = False,
) -> list[Path]:
    """Write all output formats to a directory.

    Returns list of files written.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []

    # JSON
    json_path = out / "analysis.json"
    json_path.write_text(_json_report(analysis))
    files.append(json_path)

    # CSV - LLM calls
    csv_calls = out / "llm_calls.csv"
    csv_calls.write_text(_csv_llm_calls(analysis))
    files.append(csv_calls)

    # CSV - Agent breakdown
    csv_agents = out / "agent_breakdown.csv"
    csv_agents.write_text(_csv_agent_breakdown(analysis))
    files.append(csv_agents)

    return files
