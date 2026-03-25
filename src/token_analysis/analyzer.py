"""Core analysis pipeline for token and context window analysis.

Provides three main entry points designed for both CLI and ADK agent usage:

- ``discover_agents(client, start_ns, end_ns)`` → ``AgentDiscovery``
- ``analyze_trace_tokens(client, trace_id)`` → ``TraceTokenAnalysis``
- ``analyze_window(client, start_ns, end_ns)`` → ``TokenWindowAnalysis``
"""

from __future__ import annotations

from datetime import datetime, timezone
from statistics import median, quantiles
from typing import Any

import clickhouse_connect

from . import queries
from .context import compute_context_growth, detect_accumulation, reconstruct_context
from .context_window import get_context_utilization
from .cost import estimate_cost
from .models import (
    AgentDiscovery,
    AgentTokenBreakdown,
    CostEstimate,
    DiscoveredAgent,
    LLMCallTokens,
    TokenWindowAnalysis,
    TraceTokenAnalysis,
)


# ── Agent identity resolution ───────────────────────────────────────────────


def _resolve_agent_identity(
    attrs: dict[str, str], service_name: str
) -> str:
    """Resolve agent identity from span attributes.

    Priority: gen_ai.agent.name > ioa_observe.entity.name > ServiceName
    """
    return (
        attrs.get("gen_ai.agent.name", "").strip()
        or attrs.get("ioa_observe.entity.name", "").strip()
        or service_name
    )


# ── Span parsing ────────────────────────────────────────────────────────────


def _to_int(val: str | Any) -> int:
    """Safely convert a span attribute value to int."""
    if not val:
        return 0
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0


def _timestamp_to_ns(ts: Any) -> int:
    """Convert a ClickHouse Timestamp to nanosecond epoch."""
    if isinstance(ts, datetime):
        epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return int((ts - epoch).total_seconds() * 1_000_000_000)
    if isinstance(ts, (int, float)):
        return int(ts)
    return 0


def _estimate_tokens_from_text(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English + code/JSON.

    This is a standard heuristic for GPT-family tokenizers.  It errs on
    the side of slightly over-counting, which is safer for cost estimates.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def _collect_prompt_content(attrs: dict[str, str]) -> str:
    """Collect all prompt content from gen_ai.prompt.{N}.content attributes."""
    parts: list[str] = []
    i = 0
    while True:
        content = attrs.get(f"gen_ai.prompt.{i}.content", "")
        role = attrs.get(f"gen_ai.prompt.{i}.role", "")
        if not role:
            break
        if content:
            parts.append(content)
        # Also count tool_call arguments as they consume tokens
        j = 0
        while True:
            tc_args = attrs.get(f"gen_ai.prompt.{i}.tool_calls.{j}.arguments", "")
            tc_name = attrs.get(f"gen_ai.prompt.{i}.tool_calls.{j}.name", "")
            if not tc_args and not tc_name:
                break
            if tc_args:
                parts.append(tc_args)
            if tc_name:
                parts.append(tc_name)
            j += 1
        i += 1
    return "\n".join(parts)


def _collect_completion_content(attrs: dict[str, str]) -> str:
    """Collect completion content from gen_ai.completion.0.content."""
    return attrs.get("gen_ai.completion.0.content", "")


def _parse_llm_call(row: dict[str, Any], trace_id: str) -> LLMCallTokens:
    """Parse a deduplicated LLM span row into an LLMCallTokens model."""
    attrs = row.get("span_attributes", {})
    duration_ns = row.get("duration", 0)
    service = row.get("service_name", "")

    # Resolve agent identity by walking up to parent agent spans
    # For leaf LLM spans, agent info may be on the span itself or inherited
    agent = _resolve_agent_identity(attrs, service)

    input_tokens = _to_int(
        attrs.get("gen_ai.usage.input_tokens")
        or attrs.get("gen_ai.usage.prompt_tokens")
    )
    output_tokens = _to_int(
        attrs.get("gen_ai.usage.output_tokens")
        or attrs.get("gen_ai.usage.completion_tokens")
    )
    total_tokens = _to_int(attrs.get("llm.usage.total_tokens"))

    # Estimate tokens from content when the LLM provider/instrumentation
    # doesn't report token counts (e.g. LlamaIndex traceloop spans)
    tokens_estimated = False
    if input_tokens == 0 and output_tokens == 0:
        # Collect all available input content and use the LARGEST estimate.
        # gen_ai.prompt.{N}.content has the short message text, but
        # traceloop.entity.input includes tool schemas, JSON structure,
        # etc. — which is closer to what the model actually tokenizes.
        input_candidates: list[str] = []

        prompt_content = _collect_prompt_content(attrs)
        if prompt_content:
            input_candidates.append(prompt_content)

        traceloop_input = attrs.get("traceloop.entity.input", "")
        if traceloop_input:
            input_candidates.append(traceloop_input)

        ioa_input = attrs.get("ioa_observe.entity.input", "")
        if ioa_input:
            input_candidates.append(ioa_input)

        # Use the largest content source (most complete representation)
        prompt_text = max(input_candidates, key=len) if input_candidates else ""

        # For completion, also take the largest
        output_candidates: list[str] = []
        completion_content = _collect_completion_content(attrs)
        if completion_content:
            output_candidates.append(completion_content)
        traceloop_output = attrs.get("traceloop.entity.output", "")
        if traceloop_output:
            output_candidates.append(traceloop_output)
        ioa_output = attrs.get("ioa_observe.entity.output", "")
        if ioa_output:
            output_candidates.append(ioa_output)

        completion_text = max(output_candidates, key=len) if output_candidates else ""

        if prompt_text or completion_text:
            input_tokens = _estimate_tokens_from_text(prompt_text)
            output_tokens = _estimate_tokens_from_text(completion_text)
            tokens_estimated = True

    if total_tokens == 0 and (input_tokens > 0 or output_tokens > 0):
        total_tokens = input_tokens + output_tokens

    # Context window utilization
    model_key = attrs.get("gen_ai.request.model", "")
    resp_model = attrs.get("gen_ai.response.model", "")
    ctx_size, ctx_util = get_context_utilization(
        input_tokens, model_key, resp_model
    )

    return LLMCallTokens(
        span_id=row.get("span_id", ""),
        trace_id=trace_id,
        span_name=row.get("span_name", ""),
        service_name=service,
        agent_name=agent,
        model=attrs.get("gen_ai.request.model", ""),
        response_model=attrs.get("gen_ai.response.model", ""),
        timestamp_ns=_timestamp_to_ns(row.get("timestamp")),
        duration_ms=duration_ns / 1_000_000 if duration_ns else 0.0,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=_to_int(
            attrs.get("gen_ai.usage.cache_read_input_tokens")
        ),
        reasoning_tokens=_to_int(
            attrs.get("gen_ai.usage.reasoning_tokens")
        ),
        total_tokens=total_tokens,
        tokens_estimated=tokens_estimated,
        context_window_size=ctx_size,
        context_utilization=ctx_util,
        finish_reason=attrs.get("gen_ai.completion.0.finish_reason", "")
            or attrs.get("llm.response.finish_reason", ""),
    )


# ── Per-agent breakdown ─────────────────────────────────────────────────────


def _build_agent_breakdown(
    calls: list[LLMCallTokens],
) -> dict[str, AgentTokenBreakdown]:
    """Group LLM calls by agent identity and aggregate."""
    agents: dict[str, AgentTokenBreakdown] = {}
    # Track context utilization sums for averaging
    ctx_util_sums: dict[str, float] = {}
    ctx_util_counts: dict[str, int] = {}

    for call in calls:
        agent = call.agent_name or call.service_name or "unknown"

        if agent not in agents:
            agents[agent] = AgentTokenBreakdown(
                agent_name=agent,
                service_name=call.service_name,
            )

        ab = agents[agent]
        ab.llm_call_count += 1
        ab.input_tokens += call.input_tokens
        ab.output_tokens += call.output_tokens
        ab.cache_read_tokens += call.cache_read_input_tokens
        ab.reasoning_tokens += call.reasoning_tokens
        ab.total_tokens += call.total_tokens

        model_key = call.model or call.response_model or "unknown"
        ab.models_used[model_key] = ab.models_used.get(model_key, 0) + 1

        # Track context window utilization
        if call.context_window_size > 0:
            if call.context_window_size > ab.context_window_size:
                ab.context_window_size = call.context_window_size
            ab.max_context_utilization = max(
                ab.max_context_utilization, call.context_utilization
            )
            ctx_util_sums[agent] = ctx_util_sums.get(agent, 0.0) + call.context_utilization
            ctx_util_counts[agent] = ctx_util_counts.get(agent, 0) + 1

    # Compute averages and ratios
    for name, ab in agents.items():
        if ab.llm_call_count > 0:
            ab.avg_input_per_call = round(
                ab.input_tokens / ab.llm_call_count, 1
            )
            ab.avg_output_per_call = round(
                ab.output_tokens / ab.llm_call_count, 1
            )
        if ab.input_tokens > 0:
            ab.cache_hit_ratio = round(
                ab.cache_read_tokens / ab.input_tokens, 4
            )
        if name in ctx_util_counts and ctx_util_counts[name] > 0:
            ab.avg_context_utilization = round(
                ctx_util_sums[name] / ctx_util_counts[name], 4
            )

    return agents


# ── Service chain extraction ────────────────────────────────────────────────


def _extract_service_chain(spans: list[dict[str, Any]]) -> list[str]:
    """Extract ordered unique service names from spans."""
    seen: set[str] = set()
    chain: list[str] = []
    for span in spans:
        svc = span.get("service_name", "")
        if svc and svc not in seen:
            seen.add(svc)
            chain.append(svc)
    return chain


# ── Agent discovery ──────────────────────────────────────────────────────────


def discover_agents(
    client: clickhouse_connect.driver.Client,
    start_ns: int,
    end_ns: int,
) -> AgentDiscovery:
    """Discover all agents, services, and models active in a time window.

    Use this before ``analyze_window()`` or ``analyze_trace_tokens()`` to
    find which agent names can be used with ``--agent``.

    Token counts are computed Python-side via ``_parse_llm_call`` so that
    content-based estimation works for providers that don't report token
    usage (e.g. LlamaIndex / traceloop instrumentation).

    Results are **consolidated by ServiceName** — agents from the same
    service (e.g. LLM spans resolving to ``lungo.brazil_farm`` and non-LLM
    agent spans resolving to ``brazil_farm_agent.llama_index``) are merged
    into a single row.  Non-LLM agent/tool names within the service appear
    in the ``sub_agents`` field.

    Returns an ``AgentDiscovery`` model listing every agent with its
    service, trace count, token usage, models used, and activity window.
    """
    from .client import from_nano

    # 1. Fetch all deduplicated LLM spans with full attributes
    llm_rows = queries.get_deduplicated_llm_calls_window(client, start_ns, end_ns)

    # 2. Parse each through _parse_llm_call (handles estimation)
    llm_calls = [_parse_llm_call(row, row.get("trace_id", "")) for row in llm_rows]

    # 3. Aggregate LLM calls per SERVICE (not per resolved agent name).
    #    LLM spans almost always resolve to ServiceName anyway, and this
    #    prevents double-counting when non-LLM spans use a different name.
    svc_map: dict[str, dict[str, Any]] = {}
    for call in llm_calls:
        svc = call.service_name or "unknown"
        if svc not in svc_map:
            svc_map[svc] = {
                "service_name": svc,
                "agent_names": set(),
                "trace_ids": set(),
                "llm_call_count": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_tokens": 0,
                "models": set(),
                "timestamps": [],
                "has_estimated": False,
                "span_kinds": set(),
                "sub_agents": set(),
                "context_window_size": 0,
                "max_context_utilization": 0.0,
            }
        entry = svc_map[svc]
        entry["agent_names"].add(call.agent_name or svc)
        entry["trace_ids"].add(call.trace_id)
        entry["llm_call_count"] += 1
        entry["total_input_tokens"] += call.input_tokens
        entry["total_output_tokens"] += call.output_tokens
        entry["total_tokens"] += call.total_tokens
        if call.model:
            entry["models"].add(call.model)
        entry["timestamps"].append(call.timestamp_ns)
        if call.tokens_estimated:
            entry["has_estimated"] = True
        if call.context_window_size > 0:
            entry["context_window_size"] = max(
                entry["context_window_size"], call.context_window_size
            )
            entry["max_context_utilization"] = max(
                entry["max_context_utilization"], call.context_utilization
            )

    # 4. Also discover non-LLM agents (invoke_agent, ioa_observe agent spans)
    raw = queries.discover_agents(client, start_ns, end_ns)
    non_llm_agents = raw["agents"]

    # Merge non-LLM agent info into the service-keyed map.
    # Non-LLM agents belong to the same logical service — they appear as
    # sub_agents under the service row.
    for a in non_llm_agents:
        name = a["agent_name"]
        svc = a.get("service_name", "")
        span_kinds = a.get("span_kinds", [])

        if svc in svc_map:
            # Same service as an LLM entry — merge as sub-agent
            entry = svc_map[svc]
            # Only add as sub-agent if it's a different name from the service
            if name != svc:
                entry["sub_agents"].add(name)
            entry["span_kinds"].update(span_kinds)
            # Use broader trace count
            non_llm_traces = a.get("trace_count", 0)
            if non_llm_traces > len(entry["trace_ids"]):
                entry["trace_count_override"] = max(
                    entry.get("trace_count_override", 0), non_llm_traces
                )
            # Wider time range
            if a.get("first_seen"):
                entry.setdefault("first_seen_str", a["first_seen"])
                if a["first_seen"] < entry.get("first_seen_str", "9999"):
                    entry["first_seen_str"] = a["first_seen"]
            if a.get("last_seen"):
                entry.setdefault("last_seen_str", a["last_seen"])
                if a["last_seen"] > entry.get("last_seen_str", ""):
                    entry["last_seen_str"] = a["last_seen"]
        elif name in svc_map:
            # Agent name matches a service key (unlikely but handle it)
            entry = svc_map[name]
            entry["span_kinds"].update(span_kinds)
        else:
            # Service we haven't seen from LLM spans — pure orchestrator.
            # Use the service_name as the key if available, else the agent name.
            key = svc or name
            if key not in svc_map:
                svc_map[key] = {
                    "service_name": svc,
                    "agent_names": {name},
                    "trace_ids": set(),
                    "llm_call_count": 0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_tokens": 0,
                    "models": set(),
                    "timestamps": [],
                    "has_estimated": False,
                    "span_kinds": set(span_kinds),
                    "sub_agents": {name} if name != svc else set(),
                    "trace_count_override": a.get("trace_count", 0),
                    "first_seen_str": a.get("first_seen", ""),
                    "last_seen_str": a.get("last_seen", ""),
                    "context_window_size": 0,
                    "max_context_utilization": 0.0,
                }
            else:
                entry = svc_map[key]
                if name != key:
                    entry["sub_agents"].add(name)
                entry["span_kinds"].update(span_kinds)
                non_llm_traces = a.get("trace_count", 0)
                if non_llm_traces > entry.get("trace_count_override", 0):
                    entry["trace_count_override"] = non_llm_traces

    # 5. Build DiscoveredAgent list — one per service
    agents: list[DiscoveredAgent] = []
    for entry in sorted(
        svc_map.values(),
        key=lambda e: e["total_tokens"],
        reverse=True,
    ):
        svc = entry["service_name"]
        trace_count = entry.get(
            "trace_count_override", len(entry["trace_ids"])
        )
        trace_count = max(trace_count, len(entry["trace_ids"]))
        timestamps = entry["timestamps"]

        # Pick the best display name: use the service name (which is what
        # users pass to --agent).  All distinct agent names within the
        # service are listed in sub_agents for clarity.
        display_name = svc or next(iter(entry.get("agent_names", {"unknown"})))

        # Resolve first/last seen
        first_seen = entry.get("first_seen_str", "")
        last_seen = entry.get("last_seen_str", "")
        if timestamps:
            ts_first = from_nano(min(timestamps))
            ts_last = from_nano(max(timestamps))
            if not first_seen or ts_first < first_seen:
                first_seen = ts_first
            if not last_seen or ts_last > last_seen:
                last_seen = ts_last

        agents.append(
            DiscoveredAgent(
                agent_name=display_name,
                service_name=svc,
                trace_count=trace_count,
                llm_call_count=entry["llm_call_count"],
                total_input_tokens=entry["total_input_tokens"],
                total_output_tokens=entry["total_output_tokens"],
                total_tokens=entry["total_tokens"],
                tokens_estimated=entry["has_estimated"],
                models_used=sorted(entry["models"]),
                first_seen=first_seen,
                last_seen=last_seen,
                span_kinds=sorted(entry.get("span_kinds", set())),
                sub_agents=sorted(entry.get("sub_agents", set())),
                context_window_size=entry.get("context_window_size", 0),
                max_context_utilization=entry.get("max_context_utilization", 0.0),
            )
        )

    return AgentDiscovery(
        start_time=from_nano(start_ns),
        end_time=from_nano(end_ns),
        total_agents=len(agents),
        total_services=len(raw["services"]),
        total_traces=raw["total_traces"],
        agents=agents,
        services=raw["services"],
    )


# ── Single-trace analysis ───────────────────────────────────────────────────


def analyze_trace_tokens(
    client: clickhouse_connect.driver.Client,
    trace_id: str,
    *,
    include_cost: bool = False,
    growth_factor_warn: float = 2.0,
    growth_factor_critical: float = 5.0,
) -> TraceTokenAnalysis:
    """Full token + context analysis for a single trace.

    Returns a ``TraceTokenAnalysis`` model with all breakdowns.
    """
    # 1. Fetch all spans (for tree structure + metadata)
    all_spans = queries.get_all_trace_spans(client, trace_id)

    # 2. Fetch deduplicated LLM calls (for tokens)
    llm_rows = queries.get_deduplicated_llm_calls(client, trace_id)
    llm_calls = [_parse_llm_call(row, trace_id) for row in llm_rows]

    # 3. Fetch context window spans (for prompt reconstruction)
    context_rows = queries.get_context_window_spans(client, trace_id)
    context_snapshots = []
    for i, row in enumerate(context_rows):
        attrs = row.get("span_attributes", {})
        agent = _resolve_agent_identity(attrs, row.get("service_name", ""))
        snapshot = reconstruct_context(
            span_attributes=attrs,
            span_id=row.get("span_id", ""),
            call_index=i,
            agent_name=agent,
            model=attrs.get("gen_ai.request.model", ""),
            timestamp_ns=_timestamp_to_ns(row.get("timestamp")),
        )
        context_snapshots.append(snapshot)

    # 4. Context growth + accumulation detection
    growth_steps = compute_context_growth(context_snapshots, llm_calls)
    accumulation_alerts = detect_accumulation(
        trace_id,
        llm_calls,
        growth_steps,
        growth_factor_warn=growth_factor_warn,
        growth_factor_critical=growth_factor_critical,
    )

    # 5. Per-agent breakdown
    agent_breakdown = _build_agent_breakdown(llm_calls)

    # 6. Aggregate token totals
    total_input = sum(c.input_tokens for c in llm_calls)
    total_output = sum(c.output_tokens for c in llm_calls)
    total_cache = sum(c.cache_read_input_tokens for c in llm_calls)
    total_reasoning = sum(c.reasoning_tokens for c in llm_calls)
    total_tokens = sum(c.total_tokens for c in llm_calls)
    cache_ratio = round(total_cache / total_input, 4) if total_input > 0 else 0.0

    # 7. Root span duration
    root_duration_ms = 0.0
    for span in all_spans:
        if not span.get("parent_span_id"):
            root_duration_ms = span.get("duration", 0) / 1_000_000
            break

    # 8. Max context utilization across all calls
    max_ctx_util = max(
        (c.context_utilization for c in llm_calls if c.context_window_size > 0),
        default=0.0,
    )

    # 9. Cost estimation
    cost = CostEstimate()
    if include_cost:
        cost = estimate_cost(llm_calls)

    # 10. Update agent cost from aggregate
    if include_cost:
        for agent_name, agent_cost in cost.per_agent.items():
            if agent_name in agent_breakdown:
                agent_breakdown[agent_name].estimated_cost_usd = agent_cost

    return TraceTokenAnalysis(
        trace_id=trace_id,
        total_spans=len(all_spans),
        total_llm_calls=len(llm_calls),
        total_duration_ms=root_duration_ms,
        service_chain=_extract_service_chain(all_spans),
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        total_cache_read_tokens=total_cache,
        total_reasoning_tokens=total_reasoning,
        total_tokens=total_tokens,
        overall_cache_hit_ratio=cache_ratio,
        max_context_utilization=max_ctx_util,
        llm_calls=llm_calls,
        context_snapshots=context_snapshots,
        context_growth=growth_steps,
        accumulation_alerts=accumulation_alerts,
        agent_breakdown=agent_breakdown,
        cost=cost,
    )


# ── Time-window analysis ────────────────────────────────────────────────────


def analyze_window(
    client: clickhouse_connect.driver.Client,
    start_ns: int,
    end_ns: int,
    *,
    agent_filter: str | None = None,
    include_cost: bool = False,
    limit: int = 100,
    growth_factor_warn: float = 2.0,
    growth_factor_critical: float = 5.0,
) -> TokenWindowAnalysis:
    """Aggregate token + context analysis across a time window.

    Returns a ``TokenWindowAnalysis`` model with per-trace details
    and aggregate statistics.
    """
    from .client import from_nano

    # 1. Get trace IDs
    if agent_filter:
        trace_ids = queries.get_traces_by_agent(
            client, agent_filter, start_ns, end_ns, limit=limit
        )
    else:
        trace_ids = queries.get_trace_ids(
            client, start_ns, end_ns, limit=limit
        )

    # 2. Analyze each trace
    traces: list[TraceTokenAnalysis] = []
    for tid in trace_ids:
        analysis = analyze_trace_tokens(
            client,
            tid,
            include_cost=include_cost,
            growth_factor_warn=growth_factor_warn,
            growth_factor_critical=growth_factor_critical,
        )
        traces.append(analysis)

    # 3. Aggregate
    return _aggregate_window(
        traces, start_ns, end_ns, from_nano
    )


def _aggregate_window(
    traces: list[TraceTokenAnalysis],
    start_ns: int,
    end_ns: int,
    from_nano_fn: Any,
) -> TokenWindowAnalysis:
    """Merge per-trace analyses into a window aggregate."""
    if not traces:
        return TokenWindowAnalysis(
            start_time=from_nano_fn(start_ns),
            end_time=from_nano_fn(end_ns),
        )

    # Aggregate tokens
    total_input = sum(t.total_input_tokens for t in traces)
    total_output = sum(t.total_output_tokens for t in traces)
    total_cache = sum(t.total_cache_read_tokens for t in traces)
    total_reasoning = sum(t.total_reasoning_tokens for t in traces)
    total_tokens = sum(t.total_tokens for t in traces)
    total_llm_calls = sum(t.total_llm_calls for t in traces)

    cache_ratio = round(total_cache / total_input, 4) if total_input > 0 else 0.0

    # Percentiles across traces
    trace_token_values = [t.total_tokens for t in traces if t.total_tokens > 0]
    p50_tokens = p95_tokens = p99_tokens = 0.0
    if len(trace_token_values) >= 2:
        p50_tokens = median(trace_token_values)
        quants = quantiles(trace_token_values, n=100)
        p95_tokens = quants[94] if len(quants) > 94 else trace_token_values[-1]
        p99_tokens = quants[98] if len(quants) > 98 else trace_token_values[-1]
    elif len(trace_token_values) == 1:
        p50_tokens = p95_tokens = p99_tokens = trace_token_values[0]

    # Per-call input token percentiles
    all_input_per_call = [
        c.input_tokens for t in traces for c in t.llm_calls if c.input_tokens > 0
    ]
    p50_input = p95_input = 0.0
    if len(all_input_per_call) >= 2:
        p50_input = median(all_input_per_call)
        quants = quantiles(all_input_per_call, n=100)
        p95_input = quants[94] if len(quants) > 94 else all_input_per_call[-1]
    elif len(all_input_per_call) == 1:
        p50_input = p95_input = all_input_per_call[0]

    # Context message count percentiles
    all_msg_counts = [
        s.total_messages
        for t in traces
        for s in t.context_snapshots
        if s.total_messages > 0
    ]
    p50_msgs = p95_msgs = 0.0
    if len(all_msg_counts) >= 2:
        p50_msgs = median(all_msg_counts)
        quants = quantiles(all_msg_counts, n=100)
        p95_msgs = quants[94] if len(quants) > 94 else all_msg_counts[-1]
    elif len(all_msg_counts) == 1:
        p50_msgs = p95_msgs = all_msg_counts[0]

    # Context utilization percentiles
    all_ctx_utils = [
        c.context_utilization
        for t in traces
        for c in t.llm_calls
        if c.context_window_size > 0
    ]
    p50_ctx = p95_ctx = max_ctx = 0.0
    if len(all_ctx_utils) >= 2:
        sorted_ctx = sorted(all_ctx_utils)
        p50_ctx = median(sorted_ctx)
        quants = quantiles(sorted_ctx, n=100)
        p95_ctx = quants[94] if len(quants) > 94 else sorted_ctx[-1]
        max_ctx = sorted_ctx[-1]
    elif len(all_ctx_utils) == 1:
        p50_ctx = p95_ctx = max_ctx = all_ctx_utils[0]

    # Merge agent breakdowns
    merged_agents: dict[str, AgentTokenBreakdown] = {}
    for t in traces:
        for name, ab in t.agent_breakdown.items():
            if name not in merged_agents:
                merged_agents[name] = AgentTokenBreakdown(
                    agent_name=name,
                    service_name=ab.service_name,
                )
            m = merged_agents[name]
            m.llm_call_count += ab.llm_call_count
            m.input_tokens += ab.input_tokens
            m.output_tokens += ab.output_tokens
            m.cache_read_tokens += ab.cache_read_tokens
            m.reasoning_tokens += ab.reasoning_tokens
            m.total_tokens += ab.total_tokens
            m.estimated_cost_usd += ab.estimated_cost_usd
            for model, count in ab.models_used.items():
                m.models_used[model] = m.models_used.get(model, 0) + count
            # Context window utilization
            m.context_window_size = max(m.context_window_size, ab.context_window_size)
            m.max_context_utilization = max(
                m.max_context_utilization, ab.max_context_utilization
            )

    # Recompute averages for merged agents
    for m in merged_agents.values():
        if m.llm_call_count > 0:
            m.avg_input_per_call = round(m.input_tokens / m.llm_call_count, 1)
            m.avg_output_per_call = round(m.output_tokens / m.llm_call_count, 1)
        if m.input_tokens > 0:
            m.cache_hit_ratio = round(m.cache_read_tokens / m.input_tokens, 4)
        m.estimated_cost_usd = round(m.estimated_cost_usd, 6)

    # Collect all accumulation alerts
    all_alerts = [
        alert for t in traces for alert in t.accumulation_alerts
    ]

    # Merge cost estimates
    merged_cost = CostEstimate()
    for t in traces:
        merged_cost.total_cost_usd += t.cost.total_cost_usd
        merged_cost.input_cost_usd += t.cost.input_cost_usd
        merged_cost.output_cost_usd += t.cost.output_cost_usd
        merged_cost.cached_input_cost_usd += t.cost.cached_input_cost_usd
        merged_cost.reasoning_cost_usd += t.cost.reasoning_cost_usd
        merged_cost.calls_without_pricing += t.cost.calls_without_pricing
        for model, cost_val in t.cost.per_model.items():
            merged_cost.per_model[model] = (
                merged_cost.per_model.get(model, 0.0) + cost_val
            )
        for agent, cost_val in t.cost.per_agent.items():
            merged_cost.per_agent[agent] = (
                merged_cost.per_agent.get(agent, 0.0) + cost_val
            )

    # Round merged cost
    merged_cost.total_cost_usd = round(merged_cost.total_cost_usd, 6)
    merged_cost.input_cost_usd = round(merged_cost.input_cost_usd, 6)
    merged_cost.output_cost_usd = round(merged_cost.output_cost_usd, 6)
    merged_cost.cached_input_cost_usd = round(
        merged_cost.cached_input_cost_usd, 6
    )
    merged_cost.reasoning_cost_usd = round(merged_cost.reasoning_cost_usd, 6)
    merged_cost.per_model = {
        k: round(v, 6) for k, v in merged_cost.per_model.items()
    }
    merged_cost.per_agent = {
        k: round(v, 6) for k, v in merged_cost.per_agent.items()
    }

    return TokenWindowAnalysis(
        start_time=from_nano_fn(start_ns),
        end_time=from_nano_fn(end_ns),
        trace_count=len(traces),
        total_llm_calls=total_llm_calls,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        total_cache_read_tokens=total_cache,
        total_reasoning_tokens=total_reasoning,
        total_tokens=total_tokens,
        overall_cache_hit_ratio=cache_ratio,
        p50_tokens_per_trace=round(p50_tokens, 1),
        p95_tokens_per_trace=round(p95_tokens, 1),
        p99_tokens_per_trace=round(p99_tokens, 1),
        p50_input_per_call=round(p50_input, 1),
        p95_input_per_call=round(p95_input, 1),
        p50_context_messages=round(p50_msgs, 1),
        p95_context_messages=round(p95_msgs, 1),
        p50_context_utilization=round(p50_ctx, 4),
        p95_context_utilization=round(p95_ctx, 4),
        max_context_utilization=round(max_ctx, 4),
        traces=traces,
        agent_breakdown=merged_agents,
        accumulation_alerts=all_alerts,
        cost=merged_cost,
    )
