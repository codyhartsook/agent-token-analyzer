"""ClickHouse SQL queries for token and context analysis.

All functions accept a ``clickhouse-connect`` client and optional time-window
boundaries as nanosecond integers.  Results are returned as plain Python
structures (lists of dicts / tuples) ready for the model layer.

Deduplication Strategy
----------------------
Token data is triple-duplicated across ``call_llm``, ``generate_content``,
and ``openai.chat`` spans for the same LLM call.  We use a LEFT ANTI JOIN
CTE (``LEAF_LLM_CTE``) to select only **leaf LLM spans** — spans with
``gen_ai.request.model`` that have no children also carrying that attribute.
This naturally selects ``openai.chat`` (which has no LLM children) and
excludes the wrapper spans.
"""

from __future__ import annotations

from typing import Any

import clickhouse_connect


# ── Deduplication CTEs ──────────────────────────────────────────────────────

_LEAF_LLM_CTE_WINDOW = """
    leaf_llm AS (
        SELECT s.*
        FROM otel_traces s
        LEFT ANTI JOIN otel_traces c
            ON s.SpanId = c.ParentSpanId
            AND s.TraceId = c.TraceId
            AND c.SpanAttributes['gen_ai.request.model'] != ''
        WHERE s.SpanAttributes['gen_ai.request.model'] != ''
          AND s.Timestamp >= fromUnixTimestamp64Nano({start:UInt64})
          AND s.Timestamp <= fromUnixTimestamp64Nano({end:UInt64})
    )
"""

_LEAF_LLM_CTE_TRACE = """
    leaf_llm AS (
        SELECT s.*
        FROM otel_traces s
        LEFT ANTI JOIN otel_traces c
            ON s.SpanId = c.ParentSpanId
            AND s.TraceId = c.TraceId
            AND c.SpanAttributes['gen_ai.request.model'] != ''
        WHERE s.SpanAttributes['gen_ai.request.model'] != ''
          AND s.TraceId = {tid:String}
    )
"""


# ── Trace enumeration ────────────────────────────────────────────────────────

def get_trace_ids(
    client: clickhouse_connect.driver.Client,
    start_ns: int,
    end_ns: int,
    *,
    limit: int = 100,
) -> list[str]:
    """Return distinct trace IDs within the time window, most recent first."""
    result = client.query(
        """
        SELECT TraceId, min(Timestamp) AS first_ts
        FROM otel_traces
        WHERE Timestamp >= fromUnixTimestamp64Nano({start:UInt64})
          AND Timestamp <= fromUnixTimestamp64Nano({end:UInt64})
        GROUP BY TraceId
        ORDER BY first_ts DESC
        LIMIT {lim:UInt32}
        """,
        parameters={"start": start_ns, "end": end_ns, "lim": limit},
    )
    return [row[0] for row in result.result_rows]


# ── Per-trace span retrieval ─────────────────────────────────────────────────

def get_all_trace_spans(
    client: clickhouse_connect.driver.Client,
    trace_id: str,
) -> list[dict[str, Any]]:
    """Return all spans for a single trace, ordered by timestamp."""
    result = client.query(
        """
        SELECT
            TraceId,
            SpanId,
            ParentSpanId,
            ServiceName,
            SpanName,
            SpanKind,
            Duration,
            Timestamp,
            SpanAttributes
        FROM otel_traces
        WHERE TraceId = {tid:String}
        ORDER BY Timestamp ASC
        """,
        parameters={"tid": trace_id},
    )
    columns = [
        "trace_id", "span_id", "parent_span_id", "service_name",
        "span_name", "span_kind", "duration", "timestamp",
        "span_attributes",
    ]
    return [dict(zip(columns, row)) for row in result.result_rows]


# ── Deduplicated LLM calls (per trace) ──────────────────────────────────────

def get_deduplicated_llm_calls(
    client: clickhouse_connect.driver.Client,
    trace_id: str,
) -> list[dict[str, Any]]:
    """Deduplicated LLM calls for a single trace — leaf spans only."""
    result = client.query(
        f"""
        WITH {_LEAF_LLM_CTE_TRACE}
        SELECT
            TraceId,
            SpanId,
            ParentSpanId,
            ServiceName,
            SpanName,
            Duration,
            Timestamp,
            SpanAttributes
        FROM leaf_llm
        ORDER BY Timestamp ASC
        """,
        parameters={"tid": trace_id},
    )
    columns = [
        "trace_id", "span_id", "parent_span_id", "service_name",
        "span_name", "duration", "timestamp", "span_attributes",
    ]
    return [dict(zip(columns, row)) for row in result.result_rows]


# ── Context window spans (per trace) ────────────────────────────────────────

def get_context_window_spans(
    client: clickhouse_connect.driver.Client,
    trace_id: str,
) -> list[dict[str, Any]]:
    """Get spans with prompt content for context reconstruction.

    These are always on the leaf LLM spans (``openai.chat``).
    Only returns spans that have at least ``gen_ai.prompt.0.role`` set.
    """
    result = client.query(
        """
        SELECT
            SpanId,
            ParentSpanId,
            ServiceName,
            SpanName,
            Timestamp,
            Duration,
            SpanAttributes
        FROM otel_traces
        WHERE TraceId = {tid:String}
          AND SpanAttributes['gen_ai.prompt.0.role'] != ''
        ORDER BY Timestamp ASC
        """,
        parameters={"tid": trace_id},
    )
    columns = [
        "span_id", "parent_span_id", "service_name", "span_name",
        "timestamp", "duration", "span_attributes",
    ]
    return [dict(zip(columns, row)) for row in result.result_rows]


# ── Aggregate token summary (deduplicated, window) ──────────────────────────

def get_token_summary_deduplicated(
    client: clickhouse_connect.driver.Client,
    start_ns: int,
    end_ns: int,
) -> dict[str, Any]:
    """Aggregate token stats across a time window — deduplicated."""
    result = client.query(
        f"""
        WITH {_LEAF_LLM_CTE_WINDOW}
        SELECT
            count()                                                            AS llm_calls,
            uniqExact(TraceId)                                                 AS trace_count,
            sum(toUInt64OrZero(SpanAttributes['gen_ai.usage.input_tokens']))
              + sum(toUInt64OrZero(SpanAttributes['gen_ai.usage.prompt_tokens']))
                                                                               AS input_tokens,
            sum(toUInt64OrZero(SpanAttributes['gen_ai.usage.output_tokens']))
              + sum(toUInt64OrZero(SpanAttributes['gen_ai.usage.completion_tokens']))
                                                                               AS output_tokens,
            sum(toUInt64OrZero(SpanAttributes['gen_ai.usage.cache_read_input_tokens']))
                                                                               AS cache_read,
            sum(toUInt64OrZero(SpanAttributes['gen_ai.usage.reasoning_tokens']))
                                                                               AS reasoning,
            sum(toUInt64OrZero(SpanAttributes['llm.usage.total_tokens']))      AS total_direct
        FROM leaf_llm
        """,
        parameters={"start": start_ns, "end": end_ns},
    )
    row = result.result_rows[0] if result.result_rows else (0,) * 7
    inp, out = int(row[2]), int(row[3])
    total_direct = int(row[6])
    return {
        "llm_calls": int(row[0]),
        "trace_count": int(row[1]),
        "input_tokens": inp,
        "output_tokens": out,
        "cache_read_tokens": int(row[4]),
        "reasoning_tokens": int(row[5]),
        "total_tokens": total_direct if total_direct > 0 else inp + out,
    }


# ── Per-agent token breakdown (deduplicated, window) ────────────────────────

def get_per_agent_tokens_deduplicated(
    client: clickhouse_connect.driver.Client,
    start_ns: int,
    end_ns: int,
) -> list[dict[str, Any]]:
    """Per-agent token breakdown using identity resolution + deduplication."""
    result = client.query(
        f"""
        WITH {_LEAF_LLM_CTE_WINDOW}
        SELECT
            coalesce(
                nullIf(SpanAttributes['gen_ai.agent.name'], ''),
                nullIf(SpanAttributes['ioa_observe.entity.name'], ''),
                ServiceName
            ) AS agent_name,
            any(ServiceName) AS service_name,
            count() AS llm_call_count,
            sum(toUInt64OrZero(SpanAttributes['gen_ai.usage.input_tokens'])
              + toUInt64OrZero(SpanAttributes['gen_ai.usage.prompt_tokens']))
                                                                    AS input_tokens,
            sum(toUInt64OrZero(SpanAttributes['gen_ai.usage.output_tokens'])
              + toUInt64OrZero(SpanAttributes['gen_ai.usage.completion_tokens']))
                                                                    AS output_tokens,
            sum(toUInt64OrZero(SpanAttributes['gen_ai.usage.cache_read_input_tokens']))
                                                                    AS cache_read,
            sum(toUInt64OrZero(SpanAttributes['gen_ai.usage.reasoning_tokens']))
                                                                    AS reasoning,
            sum(toUInt64OrZero(SpanAttributes['llm.usage.total_tokens']))
                                                                    AS total_direct
        FROM leaf_llm
        GROUP BY agent_name
        ORDER BY input_tokens + output_tokens DESC
        """,
        parameters={"start": start_ns, "end": end_ns},
    )
    return [
        {
            "agent_name": row[0],
            "service_name": row[1],
            "llm_call_count": int(row[2]),
            "input_tokens": int(row[3]),
            "output_tokens": int(row[4]),
            "cache_read_tokens": int(row[5]),
            "reasoning_tokens": int(row[6]),
            "total_tokens": int(row[7]) if int(row[7]) > 0 else int(row[3]) + int(row[4]),
        }
        for row in result.result_rows
    ]


# ── Token percentiles (deduplicated, window) ────────────────────────────────

def get_token_percentiles_deduplicated(
    client: clickhouse_connect.driver.Client,
    start_ns: int,
    end_ns: int,
) -> dict[str, float]:
    """p50/p95/p99 of per-trace total token usage (deduplicated)."""
    result = client.query(
        f"""
        WITH {_LEAF_LLM_CTE_WINDOW},
        per_trace AS (
            SELECT
                TraceId,
                sum(
                    toUInt64OrZero(SpanAttributes['gen_ai.usage.input_tokens'])
                    + toUInt64OrZero(SpanAttributes['gen_ai.usage.prompt_tokens'])
                    + toUInt64OrZero(SpanAttributes['gen_ai.usage.output_tokens'])
                    + toUInt64OrZero(SpanAttributes['gen_ai.usage.completion_tokens'])
                ) AS trace_tokens
            FROM leaf_llm
            GROUP BY TraceId
        )
        SELECT
            quantile(0.50)(trace_tokens) AS p50,
            quantile(0.95)(trace_tokens) AS p95,
            quantile(0.99)(trace_tokens) AS p99
        FROM per_trace
        """,
        parameters={"start": start_ns, "end": end_ns},
    )
    row = result.result_rows[0] if result.result_rows else (0, 0, 0)
    return {
        "p50": round(float(row[0]), 1),
        "p95": round(float(row[1]), 1),
        "p99": round(float(row[2]), 1),
    }


# ── Filter traces by agent ──────────────────────────────────────────────────


# ── Deduplicated LLM calls (window — raw rows for Python-side estimation) ────

def get_deduplicated_llm_calls_window(
    client: clickhouse_connect.driver.Client,
    start_ns: int,
    end_ns: int,
) -> list[dict[str, Any]]:
    """All deduplicated LLM call spans in a window — with full attributes.

    Used by ``discover_agents()`` so that Python-side token estimation
    (from prompt content) works for providers that don't report token counts.
    """
    result = client.query(
        f"""
        WITH {_LEAF_LLM_CTE_WINDOW}
        SELECT
            TraceId,
            SpanId,
            ParentSpanId,
            ServiceName,
            SpanName,
            Duration,
            Timestamp,
            SpanAttributes
        FROM leaf_llm
        ORDER BY Timestamp ASC
        """,
        parameters={"start": start_ns, "end": end_ns},
    )
    columns = [
        "trace_id", "span_id", "parent_span_id", "service_name",
        "span_name", "duration", "timestamp", "span_attributes",
    ]
    return [dict(zip(columns, row)) for row in result.result_rows]


# ── Agent discovery ──────────────────────────────────────────────────────────

def discover_agents(
    client: clickhouse_connect.driver.Client,
    start_ns: int,
    end_ns: int,
) -> dict[str, Any]:
    """Discover all agents, services, and models active in a time window.

    Uses multiple identity sources to find agents:
    - gen_ai.agent.name (explicit agent identity)
    - ioa_observe.entity.name (observe SDK entity name)
    - ioa_observe.span.kind = 'agent' (observe SDK agent spans)
    - gen_ai.operation.name = 'invoke_agent' (agent invocation spans)
    - ServiceName (always present, used as fallback)

    Returns a dict with agents, services, and trace count.
    """
    # 1. Find all named agents with their token usage (from LLM spans)
    agent_result = client.query(
        f"""
        WITH {_LEAF_LLM_CTE_WINDOW}
        SELECT
            coalesce(
                nullIf(SpanAttributes['gen_ai.agent.name'], ''),
                nullIf(SpanAttributes['ioa_observe.entity.name'], ''),
                ServiceName
            ) AS agent_name,
            any(ServiceName) AS service_name,
            uniqExact(TraceId) AS trace_count,
            count() AS llm_call_count,
            sum(toUInt64OrZero(SpanAttributes['gen_ai.usage.input_tokens'])
              + toUInt64OrZero(SpanAttributes['gen_ai.usage.prompt_tokens']))
                                                                    AS input_tokens,
            sum(toUInt64OrZero(SpanAttributes['gen_ai.usage.output_tokens'])
              + toUInt64OrZero(SpanAttributes['gen_ai.usage.completion_tokens']))
                                                                    AS output_tokens,
            groupUniqArray(SpanAttributes['gen_ai.request.model']) AS models,
            min(Timestamp) AS first_seen,
            max(Timestamp) AS last_seen
        FROM leaf_llm
        GROUP BY agent_name
        ORDER BY input_tokens + output_tokens DESC
        """,
        parameters={"start": start_ns, "end": end_ns},
    )

    llm_agents: dict[str, dict[str, Any]] = {}
    for row in agent_result.result_rows:
        models = [m for m in row[6] if m]  # filter empty strings
        inp, out = int(row[4]), int(row[5])
        llm_agents[row[0]] = {
            "agent_name": row[0],
            "service_name": row[1],
            "trace_count": int(row[2]),
            "llm_call_count": int(row[3]),
            "total_input_tokens": inp,
            "total_output_tokens": out,
            "total_tokens": inp + out,
            "models_used": models,
            "first_seen": str(row[7]),
            "last_seen": str(row[8]),
        }

    # 2. Find agents from non-LLM spans (invoke_agent, ioa_observe agent spans)
    #    These catch agents that delegate to LLM but don't have LLM spans themselves
    non_llm_result = client.query(
        """
        SELECT
            coalesce(
                nullIf(SpanAttributes['gen_ai.agent.name'], ''),
                nullIf(SpanAttributes['ioa_observe.entity.name'], ''),
                ServiceName
            ) AS agent_name,
            any(ServiceName) AS service_name,
            uniqExact(TraceId) AS trace_count,
            groupUniqArray(SpanAttributes['ioa_observe.span.kind']) AS span_kinds,
            min(Timestamp) AS first_seen,
            max(Timestamp) AS last_seen
        FROM otel_traces
        WHERE Timestamp >= fromUnixTimestamp64Nano({start:UInt64})
          AND Timestamp <= fromUnixTimestamp64Nano({end:UInt64})
          AND (
              SpanAttributes['gen_ai.agent.name'] != ''
              OR SpanAttributes['ioa_observe.span.kind'] = 'agent'
              OR SpanAttributes['gen_ai.operation.name'] = 'invoke_agent'
          )
        GROUP BY agent_name
        ORDER BY trace_count DESC
        """,
        parameters={"start": start_ns, "end": end_ns},
    )

    # Merge non-LLM agent info into the result
    all_agents: dict[str, dict[str, Any]] = dict(llm_agents)
    for row in non_llm_result.result_rows:
        name = row[0]
        span_kinds = [k for k in row[3] if k]
        if name in all_agents:
            all_agents[name]["span_kinds"] = span_kinds
            # Take the wider time range
            if str(row[4]) < all_agents[name].get("first_seen", "9999"):
                all_agents[name]["first_seen"] = str(row[4])
            if str(row[5]) > all_agents[name].get("last_seen", ""):
                all_agents[name]["last_seen"] = str(row[5])
            # Use the larger trace count
            all_agents[name]["trace_count"] = max(
                all_agents[name]["trace_count"], int(row[2])
            )
        else:
            all_agents[name] = {
                "agent_name": name,
                "service_name": row[1],
                "trace_count": int(row[2]),
                "llm_call_count": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_tokens": 0,
                "models_used": [],
                "first_seen": str(row[4]),
                "last_seen": str(row[5]),
                "span_kinds": span_kinds,
            }

    # 3. Get all distinct services
    svc_result = client.query(
        """
        SELECT DISTINCT ServiceName
        FROM otel_traces
        WHERE Timestamp >= fromUnixTimestamp64Nano({start:UInt64})
          AND Timestamp <= fromUnixTimestamp64Nano({end:UInt64})
        ORDER BY ServiceName
        """,
        parameters={"start": start_ns, "end": end_ns},
    )
    services = [row[0] for row in svc_result.result_rows]

    # 4. Total trace count
    trace_result = client.query(
        """
        SELECT uniqExact(TraceId)
        FROM otel_traces
        WHERE Timestamp >= fromUnixTimestamp64Nano({start:UInt64})
          AND Timestamp <= fromUnixTimestamp64Nano({end:UInt64})
        """,
        parameters={"start": start_ns, "end": end_ns},
    )
    total_traces = int(trace_result.result_rows[0][0]) if trace_result.result_rows else 0

    return {
        "agents": list(all_agents.values()),
        "services": services,
        "total_traces": total_traces,
    }


def get_traces_by_agent(
    client: clickhouse_connect.driver.Client,
    agent_name: str,
    start_ns: int,
    end_ns: int,
    *,
    limit: int = 100,
) -> list[str]:
    """Get trace IDs that contain spans from a specific agent."""
    result = client.query(
        """
        SELECT TraceId, min(Timestamp) AS first_ts
        FROM otel_traces
        WHERE Timestamp >= fromUnixTimestamp64Nano({start:UInt64})
          AND Timestamp <= fromUnixTimestamp64Nano({end:UInt64})
          AND (
              SpanAttributes['gen_ai.agent.name'] = {agent:String}
              OR SpanAttributes['ioa_observe.entity.name'] = {agent:String}
              OR ServiceName = {agent:String}
          )
        GROUP BY TraceId
        ORDER BY first_ts DESC
        LIMIT {lim:UInt32}
        """,
        parameters={
            "start": start_ns,
            "end": end_ns,
            "agent": agent_name,
            "lim": limit,
        },
    )
    return [row[0] for row in result.result_rows]
