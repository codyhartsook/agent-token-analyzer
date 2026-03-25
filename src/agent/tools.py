"""FunctionTools wrapping the token_analysis library API.

Each tool is an ``async def`` that Google ADK wraps via ``FunctionTool``.
They import the analysis functions directly (no subprocess) and return
``.model_dump()`` dicts so the LLM always gets structured JSON.

Session state keys used for multi-turn:
    ``last_analysis``      – the most recent result dict
    ``last_analysis_type`` – one of "discovery", "trace", "window"
"""

from __future__ import annotations

from typing import Any, Optional

from google.adk.tools.tool_context import ToolContext

from token_analysis import (
    AgentDiscovery,
    TokenWindowAnalysis,
    TraceTokenAnalysis,
    analyze_trace_tokens,
    analyze_window,
    discover_agents,
    format_report,
    get_client,
    relative_to_nano,
    to_nano,
)


# ---------------------------------------------------------------------------
# Tool 1: Discover agents
# ---------------------------------------------------------------------------

async def discover_agents_in_window(
    time_spec: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Discover all agents that have telemetry data in a time window.

    Args:
        time_spec: Relative time like '24h', '7d', '30m', or 'all' for everything.
        tool_context: ADK tool context (auto-injected, hidden from LLM).

    Returns:
        Structured JSON with discovered agents, their token usage, and models used.
    """
    try:
        client = get_client()
        if time_spec.lower() == "all":
            start_ns, end_ns = relative_to_nano("365d")
        else:
            start_ns, end_ns = relative_to_nano(time_spec)

        result: AgentDiscovery = discover_agents(client, start_ns, end_ns)
        result_dict = result.model_dump()

        # Cache in session state for format_analysis_report
        tool_context.state["last_analysis"] = result_dict
        tool_context.state["last_analysis_type"] = "discovery"

        return result_dict
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Tool 2: Analyse a single trace
# ---------------------------------------------------------------------------

async def analyze_trace(
    trace_id: str,
    include_cost: bool = False,
    accumulation_threshold: float = 2.0,
    tool_context: ToolContext = None,
) -> dict[str, Any]:
    """Deep-dive analysis of a single trace by its trace ID.

    Args:
        trace_id: The trace ID to analyse.
        include_cost: Whether to include cost estimation.
        accumulation_threshold: Context growth factor to trigger alerts (default 2.0).
        tool_context: ADK tool context (auto-injected, hidden from LLM).

    Returns:
        Structured JSON with per-call token breakdown, context windows, growth
        alerts, agent breakdown, and optional cost estimate.
    """
    try:
        client = get_client()
        result: TraceTokenAnalysis = analyze_trace_tokens(
            client,
            trace_id,
            include_cost=include_cost,
            growth_factor_warn=accumulation_threshold,
        )
        result_dict = result.model_dump()

        if tool_context is not None:
            tool_context.state["last_analysis"] = result_dict
            tool_context.state["last_analysis_type"] = "trace"

        return result_dict
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Tool 3: Analyse a time window
# ---------------------------------------------------------------------------

async def analyze_time_window(
    time_spec: str,
    agent_name: Optional[str] = None,
    include_cost: bool = False,
    limit: int = 100,
    accumulation_threshold: float = 2.0,
    tool_context: ToolContext = None,
) -> dict[str, Any]:
    """Aggregate token and context analysis across a time window.

    Args:
        time_spec: Relative time like '1h', '24h', '7d'.
        agent_name: Optional agent name to filter by.
        include_cost: Whether to include cost estimation.
        limit: Maximum number of traces to analyse (default 100).
        accumulation_threshold: Context growth factor to trigger alerts (default 2.0).
        tool_context: ADK tool context (auto-injected, hidden from LLM).

    Returns:
        Structured JSON with totals, percentiles, per-agent breakdown,
        accumulation alerts, and optional cost estimate.
    """
    try:
        client = get_client()
        start_ns, end_ns = relative_to_nano(time_spec)
        result: TokenWindowAnalysis = analyze_window(
            client,
            start_ns,
            end_ns,
            agent_filter=agent_name,
            include_cost=include_cost,
            limit=limit,
            growth_factor_warn=accumulation_threshold,
        )
        result_dict = result.model_dump()

        if tool_context is not None:
            tool_context.state["last_analysis"] = result_dict
            tool_context.state["last_analysis_type"] = "window"

        return result_dict
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Tool 4: Re-format the last analysis result
# ---------------------------------------------------------------------------

_MODEL_MAP = {
    "discovery": AgentDiscovery,
    "trace": TraceTokenAnalysis,
    "window": TokenWindowAnalysis,
}


async def format_analysis_report(
    format: str = "json",
    verbose: bool = False,
    tool_context: ToolContext = None,
) -> dict[str, Any]:
    """Re-format the most recent analysis result.

    Args:
        format: Output format — one of 'json', 'terminal', 'csv'.
        verbose: Whether to include per-call details.
        tool_context: ADK tool context (auto-injected, hidden from LLM).

    Returns:
        The formatted report as a string, or an error dict.
    """
    if tool_context is None:
        return {"error": "No tool context available."}

    last = tool_context.state.get("last_analysis")
    last_type = tool_context.state.get("last_analysis_type")

    if not last or not last_type:
        return {"error": "No previous analysis to format. Run an analysis tool first."}

    model_cls = _MODEL_MAP.get(last_type)
    if model_cls is None:
        return {"error": f"Unknown analysis type: {last_type}"}

    try:
        model_instance = model_cls.model_validate(last)
        report = format_report(model_instance, format, verbose=verbose)
        return {"format": format, "report": report}
    except Exception as e:
        return {"error": f"Formatting failed: {e}"}
