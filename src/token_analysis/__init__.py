"""Token & Context Analysis for agentic traces.

Public API for both CLI and programmatic (ADK agent) usage.

Example::

    from token_analysis import get_client, analyze_trace_tokens, analyze_window

    client = get_client()

    # Single trace analysis
    result = analyze_trace_tokens(client, "deca41fd...")
    print(result.model_dump_json(indent=2))

    # Time window analysis
    from token_analysis import relative_to_nano
    start, end = relative_to_nano("1h")
    window = analyze_window(client, start, end)
"""

from .analyzer import analyze_trace_tokens, analyze_window, discover_agents
from .client import from_nano, get_client, relative_to_nano, to_nano
from .context import reconstruct_context
from .context_window import (
    DEFAULT_CONTEXT_WINDOWS,
    ModelContextWindow,
    get_context_utilization,
    resolve_context_window,
)
from .models import (
    AgentDiscovery,
    AgentTokenBreakdown,
    ContextAccumulationAlert,
    ContextGrowthStep,
    ContextWindowSnapshot,
    CostEstimate,
    DiscoveredAgent,
    LLMCallTokens,
    ModelPricing,
    PromptMessage,
    TokenWindowAnalysis,
    ToolCallInfo,
    TraceTokenAnalysis,
)
from .report import format_report, write_reports

__all__ = [
    # Connection
    "get_client",
    "to_nano",
    "from_nano",
    "relative_to_nano",
    # Core analysis
    "discover_agents",
    "analyze_trace_tokens",
    "analyze_window",
    # Context
    "reconstruct_context",
    # Context window
    "ModelContextWindow",
    "DEFAULT_CONTEXT_WINDOWS",
    "resolve_context_window",
    "get_context_utilization",
    # Formatting
    "format_report",
    "write_reports",
    # Models
    "AgentDiscovery",
    "DiscoveredAgent",
    "TraceTokenAnalysis",
    "TokenWindowAnalysis",
    "LLMCallTokens",
    "ContextWindowSnapshot",
    "ContextGrowthStep",
    "ContextAccumulationAlert",
    "AgentTokenBreakdown",
    "PromptMessage",
    "ToolCallInfo",
    "ModelPricing",
    "CostEstimate",
]
