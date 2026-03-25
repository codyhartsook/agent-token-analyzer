"""Pydantic v2 models for token and context window analysis.

All models are designed to serialize cleanly to JSON via ``.model_dump()`` /
``.model_dump_json()`` for use as ADK agent tool responses.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, computed_field


# ── Prompt Message (reconstructed from gen_ai.prompt.{N} attributes) ────────


class ToolCallInfo(BaseModel):
    """A tool call within a prompt message."""

    index: int = 0
    name: str = ""
    arguments: str = ""


class PromptMessage(BaseModel):
    """A single message in the reconstructed conversation context."""

    index: int  # Position in the prompt (0, 1, 2, ...)
    role: str  # system, user, assistant, tool
    content: str = ""
    content_chars: int = 0
    tool_calls: list[ToolCallInfo] = Field(default_factory=list)
    tool_call_id: str = ""  # For tool-response messages


# ── Per-LLM-Call Token Breakdown ────────────────────────────────────────────


class LLMCallTokens(BaseModel):
    """Token breakdown for a single LLM API call (deduplicated)."""

    span_id: str
    trace_id: str
    span_name: str = ""
    service_name: str = ""
    agent_name: str = ""  # Resolved agent identity
    model: str = ""
    response_model: str = ""
    timestamp_ns: int = 0
    duration_ms: float = 0.0

    # Token counts
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0

    # Whether token counts were estimated from content (vs reported by the LLM API)
    tokens_estimated: bool = False

    # Context window utilization
    context_window_size: int = 0  # max_input_tokens for this model (0 = unknown)
    context_utilization: float = 0.0  # input_tokens / context_window_size

    # Finish reason
    finish_reason: str = ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cache_hit_ratio(self) -> float:
        """Fraction of input tokens served from cache."""
        if self.input_tokens == 0:
            return 0.0
        return round(self.cache_read_input_tokens / self.input_tokens, 4)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def billable_input_tokens(self) -> int:
        """Input tokens minus cached tokens (for cost estimation)."""
        return max(0, self.input_tokens - self.cache_read_input_tokens)


# ── Context Window Snapshot ─────────────────────────────────────────────────


class ContextWindowSnapshot(BaseModel):
    """Reconstructed context window for a single LLM call."""

    span_id: str
    call_index: int  # Sequential index within the trace (0-based)
    agent_name: str = ""
    model: str = ""
    timestamp_ns: int = 0

    # Reconstructed conversation
    messages: list[PromptMessage] = Field(default_factory=list)
    completion_content: str = ""
    completion_finish_reason: str = ""

    # Derived metrics
    total_messages: int = 0
    messages_by_role: dict[str, int] = Field(default_factory=dict)
    total_content_chars: int = 0
    total_tool_calls: int = 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def has_tool_results(self) -> bool:
        """Whether this context includes tool-result messages."""
        return self.messages_by_role.get("tool", 0) > 0


# ── Context Growth ──────────────────────────────────────────────────────────


class ContextGrowthStep(BaseModel):
    """Delta between two sequential LLM calls in the same trace."""

    from_call_index: int
    to_call_index: int
    from_span_id: str
    to_span_id: str
    message_count_delta: int = 0
    content_chars_delta: int = 0
    input_tokens_delta: int = 0
    new_roles: dict[str, int] = Field(default_factory=dict)
    growth_pct: float = 0.0  # Percentage growth in content chars


class ContextAccumulationAlert(BaseModel):
    """Alert: context is growing without bound between LLM calls."""

    trace_id: str
    agent_name: str = ""
    call_count: int = 0
    initial_input_tokens: int = 0
    final_input_tokens: int = 0
    growth_factor: float = 0.0
    growth_steps: list[ContextGrowthStep] = Field(default_factory=list)
    severity: str = "info"  # "info", "warning", "critical"


# ── Per-Agent Token Breakdown ───────────────────────────────────────────────


class AgentTokenBreakdown(BaseModel):
    """Token usage attributed to a single agent."""

    agent_name: str
    service_name: str = ""
    llm_call_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    avg_input_per_call: float = 0.0
    avg_output_per_call: float = 0.0
    cache_hit_ratio: float = 0.0
    models_used: dict[str, int] = Field(default_factory=dict)
    estimated_cost_usd: float = 0.0

    # Context window utilization
    context_window_size: int = 0  # context window of the primary model
    max_context_utilization: float = 0.0  # highest utilization across all calls
    avg_context_utilization: float = 0.0  # average utilization across calls


# ── Cost Estimation ─────────────────────────────────────────────────────────


class ModelPricing(BaseModel):
    """Pricing for a single model (per 1M tokens)."""

    model_pattern: str  # glob pattern, e.g. "gpt-4o*"
    input_per_1m: float  # USD per 1M input tokens
    output_per_1m: float  # USD per 1M output tokens
    cached_input_per_1m: float = 0.0
    reasoning_per_1m: float = 0.0


class CostEstimate(BaseModel):
    """Cost estimate for a set of LLM calls."""

    total_cost_usd: float = 0.0
    input_cost_usd: float = 0.0
    output_cost_usd: float = 0.0
    cached_input_cost_usd: float = 0.0
    reasoning_cost_usd: float = 0.0
    per_model: dict[str, float] = Field(default_factory=dict)
    per_agent: dict[str, float] = Field(default_factory=dict)
    calls_without_pricing: int = 0


# ── Agent Discovery ─────────────────────────────────────────────────────────


class DiscoveredAgent(BaseModel):
    """A single agent found during discovery."""

    agent_name: str
    service_name: str = ""
    trace_count: int = 0
    llm_call_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    tokens_estimated: bool = False  # True if any token counts were estimated
    models_used: list[str] = Field(default_factory=list)
    first_seen: str = ""  # ISO timestamp
    last_seen: str = ""   # ISO timestamp
    span_kinds: list[str] = Field(default_factory=list)  # agent, tool, workflow, etc.
    sub_agents: list[str] = Field(default_factory=list)  # child agent/tool names within this service

    # Context window utilization
    context_window_size: int = 0  # context window of the primary model
    max_context_utilization: float = 0.0  # peak utilization seen across all traces


class AgentDiscovery(BaseModel):
    """Result of agent discovery across a time window."""

    start_time: str | None = None
    end_time: str | None = None
    total_agents: int = 0
    total_services: int = 0
    total_traces: int = 0
    agents: list[DiscoveredAgent] = Field(default_factory=list)
    services: list[str] = Field(default_factory=list)


# ── Trace-Level Token Analysis ──────────────────────────────────────────────


class TraceTokenAnalysis(BaseModel):
    """Complete token and context analysis for a single trace."""

    trace_id: str
    total_spans: int = 0
    total_llm_calls: int = 0
    total_duration_ms: float = 0.0
    service_chain: list[str] = Field(default_factory=list)

    # Token aggregates (deduplicated)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_reasoning_tokens: int = 0
    total_tokens: int = 0
    overall_cache_hit_ratio: float = 0.0

    # Context window utilization
    max_context_utilization: float = 0.0  # highest utilization across all calls

    # Per-call breakdown
    llm_calls: list[LLMCallTokens] = Field(default_factory=list)

    # Context window analysis
    context_snapshots: list[ContextWindowSnapshot] = Field(default_factory=list)
    context_growth: list[ContextGrowthStep] = Field(default_factory=list)
    accumulation_alerts: list[ContextAccumulationAlert] = Field(default_factory=list)

    # Per-agent breakdown
    agent_breakdown: dict[str, AgentTokenBreakdown] = Field(default_factory=dict)

    # Cost
    cost: CostEstimate = Field(default_factory=CostEstimate)


# ── Time-Window Aggregate Analysis ──────────────────────────────────────────


class TokenWindowAnalysis(BaseModel):
    """Aggregate token and context analysis across a time window."""

    start_time: str | None = None
    end_time: str | None = None
    trace_count: int = 0
    total_llm_calls: int = 0

    # Token aggregates
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_reasoning_tokens: int = 0
    total_tokens: int = 0
    overall_cache_hit_ratio: float = 0.0

    # Percentiles (across traces)
    p50_tokens_per_trace: float = 0.0
    p95_tokens_per_trace: float = 0.0
    p99_tokens_per_trace: float = 0.0
    p50_input_per_call: float = 0.0
    p95_input_per_call: float = 0.0
    p50_context_messages: float = 0.0
    p95_context_messages: float = 0.0

    # Context window utilization percentiles
    p50_context_utilization: float = 0.0
    p95_context_utilization: float = 0.0
    max_context_utilization: float = 0.0

    # Per-trace details
    traces: list[TraceTokenAnalysis] = Field(default_factory=list)

    # Aggregate agent breakdown
    agent_breakdown: dict[str, AgentTokenBreakdown] = Field(default_factory=dict)

    # Accumulation alerts across all traces
    accumulation_alerts: list[ContextAccumulationAlert] = Field(default_factory=list)

    # Aggregate cost
    cost: CostEstimate = Field(default_factory=CostEstimate)
