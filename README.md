# Token & Context Analysis Agent

A natural-language ADK agent that analyzes how AI agents consume tokens, manage context windows, and incur API costs — powered by OpenTelemetry traces stored in ClickHouse.

Ask questions in plain English and get back structured JSON:

```
"Discover all agents active in the last 24 hours"
"Analyze trace abc123 with cost estimation"
"Show me token usage for the recruiter agent this week"
"Format that as CSV"
```

## Architecture

```
agent-token-analyzer/
├── src/
│   ├── token_analysis/      # Core library — ClickHouse queries, models, reports
│   ├── agent/               # ADK agent with FunctionTools
│   │   ├── agent.py         # Agent factory, session management, runner
│   │   └── tools.py         # 4 tools wrapping the analysis library
│   └── server/              # A2A server (HTTP, port 8883)
│       ├── card.py           # AgentCard definition
│       ├── agent_executor.py # A2A ↔ ADK bridge
│       ├── event_converter.py# ADK event → A2A event streaming
│       └── server.py         # uvicorn entry point
```

The agent imports the `token_analysis` library functions directly — no subprocess shelling. Every tool returns Pydantic `.model_dump()` dicts so responses are always structured JSON.

## Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- ClickHouse instance with OpenTelemetry trace data
- LiteLLM-compatible LLM provider (OpenAI, Azure, Anthropic, etc.)

### Installation

```bash
cd agent-token-analyzer

# Install dependencies
uv sync

# Copy and configure environment
cp .env.example .env
# Edit .env with your ClickHouse and LLM credentials
```

### Configuration

```env
# ClickHouse connection
DB_HOST=localhost
DB_PORT=8123
DB_USERNAME=admin
DB_PASSWORD=admin
DB_DATABASE=default

# LLM configuration
LLM_MODEL=openai/gpt-4o
# LITELLM_PROXY_BASE_URL=
# LITELLM_PROXY_API_KEY=
```

## Usage

### A2A Server

Run the agent as an A2A protocol server:

```bash
uv run python -m server.server
# Starts on http://localhost:8883
```

Verify the agent card:

```bash
curl http://localhost:8883/.well-known/agent.json
```

### CLI (still available)

The original CLI continues to work for direct analysis:

```bash
# Discover agents
uv run python -m token_analysis --discover 24h

# Analyze a trace
uv run python -m token_analysis --trace-id <trace-id> --cost

# Time window analysis
uv run python -m token_analysis --last 1h --agent RecruiterAgent --format json
```

## Agent Tools

The ADK agent exposes four tools that the LLM calls based on natural-language intent:

| Tool | What it does | Example prompt |
|------|-------------|----------------|
| `discover_agents_in_window` | Find all agents with telemetry in a time window | *"What agents have been active in the last 7 days?"* |
| `analyze_trace` | Deep-dive a single trace — per-call tokens, context snapshots, growth alerts, costs | *"Analyze trace abc123 with cost estimation"* |
| `analyze_time_window` | Aggregate analysis across a time window, optionally filtered by agent | *"Show token usage for the coder agent in the last 2 hours"* |
| `format_analysis_report` | Re-render the last result as JSON, CSV, or terminal text | *"Show that as CSV"* |

### Time spec format

Tools accept relative time specifications: a number followed by `m` (minutes), `h` (hours), or `d` (days).

| Spec | Meaning |
|------|---------|
| `30m` | Last 30 minutes |
| `2h` | Last 2 hours |
| `24h` | Last 24 hours |
| `7d` | Last 7 days |
| `all` | Everything (discover only) |

## Structured JSON Responses

All tools return Pydantic models serialized as JSON. Key response types:

**AgentDiscovery** — from `discover_agents_in_window`:
```json
{
  "total_agents": 3,
  "total_traces": 42,
  "agents": [
    {
      "agent_name": "RecruiterAgent",
      "llm_call_count": 156,
      "total_tokens": 892400,
      "models_used": ["gpt-4o"],
      "first_seen": "2026-03-24T10:00:00Z",
      "last_seen": "2026-03-25T14:30:00Z"
    }
  ]
}
```

**TraceTokenAnalysis** — from `analyze_trace`:
```json
{
  "trace_id": "abc123",
  "total_llm_calls": 8,
  "total_input_tokens": 45200,
  "total_output_tokens": 3100,
  "overall_cache_hit_ratio": 0.62,
  "agent_breakdown": { ... },
  "accumulation_alerts": [ ... ],
  "cost": { "total_cost_usd": 0.048 }
}
```

**TokenWindowAnalysis** — from `analyze_time_window`:
```json
{
  "trace_count": 15,
  "total_tokens": 1240000,
  "p50_tokens_per_trace": 82000,
  "p95_tokens_per_trace": 156000,
  "agent_breakdown": { ... },
  "cost": { "total_cost_usd": 1.24 }
}
```

## A2A Protocol

The agent is exposed via the [A2A protocol](https://github.com/a2aproject/a2a-spec) with streaming support. The final response includes:

- **TextPart** — natural-language summary of key findings
- **DataPart** — full structured JSON analysis (metadata `type: "token_analysis_discovery"`, `"token_analysis_trace"`, or `"token_analysis_window"`)

This allows other agents to consume the structured data programmatically while humans get a readable summary.

## Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov

# End-to-end tests (requires live ClickHouse)
uv run pytest -m e2e
```

## License

Apache-2.0
