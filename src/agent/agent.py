"""Single ADK agent for token & context analysis.

Creates a ``google.adk.agents.Agent`` with four ``FunctionTool`` wrappers
around the ``token_analysis`` library.  Exposes a module-level ``runner``
and ``session_service`` for the A2A server to use.
"""

from __future__ import annotations

import os
from typing import Optional

from google.adk.agents import Agent
from google.adk.apps.app import App
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.function_tool import FunctionTool

from .tools import (
    analyze_time_window,
    analyze_trace,
    discover_agents_in_window,
    format_analysis_report,
)
from .llm_config import configure_llm

configure_llm()

LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o")

# ---------------------------------------------------------------------------
# Agent instruction
# ---------------------------------------------------------------------------

AGENT_INSTRUCTION = """\
You are a Token & Context Analysis agent. You help users understand how their
AI agents consume tokens, manage context windows, and incur API costs by
querying OpenTelemetry data stored in ClickHouse.

Available tools
───────────────
• discover_agents_in_window – **Start here.** This is the primary tool for
  getting an overview of agent activity and token usage in a time window.
  It returns per-agent totals: token counts, LLM call counts, models used,
  context window utilization, and activity timestamps.
  USE THIS for: "show token usage", "what agents are active", "how many
  tokens were used", "discover agents", or any general query about a time period.
  time_spec examples: "24h", "7d", "30m", "1h", "all"

• analyze_trace – Deep-dive analysis of a single trace by its trace ID.
  Returns per-call token breakdown, context window snapshots, growth alerts,
  and cost estimates.
  USE THIS only when the user provides a specific trace ID.

• analyze_time_window – Aggregate percentile analysis across traces in a time
  window, optionally filtered to one agent. Returns p50/p95/p99 token
  distributions, accumulation alerts, and per-trace breakdowns.
  USE THIS for: statistical analysis, percentile breakdowns, or when the
  user explicitly asks for aggregate/percentile data.
  time_spec examples: "30m", "2h", "24h", "7d"
  Optionally pass agent_name to filter.

• format_analysis_report – Re-format the last analysis result as JSON, CSV,
  or terminal text. Use when the user asks to change the output format.

Tool selection guide
────────────────────
  "show me token usage for the last 24h"  → discover_agents_in_window("24h")
  "what agents are active?"               → discover_agents_in_window("24h")
  "analyze trace abc123"                  → analyze_trace("abc123")
  "show me percentile breakdown for 1h"   → analyze_time_window("1h")
  "show that as CSV"                      → format_analysis_report("csv")

When in doubt, prefer discover_agents_in_window — it gives the best overview.

Time spec format
────────────────
A number followed by m (minutes), h (hours), or d (days).
Examples: "30m", "2h", "24h", "7d".
Use "all" with discover_agents_in_window to search all available data.

Response guidelines
───────────────────
• Always return the structured JSON data from the tools.
• Include a brief natural-language summary highlighting key findings
  (top token consumers, anomalies, cost drivers) but always include the
  full structured data.
• When the user asks for a specific format (JSON, CSV, terminal), call
  format_analysis_report and return its output.
"""


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

session_service = InMemorySessionService()


async def get_or_create_session(
    app_name: str,
    user_id: str,
    session_id: str,
    state_overrides: Optional[dict] = None,
):
    """Retrieve an existing session or create a new one.

    Prevents ``AlreadyExistsError`` on repeated invocations and optionally
    seeds / updates session state.
    """
    session = await session_service.get_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )
    if session is not None:
        if state_overrides:
            for key, value in state_overrides.items():
                session.state[key] = value
        return session

    initial_state: dict = {
        "last_analysis": None,
        "last_analysis_type": None,
    }
    if state_overrides:
        initial_state.update(state_overrides)

    return await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        state=initial_state,
    )


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def create_agent() -> Agent:
    """Create the token analysis ADK agent."""
    return Agent(
        name="token_analysis_agent",
        model=LiteLlm(model=LLM_MODEL),
        instruction=AGENT_INSTRUCTION,
        description=(
            "Analyzes agent token usage, context windows, and costs "
            "from OpenTelemetry traces stored in ClickHouse"
        ),
        tools=[
            FunctionTool(func=discover_agents_in_window),
            FunctionTool(func=analyze_trace),
            FunctionTool(func=analyze_time_window),
            FunctionTool(func=format_analysis_report),
        ],
    )


# ---------------------------------------------------------------------------
# Module-level runner (created once, shared by server)
# ---------------------------------------------------------------------------

APP_NAME = "token_analysis"

_agent = create_agent()
_app = App(name=APP_NAME, root_agent=_agent)
runner = Runner(app=_app, session_service=session_service)
