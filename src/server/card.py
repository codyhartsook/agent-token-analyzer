"""A2A AgentCard for the Token Analysis agent."""

from a2a.types import AgentCard

AGENT_CARD = AgentCard(
    name="TokenAnalysisAgent",
    url="http://localhost:8883",
    description=(
        "Analyzes agent token usage, context windows, and costs from "
        "OpenTelemetry traces. Accepts natural language queries and returns "
        "structured JSON."
    ),
    version="1.0.0",
    capabilities={"streaming": True},
    skills=[],
    default_input_modes=["text/plain"],
    default_output_modes=["text/plain", "application/json"],
    supports_authenticated_extended_card=False,
)
