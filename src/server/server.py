"""A2A server entry point for the Token Analysis agent.

Run::

    uv run python -m server.server
"""

from __future__ import annotations

import asyncio
import os

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from dotenv import load_dotenv
from loguru import logger
from uvicorn import Config, Server

from server.agent_executor import TokenAnalysisAgentExecutor
from server.card import AGENT_CARD

load_dotenv()

_log = logger.bind(name="server")

PORT = int(os.getenv("TOKEN_ANALYSIS_PORT", "8883"))


async def main() -> None:
    """Start the A2A HTTP server."""
    request_handler = DefaultRequestHandler(
        agent_executor=TokenAnalysisAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=AGENT_CARD, http_handler=request_handler
    )

    config = Config(app=app.build(), host="0.0.0.0", port=PORT, loop="asyncio")
    server = Server(config)

    _log.info(f"Starting TokenAnalysisAgent A2A server on port {PORT}")
    await server.serve()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        _log.info("Shutting down on keyboard interrupt.")
    except Exception as e:
        _log.error(f"Server error: {e}")
