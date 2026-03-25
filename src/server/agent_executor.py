"""A2A AgentExecutor for the Token Analysis agent.

Streams ADK events from the agent runner, converts them to A2A protocol
events, and delivers the final structured JSON response as a DataPart.
"""

from __future__ import annotations

import time
from contextlib import aclosing
from uuid import uuid4

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    ContentTypeNotSupportedError,
    DataPart,
    InternalError,
    Message,
    Part,
    Role,
    Task,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils import new_task
from a2a.utils.errors import ServerError
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.genai import types
from loguru import logger

from agent.agent import APP_NAME, get_or_create_session, runner
from server.card import AGENT_CARD
from server.event_converter import convert_adk_to_a2a_events, create_working_status_event

_log = logger.bind(name="server.agent_executor")


class TokenAnalysisAgentExecutor(AgentExecutor):
    """Execute natural-language token analysis queries via ADK."""

    def __init__(self) -> None:
        self.agent_card = AGENT_CARD.model_dump(mode="json", exclude_none=True)

    # ------------------------------------------------------------------
    # AgentExecutor interface
    # ------------------------------------------------------------------

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        if not context or not context.message or not context.message.parts:
            raise ServerError(error=ContentTypeNotSupportedError())

        prompt = context.get_user_input()
        _log.info(f"Received prompt: {prompt!r}")

        # Session management
        task = context.current_task
        if not task:
            if context.message is None:
                raise ServerError(error=ContentTypeNotSupportedError())
            task = new_task(context.message)
            await event_queue.enqueue_event(task)

        task_id = task.id
        context_id = context.context_id or str(uuid4())
        session_id = context_id

        user_id = "anonymous"
        if context.message and context.message.metadata:
            user_id = context.message.metadata.get("user_id", "anonymous")

        try:
            t0 = time.time()

            # Ensure session exists
            await get_or_create_session(
                app_name=APP_NAME,
                user_id=user_id,
                session_id=session_id,
            )

            # Emit initial "working" status
            await event_queue.enqueue_event(
                create_working_status_event(
                    task_id=task_id,
                    context_id=context_id,
                    message_text="Processing your request...",
                    metadata={"event_type": "processing_started"},
                )
            )

            final_response = None
            event_count = 0

            content = types.Content(
                role="user", parts=[types.Part(text=prompt)]
            )
            run_config = RunConfig(streaming_mode=StreamingMode.SSE)

            async with aclosing(
                runner.run_async(
                    user_id=user_id,
                    session_id=session_id,
                    new_message=content,
                    run_config=run_config,
                )
            ) as event_stream:
                async for adk_event in event_stream:
                    event_count += 1

                    for a2a_event in convert_adk_to_a2a_events(
                        adk_event, task_id, context_id, self.agent_card["name"]
                    ):
                        await event_queue.enqueue_event(a2a_event)

                    if adk_event.is_final_response():
                        if adk_event.content and adk_event.content.parts:
                            final_response = "".join(
                                p.text or "" for p in adk_event.content.parts
                            )

            elapsed = time.time() - t0
            _log.info(
                f"Completed in {elapsed:.2f}s — {event_count} events "
                f"(user={user_id}, session={session_id})"
            )

            # Build final message
            parts: list[Part] = [
                Part(root=TextPart(text=final_response or "No response generated."))
            ]

            # Attach last structured analysis as a DataPart
            from agent.agent import session_service

            session = await session_service.get_session(
                app_name=APP_NAME, user_id=user_id, session_id=session_id
            )
            if session:
                last_analysis = session.state.get("last_analysis")
                last_type = session.state.get("last_analysis_type")
                if last_analysis:
                    parts.append(
                        Part(
                            root=DataPart(
                                data=last_analysis,
                                metadata={"type": f"token_analysis_{last_type}"},
                            )
                        )
                    )

            final_message = Message(
                message_id=str(uuid4()),
                role=Role.agent,
                metadata={"name": self.agent_card["name"]},
                parts=parts,
            )

            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    final=True,
                    status=TaskStatus(
                        state=TaskState.completed,
                        message=final_message,
                    ),
                )
            )

        except Exception as e:
            _log.error(f"Error during execution: {e}")
            raise ServerError(error=InternalError()) from e

    async def cancel(
        self, _request: RequestContext, _event_queue: EventQueue
    ) -> Task | None:
        raise ServerError(error=UnsupportedOperationError())
