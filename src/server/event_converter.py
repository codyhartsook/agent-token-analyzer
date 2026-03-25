"""Convert ADK events to A2A event types for streaming.

Lightweight local converter so we don't depend on the recruiter package.
Follows the same patterns as ``agent_recruiter.server.event_converter``.
"""

from __future__ import annotations

from typing import Any, Generator
from uuid import uuid4

from a2a.types import (
    Artifact,
    DataPart,
    Message,
    Part,
    Role,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)
from google.adk.events.event import Event as AdkEvent


def convert_adk_to_a2a_events(
    adk_event: AdkEvent,
    task_id: str,
    context_id: str,
    agent_name: str,
) -> Generator[TaskStatusUpdateEvent | TaskArtifactUpdateEvent, None, None]:
    """Yield A2A events that correspond to a single ADK event."""

    # Agent transfers
    if adk_event.actions.transfer_to_agent:
        yield _working(
            task_id,
            context_id,
            f"Transferring to {adk_event.actions.transfer_to_agent}...",
            {
                "event_type": "agent_transfer",
                "from_agent": adk_event.author,
                "to_agent": adk_event.actions.transfer_to_agent,
            },
        )

    # Function calls (tool invocations)
    function_calls = adk_event.get_function_calls()
    if function_calls and not adk_event.partial:
        for fc in function_calls:
            args_dict = dict(fc.args) if fc.args else {}
            yield _working(
                task_id,
                context_id,
                f"Calling tool: {fc.name}",
                {
                    "event_type": "tool_call",
                    "tool_name": fc.name,
                    "tool_args": args_dict,
                    "author": adk_event.author,
                },
            )

    # Function responses
    function_responses = adk_event.get_function_responses()
    if function_responses and not adk_event.partial:
        for fr in function_responses:
            yield _working(
                task_id,
                context_id,
                f"Tool {fr.name} completed",
                {
                    "event_type": "tool_response",
                    "tool_name": fr.name,
                    "author": adk_event.author,
                },
            )

    # Artifact deltas
    if adk_event.actions.artifact_delta:
        for filename, version in adk_event.actions.artifact_delta.items():
            yield TaskArtifactUpdateEvent(
                task_id=task_id,
                context_id=context_id,
                artifact=Artifact(
                    name=filename,
                    artifact_id=str(uuid4()),
                    parts=[
                        Part(
                            root=DataPart(
                                data={"version": version},
                                metadata={"filename": filename},
                            )
                        )
                    ],
                ),
            )

    # Intermediate text (non-partial, non-final, no function content)
    if (
        adk_event.content
        and adk_event.content.parts
        and not adk_event.partial
        and not function_calls
        and not function_responses
        and not adk_event.is_final_response()
    ):
        text_parts = [p.text for p in adk_event.content.parts if p.text]
        if text_parts:
            yield _working(
                task_id,
                context_id,
                "".join(text_parts),
                {"event_type": "intermediate_response", "author": adk_event.author},
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _working(
    task_id: str,
    context_id: str,
    text: str,
    metadata: dict[str, Any] | None = None,
) -> TaskStatusUpdateEvent:
    return TaskStatusUpdateEvent(
        task_id=task_id,
        context_id=context_id,
        final=False,
        status=TaskStatus(
            state=TaskState.working,
            message=Message(
                message_id=str(uuid4()),
                role=Role.agent,
                parts=[Part(root=TextPart(text=text))],
                metadata=metadata,
            ),
        ),
    )


def create_working_status_event(
    task_id: str,
    context_id: str,
    message_text: str,
    metadata: dict[str, Any] | None = None,
    final: bool = False,
) -> TaskStatusUpdateEvent:
    """Convenience wrapper matching the recruiter's helper signature."""
    return TaskStatusUpdateEvent(
        task_id=task_id,
        context_id=context_id,
        final=final,
        status=TaskStatus(
            state=TaskState.working,
            message=Message(
                message_id=str(uuid4()),
                role=Role.agent,
                parts=[Part(root=TextPart(text=message_text))],
                metadata=metadata,
            ),
        ),
    )
