"""Context window reconstruction and accumulation detection.

Parses ``gen_ai.prompt.{N}.*`` flat span attributes back into structured
conversation messages, then analyzes growth patterns between sequential
LLM calls.
"""

from __future__ import annotations

import re

from .models import (
    ContextAccumulationAlert,
    ContextGrowthStep,
    ContextWindowSnapshot,
    LLMCallTokens,
    PromptMessage,
    ToolCallInfo,
)


# ── Attribute key patterns ──────────────────────────────────────────────────

_PROMPT_KEY_RE = re.compile(r"^gen_ai\.prompt\.(\d+)\.(.+)$")
_TOOL_CALL_RE = re.compile(r"^tool_calls\.(\d+)\.(.+)$")


# ── Context reconstruction ──────────────────────────────────────────────────


def reconstruct_context(
    span_attributes: dict[str, str],
    span_id: str,
    call_index: int,
    agent_name: str = "",
    model: str = "",
    timestamp_ns: int = 0,
) -> ContextWindowSnapshot:
    """Reconstruct the full conversation context from span attributes.

    Algorithm:
        1. Scan all attributes matching ``gen_ai.prompt.{N}.*``
        2. Group by N (message index)
        3. For each message, extract role, content, tool_calls, tool_call_id
        4. Sort by N to get message order
        5. Extract completion from ``gen_ai.completion.0.*``
        6. Compute derived metrics
    """
    # Step 1-2: Group attributes by message index
    message_attrs: dict[int, dict[str, str]] = {}

    for key, value in span_attributes.items():
        m = _PROMPT_KEY_RE.match(key)
        if m:
            idx = int(m.group(1))
            rest = m.group(2)
            if idx not in message_attrs:
                message_attrs[idx] = {}
            message_attrs[idx][rest] = value

    # Step 3-4: Build PromptMessage objects
    messages: list[PromptMessage] = []
    for idx in sorted(message_attrs.keys()):
        attrs = message_attrs[idx]

        # Parse tool calls within this message
        tool_calls = _parse_tool_calls(attrs)

        content = attrs.get("content", "")
        messages.append(
            PromptMessage(
                index=idx,
                role=attrs.get("role", "unknown"),
                content=content,
                content_chars=len(content),
                tool_calls=tool_calls,
                tool_call_id=attrs.get("tool_call_id", ""),
            )
        )

    # Step 5: Extract completion
    completion_content = span_attributes.get("gen_ai.completion.0.content", "")
    completion_finish_reason = span_attributes.get(
        "gen_ai.completion.0.finish_reason", ""
    )

    # Step 6: Derive metrics
    messages_by_role: dict[str, int] = {}
    total_chars = 0
    total_tool_calls = 0
    for msg in messages:
        messages_by_role[msg.role] = messages_by_role.get(msg.role, 0) + 1
        total_chars += msg.content_chars
        total_tool_calls += len(msg.tool_calls)

    return ContextWindowSnapshot(
        span_id=span_id,
        call_index=call_index,
        agent_name=agent_name,
        model=model,
        timestamp_ns=timestamp_ns,
        messages=messages,
        completion_content=completion_content,
        completion_finish_reason=completion_finish_reason,
        total_messages=len(messages),
        messages_by_role=messages_by_role,
        total_content_chars=total_chars,
        total_tool_calls=total_tool_calls,
    )


def _parse_tool_calls(attrs: dict[str, str]) -> list[ToolCallInfo]:
    """Parse ``tool_calls.{M}.name/arguments`` from message attributes."""
    tc_data: dict[int, dict[str, str]] = {}
    for key, value in attrs.items():
        m = _TOOL_CALL_RE.match(key)
        if m:
            tc_idx = int(m.group(1))
            tc_field = m.group(2)
            if tc_idx not in tc_data:
                tc_data[tc_idx] = {}
            tc_data[tc_idx][tc_field] = value

    return [
        ToolCallInfo(
            index=idx,
            name=data.get("name", ""),
            arguments=data.get("arguments", ""),
        )
        for idx, data in sorted(tc_data.items())
    ]


# ── Context growth analysis ─────────────────────────────────────────────────


def compute_context_growth(
    snapshots: list[ContextWindowSnapshot],
    llm_calls: list[LLMCallTokens] | None = None,
) -> list[ContextGrowthStep]:
    """Compute growth deltas between sequential LLM calls.

    Compares each pair of consecutive context snapshots to measure:
    - How many messages were added
    - How much the content grew (chars)
    - Which roles were added
    """
    if len(snapshots) < 2:
        return []

    # Build a map from span_id to LLMCallTokens for token deltas
    token_map: dict[str, LLMCallTokens] = {}
    if llm_calls:
        token_map = {c.span_id: c for c in llm_calls}

    steps: list[ContextGrowthStep] = []
    for i in range(1, len(snapshots)):
        prev = snapshots[i - 1]
        curr = snapshots[i]

        msg_delta = curr.total_messages - prev.total_messages
        char_delta = curr.total_content_chars - prev.total_content_chars

        # Role breakdown of new messages
        new_roles: dict[str, int] = {}
        for role, count in curr.messages_by_role.items():
            prev_count = prev.messages_by_role.get(role, 0)
            if count > prev_count:
                new_roles[role] = count - prev_count

        growth_pct = (
            (char_delta / prev.total_content_chars * 100.0)
            if prev.total_content_chars > 0
            else 0.0
        )

        # Token delta from LLMCallTokens if available
        input_delta = 0
        prev_tokens = token_map.get(prev.span_id)
        curr_tokens = token_map.get(curr.span_id)
        if prev_tokens and curr_tokens:
            input_delta = curr_tokens.input_tokens - prev_tokens.input_tokens

        steps.append(
            ContextGrowthStep(
                from_call_index=prev.call_index,
                to_call_index=curr.call_index,
                from_span_id=prev.span_id,
                to_span_id=curr.span_id,
                message_count_delta=msg_delta,
                content_chars_delta=char_delta,
                input_tokens_delta=input_delta,
                new_roles=new_roles,
                growth_pct=round(growth_pct, 1),
            )
        )

    return steps


# ── Accumulation detection ──────────────────────────────────────────────────


def detect_accumulation(
    trace_id: str,
    llm_calls: list[LLMCallTokens],
    growth_steps: list[ContextGrowthStep],
    *,
    growth_factor_warn: float = 2.0,
    growth_factor_critical: float = 5.0,
) -> list[ContextAccumulationAlert]:
    """Flag traces where context grows without bound.

    Triggers when ``growth_factor >= growth_factor_warn``
    (final_input / initial_input).
    """
    if len(llm_calls) < 2:
        return []

    alerts: list[ContextAccumulationAlert] = []

    # Group by agent for per-agent accumulation detection
    by_agent: dict[str, list[LLMCallTokens]] = {}
    for call in llm_calls:
        agent = call.agent_name or call.service_name or "unknown"
        by_agent.setdefault(agent, []).append(call)

    for agent_name, agent_calls in by_agent.items():
        if len(agent_calls) < 2:
            continue

        initial = agent_calls[0].input_tokens
        final = agent_calls[-1].input_tokens

        if initial == 0:
            continue

        factor = final / initial

        if factor >= growth_factor_warn:
            severity = (
                "critical" if factor >= growth_factor_critical else "warning"
            )
            alerts.append(
                ContextAccumulationAlert(
                    trace_id=trace_id,
                    agent_name=agent_name,
                    call_count=len(agent_calls),
                    initial_input_tokens=initial,
                    final_input_tokens=final,
                    growth_factor=round(factor, 2),
                    growth_steps=growth_steps,
                    severity=severity,
                )
            )

    return alerts
