"""Context window size lookup for LLM models.

Provides a configurable lookup table for model context window sizes
and computes per-call utilization ratios.  Follows the same pattern
as ``cost.py`` — hardcoded table with glob matching, longest pattern wins.
"""

from __future__ import annotations

import fnmatch

from pydantic import BaseModel


# ── Model spec ─────────────────────────────────────────────────────────────


class ModelContextWindow(BaseModel):
    """Context window specification for a model pattern."""

    model_pattern: str  # glob pattern, e.g. "*gpt-4o*"
    max_input_tokens: int  # max tokens in the input context
    max_output_tokens: int  # max tokens the model can generate


# ── Default context window table ───────────────────────────────────────────
#
# More specific patterns (longer) are listed first for readability, but
# lookup sorts by pattern length descending anyway.

DEFAULT_CONTEXT_WINDOWS: list[ModelContextWindow] = [
    # OpenAI GPT-4o family (128K input, 16K output)
    ModelContextWindow(
        model_pattern="*gpt-4o-mini*",
        max_input_tokens=128_000,
        max_output_tokens=16_384,
    ),
    ModelContextWindow(
        model_pattern="*gpt-4o*",
        max_input_tokens=128_000,
        max_output_tokens=16_384,
    ),
    # OpenAI GPT-4 Turbo (128K)
    ModelContextWindow(
        model_pattern="*gpt-4-turbo*",
        max_input_tokens=128_000,
        max_output_tokens=4_096,
    ),
    # OpenAI GPT-4 32K
    ModelContextWindow(
        model_pattern="*gpt-4-32k*",
        max_input_tokens=32_768,
        max_output_tokens=4_096,
    ),
    # OpenAI GPT-4 (base 8K)
    ModelContextWindow(
        model_pattern="*gpt-4*",
        max_input_tokens=8_192,
        max_output_tokens=4_096,
    ),
    # OpenAI GPT-3.5 Turbo — Azure uses "gpt-35-turbo", OpenAI uses "gpt-3.5-turbo"
    ModelContextWindow(
        model_pattern="*gpt-35-turbo*",
        max_input_tokens=16_384,
        max_output_tokens=4_096,
    ),
    ModelContextWindow(
        model_pattern="*gpt-3.5-turbo*",
        max_input_tokens=16_384,
        max_output_tokens=4_096,
    ),
    # OpenAI o1 reasoning models
    ModelContextWindow(
        model_pattern="*o1-mini*",
        max_input_tokens=128_000,
        max_output_tokens=65_536,
    ),
    ModelContextWindow(
        model_pattern="*o1*",
        max_input_tokens=200_000,
        max_output_tokens=100_000,
    ),
    # OpenAI o3 reasoning models
    ModelContextWindow(
        model_pattern="*o3-mini*",
        max_input_tokens=200_000,
        max_output_tokens=100_000,
    ),
    ModelContextWindow(
        model_pattern="*o3*",
        max_input_tokens=200_000,
        max_output_tokens=100_000,
    ),
    # Anthropic Claude 3.5
    ModelContextWindow(
        model_pattern="*claude-3-5-sonnet*",
        max_input_tokens=200_000,
        max_output_tokens=8_192,
    ),
    ModelContextWindow(
        model_pattern="*claude-3-5-haiku*",
        max_input_tokens=200_000,
        max_output_tokens=8_192,
    ),
    # Anthropic Claude 4
    ModelContextWindow(
        model_pattern="*claude-4-sonnet*",
        max_input_tokens=200_000,
        max_output_tokens=16_384,
    ),
    ModelContextWindow(
        model_pattern="*claude-4-opus*",
        max_input_tokens=200_000,
        max_output_tokens=16_384,
    ),
    # Google Gemini
    ModelContextWindow(
        model_pattern="*gemini-1.5-pro*",
        max_input_tokens=2_097_152,
        max_output_tokens=8_192,
    ),
    ModelContextWindow(
        model_pattern="*gemini-1.5-flash*",
        max_input_tokens=1_048_576,
        max_output_tokens=8_192,
    ),
    ModelContextWindow(
        model_pattern="*gemini-2*",
        max_input_tokens=1_048_576,
        max_output_tokens=8_192,
    ),
]


# ── Lookup functions ───────────────────────────────────────────────────────


def _find_context_window(
    model_name: str,
    table: list[ModelContextWindow],
) -> ModelContextWindow | None:
    """Find matching context window for a model name using glob patterns.

    More specific patterns (longer) are checked first.
    """
    sorted_table = sorted(
        table, key=lambda e: len(e.model_pattern), reverse=True
    )
    name_lower = model_name.lower()
    for entry in sorted_table:
        if fnmatch.fnmatch(name_lower, entry.model_pattern.lower()):
            return entry
    return None


def resolve_context_window(
    model: str,
    response_model: str = "",
    table: list[ModelContextWindow] | None = None,
) -> ModelContextWindow | None:
    """Resolve context window for a model with fallback chain.

    Tries: exact model → strip provider prefix → response_model.
    Returns ``None`` if no match found.
    """
    t = table or DEFAULT_CONTEXT_WINDOWS

    if model:
        result = _find_context_window(model, t)
        if result:
            return result

        # Try stripping common provider prefixes
        for prefix in ("azure/", "openai/", "anthropic/", "google/"):
            if model.lower().startswith(prefix):
                stripped = model[len(prefix):]
                result = _find_context_window(stripped, t)
                if result:
                    return result

    if response_model and response_model != model:
        result = _find_context_window(response_model, t)
        if result:
            return result

    return None


def get_context_utilization(
    input_tokens: int,
    model: str,
    response_model: str = "",
    table: list[ModelContextWindow] | None = None,
) -> tuple[int, float]:
    """Get context window size and utilization ratio for an LLM call.

    Args:
        input_tokens: Number of input tokens used.
        model: The request model name (e.g. ``"azure/gpt-4o"``).
        response_model: The response model name (fallback).
        table: Optional custom lookup table.

    Returns:
        ``(context_window_size, utilization_ratio)`` — ``(0, 0.0)`` if
        the model is not in the lookup table.
    """
    spec = resolve_context_window(model, response_model, table)
    if spec is None or spec.max_input_tokens == 0:
        return (0, 0.0)

    utilization = round(input_tokens / spec.max_input_tokens, 4)
    return (spec.max_input_tokens, utilization)
