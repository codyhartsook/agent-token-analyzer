"""Cost estimation for LLM API calls.

Provides a configurable pricing table for common models and computes
per-call and aggregate cost estimates including cached input savings.
"""

from __future__ import annotations

import fnmatch

from .models import CostEstimate, LLMCallTokens, ModelPricing


# ── Default pricing table (USD per 1M tokens, as of early 2026) ─────────────

DEFAULT_PRICING: list[ModelPricing] = [
    # OpenAI GPT-4o family
    ModelPricing(
        model_pattern="*gpt-4o*",
        input_per_1m=2.50,
        output_per_1m=10.00,
        cached_input_per_1m=1.25,
    ),
    ModelPricing(
        model_pattern="*gpt-4o-mini*",
        input_per_1m=0.15,
        output_per_1m=0.60,
        cached_input_per_1m=0.075,
    ),
    # OpenAI o1/o3 reasoning models
    ModelPricing(
        model_pattern="*o1*",
        input_per_1m=15.00,
        output_per_1m=60.00,
        cached_input_per_1m=7.50,
        reasoning_per_1m=60.00,
    ),
    ModelPricing(
        model_pattern="*o3*",
        input_per_1m=10.00,
        output_per_1m=40.00,
        cached_input_per_1m=5.00,
        reasoning_per_1m=40.00,
    ),
    # Anthropic Claude 3.5/4
    ModelPricing(
        model_pattern="*claude-3-5-sonnet*",
        input_per_1m=3.00,
        output_per_1m=15.00,
        cached_input_per_1m=1.50,
    ),
    ModelPricing(
        model_pattern="*claude-3-5-haiku*",
        input_per_1m=0.80,
        output_per_1m=4.00,
        cached_input_per_1m=0.40,
    ),
    ModelPricing(
        model_pattern="*claude-4-sonnet*",
        input_per_1m=3.00,
        output_per_1m=15.00,
        cached_input_per_1m=1.50,
    ),
    ModelPricing(
        model_pattern="*claude-4-opus*",
        input_per_1m=15.00,
        output_per_1m=75.00,
        cached_input_per_1m=7.50,
    ),
    # Google Gemini
    ModelPricing(
        model_pattern="*gemini-1.5-pro*",
        input_per_1m=1.25,
        output_per_1m=5.00,
    ),
    ModelPricing(
        model_pattern="*gemini-2*",
        input_per_1m=1.25,
        output_per_1m=5.00,
    ),
]


# ── Pricing lookup ──────────────────────────────────────────────────────────


def _find_pricing(
    model_name: str,
    pricing_table: list[ModelPricing],
) -> ModelPricing | None:
    """Find matching pricing for a model name using glob patterns.

    More specific patterns (longer) are checked first.
    """
    # Sort by pattern length descending for best-match priority
    sorted_table = sorted(
        pricing_table, key=lambda p: len(p.model_pattern), reverse=True
    )
    name_lower = model_name.lower()
    for pricing in sorted_table:
        if fnmatch.fnmatch(name_lower, pricing.model_pattern.lower()):
            return pricing
    return None


# ── Cost computation ────────────────────────────────────────────────────────


def estimate_call_cost(
    call: LLMCallTokens,
    pricing_table: list[ModelPricing] | None = None,
) -> float:
    """Estimate cost for a single LLM call. Returns USD."""
    table = pricing_table or DEFAULT_PRICING
    pricing = _find_pricing(call.model or call.response_model, table)
    if pricing is None:
        return 0.0

    # Billable input = input - cached
    billable_input = max(0, call.input_tokens - call.cache_read_input_tokens)
    cached_input = call.cache_read_input_tokens

    cost = 0.0
    cost += billable_input * pricing.input_per_1m / 1_000_000
    cost += cached_input * pricing.cached_input_per_1m / 1_000_000
    cost += call.output_tokens * pricing.output_per_1m / 1_000_000
    if call.reasoning_tokens > 0 and pricing.reasoning_per_1m > 0:
        cost += call.reasoning_tokens * pricing.reasoning_per_1m / 1_000_000

    return cost


def estimate_cost(
    calls: list[LLMCallTokens],
    pricing_table: list[ModelPricing] | None = None,
) -> CostEstimate:
    """Estimate total cost for a list of LLM calls."""
    table = pricing_table or DEFAULT_PRICING

    total_cost = 0.0
    input_cost = 0.0
    output_cost = 0.0
    cached_cost = 0.0
    reasoning_cost = 0.0
    per_model: dict[str, float] = {}
    per_agent: dict[str, float] = {}
    no_pricing = 0

    for call in calls:
        model_key = call.model or call.response_model or "unknown"
        pricing = _find_pricing(model_key, table)

        if pricing is None:
            no_pricing += 1
            continue

        billable_input = max(
            0, call.input_tokens - call.cache_read_input_tokens
        )
        cached_input = call.cache_read_input_tokens

        c_input = billable_input * pricing.input_per_1m / 1_000_000
        c_cached = cached_input * pricing.cached_input_per_1m / 1_000_000
        c_output = call.output_tokens * pricing.output_per_1m / 1_000_000
        c_reasoning = 0.0
        if call.reasoning_tokens > 0 and pricing.reasoning_per_1m > 0:
            c_reasoning = (
                call.reasoning_tokens * pricing.reasoning_per_1m / 1_000_000
            )

        call_total = c_input + c_cached + c_output + c_reasoning
        total_cost += call_total
        input_cost += c_input
        output_cost += c_output
        cached_cost += c_cached
        reasoning_cost += c_reasoning

        per_model[model_key] = per_model.get(model_key, 0.0) + call_total
        agent_key = call.agent_name or call.service_name or "unknown"
        per_agent[agent_key] = per_agent.get(agent_key, 0.0) + call_total

    return CostEstimate(
        total_cost_usd=round(total_cost, 6),
        input_cost_usd=round(input_cost, 6),
        output_cost_usd=round(output_cost, 6),
        cached_input_cost_usd=round(cached_cost, 6),
        reasoning_cost_usd=round(reasoning_cost, 6),
        per_model={k: round(v, 6) for k, v in per_model.items()},
        per_agent={k: round(v, 6) for k, v in per_agent.items()},
        calls_without_pricing=no_pricing,
    )
