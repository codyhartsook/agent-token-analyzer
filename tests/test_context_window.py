"""Tests for context_window.py — glob matching, resolution, utilization."""

from __future__ import annotations

from token_analysis.context_window import (
    DEFAULT_CONTEXT_WINDOWS,
    ModelContextWindow,
    _find_context_window,
    get_context_utilization,
    resolve_context_window,
)


# ── _find_context_window tests ─────────────────────────────────────────────


class TestFindContextWindow:
    def test_basic_gpt4o_match(self):
        result = _find_context_window("gpt-4o", DEFAULT_CONTEXT_WINDOWS)
        assert result is not None
        assert result.max_input_tokens == 128_000

    def test_specificity_gpt4o_mini_over_gpt4o(self):
        """gpt-4o-mini should match the more specific pattern, not gpt-4o."""
        result = _find_context_window("gpt-4o-mini", DEFAULT_CONTEXT_WINDOWS)
        assert result is not None
        assert result.model_pattern == "*gpt-4o-mini*"
        assert result.max_input_tokens == 128_000

    def test_case_insensitive(self):
        result = _find_context_window("GPT-4O", DEFAULT_CONTEXT_WINDOWS)
        assert result is not None
        assert result.max_input_tokens == 128_000

    def test_claude_35_sonnet(self):
        result = _find_context_window("claude-3-5-sonnet-20241022", DEFAULT_CONTEXT_WINDOWS)
        assert result is not None
        assert result.max_input_tokens == 200_000

    def test_claude_4_opus(self):
        result = _find_context_window("claude-4-opus-20250514", DEFAULT_CONTEXT_WINDOWS)
        assert result is not None
        assert result.max_input_tokens == 200_000

    def test_gemini_15_pro(self):
        result = _find_context_window("gemini-1.5-pro", DEFAULT_CONTEXT_WINDOWS)
        assert result is not None
        assert result.max_input_tokens == 2_097_152

    def test_gemini_2(self):
        result = _find_context_window("gemini-2.0-flash", DEFAULT_CONTEXT_WINDOWS)
        assert result is not None
        assert result.max_input_tokens == 1_048_576

    def test_gpt_35_turbo(self):
        result = _find_context_window("gpt-3.5-turbo", DEFAULT_CONTEXT_WINDOWS)
        assert result is not None
        assert result.max_input_tokens == 16_384

    def test_azure_gpt35_turbo(self):
        """Azure uses gpt-35-turbo (no dot)."""
        result = _find_context_window("gpt-35-turbo", DEFAULT_CONTEXT_WINDOWS)
        assert result is not None
        assert result.max_input_tokens == 16_384

    def test_no_match(self):
        result = _find_context_window("totally-unknown-model", DEFAULT_CONTEXT_WINDOWS)
        assert result is None

    def test_o1_mini(self):
        result = _find_context_window("o1-mini", DEFAULT_CONTEXT_WINDOWS)
        assert result is not None
        assert result.model_pattern == "*o1-mini*"
        assert result.max_input_tokens == 128_000

    def test_o3(self):
        result = _find_context_window("o3-2025-04-16", DEFAULT_CONTEXT_WINDOWS)
        assert result is not None
        assert result.max_input_tokens == 200_000


# ── resolve_context_window tests ──────────────────────────────────────────


class TestResolveContextWindow:
    def test_direct_match(self):
        result = resolve_context_window("gpt-4o")
        assert result is not None
        assert result.max_input_tokens == 128_000

    def test_azure_prefix_stripping(self):
        result = resolve_context_window("azure/gpt-4o")
        assert result is not None
        assert result.max_input_tokens == 128_000

    def test_openai_prefix_stripping(self):
        result = resolve_context_window("openai/gpt-4o")
        assert result is not None
        assert result.max_input_tokens == 128_000

    def test_anthropic_prefix_stripping(self):
        result = resolve_context_window("anthropic/claude-3-5-sonnet-20241022")
        assert result is not None
        assert result.max_input_tokens == 200_000

    def test_response_model_fallback(self):
        """When model doesn't match, fall back to response_model."""
        result = resolve_context_window(
            model="custom-proxy/unknown",
            response_model="gpt-4o-2024-05-13",
        )
        assert result is not None
        assert result.max_input_tokens == 128_000

    def test_unknown_returns_none(self):
        result = resolve_context_window("totally-unknown")
        assert result is None

    def test_empty_model_with_response_model(self):
        result = resolve_context_window("", response_model="gpt-4o")
        assert result is not None
        assert result.max_input_tokens == 128_000

    def test_custom_table(self):
        custom = [
            ModelContextWindow(
                model_pattern="*custom-model*",
                max_input_tokens=50_000,
                max_output_tokens=4_096,
            )
        ]
        result = resolve_context_window("custom-model-v2", table=custom)
        assert result is not None
        assert result.max_input_tokens == 50_000


# ── get_context_utilization tests ──────────────────────────────────────────


class TestGetContextUtilization:
    def test_known_model(self):
        size, ratio = get_context_utilization(64_000, "gpt-4o")
        assert size == 128_000
        assert ratio == 0.5

    def test_unknown_model(self):
        size, ratio = get_context_utilization(100, "unknown-model")
        assert size == 0
        assert ratio == 0.0

    def test_zero_input(self):
        size, ratio = get_context_utilization(0, "gpt-4o")
        assert size == 128_000
        assert ratio == 0.0

    def test_high_utilization(self):
        size, ratio = get_context_utilization(120_000, "gpt-4o")
        assert size == 128_000
        assert ratio == 0.9375

    def test_custom_table(self):
        custom = [
            ModelContextWindow(
                model_pattern="*my-model*",
                max_input_tokens=10_000,
                max_output_tokens=1_000,
            )
        ]
        size, ratio = get_context_utilization(5_000, "my-model-v1", table=custom)
        assert size == 10_000
        assert ratio == 0.5
