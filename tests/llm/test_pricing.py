"""Tests for anchor.llm.pricing."""

from __future__ import annotations

import pytest

from anchor.llm.pricing import MODEL_PRICING, calculate_cost, _normalize_model_name


class TestCalculateCost:
    def test_known_model(self):
        cost = calculate_cost("gpt-4o", prompt_tokens=1_000_000, completion_tokens=0)
        assert cost == 2.50  # $2.50 per 1M input tokens

    def test_known_model_output(self):
        cost = calculate_cost("gpt-4o", prompt_tokens=0, completion_tokens=1_000_000)
        assert cost == 10.0  # $10 per 1M output tokens

    def test_mixed_tokens(self):
        cost = calculate_cost("gpt-4o", prompt_tokens=500_000, completion_tokens=100_000)
        assert cost == pytest.approx(1.25 + 1.0)

    def test_unknown_model_returns_none(self):
        cost = calculate_cost("unknown-model-xyz", prompt_tokens=100, completion_tokens=50)
        assert cost is None

    def test_zero_tokens(self):
        cost = calculate_cost("gpt-4o", prompt_tokens=0, completion_tokens=0)
        assert cost == 0.0

    def test_alias_normalization(self):
        """Model with date suffix should match base model pricing."""
        cost = calculate_cost("gpt-4o-2024-08-06", prompt_tokens=1_000_000, completion_tokens=0)
        assert cost == 2.50

    def test_anthropic_model(self):
        cost = calculate_cost(
            "claude-haiku-4-5-20251001",
            prompt_tokens=1_000_000,
            completion_tokens=0,
        )
        assert cost == 0.80


class TestNormalizeModelName:
    def test_strips_date_suffix_dashes(self):
        assert _normalize_model_name("gpt-4o-2024-08-06") == "gpt-4o"

    def test_strips_date_suffix_no_dashes(self):
        assert _normalize_model_name("model-20240806") == "model"

    def test_preserves_non_date_suffix(self):
        assert _normalize_model_name("gpt-4o-mini") == "gpt-4o-mini"

    def test_preserves_canonical_anthropic(self):
        result = _normalize_model_name("claude-sonnet-4-20250514")
        assert isinstance(result, str)
