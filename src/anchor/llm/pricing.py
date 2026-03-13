"""Built-in model pricing for cost tracking.

Prices are best-effort and ship with the package. Users can override
at runtime via MODEL_PRICING["model"] = {"input": X, "output": Y}.

Last updated: 2026-03-13
Prices in USD per 1M tokens.
"""

from __future__ import annotations

import re

MODEL_PRICING: dict[str, dict[str, float]] = {
    # Anthropic
    "claude-opus-4-20250514": {"input": 15.0, "output": 75.0},
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.0},
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1": {"input": 2.0, "output": 8.0},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "o3": {"input": 10.0, "output": 40.0},
    "o3-mini": {"input": 1.10, "output": 4.40},
    "o4-mini": {"input": 1.10, "output": 4.40},
    # Google
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.0},
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    # xAI (Grok)
    "grok-3": {"input": 3.0, "output": 15.0},
    "grok-3-mini": {"input": 0.30, "output": 0.50},
}


def calculate_cost(
    model: str, prompt_tokens: int, completion_tokens: int
) -> float | None:
    """Calculate USD cost. Returns None if model pricing unknown.

    Tries exact match first, then strips date suffixes for alias matching.
    """
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        normalized = _normalize_model_name(model)
        pricing = MODEL_PRICING.get(normalized)
    if pricing is None:
        return None
    return (
        prompt_tokens * pricing["input"] + completion_tokens * pricing["output"]
    ) / 1_000_000


def _normalize_model_name(model: str) -> str:
    """Strip trailing date suffixes for alias matching.

    'gpt-4o-2024-08-06' -> 'gpt-4o'
    'model-20240806' -> 'model'
    """
    return re.sub(r"-\d{4}-?\d{2}-?\d{2}$", "", model)
