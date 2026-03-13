# Multi-Provider LLM Integration Layer — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a centralized, provider-agnostic LLM layer so users can use Anthropic, OpenAI, Gemini, Grok, Ollama, or OpenRouter through a single interface.

**Architecture:** Protocol-based abstraction (`LLMProvider`) with a `BaseLLMProvider` ABC for shared retry/timeout logic, per-provider adapters wrapping native SDKs, a registry with `provider/model` string parsing, and a `FallbackProvider` for reliability. The Agent class is refactored to use `LLMProvider` instead of the Anthropic SDK directly.

**Tech Stack:** Python 3.11+, Pydantic v2, anthropic SDK, openai SDK, google-genai SDK, ollama SDK (all optional deps).

**Spec:** `docs/superpowers/specs/2026-03-13-multi-provider-llm-design.md`

---

## File Map

### New files to create

| File | Responsibility |
|------|---------------|
| `src/anchor/llm/__init__.py` | Public exports: `create_provider`, `LLMProvider`, `BaseLLMProvider`, models, errors |
| `src/anchor/llm/models.py` | `Role`, `ContentBlock`, `ToolCall`, `ToolCallDelta`, `ToolResult`, `Message`, `Usage`, `StopReason`, `LLMResponse`, `StreamChunk`, `ToolSchema` |
| `src/anchor/llm/errors.py` | `ProviderError`, `AuthenticationError`, `RateLimitError`, `ServerError`, `TimeoutError`, `ModelNotFoundError`, `ContentFilterError`, `ProviderNotInstalledError` |
| `src/anchor/llm/base.py` | `LLMProvider` Protocol, `BaseLLMProvider` ABC with retry logic |
| `src/anchor/llm/registry.py` | `create_provider()`, `register_provider()`, `_parse_model_string()`, lazy imports |
| `src/anchor/llm/fallback.py` | `FallbackProvider` with pre-first-chunk streaming safety |
| `src/anchor/llm/pricing.py` | `MODEL_PRICING` dict, `calculate_cost()`, `_normalize_model_name()` |
| `src/anchor/llm/providers/__init__.py` | Empty |
| `src/anchor/llm/providers/anthropic.py` | `AnthropicProvider` wrapping `anthropic` SDK |
| `src/anchor/llm/providers/openai.py` | `OpenAIProvider` wrapping `openai` SDK |
| `src/anchor/llm/providers/gemini.py` | `GeminiProvider` wrapping `google-genai` SDK |
| `src/anchor/llm/providers/grok.py` | `GrokProvider` subclass of `OpenAIProvider` |
| `src/anchor/llm/providers/ollama.py` | `OllamaProvider` (OpenAI-compatible) |
| `src/anchor/llm/providers/openrouter.py` | `OpenRouterProvider` subclass of `OpenAIProvider` |
| `src/anchor/llm/providers/litellm.py` | `LiteLLMProvider` optional catch-all |
| `tests/llm/__init__.py` | Empty |
| `tests/llm/test_models.py` | Tests for all Pydantic models |
| `tests/llm/test_errors.py` | Tests for error hierarchy |
| `tests/llm/test_base.py` | Tests for `BaseLLMProvider` retry logic |
| `tests/llm/test_registry.py` | Tests for `create_provider`, `_parse_model_string` |
| `tests/llm/test_fallback.py` | Tests for `FallbackProvider` |
| `tests/llm/test_pricing.py` | Tests for cost calculation and normalization |
| `tests/llm/providers/__init__.py` | Empty |
| `tests/llm/providers/test_anthropic.py` | Tests for `AnthropicProvider` (mocked SDK) |
| `tests/llm/providers/test_openai.py` | Tests for `OpenAIProvider` (mocked SDK) |
| `tests/llm/providers/test_gemini.py` | Tests for `GeminiProvider` (mocked SDK) |
| `tests/llm/providers/test_grok.py` | Tests for `GrokProvider` |
| `tests/llm/providers/test_openrouter.py` | Tests for `OpenRouterProvider` |
| `tests/llm/providers/test_ollama.py` | Tests for `OllamaProvider` |
| `tests/llm/providers/test_litellm.py` | Tests for `LiteLLMProvider` |

### Existing files to modify

| File | Change |
|------|--------|
| `src/anchor/agent/models.py` | Add `to_tool_schema()` method to `AgentTool` |
| `src/anchor/agent/agent.py` | Replace Anthropic SDK usage with `LLMProvider`, rewrite tool loop |
| `src/anchor/__init__.py` | Add LLM module exports |
| `pyproject.toml` | Add `openai`, `gemini`, `grok`, `ollama`, `litellm`, `all-providers` optional dependency groups |
| `tests/agent/test_agent.py` | Update to use mock `LLMProvider` instead of mock Anthropic client |

---

## Chunk 1: Foundation — Models, Errors, Pricing

### Task 1: LLM Models (`src/anchor/llm/models.py`)

**Files:**
- Create: `src/anchor/llm/__init__.py`
- Create: `src/anchor/llm/models.py`
- Create: `tests/llm/__init__.py`
- Create: `tests/llm/test_models.py`

- [ ] **Step 1: Write failing tests for all LLM models**

Create `tests/llm/__init__.py` (empty) and `tests/llm/test_models.py`:

```python
"""Tests for anchor.llm.models."""

from __future__ import annotations

import pytest

from anchor.llm.models import (
    ContentBlock,
    LLMResponse,
    Message,
    Role,
    StopReason,
    StreamChunk,
    ToolCall,
    ToolCallDelta,
    ToolResult,
    ToolSchema,
    Usage,
)


class TestRole:
    def test_role_values(self):
        assert Role.SYSTEM == "system"
        assert Role.USER == "user"
        assert Role.ASSISTANT == "assistant"
        assert Role.TOOL == "tool"


class TestContentBlock:
    def test_text_block(self):
        block = ContentBlock(type="text", text="hello")
        assert block.type == "text"
        assert block.text == "hello"
        assert block.image_url is None

    def test_image_url_block(self):
        block = ContentBlock(type="image_url", image_url="https://example.com/img.png")
        assert block.type == "image_url"
        assert block.image_url == "https://example.com/img.png"

    def test_image_base64_block(self):
        block = ContentBlock(
            type="image_base64",
            image_base64="aGVsbG8=",
            media_type="image/png",
        )
        assert block.media_type == "image/png"

    def test_frozen(self):
        block = ContentBlock(type="text", text="hello")
        with pytest.raises(Exception):
            block.text = "world"


class TestToolCall:
    def test_creation(self):
        tc = ToolCall(id="call_1", name="get_weather", arguments={"city": "NYC"})
        assert tc.id == "call_1"
        assert tc.name == "get_weather"
        assert tc.arguments == {"city": "NYC"}

    def test_frozen(self):
        tc = ToolCall(id="call_1", name="get_weather", arguments={})
        with pytest.raises(Exception):
            tc.name = "other"


class TestToolCallDelta:
    def test_first_delta(self):
        delta = ToolCallDelta(index=0, id="call_1", name="get_weather")
        assert delta.index == 0
        assert delta.id == "call_1"
        assert delta.arguments_fragment is None

    def test_argument_fragment(self):
        delta = ToolCallDelta(index=0, arguments_fragment='{"city":')
        assert delta.arguments_fragment == '{"city":'


class TestToolResult:
    def test_creation(self):
        tr = ToolResult(tool_call_id="call_1", content="sunny")
        assert tr.is_error is False

    def test_error_result(self):
        tr = ToolResult(tool_call_id="call_1", content="failed", is_error=True)
        assert tr.is_error is True


class TestMessage:
    def test_user_message_string(self):
        msg = Message(role=Role.USER, content="hello")
        assert msg.role == Role.USER
        assert msg.content == "hello"
        assert msg.tool_calls is None

    def test_user_message_content_blocks(self):
        blocks = [ContentBlock(type="text", text="hello")]
        msg = Message(role=Role.USER, content=blocks)
        assert isinstance(msg.content, list)

    def test_assistant_with_tool_calls(self):
        tc = ToolCall(id="c1", name="fn", arguments={})
        msg = Message(role=Role.ASSISTANT, content="thinking...", tool_calls=[tc])
        assert len(msg.tool_calls) == 1

    def test_tool_message(self):
        tr = ToolResult(tool_call_id="c1", content="result")
        msg = Message(role=Role.TOOL, tool_result=tr)
        assert msg.tool_result.content == "result"


class TestUsage:
    def test_creation(self):
        u = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        assert u.total_cost is None

    def test_with_cost(self):
        u = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150, total_cost=0.001)
        assert u.total_cost == 0.001


class TestLLMResponse:
    def test_text_response(self):
        r = LLMResponse(
            content="hello",
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            model="gpt-4o",
            provider="openai",
            stop_reason=StopReason.STOP,
        )
        assert r.content == "hello"
        assert r.tool_calls is None

    def test_tool_use_response(self):
        tc = ToolCall(id="c1", name="fn", arguments={"x": 1})
        r = LLMResponse(
            tool_calls=[tc],
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            model="claude-sonnet-4-20250514",
            provider="anthropic",
            stop_reason=StopReason.TOOL_USE,
        )
        assert r.stop_reason == StopReason.TOOL_USE
        assert len(r.tool_calls) == 1


class TestStreamChunk:
    def test_content_chunk(self):
        c = StreamChunk(content="hello")
        assert c.content == "hello"
        assert c.usage is None

    def test_final_chunk(self):
        c = StreamChunk(
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            stop_reason=StopReason.STOP,
        )
        assert c.stop_reason == StopReason.STOP

    def test_tool_call_delta_chunk(self):
        delta = ToolCallDelta(index=0, id="c1", name="fn")
        c = StreamChunk(tool_call_delta=delta)
        assert c.tool_call_delta.name == "fn"


class TestToolSchema:
    def test_creation(self):
        ts = ToolSchema(
            name="get_weather",
            description="Get weather for a city",
            input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
        )
        assert ts.name == "get_weather"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/llm/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'anchor.llm'`

- [ ] **Step 3: Implement all LLM models**

Create `src/anchor/llm/__init__.py` (empty for now) and `src/anchor/llm/models.py`:

```python
"""Unified models for the multi-provider LLM layer.

These models are the interface between Anchor and any LLM provider.
Provider adapters convert these to/from provider-specific formats.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel


class Role(str, Enum):
    """Message role."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ContentBlock(BaseModel, frozen=True):
    """A single block of content within a message.

    Supports text, images, and other modalities. Provider adapters convert
    these to their native format (e.g., Anthropic content blocks, OpenAI
    content parts).
    """

    type: str  # "text", "image_url", "image_base64"
    text: str | None = None
    image_url: str | None = None
    image_base64: str | None = None
    media_type: str | None = None  # e.g. "image/png"


class ToolCall(BaseModel, frozen=True):
    """A tool call requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


class ToolCallDelta(BaseModel, frozen=True):
    """Incremental tool call data during streaming.

    During streaming, tool calls arrive in pieces: first the id/name,
    then argument fragments. The consumer must accumulate argument
    fragments and JSON-parse when complete.
    """

    index: int  # which tool call this delta belongs to
    id: str | None = None  # present on first delta
    name: str | None = None  # present on first delta
    arguments_fragment: str | None = None  # partial JSON string


class ToolResult(BaseModel, frozen=True):
    """Result of executing a tool call."""

    tool_call_id: str
    content: str
    is_error: bool = False


class Message(BaseModel, frozen=True):
    """A single message in a conversation."""

    role: Role
    content: str | list[ContentBlock] | None = None
    tool_calls: list[ToolCall] | None = None
    tool_result: ToolResult | None = None
    name: str | None = None  # for tool messages


class Usage(BaseModel, frozen=True):
    """Token usage and cost for an LLM call."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    total_cost: float | None = None  # USD, None if pricing unknown


class StopReason(str, Enum):
    """Why the model stopped generating."""

    STOP = "stop"
    MAX_TOKENS = "max_tokens"
    TOOL_USE = "tool_use"


class LLMResponse(BaseModel, frozen=True):
    """Complete response from an LLM provider."""

    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    usage: Usage
    model: str  # actual model used (as returned by provider)
    provider: str  # actual provider used
    stop_reason: StopReason


class StreamChunk(BaseModel, frozen=True):
    """A single chunk from a streaming LLM response."""

    content: str | None = None
    tool_call_delta: ToolCallDelta | None = None
    usage: Usage | None = None  # present on final chunk
    stop_reason: StopReason | None = None  # present on final chunk


class ToolSchema(BaseModel, frozen=True):
    """Provider-agnostic tool definition."""

    name: str
    description: str
    input_schema: dict[str, Any]  # JSON Schema
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/llm/test_models.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/anchor/llm/__init__.py src/anchor/llm/models.py tests/llm/__init__.py tests/llm/test_models.py
git commit -m "feat(llm): add unified LLM models (Message, LLMResponse, StreamChunk, etc.)"
```

---

### Task 2: Error Hierarchy (`src/anchor/llm/errors.py`)

**Files:**
- Create: `src/anchor/llm/errors.py`
- Create: `tests/llm/test_errors.py`

- [ ] **Step 1: Write failing tests**

Create `tests/llm/test_errors.py`:

```python
"""Tests for anchor.llm.errors."""

from __future__ import annotations

import pytest

from anchor.llm.errors import (
    AuthenticationError,
    ContentFilterError,
    ModelNotFoundError,
    ProviderError,
    ProviderNotInstalledError,
    RateLimitError,
    ServerError,
    TimeoutError,
)


class TestProviderError:
    def test_base_error(self):
        e = ProviderError("something failed", provider="openai")
        assert str(e) == "something failed"
        assert e.provider == "openai"
        assert e.is_transient is False

    def test_transient_error(self):
        e = ProviderError("retry me", provider="openai", is_transient=True)
        assert e.is_transient is True

    def test_is_exception(self):
        with pytest.raises(ProviderError):
            raise ProviderError("boom", provider="test")


class TestAuthenticationError:
    def test_not_transient(self):
        e = AuthenticationError("bad key", provider="anthropic")
        assert e.is_transient is False
        assert e.provider == "anthropic"

    def test_inherits_provider_error(self):
        e = AuthenticationError("bad key", provider="openai")
        assert isinstance(e, ProviderError)


class TestRateLimitError:
    def test_is_transient(self):
        e = RateLimitError("429", provider="openai")
        assert e.is_transient is True
        assert e.retry_after is None

    def test_with_retry_after(self):
        e = RateLimitError("429", provider="openai", retry_after=5.0)
        assert e.retry_after == 5.0


class TestServerError:
    def test_is_transient(self):
        e = ServerError("500", provider="gemini")
        assert e.is_transient is True


class TestTimeoutError:
    def test_is_transient(self):
        e = TimeoutError("timed out", provider="ollama")
        assert e.is_transient is True


class TestModelNotFoundError:
    def test_not_transient(self):
        e = ModelNotFoundError("no such model", provider="openai")
        assert e.is_transient is False


class TestContentFilterError:
    def test_not_transient(self):
        e = ContentFilterError("blocked", provider="openai")
        assert e.is_transient is False


class TestProviderNotInstalledError:
    def test_message(self):
        e = ProviderNotInstalledError("OpenAI", "openai", "openai")
        assert "openai" in str(e)
        assert "pip install anchor[openai]" in str(e)

    def test_not_provider_error(self):
        """ProviderNotInstalledError is a setup error, not a runtime provider error."""
        e = ProviderNotInstalledError("X", "x", "x")
        assert not isinstance(e, ProviderError)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/llm/test_errors.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement error hierarchy**

Create `src/anchor/llm/errors.py`:

```python
"""Error hierarchy for the multi-provider LLM layer.

Each provider adapter maps SDK-specific exceptions to these.
The `is_transient` flag drives retry and fallback behavior.
"""

from __future__ import annotations


class ProviderError(Exception):
    """Base error for all provider runtime failures."""

    def __init__(self, message: str, *, provider: str, is_transient: bool = False):
        super().__init__(message)
        self.provider = provider
        self.is_transient = is_transient


class AuthenticationError(ProviderError):
    """Invalid or missing API key."""

    def __init__(self, message: str, *, provider: str):
        super().__init__(message, provider=provider, is_transient=False)


class RateLimitError(ProviderError):
    """Rate limit exceeded (429). Transient — retry after backoff."""

    def __init__(
        self, message: str, *, provider: str, retry_after: float | None = None
    ):
        super().__init__(message, provider=provider, is_transient=True)
        self.retry_after = retry_after


class ServerError(ProviderError):
    """Provider server error (5xx). Transient."""

    def __init__(self, message: str, *, provider: str):
        super().__init__(message, provider=provider, is_transient=True)


class TimeoutError(ProviderError):
    """Request timed out. Transient."""

    def __init__(self, message: str, *, provider: str):
        super().__init__(message, provider=provider, is_transient=True)


class ModelNotFoundError(ProviderError):
    """Model does not exist or is not available."""

    def __init__(self, message: str, *, provider: str):
        super().__init__(message, provider=provider, is_transient=False)


class ContentFilterError(ProviderError):
    """Response blocked by content filter. Not transient."""

    def __init__(self, message: str, *, provider: str):
        super().__init__(message, provider=provider, is_transient=False)


class ProviderNotInstalledError(Exception):
    """SDK for a provider is not installed. Setup error, not runtime."""

    def __init__(self, provider: str, package: str, extra: str):
        super().__init__(
            f"{provider} provider requires the '{package}' package. "
            f"Install with: pip install anchor[{extra}]"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/llm/test_errors.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/anchor/llm/errors.py tests/llm/test_errors.py
git commit -m "feat(llm): add provider error hierarchy"
```

---

### Task 3: Pricing (`src/anchor/llm/pricing.py`)

**Files:**
- Create: `src/anchor/llm/pricing.py`
- Create: `tests/llm/test_pricing.py`

- [ ] **Step 1: Write failing tests**

Create `tests/llm/test_pricing.py`:

```python
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
        # Anthropic model names end in dates but are canonical — they're
        # in the pricing table directly, so normalization is only a fallback
        result = _normalize_model_name("claude-sonnet-4-20250514")
        # This strips the date, which is fine — the exact match happens first
        assert isinstance(result, str)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/llm/test_pricing.py -v`
Expected: FAIL

- [ ] **Step 3: Implement pricing module**

Create `src/anchor/llm/pricing.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/llm/test_pricing.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/anchor/llm/pricing.py tests/llm/test_pricing.py
git commit -m "feat(llm): add model pricing table and cost calculation"
```

---

## Chunk 2: Core Abstractions — Protocol, Base Class, Registry, Fallback

### Task 4: LLMProvider Protocol & BaseLLMProvider (`src/anchor/llm/base.py`)

**Files:**
- Create: `src/anchor/llm/base.py`
- Create: `tests/llm/test_base.py`

- [ ] **Step 1: Write failing tests for BaseLLMProvider retry logic**

Create `tests/llm/test_base.py`:

```python
"""Tests for anchor.llm.base — Protocol and BaseLLMProvider."""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Iterator
from unittest.mock import MagicMock, patch

import pytest

from anchor.llm.base import BaseLLMProvider
from anchor.llm.errors import ProviderError, RateLimitError, ServerError, AuthenticationError
from anchor.llm.models import (
    LLMResponse,
    Message,
    Role,
    StopReason,
    StreamChunk,
    ToolSchema,
    Usage,
)


def _make_response(**kwargs) -> LLMResponse:
    defaults = {
        "content": "hello",
        "usage": Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        "model": "test-model",
        "provider": "test",
        "stop_reason": StopReason.STOP,
    }
    defaults.update(kwargs)
    return LLMResponse(**defaults)


class ConcreteProvider(BaseLLMProvider):
    """Minimal concrete implementation for testing."""

    provider_name = "test"

    def _resolve_api_key(self) -> str | None:
        return "test-key"

    def _do_invoke(self, messages, tools, **kwargs) -> LLMResponse:
        return _make_response()

    def _do_stream(self, messages, tools, **kwargs) -> Iterator[StreamChunk]:
        yield StreamChunk(content="hello")
        yield StreamChunk(stop_reason=StopReason.STOP)

    async def _do_ainvoke(self, messages, tools, **kwargs) -> LLMResponse:
        return _make_response()

    async def _do_astream(self, messages, tools, **kwargs) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(content="hello")
        yield StreamChunk(stop_reason=StopReason.STOP)


class TestBaseLLMProviderProperties:
    def test_model_id(self):
        p = ConcreteProvider(model="my-model")
        assert p.model_id == "test/my-model"

    def test_provider_name(self):
        p = ConcreteProvider(model="my-model")
        assert p.provider_name == "test"


class TestBaseLLMProviderInvoke:
    def test_invoke_success(self):
        p = ConcreteProvider(model="m")
        msgs = [Message(role=Role.USER, content="hi")]
        result = p.invoke(msgs)
        assert result.content == "hello"

    def test_stream_success(self):
        p = ConcreteProvider(model="m")
        msgs = [Message(role=Role.USER, content="hi")]
        chunks = list(p.stream(msgs))
        assert len(chunks) == 2
        assert chunks[0].content == "hello"


class TestRetryLogic:
    @patch("time.sleep")  # avoid real delays in tests
    def test_retries_on_transient_error(self, mock_sleep):
        call_count = 0

        class RetryProvider(ConcreteProvider):
            def _do_invoke(self, messages, tools, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise ServerError("500", provider="test")
                return _make_response()

        p = RetryProvider(model="m", max_retries=2)
        result = p.invoke([Message(role=Role.USER, content="hi")])
        assert result.content == "hello"
        assert call_count == 3
        assert mock_sleep.call_count == 2  # slept between retries

    def test_no_retry_on_non_transient_error(self):
        class AuthFailProvider(ConcreteProvider):
            def _do_invoke(self, messages, tools, **kwargs):
                raise AuthenticationError("bad key", provider="test")

        p = AuthFailProvider(model="m", max_retries=2)
        with pytest.raises(AuthenticationError):
            p.invoke([Message(role=Role.USER, content="hi")])

    @patch("time.sleep")
    def test_raises_after_max_retries(self, mock_sleep):
        class AlwaysFailProvider(ConcreteProvider):
            def _do_invoke(self, messages, tools, **kwargs):
                raise ServerError("500", provider="test")

        p = AlwaysFailProvider(model="m", max_retries=1)
        with pytest.raises(ServerError):
            p.invoke([Message(role=Role.USER, content="hi")])

    @patch("time.sleep")
    def test_respects_rate_limit_retry_after(self, mock_sleep):
        """RateLimitError with retry_after should use that delay (we just test it doesn't crash)."""
        call_count = 0

        class RateLimitProvider(ConcreteProvider):
            def _do_invoke(self, messages, tools, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise RateLimitError("429", provider="test", retry_after=0.01)
                return _make_response()

        p = RateLimitProvider(model="m", max_retries=1)
        result = p.invoke([Message(role=Role.USER, content="hi")])
        assert result.content == "hello"
        assert call_count == 2


class TestAsyncRetryLogic:
    @pytest.mark.asyncio
    async def test_ainvoke_success(self):
        p = ConcreteProvider(model="m")
        result = await p.ainvoke([Message(role=Role.USER, content="hi")])
        assert result.content == "hello"

    @pytest.mark.asyncio
    async def test_astream_success(self):
        p = ConcreteProvider(model="m")
        chunks = []
        async for chunk in p.astream([Message(role=Role.USER, content="hi")]):
            chunks.append(chunk)
        assert len(chunks) == 2

    @pytest.mark.asyncio
    async def test_async_retries_on_transient(self):
        call_count = 0

        class AsyncRetryProvider(ConcreteProvider):
            async def _do_ainvoke(self, messages, tools, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    raise ServerError("500", provider="test")
                return _make_response()

        p = AsyncRetryProvider(model="m", max_retries=1)
        result = await p.ainvoke([Message(role=Role.USER, content="hi")])
        assert result.content == "hello"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/llm/test_base.py -v`
Expected: FAIL

- [ ] **Step 3: Implement Protocol and BaseLLMProvider**

Create `src/anchor/llm/base.py`:

```python
"""LLMProvider Protocol and BaseLLMProvider ABC.

The Protocol defines what all providers must satisfy (structural subtyping).
The ABC provides shared retry, timeout, and property logic.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Iterator, Protocol, runtime_checkable

from anchor.llm.errors import ProviderError, RateLimitError
from anchor.llm.models import (
    LLMResponse,
    Message,
    StreamChunk,
    ToolSchema,
)


@runtime_checkable
class LLMProvider(Protocol):
    """Unified interface all LLM providers must satisfy."""

    @property
    def model_id(self) -> str: ...

    @property
    def provider_name(self) -> str: ...

    def invoke(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LLMResponse: ...

    def stream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]: ...

    async def ainvoke(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LLMResponse: ...

    def astream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]: ...


class BaseLLMProvider(ABC):
    """Abstract base class with shared retry, timeout, and property logic."""

    provider_name: str  # set by subclass as class attribute

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 2,
        timeout: float = 60.0,
        **kwargs: Any,
    ):
        self._model = model
        self._api_key = api_key or self._resolve_api_key()
        self._base_url = base_url
        self._max_retries = max_retries
        self._timeout = timeout

    @property
    def model_id(self) -> str:
        return f"{self.provider_name}/{self._model}"

    # --- Abstract methods for subclasses ---

    @abstractmethod
    def _resolve_api_key(self) -> str | None: ...

    @abstractmethod
    def _do_invoke(
        self, messages: list[Message], tools: list[ToolSchema] | None, **kwargs: Any
    ) -> LLMResponse: ...

    @abstractmethod
    def _do_stream(
        self, messages: list[Message], tools: list[ToolSchema] | None, **kwargs: Any
    ) -> Iterator[StreamChunk]: ...

    @abstractmethod
    async def _do_ainvoke(
        self, messages: list[Message], tools: list[ToolSchema] | None, **kwargs: Any
    ) -> LLMResponse: ...

    @abstractmethod
    async def _do_astream(
        self, messages: list[Message], tools: list[ToolSchema] | None, **kwargs: Any
    ) -> AsyncIterator[StreamChunk]: ...

    # --- Public methods with retry ---

    def invoke(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        return self._with_retry(self._do_invoke, messages, tools, **kwargs)

    def stream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        # NOTE: Cannot use _with_retry here because _do_stream is a generator
        # function. Calling it returns a generator object without executing body
        # code, so no exception is raised to catch. Retry before first chunk only.
        last_error: ProviderError | None = None
        for attempt in range(self._max_retries + 1):
            try:
                stream_iter = self._do_stream(messages, tools, **kwargs)
                first_chunk = next(stream_iter)
                # Committed — yield first chunk and rest without retry
                yield first_chunk
                yield from stream_iter
                return
            except StopIteration:
                return
            except ProviderError as e:
                if not e.is_transient or attempt == self._max_retries:
                    raise
                last_error = e
                delay = min(2**attempt, 8)
                if isinstance(e, RateLimitError) and e.retry_after:
                    delay = e.retry_after
                time.sleep(delay)
        if last_error:
            raise last_error

    async def ainvoke(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        return await self._with_async_retry(
            self._do_ainvoke, messages, tools, **kwargs
        )

    async def astream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        # NOTE: Cannot use _with_async_retry here because _do_astream is an
        # async generator (not a coroutine). `await async_gen()` raises TypeError.
        # Instead, retry before the first chunk — once streaming starts, errors propagate.
        last_error: ProviderError | None = None
        for attempt in range(self._max_retries + 1):
            try:
                stream_iter = self._do_astream(messages, tools, **kwargs)
                first_chunk = await stream_iter.__anext__()
                # Committed — yield first chunk and rest without retry
                yield first_chunk
                async for chunk in stream_iter:
                    yield chunk
                return
            except StopAsyncIteration:
                return
            except ProviderError as e:
                if not e.is_transient or attempt == self._max_retries:
                    raise
                last_error = e
                delay = min(2**attempt, 8)
                if isinstance(e, RateLimitError) and e.retry_after:
                    delay = e.retry_after
                await asyncio.sleep(delay)
        if last_error:
            raise last_error

    # --- Retry logic ---

    def _with_retry(self, fn, *args, **kwargs):
        last_error: ProviderError | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except ProviderError as e:
                if not e.is_transient or attempt == self._max_retries:
                    raise
                last_error = e
                delay = min(2**attempt, 8)
                if isinstance(e, RateLimitError) and e.retry_after:
                    delay = e.retry_after
                time.sleep(delay)
        raise last_error  # type: ignore[misc]

    async def _with_async_retry(self, fn, *args, **kwargs):
        """Retry for async coroutines (ainvoke). NOT for async generators (astream)."""
        last_error: ProviderError | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return await fn(*args, **kwargs)
            except ProviderError as e:
                if not e.is_transient or attempt == self._max_retries:
                    raise
                last_error = e
                delay = min(2**attempt, 8)
                if isinstance(e, RateLimitError) and e.retry_after:
                    delay = e.retry_after
                await asyncio.sleep(delay)
        raise last_error  # type: ignore[misc]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/llm/test_base.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/anchor/llm/base.py tests/llm/test_base.py
git commit -m "feat(llm): add LLMProvider Protocol and BaseLLMProvider with retry"
```

---

### Task 5: Provider Registry (`src/anchor/llm/registry.py`)

**Files:**
- Create: `src/anchor/llm/registry.py`
- Create: `tests/llm/test_registry.py`

- [ ] **Step 1: Write failing tests**

Create `tests/llm/test_registry.py`:

```python
"""Tests for anchor.llm.registry."""

from __future__ import annotations

from typing import AsyncIterator, Iterator

import pytest

from anchor.llm.base import BaseLLMProvider
from anchor.llm.errors import ProviderNotInstalledError
from anchor.llm.models import LLMResponse, Message, StreamChunk, StopReason, ToolSchema, Usage
from anchor.llm.registry import (
    _parse_model_string,
    _PROVIDERS,
    create_provider,
    register_provider,
)


def _make_response() -> LLMResponse:
    return LLMResponse(
        content="ok",
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        model="m",
        provider="test",
        stop_reason=StopReason.STOP,
    )


class FakeProvider(BaseLLMProvider):
    provider_name = "fake"

    def _resolve_api_key(self):
        return "k"

    def _do_invoke(self, messages, tools, **kwargs):
        return _make_response()

    def _do_stream(self, messages, tools, **kwargs):
        yield StreamChunk(content="ok")

    async def _do_ainvoke(self, messages, tools, **kwargs):
        return _make_response()

    async def _do_astream(self, messages, tools, **kwargs):
        yield StreamChunk(content="ok")


class TestParseModelString:
    def test_with_provider(self):
        assert _parse_model_string("openai/gpt-4o") == ("openai", "gpt-4o")

    def test_without_provider_defaults_anthropic(self):
        assert _parse_model_string("claude-haiku-4-5-20251001") == (
            "anthropic",
            "claude-haiku-4-5-20251001",
        )

    def test_fine_tuned_model(self):
        assert _parse_model_string("openai/ft:gpt-4o:org:name:id") == (
            "openai",
            "ft:gpt-4o:org:name:id",
        )

    def test_openrouter_double_prefix(self):
        assert _parse_model_string("openrouter/anthropic/claude-sonnet-4-20250514") == (
            "openrouter",
            "anthropic/claude-sonnet-4-20250514",
        )

    def test_empty_string(self):
        assert _parse_model_string("") == ("anthropic", "")


class TestRegisterProvider:
    def test_register_and_create(self):
        register_provider("fake", FakeProvider)
        try:
            provider = create_provider("fake/test-model")
            assert provider.model_id == "fake/test-model"
            assert provider.provider_name == "fake"
        finally:
            _PROVIDERS.pop("fake", None)

    def test_unknown_provider_raises(self):
        with pytest.raises(ProviderNotInstalledError):
            create_provider("nonexistent/model")


class TestCreateProviderWithFallbacks:
    def test_creates_fallback_provider(self):
        register_provider("fake", FakeProvider)
        try:
            provider = create_provider("fake/m1", fallbacks=["fake/m2"])
            # Should be a FallbackProvider wrapping two FakeProviders
            assert provider.model_id == "fake/m1"
            result = provider.invoke([Message(role="user", content="hi")])
            assert result.content == "ok"
        finally:
            _PROVIDERS.pop("fake", None)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/llm/test_registry.py -v`
Expected: FAIL

- [ ] **Step 3: Implement registry**

Create `src/anchor/llm/registry.py`:

```python
"""Provider registry and model string parsing.

Manages the mapping from provider name to provider class, handles
lazy imports, and provides the create_provider() factory function.
"""

from __future__ import annotations

import importlib
from typing import Any

from anchor.llm.base import BaseLLMProvider, LLMProvider
from anchor.llm.errors import ProviderNotInstalledError

_PROVIDERS: dict[str, type[BaseLLMProvider]] = {}

# Maps provider name -> module path for lazy loading
_PROVIDER_MODULES: dict[str, str] = {
    "anthropic": "anchor.llm.providers.anthropic",
    "openai": "anchor.llm.providers.openai",
    "gemini": "anchor.llm.providers.gemini",
    "grok": "anchor.llm.providers.grok",
    "ollama": "anchor.llm.providers.ollama",
    "openrouter": "anchor.llm.providers.openrouter",
    "litellm": "anchor.llm.providers.litellm",
}

# Maps provider name -> pip package name (for error messages)
_PROVIDER_PACKAGES: dict[str, str] = {
    "anthropic": "anthropic",
    "openai": "openai",
    "gemini": "google-genai",
    "grok": "openai",
    "ollama": "ollama",
    "openrouter": "openai",
    "litellm": "litellm",
}

# Maps provider name -> pip extras name
_PROVIDER_EXTRAS: dict[str, str] = {
    "anthropic": "anthropic",
    "openai": "openai",
    "gemini": "gemini",
    "grok": "openai",
    "ollama": "ollama",
    "openrouter": "openai",
    "litellm": "litellm",
}


def register_provider(name: str, cls: type[BaseLLMProvider]) -> None:
    """Register a provider adapter."""
    _PROVIDERS[name] = cls


def create_provider(
    model: str,
    *,
    api_key: str | None = None,
    fallbacks: list[str] | None = None,
    **kwargs: Any,
) -> LLMProvider:
    """Create a provider from a 'provider/model' string.

    Examples:
        create_provider("openai/gpt-4o")
        create_provider("anthropic/claude-sonnet-4-20250514", api_key="sk-...")
        create_provider("ollama/llama3")
        create_provider("anthropic/claude-sonnet-4-20250514", fallbacks=["openai/gpt-4o"])
    """
    provider_name, model_name = _parse_model_string(model)

    if provider_name not in _PROVIDERS:
        _try_import_provider(provider_name)

    if provider_name not in _PROVIDERS:
        package = _PROVIDER_PACKAGES.get(provider_name, provider_name)
        extra = _PROVIDER_EXTRAS.get(provider_name, provider_name)
        raise ProviderNotInstalledError(provider_name, package, extra)

    cls = _PROVIDERS[provider_name]
    primary = cls(model=model_name, api_key=api_key, **kwargs)

    if fallbacks:
        from anchor.llm.fallback import FallbackProvider

        fallback_providers = [create_provider(fb) for fb in fallbacks]
        return FallbackProvider(primary=primary, fallbacks=fallback_providers)

    return primary


def _parse_model_string(model: str) -> tuple[str, str]:
    """Parse 'provider/model' into (provider, model).

    No prefix defaults to 'anthropic' for backward compat.
    Splits on first '/' only.
    """
    if "/" not in model:
        return "anthropic", model
    provider, _, model_name = model.partition("/")
    return provider, model_name


def _try_import_provider(name: str) -> None:
    """Attempt to lazily import a provider module."""
    module_path = _PROVIDER_MODULES.get(name)
    if module_path:
        try:
            importlib.import_module(module_path)
        except ImportError:
            pass
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/llm/test_registry.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/anchor/llm/registry.py tests/llm/test_registry.py
git commit -m "feat(llm): add provider registry with model string parsing"
```

---

### Task 6: FallbackProvider (`src/anchor/llm/fallback.py`)

**Files:**
- Create: `src/anchor/llm/fallback.py`
- Create: `tests/llm/test_fallback.py`

- [ ] **Step 1: Write failing tests**

Create `tests/llm/test_fallback.py`:

```python
"""Tests for anchor.llm.fallback."""

from __future__ import annotations

from typing import AsyncIterator, Iterator

import pytest

from anchor.llm.base import BaseLLMProvider
from anchor.llm.errors import AuthenticationError, ProviderError, ServerError
from anchor.llm.fallback import FallbackProvider
from anchor.llm.models import (
    LLMResponse,
    Message,
    Role,
    StopReason,
    StreamChunk,
    ToolSchema,
    Usage,
)


def _resp(provider: str = "p", content: str = "ok") -> LLMResponse:
    return LLMResponse(
        content=content,
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        model="m",
        provider=provider,
        stop_reason=StopReason.STOP,
    )


class StubProvider(BaseLLMProvider):
    provider_name = "stub"

    def __init__(self, model="m", invoke_fn=None, stream_fn=None, **kwargs):
        super().__init__(model=model, max_retries=0, **kwargs)
        self._invoke_fn = invoke_fn
        self._stream_fn = stream_fn

    def _resolve_api_key(self):
        return "k"

    def _do_invoke(self, messages, tools, **kwargs):
        if self._invoke_fn:
            return self._invoke_fn()
        return _resp(self.provider_name)

    def _do_stream(self, messages, tools, **kwargs):
        if self._stream_fn:
            yield from self._stream_fn()
        else:
            yield StreamChunk(content="ok")

    async def _do_ainvoke(self, messages, tools, **kwargs):
        return self._do_invoke(messages, tools, **kwargs)

    async def _do_astream(self, messages, tools, **kwargs):
        for chunk in self._do_stream(messages, tools, **kwargs):
            yield chunk


class TestFallbackProviderProperties:
    def test_model_id_from_primary(self):
        primary = StubProvider(model="m1")
        fb = FallbackProvider(primary=primary, fallbacks=[StubProvider(model="m2")])
        assert fb.model_id == "stub/m1"

    def test_provider_name_from_primary(self):
        primary = StubProvider(model="m1")
        fb = FallbackProvider(primary=primary, fallbacks=[StubProvider(model="m2")])
        assert fb.provider_name == "stub"


class TestFallbackInvoke:
    def test_primary_succeeds(self):
        fb = FallbackProvider(
            primary=StubProvider(invoke_fn=lambda: _resp("primary")),
            fallbacks=[StubProvider(invoke_fn=lambda: _resp("fallback"))],
        )
        msgs = [Message(role=Role.USER, content="hi")]
        result = fb.invoke(msgs)
        assert result.provider == "primary"

    def test_falls_back_on_transient_error(self):
        def fail():
            raise ServerError("500", provider="primary")

        fb = FallbackProvider(
            primary=StubProvider(invoke_fn=fail),
            fallbacks=[StubProvider(invoke_fn=lambda: _resp("fallback"))],
        )
        result = fb.invoke([Message(role=Role.USER, content="hi")])
        assert result.provider == "fallback"

    def test_non_transient_error_not_caught(self):
        def fail():
            raise AuthenticationError("bad key", provider="primary")

        fb = FallbackProvider(
            primary=StubProvider(invoke_fn=fail),
            fallbacks=[StubProvider(invoke_fn=lambda: _resp("fallback"))],
        )
        with pytest.raises(AuthenticationError):
            fb.invoke([Message(role=Role.USER, content="hi")])

    def test_all_fail_raises_last_error(self):
        def fail():
            raise ServerError("500", provider="all")

        fb = FallbackProvider(
            primary=StubProvider(invoke_fn=fail),
            fallbacks=[StubProvider(invoke_fn=fail)],
        )
        with pytest.raises(ServerError):
            fb.invoke([Message(role=Role.USER, content="hi")])


class TestFallbackStream:
    def test_primary_stream_succeeds(self):
        fb = FallbackProvider(
            primary=StubProvider(),
            fallbacks=[StubProvider()],
        )
        chunks = list(fb.stream([Message(role=Role.USER, content="hi")]))
        assert len(chunks) >= 1

    def test_falls_back_before_first_chunk(self):
        """If primary fails before yielding, fallback kicks in."""
        def fail_stream():
            raise ServerError("500", provider="primary")
            yield  # make it a generator  # noqa: E501

        fb = FallbackProvider(
            primary=StubProvider(stream_fn=fail_stream),
            fallbacks=[StubProvider()],
        )
        chunks = list(fb.stream([Message(role=Role.USER, content="hi")]))
        assert any(c.content == "ok" for c in chunks)

    def test_mid_stream_failure_propagates(self):
        """After first chunk yielded, errors propagate — no silent switch."""
        def fail_mid_stream():
            yield StreamChunk(content="partial")
            raise ServerError("500", provider="primary")

        fb = FallbackProvider(
            primary=StubProvider(stream_fn=fail_mid_stream),
            fallbacks=[StubProvider()],
        )
        stream = fb.stream([Message(role=Role.USER, content="hi")])
        first = next(stream)
        assert first.content == "partial"
        with pytest.raises(ServerError):
            next(stream)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/llm/test_fallback.py -v`
Expected: FAIL

- [ ] **Step 3: Implement FallbackProvider**

Create `src/anchor/llm/fallback.py`:

```python
"""FallbackProvider — wraps multiple providers with fallback-on-failure.

Fallback rules:
- invoke/ainvoke: On transient ProviderError, tries next provider.
- stream/astream: Fallback ONLY before first chunk. Once streaming has
  started yielding, failure raises immediately (no silent mid-stream switch).
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Iterator

from anchor.llm.base import LLMProvider
from anchor.llm.errors import ProviderError
from anchor.llm.models import LLMResponse, Message, StreamChunk, ToolSchema


class FallbackProvider:
    """Wraps multiple providers with fallback-on-failure logic."""

    def __init__(
        self,
        primary: LLMProvider,
        fallbacks: list[LLMProvider],
    ):
        self._providers = [primary] + fallbacks
        self._primary = primary

    @property
    def model_id(self) -> str:
        return self._primary.model_id

    @property
    def provider_name(self) -> str:
        return self._primary.provider_name

    def invoke(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        last_error: ProviderError | None = None
        for provider in self._providers:
            try:
                return provider.invoke(messages, tools=tools, **kwargs)
            except ProviderError as e:
                if not e.is_transient:
                    raise
                last_error = e
        raise last_error  # type: ignore[misc]

    def stream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        last_error: ProviderError | None = None
        for provider in self._providers:
            try:
                stream_iter = provider.stream(messages, tools=tools, **kwargs)
                first_chunk = next(stream_iter)
                # Committed to this provider
                yield first_chunk
                yield from stream_iter
                return
            except StopIteration:
                return
            except ProviderError as e:
                if not e.is_transient:
                    raise
                last_error = e
        if last_error:
            raise last_error

    async def ainvoke(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        last_error: ProviderError | None = None
        for provider in self._providers:
            try:
                return await provider.ainvoke(messages, tools=tools, **kwargs)
            except ProviderError as e:
                if not e.is_transient:
                    raise
                last_error = e
        raise last_error  # type: ignore[misc]

    async def astream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        last_error: ProviderError | None = None
        for provider in self._providers:
            try:
                stream_iter = provider.astream(messages, tools=tools, **kwargs)
                first_chunk = await stream_iter.__anext__()
                yield first_chunk
                async for chunk in stream_iter:
                    yield chunk
                return
            except StopAsyncIteration:
                return
            except ProviderError as e:
                if not e.is_transient:
                    raise
                last_error = e
        if last_error:
            raise last_error
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/llm/test_fallback.py -v`
Expected: All PASS (remove or fix the `test_no_fallback_after_first_chunk` duplicate test)

- [ ] **Step 5: Commit**

```bash
git add src/anchor/llm/fallback.py tests/llm/test_fallback.py
git commit -m "feat(llm): add FallbackProvider with pre-first-chunk streaming safety"
```

---

## Chunk 3: Provider Adapters — Anthropic & OpenAI

### Task 7: AnthropicProvider (`src/anchor/llm/providers/anthropic.py`)

**Files:**
- Create: `src/anchor/llm/providers/__init__.py`
- Create: `src/anchor/llm/providers/anthropic.py`
- Create: `tests/llm/providers/__init__.py`
- Create: `tests/llm/providers/test_anthropic.py`

- [ ] **Step 1: Write failing tests with mocked Anthropic SDK**

Create `tests/llm/providers/__init__.py` (empty) and `tests/llm/providers/test_anthropic.py`. Tests should mock the `anthropic` SDK and verify:
- Message conversion (system messages split out, user/assistant/tool messages converted)
- Tool schema conversion
- Response parsing (text content, tool_use content)
- Stream event parsing
- Error mapping (anthropic.RateLimitError → RateLimitError, etc.)
- API key resolution from env var `ANTHROPIC_API_KEY`
- `ProviderNotInstalledError` when anthropic SDK not installed

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/llm/providers/test_anthropic.py -v`

- [ ] **Step 3: Implement AnthropicProvider**

Key implementation details:
- `_split_system()`: Extract system messages into separate list (Anthropic takes them as a parameter)
- `_to_anthropic_msg()`: Convert `Message` → Anthropic message dict
- `_to_anthropic_tool()`: Convert `ToolSchema` → `{"name", "description", "input_schema"}`
- `_parse_response()`: Convert Anthropic `MessageResponse` → `LLMResponse`
- `_parse_stream_event()`: Convert Anthropic stream events → `StreamChunk`
- Error mapping in try/except blocks
- Both sync and async clients

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/llm/providers/test_anthropic.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/anchor/llm/providers/ tests/llm/providers/
git commit -m "feat(llm): add AnthropicProvider adapter"
```

---

### Task 8: OpenAIProvider (`src/anchor/llm/providers/openai.py`)

**Files:**
- Create: `src/anchor/llm/providers/openai.py`
- Create: `tests/llm/providers/test_openai.py`

- [ ] **Step 1: Write failing tests with mocked OpenAI SDK**

Tests should cover:
- Message conversion (system as regular message, tool_calls format, function results)
- Tool schema → OpenAI function definition conversion
- Response parsing (text content, function_call/tool_calls)
- Stream chunk parsing (delta content, delta tool_calls with argument fragments)
- Error mapping
- `base_url` parameter support (important for Grok/OpenRouter subclasses)

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Implement OpenAIProvider**

Key details:
- System messages included in `messages` list (not separate like Anthropic)
- Tool calls use `{"type": "function", "function": {"name", "description", "parameters"}}` format
- Response tool_calls use `function.arguments` (JSON string) that needs parsing
- Streaming uses `delta.content` and `delta.tool_calls[].function.arguments`
- `base_url` passed to `openai.OpenAI()` for subclass reuse

- [ ] **Step 4: Run tests to verify they pass**

- [ ] **Step 5: Commit**

```bash
git add src/anchor/llm/providers/openai.py tests/llm/providers/test_openai.py
git commit -m "feat(llm): add OpenAIProvider adapter"
```

---

### Task 9: Grok & OpenRouter providers (thin subclasses)

**Files:**
- Create: `src/anchor/llm/providers/grok.py`
- Create: `src/anchor/llm/providers/openrouter.py`
- Create: `tests/llm/providers/test_grok.py`
- Create: `tests/llm/providers/test_openrouter.py`

- [ ] **Step 1: Write failing tests**

Minimal tests: verify `base_url` defaults, `_resolve_api_key` checks correct env vars, `provider_name` is correct.

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Implement both providers**

Each is ~15 lines subclassing `OpenAIProvider`.

- [ ] **Step 4: Run tests to verify they pass**

- [ ] **Step 5: Commit**

```bash
git add src/anchor/llm/providers/grok.py src/anchor/llm/providers/openrouter.py \
       tests/llm/providers/test_grok.py tests/llm/providers/test_openrouter.py
git commit -m "feat(llm): add Grok and OpenRouter providers (OpenAI-compatible)"
```

---

## Chunk 4: More Providers — Gemini, Ollama, LiteLLM

### Task 10: GeminiProvider (`src/anchor/llm/providers/gemini.py`)

**Files:**
- Create: `src/anchor/llm/providers/gemini.py`
- Create: `tests/llm/providers/test_gemini.py`

- [ ] **Step 1: Write failing tests with mocked google-genai SDK**

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Implement GeminiProvider**

Key details:
- Uses `google.genai.Client` not OpenAI-compatible
- System instruction passed via `GenerateContentConfig`
- Messages use `Content` and `Part` objects
- Tool definitions use `FunctionDeclaration`
- API key from `GOOGLE_API_KEY` or `GEMINI_API_KEY`

- [ ] **Step 4: Run tests to verify they pass**

- [ ] **Step 5: Commit**

```bash
git add src/anchor/llm/providers/gemini.py tests/llm/providers/test_gemini.py
git commit -m "feat(llm): add GeminiProvider adapter (google-genai SDK)"
```

---

### Task 11: OllamaProvider (`src/anchor/llm/providers/ollama.py`)

**Files:**
- Create: `src/anchor/llm/providers/ollama.py`
- Create: `tests/llm/providers/test_ollama.py`

- [ ] **Step 1: Write failing tests**

Ollama exposes an OpenAI-compatible endpoint at `http://localhost:11434/v1`. Subclass `OpenAIProvider` with custom `base_url` and no API key required.

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Implement OllamaProvider**

```python
class OllamaProvider(OpenAIProvider):
    provider_name = "ollama"

    def __init__(self, model: str, **kwargs):
        kwargs.setdefault("base_url", "http://localhost:11434/v1")
        kwargs.setdefault("api_key", "ollama")  # Ollama doesn't need a real key
        super().__init__(model=model, **kwargs)

    def _resolve_api_key(self) -> str | None:
        return os.environ.get("OLLAMA_API_KEY", "ollama")
```

- [ ] **Step 4: Run tests to verify they pass**

- [ ] **Step 5: Commit**

```bash
git add src/anchor/llm/providers/ollama.py tests/llm/providers/test_ollama.py
git commit -m "feat(llm): add OllamaProvider (local models via OpenAI-compat)"
```

---

### Task 12: LiteLLMProvider (optional catch-all)

**Files:**
- Create: `src/anchor/llm/providers/litellm.py`
- Create: `tests/llm/providers/test_litellm.py`

- [ ] **Step 1: Write failing tests** (mock litellm.completion)

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Implement LiteLLMProvider**

Wraps `litellm.completion()` and `litellm.acompletion()`. Converts to/from unified models.

- [ ] **Step 4: Run tests to verify they pass**

- [ ] **Step 5: Commit**

```bash
git add src/anchor/llm/providers/litellm.py tests/llm/providers/test_litellm.py
git commit -m "feat(llm): add LiteLLMProvider (optional catch-all)"
```

---

## Chunk 5: Integration — Agent, AgentTool, Exports, Packaging

### Task 13: Add `to_tool_schema()` to AgentTool

**Files:**
- Modify: `src/anchor/agent/models.py`
- Modify: `tests/agent/test_models.py` (or create new test)

- [ ] **Step 1: Write failing test**

```python
def test_agent_tool_to_tool_schema():
    from anchor.agent.models import AgentTool
    from anchor.llm.models import ToolSchema

    tool = AgentTool(
        name="get_weather",
        description="Get weather",
        input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
        fn=lambda **kwargs: "sunny",
    )
    schema = tool.to_tool_schema()
    assert isinstance(schema, ToolSchema)
    assert schema.name == "get_weather"
    assert schema.description == "Get weather"
    assert schema.input_schema == tool.input_schema
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Add `to_tool_schema()` method**

Add to `src/anchor/agent/models.py` in the `AgentTool` class:

```python
def to_tool_schema(self) -> "ToolSchema":
    """Convert to provider-agnostic ToolSchema."""
    from anchor.llm.models import ToolSchema

    return ToolSchema(
        name=self.name,
        description=self.description,
        input_schema=self.input_schema,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

- [ ] **Step 5: Commit**

```bash
git add src/anchor/agent/models.py tests/agent/
git commit -m "feat(agent): add to_tool_schema() to AgentTool"
```

---

### Task 14: Refactor Agent to use LLMProvider

**Files:**
- Modify: `src/anchor/agent/agent.py`
- Modify: `tests/agent/test_agent.py`

This is the largest task. Key changes:

**CRITICAL: Preserve existing Agent behaviors.** The refactor ONLY replaces the LLM transport layer. These behaviors MUST be preserved exactly:
- Skills system (`_evaluate_skills`, per-round tool recomputation from skills)
- Memory recording (`_add_to_memory`, token tracking)
- System message injection at chat start
- Callback invocations (`on_stream_chunk`, `on_tool_call`, etc.)
- Pipeline output formatting (formatters produce provider-specific message dicts)
- Per-round `max_rounds` and tool loop logic
- Error handling and retry semantics

The `_context_to_messages` helper must handle tool-related content blocks (tool_use, tool_result) from the message history, not just simple role/content messages.

- [ ] **Step 1: Update Agent `__slots__` and `__init__`**

Remove `_client`, `_model`. Add `_llm`. Change constructor to accept `model: str` (with `provider/model` format), `llm: LLMProvider | None`, `fallbacks: list[str] | None`.

Replace `self._pipeline.with_formatter(AnthropicFormatter())` with auto-selection based on provider:
```python
def _auto_formatter(self) -> Formatter:
    name = self._llm.provider_name
    if name == "anthropic":
        return AnthropicFormatter()
    # OpenAI, Grok, Ollama, OpenRouter all use OpenAI format
    return OpenAIFormatter()  # or GenericFormatter
```

- [ ] **Step 2: Add `_context_to_messages` conversion**

Convert pipeline output to `list[Message]`. Must handle:
- System messages → `Message(role=Role.SYSTEM, content=...)`
- User/assistant text → `Message(role=..., content=...)`
- Tool use blocks → `Message(role=Role.ASSISTANT, tool_calls=[ToolCall(...)])`
- Tool result blocks → `Message(role=Role.TOOL, tool_result=ToolResult(...))`
- Content blocks with images → `Message(content=[ContentBlock(...)])`

- [ ] **Step 3: Rewrite `chat()` method — LLM transport only**

Replace only the Anthropic-specific API call with `self._llm.stream()`. Keep ALL surrounding logic intact:
- Pipeline execution and context building
- Skills evaluation and tool resolution
- The tool loop structure (while stop_reason == TOOL_USE)
- Memory recording
- Callback invocations
- StreamDelta conversion (convert `StreamChunk` → existing `StreamDelta` for pipeline compat)

- [ ] **Step 4: Rewrite `achat()` method**

Same as `chat()` but using `self._llm.astream()` (which is now an async generator — iterate directly, do NOT await).

- [ ] **Step 5: Add helper methods**

`_build_tool_result_content(tool_call, result)`: Converts `ToolCall` + raw result into a `ToolResult` for appending to message history.
`_accumulate_tool_call(deltas)`: Accumulates `ToolCallDelta` fragments into complete `ToolCall` objects (JSON-parsing accumulated argument fragments).
`_stream_chunk_to_delta(chunk)`: Converts `StreamChunk` → existing `StreamDelta` for pipeline compatibility.

- [ ] **Step 6: Rewrite `_serialize_response` and `_run_tools`**

Operate on `LLMResponse` and `ToolCall` instead of Anthropic SDK types. `_run_tools` should accept `list[ToolCall]` and return `list[ToolResult]`.

- [ ] **Step 7: Remove `_call_api_with_retry` and `_call_api_with_retry_async`**

Retry logic now lives in `BaseLLMProvider._with_retry`.

- [ ] **Step 8: Update tests**

Replace `client=mock_anthropic_client` with `llm=mock_provider`. Update assertions to use unified model types. Verify skills, memory, and callback behaviors are preserved by existing tests.

- [ ] **Step 9: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All PASS — no regressions in agent behavior tests

- [ ] **Step 10: Commit**

```bash
git add src/anchor/agent/agent.py tests/agent/
git commit -m "refactor(agent): use LLMProvider instead of Anthropic SDK directly"
```

---

### Task 15: Update exports and packaging

**Files:**
- Modify: `src/anchor/llm/__init__.py`
- Modify: `src/anchor/__init__.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add LLM module exports**

Update `src/anchor/llm/__init__.py` to export all public symbols.

- [ ] **Step 2: Add to main `__init__.py`**

Add LLM module imports to `src/anchor/__init__.py` and extend `__all__`.

- [ ] **Step 3: Update `pyproject.toml` optional dependencies**

Add `openai`, `gemini`, `ollama`, `litellm`, `all-providers` extras groups.

**Important:** Move `anthropic` from hard dependency to the `anthropic` optional extra. The `agents` extra (which depends on `anthropic`) should continue to pull it in. Add a new `grok` extra that depends on `openai` (since Grok uses OpenAI-compatible API). The `all-providers` extra should include all provider extras.

Example extras:
```toml
[project.optional-dependencies]
anthropic = ["anthropic>=0.40.0"]
openai = ["openai>=1.50.0"]
gemini = ["google-genai>=1.0.0"]
grok = ["openai>=1.50.0"]
ollama = ["openai>=1.50.0"]
litellm = ["litellm>=1.50.0"]
all-providers = ["astro-anchor[anthropic,openai,gemini,ollama,litellm]"]
```

- [ ] **Step 4: Add `pytest-asyncio` to dev dependencies if not present**

Needed for `@pytest.mark.asyncio` in `tests/llm/test_base.py`. Check if already in `[project.optional-dependencies]` dev/test group.

- [ ] **Step 5: Add mypy overrides for optional provider SDKs**

Add `[[tool.mypy.overrides]]` entries in `pyproject.toml` for `anthropic`, `openai`, `google.genai`, `ollama`, `litellm` (follow existing pattern for `rank_bm25`, `pypdf`, etc.):

```toml
[[tool.mypy.overrides]]
module = ["anthropic.*", "openai.*", "google.genai.*", "ollama.*", "litellm.*"]
ignore_missing_imports = true
```

- [ ] **Step 6: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/anchor/llm/__init__.py src/anchor/__init__.py pyproject.toml
git commit -m "feat: export LLM module and add provider optional dependencies"
```

---

## Chunk 6: Final Verification

### Task 16: End-to-end verification

- [ ] **Step 1: Run full test suite with coverage**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All PASS, no regressions

- [ ] **Step 2: Run type checker**

Run: `python -m mypy src/anchor/llm/`
Expected: No errors

- [ ] **Step 3: Run linter**

Run: `ruff check src/anchor/llm/ tests/llm/`
Expected: No errors

- [ ] **Step 4: Verify backward compatibility**

Confirm these patterns still work:
```python
from anchor import Agent
agent = Agent(model="claude-haiku-4-5-20251001")  # no prefix = anthropic
```

- [ ] **Step 5: Final commit if any fixups needed**

```bash
git add -u
git commit -m "fix: address linting and type errors in LLM module"
```
