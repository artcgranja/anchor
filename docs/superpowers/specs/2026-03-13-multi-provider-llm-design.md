# Multi-Provider LLM Integration Layer

**Date:** 2026-03-13
**Status:** Draft
**Author:** Claude + Arthur

## Summary

Add a centralized, provider-agnostic LLM layer to Anchor so users can use any LLM provider (Anthropic, OpenAI, Gemini, Grok, Ollama, OpenRouter) through a single interface. Follows the same architecture pattern used by LangChain, LlamaIndex, and Agno: Protocol-based abstraction with direct SDK adapters per provider.

## Motivation

Anchor's `Agent` class is currently hardcoded to Anthropic. The `ContextPipeline` is already provider-agnostic (formatters for Anthropic, OpenAI, Generic), but the Agent — the primary user-facing API — only works with one provider. Users need to:

- Use the model of their choice (cost, performance, privacy tradeoffs)
- Switch providers without rewriting application code
- Use local models (Ollama) for development and testing
- Set up fallback chains for reliability
- Track costs across providers

## Design Decisions

### Why direct SDK adapters (not LiteLLM as core)

Research across the top frameworks:

| Framework | Approach | Uses LiteLLM as core? |
|-----------|----------|----------------------|
| LangChain | Direct SDK adapters per provider | No (separate packages) |
| LlamaIndex | Direct SDK adapters per provider | No (103 separate packages) |
| Agno | Direct SDK adapters per provider | No (41 provider dirs) |
| CrewAI | Native adapters + LiteLLM fallback | Hybrid |
| Open Interpreter | LiteLLM only | Yes |

LiteLLM is ~30MB with heavy transitive deps (boto3, tokenizers). The major frameworks all use thin direct SDK wrappers. LiteLLM is offered as one optional adapter, not the core.

### Model identifier format

Uses the `provider/model` string convention: `"openai/gpt-4o"`, `"anthropic/claude-sonnet-4-20250514"`, `"ollama/llama3"`.

This is the 2026 standard used by LiteLLM, OpenRouter, aisuite, and any-llm. Auto-detects provider from prefix.

**Edge cases:**
- No prefix: defaults to `anthropic/` for backward compat (`"claude-haiku-4-5-20251001"` → `("anthropic", "claude-haiku-4-5-20251001")`)
- Fine-tuned models: `"openai/ft:gpt-4o:my-org:custom:id"` → `("openai", "ft:gpt-4o:my-org:custom:id")` — splits on first `/` only
- OpenRouter double-prefix: `"openrouter/anthropic/claude-sonnet-4-20250514"` → `("openrouter", "anthropic/claude-sonnet-4-20250514")` — correct, OpenRouter expects the full `provider/model` as the model name

### Backward compatibility

If the model string has no `/` prefix, it defaults to `anthropic/` to preserve existing behavior. `Agent(model="claude-haiku-4-5-20251001")` continues to work.

## Architecture

### Module structure

```
src/anchor/llm/
├── __init__.py              # Public exports: create_provider, LLMProvider, etc.
├── base.py                  # LLMProvider Protocol, BaseLLMProvider ABC
├── models.py                # Message, LLMResponse, StreamChunk, ToolSchema, Usage
├── registry.py              # Provider registry, model string parsing, create_provider()
├── fallback.py              # FallbackProvider (wraps multiple providers)
├── errors.py                # ProviderError, ModelNotFoundError, etc.
├── pricing.py               # Built-in model pricing table, cost calculation
├── providers/
│   ├── __init__.py
│   ├── anthropic.py         # AnthropicProvider
│   ├── openai.py            # OpenAIProvider
│   ├── gemini.py            # GeminiProvider
│   ├── grok.py              # GrokProvider (OpenAI-compatible)
│   ├── ollama.py            # OllamaProvider (OpenAI-compatible or native)
│   ├── openrouter.py        # OpenRouterProvider (OpenAI-compatible)
│   └── litellm.py           # LiteLLMProvider (optional catch-all)
```

### Core Protocol

```python
from __future__ import annotations
from typing import Protocol, Iterator, AsyncIterator

class LLMProvider(Protocol):
    """Unified interface all LLM providers must satisfy."""

    @property
    def model_id(self) -> str:
        """Full model identifier, e.g. 'openai/gpt-4o'."""
        ...

    @property
    def provider_name(self) -> str:
        """Provider name, e.g. 'openai'."""
        ...

    def invoke(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: list[str] | None = None,
        **kwargs,
    ) -> LLMResponse:
        ...

    def stream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: list[str] | None = None,
        **kwargs,
    ) -> Iterator[StreamChunk]:
        ...

    async def ainvoke(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: list[str] | None = None,
        **kwargs,
    ) -> LLMResponse:
        ...

    async def astream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: list[str] | None = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        ...
```

### Unified message models

```python
from pydantic import BaseModel
from enum import Enum
from typing import Any

class Role(str, Enum):
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
    # For image content
    image_url: str | None = None
    image_base64: str | None = None
    media_type: str | None = None  # e.g. "image/png"

class ToolCall(BaseModel, frozen=True):
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
    tool_call_id: str
    content: str
    is_error: bool = False

class Message(BaseModel, frozen=True):
    role: Role
    content: str | list[ContentBlock] | None = None
    tool_calls: list[ToolCall] | None = None
    tool_result: ToolResult | None = None
    name: str | None = None  # for tool messages

class Usage(BaseModel, frozen=True):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    total_cost: float | None = None  # USD, None if pricing unknown

class StopReason(str, Enum):
    STOP = "stop"              # natural stop
    MAX_TOKENS = "max_tokens"  # hit token limit
    TOOL_USE = "tool_use"      # wants to call a tool

class LLMResponse(BaseModel, frozen=True):
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    usage: Usage
    model: str                 # actual model used (as returned by provider)
    provider: str              # actual provider used
    stop_reason: StopReason

class StreamChunk(BaseModel, frozen=True):
    content: str | None = None
    tool_call_delta: ToolCallDelta | None = None
    usage: Usage | None = None        # present on final chunk
    stop_reason: StopReason | None = None  # present on final chunk
```

**Relationship to existing `StreamResult`/`StreamUsage`/`StreamDelta` models** (`src/anchor/models/streaming.py`): These existing models are pipeline-level streaming abstractions. The new `StreamChunk` and `Usage` are LLM-layer models — they represent what comes back from a provider API call. The existing models remain untouched; the Agent will internally convert `StreamChunk` → `StreamDelta` where needed for pipeline compatibility.

### ToolSchema

```python
class ToolSchema(BaseModel, frozen=True):
    name: str
    description: str
    input_schema: dict[str, Any]  # JSON Schema
```

**Conversion from AgentTool**: `AgentTool` gains a new method `to_tool_schema() -> ToolSchema` that extracts the provider-agnostic schema. This replaces the current pattern of calling `to_anthropic_schema()` directly in the Agent's chat loop. Each provider adapter then converts `ToolSchema` to its native format internally.

```python
# In AgentTool (agent/models.py)
def to_tool_schema(self) -> ToolSchema:
    """Convert to provider-agnostic ToolSchema."""
    return ToolSchema(
        name=self.name,
        description=self.description,
        input_schema=self.input_schema,
    )
```

### Error hierarchy

```python
# src/anchor/llm/errors.py

class ProviderError(Exception):
    """Base error for all provider failures."""
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
    def __init__(self, message: str, *, provider: str, retry_after: float | None = None):
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
    """SDK for a provider is not installed."""
    def __init__(self, provider: str, package: str, extra: str):
        super().__init__(
            f"{provider} provider requires the '{package}' package. "
            f"Install with: pip install anchor[{extra}]"
        )
```

Each provider adapter maps SDK-specific exceptions to these. For example, `anthropic.RateLimitError` → `RateLimitError`, `openai.AuthenticationError` → `AuthenticationError`.

### BaseLLMProvider (shared logic)

```python
class BaseLLMProvider(ABC):
    """Abstract base class with shared retry, timeout, and metric logic."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 2,
        timeout: float = 60.0,
        **kwargs,
    ):
        self._model = model
        self._api_key = api_key or self._resolve_api_key()
        self._base_url = base_url
        self._max_retries = max_retries
        self._timeout = timeout

    @property
    def model_id(self) -> str:
        return f"{self.provider_name}/{self._model}"

    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @abstractmethod
    def _resolve_api_key(self) -> str | None:
        """Resolve API key from environment. Each provider knows its env var."""
        ...

    # -- Sync abstract methods (provider-specific) --

    @abstractmethod
    def _do_invoke(self, messages: list[Message], tools: list[ToolSchema] | None, **kwargs) -> LLMResponse:
        """Provider-specific invoke."""
        ...

    @abstractmethod
    def _do_stream(self, messages: list[Message], tools: list[ToolSchema] | None, **kwargs) -> Iterator[StreamChunk]:
        """Provider-specific stream."""
        ...

    # -- Async abstract methods (provider-specific) --

    @abstractmethod
    async def _do_ainvoke(self, messages: list[Message], tools: list[ToolSchema] | None, **kwargs) -> LLMResponse:
        """Provider-specific async invoke."""
        ...

    @abstractmethod
    async def _do_astream(self, messages: list[Message], tools: list[ToolSchema] | None, **kwargs) -> AsyncIterator[StreamChunk]:
        """Provider-specific async stream."""
        ...

    # -- Public methods with retry wrapping --

    def invoke(self, messages, *, tools=None, **kwargs) -> LLMResponse:
        """Invoke with retry logic for transient errors (429, 503, timeouts)."""
        return self._with_retry(self._do_invoke, messages, tools, **kwargs)

    def stream(self, messages, *, tools=None, **kwargs) -> Iterator[StreamChunk]:
        """Stream with retry (only retries before first chunk is yielded)."""
        return self._with_retry(self._do_stream, messages, tools, **kwargs)

    async def ainvoke(self, messages, *, tools=None, **kwargs) -> LLMResponse:
        """Async invoke with retry."""
        return await self._with_async_retry(self._do_ainvoke, messages, tools, **kwargs)

    async def astream(self, messages, *, tools=None, **kwargs) -> AsyncIterator[StreamChunk]:
        """Async stream with retry (only retries before first chunk)."""
        return await self._with_async_retry(self._do_astream, messages, tools, **kwargs)

    # -- Retry logic --

    def _with_retry(self, fn, *args, **kwargs):
        """Exponential backoff retry for transient errors.

        Retries up to self._max_retries times with exponential backoff
        (1s, 2s, 4s). Only retries on ProviderError with is_transient=True.
        """
        last_error = None
        for attempt in range(self._max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except ProviderError as e:
                if not e.is_transient or attempt == self._max_retries:
                    raise
                last_error = e
                delay = min(2 ** attempt, 8)
                if isinstance(e, RateLimitError) and e.retry_after:
                    delay = e.retry_after
                time.sleep(delay)
        raise last_error

    async def _with_async_retry(self, fn, *args, **kwargs):
        """Async version of _with_retry."""
        last_error = None
        for attempt in range(self._max_retries + 1):
            try:
                return await fn(*args, **kwargs)
            except ProviderError as e:
                if not e.is_transient or attempt == self._max_retries:
                    raise
                last_error = e
                delay = min(2 ** attempt, 8)
                if isinstance(e, RateLimitError) and e.retry_after:
                    delay = e.retry_after
                await asyncio.sleep(delay)
        raise last_error
```

### Provider registry

```python
# Registry maps provider prefix -> provider class
_PROVIDERS: dict[str, type[BaseLLMProvider]] = {}

def register_provider(name: str, cls: type[BaseLLMProvider]):
    """Register a provider adapter. Called at import time by each adapter."""
    _PROVIDERS[name] = cls

def create_provider(
    model: str,
    *,
    api_key: str | None = None,
    fallbacks: list[str] | None = None,
    **kwargs,
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
        # Try lazy import
        _try_import_provider(provider_name)

    if provider_name not in _PROVIDERS:
        raise ProviderNotInstalledError(
            provider_name,
            _PROVIDER_PACKAGES.get(provider_name, provider_name),
            provider_name,
        )

    cls = _PROVIDERS[provider_name]
    primary = cls(model=model_name, api_key=api_key, **kwargs)

    if fallbacks:
        fallback_providers = [create_provider(fb) for fb in fallbacks]
        return FallbackProvider(primary=primary, fallbacks=fallback_providers)

    return primary

def _parse_model_string(model: str) -> tuple[str, str]:
    """Parse 'provider/model' into (provider, model).

    No prefix defaults to 'anthropic' for backward compat.
    Splits on first '/' only — preserves slashes in model name
    (important for OpenRouter's 'provider/model' model names and
    OpenAI fine-tuned model IDs like 'ft:gpt-4o:org:name:id').
    """
    if "/" not in model:
        return "anthropic", model
    provider, _, model_name = model.partition("/")
    return provider, model_name

# Maps provider name -> (module_path, env_var_for_key)
_PROVIDER_MODULES: dict[str, str] = {
    "anthropic": "anchor.llm.providers.anthropic",
    "openai": "anchor.llm.providers.openai",
    "gemini": "anchor.llm.providers.gemini",
    "grok": "anchor.llm.providers.grok",
    "ollama": "anchor.llm.providers.ollama",
    "openrouter": "anchor.llm.providers.openrouter",
    "litellm": "anchor.llm.providers.litellm",
}

_PROVIDER_PACKAGES: dict[str, str] = {
    "anthropic": "anthropic",
    "openai": "openai",
    "gemini": "google-genai",
    "grok": "openai",
    "ollama": "ollama",
    "openrouter": "openai",
    "litellm": "litellm",
}

def _try_import_provider(name: str) -> None:
    """Attempt to lazily import a provider module."""
    module_path = _PROVIDER_MODULES.get(name)
    if module_path:
        import importlib
        try:
            importlib.import_module(module_path)
        except ImportError:
            pass  # Will be caught by create_provider's error handling
```

### FallbackProvider

```python
class FallbackProvider:
    """Wraps multiple providers with fallback-on-failure logic.

    Fallback behavior:
    - invoke/ainvoke: If primary raises transient ProviderError, tries next provider.
    - stream/astream: Fallback only happens BEFORE the first chunk is yielded.
      Once streaming has started yielding chunks, a failure raises immediately
      (no silent switch to another provider mid-stream).
    """

    def __init__(
        self,
        primary: LLMProvider,
        fallbacks: list[LLMProvider],
    ):
        self._providers = [primary] + fallbacks
        self._primary = primary

    @property
    def model_id(self) -> str:
        """Returns primary provider's model_id."""
        return self._primary.model_id

    @property
    def provider_name(self) -> str:
        """Returns primary provider's name."""
        return self._primary.provider_name

    def invoke(self, messages, **kwargs) -> LLMResponse:
        last_error = None
        for provider in self._providers:
            try:
                return provider.invoke(messages, **kwargs)
            except ProviderError as e:
                if not e.is_transient:
                    raise
                last_error = e
        raise last_error

    def stream(self, messages, **kwargs) -> Iterator[StreamChunk]:
        """Stream with pre-first-chunk fallback only.

        Tries each provider. If a provider fails before yielding any
        chunks, moves to the next. Once a chunk has been yielded,
        any failure raises immediately — no silent mid-stream switch.
        """
        last_error = None
        for provider in self._providers:
            try:
                stream_iter = provider.stream(messages, **kwargs)
                first_chunk = next(stream_iter)
                # First chunk received — commit to this provider
                yield first_chunk
                yield from stream_iter
                return
            except StopIteration:
                return  # Empty stream, that's fine
            except ProviderError as e:
                if not e.is_transient:
                    raise
                last_error = e
        raise last_error

    async def ainvoke(self, messages, **kwargs) -> LLMResponse:
        last_error = None
        for provider in self._providers:
            try:
                return await provider.ainvoke(messages, **kwargs)
            except ProviderError as e:
                if not e.is_transient:
                    raise
                last_error = e
        raise last_error

    async def astream(self, messages, **kwargs) -> AsyncIterator[StreamChunk]:
        """Async stream with pre-first-chunk fallback only."""
        last_error = None
        for provider in self._providers:
            try:
                stream_iter = provider.astream(messages, **kwargs)
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
        raise last_error
```

### Example provider adapter (Anthropic)

```python
class AnthropicProvider(BaseLLMProvider):
    """Adapter for Anthropic's Messages API."""

    provider_name = "anthropic"

    def __init__(self, model: str, **kwargs):
        super().__init__(model=model, **kwargs)
        try:
            import anthropic as _anthropic
        except ImportError:
            raise ProviderNotInstalledError("Anthropic", "anthropic", "anthropic")
        self._anthropic = _anthropic
        self._client = _anthropic.Anthropic(
            api_key=self._api_key,
            max_retries=0,  # we handle retries ourselves
            timeout=self._timeout,
        )
        self._async_client = _anthropic.AsyncAnthropic(
            api_key=self._api_key,
            max_retries=0,
            timeout=self._timeout,
        )

    def _resolve_api_key(self) -> str | None:
        return os.environ.get("ANTHROPIC_API_KEY")

    def _do_invoke(self, messages, tools, **kwargs) -> LLMResponse:
        system_msgs, chat_msgs = self._split_system(messages)
        api_messages = [self._to_anthropic_msg(m) for m in chat_msgs]
        api_tools = [self._to_anthropic_tool(t) for t in (tools or [])]

        try:
            response = self._client.messages.create(
                model=self._model,
                system=system_msgs or self._anthropic.NOT_GIVEN,
                messages=api_messages,
                tools=api_tools or self._anthropic.NOT_GIVEN,
                max_tokens=kwargs.get("max_tokens", 4096),
                temperature=kwargs.get("temperature", self._anthropic.NOT_GIVEN),
            )
        except self._anthropic.RateLimitError as e:
            raise RateLimitError(str(e), provider="anthropic")
        except self._anthropic.AuthenticationError as e:
            raise AuthenticationError(str(e), provider="anthropic")
        except self._anthropic.APIStatusError as e:
            if e.status_code >= 500:
                raise ServerError(str(e), provider="anthropic")
            raise ProviderError(str(e), provider="anthropic")

        return self._parse_response(response)

    def _do_stream(self, messages, tools, **kwargs) -> Iterator[StreamChunk]:
        system_msgs, chat_msgs = self._split_system(messages)
        api_messages = [self._to_anthropic_msg(m) for m in chat_msgs]
        api_tools = [self._to_anthropic_tool(t) for t in (tools or [])]

        with self._client.messages.stream(
            model=self._model,
            system=system_msgs or self._anthropic.NOT_GIVEN,
            messages=api_messages,
            tools=api_tools or self._anthropic.NOT_GIVEN,
            max_tokens=kwargs.get("max_tokens", 4096),
        ) as stream:
            for event in stream:
                yield self._parse_stream_event(event)

    async def _do_ainvoke(self, messages, tools, **kwargs) -> LLMResponse:
        """Async invoke using AsyncAnthropic client."""
        system_msgs, chat_msgs = self._split_system(messages)
        api_messages = [self._to_anthropic_msg(m) for m in chat_msgs]
        api_tools = [self._to_anthropic_tool(t) for t in (tools or [])]

        try:
            response = await self._async_client.messages.create(
                model=self._model,
                system=system_msgs or self._anthropic.NOT_GIVEN,
                messages=api_messages,
                tools=api_tools or self._anthropic.NOT_GIVEN,
                max_tokens=kwargs.get("max_tokens", 4096),
                temperature=kwargs.get("temperature", self._anthropic.NOT_GIVEN),
            )
        except self._anthropic.RateLimitError as e:
            raise RateLimitError(str(e), provider="anthropic")
        except self._anthropic.AuthenticationError as e:
            raise AuthenticationError(str(e), provider="anthropic")
        except self._anthropic.APIStatusError as e:
            if e.status_code >= 500:
                raise ServerError(str(e), provider="anthropic")
            raise ProviderError(str(e), provider="anthropic")

        return self._parse_response(response)

    async def _do_astream(self, messages, tools, **kwargs) -> AsyncIterator[StreamChunk]:
        """Async stream using AsyncAnthropic client."""
        system_msgs, chat_msgs = self._split_system(messages)
        api_messages = [self._to_anthropic_msg(m) for m in chat_msgs]
        api_tools = [self._to_anthropic_tool(t) for t in (tools or [])]

        async with self._async_client.messages.stream(
            model=self._model,
            system=system_msgs or self._anthropic.NOT_GIVEN,
            messages=api_messages,
            tools=api_tools or self._anthropic.NOT_GIVEN,
            max_tokens=kwargs.get("max_tokens", 4096),
        ) as stream:
            async for event in stream:
                yield self._parse_stream_event(event)

    # --- Conversion helpers ---

    def _split_system(self, messages: list[Message]) -> tuple[list[dict] | None, list[Message]]:
        """Split system messages (Anthropic takes them separately)."""
        system = []
        chat = []
        for m in messages:
            if m.role == Role.SYSTEM:
                text = m.content if isinstance(m.content, str) else ""
                system.append({"type": "text", "text": text})
            else:
                chat.append(m)
        return system or None, chat

    def _to_anthropic_msg(self, msg: Message) -> dict:
        """Convert unified Message to Anthropic message dict."""
        ...

    def _to_anthropic_tool(self, tool: ToolSchema) -> dict:
        """Convert ToolSchema to Anthropic tool definition."""
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
        }

    def _parse_response(self, response) -> LLMResponse:
        """Convert Anthropic response to unified LLMResponse."""
        ...

    def _parse_stream_event(self, event) -> StreamChunk:
        """Convert Anthropic stream event to unified StreamChunk."""
        ...
```

### Example provider adapter (OpenAI)

```python
class OpenAIProvider(BaseLLMProvider):
    """Adapter for OpenAI Chat Completions API."""

    provider_name = "openai"

    def __init__(self, model: str, **kwargs):
        super().__init__(model=model, **kwargs)
        try:
            import openai as _openai
        except ImportError:
            raise ProviderNotInstalledError("OpenAI", "openai", "openai")
        self._openai = _openai
        self._client = _openai.OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            max_retries=0,
            timeout=self._timeout,
        )
        self._async_client = _openai.AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            max_retries=0,
            timeout=self._timeout,
        )

    def _resolve_api_key(self) -> str | None:
        return os.environ.get("OPENAI_API_KEY")

    def _do_invoke(self, messages, tools, **kwargs) -> LLMResponse:
        api_messages = [self._to_openai_msg(m) for m in messages]
        api_tools = [self._to_openai_tool(t) for t in (tools or [])]

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=api_messages,
                tools=api_tools or self._openai.NOT_GIVEN,
                max_tokens=kwargs.get("max_tokens"),
                temperature=kwargs.get("temperature"),
            )
        except self._openai.RateLimitError as e:
            raise RateLimitError(str(e), provider=self.provider_name)
        except self._openai.AuthenticationError as e:
            raise AuthenticationError(str(e), provider=self.provider_name)
        except self._openai.APIStatusError as e:
            if e.status_code >= 500:
                raise ServerError(str(e), provider=self.provider_name)
            raise ProviderError(str(e), provider=self.provider_name)

        return self._parse_response(response)

    def _do_stream(self, messages, tools, **kwargs) -> Iterator[StreamChunk]:
        """OpenAI streaming."""
        ...

    async def _do_ainvoke(self, messages, tools, **kwargs) -> LLMResponse:
        """Async invoke using AsyncOpenAI."""
        ...

    async def _do_astream(self, messages, tools, **kwargs) -> AsyncIterator[StreamChunk]:
        """Async stream using AsyncOpenAI."""
        ...
```

### Gemini provider adapter

Gemini uses the `google-genai` SDK which has its own message format (not OpenAI-compatible), so it needs a full adapter:

```python
class GeminiProvider(BaseLLMProvider):
    """Adapter for Google Gemini via google-genai SDK."""

    provider_name = "gemini"

    def __init__(self, model: str, **kwargs):
        super().__init__(model=model, **kwargs)
        try:
            from google import genai
        except ImportError:
            raise ProviderNotInstalledError("Gemini", "google-genai", "gemini")
        self._genai = genai
        self._client = genai.Client(api_key=self._api_key)

    def _resolve_api_key(self) -> str | None:
        return os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

    def _do_invoke(self, messages, tools, **kwargs) -> LLMResponse:
        contents = [self._to_gemini_content(m) for m in messages if m.role != Role.SYSTEM]
        system_instruction = self._extract_system(messages)
        gemini_tools = [self._to_gemini_tool(t) for t in (tools or [])]

        config = self._genai.types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=gemini_tools or None,
            max_output_tokens=kwargs.get("max_tokens"),
            temperature=kwargs.get("temperature"),
        )

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            # Map google-genai exceptions to our error hierarchy
            raise ProviderError(str(e), provider="gemini")

        return self._parse_response(response)

    # ... stream, ainvoke, astream, conversion helpers
```

### OpenAI-compatible providers (Grok, OpenRouter)

Grok and OpenRouter both expose OpenAI-compatible APIs. Their adapters are thin subclasses:

```python
class GrokProvider(OpenAIProvider):
    """Grok via xAI's OpenAI-compatible API."""

    provider_name = "grok"

    def __init__(self, model: str, **kwargs):
        kwargs.setdefault("base_url", "https://api.x.ai/v1")
        super().__init__(model=model, **kwargs)

    def _resolve_api_key(self) -> str | None:
        return os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")


class OpenRouterProvider(OpenAIProvider):
    """OpenRouter unified gateway."""

    provider_name = "openrouter"

    def __init__(self, model: str, **kwargs):
        kwargs.setdefault("base_url", "https://openrouter.ai/api/v1")
        super().__init__(model=model, **kwargs)

    def _resolve_api_key(self) -> str | None:
        return os.environ.get("OPENROUTER_API_KEY")
```

### Agent integration

The Agent class changes from importing `anthropic` directly to using `LLMProvider`. The full tool loop migration is detailed here.

**Current Agent `__slots__`** (to be updated):
```python
# Remove: "_client", "_model"
# Add: "_llm"
# Keep: "_formatter" (already exists)
__slots__ = (
    "_llm",           # LLMProvider (replaces _client + _model)
    "_formatter",     # Formatter
    "_pipeline",      # ContextPipeline
    "_memory",        # MemoryManager | None
    "_tools",         # list[AgentTool]
    "_skills",        # SkillRegistry
    # ... rest unchanged
)
```

**Constructor changes:**
```python
class Agent:
    def __init__(
        self,
        model: str = "anthropic/claude-haiku-4-5-20251001",
        *,
        llm: LLMProvider | None = None,  # replaces client: Any
        api_key: str | None = None,
        fallbacks: list[str] | None = None,
        max_response_tokens: int = 4096,
        ...
    ):
        # Use explicit provider or create from model string
        if llm is not None:
            self._llm = llm
        else:
            self._llm = create_provider(
                model, api_key=api_key, fallbacks=fallbacks
            )
        self._max_response_tokens = max_response_tokens

        # Auto-select formatter based on provider
        if formatter is None:
            self._formatter = self._auto_formatter(self._llm.provider_name)

    def _auto_formatter(self, provider: str) -> Formatter:
        if provider == "anthropic":
            return AnthropicFormatter()
        return OpenAIFormatter()  # openai, grok, ollama, gemini, openrouter
```

**Tool loop migration** (replaces the current Anthropic-specific `chat()` method):

```python
def chat(self, user_input: str) -> Iterator[str]:
    """Multi-turn chat with tool use loop.

    Migrated from Anthropic-specific to provider-agnostic.
    Key changes:
    - Uses self._llm.stream() instead of self._client.messages.stream()
    - Uses unified Message/ToolCall/LLMResponse instead of Anthropic SDK types
    - _serialize_response and _run_tools operate on unified types
    """
    # 1. Add user message to conversation
    self._add_user_message(user_input)

    # 2. Build context via pipeline
    context = self._pipeline.build(user_input)

    # 3. Convert pipeline output to list[Message]
    messages = self._context_to_messages(context)

    # 4. Get tool schemas
    tool_schemas = [t.to_tool_schema() for t in self._tools]

    # 5. Multi-round tool loop
    for round_num in range(self._max_rounds):
        response_text = ""
        tool_calls = []

        # Stream response
        for chunk in self._llm.stream(
            messages,
            tools=tool_schemas if self._tools else None,
            max_tokens=self._max_response_tokens,
        ):
            if chunk.content:
                response_text += chunk.content
                yield chunk.content
            if chunk.tool_call_delta:
                self._accumulate_tool_call(tool_calls, chunk.tool_call_delta)
            if chunk.stop_reason == StopReason.STOP:
                # No tool calls — we're done
                self._add_assistant_message(response_text)
                return
            if chunk.stop_reason == StopReason.TOOL_USE:
                break

        # Handle tool calls
        if tool_calls:
            # Add assistant message with tool calls
            messages.append(Message(
                role=Role.ASSISTANT,
                content=response_text or None,
                tool_calls=tool_calls,
            ))
            # Execute tools and add results
            for tc in tool_calls:
                result = self._execute_tool(tc)
                messages.append(Message(
                    role=Role.TOOL,
                    tool_result=ToolResult(
                        tool_call_id=tc.id,
                        content=result,
                    ),
                ))
            # Continue loop for next round
        else:
            break

def _context_to_messages(self, context: ContextResult) -> list[Message]:
    """Convert pipeline ContextResult to list[Message] for the provider.

    The pipeline's formatted_output is already in the right format for
    the selected formatter. We extract the message list from it.
    """
    output = context.formatted_output
    messages = []

    # Anthropic format: {"system": [...], "messages": [...]}
    # OpenAI format: {"messages": [...]}
    if isinstance(output, dict):
        if "system" in output:
            # Anthropic format — convert system blocks to system Message
            system_text = " ".join(b["text"] for b in output["system"])
            messages.append(Message(role=Role.SYSTEM, content=system_text))
        for msg in output.get("messages", []):
            messages.append(Message(
                role=Role(msg["role"]),
                content=msg.get("content"),
            ))

    return messages
```

**Methods that change:**

| Current method | Change |
|---|---|
| `_call_api_with_retry` | Removed — retry now lives in `BaseLLMProvider._with_retry` |
| `_call_api_with_retry_async` | Removed — retry now in `BaseLLMProvider._with_async_retry` |
| `_serialize_response` | Rewritten to operate on `LLMResponse` instead of Anthropic SDK types |
| `_run_tools` | Rewritten to take `list[ToolCall]` instead of Anthropic content blocks |
| `_build_tool_result_content` | Simplified — returns `ToolResult` instead of Anthropic tool_result dict |

**Test migration:** The existing `Agent` tests that pass `client=mock_anthropic_client` will need updating to pass `llm=mock_llm_provider` instead. This is a breaking change for test code only; the public API remains backward-compatible.

### Cost tracking

```python
# src/anchor/llm/pricing.py

# Built-in pricing table (subset shown)
# Last updated: 2026-03-13
# Prices in USD per 1M tokens
MODEL_PRICING: dict[str, dict[str, float]] = {
    # Anthropic
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.0},
    "claude-opus-4-20250514": {"input": 15.0, "output": 75.0},
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1": {"input": 2.0, "output": 8.0},
    # Google
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.0},
    # ... more models
}

def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    """Calculate USD cost. Returns None if model pricing unknown.

    Tries exact match first, then strips date suffixes for alias matching
    (e.g. 'gpt-4o-2024-08-06' -> 'gpt-4o').
    """
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        # Try stripping common date suffixes for alias resolution
        normalized = _normalize_model_name(model)
        pricing = MODEL_PRICING.get(normalized)
    if pricing is None:
        return None
    return (prompt_tokens * pricing["input"] + completion_tokens * pricing["output"]) / 1_000_000

def _normalize_model_name(model: str) -> str:
    """Strip date suffixes and common aliases.

    'gpt-4o-2024-08-06' -> 'gpt-4o'
    'claude-sonnet-4-20250514' -> 'claude-sonnet-4-20250514' (keep, it's the canonical name)
    """
    import re
    # Strip trailing date patterns like -YYYY-MM-DD or -YYYYMMDD
    return re.sub(r'-\d{4}-?\d{2}-?\d{2}$', '', model)
```

**Staleness strategy:** The pricing table is best-effort and ships with the package. It includes a `# Last updated` comment. Users can override/extend pricing at runtime:

```python
from anchor.llm.pricing import MODEL_PRICING

# Add custom model pricing
MODEL_PRICING["my-fine-tuned-model"] = {"input": 5.0, "output": 15.0}
```

### Packaging (pyproject.toml)

```toml
[project.optional-dependencies]
anthropic = ["anthropic>=0.40"]
openai = ["openai>=1.50"]
gemini = ["google-genai>=1.0"]
ollama = ["ollama>=0.4"]
litellm = ["litellm>=1.50"]
all-providers = [
    "anthropic>=0.40",
    "openai>=1.50",
    "google-genai>=1.0",
    "ollama>=0.4",
]
```

Note: `grok` and `openrouter` extras are not needed — they use the `openai` SDK with a custom `base_url`. Installing `anchor[openai]` unlocks OpenAI + Grok + OpenRouter.

## What's NOT in scope

- **Proactive rate limiting** — gateway/proxy concern, not SDK
- **Prompt caching** — provider-specific feature, passed through via `**kwargs`
- **Embeddings** — separate concern, could be a future addition
- **Image/audio generation** — out of scope for this design
- **Provider-specific features** (extended thinking, prompt caching) — accessible via `**kwargs` passthrough
- **Connection health checks** — could be added later as `provider.ping()` but not needed for v1

## Migration path

1. Existing `Agent(model="claude-haiku-4-5-20251001")` continues working (defaults to `anthropic/`)
2. Existing `Agent(api_key="sk-ant-...")` continues working
3. No breaking changes to `ContextPipeline` or formatters
4. `AgentTool` gains `to_tool_schema()` — existing `to_anthropic_schema()` / `to_openai_schema()` remain
5. `Agent(client=...)` parameter is replaced by `Agent(llm=...)` — **breaking change for test code** that injects mock Anthropic clients. Migration: replace `client=mock_client` with `llm=mock_provider`
6. Existing `StreamResult`/`StreamUsage` models in `models/streaming.py` are untouched — they coexist with the new `StreamChunk`/`Usage` at the LLM layer

## Testing strategy

- Unit tests per provider adapter (mock SDK responses)
- Integration tests with real API calls (marked `@pytest.mark.integration`, skipped in CI)
- Test fallback chain behavior (primary fails -> secondary succeeds)
- Test streaming fallback (fails before first chunk -> switches; fails after first chunk -> raises)
- Test model string parsing edge cases (no prefix, fine-tuned models, OpenRouter double-prefix)
- Test backward compatibility (no-prefix defaults to anthropic)
- Test cost calculation accuracy and alias normalization
- Test lazy import error messages when SDK not installed
- Test retry with exponential backoff (mock transient errors)
- Test error mapping (each SDK's exceptions -> unified ProviderError hierarchy)
- Test `AgentTool.to_tool_schema()` conversion
- Test Agent tool loop with mock LLMProvider (end-to-end)
