"""LiteLLMProvider adapter for the multi-provider LLM layer.

LiteLLM is an optional catch-all that routes to 100+ LLM providers
using a unified OpenAI-compatible interface. It handles API keys
internally via environment variables, so `_resolve_api_key()` returns None.

The `litellm` SDK is imported lazily inside methods so this module
can be imported even when the SDK is not installed (import fails only
when you actually try to use the provider).

Self-registers via register_provider() at module import time.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterator

from anchor.llm.base import BaseLLMProvider
from anchor.llm.errors import (
    AuthenticationError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
    ServerError,
    TimeoutError,
)
from anchor.llm.models import (
    LLMResponse,
    Message,
    Role,
    StopReason,
    StreamChunk,
    ToolCall,
    ToolCallDelta,
    ToolSchema,
    Usage,
)
from anchor.llm.registry import register_provider


# ---------------------------------------------------------------------------
# Stop reason mapping
# ---------------------------------------------------------------------------

_STOP_REASON_MAP: dict[str, StopReason] = {
    "stop": StopReason.STOP,
    "length": StopReason.MAX_TOKENS,
    "tool_calls": StopReason.TOOL_USE,
}


def _map_stop_reason(finish_reason: str | None) -> StopReason:
    if finish_reason is None:
        return StopReason.STOP
    return _STOP_REASON_MAP.get(finish_reason, StopReason.STOP)


# ---------------------------------------------------------------------------
# LiteLLMProvider
# ---------------------------------------------------------------------------


class LiteLLMProvider(BaseLLMProvider):
    """Adapter for the LiteLLM library.

    LiteLLM acts as a catch-all, routing to 100+ providers using an
    OpenAI-compatible response format. It manages API key resolution
    internally through environment variables (e.g., OPENAI_API_KEY,
    ANTHROPIC_API_KEY, etc.), so _resolve_api_key() returns None.

    Usage:
        provider = LiteLLMProvider(model="gpt-4o")
        # or for other providers:
        provider = LiteLLMProvider(model="anthropic/claude-3-5-sonnet-20241022")
        provider = LiteLLMProvider(model="gemini/gemini-2.0-flash")
    """

    provider_name = "litellm"

    # ------------------------------------------------------------------
    # BaseLLMProvider abstract method implementations
    # ------------------------------------------------------------------

    def _resolve_api_key(self) -> str | None:
        """LiteLLM manages API keys via environment variables internally."""
        return None

    def _do_invoke(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        **kwargs: Any,
    ) -> LLMResponse:
        import litellm  # noqa: PLC0415 — lazy import

        converted = self._convert_messages(messages)

        call_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": converted,
            "max_tokens": kwargs.get("max_tokens", 4096),
        }
        if tools:
            call_kwargs["tools"] = [self._convert_tool(t) for t in tools]
        if kwargs.get("temperature") is not None:
            call_kwargs["temperature"] = kwargs["temperature"]
        if kwargs.get("stop"):
            call_kwargs["stop"] = kwargs["stop"]

        try:
            response = litellm.completion(**call_kwargs)
        except Exception as exc:
            raise self._map_error(exc) from exc

        return self._parse_response(response)

    def _do_stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        import litellm  # noqa: PLC0415 — lazy import

        converted = self._convert_messages(messages)

        call_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": converted,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "stream": True,
        }
        if tools:
            call_kwargs["tools"] = [self._convert_tool(t) for t in tools]
        if kwargs.get("temperature") is not None:
            call_kwargs["temperature"] = kwargs["temperature"]
        if kwargs.get("stop"):
            call_kwargs["stop"] = kwargs["stop"]

        try:
            stream = litellm.completion(**call_kwargs)
            for raw_chunk in stream:
                chunk = self._parse_stream_chunk(raw_chunk)
                if chunk is not None:
                    yield chunk
        except Exception as exc:
            raise self._map_error(exc) from exc

    async def _do_ainvoke(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        **kwargs: Any,
    ) -> LLMResponse:
        import litellm  # noqa: PLC0415 — lazy import

        converted = self._convert_messages(messages)

        call_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": converted,
            "max_tokens": kwargs.get("max_tokens", 4096),
        }
        if tools:
            call_kwargs["tools"] = [self._convert_tool(t) for t in tools]
        if kwargs.get("temperature") is not None:
            call_kwargs["temperature"] = kwargs["temperature"]
        if kwargs.get("stop"):
            call_kwargs["stop"] = kwargs["stop"]

        try:
            response = await litellm.acompletion(**call_kwargs)
        except Exception as exc:
            raise self._map_error(exc) from exc

        return self._parse_response(response)

    async def _do_astream(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        import litellm  # noqa: PLC0415 — lazy import

        converted = self._convert_messages(messages)

        call_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": converted,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "stream": True,
        }
        if tools:
            call_kwargs["tools"] = [self._convert_tool(t) for t in tools]
        if kwargs.get("temperature") is not None:
            call_kwargs["temperature"] = kwargs["temperature"]
        if kwargs.get("stop"):
            call_kwargs["stop"] = kwargs["stop"]

        try:
            stream = await litellm.acompletion(**call_kwargs)
            async for raw_chunk in stream:
                chunk = self._parse_stream_chunk(raw_chunk)
                if chunk is not None:
                    yield chunk
        except Exception as exc:
            raise self._map_error(exc) from exc

    # ------------------------------------------------------------------
    # Message conversion helpers
    # ------------------------------------------------------------------

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert Anchor messages to OpenAI-compatible format used by LiteLLM.

        LiteLLM mirrors the OpenAI Chat Completions wire format:
        - System messages stay in the messages list
        - Tool results use role='tool' with tool_call_id
        - Assistant tool calls use JSON strings for arguments
        """
        converted: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                converted.append({"role": "system", "content": msg.content or ""})
                continue

            if msg.role == Role.TOOL:
                # Tool result → role='tool' with tool_call_id
                if msg.tool_result is not None:
                    converted.append(
                        {
                            "role": "tool",
                            "tool_call_id": msg.tool_result.tool_call_id,
                            "content": msg.tool_result.content,
                        }
                    )
                continue

            if msg.role == Role.ASSISTANT and msg.tool_calls:
                # Assistant message with tool calls
                litellm_msg: dict[str, Any] = {"role": "assistant"}
                if msg.content:
                    litellm_msg["content"] = msg.content
                litellm_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            # LiteLLM expects arguments as a JSON string
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in msg.tool_calls
                ]
                converted.append(litellm_msg)
                continue

            # Regular user / assistant messages
            role_str = "user" if msg.role == Role.USER else "assistant"
            if isinstance(msg.content, str):
                converted.append({"role": role_str, "content": msg.content})
            elif isinstance(msg.content, list):
                blocks: list[dict[str, Any]] = []
                for block in msg.content:
                    if block.type == "text" and block.text is not None:
                        blocks.append({"type": "text", "text": block.text})
                    elif block.type == "image_url" and block.image_url is not None:
                        blocks.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": block.image_url},
                            }
                        )
                    elif block.type == "image_base64" and block.image_base64 is not None:
                        media_type = block.media_type or "image/png"
                        blocks.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{block.image_base64}"
                                },
                            }
                        )
                converted.append({"role": role_str, "content": blocks})

        return converted

    # ------------------------------------------------------------------
    # Tool schema conversion
    # ------------------------------------------------------------------

    def _convert_tool(self, tool: ToolSchema) -> dict[str, Any]:
        """Convert a ToolSchema to OpenAI-compatible tool definition format.

        LiteLLM uses the same OpenAI format with 'parameters' (not 'input_schema').
        """
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
        }

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse a LiteLLM response (OpenAI-compatible) into an LLMResponse."""
        choice = response.choices[0]
        message = choice.message

        content = message.content if message.content else None

        tool_calls: list[ToolCall] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                # arguments comes as a JSON string from LiteLLM
                try:
                    arguments = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    arguments = {}
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=arguments,
                    )
                )

        usage = Usage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls if tool_calls else None,
            usage=usage,
            model=response.model,
            provider=self.provider_name,
            stop_reason=_map_stop_reason(choice.finish_reason),
        )

    # ------------------------------------------------------------------
    # Stream chunk parsing
    # ------------------------------------------------------------------

    def _parse_stream_chunk(self, chunk: Any) -> StreamChunk | None:
        """Parse a single LiteLLM stream chunk into a StreamChunk, or None."""
        if not chunk.choices:
            return None

        choice = chunk.choices[0]
        delta = choice.delta
        finish_reason = choice.finish_reason

        # Handle finish reason first
        if finish_reason is not None:
            return StreamChunk(stop_reason=_map_stop_reason(finish_reason))

        # Handle tool call deltas
        if delta.tool_calls:
            tc_delta = delta.tool_calls[0]
            func_delta = tc_delta.function
            return StreamChunk(
                tool_call_delta=ToolCallDelta(
                    index=tc_delta.index,
                    id=tc_delta.id if tc_delta.id else None,
                    name=func_delta.name if func_delta.name else None,
                    arguments_fragment=func_delta.arguments if func_delta.arguments else None,
                )
            )

        # Handle text content
        if delta.content is not None:
            return StreamChunk(content=delta.content)

        return None

    # ------------------------------------------------------------------
    # Error mapping
    # ------------------------------------------------------------------

    def _map_error(self, exc: Exception) -> ProviderError:
        """Map a LiteLLM exception to our error hierarchy.

        LiteLLM raises its own exception types that mirror the OpenAI names.
        Uses class name matching so this works with both real SDK exceptions
        and dynamically-created test mocks.
        """
        # Walk the full MRO and collect all class names — handles
        # both real SDK exceptions and dynamically-created test mocks.
        mro_names = {cls.__name__ for cls in type(exc).__mro__}

        if "AuthenticationError" in mro_names:
            return AuthenticationError(str(exc), provider=self.provider_name)

        if "RateLimitError" in mro_names:
            return RateLimitError(str(exc), provider=self.provider_name)

        if "NotFoundError" in mro_names or "ModelNotFoundError" in mro_names:
            return ModelNotFoundError(str(exc), provider=self.provider_name)

        if "APIConnectionError" in mro_names or "APIConnectTimeoutError" in mro_names:
            return TimeoutError(str(exc), provider=self.provider_name)

        if "APITimeoutError" in mro_names or "Timeout" in mro_names:
            return TimeoutError(str(exc), provider=self.provider_name)

        if "APIStatusError" in mro_names:
            status_code = getattr(exc, "status_code", 0)
            if status_code >= 500:
                return ServerError(str(exc), provider=self.provider_name)
            return ProviderError(str(exc), provider=self.provider_name, is_transient=False)

        if "ServiceUnavailableError" in mro_names:
            return ServerError(str(exc), provider=self.provider_name)

        # Fallback
        return ProviderError(str(exc), provider=self.provider_name)


# ---------------------------------------------------------------------------
# Self-registration
# ---------------------------------------------------------------------------

register_provider("litellm", LiteLLMProvider)
