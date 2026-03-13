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
            assert provider.model_id == "fake/m1"
            result = provider.invoke([Message(role="user", content="hi")])
            assert result.content == "ok"
        finally:
            _PROVIDERS.pop("fake", None)
