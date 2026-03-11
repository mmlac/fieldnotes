"""Tests for AnthropicProvider complete method."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from worker.models.providers.anthropic import AnthropicProvider
from worker.models.base import CompletionRequest, CompletionResponse


@pytest.fixture()
def provider() -> AnthropicProvider:
    p = AnthropicProvider()
    p._client = MagicMock()
    p._api_key = "sk-test-key"
    return p


def _text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _tool_use_block(id: str, name: str, input: dict) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=id, name=name, input=input)


def _usage(input_tokens: int = 10, output_tokens: int = 5, cache_read: int = 0) -> SimpleNamespace:
    ns = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    if cache_read:
        ns.cache_read_input_tokens = cache_read
    return ns


def _mock_response(content: list, usage=None) -> SimpleNamespace:
    return SimpleNamespace(
        content=content,
        usage=usage or _usage(),
    )


class TestAnthropicProviderType:
    def test_provider_type(self, provider: AnthropicProvider) -> None:
        assert provider.provider_type == "anthropic"


class TestAnthropicProviderConfigure:
    def test_configure_with_api_key(self) -> None:
        p = AnthropicProvider()
        with patch("worker.models.providers.anthropic.anthropic.Anthropic") as mock_cls:
            p.configure({"api_key": "sk-from-config"})
        assert p._api_key == "sk-from-config"
        mock_cls.assert_called_once_with(api_key="sk-from-config")

    def test_configure_from_env(self) -> None:
        p = AnthropicProvider()
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-from-env"}):
            with patch("worker.models.providers.anthropic.anthropic.Anthropic"):
                p.configure({})
        assert p._api_key == "sk-from-env"

    def test_configure_raises_without_key(self) -> None:
        p = AnthropicProvider()
        with patch.dict("os.environ", {}, clear=True):
            # Remove ANTHROPIC_API_KEY if present
            import os
            env = os.environ.copy()
            env.pop("ANTHROPIC_API_KEY", None)
            with patch.dict("os.environ", env, clear=True):
                with pytest.raises(ValueError, match="API key required"):
                    p.configure({})


class TestAnthropicComplete:
    def test_basic_completion(self, provider: AnthropicProvider) -> None:
        provider._client.messages.create.return_value = _mock_response(
            content=[_text_block("Hello back!")],
            usage=_usage(input_tokens=10, output_tokens=5),
        )

        req = CompletionRequest(
            system="You are helpful.",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100,
            temperature=0.0,
        )
        resp = provider.complete("claude-sonnet-4-20250514", req)

        assert isinstance(resp, CompletionResponse)
        assert resp.text == "Hello back!"
        assert resp.input_tokens == 10
        assert resp.output_tokens == 5
        assert resp.tool_calls is None

    def test_sends_correct_kwargs(self, provider: AnthropicProvider) -> None:
        provider._client.messages.create.return_value = _mock_response(
            content=[_text_block("ok")],
        )

        req = CompletionRequest(
            system="sys prompt",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=256,
            temperature=0.7,
        )
        provider.complete("claude-sonnet-4-20250514", req)

        call_kwargs = provider._client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-sonnet-4-20250514"
        assert call_kwargs["max_tokens"] == 256
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["system"] == "sys prompt"
        assert call_kwargs["messages"] == [{"role": "user", "content": "hi"}]

    def test_no_system_when_empty(self, provider: AnthropicProvider) -> None:
        provider._client.messages.create.return_value = _mock_response(
            content=[_text_block("ok")],
        )

        req = CompletionRequest(
            system="",
            messages=[{"role": "user", "content": "hi"}],
        )
        provider.complete("m", req)

        call_kwargs = provider._client.messages.create.call_args.kwargs
        assert "system" not in call_kwargs

    def test_tool_use_response(self, provider: AnthropicProvider) -> None:
        provider._client.messages.create.return_value = _mock_response(
            content=[
                _text_block("Let me search for that."),
                _tool_use_block("toolu_123", "search", {"query": "test"}),
            ],
        )

        req = CompletionRequest(
            system="sys",
            messages=[{"role": "user", "content": "search for test"}],
            tools=[{"name": "search", "description": "Search", "input_schema": {"type": "object"}}],
        )
        resp = provider.complete("m", req)

        assert resp.text == "Let me search for that."
        assert resp.tool_calls is not None
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0]["function"]["name"] == "search"
        assert resp.tool_calls[0]["function"]["arguments"] == {"query": "test"}

    def test_tools_passed_in_kwargs(self, provider: AnthropicProvider) -> None:
        provider._client.messages.create.return_value = _mock_response(
            content=[_text_block("ok")],
        )

        tools = [{"name": "foo", "description": "Foo", "input_schema": {"type": "object"}}]
        req = CompletionRequest(
            system="sys",
            messages=[{"role": "user", "content": "hi"}],
            tools=tools,
        )
        provider.complete("m", req)

        call_kwargs = provider._client.messages.create.call_args.kwargs
        assert call_kwargs["tools"] == tools

    def test_openai_format_tools_converted(self, provider: AnthropicProvider) -> None:
        """Tools in OpenAI format are converted to Anthropic format before API call."""
        provider._client.messages.create.return_value = _mock_response(
            content=[_text_block("ok")],
        )

        openai_tools = [{
            "type": "function",
            "function": {
                "name": "extract",
                "description": "Extract entities",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
        }]
        req = CompletionRequest(
            system="sys",
            messages=[{"role": "user", "content": "hi"}],
            tools=openai_tools,
        )
        provider.complete("m", req)

        call_kwargs = provider._client.messages.create.call_args.kwargs
        converted = call_kwargs["tools"]
        assert len(converted) == 1
        assert converted[0]["name"] == "extract"
        assert converted[0]["description"] == "Extract entities"
        assert converted[0]["input_schema"]["type"] == "object"
        assert "function" not in converted[0]
        assert "type" not in converted[0]

    def test_no_tools_key_when_none(self, provider: AnthropicProvider) -> None:
        provider._client.messages.create.return_value = _mock_response(
            content=[_text_block("ok")],
        )

        req = CompletionRequest(
            system="sys",
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
        )
        provider.complete("m", req)

        call_kwargs = provider._client.messages.create.call_args.kwargs
        assert "tools" not in call_kwargs

    def test_cached_tokens_extracted(self, provider: AnthropicProvider) -> None:
        provider._client.messages.create.return_value = _mock_response(
            content=[_text_block("ok")],
            usage=_usage(input_tokens=100, output_tokens=20, cache_read=50),
        )

        req = CompletionRequest(
            system="sys",
            messages=[{"role": "user", "content": "hi"}],
        )
        resp = provider.complete("m", req)

        assert resp.cached_tokens == 50

    def test_cached_tokens_zero_when_absent(self, provider: AnthropicProvider) -> None:
        usage = SimpleNamespace(input_tokens=100, output_tokens=20)
        provider._client.messages.create.return_value = _mock_response(
            content=[_text_block("ok")],
            usage=usage,
        )

        req = CompletionRequest(
            system="sys",
            messages=[{"role": "user", "content": "hi"}],
        )
        resp = provider.complete("m", req)

        assert resp.cached_tokens == 0

    def test_embed_raises_not_implemented(self, provider: AnthropicProvider) -> None:
        from worker.models.base import EmbedRequest
        with pytest.raises(NotImplementedError, match="anthropic does not support embedding"):
            provider.embed("m", EmbedRequest(texts=["test"]))


class TestConvertTools:
    def test_converts_openai_format(self) -> None:
        tools = [{
            "type": "function",
            "function": {
                "name": "my_tool",
                "description": "Does stuff",
                "parameters": {"type": "object", "properties": {}},
            },
        }]
        result = AnthropicProvider._convert_tools(tools)
        assert result == [{
            "name": "my_tool",
            "description": "Does stuff",
            "input_schema": {"type": "object", "properties": {}},
        }]

    def test_passes_through_anthropic_format(self) -> None:
        tools = [{"name": "foo", "description": "Bar", "input_schema": {"type": "object"}}]
        result = AnthropicProvider._convert_tools(tools)
        assert result == tools

    def test_handles_missing_description(self) -> None:
        tools = [{"type": "function", "function": {"name": "t", "parameters": {"type": "object"}}}]
        result = AnthropicProvider._convert_tools(tools)
        assert result[0]["description"] == ""

    def test_handles_missing_parameters(self) -> None:
        tools = [{"type": "function", "function": {"name": "t", "description": "d"}}]
        result = AnthropicProvider._convert_tools(tools)
        assert result[0]["input_schema"] == {"type": "object", "properties": {}}


class TestAnthropicProviderRegistration:
    def test_registered_in_model_registry(self) -> None:
        from worker.models.registry import get_provider

        cls = get_provider("anthropic")
        assert cls is AnthropicProvider

    def test_get_client_raises_without_configure(self) -> None:
        p = AnthropicProvider()
        with pytest.raises(RuntimeError, match="configure.*must be called"):
            p._get_client()
