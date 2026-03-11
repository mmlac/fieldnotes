"""Tests for OpenAIProvider complete and embed methods."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from worker.models.providers.openai import OpenAIProvider
from worker.models.base import (
    CompletionRequest,
    CompletionResponse,
    EmbedRequest,
    EmbedResponse,
)


@pytest.fixture()
def provider() -> OpenAIProvider:
    p = OpenAIProvider()
    p._api_key = "test-key"
    p._client = MagicMock()
    return p


def _make_completion_response(
    content: str = "Hello!",
    tool_calls=None,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
):
    """Build a mock OpenAI ChatCompletion response."""
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    return SimpleNamespace(choices=[choice], usage=usage)


def _make_embedding_response(
    embeddings: list[list[float]],
    total_tokens: int = 10,
):
    """Build a mock OpenAI Embedding response."""
    data = [SimpleNamespace(embedding=vec) for vec in embeddings]
    usage = SimpleNamespace(total_tokens=total_tokens)
    return SimpleNamespace(data=data, usage=usage)


class TestOpenAIProviderType:
    def test_provider_type(self, provider: OpenAIProvider) -> None:
        assert provider.provider_type == "openai"


class TestOpenAIProviderConfigure:
    def test_configure_with_api_key(self) -> None:
        p = OpenAIProvider()
        p.configure({"api_key": "sk-test123"})
        assert p._api_key == "sk-test123"
        assert p._client is not None

    def test_configure_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        p = OpenAIProvider()
        p.configure({})
        assert p._api_key == "sk-from-env"

    def test_configure_raises_without_key(self, monkeypatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        p = OpenAIProvider()
        with pytest.raises(ValueError, match="API key"):
            p.configure({})

    def test_configure_sets_timeouts(self) -> None:
        p = OpenAIProvider()
        p.configure({"api_key": "sk-test", "completion_timeout": 60, "embed_timeout": 15})
        assert p._completion_timeout == 60.0
        assert p._embed_timeout == 15.0


class TestOpenAIComplete:
    def test_basic_completion(self, provider: OpenAIProvider) -> None:
        provider._client.chat.completions.create.return_value = _make_completion_response(
            content="Hello back!", prompt_tokens=10, completion_tokens=5,
        )

        req = CompletionRequest(
            system="You are helpful.",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100,
            temperature=0.0,
        )
        resp = provider.complete("gpt-4o", req)

        assert isinstance(resp, CompletionResponse)
        assert resp.text == "Hello back!"
        assert resp.input_tokens == 10
        assert resp.output_tokens == 5
        assert resp.tool_calls is None

    def test_sends_correct_kwargs(self, provider: OpenAIProvider) -> None:
        provider._client.chat.completions.create.return_value = _make_completion_response()

        req = CompletionRequest(
            system="sys prompt",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=256,
            temperature=0.7,
        )
        provider.complete("gpt-4o", req)

        call_kwargs = provider._client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["max_tokens"] == 256
        assert call_kwargs["temperature"] == 0.7
        messages = call_kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "sys prompt"}
        assert messages[1] == {"role": "user", "content": "hi"}

    def test_no_system_message_when_empty(self, provider: OpenAIProvider) -> None:
        provider._client.chat.completions.create.return_value = _make_completion_response()

        req = CompletionRequest(
            system="",
            messages=[{"role": "user", "content": "hi"}],
        )
        provider.complete("m", req)

        call_kwargs = provider._client.chat.completions.create.call_args.kwargs
        assert all(m["role"] != "system" for m in call_kwargs["messages"])

    def test_tool_calls_returned(self, provider: OpenAIProvider) -> None:
        tc = SimpleNamespace(
            function=SimpleNamespace(
                name="search",
                arguments=json.dumps({"q": "test"}),
            )
        )
        provider._client.chat.completions.create.return_value = _make_completion_response(
            content="", tool_calls=[tc],
        )

        req = CompletionRequest(
            system="sys",
            messages=[{"role": "user", "content": "search for test"}],
            tools=[{"type": "function", "function": {"name": "search"}}],
        )
        resp = provider.complete("m", req)
        assert resp.tool_calls == [{"function": {"name": "search", "arguments": {"q": "test"}}}]

    def test_malformed_tool_call_json_uses_raw_string(self, provider: OpenAIProvider) -> None:
        tc = SimpleNamespace(
            function=SimpleNamespace(
                name="search",
                arguments="{bad json",
            )
        )
        provider._client.chat.completions.create.return_value = _make_completion_response(
            content="", tool_calls=[tc],
        )

        req = CompletionRequest(
            system="sys",
            messages=[{"role": "user", "content": "search"}],
            tools=[{"type": "function", "function": {"name": "search"}}],
        )
        resp = provider.complete("m", req)
        assert resp.tool_calls == [{"function": {"name": "search", "arguments": "{bad json"}}]

    def test_tools_in_kwargs_when_provided(self, provider: OpenAIProvider) -> None:
        provider._client.chat.completions.create.return_value = _make_completion_response()

        tools = [{"type": "function", "function": {"name": "foo"}}]
        req = CompletionRequest(
            system="sys",
            messages=[{"role": "user", "content": "hi"}],
            tools=tools,
        )
        provider.complete("m", req)

        call_kwargs = provider._client.chat.completions.create.call_args.kwargs
        assert call_kwargs["tools"] == tools

    def test_no_tools_key_when_none(self, provider: OpenAIProvider) -> None:
        provider._client.chat.completions.create.return_value = _make_completion_response()

        req = CompletionRequest(
            system="sys",
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
        )
        provider.complete("m", req)

        call_kwargs = provider._client.chat.completions.create.call_args.kwargs
        assert "tools" not in call_kwargs

    def test_unconfigured_raises(self) -> None:
        p = OpenAIProvider()
        req = CompletionRequest(system="s", messages=[{"role": "user", "content": "hi"}])
        with pytest.raises(RuntimeError, match="not configured"):
            p.complete("m", req)


class TestOpenAIEmbed:
    def test_single_text_embedding(self, provider: OpenAIProvider) -> None:
        provider._client.embeddings.create.return_value = _make_embedding_response(
            embeddings=[[0.1, 0.2, 0.3]], total_tokens=5,
        )

        req = EmbedRequest(texts=["hello world"])
        resp = provider.embed("text-embedding-3-small", req)

        assert isinstance(resp, EmbedResponse)
        assert len(resp.vectors) == 1
        assert resp.vectors[0] == [0.1, 0.2, 0.3]
        assert resp.model == "text-embedding-3-small"
        assert resp.input_tokens == 5

    def test_multiple_texts_embedding(self, provider: OpenAIProvider) -> None:
        provider._client.embeddings.create.return_value = _make_embedding_response(
            embeddings=[[0.1, 0.2], [0.3, 0.4]], total_tokens=7,
        )

        req = EmbedRequest(texts=["hello", "world"])
        resp = provider.embed("model", req)

        assert len(resp.vectors) == 2
        assert resp.vectors[0] == [0.1, 0.2]
        assert resp.vectors[1] == [0.3, 0.4]
        assert resp.input_tokens == 7

    def test_embed_passes_texts_as_input(self, provider: OpenAIProvider) -> None:
        provider._client.embeddings.create.return_value = _make_embedding_response(
            embeddings=[[0.1]], total_tokens=1,
        )

        req = EmbedRequest(texts=["test text"])
        provider.embed("model", req)

        call_kwargs = provider._client.embeddings.create.call_args.kwargs
        assert call_kwargs["model"] == "model"
        assert call_kwargs["input"] == ["test text"]


class TestOpenAIProviderRegistration:
    def test_registered_in_model_registry(self) -> None:
        from worker.models.registry import get_provider

        cls = get_provider("openai")
        assert cls is OpenAIProvider
