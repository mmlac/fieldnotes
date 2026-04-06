"""Tests for OllamaProvider complete and embed methods."""

from unittest.mock import patch

import httpx
import pytest

from worker.models.providers.ollama import OllamaProvider
from worker.models.base import (
    CompletionRequest,
    CompletionResponse,
    EmbedRequest,
    EmbedResponse,
)


@pytest.fixture()
def provider() -> OllamaProvider:
    p = OllamaProvider()
    # Use a pre-validated URL to avoid DNS lookups in tests
    p._base_url = "http://localhost:11434"
    return p


def _mock_response(data: dict, status_code: int = 200) -> httpx.Response:
    """Build a mock httpx.Response."""
    resp = httpx.Response(
        status_code=status_code,
        json=data,
        request=httpx.Request("POST", "http://localhost"),
    )
    return resp


class TestOllamaProviderType:
    def test_provider_type(self, provider: OllamaProvider) -> None:
        assert provider.provider_type == "ollama"


class TestOllamaProviderConfigure:
    def test_sets_timeouts(self) -> None:
        p = OllamaProvider()
        p.configure({"completion_timeout": 60, "embed_timeout": 15})
        assert p._completion_timeout == 60.0
        assert p._embed_timeout == 15.0

    def test_env_vars_override_config(self, monkeypatch) -> None:
        monkeypatch.setenv("OLLAMA_COMPLETION_TIMEOUT", "99")
        monkeypatch.setenv("OLLAMA_EMBED_TIMEOUT", "42")
        p = OllamaProvider()
        p.configure({"completion_timeout": 60, "embed_timeout": 15})
        assert p._completion_timeout == 99.0
        assert p._embed_timeout == 42.0

    def test_env_vars_override_defaults(self, monkeypatch) -> None:
        monkeypatch.setenv("OLLAMA_COMPLETION_TIMEOUT", "120")
        monkeypatch.setenv("OLLAMA_EMBED_TIMEOUT", "90")
        p = OllamaProvider()
        p.configure({})
        assert p._completion_timeout == 120.0
        assert p._embed_timeout == 90.0

    @patch(
        "worker.models.providers.ollama._validate_ollama_url",
        side_effect=lambda url: url,
    )
    def test_strips_trailing_slash(self, _mock_validate) -> None:
        p = OllamaProvider()
        p.configure({"base_url": "http://localhost:11434/"})
        assert not p._base_url.endswith("/")


class TestOllamaComplete:
    @patch("worker.models.providers.ollama.httpx.post")
    def test_basic_completion(self, mock_post, provider: OllamaProvider) -> None:
        mock_post.return_value = _mock_response(
            {
                "message": {"content": "Hello back!", "role": "assistant"},
                "prompt_eval_count": 10,
                "eval_count": 5,
            }
        )

        req = CompletionRequest(
            system="You are helpful.",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100,
            temperature=0.0,
        )
        resp = provider.complete("qwen3.5:27b", req)

        assert isinstance(resp, CompletionResponse)
        assert resp.text == "Hello back!"
        assert resp.input_tokens == 10
        assert resp.output_tokens == 5
        assert resp.tool_calls is None

    @patch("worker.models.providers.ollama.httpx.post")
    def test_sends_correct_payload(self, mock_post, provider: OllamaProvider) -> None:
        mock_post.return_value = _mock_response(
            {
                "message": {"content": "ok"},
            }
        )

        req = CompletionRequest(
            system="sys prompt",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=256,
            temperature=0.7,
        )
        provider.complete("model-x", req)

        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["model"] == "model-x"
        assert payload["stream"] is False
        assert payload["options"]["temperature"] == 0.7
        assert payload["options"]["num_predict"] == 256
        # System message should be first
        assert payload["messages"][0] == {"role": "system", "content": "sys prompt"}
        assert payload["messages"][1] == {"role": "user", "content": "hi"}

    @patch("worker.models.providers.ollama.httpx.post")
    def test_no_system_message_when_empty(
        self, mock_post, provider: OllamaProvider
    ) -> None:
        mock_post.return_value = _mock_response({"message": {"content": "ok"}})

        req = CompletionRequest(
            system="",
            messages=[{"role": "user", "content": "hi"}],
        )
        provider.complete("m", req)

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get(
            "json"
        )
        # Empty system string is falsy, so no system message
        assert all(m["role"] != "system" for m in payload["messages"])

    @patch("worker.models.providers.ollama.httpx.post")
    def test_tool_calls_returned(self, mock_post, provider: OllamaProvider) -> None:
        tool_calls = [{"function": {"name": "search", "arguments": {"q": "test"}}}]
        mock_post.return_value = _mock_response(
            {
                "message": {"content": "", "tool_calls": tool_calls},
            }
        )

        req = CompletionRequest(
            system="sys",
            messages=[{"role": "user", "content": "search for test"}],
            tools=[{"type": "function", "function": {"name": "search"}}],
        )
        resp = provider.complete("m", req)
        assert resp.tool_calls == tool_calls

    @patch("worker.models.providers.ollama.httpx.post")
    def test_tools_in_payload_when_provided(
        self, mock_post, provider: OllamaProvider
    ) -> None:
        mock_post.return_value = _mock_response({"message": {"content": "ok"}})

        tools = [{"type": "function", "function": {"name": "foo"}}]
        req = CompletionRequest(
            system="sys",
            messages=[{"role": "user", "content": "hi"}],
            tools=tools,
        )
        provider.complete("m", req)

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get(
            "json"
        )
        assert payload["tools"] == tools

    @patch("worker.models.providers.ollama.httpx.post")
    def test_no_tools_key_when_none(self, mock_post, provider: OllamaProvider) -> None:
        mock_post.return_value = _mock_response({"message": {"content": "ok"}})

        req = CompletionRequest(
            system="sys",
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
        )
        provider.complete("m", req)

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get(
            "json"
        )
        assert "tools" not in payload

    @patch("worker.models.providers.ollama.httpx.post")
    def test_posts_to_chat_endpoint(self, mock_post, provider: OllamaProvider) -> None:
        mock_post.return_value = _mock_response({"message": {"content": "ok"}})

        req = CompletionRequest(
            system="s", messages=[{"role": "user", "content": "hi"}]
        )
        provider.complete("m", req)

        url = (
            mock_post.call_args[0][0]
            if mock_post.call_args[0]
            else mock_post.call_args.kwargs.get("url", "")
        )
        assert url == "http://localhost:11434/api/chat"


class TestOllamaEmbed:
    @patch("worker.models.providers.ollama.httpx.post")
    def test_single_text_embedding(self, mock_post, provider: OllamaProvider) -> None:
        mock_post.return_value = _mock_response(
            {
                "embeddings": [[0.1, 0.2, 0.3]],
                "prompt_eval_count": 5,
            }
        )

        req = EmbedRequest(texts=["hello world"])
        resp = provider.embed("nomic-embed-text", req)

        assert isinstance(resp, EmbedResponse)
        assert len(resp.vectors) == 1
        assert resp.vectors[0] == [0.1, 0.2, 0.3]
        assert resp.model == "nomic-embed-text"
        assert resp.input_tokens == 5

    @patch("worker.models.providers.ollama.httpx.post")
    def test_multiple_texts_embedding(
        self, mock_post, provider: OllamaProvider
    ) -> None:
        mock_post.return_value = _mock_response(
            {
                "embeddings": [[0.1, 0.2], [0.3, 0.4]],
                "prompt_eval_count": 7,
            }
        )

        req = EmbedRequest(texts=["hello", "world"])
        resp = provider.embed("model", req)

        assert len(resp.vectors) == 2
        assert resp.vectors[0] == [0.1, 0.2]
        assert resp.vectors[1] == [0.3, 0.4]
        assert resp.input_tokens == 7
        assert mock_post.call_count == 1

    @patch("worker.models.providers.ollama.httpx.post")
    def test_posts_to_embed_endpoint(self, mock_post, provider: OllamaProvider) -> None:
        mock_post.return_value = _mock_response(
            {
                "embeddings": [[0.1]],
                "prompt_eval_count": 1,
            }
        )

        req = EmbedRequest(texts=["test"])
        provider.embed("m", req)

        url = (
            mock_post.call_args[0][0]
            if mock_post.call_args[0]
            else mock_post.call_args.kwargs.get("url", "")
        )
        assert url == "http://localhost:11434/api/embed"

    @patch("worker.models.providers.ollama.httpx.post")
    def test_embed_payload_format(self, mock_post, provider: OllamaProvider) -> None:
        mock_post.return_value = _mock_response(
            {
                "embeddings": [[0.1]],
                "prompt_eval_count": 1,
            }
        )

        req = EmbedRequest(texts=["test text"])
        provider.embed("nomic", req)

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get(
            "json"
        )
        assert payload["model"] == "nomic"
        assert payload["input"] == ["test text"]


class TestOllamaProviderRegistration:
    def test_registered_in_model_registry(self) -> None:
        from worker.models.registry import get_provider

        cls = get_provider("ollama")
        assert cls is OllamaProvider
