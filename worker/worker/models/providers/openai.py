"""OpenAIProvider — calls the OpenAI API for chat completions and embeddings."""

import json
import logging
import os
from typing import Any

import openai
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from collections.abc import Iterator

from ..base import (
    CompletionRequest,
    CompletionResponse,
    EmbedRequest,
    EmbedResponse,
    ModelProvider,
    StreamChunk,
)
from ..registry import register

logger = logging.getLogger(__name__)

_openai_retry = retry(
    retry=retry_if_exception_type((openai.APITimeoutError, openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError)),
    stop=stop_after_attempt(4),
    wait=wait_exponential_jitter(initial=0.5, max=10),
    before_sleep=lambda rs: logger.warning(
        "OpenAI call failed (%s), retry %d", rs.outcome.exception(), rs.attempt_number
    ),
    reraise=True,
)


@register
class OpenAIProvider(ModelProvider):
    """Model provider that talks to the OpenAI API."""

    _api_key: str | None = None
    _client: openai.OpenAI | None = None
    _completion_timeout: float = 120.0
    _embed_timeout: float = 30.0

    @property
    def provider_type(self) -> str:
        return "openai"

    def configure(self, cfg: dict[str, Any]) -> None:
        self._api_key = cfg.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "OpenAI provider requires an API key: set 'api_key' in config "
                "or the OPENAI_API_KEY environment variable"
            )
        self._completion_timeout = float(cfg.get("completion_timeout", self._completion_timeout))
        self._embed_timeout = float(cfg.get("embed_timeout", self._embed_timeout))
        self._client = openai.OpenAI(api_key=self._api_key)

    def _get_client(self) -> openai.OpenAI:
        if self._client is None:
            raise RuntimeError("OpenAIProvider not configured — call configure() first")
        return self._client

    @_openai_retry
    def complete(self, model: str, req: CompletionRequest) -> CompletionResponse:
        client = self._get_client()

        messages: list[dict[str, Any]] = []
        if req.system:
            messages.append({"role": "system", "content": req.system})
        messages.extend(req.messages)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "timeout": req.timeout if req.timeout is not None else self._completion_timeout,
        }
        if req.tools:
            kwargs["tools"] = req.tools

        response = client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        tool_calls = None
        if choice.message.tool_calls:
            parsed: list[dict[str, Any]] = []
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    logger.warning(
                        "Malformed tool-call JSON from model (name=%s), using raw string",
                        tc.function.name,
                    )
                    args = tc.function.arguments
                parsed.append({"function": {"name": tc.function.name, "arguments": args}})
            tool_calls = parsed

        usage = response.usage
        return CompletionResponse(
            text=choice.message.content or "",
            tool_calls=tool_calls,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            cached_tokens=getattr(usage, "prompt_tokens_details", None)
            and getattr(usage.prompt_tokens_details, "cached_tokens", 0)
            or 0,
        )

    def stream_complete(self, model: str, req: CompletionRequest) -> Iterator[StreamChunk]:
        client = self._get_client()

        messages: list[dict[str, Any]] = []
        if req.system:
            messages.append({"role": "system", "content": req.system})
        messages.extend(req.messages)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "timeout": req.timeout if req.timeout is not None else self._completion_timeout,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        stream = client.chat.completions.create(**kwargs)
        for event in stream:
            choice = event.choices[0] if event.choices else None
            if choice and choice.delta and choice.delta.content:
                yield StreamChunk(text=choice.delta.content)
            # Final chunk with usage stats
            if event.usage:
                yield StreamChunk(
                    input_tokens=event.usage.prompt_tokens,
                    output_tokens=event.usage.completion_tokens,
                    done=True,
                )

    @_openai_retry
    def embed(self, model: str, req: EmbedRequest) -> EmbedResponse:
        client = self._get_client()

        response = client.embeddings.create(
            model=model,
            input=req.texts,
            timeout=req.timeout if req.timeout is not None else self._embed_timeout,
        )

        vectors = [item.embedding for item in response.data]
        input_tokens = response.usage.total_tokens if response.usage else 0

        return EmbedResponse(
            vectors=vectors,
            model=model,
            input_tokens=input_tokens,
        )
