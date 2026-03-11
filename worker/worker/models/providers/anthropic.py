"""AnthropicProvider — calls the Anthropic Messages API for completions."""

import logging
import os
from typing import Any

import anthropic
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from ..base import (
    CompletionRequest,
    CompletionResponse,
    ModelProvider,
)
from ..registry import register

logger = logging.getLogger(__name__)

_anthropic_retry = retry(
    retry=retry_if_exception_type((
        anthropic.APIConnectionError,
        anthropic.RateLimitError,
        anthropic.InternalServerError,
    )),
    stop=stop_after_attempt(4),
    wait=wait_exponential_jitter(initial=0.5, max=10),
    before_sleep=lambda rs: logger.warning(
        "Anthropic call failed (%s), retry %d", rs.outcome.exception(), rs.attempt_number
    ),
    reraise=True,
)


@register
class AnthropicProvider(ModelProvider):
    """Model provider that talks to the Anthropic Messages API."""

    _client: anthropic.Anthropic | None = None
    _api_key: str | None = None

    @property
    def provider_type(self) -> str:
        return "anthropic"

    def configure(self, cfg: dict[str, Any]) -> None:
        self._api_key = cfg.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Anthropic API key required: set api_key in config or ANTHROPIC_API_KEY env var"
            )
        self._client = anthropic.Anthropic(api_key=self._api_key)

    def _get_client(self) -> anthropic.Anthropic:
        if self._client is None:
            raise RuntimeError("AnthropicProvider.configure() must be called before use")
        return self._client

    @_anthropic_retry
    def complete(self, model: str, req: CompletionRequest) -> CompletionResponse:
        client = self._get_client()

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "messages": req.messages,
        }
        if req.system:
            kwargs["system"] = req.system
        if req.tools:
            kwargs["tools"] = req.tools

        response = client.messages.create(**kwargs)

        # Extract text and tool_calls from content blocks
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "type": "tool_use",
                    "name": block.name,
                    "input": block.input,
                })

        usage = response.usage
        cached_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0

        return CompletionResponse(
            text="\n".join(text_parts),
            tool_calls=tool_calls or None,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached_tokens=cached_tokens,
        )
