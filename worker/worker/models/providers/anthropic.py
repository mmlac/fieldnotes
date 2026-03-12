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

from collections.abc import Iterator

from worker.log_sanitizer import sanitize_exception

from ..base import (
    CompletionRequest,
    CompletionResponse,
    ModelProvider,
    StreamChunk,
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
        "Anthropic call failed (%s), retry %d",
        sanitize_exception(rs.outcome.exception()),
        rs.attempt_number,
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

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI-format tool definitions to Anthropic format.

        OpenAI format: {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
        Anthropic format: {"name": ..., "description": ..., "input_schema": ...}
        """
        converted = []
        for tool in tools:
            if "function" in tool:
                func = tool["function"]
                converted.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                })
            else:
                # Already in Anthropic format or unknown — pass through
                converted.append(tool)
        return converted

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
            kwargs["tools"] = self._convert_tools(req.tools)

        if req.timeout is not None:
            kwargs["timeout"] = req.timeout

        response = client.messages.create(**kwargs)

        # Extract text and tool_calls from content blocks.
        # Normalise to OpenAI-style {"function": {"name": ..., "arguments": ...}}
        # so downstream parsers work uniformly across providers.
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({
                    "function": {
                        "name": block.name,
                        "arguments": block.input,
                    },
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

    def stream_complete(self, model: str, req: CompletionRequest) -> Iterator[StreamChunk]:
        client = self._get_client()

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "messages": req.messages,
        }
        if req.system:
            kwargs["system"] = req.system
        if req.timeout is not None:
            kwargs["timeout"] = req.timeout

        with client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield StreamChunk(text=text)

            message = stream.get_final_message()
            usage = message.usage
            yield StreamChunk(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                done=True,
            )
