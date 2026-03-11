"""OllamaProvider — calls a local Ollama HTTP API for completions and embeddings."""

import logging
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from ..base import (
    CompletionRequest,
    CompletionResponse,
    EmbedRequest,
    EmbedResponse,
    ModelProvider,
)
from ..registry import register

logger = logging.getLogger(__name__)


def _is_retryable_httpx(exc: BaseException) -> bool:
    """Return True for transient httpx errors worth retrying."""
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code >= 500:
        return True
    return False


_ollama_retry = retry(
    retry=retry_if_exception(_is_retryable_httpx),
    stop=stop_after_attempt(4),
    wait=wait_exponential_jitter(initial=0.5, max=10),
    before_sleep=lambda rs: logger.warning(
        "Ollama call failed (%s), retry %d", rs.outcome.exception(), rs.attempt_number
    ),
    reraise=True,
)


@register
class OllamaProvider(ModelProvider):
    """Model provider that talks to a local Ollama instance."""

    _base_url: str = "http://localhost:11434"
    _completion_timeout: float = 120.0
    _embed_timeout: float = 30.0

    @property
    def provider_type(self) -> str:
        return "ollama"

    def configure(self, cfg: dict[str, Any]) -> None:
        self._base_url = cfg.get("base_url", self._base_url).rstrip("/")
        self._completion_timeout = float(cfg.get("completion_timeout", self._completion_timeout))
        self._embed_timeout = float(cfg.get("embed_timeout", self._embed_timeout))

    @_ollama_retry
    def complete(self, model: str, req: CompletionRequest) -> CompletionResponse:
        messages: list[dict[str, Any]] = []
        if req.system:
            messages.append({"role": "system", "content": req.system})
        messages.extend(req.messages)

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": req.temperature,
            },
        }
        if req.tools:
            payload["tools"] = req.tools

        resp = httpx.post(
            f"{self._base_url}/api/chat",
            json=payload,
            timeout=self._completion_timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        message = data.get("message", {})
        tool_calls = message.get("tool_calls") or None

        return CompletionResponse(
            text=message.get("content", ""),
            tool_calls=tool_calls,
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
        )

    @_ollama_retry
    def embed(self, model: str, req: EmbedRequest) -> EmbedResponse:
        vectors: list[list[float]] = []
        total_tokens = 0

        for text in req.texts:
            resp = httpx.post(
                f"{self._base_url}/api/embeddings",
                json={"model": model, "prompt": text},
                timeout=self._embed_timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            vectors.append(data["embedding"])
            total_tokens += data.get("prompt_eval_count", 0)

        return EmbedResponse(
            vectors=vectors,
            model=model,
            input_tokens=total_tokens,
        )
