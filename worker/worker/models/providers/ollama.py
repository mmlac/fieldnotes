"""OllamaProvider — calls a local Ollama HTTP API for completions and embeddings."""

from typing import Any

import httpx

from ..base import (
    CompletionRequest,
    CompletionResponse,
    EmbedRequest,
    EmbedResponse,
    ModelProvider,
)
from ..registry import register


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
