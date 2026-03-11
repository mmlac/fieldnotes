"""OllamaProvider — calls a local Ollama HTTP API for completions and embeddings."""

import ipaddress
import logging
import socket
from urllib.parse import urlparse
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

# Cloud metadata and link-local CIDRs that must never be reached via base_url.
_BLOCKED_NETWORKS = [
    ipaddress.ip_network("169.254.0.0/16"),   # AWS/GCP metadata & link-local
    ipaddress.ip_network("fe80::/10"),         # IPv6 link-local
]

_ALLOWED_SCHEMES = {"http", "https"}


def _validate_ollama_url(url: str) -> str:
    """Validate and return *url* or raise ``ValueError``.

    Rules:
    * Scheme must be http or https.
    * Host must not resolve to a cloud-metadata or link-local address.
    """
    parsed = urlparse(url)

    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(
            f"Ollama base_url must use http or https (got {parsed.scheme!r})"
        )

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("Ollama base_url has no hostname")

    # Resolve hostname to detect internal IPs
    try:
        addrinfos = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise ValueError(f"Cannot resolve Ollama base_url host {hostname!r}: {exc}") from exc

    for _family, _type, _proto, _canonname, sockaddr in addrinfos:
        addr = ipaddress.ip_address(sockaddr[0])
        for net in _BLOCKED_NETWORKS:
            if addr in net:
                raise ValueError(
                    f"Ollama base_url resolves to blocked address {addr} "
                    f"(in {net})"
                )

    return url


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
        raw_url = cfg.get("base_url", self._base_url).rstrip("/")
        self._base_url = _validate_ollama_url(raw_url)
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
                "num_predict": req.max_tokens,
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
