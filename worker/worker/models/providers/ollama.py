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

from collections.abc import Iterator

from worker.log_sanitizer import sanitize_exception

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

# Private, reserved, and internal CIDRs that must never be reached via
# user-supplied base_url.  The built-in default (localhost:11434) bypasses
# validation, so blocking loopback here does not break normal usage.
_BLOCKED_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),  # RFC 1918
    ipaddress.ip_network("172.16.0.0/12"),  # RFC 1918
    ipaddress.ip_network("192.168.0.0/16"),  # RFC 1918
    ipaddress.ip_network("127.0.0.0/8"),  # Loopback
    ipaddress.ip_network("169.254.0.0/16"),  # Link-local / cloud metadata
    ipaddress.ip_network("100.64.0.0/10"),  # CGNAT (RFC 6598)
    ipaddress.ip_network("::1/128"),  # IPv6 loopback
    ipaddress.ip_network("::ffff:0:0/96"),  # IPv4-mapped IPv6 (RFC 4291 §2.5.5.2)
    ipaddress.ip_network("fe80::/10"),  # IPv6 link-local
    ipaddress.ip_network("fd00::/8"),  # IPv6 ULA
]

_ALLOWED_SCHEMES = {"http", "https"}

_DEFAULT_BASE_URL = "http://localhost:11434"


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
        raise ValueError(
            f"Cannot resolve Ollama base_url host {hostname!r}: {exc}"
        ) from exc

    for _family, _type, _proto, _canonname, sockaddr in addrinfos:
        addr = ipaddress.ip_address(sockaddr[0])
        for net in _BLOCKED_NETWORKS:
            if addr in net:
                raise ValueError(
                    f"Ollama base_url resolves to blocked address {addr} (in {net})"
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
        "Ollama call failed (%s), retry %d",
        sanitize_exception(rs.outcome.exception()),
        rs.attempt_number,
    ),
    reraise=True,
)


@register
class OllamaProvider(ModelProvider):
    """Model provider that talks to a local Ollama instance."""

    _base_url: str = "http://localhost:11434"
    _completion_timeout: float = 600.0
    _embed_timeout: float = 30.0

    @property
    def provider_type(self) -> str:
        return "ollama"

    def configure(self, cfg: dict[str, Any]) -> None:
        raw_url = cfg.get("base_url")
        if raw_url is not None:
            normalized = raw_url.rstrip("/")
            if normalized == _DEFAULT_BASE_URL:
                # Explicitly supplying the default localhost URL is always
                # trusted — no SSRF risk, and blocking it would prevent
                # the most common Ollama configuration pattern.
                self._base_url = normalized
            else:
                self._base_url = _validate_ollama_url(normalized)
        # When no base_url is provided the built-in default
        # (http://localhost:11434) is trusted and skips validation.
        self._completion_timeout = float(
            cfg.get("completion_timeout", self._completion_timeout)
        )
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
            timeout=req.timeout
            if req.timeout is not None
            else self._completion_timeout,
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

    def stream_complete(
        self, model: str, req: CompletionRequest
    ) -> Iterator[StreamChunk]:
        messages: list[dict[str, Any]] = []
        if req.system:
            messages.append({"role": "system", "content": req.system})
        messages.extend(req.messages)

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": req.temperature,
                "num_predict": req.max_tokens,
            },
        }

        timeout = req.timeout if req.timeout is not None else self._completion_timeout
        with httpx.stream(
            "POST",
            f"{self._base_url}/api/chat",
            json=payload,
            timeout=timeout,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                import json as _json

                data = _json.loads(line)
                message = data.get("message", {})
                token = message.get("content", "")
                done = data.get("done", False)
                chunk = StreamChunk(text=token, done=done)
                if done:
                    chunk.input_tokens = data.get("prompt_eval_count", 0)
                    chunk.output_tokens = data.get("eval_count", 0)
                yield chunk

    @_ollama_retry
    def embed(self, model: str, req: EmbedRequest) -> EmbedResponse:
        resp = httpx.post(
            f"{self._base_url}/api/embed",
            json={"model": model, "input": req.texts},
            timeout=req.timeout if req.timeout is not None else self._embed_timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        return EmbedResponse(
            vectors=data["embeddings"],
            model=model,
            input_tokens=data.get("prompt_eval_count", 0),
        )
