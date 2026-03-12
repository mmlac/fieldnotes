"""ModelRegistry — three-layer config resolution (providers → models → roles).

Single entry point for the pipeline to access any model. Resolution layers:
  1. [modelproviders.*] → configured provider instances
  2. [models.*] → alias → ResolvedModel (skipping 'roles' key)
  3. [models.roles] → role → ResolvedModel
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass

from .base import (
    CompletionRequest,
    CompletionResponse,
    EmbedRequest,
    EmbedResponse,
    ModelProvider,
    StreamChunk,
)
from .registry import get_provider
from ..config import Config
from ..metrics import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DURATION,
    LLM_ERRORS,
    LLM_REQUEST_DURATION,
    LLM_TOKENS,
)

logger = logging.getLogger(__name__)


@dataclass
class ResolvedModel:
    """A fully resolved model: alias + model string + configured provider instance."""

    alias: str
    model: str
    provider: ModelProvider

    def complete(self, req: CompletionRequest, *, task: str = "unknown") -> CompletionResponse:
        """Run a completion with latency, token, and error tracking."""
        start = time.monotonic()
        try:
            resp = self.provider.complete(self.model, req)
        except Exception as exc:
            elapsed = time.monotonic() - start
            LLM_REQUEST_DURATION.labels(model=self.model, task=task).observe(elapsed)
            LLM_ERRORS.labels(
                model=self.model, task=task, error_type=type(exc).__name__,
            ).inc()
            raise
        elapsed = time.monotonic() - start
        LLM_REQUEST_DURATION.labels(model=self.model, task=task).observe(elapsed)
        if resp.input_tokens or resp.output_tokens:
            LLM_TOKENS.labels(model=self.model, task=task, direction="input").inc(resp.input_tokens)
            LLM_TOKENS.labels(model=self.model, task=task, direction="output").inc(resp.output_tokens)
        return resp

    def stream_complete(self, req: CompletionRequest, *, task: str = "unknown") -> Iterator[StreamChunk]:
        """Stream a completion with metrics tracking on the final chunk."""
        start = time.monotonic()
        try:
            for chunk in self.provider.stream_complete(self.model, req):
                if chunk.done:
                    elapsed = time.monotonic() - start
                    LLM_REQUEST_DURATION.labels(model=self.model, task=task).observe(elapsed)
                    if chunk.input_tokens or chunk.output_tokens:
                        LLM_TOKENS.labels(model=self.model, task=task, direction="input").inc(chunk.input_tokens)
                        LLM_TOKENS.labels(model=self.model, task=task, direction="output").inc(chunk.output_tokens)
                yield chunk
        except Exception as exc:
            elapsed = time.monotonic() - start
            LLM_REQUEST_DURATION.labels(model=self.model, task=task).observe(elapsed)
            LLM_ERRORS.labels(
                model=self.model, task=task, error_type=type(exc).__name__,
            ).inc()
            raise

    def embed(self, req: EmbedRequest, *, task: str = "embed") -> EmbedResponse:
        """Run an embedding call with latency, token, and error tracking."""
        start = time.monotonic()
        try:
            resp = self.provider.embed(self.model, req)
        except Exception as exc:
            elapsed = time.monotonic() - start
            EMBEDDING_DURATION.labels(model=self.model).observe(elapsed)
            LLM_ERRORS.labels(
                model=self.model, task=task, error_type=type(exc).__name__,
            ).inc()
            raise
        elapsed = time.monotonic() - start
        EMBEDDING_DURATION.labels(model=self.model).observe(elapsed)
        EMBEDDING_BATCH_SIZE.labels(model=self.model).observe(len(req.texts))
        if resp.input_tokens:
            LLM_TOKENS.labels(model=self.model, task=task, direction="input").inc(resp.input_tokens)
        return resp


class ModelRegistry:
    """Resolves parsed config into ready-to-use model handles.

    Three-layer resolution:
      1. [modelproviders.*] → provider instances (configured via .configure())
      2. [models.*] → alias → ResolvedModel
      3. [models.roles] → role → ResolvedModel
    """

    def __init__(self, cfg: Config) -> None:
        # Layer 1: build configured provider instances
        providers: dict[str, ModelProvider] = {}
        for name, pcfg in cfg.providers.items():
            cls = get_provider(pcfg.type)
            instance = cls()
            instance.configure(pcfg.settings)
            providers[name] = instance

        # Layer 2: alias → ResolvedModel
        self._by_alias: dict[str, ResolvedModel] = {}
        for alias, mcfg in cfg.models.items():
            provider = providers.get(mcfg.provider)
            if provider is None:
                raise ValueError(
                    f"Model {alias!r} references provider {mcfg.provider!r}, "
                    f"but no such provider is configured. "
                    f"Known providers: {sorted(providers)}"
                )
            self._by_alias[alias] = ResolvedModel(
                alias=alias,
                model=mcfg.model,
                provider=provider,
            )

        # Layer 3: role → ResolvedModel
        self._by_role: dict[str, ResolvedModel] = {}
        for role, alias in cfg.roles.mapping.items():
            resolved = self._by_alias.get(alias)
            if resolved is None:
                raise ValueError(
                    f"Role {role!r} maps to alias {alias!r}, "
                    f"but no such model is defined. "
                    f"Known aliases: {sorted(self._by_alias)}"
                )
            self._by_role[role] = resolved

    def for_role(self, role: str) -> ResolvedModel:
        """Look up a model by pipeline role (e.g. 'extraction', 'embedding')."""
        resolved = self._by_role.get(role)
        if resolved is None:
            raise KeyError(
                f"No model configured for role {role!r}. "
                f"Known roles: {sorted(self._by_role)}"
            )
        return resolved

    def by_alias(self, alias: str) -> ResolvedModel:
        """Look up a model by its config alias."""
        resolved = self._by_alias.get(alias)
        if resolved is None:
            raise KeyError(
                f"No model configured with alias {alias!r}. "
                f"Known aliases: {sorted(self._by_alias)}"
            )
        return resolved
