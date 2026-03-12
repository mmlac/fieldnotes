"""ModelRegistry — three-layer config resolution (providers → models → roles).

Single entry point for the pipeline to access any model. Resolution layers:
  1. [modelproviders.*] → configured provider instances
  2. [models.*] → alias → ResolvedModel (skipping 'roles' key)
  3. [models.roles] → role → ResolvedModel
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from .base import (
    CompletionRequest,
    CompletionResponse,
    EmbedRequest,
    EmbedResponse,
    ModelProvider,
    StreamChunk,
)
from .rate_limiter import (
    ConcurrencyLimiter,
    RateLimiter,
    RateLimitExceeded,
    TokenBudget,
    TokenBudgetExhausted,
)
from .registry import get_provider
from ..circuit_breaker import CircuitBreaker, CircuitOpenError, get_breaker
from ..config import Config
from ..metrics import (
    CIRCUIT_BREAKER_REJECTIONS,
    CONCURRENCY_LIMIT_WAITS,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DURATION,
    LLM_ERRORS,
    LLM_REQUEST_DURATION,
    LLM_TOKENS,
    RATE_LIMIT_REJECTIONS,
    RATE_LIMIT_WAITS,
    TOKEN_BUDGET_REJECTIONS,
    TOKEN_BUDGET_USED,
)

logger = logging.getLogger(__name__)

# LLM circuit breaker: one breaker per provider type (e.g. "llm:openai",
# "llm:anthropic", "llm:ollama").  Created lazily on first use.
_llm_breaker_lock = threading.Lock()


def _llm_breaker(provider_type: str) -> CircuitBreaker:
    """Get or create a circuit breaker for the given LLM provider type."""
    name = f"llm:{provider_type}"
    breaker = get_breaker(name)
    if breaker is not None:
        return breaker
    with _llm_breaker_lock:
        # Double-check after acquiring lock
        breaker = get_breaker(name)
        if breaker is not None:
            return breaker
        return CircuitBreaker(
            name,
            failure_threshold=5,
            recovery_timeout=120.0,
        )


@dataclass
class ResolvedModel:
    """A fully resolved model: alias + model string + configured provider instance.

    Rate limiting, token budgets, and concurrency limits are optional and
    injected by :class:`ModelRegistry` from the ``[rate_limits]`` config.
    """

    alias: str
    model: str
    provider: ModelProvider
    _rate_limiter: RateLimiter | None = field(default=None, repr=False)
    _token_budget: TokenBudget | None = field(default=None, repr=False)
    _concurrency_limiter: ConcurrencyLimiter | None = field(default=None, repr=False)

    # -- internal helpers ---------------------------------------------------

    def _pre_request(self, *, task: str) -> None:
        """Run pre-request gates: token budget check, rate limit, concurrency."""
        if self._token_budget is not None:
            try:
                self._token_budget.check()
            except TokenBudgetExhausted:
                TOKEN_BUDGET_REJECTIONS.inc()
                raise
        if self._rate_limiter is not None:
            t0 = time.monotonic()
            try:
                self._rate_limiter.acquire()
            except RateLimitExceeded:
                RATE_LIMIT_REJECTIONS.labels(
                    provider=self.provider.provider_type,
                ).inc()
                raise
            waited = time.monotonic() - t0
            if waited > 0.05:
                RATE_LIMIT_WAITS.labels(
                    provider=self.provider.provider_type,
                ).inc()
        if self._concurrency_limiter is not None:
            t0 = time.monotonic()
            ok = self._concurrency_limiter.acquire()
            if not ok:
                raise RateLimitExceeded(
                    "Timed out waiting for concurrency slot"
                )
            waited = time.monotonic() - t0
            if waited > 0.05:
                CONCURRENCY_LIMIT_WAITS.inc()

    def _post_request(
        self,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Run post-request accounting: release concurrency, record tokens."""
        if self._concurrency_limiter is not None:
            self._concurrency_limiter.release()
        if self._token_budget is not None:
            self._token_budget.record(input_tokens, output_tokens)
            TOKEN_BUDGET_USED.set(self._token_budget.used)

    def _release_concurrency(self) -> None:
        """Release the concurrency slot on error paths."""
        if self._concurrency_limiter is not None:
            self._concurrency_limiter.release()

    # -- public API ---------------------------------------------------------

    def complete(self, req: CompletionRequest, *, task: str = "unknown") -> CompletionResponse:
        """Run a completion with latency, token, error, and circuit breaker tracking."""
        breaker = _llm_breaker(self.provider.provider_type)
        if not breaker.allow_request():
            CIRCUIT_BREAKER_REJECTIONS.labels(service=breaker.name).inc()
            raise CircuitOpenError(breaker.name)

        self._pre_request(task=task)
        start = time.monotonic()
        try:
            resp = self.provider.complete(self.model, req)
        except CircuitOpenError:
            self._release_concurrency()
            raise
        except Exception as exc:
            self._release_concurrency()
            elapsed = time.monotonic() - start
            LLM_REQUEST_DURATION.labels(model=self.model, task=task).observe(elapsed)
            LLM_ERRORS.labels(
                model=self.model, task=task, error_type=type(exc).__name__,
            ).inc()
            breaker.record_failure()
            raise
        elapsed = time.monotonic() - start
        LLM_REQUEST_DURATION.labels(model=self.model, task=task).observe(elapsed)
        if resp.input_tokens or resp.output_tokens:
            LLM_TOKENS.labels(model=self.model, task=task, direction="input").inc(resp.input_tokens)
            LLM_TOKENS.labels(model=self.model, task=task, direction="output").inc(resp.output_tokens)
        breaker.record_success()
        self._post_request(resp.input_tokens, resp.output_tokens)
        return resp

    def stream_complete(self, req: CompletionRequest, *, task: str = "unknown") -> Iterator[StreamChunk]:
        """Stream a completion with metrics tracking on the final chunk."""
        breaker = _llm_breaker(self.provider.provider_type)
        if not breaker.allow_request():
            CIRCUIT_BREAKER_REJECTIONS.labels(service=breaker.name).inc()
            raise CircuitOpenError(breaker.name)

        self._pre_request(task=task)
        start = time.monotonic()
        try:
            for chunk in self.provider.stream_complete(self.model, req):
                if chunk.done:
                    elapsed = time.monotonic() - start
                    LLM_REQUEST_DURATION.labels(model=self.model, task=task).observe(elapsed)
                    if chunk.input_tokens or chunk.output_tokens:
                        LLM_TOKENS.labels(model=self.model, task=task, direction="input").inc(chunk.input_tokens)
                        LLM_TOKENS.labels(model=self.model, task=task, direction="output").inc(chunk.output_tokens)
                    breaker.record_success()
                    self._post_request(chunk.input_tokens, chunk.output_tokens)
                yield chunk
        except CircuitOpenError:
            self._release_concurrency()
            raise
        except Exception as exc:
            self._release_concurrency()
            elapsed = time.monotonic() - start
            LLM_REQUEST_DURATION.labels(model=self.model, task=task).observe(elapsed)
            LLM_ERRORS.labels(
                model=self.model, task=task, error_type=type(exc).__name__,
            ).inc()
            breaker.record_failure()
            raise

    def embed(self, req: EmbedRequest, *, task: str = "embed") -> EmbedResponse:
        """Run an embedding call with latency, token, error, and circuit breaker tracking."""
        breaker = _llm_breaker(self.provider.provider_type)
        if not breaker.allow_request():
            CIRCUIT_BREAKER_REJECTIONS.labels(service=breaker.name).inc()
            raise CircuitOpenError(breaker.name)

        self._pre_request(task=task)
        start = time.monotonic()
        try:
            resp = self.provider.embed(self.model, req)
        except CircuitOpenError:
            self._release_concurrency()
            raise
        except Exception as exc:
            self._release_concurrency()
            elapsed = time.monotonic() - start
            EMBEDDING_DURATION.labels(model=self.model).observe(elapsed)
            LLM_ERRORS.labels(
                model=self.model, task=task, error_type=type(exc).__name__,
            ).inc()
            breaker.record_failure()
            raise
        elapsed = time.monotonic() - start
        EMBEDDING_DURATION.labels(model=self.model).observe(elapsed)
        EMBEDDING_BATCH_SIZE.labels(model=self.model).observe(len(req.texts))
        if resp.input_tokens:
            LLM_TOKENS.labels(model=self.model, task=task, direction="input").inc(resp.input_tokens)
        breaker.record_success()
        self._post_request(resp.input_tokens, 0)
        return resp


class ModelRegistry:
    """Resolves parsed config into ready-to-use model handles.

    Three-layer resolution:
      1. [modelproviders.*] → provider instances (configured via .configure())
      2. [models.*] → alias → ResolvedModel
      3. [models.roles] → role → ResolvedModel

    Rate limiters, token budgets, and concurrency limits from ``[rate_limits]``
    are injected into every :class:`ResolvedModel`.
    """

    def __init__(self, cfg: Config) -> None:
        # Layer 1: build configured provider instances
        providers: dict[str, ModelProvider] = {}
        for name, pcfg in cfg.providers.items():
            cls = get_provider(pcfg.type)
            instance = cls()
            instance.configure(pcfg.settings)
            providers[name] = instance

        # Build shared rate-limit controls from [rate_limits] config
        rl = cfg.rate_limits

        # Per-provider rate limiters (one per provider *type*)
        rate_limiters: dict[str, RateLimiter] = {}
        if rl.requests_per_minute > 0:
            for prov in providers.values():
                pt = prov.provider_type
                if pt not in rate_limiters:
                    rate_limiters[pt] = RateLimiter(rl.requests_per_minute)

        # Shared token budget (global, not per-provider)
        token_budget: TokenBudget | None = None
        if rl.daily_token_budget > 0:
            token_budget = TokenBudget(rl.daily_token_budget)

        # Shared concurrency limiter (global)
        concurrency_limiter: ConcurrencyLimiter | None = None
        if rl.max_concurrency > 0:
            concurrency_limiter = ConcurrencyLimiter(rl.max_concurrency)

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
                _rate_limiter=rate_limiters.get(provider.provider_type),
                _token_budget=token_budget,
                _concurrency_limiter=concurrency_limiter,
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
