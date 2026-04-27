"""Rate-limit-aware wrapper for Slack Web API calls.

Slack's tiered rate limits (Tier 2/3 endpoints — ``conversations.history``,
``conversations.replies``, ``users.list`` — cap at 50-100 calls/min) bite
the source's polling and backfill loops on busy workspaces.  A single 429
response surfaced as a bare ``SlackApiError`` would crash the source.

This helper wraps each API call so that ``error='ratelimited'`` responses
trigger Retry-After-aware sleep + bounded exponential backoff, leaving
all other ``SlackApiError`` codes (e.g. ``channel_not_found``) untouched
for callers to handle.

Behavior:

* On a ratelimited response, sleep ``Retry-After + jitter`` seconds.
  ``Retry-After`` defaults to 30s when the header is absent.  Jitter is
  uniform in ``(0, 5]`` seconds — strictly positive so back-to-back
  callers never sleep for exactly the same duration.
* Up to ``max_retries=3`` retries with exponential backoff factors
  ``1x, 2x, 4x`` of the Retry-After value.  After the 4th failure the
  helper raises :class:`RateLimitedError` carrying the last response.
* Each ratelimit hit increments
  :data:`worker.metrics.SLACK_RATE_LIMIT_HITS`.

The Slack SDK's :class:`slack_sdk.http_retry.RateLimitErrorRetryHandler`
does the basic Retry-After sleep but does not implement exponential
backoff, raise a typed terminal error, or feed our metrics — hand-rolling
keeps the contract explicit and testable without monkeying with the
WebClient's retry plumbing.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Callable, TypeVar

from slack_sdk.errors import SlackApiError

from worker.metrics import SLACK_RATE_LIMIT_HITS

logger = logging.getLogger(__name__)

DEFAULT_RETRY_AFTER_SECONDS = 30
DEFAULT_MAX_RETRIES = 3
DEFAULT_JITTER_MAX_SECONDS = 5.0

T = TypeVar("T")


class RateLimitedError(Exception):
    """Raised when Slack rate-limit retries are exhausted.

    ``last_error`` is the final :class:`SlackApiError` whose response
    headers carried the unhandled Retry-After signal, kept so callers
    can log or surface details if needed.
    """

    def __init__(self, message: str, *, last_error: SlackApiError) -> None:
        super().__init__(message)
        self.last_error = last_error


def _is_rate_limited(exc: SlackApiError) -> bool:
    """True iff *exc* represents a Slack 429 / ``ratelimited`` response."""
    resp = getattr(exc, "response", None)
    if resp is None:
        return False
    err_code = ""
    try:
        # SlackResponse.get(key) returns the body field; falsy on missing.
        err_code = (resp.get("error") if hasattr(resp, "get") else "") or ""
    except Exception:
        err_code = ""
    if err_code == "ratelimited":
        return True
    return getattr(resp, "status_code", None) == 429


def _retry_after_seconds(exc: SlackApiError) -> int:
    """Read ``Retry-After`` from *exc* response, defaulting to 30s.

    Tolerates header values delivered as ``str``, ``int``, or
    ``list[str]`` (urllib3 sometimes wraps multi-valued headers).
    """
    resp = getattr(exc, "response", None)
    if resp is None:
        return DEFAULT_RETRY_AFTER_SECONDS
    headers = getattr(resp, "headers", None) or {}
    raw: Any = None
    if hasattr(headers, "items"):
        for k, v in headers.items():
            if str(k).lower() == "retry-after":
                raw = v
                break
    if raw is None:
        return DEFAULT_RETRY_AFTER_SECONDS
    if isinstance(raw, (list, tuple)):
        raw = raw[0] if raw else None
    if raw is None:
        return DEFAULT_RETRY_AFTER_SECONDS
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return DEFAULT_RETRY_AFTER_SECONDS


def call_with_rate_limit_retry(
    func: Callable[..., T],
    *args: Any,
    max_retries: int = DEFAULT_MAX_RETRIES,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[float], float] | None = None,
    **kwargs: Any,
) -> T:
    """Invoke *func*, retrying on Slack ``ratelimited`` responses.

    Up to *max_retries* additional attempts are made on ratelimited
    failures, with sleeps of ``Retry-After * (1, 2, 4) + jitter`` seconds
    between attempts.  Non-ratelimited :class:`SlackApiError` instances
    propagate immediately.  After exhaustion the helper raises
    :class:`RateLimitedError`.

    *sleep* and *jitter* are injectable for tests.  *jitter* receives the
    configured maximum (5.0) and must return a strictly positive float so
    that simultaneous retries from multiple call sites don't pile up.
    """
    if jitter is None:
        # Strictly positive draw so the contract — "jitter is added" — is
        # observable in tests; collapses to time-honored uniform(0, hi]
        # otherwise.
        def jitter(hi: float) -> float:
            j = random.uniform(0.0, hi)
            # Avoid the (vanishingly rare) j==0.0 draw so callers can
            # always assert "wait > base".
            return j if j > 0 else hi / 2.0

    attempt = 0
    while True:
        try:
            return func(*args, **kwargs)
        except SlackApiError as exc:
            if not _is_rate_limited(exc):
                raise
            SLACK_RATE_LIMIT_HITS.inc()
            if attempt >= max_retries:
                raise RateLimitedError(
                    f"Slack rate limit exceeded after {max_retries} retries",
                    last_error=exc,
                ) from exc
            base = _retry_after_seconds(exc)
            backoff_factor = 2**attempt  # 1, 2, 4
            jitter_s = jitter(DEFAULT_JITTER_MAX_SECONDS)
            wait = base * backoff_factor + jitter_s
            logger.warning(
                "Slack rate limited; sleeping %.2fs (base=%ds backoff=%dx "
                "jitter=%.2fs) attempt %d/%d",
                wait,
                base,
                backoff_factor,
                jitter_s,
                attempt + 1,
                max_retries,
            )
            sleep(wait)
            attempt += 1
