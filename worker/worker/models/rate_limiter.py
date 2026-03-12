"""Rate limiting, token budgets, and concurrency control for LLM API calls.

Provides three complementary controls:
  - RateLimiter: sliding-window RPM (requests per minute) per provider
  - TokenBudget: daily token budget with warning at 80% and hard stop at 100%
  - ConcurrencyLimiter: semaphore-based cap on parallel LLM calls
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class RateLimitExceeded(Exception):
    """Raised when a request would exceed the configured RPM limit."""


class TokenBudgetExhausted(Exception):
    """Raised when the daily token budget has been reached."""


# ---------------------------------------------------------------------------
# RateLimiter — sliding-window RPM
# ---------------------------------------------------------------------------


class RateLimiter:
    """Sliding-window rate limiter that enforces requests-per-minute.

    Each provider gets its own instance.  Callers must invoke :meth:`acquire`
    before making an API call — it blocks until a slot is available or raises
    after *timeout* seconds.
    """

    def __init__(self, rpm: int) -> None:
        if rpm < 1:
            raise ValueError(f"rpm must be >= 1, got {rpm}")
        self._rpm = rpm
        self._lock = threading.Lock()
        # Timestamps (monotonic) of recent requests within the sliding window.
        self._timestamps: list[float] = []

    @property
    def rpm(self) -> int:
        return self._rpm

    def acquire(self, timeout: float = 60.0) -> None:
        """Block until a request slot is available within the RPM window.

        Raises :class:`RateLimitExceeded` if *timeout* elapses without a slot.
        """
        deadline = time.monotonic() + timeout
        while True:
            now = time.monotonic()
            if now > deadline:
                raise RateLimitExceeded(
                    f"Timed out after {timeout:.0f}s waiting for rate limit "
                    f"slot ({self._rpm} RPM)"
                )
            with self._lock:
                cutoff = now - 60.0
                # Prune timestamps older than the 60-second window
                self._timestamps = [
                    ts for ts in self._timestamps if ts > cutoff
                ]
                if len(self._timestamps) < self._rpm:
                    self._timestamps.append(now)
                    return
                # Compute how long until the oldest entry ages out
                wait = self._timestamps[0] - cutoff
            time.sleep(min(wait + 0.01, deadline - time.monotonic()))


# ---------------------------------------------------------------------------
# TokenBudget — daily cap with warning/hard-stop
# ---------------------------------------------------------------------------


_WARN_THRESHOLD = 0.80
_HARD_THRESHOLD = 1.00


class TokenBudget:
    """Tracks daily token usage and enforces a hard budget.

    Warns at 80% of the budget (via logging) and raises
    :class:`TokenBudgetExhausted` at 100%.
    """

    def __init__(self, daily_limit: int) -> None:
        if daily_limit < 1:
            raise ValueError(f"daily_limit must be >= 1, got {daily_limit}")
        self._daily_limit = daily_limit
        self._lock = threading.Lock()
        self._used: int = 0
        # Day boundary (midnight UTC) when the counter last reset.
        self._day: int = self._current_day()
        self._warned = False

    @property
    def daily_limit(self) -> int:
        return self._daily_limit

    @property
    def used(self) -> int:
        with self._lock:
            self._maybe_reset()
            return self._used

    @property
    def remaining(self) -> int:
        with self._lock:
            self._maybe_reset()
            return max(0, self._daily_limit - self._used)

    @staticmethod
    def _current_day() -> int:
        """Return the current day as days-since-epoch (UTC)."""
        return int(time.time()) // 86400

    def _maybe_reset(self) -> None:
        """Reset counters if the day has rolled over.  Caller must hold lock."""
        today = self._current_day()
        if today != self._day:
            self._used = 0
            self._day = today
            self._warned = False

    def check(self) -> None:
        """Raise :class:`TokenBudgetExhausted` if the budget is fully spent."""
        with self._lock:
            self._maybe_reset()
            if self._used >= self._daily_limit:
                raise TokenBudgetExhausted(
                    f"Daily token budget exhausted: "
                    f"{self._used:,}/{self._daily_limit:,} tokens used"
                )

    def record(self, input_tokens: int, output_tokens: int) -> None:
        """Record token usage from a completed call.

        Logs a warning when 80% of the budget is consumed.
        """
        total = input_tokens + output_tokens
        if total <= 0:
            return
        with self._lock:
            self._maybe_reset()
            self._used += total
            ratio = self._used / self._daily_limit
            if ratio >= _WARN_THRESHOLD and not self._warned:
                self._warned = True
                logger.warning(
                    "Token budget %.0f%% consumed: %s/%s tokens",
                    ratio * 100,
                    f"{self._used:,}",
                    f"{self._daily_limit:,}",
                )


# ---------------------------------------------------------------------------
# ConcurrencyLimiter — bounded parallel LLM calls
# ---------------------------------------------------------------------------


class ConcurrencyLimiter:
    """Semaphore wrapper that limits the number of concurrent LLM calls."""

    def __init__(self, max_concurrency: int) -> None:
        if max_concurrency < 1:
            raise ValueError(
                f"max_concurrency must be >= 1, got {max_concurrency}"
            )
        self._semaphore = threading.Semaphore(max_concurrency)
        self._max = max_concurrency

    @property
    def max_concurrency(self) -> int:
        return self._max

    def acquire(self, timeout: float = 300.0) -> bool:
        """Acquire a slot.  Returns True on success, False on timeout."""
        return self._semaphore.acquire(timeout=timeout)

    def release(self) -> None:
        self._semaphore.release()


# ---------------------------------------------------------------------------
# RateLimitConfig — parsed from [rate_limits] TOML section
# ---------------------------------------------------------------------------


@dataclass
class RateLimitConfig:
    """Parsed ``[rate_limits]`` section."""

    requests_per_minute: int = 0    # 0 = disabled
    daily_token_budget: int = 0     # 0 = disabled (unlimited)
    max_concurrency: int = 0        # 0 = disabled (unlimited)
