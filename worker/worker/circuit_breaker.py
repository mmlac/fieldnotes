"""Thread-safe circuit breaker for downstream service protection.

Implements the standard three-state circuit breaker pattern:

  CLOSED  →  (N consecutive failures)  →  OPEN
  OPEN    →  (backoff expires)         →  HALF_OPEN
  HALF_OPEN → (probe succeeds)         →  CLOSED
  HALF_OPEN → (probe fails)            →  OPEN

Usage::

    neo4j_cb = CircuitBreaker("neo4j", failure_threshold=5, recovery_timeout=60)

    if not neo4j_cb.allow_request():
        raise CircuitOpenError("neo4j")

    try:
        result = do_neo4j_call()
        neo4j_cb.record_success()
    except ServiceUnavailable:
        neo4j_cb.record_failure()
        raise

Or as a decorator / context manager::

    with neo4j_cb:
        do_neo4j_call()

The module also provides a global registry so that all circuit breaker
instances can be queried for health reporting (e.g. ingest_status).
"""

from __future__ import annotations

import enum
import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global registry
# ---------------------------------------------------------------------------

_registry_lock = threading.Lock()
_registry: dict[str, CircuitBreaker] = {}


def get_breaker(name: str) -> CircuitBreaker | None:
    """Return the circuit breaker registered under *name*, or None."""
    with _registry_lock:
        return _registry.get(name)


def all_breakers() -> dict[str, CircuitBreaker]:
    """Return a snapshot of all registered circuit breakers."""
    with _registry_lock:
        return dict(_registry)


def _register(breaker: CircuitBreaker) -> None:
    with _registry_lock:
        _registry[breaker.name] = breaker


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


class State(enum.Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when a call is rejected because the circuit is open."""

    def __init__(self, name: str) -> None:
        self.breaker_name = name
        super().__init__(f"Circuit breaker '{name}' is OPEN — service unavailable")


class CircuitBreaker:
    """Thread-safe circuit breaker.

    Parameters
    ----------
    name:
        Identifier for this breaker (e.g. "neo4j", "qdrant", "openai").
    failure_threshold:
        Number of consecutive failures before opening the circuit.
    recovery_timeout:
        Seconds to wait in OPEN state before allowing a probe (HALF_OPEN).
    half_open_max_calls:
        Maximum concurrent probe calls allowed in HALF_OPEN state.
    """

    def __init__(
        self,
        name: str,
        *,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 1,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self._lock = threading.Lock()
        self._state = State.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float = 0.0
        self._opened_at: float = 0.0
        self._half_open_calls = 0

        _register(self)

    # -- state queries -------------------------------------------------------

    @property
    def state(self) -> State:
        with self._lock:
            return self._current_state()

    @property
    def failure_count(self) -> int:
        with self._lock:
            return self._failure_count

    def _current_state(self) -> State:
        """Return effective state, auto-transitioning OPEN → HALF_OPEN on timeout."""
        if self._state == State.OPEN:
            if time.monotonic() - self._opened_at >= self.recovery_timeout:
                self._state = State.HALF_OPEN
                self._half_open_calls = 0
                logger.info(
                    "Circuit breaker '%s': OPEN → HALF_OPEN (recovery timeout elapsed)",
                    self.name,
                )
        return self._state

    # -- request gating ------------------------------------------------------

    def allow_request(self) -> bool:
        """Return True if the call should proceed, False if circuit is open."""
        with self._lock:
            state = self._current_state()
            if state == State.CLOSED:
                return True
            if state == State.HALF_OPEN:
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
            # OPEN
            return False

    # -- outcome recording ---------------------------------------------------

    def record_success(self) -> None:
        """Record a successful call. Resets failure count; closes circuit if half-open."""
        with self._lock:
            state = self._current_state()
            if state == State.HALF_OPEN:
                self._state = State.CLOSED
                self._failure_count = 0
                self._half_open_calls = 0
                logger.info(
                    "Circuit breaker '%s': HALF_OPEN → CLOSED (probe succeeded)",
                    self.name,
                )
            elif state == State.CLOSED:
                self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call. Opens circuit after threshold consecutive failures."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            state = self._current_state()

            if state == State.HALF_OPEN:
                self._state = State.OPEN
                self._opened_at = time.monotonic()
                logger.warning(
                    "Circuit breaker '%s': HALF_OPEN → OPEN (probe failed)",
                    self.name,
                )
            elif state == State.CLOSED and self._failure_count >= self.failure_threshold:
                self._state = State.OPEN
                self._opened_at = time.monotonic()
                logger.warning(
                    "Circuit breaker '%s': CLOSED → OPEN after %d consecutive failures",
                    self.name,
                    self._failure_count,
                )

    # -- manual control ------------------------------------------------------

    def reset(self) -> None:
        """Force-close the circuit breaker (e.g. after manual verification)."""
        with self._lock:
            self._state = State.CLOSED
            self._failure_count = 0
            self._half_open_calls = 0
            logger.info("Circuit breaker '%s': manually reset to CLOSED", self.name)

    # -- context manager (records success/failure automatically) ---------------

    def __enter__(self) -> CircuitBreaker:
        if not self.allow_request():
            raise CircuitOpenError(self.name)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        if exc_type is None:
            self.record_success()
        else:
            self.record_failure()

    # -- introspection -------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Return a dict describing the current breaker state for health reporting."""
        with self._lock:
            state = self._current_state()
            info: dict[str, Any] = {
                "state": state.value,
                "failure_count": self._failure_count,
                "failure_threshold": self.failure_threshold,
                "recovery_timeout": self.recovery_timeout,
            }
            if state == State.OPEN:
                remaining = self.recovery_timeout - (time.monotonic() - self._opened_at)
                info["recovery_remaining_s"] = round(max(0, remaining), 1)
            return info

    def __repr__(self) -> str:
        return f"CircuitBreaker({self.name!r}, state={self.state.value})"
