"""Tests for the circuit breaker module."""

import time

import pytest

from worker.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    State,
    all_breakers,
    get_breaker,
)


class TestCircuitBreakerStates:
    """Test state transitions in the circuit breaker."""

    def test_starts_closed(self):
        cb = CircuitBreaker("test-closed", failure_threshold=3, recovery_timeout=10)
        assert cb.state == State.CLOSED

    def test_stays_closed_below_threshold(self):
        cb = CircuitBreaker("test-below", failure_threshold=3, recovery_timeout=10)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == State.CLOSED
        assert cb.failure_count == 2

    def test_opens_at_threshold(self):
        cb = CircuitBreaker("test-threshold", failure_threshold=3, recovery_timeout=10)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == State.OPEN

    def test_open_rejects_requests(self):
        cb = CircuitBreaker("test-reject", failure_threshold=2, recovery_timeout=10)
        cb.record_failure()
        cb.record_failure()
        assert not cb.allow_request()

    def test_half_open_after_recovery_timeout(self):
        cb = CircuitBreaker("test-halfopen", failure_threshold=2, recovery_timeout=0.1)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == State.OPEN
        time.sleep(0.15)
        assert cb.state == State.HALF_OPEN
        assert cb.allow_request()

    def test_half_open_success_closes(self):
        cb = CircuitBreaker("test-close", failure_threshold=2, recovery_timeout=0.1)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)
        assert cb.allow_request()  # enters half-open, allows probe
        cb.record_success()
        assert cb.state == State.CLOSED
        assert cb.failure_count == 0

    def test_half_open_failure_reopens(self):
        cb = CircuitBreaker("test-reopen", failure_threshold=2, recovery_timeout=0.1)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)
        assert cb.allow_request()
        cb.record_failure()
        assert cb.state == State.OPEN

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker("test-reset", failure_threshold=3, recovery_timeout=10)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.failure_count == 0
        # Should need 3 more failures to open
        cb.record_failure()
        cb.record_failure()
        assert cb.state == State.CLOSED


class TestCircuitBreakerContextManager:
    """Test the context manager interface."""

    def test_success_path(self):
        cb = CircuitBreaker("test-ctx-ok", failure_threshold=3, recovery_timeout=10)
        with cb:
            pass  # success
        assert cb.failure_count == 0

    def test_failure_path(self):
        cb = CircuitBreaker("test-ctx-fail", failure_threshold=3, recovery_timeout=10)
        with pytest.raises(ValueError):
            with cb:
                raise ValueError("boom")
        assert cb.failure_count == 1

    def test_context_raises_when_open(self):
        cb = CircuitBreaker("test-ctx-open", failure_threshold=1, recovery_timeout=10)
        cb.record_failure()
        with pytest.raises(CircuitOpenError) as exc_info:
            with cb:
                pass
        assert exc_info.value.breaker_name == "test-ctx-open"


class TestCircuitBreakerStatus:
    """Test the status reporting method."""

    def test_closed_status(self):
        cb = CircuitBreaker("test-status", failure_threshold=5, recovery_timeout=60)
        status = cb.status()
        assert status["state"] == "closed"
        assert status["failure_count"] == 0
        assert "recovery_remaining_s" not in status

    def test_open_status_includes_recovery(self):
        cb = CircuitBreaker(
            "test-open-status", failure_threshold=1, recovery_timeout=60
        )
        cb.record_failure()
        status = cb.status()
        assert status["state"] == "open"
        assert "recovery_remaining_s" in status
        assert status["recovery_remaining_s"] > 0


class TestCircuitBreakerRegistry:
    """Test the global registry."""

    def test_breaker_registered_on_creation(self):
        name = "test-registry-auto"
        cb = CircuitBreaker(name, failure_threshold=3, recovery_timeout=10)
        assert get_breaker(name) is cb

    def test_all_breakers_returns_snapshot(self):
        _ = CircuitBreaker("test-all-1", failure_threshold=3, recovery_timeout=10)
        _ = CircuitBreaker("test-all-2", failure_threshold=3, recovery_timeout=10)
        breakers = all_breakers()
        assert "test-all-1" in breakers
        assert "test-all-2" in breakers


class TestCircuitBreakerReset:
    """Test manual reset."""

    def test_reset_closes_open_breaker(self):
        cb = CircuitBreaker(
            "test-manual-reset", failure_threshold=1, recovery_timeout=999
        )
        cb.record_failure()
        assert cb.state == State.OPEN
        cb.reset()
        assert cb.state == State.CLOSED
        assert cb.failure_count == 0
        assert cb.allow_request()


class TestHalfOpenMaxCalls:
    """Test that half-open limits concurrent probes."""

    def test_limits_concurrent_probes(self):
        cb = CircuitBreaker(
            "test-halfopen-limit",
            failure_threshold=1,
            recovery_timeout=0.1,
            half_open_max_calls=1,
        )
        cb.record_failure()
        time.sleep(0.15)
        assert cb.allow_request()  # first probe allowed
        assert not cb.allow_request()  # second probe rejected
