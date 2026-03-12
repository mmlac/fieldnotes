"""Tests for rate limiting, token budgets, and concurrency controls."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from worker.config import RateLimitConfig, _parse
from worker.models.rate_limiter import (
    ConcurrencyLimiter,
    RateLimiter,
    RateLimitExceeded,
    TokenBudget,
    TokenBudgetExhausted,
)


# ------------------------------------------------------------------
# RateLimiter
# ------------------------------------------------------------------


class TestRateLimiter:
    def test_allows_up_to_rpm(self) -> None:
        limiter = RateLimiter(rpm=5)
        for _ in range(5):
            limiter.acquire(timeout=1.0)

    def test_blocks_beyond_rpm(self) -> None:
        limiter = RateLimiter(rpm=2)
        limiter.acquire(timeout=1.0)
        limiter.acquire(timeout=1.0)
        with pytest.raises(RateLimitExceeded):
            limiter.acquire(timeout=0.1)

    def test_invalid_rpm_raises(self) -> None:
        with pytest.raises(ValueError, match="rpm must be >= 1"):
            RateLimiter(rpm=0)

    def test_rpm_property(self) -> None:
        limiter = RateLimiter(rpm=42)
        assert limiter.rpm == 42

    def test_slots_replenish_after_window(self) -> None:
        limiter = RateLimiter(rpm=1)
        # Inject a timestamp 61 seconds in the past so it's outside the window
        limiter._timestamps = [time.monotonic() - 61.0]
        limiter.acquire(timeout=0.1)  # should succeed — old ts pruned


# ------------------------------------------------------------------
# TokenBudget
# ------------------------------------------------------------------


class TestTokenBudget:
    def test_record_and_check(self) -> None:
        budget = TokenBudget(daily_limit=100)
        budget.record(40, 20)
        assert budget.used == 60
        assert budget.remaining == 40
        budget.check()  # should not raise

    def test_exhausted_raises(self) -> None:
        budget = TokenBudget(daily_limit=50)
        budget.record(30, 20)
        with pytest.raises(TokenBudgetExhausted):
            budget.check()

    def test_warning_at_80_percent(self, caplog: pytest.LogCaptureFixture) -> None:
        budget = TokenBudget(daily_limit=100)
        import logging
        with caplog.at_level(logging.WARNING):
            budget.record(80, 0)
        assert "80%" in caplog.text or "Token budget" in caplog.text

    def test_warning_fires_once(self, caplog: pytest.LogCaptureFixture) -> None:
        budget = TokenBudget(daily_limit=100)
        import logging
        with caplog.at_level(logging.WARNING):
            budget.record(81, 0)
            caplog.clear()
            budget.record(5, 0)
        assert "Token budget" not in caplog.text

    def test_daily_reset(self) -> None:
        budget = TokenBudget(daily_limit=100)
        budget.record(100, 0)
        # Simulate day rollover
        with patch.object(TokenBudget, "_current_day", return_value=budget._day + 1):
            assert budget.used == 0
            assert budget.remaining == 100
            budget.check()  # should not raise

    def test_invalid_limit_raises(self) -> None:
        with pytest.raises(ValueError, match="daily_limit must be >= 1"):
            TokenBudget(daily_limit=0)

    def test_zero_tokens_not_recorded(self) -> None:
        budget = TokenBudget(daily_limit=100)
        budget.record(0, 0)
        assert budget.used == 0


# ------------------------------------------------------------------
# ConcurrencyLimiter
# ------------------------------------------------------------------


class TestConcurrencyLimiter:
    def test_acquire_release(self) -> None:
        limiter = ConcurrencyLimiter(max_concurrency=2)
        assert limiter.acquire(timeout=0.1)
        assert limiter.acquire(timeout=0.1)
        # Third should block
        assert not limiter.acquire(timeout=0.05)
        limiter.release()
        assert limiter.acquire(timeout=0.1)

    def test_max_concurrency_property(self) -> None:
        limiter = ConcurrencyLimiter(max_concurrency=3)
        assert limiter.max_concurrency == 3

    def test_invalid_concurrency_raises(self) -> None:
        with pytest.raises(ValueError, match="max_concurrency must be >= 1"):
            ConcurrencyLimiter(max_concurrency=0)


# ------------------------------------------------------------------
# Config parsing
# ------------------------------------------------------------------


class TestRateLimitConfig:
    def test_defaults_when_absent(self) -> None:
        cfg = _parse({})
        assert cfg.rate_limits.requests_per_minute == 0
        assert cfg.rate_limits.daily_token_budget == 0
        assert cfg.rate_limits.max_concurrency == 0

    def test_parses_all_fields(self) -> None:
        raw = {
            "rate_limits": {
                "requests_per_minute": 60,
                "daily_token_budget": 1_000_000,
                "max_concurrency": 4,
            }
        }
        cfg = _parse(raw)
        assert cfg.rate_limits.requests_per_minute == 60
        assert cfg.rate_limits.daily_token_budget == 1_000_000
        assert cfg.rate_limits.max_concurrency == 4

    def test_partial_override(self) -> None:
        raw = {"rate_limits": {"max_concurrency": 8}}
        cfg = _parse(raw)
        assert cfg.rate_limits.requests_per_minute == 0
        assert cfg.rate_limits.max_concurrency == 8

    def test_negative_value_raises(self) -> None:
        raw = {"rate_limits": {"requests_per_minute": -1}}
        with pytest.raises(ValueError, match="must be >= 0"):
            _parse(raw)

    def test_wrong_type_raises(self) -> None:
        raw = {"rate_limits": {"requests_per_minute": "fast"}}
        with pytest.raises(TypeError):
            _parse(raw)


# ------------------------------------------------------------------
# Integration: ResolvedModel with limiters
# ------------------------------------------------------------------


class TestResolvedModelWithLimiters:
    def _make_model(
        self,
        rate_limiter: RateLimiter | None = None,
        token_budget: TokenBudget | None = None,
        concurrency_limiter: ConcurrencyLimiter | None = None,
    ):
        from worker.models.base import CompletionResponse
        from worker.models.resolver import ResolvedModel

        provider = MagicMock()
        provider.provider_type = "openai"
        provider.complete.return_value = CompletionResponse(
            text="ok", input_tokens=10, output_tokens=5,
        )
        provider.embed.return_value = MagicMock(
            vectors=[[0.1, 0.2]], model="test", input_tokens=3,
        )

        return ResolvedModel(
            alias="test",
            model="gpt-4",
            provider=provider,
            _rate_limiter=rate_limiter,
            _token_budget=token_budget,
            _concurrency_limiter=concurrency_limiter,
        )

    def test_complete_records_token_budget(self) -> None:
        from worker.models.base import CompletionRequest
        budget = TokenBudget(daily_limit=1000)
        model = self._make_model(token_budget=budget)
        model.complete(CompletionRequest(system="test", messages=[]))
        assert budget.used == 15  # 10 input + 5 output

    def test_complete_respects_rate_limit(self) -> None:
        from worker.models.base import CompletionRequest
        limiter = RateLimiter(rpm=1)
        model = self._make_model(rate_limiter=limiter)
        model.complete(CompletionRequest(system="test", messages=[]))
        with pytest.raises(RateLimitExceeded):
            model.complete(CompletionRequest(system="test", messages=[]))

    def test_complete_blocked_by_token_budget(self) -> None:
        from worker.models.base import CompletionRequest
        budget = TokenBudget(daily_limit=10)
        budget.record(10, 0)  # exhaust budget
        model = self._make_model(token_budget=budget)
        with pytest.raises(TokenBudgetExhausted):
            model.complete(CompletionRequest(system="test", messages=[]))

    def test_concurrency_released_on_error(self) -> None:
        from worker.models.base import CompletionRequest
        from worker.models.resolver import ResolvedModel

        provider = MagicMock()
        provider.provider_type = "openai"
        provider.complete.side_effect = RuntimeError("boom")

        limiter = ConcurrencyLimiter(max_concurrency=1)
        model = ResolvedModel(
            alias="test", model="gpt-4", provider=provider,
            _concurrency_limiter=limiter,
        )
        with pytest.raises(RuntimeError):
            model.complete(CompletionRequest(system="test", messages=[]))
        # Slot should be released — next acquire should succeed
        assert limiter.acquire(timeout=0.1)

    def test_no_limiters_works_as_before(self) -> None:
        from worker.models.base import CompletionRequest
        model = self._make_model()  # no limiters
        resp = model.complete(CompletionRequest(system="test", messages=[]))
        assert resp.text == "ok"
