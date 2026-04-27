"""Tests for the Slack rate-limit-aware retry helper.

Covers:
- ``ratelimited`` SlackApiError triggers Retry-After-aware sleep + retry.
- Retry-After header is honored; falls back to 30s when absent.
- Exponential backoff factors of 1x, 2x, 4x of Retry-After.
- Jitter is positive and added on top of the base sleep.
- After ``max_retries`` consecutive ratelimited responses, the helper
  raises :class:`RateLimitedError` carrying the last SlackApiError.
- Non-``ratelimited`` SlackApiError codes propagate immediately.
- ``SLACK_RATE_LIMIT_HITS`` counter increments per ratelimit hit.
"""

from __future__ import annotations

from typing import Any

import pytest
from slack_sdk.errors import SlackApiError

from worker.metrics import SLACK_RATE_LIMIT_HITS
from worker.sources._slack_rate_limit import (
    DEFAULT_RETRY_AFTER_SECONDS,
    RateLimitedError,
    _is_rate_limited,
    _retry_after_seconds,
    call_with_rate_limit_retry,
)


class _FakeResponse:
    """Minimal SlackResponse-shaped object for raising SlackApiError."""

    def __init__(
        self,
        *,
        error: str = "ratelimited",
        retry_after: int | str | None = None,
        status_code: int = 429,
    ) -> None:
        self._data = {"ok": False, "error": error}
        self.headers: dict[str, Any] = {}
        if retry_after is not None:
            self.headers["Retry-After"] = retry_after
        self.status_code = status_code

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)


def _ratelimited_error(retry_after: int | str | None = 2) -> SlackApiError:
    resp = _FakeResponse(error="ratelimited", retry_after=retry_after, status_code=429)
    return SlackApiError("ratelimited", resp)


def _other_error(code: str = "channel_not_found") -> SlackApiError:
    resp = _FakeResponse(error=code, status_code=400)
    return SlackApiError(code, resp)


# ---------------------------------------------------------------------------
# Helpers under test: low-level predicates
# ---------------------------------------------------------------------------


def test_is_rate_limited_true_on_ratelimited_code() -> None:
    assert _is_rate_limited(_ratelimited_error())


def test_is_rate_limited_true_on_429_status_alone() -> None:
    # No body code, but status 429 — still rate-limited.
    resp = _FakeResponse(error="", retry_after=1, status_code=429)
    exc = SlackApiError("ratelimited", resp)
    assert _is_rate_limited(exc)


def test_is_rate_limited_false_on_other_codes() -> None:
    assert not _is_rate_limited(_other_error("channel_not_found"))


def test_retry_after_seconds_reads_header() -> None:
    assert _retry_after_seconds(_ratelimited_error(retry_after=7)) == 7


def test_retry_after_seconds_default_when_missing() -> None:
    assert (
        _retry_after_seconds(_ratelimited_error(retry_after=None))
        == DEFAULT_RETRY_AFTER_SECONDS
    )


def test_retry_after_seconds_handles_string_header() -> None:
    assert _retry_after_seconds(_ratelimited_error(retry_after="11")) == 11


# ---------------------------------------------------------------------------
# Helpers under test: retry orchestration
# ---------------------------------------------------------------------------


class _Caller:
    """Records every call and yields a configurable sequence of outcomes."""

    def __init__(self, outcomes: list[Any]) -> None:
        self._outcomes = list(outcomes)
        self.calls = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def test_rate_limit_retries_with_retry_after() -> None:
    """One ratelimited response, then success — helper sleeps and returns."""
    sleeps: list[float] = []
    caller = _Caller([_ratelimited_error(retry_after=2), {"ok": True, "value": 42}])

    result = call_with_rate_limit_retry(
        caller,
        sleep=sleeps.append,
        jitter=lambda hi: 1.0,  # deterministic
    )

    assert result == {"ok": True, "value": 42}
    assert caller.calls == 2
    # 1 retry → 1 sleep at base*1 + jitter = 2 + 1 = 3.
    assert sleeps == [3.0]


def test_rate_limit_max_retries() -> None:
    """4 consecutive ratelimited → RateLimitedError after 3 retries."""
    sleeps: list[float] = []
    caller = _Caller([_ratelimited_error(retry_after=2)] * 4)

    with pytest.raises(RateLimitedError) as ei:
        call_with_rate_limit_retry(
            caller, sleep=sleeps.append, jitter=lambda hi: 0.5
        )

    assert caller.calls == 4
    # Retries 1, 2, 3 with backoff factors 1x, 2x, 4x.
    assert sleeps == [2 * 1 + 0.5, 2 * 2 + 0.5, 2 * 4 + 0.5]
    assert isinstance(ei.value.last_error, SlackApiError)


def test_rate_limit_jitter_added() -> None:
    """Each sleep duration equals base*backoff + a strictly positive jitter."""
    sleeps: list[float] = []
    jitter_calls: list[float] = []

    def _jitter(hi: float) -> float:
        jitter_calls.append(hi)
        return 1.5  # > 0

    caller = _Caller([_ratelimited_error(retry_after=4), {"ok": True}])
    call_with_rate_limit_retry(caller, sleep=sleeps.append, jitter=_jitter)

    assert sleeps == [4 * 1 + 1.5]
    # Jitter ceiling exposed to the supplier was the configured 5s max.
    assert jitter_calls == [5.0]


def test_no_retry_on_other_errors() -> None:
    """Non-ratelimited SlackApiError surfaces immediately, no sleep."""
    sleeps: list[float] = []
    caller = _Caller([_other_error("channel_not_found")])

    with pytest.raises(SlackApiError) as ei:
        call_with_rate_limit_retry(caller, sleep=sleeps.append)

    assert caller.calls == 1
    assert sleeps == []
    # Original error is preserved (not wrapped in RateLimitedError).
    assert not isinstance(ei.value, RateLimitedError)


def test_metric_increments() -> None:
    """``SLACK_RATE_LIMIT_HITS`` increments once per ratelimit hit."""
    before = SLACK_RATE_LIMIT_HITS._value.get()
    sleeps: list[float] = []
    caller = _Caller([_ratelimited_error(retry_after=1)] * 2 + [{"ok": True}])

    call_with_rate_limit_retry(caller, sleep=sleeps.append, jitter=lambda hi: 0.0)

    after = SLACK_RATE_LIMIT_HITS._value.get()
    # Two ratelimited responses absorbed → +2 hits recorded.
    assert after - before == 2


def test_metric_increments_on_exhaustion() -> None:
    """All retries exhausted: every attempt's ratelimit still counts."""
    before = SLACK_RATE_LIMIT_HITS._value.get()
    sleeps: list[float] = []
    caller = _Caller([_ratelimited_error(retry_after=1)] * 4)

    with pytest.raises(RateLimitedError):
        call_with_rate_limit_retry(
            caller, sleep=sleeps.append, jitter=lambda hi: 0.0
        )

    after = SLACK_RATE_LIMIT_HITS._value.get()
    # 1 initial + 3 retries all ratelimited → +4 hits.
    assert after - before == 4


def test_function_args_passed_through() -> None:
    """Positional/keyword args are forwarded unchanged on success."""
    captured: dict[str, Any] = {}

    def _api(channel: str, *, limit: int) -> dict[str, Any]:
        captured["channel"] = channel
        captured["limit"] = limit
        return {"ok": True}

    call_with_rate_limit_retry(_api, "C123", limit=200, sleep=lambda _: None)
    assert captured == {"channel": "C123", "limit": 200}
