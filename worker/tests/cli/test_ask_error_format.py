"""Tests for the ask CLI's user-facing error formatter."""

from __future__ import annotations

import httpx

from worker.cli.ask import _format_error_for_user


def test_timeout_produces_actionable_message() -> None:
    # httpx.ReadTimeout().__str__() is empty, which is what made the
    # original "error: {exc}" rendering look like a silent failure.
    msg = _format_error_for_user(httpx.ReadTimeout("read timed out"))
    assert "timed out" in msg
    # Steers the user toward the knob that actually fixes it.
    assert "completion_timeout" in msg or "OLLAMA_COMPLETION_TIMEOUT" in msg


def test_http_status_error_surfaces_code() -> None:
    resp = httpx.Response(
        status_code=500,
        request=httpx.Request("POST", "http://localhost/api/chat"),
    )
    exc = httpx.HTTPStatusError("Server error", request=resp.request, response=resp)
    msg = _format_error_for_user(exc)
    assert "500" in msg


def test_falls_back_to_class_name_for_silent_exceptions() -> None:
    class _Quiet(Exception):
        pass

    assert _format_error_for_user(_Quiet()) == "_Quiet"
