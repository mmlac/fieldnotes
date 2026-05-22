"""Tests for the question-time heuristic extractor."""

from __future__ import annotations

from datetime import date

import pytest

from worker.query._question_time import extract_date_window, mentions_journal


TODAY = date(2026, 5, 22)  # a Friday — used to make weekly arithmetic deterministic


@pytest.mark.parametrize(
    "question, expected",
    [
        # The motivating case from the user's report.
        ("Summarize my journal entries of the last seven days",
         (date(2026, 5, 16), TODAY)),
        # Digit form is the same.
        ("notes from the last 7 days", (date(2026, 5, 16), TODAY)),
        # Bare "last week" defaults to N=1 → 7 days inclusive ending today.
        ("what happened last week", (date(2026, 5, 16), TODAY)),
        ("last month", (date(2026, 4, 23), TODAY)),       # 30 days inclusive
        ("last year", (date(2025, 5, 23), TODAY)),        # 365 days inclusive
        # "past" / "previous" / "in the last" all alias to the same shape.
        ("in the past 3 days", (date(2026, 5, 20), TODAY)),
        ("previous 2 weeks", (date(2026, 5, 9), TODAY)),
        ("in the last fourteen days", (date(2026, 5, 9), TODAY)),
        # Calendar-aligned forms.
        ("this week's notes",
         (date(2026, 5, 18), TODAY)),    # Mon..today (TODAY is Friday)
        ("this month", (date(2026, 5, 1), TODAY)),
        ("this year so far", (date(2026, 1, 1), TODAY)),
        # Single-day shortcuts.
        ("what did I do today", (TODAY, TODAY)),
        ("yesterday's plan", (date(2026, 5, 21), date(2026, 5, 21))),
    ],
)
def test_extract_date_window_recognised_forms(
    question: str, expected: tuple[date, date]
) -> None:
    assert extract_date_window(question, today=TODAY) == expected


@pytest.mark.parametrize(
    "question",
    [
        "tell me about the project",
        "what's in my notes",
        "summarize everything",
        "",
        "list all people",
    ],
)
def test_extract_date_window_no_match_returns_none(question: str) -> None:
    assert extract_date_window(question, today=TODAY) is None


@pytest.mark.parametrize(
    "question, expected",
    [
        ("journal entries from last week", True),
        ("summarize my journal", True),
        ("my journaling habit", True),
        ("show notes mentioning Bob", False),
        ("calendar for tomorrow", False),
        ("", False),
    ],
)
def test_mentions_journal(question: str, expected: bool) -> None:
    assert mentions_journal(question) is expected


def test_extract_date_window_default_today_uses_real_clock() -> None:
    """Sanity: omitting today= falls back to date.today() (no crash)."""
    rng = extract_date_window("today")
    assert rng is not None
    assert rng[0] == rng[1]
