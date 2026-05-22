"""Heuristic extractor: pull a date window out of a free-form question.

Used by the ask pipeline to:
  - Tighten the system prompt with explicit date bounds.
  - Post-filter vector results so off-window noise (e.g. a six-month-old
    PDF that matches "journal" semantically) doesn't reach the LLM.

Recognised patterns (case-insensitive):
    today, today's
    yesterday, yesterday's
    this week | this month | this year
    last week | last month | last year
    last N day(s)/week(s)/month(s)/year(s)        — N is a digit or
                                                    word number (one..ten,
                                                    fourteen, thirty)
    past/previous N <unit>                         — same shape
    in the last N <unit>                           — same shape

Returned ranges are inclusive on both ends.

Ambiguities resolved by convention:
  - "last week"  = the rolling 7 days ending today (today − 6, today).
  - "this week" = the ISO-week-to-date (Monday of this week .. today).
The first matches what people usually mean when summarising journals;
the second is the canonical calendar form for status reports.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

# Word-number forms we accept. Keeping this small on purpose — if a user
# types something exotic, they can use digits.
_WORD_NUM: dict[str, int] = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "fourteen": 14,
    "thirty": 30,
}

_NUM_PAT = r"\d+|" + "|".join(_WORD_NUM)

_RE_LAST_N = re.compile(
    rf"\b(?:in\s+the\s+)?(?:last|past|previous)\s+"
    rf"(?:({_NUM_PAT})\s+)?"
    r"(day|days|week|weeks|month|months|year|years)\b",
    re.IGNORECASE,
)
_RE_THIS = re.compile(r"\bthis\s+(week|month|year)\b", re.IGNORECASE)
_RE_TODAY = re.compile(r"\btoday(?:'s)?\b", re.IGNORECASE)
_RE_YESTERDAY = re.compile(r"\byesterday(?:'s)?\b", re.IGNORECASE)


def extract_date_window(
    question: str, today: date | None = None
) -> tuple[date, date] | None:
    """If *question* contains a recognised time constraint, return ``(start, end)``.

    Both bounds are inclusive. Returns ``None`` when no time hint is detected.
    Caller decides what to do with the window — typically filter vector results
    and/or pass the bounds to the LLM via the system prompt.
    """
    if today is None:
        today = date.today()
    q = question.strip()

    if _RE_TODAY.search(q):
        return (today, today)
    if _RE_YESTERDAY.search(q):
        d = today - timedelta(days=1)
        return (d, d)

    m = _RE_THIS.search(q)
    if m:
        unit = m.group(1).lower()
        if unit == "week":
            start = today - timedelta(days=today.weekday())  # Monday
        elif unit == "month":
            start = today.replace(day=1)
        elif unit == "year":
            start = today.replace(month=1, day=1)
        else:
            return None
        return (start, today)

    m = _RE_LAST_N.search(q)
    if m:
        n_raw = m.group(1)
        unit = m.group(2).lower()
        if n_raw is None:
            n = 1  # bare "last week" / "last month" / "last year"
        elif n_raw.isdigit():
            n = int(n_raw)
        else:
            n = _WORD_NUM.get(n_raw.lower(), 1)
        if unit.startswith("day"):
            days = n
        elif unit.startswith("week"):
            days = n * 7
        elif unit.startswith("month"):
            days = n * 30
        elif unit.startswith("year"):
            days = n * 365
        else:
            return None
        return (today - timedelta(days=days - 1), today)

    return None


def mentions_journal(question: str) -> bool:
    """Return True if the question explicitly refers to journal entries.

    Used by the vector post-filter: when the user is asking about journals
    specifically, drop vector results that don't live in a journal folder
    (per the configured ``[retrieval] journal_folder_patterns``).
    """
    return bool(re.search(r"\bjournal(s|ing|ed|\b)", question, re.IGNORECASE))
