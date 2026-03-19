"""Shared time-parsing utilities for fieldnotes query modules."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

# Relative time patterns: "24h", "7d", "2w", "3m"
_RELATIVE_RE = re.compile(
    r"^(\d+)\s*(h|hr|hour|hours|d|day|days|w|week|weeks|m|mo|month|months)$",
    re.IGNORECASE,
)


def parse_relative_time(s: str) -> datetime:
    """Parse a relative or absolute time string into a UTC datetime.

    Supports:
    - Relative: "24h", "7d", "2w", "3m"
    - Named: "yesterday", "last week"
    - ISO 8601 strings: "2026-03-01", "2026-03-01T12:00:00Z"
    - Special: "now"
    """
    s = s.strip().lower()
    now = datetime.now(timezone.utc)

    if s in ("now", ""):
        return now

    if s == "yesterday":
        return (now - timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

    if s == "last week":
        return now - timedelta(weeks=1)

    m = _RELATIVE_RE.match(s)
    if m:
        n = int(m.group(1))
        unit = m.group(2).lower()
        if unit in ("h", "hr", "hour", "hours"):
            return now - timedelta(hours=n)
        if unit in ("d", "day", "days"):
            return now - timedelta(days=n)
        if unit in ("w", "week", "weeks"):
            return now - timedelta(weeks=n)
        if unit in ("m", "mo", "month", "months"):
            # Approximate months as 30 days each.
            return now - timedelta(days=n * 30)

    # Try ISO 8601 parsing (date or datetime).
    for fmt in (
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d",
    ):
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue

    raise ValueError(
        f"Cannot parse time string: {s!r}. "
        "Use ISO 8601, a relative value like '24h'/'7d', or 'now'/'yesterday'."
    )
