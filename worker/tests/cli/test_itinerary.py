"""Tests for the ``fieldnotes itinerary`` CLI command (fn-wbc.2).

End-to-end exercise of :mod:`worker.cli.itinerary` against a real Neo4j
instance, seeded per test from a single Cypher fixture.  Skipped
automatically when no Neo4j is reachable, mirroring the pattern in
``tests/query/test_itinerary.py``.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from datetime import date, datetime, timedelta, timezone
from typing import Any, Generator
from unittest.mock import patch

import pytest
from neo4j import Driver, GraphDatabase

from worker.cli import _build_parser
from worker.cli.itinerary import run_itinerary
from worker.config import CalendarAccountConfig, Config, Neo4jConfig

_NEO4J_URI = "bolt://localhost:7687"
_NEO4J_USER = "neo4j"
_NEO4J_PASSWORD = "testpassword"


def _neo4j_available() -> bool:
    try:
        with GraphDatabase.driver(_NEO4J_URI, auth=(_NEO4J_USER, _NEO4J_PASSWORD)) as d:
            d.verify_connectivity()
        return True
    except Exception:
        return False


_NEEDS_NEO4J = pytest.mark.skipif(
    not _neo4j_available(),
    reason="Neo4j not available at bolt://localhost:7687",
)


_TARGET_DAY = date(2026, 4, 27)
_NOW = datetime(2026, 4, 27, 12, 0, 0, tzinfo=timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _slack_ts(dt: datetime) -> str:
    return f"{dt.timestamp():.6f}"


def _make_config(*, with_accounts: bool = True) -> Config:
    cfg = Config(
        neo4j=Neo4jConfig(uri=_NEO4J_URI, user=_NEO4J_USER, password=_NEO4J_PASSWORD)
    )
    if with_accounts:
        cfg.google_calendar = {
            "work": CalendarAccountConfig(name="work"),
            "personal": CalendarAccountConfig(name="personal"),
        }
    return cfg


@pytest.fixture
def driver() -> Generator[Driver, None, None]:
    drv = GraphDatabase.driver(_NEO4J_URI, auth=(_NEO4J_USER, _NEO4J_PASSWORD))
    with drv.session() as s:
        s.run("MATCH (n) DETACH DELETE n")
    yield drv
    with drv.session() as s:
        s.run("MATCH (n) DETACH DELETE n")
    drv.close()


@pytest.fixture
def seeded(driver: Driver) -> dict[str, int]:
    """Seed events + linked tasks/threads on the target day.

    Mirrors the structure used by ``tests/query/test_itinerary.py`` so the
    CLI render assertions key off the same dataset.
    """
    work_meet_start = datetime(2026, 4, 27, 9, 0, 0, tzinfo=timezone.utc)
    work_meet_end = datetime(2026, 4, 27, 10, 0, 0, tzinfo=timezone.utc)
    crowded_start = datetime(2026, 4, 27, 11, 0, 0, tzinfo=timezone.utc)
    crowded_end = datetime(2026, 4, 27, 11, 30, 0, tzinfo=timezone.utc)
    lunch_start = datetime(2026, 4, 27, 12, 30, 0, tzinfo=timezone.utc)
    lunch_end = datetime(2026, 4, 27, 13, 30, 0, tzinfo=timezone.utc)
    next_day_start = datetime(2026, 4, 28, 9, 0, 0, tzinfo=timezone.utc)
    next_day_end = datetime(2026, 4, 28, 10, 0, 0, tzinfo=timezone.utc)

    cypher = """
    MERGE (me:Person {email: 'me@example.com'})
      SET me.name = 'Me Self', me.is_self = true
    MERGE (alice:Person {email: 'alice@example.com'})
      SET alice.name = 'Alice Example'
    MERGE (bob:Person {email: 'bob@example.com'})
      SET bob.name = 'Bob Builder'
    MERGE (carol:Person {email: 'carol@example.com'})
      SET carol.name = 'Carol Stranger'
    MERGE (dan:Person {email: 'dan@example.com'})
      SET dan.name = 'Dan Doer'
    MERGE (eve:Person {email: 'eve@example.com'})
      SET eve.name = 'Eve Engineer'
    MERGE (frank:Person {email: 'frank@example.com'})
      SET frank.name = 'Frank Founder'

    MERGE (work_meet:CalendarEvent {source_id: 'cal://work/work_meet'})
      SET work_meet.summary = 'Q2 sync',
          work_meet.start_time = $work_start,
          work_meet.end_time = $work_end,
          work_meet.account = 'work',
          work_meet.calendar_id = 'work/primary',
          work_meet.location = 'Conference Room A'
    MERGE (work_meet)-[:ORGANIZED_BY]->(me)
    MERGE (work_meet)-[:ATTENDED_BY]->(me)
    MERGE (work_meet)-[:ATTENDED_BY]->(alice)
    MERGE (work_meet)-[:ATTENDED_BY]->(bob)

    MERGE (lunch:CalendarEvent {source_id: 'cal://personal/lunch'})
      SET lunch.summary = 'Lunch with Alice',
          lunch.start_time = $lunch_start,
          lunch.end_time = $lunch_end,
          lunch.account = 'personal',
          lunch.calendar_id = 'personal/primary'
    MERGE (lunch)-[:ORGANIZED_BY]->(me)
    MERGE (lunch)-[:ATTENDED_BY]->(me)
    MERGE (lunch)-[:ATTENDED_BY]->(alice)

    MERGE (crowded:CalendarEvent {source_id: 'cal://work/crowded'})
      SET crowded.summary = 'All hands',
          crowded.start_time = $crowded_start,
          crowded.end_time = $crowded_end,
          crowded.account = 'work',
          crowded.calendar_id = 'work/primary'
    MERGE (crowded)-[:ORGANIZED_BY]->(me)
    MERGE (crowded)-[:ATTENDED_BY]->(me)
    MERGE (crowded)-[:ATTENDED_BY]->(alice)
    MERGE (crowded)-[:ATTENDED_BY]->(bob)
    MERGE (crowded)-[:ATTENDED_BY]->(carol)
    MERGE (crowded)-[:ATTENDED_BY]->(dan)
    MERGE (crowded)-[:ATTENDED_BY]->(eve)
    MERGE (crowded)-[:ATTENDED_BY]->(frank)

    MERGE (offday:CalendarEvent {source_id: 'cal://work/offday'})
      SET offday.summary = 'Tomorrow standup',
          offday.start_time = $offday_start,
          offday.end_time = $offday_end,
          offday.account = 'work'
    MERGE (offday)-[:ORGANIZED_BY]->(me)

    MERGE (proj:Project {source_id: 'omnifocus-project:Work'})
      SET proj.name = 'Work', proj.source = 'omnifocus'
    MERGE (t1:Task {source_id: 'of://t1'})
      SET t1.name = 'Email Alice about Q2',
          t1.status = 'active',
          t1.flagged = true,
          t1.due_date = $now_iso,
          t1.modification_date = $now_iso
    MERGE (t1)-[:IN_PROJECT]->(proj)
    MERGE (t1)-[:MENTIONS]->(alice)

    MERGE (thread_q2:Thread {source_id: 'gmail://work/thread/q2'})
      SET thread_q2.subject = 'Q2 planning thread'
    MERGE (em1:Email {source_id: 'gmail://work/em1'})
      SET em1.subject = 'Q2 planning', em1.date = $em_recent
    MERGE (em2:Email {source_id: 'gmail://work/em2'})
      SET em2.subject = 'Re: Q2 planning', em2.date = $em_recent2
    MERGE (em1)-[:PART_OF]->(thread_q2)
    MERGE (em2)-[:PART_OF]->(thread_q2)
    MERGE (em1)-[:TO]->(alice)
    MERGE (em2)-[:TO]->(bob)
    MERGE (me)-[:SENT]->(em1)
    MERGE (me)-[:SENT]->(em2)

    RETURN id(work_meet) AS work_meet,
           id(lunch) AS lunch,
           id(crowded) AS crowded,
           id(offday) AS offday
    """
    em_recent = _NOW - timedelta(days=2)
    em_recent2 = _NOW - timedelta(days=1)
    params = {
        "work_start": _iso(work_meet_start),
        "work_end": _iso(work_meet_end),
        "lunch_start": _iso(lunch_start),
        "lunch_end": _iso(lunch_end),
        "crowded_start": _iso(crowded_start),
        "crowded_end": _iso(crowded_end),
        "offday_start": _iso(next_day_start),
        "offday_end": _iso(next_day_end),
        "now_iso": _iso(_NOW),
        "em_recent": _iso(em_recent),
        "em_recent2": _iso(em_recent2),
    }
    with driver.session() as s:
        rec = s.run(cypher, **params).single()
        assert rec is not None
        return {k: int(v) for k, v in rec.data().items()}


def _run(*, with_accounts: bool = True, **kwargs: Any) -> tuple[int, str, str]:
    """Invoke :func:`run_itinerary` with a patched ``load_config`` and capture I/O.

    Defaults ``brief=True`` so the LLM path is skipped — these tests
    cover the non-LLM rendering and JSON schema.  LLM coverage lives in
    :mod:`worker.tests.cli.test_itinerary_summary`.
    """
    kwargs.setdefault("brief", True)
    cfg = _make_config(with_accounts=with_accounts)
    out = io.StringIO()
    err = io.StringIO()
    with patch("worker.cli.itinerary.load_config", return_value=cfg):
        with redirect_stdout(out), redirect_stderr(err):
            code = run_itinerary(**kwargs)
    return code, out.getvalue(), err.getvalue()


# ---------------------------------------------------------------------------
# Parser plumbing
# ---------------------------------------------------------------------------


class TestItineraryParser:
    def test_itinerary_default_day(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["itinerary"])
        assert args.command == "itinerary"
        assert args.day == "today"
        assert args.account is None
        assert args.brief is False
        assert args.horizon == "30d"
        assert args.json_output is False

    def test_itinerary_brief_flag_parses(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["itinerary", "--brief"])
        assert args.brief is True

    def test_itinerary_account_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["itinerary", "--account", "work"])
        assert args.account == "work"

    def test_itinerary_horizon_default_30d(self) -> None:
        """Default --horizon must be exactly '30d' (acceptance criterion)."""
        parser = _build_parser()
        args = parser.parse_args(["itinerary"])
        assert args.horizon == "30d"

    def test_itinerary_json_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["itinerary", "--json"])
        assert args.json_output is True


# ---------------------------------------------------------------------------
# Day-resolution and validation
# ---------------------------------------------------------------------------


def test_itinerary_invalid_day_exits_nonzero() -> None:
    code, _out, err = _run(day="garbage")
    assert code == 2
    assert "today" in err.lower() or "yyyy-mm-dd" in err.lower()


def test_itinerary_unknown_account_exits_nonzero_with_valid_list() -> None:
    code, _out, err = _run(day="2026-04-27", account="bogus")
    assert code == 2
    assert "bogus" in err
    # Configured accounts must be enumerated in the error message.
    assert "work" in err
    assert "personal" in err


# ---------------------------------------------------------------------------
# Rich rendering
# ---------------------------------------------------------------------------


@_NEEDS_NEO4J
def test_itinerary_renders_all_sections(seeded: dict[str, int]) -> None:
    code, out, _err = _run(day="2026-04-27")
    assert code == 0
    # Day-level header
    assert "Itinerary - " in out
    assert "2026-04-27" in out
    assert "events" in out
    # Event title rendered
    assert "Q2 sync" in out
    # Account label rendered
    assert "[work]" in out
    # Section labels
    assert "Attendees:" in out
    assert "Tasks:" in out or "Tasks: -" in out
    assert "Notes:" in out or "Notes: -" in out
    assert "Thread:" in out
    # Linked task seeded for work_meet must appear.
    assert "Email Alice about Q2" in out


@_NEEDS_NEO4J
def test_itinerary_specific_iso_date(seeded: dict[str, int]) -> None:
    code, out, _err = _run(day="2026-04-27")
    assert code == 0
    assert "Q2 sync" in out
    # The off-day event must NOT show up.
    assert "Tomorrow standup" not in out


@_NEEDS_NEO4J
def test_itinerary_empty_day_prints_no_events(seeded: dict[str, int]) -> None:
    code, out, _err = _run(day="2030-01-01")
    assert code == 0
    assert "No events scheduled." in out


@_NEEDS_NEO4J
def test_itinerary_account_filter_applies(seeded: dict[str, int]) -> None:
    code, out, _err = _run(day="2026-04-27", account="work")
    assert code == 0
    assert "Q2 sync" in out
    # personal-account event must be filtered out
    assert "Lunch with Alice" not in out


@_NEEDS_NEO4J
def test_itinerary_attendee_overflow_renders_plus_n_others(
    seeded: dict[str, int],
) -> None:
    code, out, _err = _run(day="2026-04-27")
    assert code == 0
    # "All hands" has 6 non-self attendees → "+3 others" overflow.
    assert "All hands" in out
    assert "+3 others" in out


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


_REQUIRED_EVENT_KEYS = {
    "event_id",
    "source_id",
    "title",
    "start",
    "end",
    "account",
    "calendar_id",
    "organizer",
    "attendees",
    "location",
    "html_link",
    "linked",
    "next_brief",
}


@_NEEDS_NEO4J
def test_itinerary_json_schema_stable(seeded: dict[str, int]) -> None:
    code, out, _err = _run(day="2026-04-27", json_output=True)
    assert code == 0
    payload = json.loads(out)  # roundtrips
    assert payload["day"] == "2026-04-27"
    assert isinstance(payload["timezone"], str) and payload["timezone"]
    assert isinstance(payload["events"], list)
    # At least three on-day events seeded (work_meet + lunch + crowded)
    assert len(payload["events"]) >= 3
    sample = payload["events"][0]
    assert _REQUIRED_EVENT_KEYS.issubset(sample.keys())
    assert {"tasks", "notes", "thread"} <= set(sample["linked"].keys())
    assert isinstance(sample["linked"]["tasks"], list)
    assert isinstance(sample["linked"]["notes"], list)
    # Locate the work_meet event and assert organizer + attendees shape.
    work_meet = next(e for e in payload["events"] if e["title"] == "Q2 sync")
    assert work_meet["account"] == "work"
    assert isinstance(work_meet["organizer"], dict)
    assert {"name", "email"} <= set(work_meet["organizer"].keys())
    assert work_meet["attendees"]
    for att in work_meet["attendees"]:
        assert {"name", "email"} <= set(att.keys())


@_NEEDS_NEO4J
def test_itinerary_json_brief_flag_emits_null_next_brief(
    seeded: dict[str, int],
) -> None:
    """With ``--brief``, no LLM is invoked and ``next_brief`` is null."""
    code, out, _err = _run(day="2026-04-27", brief=True, json_output=True)
    assert code == 0
    payload = json.loads(out)
    assert payload["events"]
    for ev in payload["events"]:
        assert ev["next_brief"] is None


@_NEEDS_NEO4J
def test_itinerary_json_empty_day_emits_empty_events(
    seeded: dict[str, int],
) -> None:
    code, out, _err = _run(day="2030-01-01", json_output=True)
    assert code == 0
    payload = json.loads(out)
    assert payload["events"] == []
    assert payload["day"] == "2030-01-01"


# ---------------------------------------------------------------------------
# --brief no-op behavior
# ---------------------------------------------------------------------------


@_NEEDS_NEO4J
def test_itinerary_brief_flag_skips_llm_and_keeps_schema(
    seeded: dict[str, int],
) -> None:
    """``--brief`` skips the LLM but produces the same schema shape."""
    code, out, _err = _run(day="2026-04-27", brief=True, json_output=True)
    assert code == 0
    payload = json.loads(out)
    assert payload["events"]
    for ev in payload["events"]:
        assert "next_brief" in ev
        assert ev["next_brief"] is None


# ---------------------------------------------------------------------------
# --horizon threading: passing a custom horizon must succeed
# ---------------------------------------------------------------------------


@_NEEDS_NEO4J
def test_itinerary_horizon_default_30d_runtime(seeded: dict[str, int]) -> None:
    """The default horizon (30d) and an explicit '30d' must produce the same
    output — guards against regressions where the parser default drifts.
    """
    code1, out1, _ = _run(day="2026-04-27", json_output=True)
    code2, out2, _ = _run(day="2026-04-27", horizon="30d", json_output=True)
    assert code1 == code2 == 0
    assert json.loads(out1) == json.loads(out2)
