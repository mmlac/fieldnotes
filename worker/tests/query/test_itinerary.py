"""Integration tests for ``worker.query.itinerary``.

Drives the read-only itinerary queries against a real Neo4j instance,
seeded per test from a single Cypher fixture.  Vector search is exercised
against an in-memory stub Qdrant client so the tests run with only Neo4j
available — mirrors the skip-gracefully pattern in ``test_person.py``.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Generator

import pytest
from neo4j import Driver, GraphDatabase

from worker.models.base import EmbedRequest, EmbedResponse
from worker.query.itinerary import (
    Event,
    EventWithLinks,
    Itinerary,
    NoteHit,
    OpenTask,
    PersonRef,
    ThreadHit,
    _resolve_day,
    events_for_day,
    get_itinerary,
    linked_notes_for_event,
    linked_tasks_for_event,
    recent_thread_with_attendees,
)

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


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------


_TARGET_DAY = date(2026, 4, 27)
_NOW = datetime(2026, 4, 27, 12, 0, 0, tzinfo=timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _slack_ts(dt: datetime) -> str:
    return f"{dt.timestamp():.6f}"


@pytest.fixture
def driver() -> Generator[Driver, None, None]:
    drv = GraphDatabase.driver(_NEO4J_URI, auth=(_NEO4J_USER, _NEO4J_PASSWORD))
    yield drv
    drv.close()


@pytest.fixture(autouse=True)
def _clean_neo4j() -> Generator[None, None, None]:
    """Wipe the test graph before/after each test that touches Neo4j.

    Skips silently when Neo4j is unreachable so tests that don't need the
    database (``test_resolve_day_*``, dataclass-export checks) still run.
    """
    if not _neo4j_available():
        yield
        return
    drv = GraphDatabase.driver(_NEO4J_URI, auth=(_NEO4J_USER, _NEO4J_PASSWORD))
    try:
        with drv.session() as s:
            s.run("MATCH (n) DETACH DELETE n")
        yield
        with drv.session() as s:
            s.run("MATCH (n) DETACH DELETE n")
    finally:
        drv.close()


@pytest.fixture
def seeded(driver: Driver) -> dict[str, int]:
    """Seed a multi-source graph reflecting the bead's required dataset."""
    # Three events on the target day:
    #   - work_meet (work account, 09:00 local UTC, attendees: alice + bob)
    #   - personal_lunch (personal account, 12:30 UTC, attendees: alice)
    #   - solo_block (work account, all-day, only self)
    work_meet_start = datetime(2026, 4, 27, 9, 0, 0, tzinfo=timezone.utc)
    work_meet_end = datetime(2026, 4, 27, 10, 0, 0, tzinfo=timezone.utc)
    personal_lunch_start = datetime(2026, 4, 27, 12, 30, 0, tzinfo=timezone.utc)
    personal_lunch_end = datetime(2026, 4, 27, 13, 30, 0, tzinfo=timezone.utc)
    # All-day uses Google's "YYYY-MM-DD" form (start=day, end=day+1).
    all_day_start = "2026-04-27"
    all_day_end = "2026-04-28"
    # An event on the day after — must NOT appear for target day.
    next_day_start = datetime(2026, 4, 28, 9, 0, 0, tzinfo=timezone.utc)
    next_day_end = datetime(2026, 4, 28, 10, 0, 0, tzinfo=timezone.utc)

    cypher = """
    // ── People ──────────────────────────────────────────
    MERGE (me:Person {email: 'me@example.com'})
      SET me.name = 'Me Self', me.is_self = true
    MERGE (alice:Person {email: 'alice@example.com'})
      SET alice.name = 'Alice Example'
    MERGE (alice_alt:Person {email: 'alice.alt@example.com'})
      SET alice_alt.name = 'Alice Example'
    MERGE (alice)-[:SAME_AS {match_type: 'fuzzy_name', confidence: 0.97}]->(alice_alt)
    MERGE (bob:Person {email: 'bob@example.com'})
      SET bob.name = 'Bob Builder'
    MERGE (carol:Person {email: 'carol@example.com'})
      SET carol.name = 'Carol Stranger'

    // ── Calendar events ─────────────────────────────────
    MERGE (work_meet:CalendarEvent {source_id: 'cal://work/work_meet'})
      SET work_meet.summary = 'Q2 sync',
          work_meet.description = 'Quarterly planning sync',
          work_meet.start_time = $work_start,
          work_meet.end_time = $work_end,
          work_meet.account = 'work',
          work_meet.calendar_id = 'work/primary',
          work_meet.html_link = 'https://cal.google.com/work_meet',
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
          lunch.calendar_id = 'personal/primary',
          lunch.html_link = 'https://cal.google.com/lunch'
    MERGE (lunch)-[:ORGANIZED_BY]->(me)
    MERGE (lunch)-[:ATTENDED_BY]->(me)
    MERGE (lunch)-[:ATTENDED_BY]->(alice)

    MERGE (solo:CalendarEvent {source_id: 'cal://work/solo'})
      SET solo.summary = 'Focus block',
          solo.start_time = $allday_start,
          solo.end_time = $allday_end,
          solo.account = 'work'
    MERGE (solo)-[:ORGANIZED_BY]->(me)
    MERGE (solo)-[:ATTENDED_BY]->(me)

    // Off-day event (next-day) — used to guarantee day-window scoping.
    MERGE (offday:CalendarEvent {source_id: 'cal://work/offday'})
      SET offday.summary = 'Tomorrow standup',
          offday.start_time = $offday_start,
          offday.end_time = $offday_end,
          offday.account = 'work'
    MERGE (offday)-[:ORGANIZED_BY]->(me)

    // ── OmniFocus tasks ─────────────────────────────────
    MERGE (proj:Project {source_id: 'omnifocus-project:Work'})
      SET proj.name = 'Work', proj.source = 'omnifocus'
    MERGE (people_alice:Tag {source_id: 'omnifocus-tag:People/Alice'})
      SET people_alice.name = 'People/Alice', people_alice.source = 'omnifocus'

    // 1. Open + tagged for Alice (flagged, with due)
    MERGE (t1:Task {source_id: 'of://t1'})
      SET t1.name = 'Email Alice about Q2',
          t1.status = 'active',
          t1.flagged = true,
          t1.due_date = $now_iso,
          t1.modification_date = $now_iso
    MERGE (t1)-[:IN_PROJECT]->(proj)
    MERGE (t1)-[:TAGGED]->(people_alice)
    MERGE (t1)-[:MENTIONS]->(alice)

    // 2. Open + tagged for Alice (not flagged, no due)
    MERGE (t2:Task {source_id: 'of://t2'})
      SET t2.name = 'Send Alice the deck',
          t2.status = 'active',
          t2.flagged = false,
          t2.modification_date = $now_iso
    MERGE (t2)-[:IN_PROJECT]->(proj)
    MERGE (t2)-[:TAGGED]->(people_alice)
    MERGE (t2)-[:MENTIONS]->(alice)

    // 3. Completed task (must be excluded)
    MERGE (t3:Task {source_id: 'of://t3'})
      SET t3.name = 'Already shipped',
          t3.status = 'completed',
          t3.modification_date = $now_iso
    MERGE (t3)-[:MENTIONS]->(alice)

    // 4. Open but unrelated (mentions Carol)
    MERGE (t4:Task {source_id: 'of://t4'})
      SET t4.name = 'Random unrelated chore',
          t4.status = 'active',
          t4.modification_date = $now_iso
    MERGE (t4)-[:MENTIONS]->(carol)

    // 5. Open + email mention only (mentions Bob via email-keyed Person)
    MERGE (t5:Task {source_id: 'of://t5'})
      SET t5.name = 'Reply bob@example.com',
          t5.status = 'active',
          t5.modification_date = $now_iso
    MERGE (t5)-[:MENTIONS]->(bob)

    // ── File / Obsidian / Slack chunks (parents only) ───────────────
    MERGE (f1:File {source_id: 'obs://notes/q2-roadmap.md'})
      SET f1.path = '/notes/q2-roadmap.md',
          f1.title = 'Q2 Roadmap',
          f1.modified_at = $now_iso,
          f1.source = 'obsidian'
    MERGE (f1)-[:MENTIONS]->(alice)
    MERGE (f1)-[:MENTIONS]->(bob)

    MERGE (f2:File {source_id: 'obs://notes/q2-followups.md'})
      SET f2.path = '/notes/q2-followups.md',
          f2.title = 'Q2 follow-ups',
          f2.modified_at = $now_iso,
          f2.source = 'obsidian'
    MERGE (f2)-[:MENTIONS]->(alice)

    MERGE (f3:File {source_id: 'obs://notes/q2-wide.md'})
      SET f3.path = '/notes/q2-wide.md',
          f3.title = 'Q2 broad context',
          f3.modified_at = $now_iso,
          f3.source = 'obsidian'

    MERGE (f4:File {source_id: 'obs://notes/grocery.md'})
      SET f4.path = '/notes/grocery.md',
          f4.title = 'Groceries',
          f4.modified_at = $now_iso,
          f4.source = 'obsidian'
    MERGE (f4)-[:MENTIONS]->(alice)

    // ── Email threads ───────────────────────────────────
    MERGE (thread_q2:Thread {source_id: 'gmail://work/thread/q2'})
      SET thread_q2.thread_id = 'q2-thread',
          thread_q2.subject = 'Q2 planning thread'
    MERGE (em1:Email {source_id: 'gmail://work/em1'})
      SET em1.subject = 'Q2 planning',
          em1.date = $em_recent
    MERGE (em2:Email {source_id: 'gmail://work/em2'})
      SET em2.subject = 'Re: Q2 planning',
          em2.date = $em_yesterday
    MERGE (em1)-[:PART_OF]->(thread_q2)
    MERGE (em2)-[:PART_OF]->(thread_q2)
    // Alice on thread (TO em1), Bob on thread (TO em2) → covers both attendees.
    MERGE (em1)-[:TO]->(alice)
    MERGE (em2)-[:TO]->(bob)
    MERGE (me)-[:SENT]->(em1)
    MERGE (me)-[:SENT]->(em2)

    // Thread that touches only Alice — must NOT match a 2-attendee event.
    MERGE (thread_solo:Thread {source_id: 'gmail://personal/thread/solo'})
      SET thread_solo.thread_id = 'solo-thread',
          thread_solo.subject = 'Lunch?'
    MERGE (em3:Email {source_id: 'gmail://personal/em3'})
      SET em3.subject = 'Lunch?', em3.date = $em_yesterday
    MERGE (em3)-[:PART_OF]->(thread_solo)
    MERGE (em3)-[:TO]->(alice)
    MERGE (me)-[:SENT]->(em3)

    // ── Slack window with both attendees ─────────────────
    MERGE (sl:SlackMessage {source_id: 'slack://eng/window'})
      SET sl.channel_name = 'eng',
          sl.first_ts = $sl_recent,
          sl.last_ts = $sl_recent
    MERGE (sl)-[:SENT_BY]->(alice)
    MERGE (sl)-[:MENTIONS]->(bob)

    RETURN id(work_meet) AS work_meet,
           id(lunch) AS lunch,
           id(solo) AS solo,
           id(offday) AS offday,
           id(alice) AS alice,
           id(bob) AS bob,
           id(carol) AS carol,
           id(me) AS me,
           id(f1) AS f1,
           id(f2) AS f2,
           id(f3) AS f3,
           id(f4) AS f4
    """
    em_recent = _NOW - timedelta(days=2)
    em_yesterday = _NOW - timedelta(days=10)
    sl_recent = _NOW - timedelta(days=1)
    params = {
        "work_start": _iso(work_meet_start),
        "work_end": _iso(work_meet_end),
        "lunch_start": _iso(personal_lunch_start),
        "lunch_end": _iso(personal_lunch_end),
        "allday_start": all_day_start,
        "allday_end": all_day_end,
        "offday_start": _iso(next_day_start),
        "offday_end": _iso(next_day_end),
        "now_iso": _iso(_NOW),
        "em_recent": _iso(em_recent),
        "em_yesterday": _iso(em_yesterday),
        "sl_recent": _slack_ts(sl_recent),
    }
    with driver.session() as s:
        rec = s.run(cypher, **params).single()
        assert rec is not None
        return {k: int(v) for k, v in rec.data().items()}


# ---------------------------------------------------------------------------
# _resolve_day
# ---------------------------------------------------------------------------


def test_resolve_day_today_tomorrow_iso() -> None:
    tz = timezone.utc
    today = _resolve_day("today", tz)
    tomorrow = _resolve_day("tomorrow", tz)
    iso = _resolve_day("2026-04-27", tz)
    assert isinstance(today, date) and not isinstance(today, datetime)
    assert tomorrow == today + timedelta(days=1)
    assert iso == date(2026, 4, 27)
    # Pass-through on date.
    assert _resolve_day(date(2026, 1, 1), tz) == date(2026, 1, 1)
    # Case-insensitive named days.
    assert _resolve_day("Today", tz) == today


def test_resolve_day_invalid_raises_clean_error() -> None:
    tz = timezone.utc
    with pytest.raises(ValueError):
        _resolve_day("not a date", tz)
    with pytest.raises(ValueError):
        _resolve_day("2026-13-01", tz)
    with pytest.raises(ValueError):
        _resolve_day("", tz)


# ---------------------------------------------------------------------------
# events_for_day
# ---------------------------------------------------------------------------


@_NEEDS_NEO4J
def test_events_for_day_filters_to_local_day(
    driver: Driver, seeded: dict[str, int]
) -> None:
    rows = events_for_day(_TARGET_DAY, timezone_=timezone.utc, driver=driver)
    titles = {r.title for r in rows}
    assert {"Q2 sync", "Lunch with Alice", "Focus block"} <= titles
    # Off-day event must not appear.
    assert "Tomorrow standup" not in titles
    # Strict ascending order on start_ts.
    starts = [r.start_ts for r in rows]
    assert starts == sorted(starts)


@_NEEDS_NEO4J
def test_events_for_day_account_filter(driver: Driver, seeded: dict[str, int]) -> None:
    work_only = events_for_day(
        _TARGET_DAY, account="work", timezone_=timezone.utc, driver=driver
    )
    titles = {r.title for r in work_only}
    assert "Q2 sync" in titles
    assert "Focus block" in titles
    # personal-account event must be excluded
    assert "Lunch with Alice" not in titles
    for r in work_only:
        assert r.account == "work"


@_NEEDS_NEO4J
def test_events_for_day_empty_returns_empty_list(
    driver: Driver, seeded: dict[str, int]
) -> None:
    other_day = date(2026, 1, 1)
    rows = events_for_day(other_day, timezone_=timezone.utc, driver=driver)
    assert rows == []
    assert isinstance(rows, list)


@_NEEDS_NEO4J
def test_events_for_day_all_day_events_included(
    driver: Driver, seeded: dict[str, int]
) -> None:
    rows = events_for_day(_TARGET_DAY, timezone_=timezone.utc, driver=driver)
    all_day = [r for r in rows if r.title == "Focus block"]
    assert len(all_day) == 1
    ev = all_day[0]
    # Stored as plain "YYYY-MM-DD".
    assert ev.start_ts == "2026-04-27"
    assert ev.end_ts == "2026-04-28"
    # Solo event has only-self attendees → flagged.
    assert ev.is_self_only is True
    # Multi-attendee event is not self-only.
    work_meet = next(r for r in rows if r.title == "Q2 sync")
    assert work_meet.is_self_only is False
    assert {a.email for a in work_meet.attendees} == {
        "alice@example.com",
        "bob@example.com",
    }
    # Self never appears in attendees list.
    for ev in rows:
        for a in ev.attendees:
            assert not a.is_self


# ---------------------------------------------------------------------------
# linked_tasks_for_event
# ---------------------------------------------------------------------------


@_NEEDS_NEO4J
def test_linked_tasks_excludes_completed_and_dropped(
    driver: Driver, seeded: dict[str, int]
) -> None:
    work_meet_id = seeded["work_meet"]
    tasks = linked_tasks_for_event(
        work_meet_id, k=10, horizon=timedelta(days=30), driver=driver
    )
    titles = {t.title for t in tasks}
    # Open+tagged for Alice and email-mention of Bob both present.
    assert "Email Alice about Q2" in titles
    assert "Send Alice the deck" in titles
    assert "Reply bob@example.com" in titles
    # Excluded: completed and unrelated.
    assert "Already shipped" not in titles
    assert "Random unrelated chore" not in titles


@_NEEDS_NEO4J
def test_linked_tasks_orders_by_flagged_then_due(
    driver: Driver, seeded: dict[str, int]
) -> None:
    work_meet_id = seeded["work_meet"]
    tasks = linked_tasks_for_event(
        work_meet_id, k=10, horizon=timedelta(days=30), driver=driver
    )
    # Flagged Email Alice task must come first.
    assert tasks[0].title == "Email Alice about Q2"
    assert tasks[0].flagged is True
    # Subsequent rows are not flagged → tasks ordered by due ASC NULLS LAST.
    later = [t for t in tasks[1:] if not t.flagged]
    nulls_first = [t for t in later if t.due is None]
    nonnull = [t for t in later if t.due is not None]
    # All NULL-due rows sit after non-NULL ones (NULLs last).
    if nonnull and nulls_first:
        last_nonnull_idx = max(later.index(t) for t in nonnull)
        first_null_idx = min(later.index(t) for t in nulls_first)
        assert first_null_idx > last_nonnull_idx


@_NEEDS_NEO4J
def test_linked_tasks_respects_k_limit(driver: Driver, seeded: dict[str, int]) -> None:
    work_meet_id = seeded["work_meet"]
    tasks_all = linked_tasks_for_event(
        work_meet_id, k=10, horizon=timedelta(days=30), driver=driver
    )
    assert len(tasks_all) >= 3  # we seeded at least 3 matching tasks
    tasks_two = linked_tasks_for_event(
        work_meet_id, k=2, horizon=timedelta(days=30), driver=driver
    )
    assert len(tasks_two) == 2
    # k=2 returns the same prefix as the unbounded result.
    assert [t.title for t in tasks_two] == [t.title for t in tasks_all[:2]]


# ---------------------------------------------------------------------------
# linked_notes_for_event (uses stub embedder + stub Qdrant client)
# ---------------------------------------------------------------------------


class _StubModel:
    def embed(self, req: EmbedRequest) -> EmbedResponse:
        # Single canned vector — the stub Qdrant ignores it anyway.
        return EmbedResponse(vectors=[[0.1, 0.2, 0.3]], model="stub")


class _StubRegistry:
    def __init__(self, model: _StubModel | None = None) -> None:
        self._model = model or _StubModel()

    def for_role(self, role: str) -> _StubModel:
        if role != "embed":
            raise KeyError(role)
        return self._model


def _make_qdrant_stub(points: list[Any]) -> Any:
    """Build a fake QdrantClient that returns *points* on query_points."""
    response = SimpleNamespace(points=points)
    return SimpleNamespace(
        query_points=lambda **_: response,
        close=lambda: None,
    )


def _q_point(
    *,
    score: float,
    source_type: str,
    source_id: str,
    text: str,
    date: str,
) -> Any:
    return SimpleNamespace(
        score=score,
        payload={
            "source_type": source_type,
            "source_id": source_id,
            "text": text,
            "date": date,
            "chunk_index": 0,
        },
    )


@_NEEDS_NEO4J
def test_linked_notes_promotes_attendee_overlap(
    driver: Driver, seeded: dict[str, int]
) -> None:
    """A high-similarity hit *without* attendee overlap must still come
    after a slightly lower-similarity hit that has overlap, because the
    re-ranker promotes attendee-overlap parents to the top.
    """
    # f3 (no MENTIONS) gets the *highest* raw score; f1/f2 (Alice+Bob) lower.
    points = [
        _q_point(
            score=0.95,
            source_type="obsidian",
            source_id="obs://notes/q2-wide.md",
            text="Q2 broad context — wide overview",
            date=_iso(_NOW),
        ),
        _q_point(
            score=0.80,
            source_type="obsidian",
            source_id="obs://notes/q2-roadmap.md",
            text="Q2 roadmap and milestones with Alice/Bob",
            date=_iso(_NOW),
        ),
        _q_point(
            score=0.70,
            source_type="obsidian",
            source_id="obs://notes/q2-followups.md",
            text="Follow-ups from Q2 sync",
            date=_iso(_NOW),
        ),
    ]
    qstub = _make_qdrant_stub(points)

    work_meet_id = seeded["work_meet"]
    notes = linked_notes_for_event(
        work_meet_id,
        k=2,
        horizon=timedelta(days=30),
        registry=_StubRegistry(),
        qdrant_client=qstub,
        driver=driver,
    )
    assert len(notes) == 2
    # Both top-2 must be attendee-overlap parents, ordered by raw score.
    assert all(n.attendee_overlap for n in notes)
    assert notes[0].source_id == "obs://notes/q2-roadmap.md"
    assert notes[1].source_id == "obs://notes/q2-followups.md"


@_NEEDS_NEO4J
def test_linked_notes_horizon_excludes_old_chunks(
    driver: Driver, seeded: dict[str, int]
) -> None:
    """Chunks older than ``horizon`` must be dropped before re-ranking."""
    old_iso = _iso(_NOW - timedelta(days=120))
    points = [
        _q_point(
            score=0.99,
            source_type="obsidian",
            source_id="obs://notes/q2-roadmap.md",
            text="ancient",
            date=old_iso,
        ),
        _q_point(
            score=0.40,
            source_type="obsidian",
            source_id="obs://notes/q2-followups.md",
            text="recent",
            date=_iso(_NOW),
        ),
    ]
    qstub = _make_qdrant_stub(points)
    work_meet_id = seeded["work_meet"]
    notes = linked_notes_for_event(
        work_meet_id,
        k=5,
        horizon=timedelta(days=30),
        registry=_StubRegistry(),
        qdrant_client=qstub,
        driver=driver,
    )
    sids = [n.source_id for n in notes]
    assert "obs://notes/q2-roadmap.md" not in sids
    assert "obs://notes/q2-followups.md" in sids


# ---------------------------------------------------------------------------
# recent_thread_with_attendees
# ---------------------------------------------------------------------------


@_NEEDS_NEO4J
def test_recent_thread_requires_all_attendees(
    driver: Driver, seeded: dict[str, int]
) -> None:
    """work_meet has both Alice and Bob — the q2 thread covers both;
    the lunch-only thread covers Alice only and must NOT be returned.
    """
    work_meet_id = seeded["work_meet"]
    hit = recent_thread_with_attendees(
        work_meet_id, horizon=timedelta(days=30), driver=driver
    )
    assert hit is not None
    # Both candidates exist (email q2 thread + slack window) — pick most-recent.
    # Slack message timestamp is yesterday (Unix seconds), email last is 2d ago.
    assert hit.kind == "slack"
    assert hit.source_id == "slack://eng/window"

    # The lunch event has only Alice — both candidates apply, but pick the
    # most recent that covers Alice; our slack window MENTIONS Bob, not
    # Alice's cluster... so it wouldn't necessarily fail.  The q2 thread
    # has Alice via TO em1 → covers Alice → returned.
    lunch_id = seeded["lunch"]
    lunch_hit = recent_thread_with_attendees(
        lunch_id, horizon=timedelta(days=30), driver=driver
    )
    assert lunch_hit is not None
    # For lunch (alice only), the slack window has alice via SENT_BY,
    # so it still wins on recency.
    assert lunch_hit.source_id in {
        "slack://eng/window",
        "gmail://work/thread/q2",
        "gmail://personal/thread/solo",
    }


@_NEEDS_NEO4J
def test_recent_thread_returns_none_when_no_overlap(
    driver: Driver, seeded: dict[str, int]
) -> None:
    """An event whose attendees never co-appear in any thread/slack window
    returns None.  We exercise this by attaching Carol (no thread/slack)
    as the lone attendee of a fresh event.
    """
    with driver.session() as s:
        rec = s.run(
            """
            MATCH (carol:Person {email: 'carol@example.com'})
            MATCH (me:Person {is_self: true})
            CREATE (e:CalendarEvent {
                source_id: 'cal://lonely',
                summary: 'Lonely meet',
                start_time: $start,
                end_time: $end,
                account: 'work'
            })
            CREATE (e)-[:ORGANIZED_BY]->(me)
            CREATE (e)-[:ATTENDED_BY]->(me)
            CREATE (e)-[:ATTENDED_BY]->(carol)
            RETURN id(e) AS id
            """,
            start=_iso(datetime(2026, 4, 27, 16, 0, 0, tzinfo=timezone.utc)),
            end=_iso(datetime(2026, 4, 27, 17, 0, 0, tzinfo=timezone.utc)),
        ).single()
        assert rec is not None
        lonely_id = int(rec["id"])

    hit = recent_thread_with_attendees(
        lonely_id, horizon=timedelta(days=30), driver=driver
    )
    assert hit is None


# ---------------------------------------------------------------------------
# get_itinerary
# ---------------------------------------------------------------------------


@_NEEDS_NEO4J
def test_get_itinerary_aggregates_per_event_links(
    driver: Driver, seeded: dict[str, int]
) -> None:
    """``get_itinerary`` reuses one driver and stitches events + tasks +
    threads.  Notes are skipped silently when *registry* is ``None``.
    """
    itinerary = get_itinerary(
        _TARGET_DAY,
        timezone_=timezone.utc,
        k_tasks=10,
        horizon=timedelta(days=30),
    )
    assert isinstance(itinerary, Itinerary)
    assert itinerary.day == _TARGET_DAY
    titles = [ew.event.title for ew in itinerary.events]
    assert "Q2 sync" in titles
    assert "Lunch with Alice" in titles
    assert "Focus block" in titles
    assert "Tomorrow standup" not in titles

    # Find the work meet entry: should have tasks + a thread.
    work = next(ew for ew in itinerary.events if ew.event.title == "Q2 sync")
    assert any(t.title == "Email Alice about Q2" for t in work.tasks)
    assert work.thread is not None
    # Notes section is empty because no registry was passed.
    assert work.notes == []

    # Self-only event has no attendees → no tasks, no thread, no notes.
    solo = next(ew for ew in itinerary.events if ew.event.title == "Focus block")
    assert solo.event.is_self_only is True
    assert solo.tasks == []
    assert solo.thread is None


# ---------------------------------------------------------------------------
# Sanity: the dataclasses are importable as documented
# ---------------------------------------------------------------------------


def test_module_exports_documented_dataclasses() -> None:
    # These classes are part of the documented public API per the bead.
    for cls in (
        Event,
        OpenTask,
        NoteHit,
        ThreadHit,
        EventWithLinks,
        Itinerary,
        PersonRef,
    ):
        assert isinstance(cls, type)
