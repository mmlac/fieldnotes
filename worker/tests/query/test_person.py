"""Integration tests for ``worker.query.person``.

Drives the read-only profile queries against a real Neo4j instance,
seeded per test from a single Cypher fixture.  Skipped automatically
when no Neo4j is reachable (mirrors ``tests/test_slack_e2e.py``).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Generator

import pytest
from neo4j import Driver, GraphDatabase

from worker.query.person import (
    Person,
    PersonProfile,
    files_mentioning,
    find_person,
    get_profile,
    identity_cluster,
    open_tasks,
    recent_interactions,
    related_people,
    top_topics,
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


pytestmark = pytest.mark.skipif(
    not _neo4j_available(),
    reason="Neo4j not available at bolt://localhost:7687",
)


# ---------------------------------------------------------------------------
# Reference timestamps used throughout the seed
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 4, 27, 12, 0, 0, tzinfo=timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# Slack-style ts: Unix seconds with .micro fractional part.
def _slack_ts(dt: datetime) -> str:
    return f"{dt.timestamp():.6f}"


@pytest.fixture
def driver() -> Generator[Driver, None, None]:
    drv = GraphDatabase.driver(_NEO4J_URI, auth=(_NEO4J_USER, _NEO4J_PASSWORD))
    yield drv
    drv.close()


@pytest.fixture(autouse=True)
def _clean_neo4j(driver: Driver) -> Generator[None, None, None]:
    with driver.session() as s:
        s.run("MATCH (n) DETACH DELETE n")
    yield
    with driver.session() as s:
        s.run("MATCH (n) DETACH DELETE n")


@pytest.fixture
def seeded(driver: Driver) -> dict[str, int]:
    """Seed the graph and return a map of named-anchor → Neo4j internal id."""
    t_now = _NOW
    t_yesterday = t_now - timedelta(days=1)
    t_2d = t_now - timedelta(days=2)
    t_3d = t_now - timedelta(days=3)
    t_4d = t_now - timedelta(days=4)
    t_5d = t_now - timedelta(days=5)
    t_2w = t_now - timedelta(days=14)
    t_3mo = t_now - timedelta(days=90)

    cypher = """
    // Subject Person — has two emails linked by SAME_AS, plus a Slack identity
    MERGE (p_main:Person {email: 'alice@example.com'})
      SET p_main.name = 'Alice Example',
          p_main.slack_user_id = 'U-ALICE',
          p_main.team_id = 'T-TEAM'
    MERGE (p_alt:Person {email: 'alice.alt@example.com'})
      SET p_alt.name = 'Alice Example'
    MERGE (p_main)-[r1:SAME_AS]->(p_alt)
      SET r1.match_type = 'fuzzy_name',
          r1.confidence = 0.97,
          r1.cross_source = true

    // A related Person who shares 3 documents with Alice
    MERGE (p_bob:Person {email: 'bob@example.com'})
      SET p_bob.name = 'Bob Builder'

    // Two unrelated Persons (one fuzzy-name conflict, one stranger)
    MERGE (p_alicia:Person {email: 'alicia@example.com'})
      SET p_alicia.name = 'Alicia Example'
    MERGE (p_stranger:Person {email: 'stranger@example.com'})
      SET p_stranger.name = 'Stranger Person'

    // Emails — 5 spanning multiple sources/dates
    MERGE (e1:Email {source_id: 'gmail://1'})
      SET e1.subject = 'Project kickoff',  e1.date = $d_now
    MERGE (e2:Email {source_id: 'gmail://2'})
      SET e2.subject = 'Re: weekly sync',  e2.date = $d_y
    MERGE (e3:Email {source_id: 'gmail://3'})
      SET e3.subject = 'Travel plans',     e3.date = $d_2d
    MERGE (e4:Email {source_id: 'gmail://4'})
      SET e4.subject = 'Old thread',       e4.date = $d_3mo
    MERGE (e5:Email {source_id: 'gmail://5'})
      SET e5.subject = 'Coffee?',          e5.date = $d_3d

    MERGE (p_main)-[:SENT]->(e1)
    MERGE (e2)-[:TO]->(p_main)
    MERGE (e3)-[:MENTIONS]->(p_alt)
    MERGE (e4)-[:TO]->(p_main)
    MERGE (e5)-[:TO]->(p_main)
    MERGE (e2)-[:TO]->(p_bob)
    MERGE (e1)-[:TO]->(p_bob)

    // Calendar events — 1 organized, 1 attended
    MERGE (cal1:CalendarEvent {source_id: 'cal://1'})
      SET cal1.summary = 'Q2 planning',  cal1.start_time = $d_2d
    MERGE (cal2:CalendarEvent {source_id: 'cal://2'})
      SET cal2.summary = '1:1 with Bob', cal2.start_time = $d_4d
    MERGE (cal1)-[:ORGANIZED_BY]->(p_main)
    MERGE (cal2)-[:ATTENDED_BY]->(p_main)
    MERGE (cal2)-[:ORGANIZED_BY]->(p_bob)

    // Slack — 1 authored, 2 mentioning
    MERGE (s1:SlackMessage {source_id: 'slack://1'})
      SET s1.channel_name = 'eng', s1.first_ts = $sl_y, s1.last_ts = $sl_y
    MERGE (s2:SlackMessage {source_id: 'slack://2'})
      SET s2.channel_name = 'eng', s2.first_ts = $sl_2d, s2.last_ts = $sl_2d
    MERGE (s3:SlackMessage {source_id: 'slack://3'})
      SET s3.channel_name = 'random', s3.first_ts = $sl_3d, s3.last_ts = $sl_3d
    MERGE (s1)-[:SENT_BY]->(p_main)
    MERGE (s2)-[:MENTIONS]->(p_main)
    MERGE (s3)-[:MENTIONS]->(p_alt)
    MERGE (s2)-[:SENT_BY]->(p_bob)

    // OmniFocus tasks — 1 open + tagged, 1 completed
    MERGE (t_open:Task {source_id: 'of://open'})
      SET t_open.name = 'Email Alice about Q2',
          t_open.status = 'active',
          t_open.flagged = true,
          t_open.due_date = $d_y,
          t_open.defer_date = $d_now
    MERGE (t_drop:Task {source_id: 'of://dropped'})
      SET t_drop.name = 'Drop me',
          t_drop.status = 'dropped'
    MERGE (t_done:Task {source_id: 'of://done'})
      SET t_done.name = 'Already shipped',
          t_done.status = 'completed'
    MERGE (proj:Project {source_id: 'omnifocus-project:Work'})
      SET proj.name = 'Work', proj.source = 'omnifocus'
    MERGE (tag:Tag {source_id: 'omnifocus-tag:urgent'})
      SET tag.name = 'urgent', tag.source = 'omnifocus'
    MERGE (t_open)-[:IN_PROJECT]->(proj)
    MERGE (t_open)-[:TAGGED]->(tag)
    MERGE (t_open)-[:MENTIONS]->(p_main)
    MERGE (t_drop)-[:MENTIONS]->(p_main)
    MERGE (t_done)-[:MENTIONS]->(p_main)

    // Topic linked to several docs
    MERGE (topic:Topic {name: 'Q2 planning'})
      SET topic.source = 'user'
    MERGE (e1)-[:TAGGED]->(topic)
    MERGE (cal1)-[:TAGGED]->(topic)
    MERGE (s1)-[:TAGGED]->(topic)
    MERGE (topic_other:Topic {name: 'Travel'})
      SET topic_other.source = 'user'
    MERGE (e3)-[:TAGGED]->(topic_other)

    // Files — File MENTIONS Person, ordered by modified_at
    MERGE (f1:File {source_id: '/notes/a.md'})
      SET f1.path = '/notes/a.md', f1.modified_at = $d_now,  f1.source = 'obsidian'
    MERGE (f2:File {source_id: '/notes/b.md'})
      SET f2.path = '/notes/b.md', f2.modified_at = $d_2w,  f2.source = 'obsidian'
    MERGE (f3:File {source_id: '/notes/c.md'})
      SET f3.path = '/notes/c.md', f3.modified_at = $d_5d,  f3.source = 'obsidian'
    MERGE (f1)-[:MENTIONS]->(p_main)
    MERGE (f2)-[:MENTIONS]->(p_main)
    MERGE (f3)-[:MENTIONS]->(p_alt)

    // Bob shares 3 docs with Alice (e1, e2 by TO, cal2 by ORGANIZED_BY)
    // Plus s2 (SENT_BY): Bob authored a slack msg that mentions Alice → 4
    // Stranger shares 0 documents with Alice.

    RETURN id(p_main)     AS p_main_id,
           id(p_alt)      AS p_alt_id,
           id(p_bob)      AS p_bob_id,
           id(p_alicia)   AS p_alicia_id,
           id(p_stranger) AS p_stranger_id
    """
    params = {
        "d_now": _iso(t_now),
        "d_y": _iso(t_yesterday),
        "d_2d": _iso(t_2d),
        "d_3d": _iso(t_3d),
        "d_4d": _iso(t_4d),
        "d_5d": _iso(t_5d),
        "d_2w": _iso(t_2w),
        "d_3mo": _iso(t_3mo),
        "sl_y": _slack_ts(t_yesterday),
        "sl_2d": _slack_ts(t_2d),
        "sl_3d": _slack_ts(t_3d),
    }
    with driver.session() as s:
        rec = s.run(cypher, **params).single()
        assert rec is not None
        return {k: int(v) for k, v in rec.data().items()}


# ---------------------------------------------------------------------------
# find_person
# ---------------------------------------------------------------------------


def test_find_person_by_email_canonicalizes_googlemail(
    driver: Driver, seeded: dict[str, int]
) -> None:
    # Seed an alternative googlemail.com row that should canonicalize to gmail.com
    with driver.session() as s:
        s.run("MERGE (p:Person {email: 'carol@gmail.com'}) SET p.name = 'Carol Gmail'")

    person = find_person("Carol@GoogleMail.com", driver=driver)
    assert isinstance(person, Person)
    assert person.email == "carol@gmail.com"


def test_find_person_by_slack_user_id(driver: Driver, seeded: dict[str, int]) -> None:
    person = find_person("slack-user:T-TEAM/U-ALICE", driver=driver)
    assert isinstance(person, Person)
    assert person.email == "alice@example.com"


def test_find_person_fuzzy_name_returns_list_on_ambiguity(
    driver: Driver, seeded: dict[str, int]
) -> None:
    result = find_person("Alic Example", driver=driver)
    # Both 'Alice Example' (canonicalized to one cluster) and 'Alicia Example'
    # match above the 90 cutoff.  After canonicalisation we expect at least
    # two distinct Persons → list.
    assert isinstance(result, list)
    assert len(result) >= 2
    emails = {p.email for p in result}
    assert "alice@example.com" in emails
    assert "alicia@example.com" in emails


def test_find_person_returns_none_on_miss(
    driver: Driver, seeded: dict[str, int]
) -> None:
    assert find_person("ghost@nowhere.invalid", driver=driver) is None
    assert find_person("Zzz Unmatchable Name", driver=driver) is None
    assert find_person("slack-user:T-TEAM/U-MISSING", driver=driver) is None


# ---------------------------------------------------------------------------
# recent_interactions
# ---------------------------------------------------------------------------


def test_recent_interactions_orders_desc_and_respects_limit(
    driver: Driver, seeded: dict[str, int]
) -> None:
    pid = seeded["p_main_id"]
    # Resolve via find_person so we exercise canonicalisation.
    canonical = find_person("alice@example.com", driver=driver)
    assert isinstance(canonical, Person)

    rows = recent_interactions(
        canonical.id, since=_NOW - timedelta(days=365), limit=3, driver=driver
    )
    assert len(rows) == 3
    # Strict descending order
    times = [r.timestamp for r in rows]
    assert times == sorted(times, reverse=True)
    # The newest is the email "Project kickoff" at d_now.
    assert "Project kickoff" in rows[0].title
    # Sanity: result still references the canonical Person (the limit
    # came from Python-side pagination, not from a missing alias edge).
    assert canonical.id == pid


def test_recent_interactions_filters_by_since(
    driver: Driver, seeded: dict[str, int]
) -> None:
    canonical = find_person("alice@example.com", driver=driver)
    assert isinstance(canonical, Person)

    # Window: last 36 hours — excludes 2-day-old, 3-day-old, 3-month-old rows.
    rows = recent_interactions(
        canonical.id,
        since=_NOW - timedelta(hours=36),
        limit=50,
        driver=driver,
    )
    titles = [r.title for r in rows]
    assert "Project kickoff" in titles  # d_now
    assert "Re: weekly sync" in titles  # yesterday
    assert "Travel plans" not in titles  # 2 days
    assert "Old thread" not in titles  # 3 months


# ---------------------------------------------------------------------------
# top_topics
# ---------------------------------------------------------------------------


def test_top_topics_counts_distinct_docs(
    driver: Driver, seeded: dict[str, int]
) -> None:
    canonical = find_person("alice@example.com", driver=driver)
    assert isinstance(canonical, Person)

    topics = top_topics(canonical.id, k=10, driver=driver)
    by_name = {t.topic_name: t.doc_count for t in topics}
    # Q2 planning: e1 (SENT), cal1 (ORGANIZED_BY), s1 (SENT_BY) → 3
    assert by_name.get("Q2 planning") == 3
    # Travel: e3 mentions p_alt (cluster) → 1
    assert by_name.get("Travel") == 1


# ---------------------------------------------------------------------------
# related_people
# ---------------------------------------------------------------------------


def test_related_people_excludes_same_as_cluster(
    driver: Driver, seeded: dict[str, int]
) -> None:
    canonical = find_person("alice@example.com", driver=driver)
    assert isinstance(canonical, Person)

    rows = related_people(canonical.id, k=10, driver=driver)
    emails = {r.email for r in rows}
    # p_alt is in Alice's SAME_AS cluster → must NOT appear.
    assert "alice.alt@example.com" not in emails
    # Bob shares multiple docs → must appear.
    assert "bob@example.com" in emails
    bob_row = next(r for r in rows if r.email == "bob@example.com")
    assert bob_row.shared_count >= 3


# ---------------------------------------------------------------------------
# open_tasks
# ---------------------------------------------------------------------------


def test_open_tasks_excludes_completed_and_dropped(
    driver: Driver, seeded: dict[str, int]
) -> None:
    canonical = find_person("alice@example.com", driver=driver)
    assert isinstance(canonical, Person)

    tasks = open_tasks(canonical.id, driver=driver)
    titles = [t.title for t in tasks]
    assert "Email Alice about Q2" in titles
    assert "Drop me" not in titles
    assert "Already shipped" not in titles

    only_open = next(t for t in tasks if t.title == "Email Alice about Q2")
    assert only_open.project == "Work"
    assert "urgent" in only_open.tags
    assert only_open.flagged is True
    assert only_open.due is not None


# ---------------------------------------------------------------------------
# files_mentioning
# ---------------------------------------------------------------------------


def test_files_mentioning_orders_by_mtime(
    driver: Driver, seeded: dict[str, int]
) -> None:
    canonical = find_person("alice@example.com", driver=driver)
    assert isinstance(canonical, Person)

    files = files_mentioning(canonical.id, k=10, driver=driver)
    paths = [f.path for f in files]
    # Three files mention some cluster member.
    assert paths == ["/notes/a.md", "/notes/c.md", "/notes/b.md"]


# ---------------------------------------------------------------------------
# identity_cluster
# ---------------------------------------------------------------------------


def test_identity_cluster_returns_match_type_and_confidence(
    driver: Driver, seeded: dict[str, int]
) -> None:
    canonical = find_person("alice@example.com", driver=driver)
    assert isinstance(canonical, Person)

    members = identity_cluster(canonical.id, driver=driver)
    assert len(members) == 1
    m = members[0]
    assert m.member == "alice.alt@example.com"
    assert m.match_type == "fuzzy_name"
    assert m.confidence is not None and m.confidence > 0.9
    assert m.cross_source is True


# ---------------------------------------------------------------------------
# get_profile
# ---------------------------------------------------------------------------


def test_get_profile_aggregates_all_sections(
    driver: Driver, seeded: dict[str, int]
) -> None:
    profile = get_profile(
        "alice@example.com",
        since=_NOW - timedelta(days=365),
        limit=20,
    )
    assert isinstance(profile, PersonProfile)
    assert profile.person.email == "alice@example.com"
    assert profile.recent_interactions  # populated
    assert profile.top_topics
    assert profile.related_people
    assert profile.open_tasks
    assert profile.files
    assert profile.identity_cluster
