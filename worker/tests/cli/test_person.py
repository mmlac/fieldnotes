"""Tests for the ``fieldnotes person`` CLI command.

Drives :mod:`worker.cli.person` end-to-end against a real Neo4j
instance, seeded per test from a single Cypher fixture.  Skipped
automatically when no Neo4j is reachable (mirrors
``tests/query/test_person.py``).
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from typing import Any, Generator
from unittest.mock import patch

import pytest
from neo4j import Driver, GraphDatabase

from worker.cli import _build_parser
from worker.cli.person import run_person
from worker.config import Config, MeConfig, Neo4jConfig

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


_NOW = datetime(2026, 4, 27, 12, 0, 0, tzinfo=timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _slack_ts(dt: datetime) -> str:
    return f"{dt.timestamp():.6f}"


def _make_config(*, with_me: bool = False) -> Config:
    cfg = Config(
        neo4j=Neo4jConfig(uri=_NEO4J_URI, user=_NEO4J_USER, password=_NEO4J_PASSWORD)
    )
    if with_me:
        cfg.me = MeConfig(emails=["alice@example.com"], name="Alice Example")
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
    """Mirrors ``tests/query/test_person.py``: seed Alice with multi-source edges."""
    t_now = _NOW
    t_y = t_now - timedelta(days=1)
    t_2d = t_now - timedelta(days=2)
    t_3d = t_now - timedelta(days=3)
    t_4d = t_now - timedelta(days=4)
    t_5d = t_now - timedelta(days=5)
    t_2w = t_now - timedelta(days=14)
    t_3mo = t_now - timedelta(days=90)

    cypher = """
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

    MERGE (p_bob:Person {email: 'bob@example.com'})
      SET p_bob.name = 'Bob Builder'
    MERGE (p_alicia:Person {email: 'alicia@example.com'})
      SET p_alicia.name = 'Alicia Example'

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

    MERGE (cal1:CalendarEvent {source_id: 'cal://1'})
      SET cal1.summary = 'Q2 planning',  cal1.start_time = $d_2d
    MERGE (cal2:CalendarEvent {source_id: 'cal://2'})
      SET cal2.summary = '1:1 with Bob', cal2.start_time = $d_4d
    MERGE (cal1)-[:ORGANIZED_BY]->(p_main)
    MERGE (cal2)-[:ATTENDED_BY]->(p_main)
    MERGE (cal2)-[:ORGANIZED_BY]->(p_bob)

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

    MERGE (t_open:Task {source_id: 'of://open'})
      SET t_open.name = 'Email Alice about Q2',
          t_open.status = 'active',
          t_open.flagged = true,
          t_open.due_date = $d_y,
          t_open.defer_date = $d_now
    MERGE (proj:Project {source_id: 'omnifocus-project:Work'})
      SET proj.name = 'Work', proj.source = 'omnifocus'
    MERGE (tag:Tag {source_id: 'omnifocus-tag:urgent'})
      SET tag.name = 'urgent', tag.source = 'omnifocus'
    MERGE (t_open)-[:IN_PROJECT]->(proj)
    MERGE (t_open)-[:TAGGED]->(tag)
    MERGE (t_open)-[:MENTIONS]->(p_main)

    MERGE (topic:Topic {name: 'Q2 planning'}) SET topic.source = 'user'
    MERGE (e1)-[:TAGGED]->(topic)
    MERGE (cal1)-[:TAGGED]->(topic)
    MERGE (s1)-[:TAGGED]->(topic)
    MERGE (topic_other:Topic {name: 'Travel'}) SET topic_other.source = 'user'
    MERGE (e3)-[:TAGGED]->(topic_other)

    MERGE (f1:File {source_id: '/notes/a.md'})
      SET f1.path = '/notes/a.md', f1.modified_at = $d_now,  f1.source = 'obsidian'
    MERGE (f2:File {source_id: '/notes/b.md'})
      SET f2.path = '/notes/b.md', f2.modified_at = $d_2w,  f2.source = 'obsidian'
    MERGE (f3:File {source_id: '/notes/c.md'})
      SET f3.path = '/notes/c.md', f3.modified_at = $d_5d,  f3.source = 'obsidian'
    MERGE (f1)-[:MENTIONS]->(p_main)
    MERGE (f2)-[:MENTIONS]->(p_main)
    MERGE (f3)-[:MENTIONS]->(p_alt)

    RETURN id(p_main) AS p_main_id
    """
    params = {
        "d_now": _iso(t_now),
        "d_y": _iso(t_y),
        "d_2d": _iso(t_2d),
        "d_3d": _iso(t_3d),
        "d_4d": _iso(t_4d),
        "d_5d": _iso(t_5d),
        "d_2w": _iso(t_2w),
        "d_3mo": _iso(t_3mo),
        "sl_y": _slack_ts(t_y),
        "sl_2d": _slack_ts(t_2d),
        "sl_3d": _slack_ts(t_3d),
    }
    with driver.session() as s:
        rec = s.run(cypher, **params).single()
        assert rec is not None
        return {k: int(v) for k, v in rec.data().items()}


def _run(
    *,
    with_me: bool = False,
    **kwargs: Any,
) -> tuple[int, str, str]:
    """Invoke ``run_person`` with a patched ``load_config`` and capture I/O."""
    cfg = _make_config(with_me=with_me)
    out = io.StringIO()
    err = io.StringIO()
    with patch("worker.cli.person.load_config", return_value=cfg):
        with redirect_stdout(out), redirect_stderr(err):
            code = run_person(**kwargs)
    return code, out.getvalue(), err.getvalue()


# ---------------------------------------------------------------------------
# Parser argument plumbing
# ---------------------------------------------------------------------------


class TestPersonParser:
    def test_person_basic(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["person", "alice@example.com"])
        assert args.command == "person"
        assert args.identifier == "alice@example.com"
        assert args.use_self is False
        assert args.search is None
        assert args.json_output is False

    def test_person_since_default_30d(self) -> None:
        """Default --since must be exactly '30d' (acceptance criterion)."""
        parser = _build_parser()
        args = parser.parse_args(["person", "x@y"])
        assert args.since == "30d"
        assert args.limit == 10

    def test_person_self_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["person", "--self"])
        assert args.use_self is True
        assert args.identifier is None

    def test_person_search_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["person", "--search", "Alice"])
        assert args.search == "Alice"

    def test_person_json_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["person", "x@y", "--json"])
        assert args.json_output is True


# ---------------------------------------------------------------------------
# Rich rendering: full sections from a populated graph
# ---------------------------------------------------------------------------


@_NEEDS_NEO4J
def test_person_email_renders_all_sections(seeded: dict[str, int]) -> None:
    code, out, _err = _run(identifier="alice@example.com", since="365d")
    assert code == 0
    # Header
    assert "Alice Example" in out
    assert "alice@example.com" in out
    assert "alice.alt@example.com" in out  # SAME_AS alias
    # Section titles (Rich tables render the title on its own line)
    assert "Recent interactions" in out
    assert "Top topics" in out
    assert "Q2 planning" in out
    assert "Related people" in out
    assert "bob@example.com" in out
    assert "Open tasks" in out
    assert "Email Alice about Q2" in out
    assert "Files mentioning" in out
    assert "/notes/a.md" in out
    assert "Identity cluster" in out


# ---------------------------------------------------------------------------
# Unknown identifier → exit non-zero
# ---------------------------------------------------------------------------


@_NEEDS_NEO4J
def test_person_unknown_exits_nonzero(seeded: dict[str, int]) -> None:
    code, _out, err = _run(identifier="ghost@nowhere.invalid")
    assert code == 1
    assert "No Person found" in err
    assert "ghost@nowhere.invalid" in err


# ---------------------------------------------------------------------------
# Ambiguous fuzzy name → disambiguation table on stderr, non-zero exit
# ---------------------------------------------------------------------------


@_NEEDS_NEO4J
def test_person_ambiguous_name_shows_disambiguation(seeded: dict[str, int]) -> None:
    code, _out, err = _run(identifier="Alic Example")
    assert code == 1
    assert "Ambiguous" in err
    # Both candidate emails should be listed
    assert "alice@example.com" in err
    assert "alicia@example.com" in err


# ---------------------------------------------------------------------------
# --self flag
# ---------------------------------------------------------------------------


def test_person_self_without_me_block_errors() -> None:  # No Neo4j needed
    code, _out, err = _run(identifier=None, use_self=True, with_me=False)
    assert code == 2
    assert "[me]" in err
    assert "config" in err


@_NEEDS_NEO4J
def test_person_self_with_me_block_resolves(
    driver: Driver, seeded: dict[str, int]
) -> None:
    # Flag Alice as self in the graph
    with driver.session() as s:
        s.run("MATCH (p:Person {email: 'alice@example.com'}) SET p.is_self = true")
    code, out, _err = _run(identifier=None, use_self=True, with_me=True, since="365d")
    assert code == 0
    assert "Alice Example" in out
    assert "(self)" in out


# ---------------------------------------------------------------------------
# --json schema
# ---------------------------------------------------------------------------


_REQUIRED_JSON_KEYS = {
    "identifier",
    "resolved",
    "sources_present",
    "last_seen",
    "total_interactions",
    "recent_interactions",
    "top_topics",
    "related_people",
    "open_tasks",
    "files_mentioning",
    "identity_cluster",
}


@_NEEDS_NEO4J
def test_person_json_schema_stable(seeded: dict[str, int]) -> None:
    code, out, _err = _run(
        identifier="alice@example.com",
        since="365d",
        json_output=True,
    )
    assert code == 0
    payload = json.loads(out)  # roundtrips
    assert _REQUIRED_JSON_KEYS.issubset(payload.keys())
    assert payload["identifier"] == "alice@example.com"
    assert payload["resolved"]["email"] == "alice@example.com"
    assert payload["resolved"]["is_self"] is False
    assert isinstance(payload["sources_present"], list)
    assert isinstance(payload["total_interactions"], int)
    assert payload["total_interactions"] > 0
    # Recent interaction row schema
    assert payload["recent_interactions"]
    sample = payload["recent_interactions"][0]
    for field_name in ("timestamp", "source_type", "title", "snippet", "edge_kind"):
        assert field_name in sample
    # Top topics row schema
    assert payload["top_topics"]
    assert {"topic_name", "doc_count"} <= set(payload["top_topics"][0].keys())
    # Related people row schema
    assert payload["related_people"]
    assert {"name", "email", "shared_count"} <= set(payload["related_people"][0].keys())


@_NEEDS_NEO4J
def test_person_json_empty_sections_are_arrays(driver: Driver) -> None:
    """A Person with no edges must produce empty arrays, not crash."""
    with driver.session() as s:
        s.run("MERGE (p:Person {email: 'lonely@example.com'}) SET p.name = 'Lonely'")
    code, out, _err = _run(identifier="lonely@example.com", json_output=True)
    assert code == 0
    payload = json.loads(out)
    for arr_key in (
        "recent_interactions",
        "top_topics",
        "related_people",
        "open_tasks",
        "files_mentioning",
        "identity_cluster",
        "sources_present",
    ):
        assert payload[arr_key] == [], f"{arr_key} should be empty list"
    assert payload["last_seen"] is None
    assert payload["total_interactions"] == 0


# ---------------------------------------------------------------------------
# --limit caps row counts
# ---------------------------------------------------------------------------


@_NEEDS_NEO4J
def test_person_limit_caps_rows(seeded: dict[str, int]) -> None:
    code, out, _err = _run(
        identifier="alice@example.com",
        since="365d",
        limit=2,
        json_output=True,
    )
    assert code == 0
    payload = json.loads(out)
    assert len(payload["recent_interactions"]) <= 2
    assert len(payload["top_topics"]) <= 2
    assert len(payload["related_people"]) <= 2
    assert len(payload["files_mentioning"]) <= 2
