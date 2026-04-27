"""Tests for ``fieldnotes migrate gmail-multiaccount`` (fn-3t4)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from worker.cli import _build_parser
from worker.cli.migrate import (
    MigrateLockHeld,
    acquire_migrate_lock,
    detect_interleaved,
    detect_neo4j_counts,
    detect_qdrant_count,
    detect_queue_counts,
    migrate_config,
    migrate_files,
    migrate_neo4j,
    migrate_qdrant,
    migrate_queue,
    release_migrate_lock,
    resolve_account,
    rewrite_source_id,
    run_migrate_gmail_multiaccount,
    validate_post_pass,
)


# ─── Helpers ──────────────────────────────────────────────────────────


def _seed_queue_db(db_path: Path) -> None:
    """Create a queue.db with mixed old- and new-shape rows."""
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE queue (
            id TEXT PRIMARY KEY,
            source_type TEXT NOT NULL,
            source_id TEXT NOT NULL,
            operation TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            payload TEXT NOT NULL,
            blob_path TEXT,
            enqueued_at TEXT NOT NULL,
            started_at TEXT,
            attempts INTEGER NOT NULL DEFAULT 0,
            error TEXT
        );
        CREATE TABLE cursors (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )
    rows = [
        # Old-shape Gmail thread + message (pending + processing).
        ("g1", "gmail", "gmail://thread/T1", "created", "pending"),
        ("g2", "gmail", "gmail://message/M1", "created", "processing"),
        ("g3", "gmail", "gmail://thread/T2", "created", "failed"),
        # Old-shape Calendar event + series.
        ("c1", "calendar", "google-calendar://event/E1", "created", "pending"),
        ("c2", "calendar", "google-calendar://series/S1", "created", "pending"),
        # Cross-source: Slack and Obsidian must be untouched.
        ("s1", "slack", "slack://team/T/channel/C/ts/123.456", "created", "pending"),
        ("o1", "obsidian", "obsidian:///notes/foo.md", "modified", "pending"),
        # File source untouched.
        ("f1", "files", "file:///home/me/doc.txt", "created", "pending"),
    ]
    for row_id, source_type, source_id, operation, status in rows:
        payload = json.dumps(
            {
                "id": row_id,
                "source_type": source_type,
                "source_id": source_id,
                "operation": operation,
            }
        )
        conn.execute(
            "INSERT INTO queue "
            "(id, source_type, source_id, operation, status, payload, "
            " enqueued_at) "
            "VALUES (?, ?, ?, ?, ?, ?, '2026-04-26T00:00:00Z')",
            (row_id, source_type, source_id, operation, status, payload),
        )
    conn.execute(
        "INSERT INTO cursors VALUES ('gmail', '{\"x\":1}', '2026-04-26T00:00:00Z')"
    )
    conn.execute(
        "INSERT INTO cursors VALUES "
        "('calendar', '{\"y\":2}', '2026-04-26T00:00:00Z')"
    )
    conn.execute(
        "INSERT INTO cursors VALUES "
        "('files', '{\"z\":3}', '2026-04-26T00:00:00Z')"
    )
    conn.commit()
    conn.close()


class _FakeNeo4jSession:
    """Tiny in-memory Cypher executor sufficient for the migrate path.

    Stores ``Document``-style nodes with a ``source_id`` plus
    ``Chunk`` nodes with an ``id`` and ``Person`` nodes with a
    ``source_id``.  Supports the small set of MATCH/SET/RETURN queries
    issued by the migrate code; everything else is a no-op.
    """

    def __init__(self) -> None:
        self.docs: list[dict[str, Any]] = []
        self.chunks: list[dict[str, Any]] = []
        self.persons: list[dict[str, Any]] = []
        self.entered = 0

    def __enter__(self) -> "_FakeNeo4jSession":
        self.entered += 1
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    # ---- API used by the migrate code ----

    def run(self, query: str, **params: Any) -> "_FakeNeo4jResult":
        q = " ".join(query.split())  # collapse whitespace

        # Detection: count of source_id-bearing nodes for old-shape prefixes.
        if "MATCH (n) WHERE n.source_id IS NOT NULL" in q and "RETURN count(n)" in q and "STARTS WITH" in q and "SET" not in q:
            count = sum(
                1 for d in self.docs
                if any(d["source_id"].startswith(p) for p in self._old_or_new_prefixes(q))
            )
            return _FakeNeo4jResult([{"_scalar": count}])

        if "MATCH (c:Chunk)" in q and "RETURN count(c)" in q and "SET" not in q:
            count = sum(
                1 for c in self.chunks
                if any(c["id"].startswith(p) for p in self._old_or_new_prefixes(q))
            )
            return _FakeNeo4jResult([{"_scalar": count}])

        if "MATCH (p:Person)" in q and "RETURN count(p)" in q and "SET" not in q:
            count = sum(
                1 for p in self.persons
                if any(p["source_id"].startswith(pref) for pref in self._old_or_new_prefixes(q))
            )
            return _FakeNeo4jResult([{"_scalar": count}])

        # Mutation: SET source_id on source nodes.
        if "MATCH (n) WHERE n.source_id" in q and "SET n.source_id" in q:
            old_prefix = params["old_prefix"]
            new_prefix = params["new_prefix"]
            cut = params["cut"]
            n = 0
            for d in self.docs:
                if d["source_id"].startswith(old_prefix):
                    d["source_id"] = new_prefix + d["source_id"][cut:]
                    n += 1
            return _FakeNeo4jResult([{"c": n}])

        if "MATCH (c:Chunk)" in q and "SET c.id" in q:
            old_prefix = params["old_prefix"]
            new_prefix = params["new_prefix"]
            cut = params["cut"]
            n = 0
            for c in self.chunks:
                if c["id"].startswith(old_prefix):
                    c["id"] = new_prefix + c["id"][cut:]
                    n += 1
            return _FakeNeo4jResult([{"c": n}])

        if "MATCH (p:Person)" in q and "SET p.source_id" in q:
            old_prefix = params["old_prefix"]
            new_prefix = params["new_prefix"]
            cut = params["cut"]
            n = 0
            for p in self.persons:
                if p["source_id"].startswith(old_prefix):
                    p["source_id"] = new_prefix + p["source_id"][cut:]
                    n += 1
            return _FakeNeo4jResult([{"c": n}])

        return _FakeNeo4jResult([])

    @staticmethod
    def _old_or_new_prefixes(query: str) -> list[str]:
        """Pull literal prefix strings out of the query (for count probes)."""
        prefixes: list[str] = []
        for needle in (
            "gmail://",
            "google-calendar://",
        ):
            idx = 0
            while True:
                start = query.find(f"'{needle}", idx)
                if start == -1:
                    break
                end = query.find("'", start + 1)
                if end == -1:
                    break
                prefixes.append(query[start + 1: end])
                idx = end + 1
        return prefixes


class _FakeNeo4jResult:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def single(self) -> Any:
        return self._rows[0] if self._rows else None

    def __iter__(self):
        return iter(self._rows)


class _FakeQdrantPoint:
    def __init__(self, point_id: int, payload: dict[str, Any], vector: list[float]):
        self.id = point_id
        self.payload = payload
        self.vector = vector


class _FakeQdrantClient:
    """In-memory Qdrant stand-in supporting scroll + set_payload."""

    def __init__(self) -> None:
        self.points: dict[int, _FakeQdrantPoint] = {}

    def add(self, point_id: int, source_id: str, *, source_type: str = "gmail") -> None:
        self.points[point_id] = _FakeQdrantPoint(
            point_id,
            {"source_type": source_type, "source_id": source_id, "chunk_index": 0,
             "text": f"chunk-{point_id}"},
            [0.1, 0.2, 0.3, point_id * 1.0],
        )

    def scroll(self, *, collection_name: str, limit: int, offset: Any,
               with_payload: bool, with_vectors: bool) -> tuple[list, Any]:
        all_points = list(self.points.values())
        start = 0
        if offset is not None:
            for i, p in enumerate(all_points):
                if p.id == offset:
                    start = i
                    break
        page = all_points[start: start + limit]
        next_offset = (
            all_points[start + limit].id if start + limit < len(all_points) else None
        )
        return page, next_offset

    def set_payload(self, *, collection_name: str, payload: dict[str, Any], points: list[int]):
        for pid in points:
            point = self.points[pid]
            new_payload = dict(point.payload or {})
            new_payload.update(payload)
            point.payload = new_payload
            # NB: vector left untouched — the test asserts this.


# ─── Parser tests ─────────────────────────────────────────────────────


class TestParser:
    def test_subcommand_present(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["migrate", "gmail-multiaccount"])
        assert args.command == "migrate"
        assert args.migrate_command == "gmail-multiaccount"
        assert args.account is None
        assert not args.yes
        assert not args.dry_run
        assert not args.force_running

    def test_all_flags(self) -> None:
        parser = _build_parser()
        args = parser.parse_args([
            "migrate", "gmail-multiaccount",
            "--account", "personal",
            "--yes",
            "--dry-run",
            "--force-running",
        ])
        assert args.account == "personal"
        assert args.yes is True
        assert args.dry_run is True
        assert args.force_running is True


# ─── Pure-function tests ──────────────────────────────────────────────


class TestRewriteSourceId:
    def test_thread(self) -> None:
        assert rewrite_source_id("gmail://thread/abc", "personal") == \
            "gmail://personal/thread/abc"

    def test_message(self) -> None:
        assert rewrite_source_id("gmail://message/m1", "work") == \
            "gmail://work/message/m1"

    def test_calendar_event(self) -> None:
        assert rewrite_source_id("google-calendar://event/E1", "default") == \
            "google-calendar://default/event/E1"

    def test_calendar_series(self) -> None:
        assert rewrite_source_id("google-calendar://series/S1", "default") == \
            "google-calendar://default/series/S1"

    def test_attendee_fallback(self) -> None:
        # Attendee fallback Person source_ids hit the same rewrite path.
        assert rewrite_source_id(
            "google-calendar://event/E1/attendee/3", "personal"
        ) == "google-calendar://personal/event/E1/attendee/3"

    def test_already_namespaced_returns_none(self) -> None:
        assert rewrite_source_id("gmail://personal/thread/abc", "default") is None

    def test_unrelated_scheme_returns_none(self) -> None:
        assert rewrite_source_id("slack://team/T/channel/C", "default") is None


class TestResolveAccount:
    def test_explicit_valid(self) -> None:
        assert resolve_account("personal", yes=False) == "personal"

    def test_explicit_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            resolve_account("Invalid", yes=False)

    def test_explicit_too_long_raises(self) -> None:
        with pytest.raises(ValueError):
            resolve_account("a-very-long-name-that-exceeds-thirty-one-chars", yes=False)

    def test_yes_defaults_to_default(self) -> None:
        assert resolve_account(None, yes=True) == "default"

    def test_prompt_empty_returns_default(self) -> None:
        assert resolve_account(None, yes=False, input_fn=lambda _: "") == "default"

    def test_prompt_returns_user_value(self) -> None:
        assert resolve_account(None, yes=False, input_fn=lambda _: "work") == "work"


# ─── Queue migration ─────────────────────────────────────────────────


class TestMigrateQueue:
    def test_retags_old_shape_rows(self, tmp_path: Path) -> None:
        db = tmp_path / "queue.db"
        _seed_queue_db(db)

        result = migrate_queue(db, "personal")
        assert result["rows"] == 5
        assert result["cursors"] == 2

        conn = sqlite3.connect(str(db))
        try:
            sids = {r[0] for r in conn.execute("SELECT source_id FROM queue")}
            assert "gmail://personal/thread/T1" in sids
            assert "gmail://personal/thread/T2" in sids
            assert "gmail://personal/message/M1" in sids
            assert "google-calendar://personal/event/E1" in sids
            assert "google-calendar://personal/series/S1" in sids
            # Slack/Obsidian/Files unaffected.
            assert "slack://team/T/channel/C/ts/123.456" in sids
            assert "obsidian:///notes/foo.md" in sids
            assert "file:///home/me/doc.txt" in sids
        finally:
            conn.close()

    def test_payload_json_updated(self, tmp_path: Path) -> None:
        db = tmp_path / "queue.db"
        _seed_queue_db(db)
        migrate_queue(db, "personal")

        conn = sqlite3.connect(str(db))
        try:
            for row_id, expected_sid in [
                ("g1", "gmail://personal/thread/T1"),
                ("c1", "google-calendar://personal/event/E1"),
            ]:
                row = conn.execute(
                    "SELECT payload FROM queue WHERE id = ?", (row_id,)
                ).fetchone()
                payload = json.loads(row[0])
                assert payload["source_id"] == expected_sid
        finally:
            conn.close()

    def test_cursors_renamed(self, tmp_path: Path) -> None:
        db = tmp_path / "queue.db"
        _seed_queue_db(db)
        migrate_queue(db, "personal")

        conn = sqlite3.connect(str(db))
        try:
            keys = {r[0] for r in conn.execute("SELECT key FROM cursors")}
            assert "gmail:personal" in keys
            assert "calendar:personal" in keys
            assert "files" in keys  # untouched
            assert "gmail" not in keys
            assert "calendar" not in keys
        finally:
            conn.close()

    def test_idempotent_second_run_is_noop(self, tmp_path: Path) -> None:
        db = tmp_path / "queue.db"
        _seed_queue_db(db)
        migrate_queue(db, "personal")
        result = migrate_queue(db, "personal")
        assert result == {"rows": 0, "cursors": 0}

    def test_missing_db_is_noop(self, tmp_path: Path) -> None:
        result = migrate_queue(tmp_path / "missing.db", "personal")
        assert result == {"rows": 0, "cursors": 0}


# ─── Neo4j migration ────────────────────────────────────────────────


class TestMigrateNeo4j:
    def test_rewrites_documents_chunks_persons(self) -> None:
        sess = _FakeNeo4jSession()
        sess.docs.append({"source_id": "gmail://thread/T1"})
        sess.docs.append({"source_id": "gmail://message/M1"})
        sess.docs.append({"source_id": "google-calendar://event/E1"})
        sess.docs.append({"source_id": "obsidian:///notes/x.md"})  # untouched
        sess.chunks.append({"id": "gmail://thread/T1:chunk:0"})
        sess.chunks.append({"id": "gmail://thread/T1:chunk:1"})
        sess.persons.append({"source_id": "google-calendar://event/E1/attendee/2"})

        out = migrate_neo4j(sess, "personal")

        assert out == {"documents": 3, "chunks": 2, "persons": 1}
        assert {d["source_id"] for d in sess.docs} == {
            "gmail://personal/thread/T1",
            "gmail://personal/message/M1",
            "google-calendar://personal/event/E1",
            "obsidian:///notes/x.md",
        }
        assert {c["id"] for c in sess.chunks} == {
            "gmail://personal/thread/T1:chunk:0",
            "gmail://personal/thread/T1:chunk:1",
        }
        assert sess.persons[0]["source_id"] == \
            "google-calendar://personal/event/E1/attendee/2"


# ─── Qdrant migration ───────────────────────────────────────────────


class TestMigrateQdrant:
    def test_rewrites_payload_preserves_vectors(self) -> None:
        client = _FakeQdrantClient()
        client.add(1, "gmail://thread/T1")
        client.add(2, "gmail://message/M1")
        client.add(3, "slack://team/T/channel/C/ts/1.1", source_type="slack")  # untouched
        client.add(4, "google-calendar://event/E1")

        before = {pid: list(p.vector) for pid, p in client.points.items()}

        out = migrate_qdrant(client, "fieldnotes", "personal", batch=2)

        assert out == {"points": 3}
        assert client.points[1].payload["source_id"] == \
            "gmail://personal/thread/T1"
        assert client.points[2].payload["source_id"] == \
            "gmail://personal/message/M1"
        assert client.points[3].payload["source_id"] == \
            "slack://team/T/channel/C/ts/1.1"
        assert client.points[4].payload["source_id"] == \
            "google-calendar://personal/event/E1"
        # Vector bytes unchanged.
        for pid, vec in before.items():
            assert client.points[pid].vector == vec


# ─── File renames + cursor cleanup ───────────────────────────────────


class TestMigrateFiles:
    def test_renames_token_files(self, tmp_path: Path) -> None:
        (tmp_path / "gmail_token.json").write_text('{"refresh":"x"}')
        (tmp_path / "calendar_token.json").write_text('{"refresh":"y"}')

        result = migrate_files(tmp_path, "personal")

        assert (tmp_path / "gmail_token-personal.json").exists()
        assert (tmp_path / "calendar_token-personal.json").exists()
        assert not (tmp_path / "gmail_token.json").exists()
        assert not (tmp_path / "calendar_token.json").exists()
        assert "gmail_token.json → gmail_token-personal.json" in result["renamed"]
        assert "calendar_token.json → calendar_token-personal.json" in result["renamed"]

    def test_missing_files_logged_not_failed(self, tmp_path: Path) -> None:
        result = migrate_files(tmp_path, "personal")
        assert result["renamed"] == []
        assert result["deleted"] == []

    def test_deletes_legacy_cursor_files(self, tmp_path: Path) -> None:
        (tmp_path / "gmail_cursor.json").write_text("{}")
        (tmp_path / "calendar_cursor.json").write_text("{}")

        result = migrate_files(tmp_path, "personal")

        assert not (tmp_path / "gmail_cursor.json").exists()
        assert not (tmp_path / "calendar_cursor.json").exists()
        assert "gmail_cursor.json" in result["deleted"]
        assert "calendar_cursor.json" in result["deleted"]

    def test_deletes_per_account_cursor_variants(self, tmp_path: Path) -> None:
        (tmp_path / "gmail_cursor-personal.json").write_text("{}")
        (tmp_path / "calendar_cursor-work.json").write_text("{}")

        result = migrate_files(tmp_path, "personal")

        assert not (tmp_path / "gmail_cursor-personal.json").exists()
        assert not (tmp_path / "calendar_cursor-work.json").exists()
        assert "gmail_cursor-personal.json" in result["deleted"]
        assert "calendar_cursor-work.json" in result["deleted"]

    def test_unrelated_files_untouched(self, tmp_path: Path) -> None:
        (tmp_path / "foo.json").write_text("{}")
        (tmp_path / "gmail_token-default.json").write_text("{}")  # already migrated

        result = migrate_files(tmp_path, "personal")

        assert (tmp_path / "foo.json").exists()
        assert (tmp_path / "gmail_token-default.json").exists()
        assert "foo.json" not in result["deleted"]
        assert "gmail_token-default.json" not in result["deleted"]


# ─── Config rewrite ──────────────────────────────────────────────────


class TestMigrateConfig:
    def test_wraps_flat_sections(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "config.toml"
        cfg_path.write_text(
            """
[core]
data_dir = "~/.fieldnotes/data"

[sources.gmail]
client_secrets_path = "~/.fieldnotes/gmail_credentials.json"
poll_interval_seconds = 300
max_initial_threads = 500

[sources.google_calendar]
client_secrets_path = "~/.fieldnotes/gmail_credentials.json"
calendar_ids = ["primary"]
""".lstrip()
        )

        rewrote = migrate_config(cfg_path, "personal")
        assert rewrote is True

        # Backup created.
        backups = list(tmp_path.glob("config.toml.backup-*"))
        assert len(backups) == 1
        # Round-trip via tomllib (read-only) verifies the new shape.
        import tomllib
        new = tomllib.loads(cfg_path.read_text())
        assert "personal" in new["sources"]["gmail"]
        assert (
            new["sources"]["gmail"]["personal"]["client_secrets_path"]
            == "~/.fieldnotes/gmail_credentials.json"
        )
        assert new["sources"]["gmail"]["personal"]["poll_interval_seconds"] == 300
        assert "personal" in new["sources"]["google_calendar"]
        assert (
            new["sources"]["google_calendar"]["personal"]["calendar_ids"]
            == ["primary"]
        )

    def test_already_multi_account_is_noop(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "config.toml"
        cfg_path.write_text(
            """
[sources.gmail.personal]
client_secrets_path = "~/.fieldnotes/gmail_credentials.json"
""".lstrip()
        )
        assert migrate_config(cfg_path, "personal") is False
        assert not list(tmp_path.glob("config.toml.backup-*"))

    def test_missing_config_is_noop(self, tmp_path: Path) -> None:
        assert migrate_config(tmp_path / "nope.toml", "personal") is False


# ─── Detection counts ────────────────────────────────────────────────


class TestDetection:
    def test_queue_counts(self, tmp_path: Path) -> None:
        db = tmp_path / "queue.db"
        _seed_queue_db(db)
        assert detect_queue_counts(db) == (5, 2)

    def test_queue_counts_missing_db(self, tmp_path: Path) -> None:
        assert detect_queue_counts(tmp_path / "missing.db") == (0, 0)

    def test_neo4j_counts(self) -> None:
        sess = _FakeNeo4jSession()
        sess.docs.append({"source_id": "gmail://thread/T1"})
        sess.docs.append({"source_id": "obsidian:///x.md"})
        sess.chunks.append({"id": "gmail://thread/T1:chunk:0"})
        sess.persons.append({"source_id": "google-calendar://event/E1/attendee/0"})
        assert detect_neo4j_counts(sess) == (1, 1, 1)

    def test_qdrant_counts(self) -> None:
        client = _FakeQdrantClient()
        client.add(1, "gmail://thread/T1")
        client.add(2, "slack://team/T/channel/C/ts/1.0", source_type="slack")
        client.add(3, "google-calendar://event/E1")
        assert detect_qdrant_count(client, "fieldnotes") == 2


# ─── Interleaved-migration refusal ───────────────────────────────────


class TestInterleavedDetection:
    def test_detects_new_shape_documents(self) -> None:
        sess = _FakeNeo4jSession()
        sess.docs.append({"source_id": "gmail://personal/thread/X"})
        msg = detect_interleaved(sess, "personal")
        assert msg is not None
        assert "personal" in msg

    def test_detects_promoted_cursor_key(self, tmp_path: Path) -> None:
        sess = _FakeNeo4jSession()
        db = tmp_path / "queue.db"
        _seed_queue_db(db)
        # Promote a cursor key as if a daemon already wrote it.
        conn = sqlite3.connect(str(db))
        conn.execute(
            "INSERT INTO cursors VALUES "
            "('gmail:personal', '{\"x\":1}', '2026-04-26T00:00:00Z')"
        )
        conn.commit()
        conn.close()
        msg = detect_interleaved(sess, "personal", db_path=db)
        assert msg is not None
        assert "gmail:personal" in msg

    def test_clean_state_returns_none(self, tmp_path: Path) -> None:
        sess = _FakeNeo4jSession()
        db = tmp_path / "queue.db"
        _seed_queue_db(db)
        assert detect_interleaved(sess, "personal", db_path=db) is None


# ─── End-to-end orchestrator ─────────────────────────────────────────


@pytest.fixture
def env(tmp_path: Path):
    """Build a fully-populated migration environment in tmp_path."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    db = data_dir / "queue.db"
    _seed_queue_db(db)

    # Token + cursor files.
    (data_dir / "gmail_token.json").write_text("{}")
    (data_dir / "calendar_token.json").write_text("{}")
    (data_dir / "gmail_cursor.json").write_text("{}")
    (data_dir / "calendar_cursor.json").write_text("{}")

    # Neo4j fake.
    sess = _FakeNeo4jSession()
    sess.docs.append({"source_id": "gmail://thread/T1"})
    sess.docs.append({"source_id": "gmail://message/M1"})
    sess.docs.append({"source_id": "google-calendar://event/E1"})
    sess.docs.append({"source_id": "obsidian:///x.md"})  # untouched
    sess.chunks.append({"id": "gmail://thread/T1:chunk:0"})
    sess.persons.append({"source_id": "google-calendar://event/E1/attendee/0"})

    # Qdrant fake.
    qdrant = _FakeQdrantClient()
    qdrant.add(1, "gmail://thread/T1")
    qdrant.add(2, "google-calendar://event/E1")
    qdrant.add(3, "slack://team/T/channel/C/ts/1.0", source_type="slack")  # untouched

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[core]
data_dir = "{data_dir}"

[neo4j]
password = "x"

[sources.gmail]
client_secrets_path = "~/.fieldnotes/gmail_credentials.json"

[sources.google_calendar]
client_secrets_path = "~/.fieldnotes/gmail_credentials.json"
calendar_ids = ["primary"]

[me]
emails = ["me@example.com"]
""".lstrip().replace("{data_dir}", str(data_dir))
    )

    return {
        "data_dir": data_dir,
        "db": db,
        "sess": sess,
        "qdrant": qdrant,
        "config_path": config_path,
    }


class TestRunMigrate:
    def test_refuses_when_daemon_running(self, env, capsys) -> None:
        rc = run_migrate_gmail_multiaccount(
            account="personal",
            yes=True,
            data_dir=env["data_dir"],
            daemon_detector=lambda: (True, "fake"),
            neo4j_session_factory=lambda: env["sess"],
            qdrant_factory=lambda: env["qdrant"],
        )
        assert rc == 2
        err = capsys.readouterr().err
        assert "daemon is running" in err
        # No mutations.
        sids = sqlite3.connect(str(env["db"])).execute(
            "SELECT source_id FROM queue WHERE id = 'g1'"
        ).fetchone()
        assert sids[0] == "gmail://thread/T1"

    def test_force_running_proceeds_with_warning(self, env, caplog) -> None:
        with caplog.at_level("WARNING"):
            rc = run_migrate_gmail_multiaccount(
                account="personal",
                yes=True,
                force_running=True,
                data_dir=env["data_dir"],
                daemon_detector=lambda: (True, "fake"),
                neo4j_session_factory=lambda: env["sess"],
                qdrant_factory=lambda: env["qdrant"],
            )
        assert rc == 0
        assert any("DAEMON STILL RUNNING" in rec.message for rec in caplog.records)

    def test_dry_run_makes_no_mutations(self, env, capsys) -> None:
        rc = run_migrate_gmail_multiaccount(
            account="personal",
            yes=True,
            dry_run=True,
            data_dir=env["data_dir"],
            daemon_detector=lambda: (False, None),
            neo4j_session_factory=lambda: env["sess"],
            qdrant_factory=lambda: env["qdrant"],
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "Migration target" in out
        assert "Dry run" in out
        # Queue + Neo4j + Qdrant + files unchanged.
        assert env["sess"].docs[0]["source_id"] == "gmail://thread/T1"
        assert env["qdrant"].points[1].payload["source_id"] == "gmail://thread/T1"
        assert (env["data_dir"] / "gmail_token.json").exists()
        assert (env["data_dir"] / "gmail_cursor.json").exists()
        sid = sqlite3.connect(str(env["db"])).execute(
            "SELECT source_id FROM queue WHERE id = 'g1'"
        ).fetchone()
        assert sid[0] == "gmail://thread/T1"

    def test_full_happy_path(self, env, capsys) -> None:
        rc = run_migrate_gmail_multiaccount(
            account="personal",
            yes=True,
            data_dir=env["data_dir"],
            daemon_detector=lambda: (False, None),
            neo4j_session_factory=lambda: env["sess"],
            qdrant_factory=lambda: env["qdrant"],
            config_path=env["config_path"],
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "Retagged" in out

        # Queue retagged.
        sids = {r[0] for r in sqlite3.connect(str(env["db"])).execute(
            "SELECT source_id FROM queue"
        )}
        assert "gmail://personal/thread/T1" in sids
        assert "google-calendar://personal/event/E1" in sids
        assert "slack://team/T/channel/C/ts/123.456" in sids  # untouched

        # Neo4j retagged.
        doc_ids = {d["source_id"] for d in env["sess"].docs}
        assert "gmail://personal/thread/T1" in doc_ids
        assert "obsidian:///x.md" in doc_ids

        # Qdrant retagged + vectors preserved.
        assert env["qdrant"].points[1].payload["source_id"] == \
            "gmail://personal/thread/T1"
        assert env["qdrant"].points[3].payload["source_id"] == \
            "slack://team/T/channel/C/ts/1.0"
        assert env["qdrant"].points[1].vector == [0.1, 0.2, 0.3, 1.0]

        # Files renamed + deleted.
        assert (env["data_dir"] / "gmail_token-personal.json").exists()
        assert (env["data_dir"] / "calendar_token-personal.json").exists()
        assert not (env["data_dir"] / "gmail_cursor.json").exists()
        assert not (env["data_dir"] / "calendar_cursor.json").exists()

        # Config rewritten.
        import tomllib
        new_cfg = tomllib.loads(env["config_path"].read_text())
        assert "personal" in new_cfg["sources"]["gmail"]
        assert "personal" in new_cfg["sources"]["google_calendar"]

    def test_idempotent_second_run(self, env) -> None:
        # First run.
        run_migrate_gmail_multiaccount(
            account="personal",
            yes=True,
            data_dir=env["data_dir"],
            daemon_detector=lambda: (False, None),
            neo4j_session_factory=lambda: env["sess"],
            qdrant_factory=lambda: env["qdrant"],
            config_path=env["config_path"],
        )
        # Second run is a refusal — interleaved-migration check fires
        # because account=personal Documents are now present.
        rc = run_migrate_gmail_multiaccount(
            account="personal",
            yes=True,
            data_dir=env["data_dir"],
            daemon_detector=lambda: (False, None),
            neo4j_session_factory=lambda: env["sess"],
            qdrant_factory=lambda: env["qdrant"],
            config_path=env["config_path"],
        )
        assert rc == 3

    def test_refuses_interleaved_state(self, env, capsys) -> None:
        # Pre-existing new-shape doc.
        env["sess"].docs.append({"source_id": "gmail://personal/thread/PREEXIST"})
        rc = run_migrate_gmail_multiaccount(
            account="personal",
            yes=True,
            data_dir=env["data_dir"],
            daemon_detector=lambda: (False, None),
            neo4j_session_factory=lambda: env["sess"],
            qdrant_factory=lambda: env["qdrant"],
        )
        assert rc == 3
        err = capsys.readouterr().err
        assert "Interleaved migration" in err

    def test_invalid_account_label_rejected(self, env) -> None:
        rc = run_migrate_gmail_multiaccount(
            account="Invalid-Caps",
            yes=True,
            data_dir=env["data_dir"],
            daemon_detector=lambda: (False, None),
            neo4j_session_factory=lambda: env["sess"],
            qdrant_factory=lambda: env["qdrant"],
        )
        assert rc == 2


# ─── Post-pass validation (race detection) ───────────────────────────


class TestValidatePostPass:
    def test_clean_db_is_empty(self, tmp_path: Path) -> None:
        db = tmp_path / "queue.db"
        _seed_queue_db(db)
        migrate_queue(db, "personal")
        rows, keys = validate_post_pass(db)
        assert rows == []
        assert keys == []

    def test_detects_late_inserted_old_shape_row(
        self, tmp_path: Path
    ) -> None:
        db = tmp_path / "queue.db"
        _seed_queue_db(db)
        migrate_queue(db, "personal")
        # Simulate a daemon writing an old-shape row *after* the
        # migrate's UPDATE pass committed.
        conn = sqlite3.connect(str(db))
        conn.execute(
            "INSERT INTO queue "
            "(id, source_type, source_id, operation, status, payload, "
            " enqueued_at) "
            "VALUES ('late1', 'gmail', 'gmail://thread/LATE', "
            "'created', 'pending', '{}', '2026-04-26T01:00:00Z')"
        )
        conn.commit()
        conn.close()
        rows, keys = validate_post_pass(db)
        assert rows == [("late1", "gmail://thread/LATE")]
        assert keys == []

    def test_detects_bare_cursor_key(self, tmp_path: Path) -> None:
        db = tmp_path / "queue.db"
        _seed_queue_db(db)
        migrate_queue(db, "personal")
        # Simulate a daemon promoting an old-style cursor key after the
        # migrate ran.
        conn = sqlite3.connect(str(db))
        conn.execute(
            "INSERT INTO cursors VALUES "
            "('gmail', '{\"x\":9}', '2026-04-26T02:00:00Z')"
        )
        conn.commit()
        conn.close()
        rows, keys = validate_post_pass(db)
        assert rows == []
        assert keys == ["gmail"]

    def test_missing_db_returns_empty(self, tmp_path: Path) -> None:
        rows, keys = validate_post_pass(tmp_path / "nope.db")
        assert rows == []
        assert keys == []


# ─── Advisory lockfile ──────────────────────────────────────────────


class TestMigrateLock:
    def test_acquire_creates_lockfile(self, tmp_path: Path) -> None:
        lock = tmp_path / "migrate.lock"
        fd = acquire_migrate_lock(lock)
        try:
            assert lock.exists()
            assert lock.read_text().strip().isdigit()
        finally:
            release_migrate_lock(lock, fd)
        assert not lock.exists()

    def test_concurrent_acquire_raises(self, tmp_path: Path) -> None:
        lock = tmp_path / "migrate.lock"
        fd = acquire_migrate_lock(lock)
        try:
            with pytest.raises(MigrateLockHeld) as exc_info:
                acquire_migrate_lock(lock)
            assert "another fieldnotes migrate" in str(exc_info.value).lower()
            assert exc_info.value.lock_path == lock
        finally:
            release_migrate_lock(lock, fd)

    def test_release_after_unlinked_is_safe(self, tmp_path: Path) -> None:
        lock = tmp_path / "migrate.lock"
        fd = acquire_migrate_lock(lock)
        # Someone else removed the file.
        lock.unlink()
        # Should not raise.
        release_migrate_lock(lock, fd)


# ─── Race-window integration tests (fn-3ox) ─────────────────────────


class TestRaceFixes:
    def test_post_pass_detects_late_inserts(self, env, capsys) -> None:
        """Simulate a daemon write between migrate_queue's UPDATE pass
        and the post-pass scan: post-pass reports N=1 and exits non-zero.
        """
        from worker.cli import migrate as migrate_mod

        original_migrate_queue = migrate_mod.migrate_queue

        def racing_migrate_queue(db_path: Path, account: str):
            result = original_migrate_queue(db_path, account)
            # After the UPDATE pass committed, a daemon (still alive
            # under --force-running) sneaks in an old-shape row.
            conn = sqlite3.connect(str(db_path))
            conn.execute(
                "INSERT INTO queue "
                "(id, source_type, source_id, operation, status, "
                " payload, enqueued_at) "
                "VALUES ('race1', 'gmail', 'gmail://thread/RACE', "
                "'created', 'pending', '{}', "
                "'2026-04-26T03:00:00Z')"
            )
            conn.commit()
            conn.close()
            return result

        migrate_mod.migrate_queue = racing_migrate_queue
        try:
            rc = run_migrate_gmail_multiaccount(
                account="personal",
                yes=True,
                force_running=True,
                data_dir=env["data_dir"],
                daemon_detector=lambda: (True, "racing-daemon"),
                neo4j_session_factory=lambda: env["sess"],
                qdrant_factory=lambda: env["qdrant"],
            )
        finally:
            migrate_mod.migrate_queue = original_migrate_queue

        assert rc == 5
        err = capsys.readouterr().err
        assert "gmail://thread/RACE" in err
        assert "Post-migration validation found 1 old-shape rows" in err
        assert "stop the daemon" in err
        # Neo4j/Qdrant migration was *not* run (we short-circuited).
        assert env["sess"].docs[0]["source_id"] == "gmail://thread/T1"

    def test_post_pass_clean_when_serialized(self, env, capsys) -> None:
        """With the lockfile already held by another invocation and
        --no-force-running (default), migrate refuses to start with a
        clear, actionable message.
        """
        lock_path = env["data_dir"] / "migrate.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        held_fd = acquire_migrate_lock(lock_path)
        try:
            rc = run_migrate_gmail_multiaccount(
                account="personal",
                yes=True,
                data_dir=env["data_dir"],
                daemon_detector=lambda: (False, None),
                neo4j_session_factory=lambda: env["sess"],
                qdrant_factory=lambda: env["qdrant"],
            )
        finally:
            release_migrate_lock(lock_path, held_fd)
        assert rc == 4
        err = capsys.readouterr().err
        assert "another fieldnotes migrate is in progress" in err.lower()
        assert "--no-serialize" in err
        # Queue untouched: no rows were rewritten.
        sids = {
            r[0]
            for r in sqlite3.connect(str(env["db"])).execute(
                "SELECT source_id FROM queue"
            )
        }
        assert "gmail://thread/T1" in sids

    def test_post_pass_clean_after_full_migrate(self, env) -> None:
        """End-to-end happy path: post-pass scan finds zero old-shape
        rows after the migrate completes successfully.
        """
        rc = run_migrate_gmail_multiaccount(
            account="personal",
            yes=True,
            data_dir=env["data_dir"],
            daemon_detector=lambda: (False, None),
            neo4j_session_factory=lambda: env["sess"],
            qdrant_factory=lambda: env["qdrant"],
            config_path=env["config_path"],
        )
        assert rc == 0
        rows, keys = validate_post_pass(env["db"])
        assert rows == []
        assert keys == []
        # Lockfile cleaned up.
        assert not (env["data_dir"] / "migrate.lock").exists()

    def test_concurrent_migrates_one_wins(self, env, tmp_path) -> None:
        """Two parallel migrate invocations: one wins the advisory
        lock and proceeds; the other refuses with rc=4.

        Determinism: monkey-patch ``detect_queue_counts`` to block the
        winning thread inside the locked section until the loser has
        attempted (and failed) acquisition.
        """
        import threading

        from worker.cli import migrate as migrate_mod

        original_detect = migrate_mod.detect_queue_counts
        winner_in_lock = threading.Event()
        loser_done = threading.Event()
        first_call_seen = threading.Lock()
        first_call_taken = {"taken": False}

        def slow_detect(db_path: Path):
            with first_call_seen:
                is_first = not first_call_taken["taken"]
                first_call_taken["taken"] = True
            if is_first:
                winner_in_lock.set()
                loser_done.wait(timeout=5.0)
            return original_detect(db_path)

        migrate_mod.detect_queue_counts = slow_detect

        # Build a separate env for the loser so its in-memory Neo4j /
        # Qdrant fakes don't collide with the winner's; both share
        # ``data_dir`` (and queue.db) by design — that's the point.
        loser_sess = _FakeNeo4jSession()
        loser_sess.docs.append({"source_id": "gmail://thread/T1"})
        loser_qdrant = _FakeQdrantClient()
        loser_qdrant.add(1, "gmail://thread/T1")

        results: dict[str, int] = {}

        def run_winner() -> None:
            results["winner"] = run_migrate_gmail_multiaccount(
                account="personal",
                yes=True,
                data_dir=env["data_dir"],
                daemon_detector=lambda: (False, None),
                neo4j_session_factory=lambda: env["sess"],
                qdrant_factory=lambda: env["qdrant"],
                config_path=env["config_path"],
            )

        def run_loser() -> None:
            winner_in_lock.wait(timeout=5.0)
            results["loser"] = run_migrate_gmail_multiaccount(
                account="personal",
                yes=True,
                data_dir=env["data_dir"],
                daemon_detector=lambda: (False, None),
                neo4j_session_factory=lambda: loser_sess,
                qdrant_factory=lambda: loser_qdrant,
                config_path=env["config_path"],
            )
            loser_done.set()

        winner_t = threading.Thread(target=run_winner)
        loser_t = threading.Thread(target=run_loser)
        winner_t.start()
        loser_t.start()
        winner_t.join(timeout=15.0)
        loser_t.join(timeout=15.0)

        migrate_mod.detect_queue_counts = original_detect

        assert not winner_t.is_alive() and not loser_t.is_alive()
        assert results["loser"] == 4
        assert results["winner"] == 0
        # Winner's migrate landed.
        sids = {
            r[0]
            for r in sqlite3.connect(str(env["db"])).execute(
                "SELECT source_id FROM queue"
            )
        }
        assert "gmail://personal/thread/T1" in sids


# ─── --no-serialize parser flag ──────────────────────────────────────


class TestParserNoSerialize:
    def test_default_no_serialize_false(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["migrate", "gmail-multiaccount"])
        assert args.no_serialize is False

    def test_no_serialize_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(
            ["migrate", "gmail-multiaccount", "--no-serialize"]
        )
        assert args.no_serialize is True


class TestNoSerializeBypass:
    def test_no_serialize_bypasses_lock_contention(
        self, env
    ) -> None:
        """With ``serialize=False``, migrate ignores an existing
        lockfile and proceeds.  This is the documented escape hatch.
        """
        lock_path = env["data_dir"] / "migrate.lock"
        held_fd = acquire_migrate_lock(lock_path)
        try:
            rc = run_migrate_gmail_multiaccount(
                account="personal",
                yes=True,
                serialize=False,
                data_dir=env["data_dir"],
                daemon_detector=lambda: (False, None),
                neo4j_session_factory=lambda: env["sess"],
                qdrant_factory=lambda: env["qdrant"],
                config_path=env["config_path"],
            )
        finally:
            release_migrate_lock(lock_path, held_fd)
        assert rc == 0
