"""Tests for the reindex-references CLI command.

All tests mock out Neo4j so they run without running services.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from worker.cli.reindex_references import (
    SUPPORTED_LABELS,
    _LABEL_SPEC,
    _fetch_nodes,
    _write_hints_tx,
    run_reindex_references,
)
from worker.parsers.base import GraphHint


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_record(source_id: str, chunks: list[str]) -> MagicMock:
    rec = MagicMock()
    rec.__getitem__ = lambda self, k: source_id if k == "source_id" else chunks
    return rec


def _neo4j_session_from_records(records: list[MagicMock]) -> MagicMock:
    result = MagicMock()
    result.__iter__ = MagicMock(return_value=iter(records))
    session = MagicMock()
    session.run.return_value = result
    return session


def _make_driver(session: MagicMock) -> MagicMock:
    driver = MagicMock()
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=session)
    ctx.__exit__ = MagicMock(return_value=False)
    driver.session.return_value = ctx
    return driver


# ── _fetch_nodes ─────────────────────────────────────────────────────────────


class TestFetchNodes:
    def test_returns_source_id_and_text(self):
        records = [_make_record("gmail://acc/message/1", ["Hello world"])]
        session = _neo4j_session_from_records(records)
        rows = _fetch_nodes(session, "Email", None)
        assert rows == [("gmail://acc/message/1", "Hello world")]

    def test_joins_multiple_chunks(self):
        records = [_make_record("doc:1", ["chunk one", "chunk two"])]
        session = _neo4j_session_from_records(records)
        rows = _fetch_nodes(session, "Email", None)
        assert rows == [("doc:1", "chunk one\nchunk two")]

    def test_skips_none_source_id(self):
        records = [_make_record(None, ["text"])]
        session = _neo4j_session_from_records(records)
        rows = _fetch_nodes(session, "Email", None)
        assert rows == []

    def test_source_type_filter_passed_to_query(self):
        session = _neo4j_session_from_records([])
        _fetch_nodes(session, "File", "obsidian")
        call_args = session.run.call_args
        assert "st" in call_args.kwargs or (
            len(call_args.args) >= 2 and call_args.args[1] == "obsidian"
        ) or call_args.kwargs.get("st") == "obsidian"

    def test_empty_chunks_returns_empty_text(self):
        records = [_make_record("doc:2", [])]
        session = _neo4j_session_from_records(records)
        rows = _fetch_nodes(session, "Email", None)
        assert rows == [("doc:2", "")]


# ── _write_hints_tx ───────────────────────────────────────────────────────────


class TestWriteHintsTx:
    def test_calls_write_graph_hint_for_each_hint(self):
        tx = MagicMock()
        hint1 = GraphHint(
            subject_id="doc:1",
            subject_label="Email",
            predicate="REFERENCES",
            object_id="slack://T1/C1/123",
            object_label="SlackMessage",
        )
        hint2 = GraphHint(
            subject_id="doc:1",
            subject_label="Email",
            predicate="REFERENCES",
            object_id="gmail://acc/message/abc",
            object_label="Email",
        )
        with patch("worker.cli.reindex_references._write_graph_hint") as mock_wgh:
            _write_hints_tx(tx, [hint1, hint2])
            assert mock_wgh.call_count == 2
            mock_wgh.assert_any_call(tx, hint1)
            mock_wgh.assert_any_call(tx, hint2)

    def test_empty_hints_is_no_op(self):
        tx = MagicMock()
        with patch("worker.cli.reindex_references._write_graph_hint") as mock_wgh:
            _write_hints_tx(tx, [])
            mock_wgh.assert_not_called()


# ── Fixture corpus ────────────────────────────────────────────────────────────
#
# Three documents, each referencing a different source type:
#   CalendarEvent  → obsidian note
#   File (obsidian) → gmail message
#   Email          → slack permalink
#
# After backfill, there should be exactly 3 REFERENCES edges.

_CALENDAR_TEXT = (
    "See obsidian://open?vault=Personal&file=Meetings%2FKris.md for prep notes."
)
_OBSIDIAN_TEXT = "Reference: gmail://acct@gmail.com/message/abc123 from last week."
_EMAIL_TEXT = (
    "Thread in https://myteam.slack.com/archives/C01234567/p1234567890000100 here."
)


def _make_fixture_session() -> MagicMock:
    """Session that returns the 3-document fixture corpus across 4 label queries."""

    def _run(query, **kwargs):
        result = MagicMock()
        st = kwargs.get("st")
        label_in_query = ""
        for lbl in ("CalendarEvent", "Email", "SlackMessage", "File"):
            if f"MATCH (n:{lbl}" in query:
                label_in_query = lbl
                break

        if label_in_query == "CalendarEvent":
            records = [_make_record("google-calendar://acc/event/evt1", [_CALENDAR_TEXT])]
        elif label_in_query == "File" and st == "obsidian":
            records = [_make_record("/vault/Meetings/Kris.md", [_OBSIDIAN_TEXT])]
        elif label_in_query == "Email":
            records = [_make_record("gmail://acct@gmail.com/message/msg1", [_EMAIL_TEXT])]
        elif label_in_query == "SlackMessage":
            records = []
        else:
            records = []

        result.__iter__ = MagicMock(return_value=iter(records))
        return result

    session = MagicMock()
    session.run.side_effect = _run
    return session


# ── TestReindexReferences_DryRun ──────────────────────────────────────────────


class TestReindexReferences_DryRun:
    def test_dry_run_prints_count_and_writes_nothing(self, capsys):
        session = _make_fixture_session()
        driver = _make_driver(session)

        with patch("worker.cli.reindex_references.GraphDatabase") as mock_gdb, patch(
            "worker.cli.reindex_references.load_config"
        ) as mock_cfg:
            mock_gdb.driver.return_value = driver
            mock_cfg.return_value.neo4j.uri = "bolt://localhost:7687"
            mock_cfg.return_value.neo4j.user = "neo4j"
            mock_cfg.return_value.neo4j.password = ""
            mock_cfg.return_value.sources = {}

            rc = run_reindex_references(dry_run=True)

        assert rc == 0
        captured = capsys.readouterr()
        assert "Dry run" in captured.out
        assert "REFERENCES edge" in captured.out
        session.execute_write.assert_not_called()

    def test_dry_run_counts_three_edges_for_fixture(self, capsys):
        session = _make_fixture_session()
        driver = _make_driver(session)

        with patch("worker.cli.reindex_references.GraphDatabase") as mock_gdb, patch(
            "worker.cli.reindex_references.load_config"
        ) as mock_cfg, patch(
            "worker.cli.reindex_references._configure_vault_map"
        ):
            mock_gdb.driver.return_value = driver
            mock_cfg.return_value.neo4j.uri = "bolt://localhost:7687"
            mock_cfg.return_value.neo4j.user = "neo4j"
            mock_cfg.return_value.neo4j.password = ""
            mock_cfg.return_value.sources = {}

            rc = run_reindex_references(dry_run=True)

        assert rc == 0
        captured = capsys.readouterr()
        assert "3 REFERENCES edge" in captured.out


# ── TestReindexReferences_LiveRun ─────────────────────────────────────────────


class TestReindexReferences_LiveRun:
    def test_live_run_calls_execute_write(self):
        session = _make_fixture_session()
        driver = _make_driver(session)

        with patch("worker.cli.reindex_references.GraphDatabase") as mock_gdb, patch(
            "worker.cli.reindex_references.load_config"
        ) as mock_cfg, patch(
            "worker.cli.reindex_references._configure_vault_map"
        ):
            mock_gdb.driver.return_value = driver
            mock_cfg.return_value.neo4j.uri = "bolt://localhost:7687"
            mock_cfg.return_value.neo4j.user = "neo4j"
            mock_cfg.return_value.neo4j.password = ""
            mock_cfg.return_value.sources = {}

            rc = run_reindex_references(dry_run=False)

        assert rc == 0
        # Should have called execute_write once per node with hints (3 nodes × 1 call each)
        assert session.execute_write.call_count == 3

    def test_live_run_prints_edge_count(self, capsys):
        session = _make_fixture_session()
        driver = _make_driver(session)

        with patch("worker.cli.reindex_references.GraphDatabase") as mock_gdb, patch(
            "worker.cli.reindex_references.load_config"
        ) as mock_cfg, patch(
            "worker.cli.reindex_references._configure_vault_map"
        ):
            mock_gdb.driver.return_value = driver
            mock_cfg.return_value.neo4j.uri = "bolt://localhost:7687"
            mock_cfg.return_value.neo4j.user = "neo4j"
            mock_cfg.return_value.neo4j.password = ""
            mock_cfg.return_value.sources = {}

            run_reindex_references(dry_run=False)

        captured = capsys.readouterr()
        assert "Created" in captured.out
        assert "REFERENCES edge" in captured.out


# ── TestReindexReferences_Idempotent ─────────────────────────────────────────


class TestReindexReferences_Idempotent:
    def test_running_twice_produces_same_call_count(self):
        """execute_write is called the same number of times on both runs.

        Idempotency in the graph is guaranteed by MERGE in _write_graph_hint;
        here we verify that the caller produces the same write calls.
        """
        def _make_session():
            return _make_fixture_session()

        def _make_drv(sess):
            return _make_driver(sess)

        with patch("worker.cli.reindex_references.GraphDatabase") as mock_gdb, patch(
            "worker.cli.reindex_references.load_config"
        ) as mock_cfg, patch(
            "worker.cli.reindex_references._configure_vault_map"
        ):
            mock_cfg.return_value.neo4j.uri = "bolt://localhost:7687"
            mock_cfg.return_value.neo4j.user = "neo4j"
            mock_cfg.return_value.neo4j.password = ""
            mock_cfg.return_value.sources = {}

            session1 = _make_session()
            mock_gdb.driver.return_value = _make_drv(session1)
            run_reindex_references(dry_run=False)
            first_count = session1.execute_write.call_count

            session2 = _make_session()
            mock_gdb.driver.return_value = _make_drv(session2)
            run_reindex_references(dry_run=False)
            second_count = session2.execute_write.call_count

        assert first_count == second_count


# ── TestReindexReferences_LabelFilter ────────────────────────────────────────


class TestReindexReferences_LabelFilter:
    def test_label_filter_scopes_to_calendar_event_only(self):
        """--label CalendarEvent should only query CalendarEvent nodes."""
        session = _make_fixture_session()
        driver = _make_driver(session)

        with patch("worker.cli.reindex_references.GraphDatabase") as mock_gdb, patch(
            "worker.cli.reindex_references.load_config"
        ) as mock_cfg, patch(
            "worker.cli.reindex_references._configure_vault_map"
        ):
            mock_gdb.driver.return_value = driver
            mock_cfg.return_value.neo4j.uri = "bolt://localhost:7687"
            mock_cfg.return_value.neo4j.user = "neo4j"
            mock_cfg.return_value.neo4j.password = ""
            mock_cfg.return_value.sources = {}

            rc = run_reindex_references(label="CalendarEvent", dry_run=False)

        assert rc == 0
        # Only one node should have been processed (the CalendarEvent)
        assert session.execute_write.call_count == 1

    def test_unknown_label_returns_error(self, capsys):
        with patch("worker.cli.reindex_references.GraphDatabase"), patch(
            "worker.cli.reindex_references.load_config"
        ):
            rc = run_reindex_references(label="UnknownLabel")

        assert rc == 1
        captured = capsys.readouterr()
        assert "unknown label" in captured.err

    def test_obsidian_note_label_uses_source_type_filter(self):
        session = _make_fixture_session()
        driver = _make_driver(session)

        with patch("worker.cli.reindex_references.GraphDatabase") as mock_gdb, patch(
            "worker.cli.reindex_references.load_config"
        ) as mock_cfg, patch(
            "worker.cli.reindex_references._configure_vault_map"
        ):
            mock_gdb.driver.return_value = driver
            mock_cfg.return_value.neo4j.uri = "bolt://localhost:7687"
            mock_cfg.return_value.neo4j.user = "neo4j"
            mock_cfg.return_value.neo4j.password = ""
            mock_cfg.return_value.sources = {}

            rc = run_reindex_references(label="ObsidianNote", dry_run=True)

        assert rc == 0
        # Session.run should have been called once with st="obsidian"
        calls_with_st = [
            c for c in session.run.call_args_list if c.kwargs.get("st") == "obsidian"
        ]
        assert len(calls_with_st) == 1


# ── TestReindexReferences_HandlesMissingVaultMap ─────────────────────────────


class TestReindexReferences_HandlesMissingVaultMap:
    def test_obsidian_url_without_vault_map_produces_dangling_edge(self, capsys):
        """obsidian:// URL with no vault map → URL used as object_id, no crash."""
        obsidian_text = (
            "See obsidian://open?vault=Personal&file=Meetings%2FNote.md for notes."
        )

        def _run(query, **kwargs):
            result = MagicMock()
            if "MATCH (n:Email" in query:
                records = [
                    _make_record(
                        "gmail://acc/message/1", [obsidian_text]
                    )
                ]
            else:
                records = []
            result.__iter__ = MagicMock(return_value=iter(records))
            return result

        session = MagicMock()
        session.run.side_effect = _run
        driver = _make_driver(session)

        with patch("worker.cli.reindex_references.GraphDatabase") as mock_gdb, patch(
            "worker.cli.reindex_references.load_config"
        ) as mock_cfg, patch(
            "worker.cli.reindex_references._configure_vault_map"
        ), patch("worker.parsers.base._obsidian_vaults", None):
            mock_gdb.driver.return_value = driver
            mock_cfg.return_value.neo4j.uri = "bolt://localhost:7687"
            mock_cfg.return_value.neo4j.user = "neo4j"
            mock_cfg.return_value.neo4j.password = ""
            mock_cfg.return_value.sources = {}

            rc = run_reindex_references(label="Email", dry_run=True)

        assert rc == 0
        captured = capsys.readouterr()
        assert "1 REFERENCES edge" in captured.out

    def test_no_crash_when_vault_map_none(self):
        """Command must not raise even when vault map is completely absent."""
        session = _make_fixture_session()
        driver = _make_driver(session)

        with patch("worker.cli.reindex_references.GraphDatabase") as mock_gdb, patch(
            "worker.cli.reindex_references.load_config"
        ) as mock_cfg, patch(
            "worker.cli.reindex_references._configure_vault_map"
        ), patch("worker.parsers.base._obsidian_vaults", None):
            mock_gdb.driver.return_value = driver
            mock_cfg.return_value.neo4j.uri = "bolt://localhost:7687"
            mock_cfg.return_value.neo4j.user = "neo4j"
            mock_cfg.return_value.neo4j.password = ""
            mock_cfg.return_value.sources = {}

            rc = run_reindex_references(dry_run=True)

        assert rc == 0


# ── Module-level constants ────────────────────────────────────────────────────


class TestModuleConstants:
    def test_supported_labels_covers_four_types(self):
        assert set(SUPPORTED_LABELS) == {
            "CalendarEvent",
            "Email",
            "SlackMessage",
            "ObsidianNote",
        }

    def test_label_spec_keys_match_supported_labels(self):
        assert set(_LABEL_SPEC) == set(SUPPORTED_LABELS)
