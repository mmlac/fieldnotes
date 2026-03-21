"""Tests for the daily digest module.

Covers DigestQuerier (unit, with mocked Neo4j), the run_digest() CLI
function, and MCP tool registration.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from worker.query.digest import (
    DigestQuerier,
    DigestResult,
    SourceActivity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_querier(
    source_rows_by_call: list[list[dict[str, Any]]] | None = None,
    connections_rows: list[dict[str, Any]] | None = None,
    topics_rows: list[dict[str, Any]] | None = None,
    neo4j_side_effect: Exception | None = None,
) -> tuple[DigestQuerier, MagicMock]:
    """Build a DigestQuerier with a mocked Neo4j driver.

    source_rows_by_call: list of row lists, one per source-type execute_read call.
    connections_rows / topics_rows: rows for those specific queries.
    """
    with patch("worker.query.digest.GraphDatabase") as mock_gdb_cls:
        mock_driver = MagicMock()
        mock_gdb_cls.driver.return_value = mock_driver

        # Build the per-call return queue.
        # _query_sources makes one execute_read per source type (5 total by default).
        # _query_new_connections and _query_new_topics each open their own session.
        # We handle per-session mocking below.
        call_queue: list[list[dict]] = list(source_rows_by_call or [])
        # Pad with empty lists for any missing source calls.
        while len(call_queue) < 5:
            call_queue.append([])

        call_index: list[int] = [0]

        def _make_session_for_sources() -> MagicMock:
            session = MagicMock()
            session.__enter__ = lambda s: s
            session.__exit__ = MagicMock(return_value=False)

            def _execute_read(fn: Any) -> Any:
                idx = call_index[0]
                call_index[0] += 1
                rows = call_queue[idx] if idx < len(call_queue) else []
                return fn(
                    MagicMock(
                        **{"run.return_value": MagicMock(**{"data.return_value": rows})}
                    )
                )

            session.execute_read.side_effect = _execute_read
            return session

        def _make_session_for_meta(rows: list[dict]) -> MagicMock:
            session = MagicMock()
            session.__enter__ = lambda s: s
            session.__exit__ = MagicMock(return_value=False)
            session.execute_read.side_effect = lambda fn: fn(
                MagicMock(
                    **{"run.return_value": MagicMock(**{"data.return_value": rows})}
                )
            )
            return session

        sessions: list[MagicMock] = []

        if neo4j_side_effect is not None:
            err_session = MagicMock()
            err_session.__enter__ = lambda s: s
            err_session.__exit__ = MagicMock(return_value=False)
            err_session.execute_read.side_effect = neo4j_side_effect
            mock_driver.session.return_value = err_session
        else:
            # _query_sources opens one session (all 5 source queries share it).
            # _query_new_connections opens its own session.
            # _query_new_topics opens its own session.
            conns = (
                connections_rows
                if connections_rows is not None
                else [{"new_connections": 0}]
            )
            tops = topics_rows if topics_rows is not None else [{"new_topics": 0}]

            sessions = [
                _make_session_for_sources(),
                _make_session_for_meta(conns),
                _make_session_for_meta(tops),
            ]
            session_index: list[int] = [0]

            def _next_session() -> MagicMock:
                idx = session_index[0]
                session_index[0] += 1
                return sessions[idx] if idx < len(sessions) else sessions[-1]

            mock_driver.session.side_effect = _next_session

        from worker.config import Neo4jConfig

        querier = DigestQuerier(
            Neo4jConfig(uri="bolt://localhost:7687", user="neo4j", password="test")
        )
        querier._driver = mock_driver
        return querier, mock_driver


def _source_row(
    created: int = 0,
    modified: int = 0,
    completed: int = 0,
    highlights: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "created_count": created,
        "modified_count": modified,
        "completed_count": completed,
        "highlights": highlights or [],
    }


# ---------------------------------------------------------------------------
# DigestQuerier unit tests
# ---------------------------------------------------------------------------


class TestDigestDefault24h:
    """Default query returns sources for all active source types."""

    @patch("worker.query.digest.GraphDatabase")
    def test_three_source_types_returned(self, mock_gdb_cls: MagicMock) -> None:
        # obsidian=2 modified, omnifocus=1 completed, gmail=3 new, repos=0, apps=0
        source_rows = [
            [_source_row(modified=2, highlights=["Note A", "Note B"])],  # obsidian
            [_source_row(completed=1, highlights=["Task 1"])],  # omnifocus
            [_source_row(created=3, highlights=["Subject 1"])],  # gmail
            [],  # repositories (empty)
            [],  # apps (empty)
        ]
        querier, _ = _make_querier(source_rows_by_call=source_rows)
        result = querier.query(since="24h", until="now")

        assert result.error is None
        source_types = {s.source_type for s in result.sources}
        assert "obsidian" in source_types
        assert "omnifocus" in source_types
        assert "gmail" in source_types
        assert len(result.sources) == 3

    @patch("worker.query.digest.GraphDatabase")
    def test_since_defaults_to_approx_24h_ago(self, mock_gdb_cls: MagicMock) -> None:
        querier, _ = _make_querier()
        before = datetime.now(timezone.utc)
        result = querier.query()

        since_dt = datetime.strptime(result.since, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
        expected_since = before - timedelta(hours=24)
        assert abs((since_dt - expected_since).total_seconds()) < 10


class TestDigestCustomRange:
    @patch("worker.query.digest.GraphDatabase")
    def test_7d_since_timestamp_is_7_days_ago(self, mock_gdb_cls: MagicMock) -> None:
        querier, _ = _make_querier()
        before = datetime.now(timezone.utc)
        result = querier.query(since="7d")

        since_dt = datetime.strptime(result.since, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
        expected = before - timedelta(days=7)
        assert abs((since_dt - expected).total_seconds()) < 10


class TestDigestActivityCounts:
    @patch("worker.query.digest.GraphDatabase")
    def test_counts_are_correct_per_source_type(self, mock_gdb_cls: MagicMock) -> None:
        source_rows = [
            [_source_row(modified=5, highlights=["a", "b"])],  # obsidian: 5 modified
            [
                _source_row(completed=2, modified=3, highlights=["t1"])
            ],  # omnifocus: 2 completed + 3 modified
            [_source_row(created=10, highlights=["e1"])],  # gmail: 10 new
            [],
            [],
        ]
        querier, _ = _make_querier(source_rows_by_call=source_rows)
        result = querier.query()

        sources = {s.source_type: s for s in result.sources}

        assert sources["obsidian"].modified == 5
        # omnifocus: completed (2) is folded into modified
        omnifocus = sources["omnifocus"]
        assert getattr(omnifocus, "_completed", 0) == 2
        assert omnifocus.modified == 5  # 2 completed + 3 modified
        assert sources["gmail"].created == 10


class TestDigestHighlightsLimitedTo5:
    @patch("worker.query.digest.GraphDatabase")
    def test_highlights_at_most_5_entries(self, mock_gdb_cls: MagicMock) -> None:
        # Neo4j returns 20 titles, but the Cypher query uses $limit=5.
        # The querier passes _HIGHLIGHTS_LIMIT=5 to the query. We simulate
        # Neo4j already having applied the limit.
        titles = [f"Note {i}" for i in range(5)]
        source_rows = [
            [_source_row(modified=20, highlights=titles)],
            [],
            [],
            [],
            [],
        ]
        querier, _ = _make_querier(source_rows_by_call=source_rows)
        result = querier.query()

        obsidian = next(s for s in result.sources if s.source_type == "obsidian")
        assert len(obsidian.highlights) <= 5


class TestDigestNewConnectionsCount:
    @patch("worker.query.digest.GraphDatabase")
    def test_new_connections_is_7(self, mock_gdb_cls: MagicMock) -> None:
        source_rows = [[_source_row(modified=1)]] + [[] for _ in range(4)]
        querier, _ = _make_querier(
            source_rows_by_call=source_rows,
            connections_rows=[{"new_connections": 7}],
        )
        result = querier.query()

        assert result.new_connections == 7


class TestDigestNewTopicsCount:
    @patch("worker.query.digest.GraphDatabase")
    def test_new_topics_is_2(self, mock_gdb_cls: MagicMock) -> None:
        source_rows = [[_source_row(modified=1)]] + [[] for _ in range(4)]
        querier, _ = _make_querier(
            source_rows_by_call=source_rows,
            topics_rows=[{"new_topics": 2}],
        )
        result = querier.query()

        assert result.new_topics == 2


class TestDigestNoActivity:
    @patch("worker.query.digest.GraphDatabase")
    def test_no_activity_empty_sources_no_error(self, mock_gdb_cls: MagicMock) -> None:
        querier, _ = _make_querier(
            source_rows_by_call=[[] for _ in range(5)],
            connections_rows=[{"new_connections": 0}],
            topics_rows=[{"new_topics": 0}],
        )
        result = querier.query()

        assert result.error is None
        assert result.sources == []
        assert result.new_connections == 0
        assert result.new_topics == 0


class TestDigestWithSummarize:
    @patch("worker.query.digest.GraphDatabase")
    def test_summarize_true_populates_summary(self, mock_gdb_cls: MagicMock) -> None:
        from worker.cli.digest import run_digest

        with (
            patch("worker.cli.digest.load_config") as mock_load_cfg,
            patch("worker.cli.digest.DigestQuerier") as mock_querier_cls,
            patch("worker.cli.digest._generate_summary") as mock_gen_summary,
        ):
            mock_load_cfg.return_value = MagicMock()
            mock_querier = MagicMock()
            mock_querier_cls.return_value.__enter__ = lambda s: mock_querier
            mock_querier_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_querier.query.return_value = DigestResult(
                since="2026-03-18T00:00:00Z",
                until="2026-03-19T00:00:00Z",
                sources=[
                    SourceActivity(
                        source_type="obsidian", modified=2, highlights=["Note A"]
                    ),
                ],
                new_connections=1,
                new_topics=0,
            )
            mock_gen_summary.return_value = (
                "You had a busy day with 2 Obsidian changes."
            )

            rc = run_digest(summarize=True)
            assert rc == 0
            mock_gen_summary.assert_called_once()

    @patch("worker.query.digest.GraphDatabase")
    def test_summary_field_is_populated(self, mock_gdb_cls: MagicMock) -> None:
        from worker.cli.digest import run_digest

        with (
            patch("worker.cli.digest.load_config") as mock_load_cfg,
            patch("worker.cli.digest.DigestQuerier") as mock_querier_cls,
            patch("worker.cli.digest._generate_summary") as mock_gen_summary,
        ):
            mock_load_cfg.return_value = MagicMock()
            mock_querier = MagicMock()
            mock_querier_cls.return_value.__enter__ = lambda s: mock_querier
            mock_querier_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_querier.query.return_value = DigestResult(
                since="2026-03-18T00:00:00Z",
                until="2026-03-19T00:00:00Z",
                sources=[
                    SourceActivity(
                        source_type="gmail", created=5, highlights=["Email 1"]
                    ),
                ],
            )
            summary_text = "5 emails received today."
            mock_gen_summary.return_value = summary_text

            rc = run_digest(summarize=True)
            assert rc == 0
            # Verify _generate_summary was called with a DigestResult
            call_args = mock_gen_summary.call_args
            assert isinstance(call_args[0][0], DigestResult)


class TestDigestWithoutSummarize:
    @patch("worker.query.digest.GraphDatabase")
    def test_summarize_false_no_llm_call(self, mock_gdb_cls: MagicMock) -> None:
        from worker.cli.digest import run_digest

        with (
            patch("worker.cli.digest.load_config") as mock_load_cfg,
            patch("worker.cli.digest.DigestQuerier") as mock_querier_cls,
            patch("worker.cli.digest._generate_summary") as mock_gen_summary,
        ):
            mock_load_cfg.return_value = MagicMock()
            mock_querier = MagicMock()
            mock_querier_cls.return_value.__enter__ = lambda s: mock_querier
            mock_querier_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_querier.query.return_value = DigestResult(
                since="2026-03-18T00:00:00Z",
                until="2026-03-19T00:00:00Z",
                sources=[],
            )

            rc = run_digest(summarize=False)
            assert rc == 0
            mock_gen_summary.assert_not_called()

    @patch("worker.query.digest.GraphDatabase")
    def test_default_no_llm_call(self, mock_gdb_cls: MagicMock) -> None:
        from worker.cli.digest import run_digest

        with (
            patch("worker.cli.digest.load_config") as mock_load_cfg,
            patch("worker.cli.digest.DigestQuerier") as mock_querier_cls,
            patch("worker.cli.digest._generate_summary") as mock_gen_summary,
        ):
            mock_load_cfg.return_value = MagicMock()
            mock_querier = MagicMock()
            mock_querier_cls.return_value.__enter__ = lambda s: mock_querier
            mock_querier_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_querier.query.return_value = DigestResult(
                since="2026-03-18T00:00:00Z",
                until="2026-03-19T00:00:00Z",
            )

            rc = run_digest()
            assert rc == 0
            mock_gen_summary.assert_not_called()


class TestDigestSourceTypeMissing:
    @patch("worker.query.digest.GraphDatabase")
    def test_missing_source_types_excluded(self, mock_gdb_cls: MagicMock) -> None:
        # Only obsidian and gmail have activity; omnifocus/repos/apps return nothing.
        source_rows = [
            [_source_row(modified=3, highlights=["Note"])],  # obsidian
            [],  # omnifocus: no activity
            [_source_row(created=2, highlights=["Email"])],  # gmail
            [],  # repositories
            [],  # apps
        ]
        querier, _ = _make_querier(source_rows_by_call=source_rows)
        result = querier.query()

        types = {s.source_type for s in result.sources}
        assert "obsidian" in types
        assert "gmail" in types
        assert "omnifocus" not in types
        assert "repositories" not in types
        assert "apps" not in types


# ---------------------------------------------------------------------------
# CLI output tests
# ---------------------------------------------------------------------------


class TestCliDigestHumanOutput:
    @patch("worker.cli.digest.DigestQuerier")
    @patch("worker.cli.digest.load_config")
    def test_human_output_contains_source_names_and_counts(
        self,
        mock_load_config: MagicMock,
        mock_querier_cls: MagicMock,
        capsys: pytest.CaptureFixture,
    ) -> None:
        from worker.cli.digest import run_digest

        mock_load_config.return_value = MagicMock()
        mock_querier = MagicMock()
        mock_querier_cls.return_value.__enter__ = lambda s: mock_querier
        mock_querier_cls.return_value.__exit__ = MagicMock(return_value=False)

        activity = SourceActivity(
            source_type="obsidian",
            modified=3,
            highlights=["Note Alpha", "Note Beta"],
        )
        mock_querier.query.return_value = DigestResult(
            since="2026-03-18T00:00:00Z",
            until="2026-03-19T00:00:00Z",
            sources=[activity],
            new_connections=2,
            new_topics=1,
        )

        rc = run_digest()
        captured = capsys.readouterr()

        assert rc == 0
        assert "Obsidian" in captured.out
        assert "Note Alpha" in captured.out or "Note Beta" in captured.out
        assert "Cross-source" in captured.out
        assert "Topics" in captured.out

    @patch("worker.cli.digest.DigestQuerier")
    @patch("worker.cli.digest.load_config")
    def test_human_output_no_activity_message(
        self,
        mock_load_config: MagicMock,
        mock_querier_cls: MagicMock,
        capsys: pytest.CaptureFixture,
    ) -> None:
        from worker.cli.digest import run_digest

        mock_load_config.return_value = MagicMock()
        mock_querier = MagicMock()
        mock_querier_cls.return_value.__enter__ = lambda s: mock_querier
        mock_querier_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_querier.query.return_value = DigestResult(
            since="2026-03-18T00:00:00Z",
            until="2026-03-19T00:00:00Z",
        )

        rc = run_digest()
        captured = capsys.readouterr()

        assert rc == 0
        assert "No activity" in captured.out


class TestCliDigestJsonOutput:
    @patch("worker.cli.digest.DigestQuerier")
    @patch("worker.cli.digest.load_config")
    def test_json_output_has_required_fields(
        self,
        mock_load_config: MagicMock,
        mock_querier_cls: MagicMock,
        capsys: pytest.CaptureFixture,
    ) -> None:
        from worker.cli.digest import run_digest

        mock_load_config.return_value = MagicMock()
        mock_querier = MagicMock()
        mock_querier_cls.return_value.__enter__ = lambda s: mock_querier
        mock_querier_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_querier.query.return_value = DigestResult(
            since="2026-03-18T00:00:00Z",
            until="2026-03-19T00:00:00Z",
            sources=[
                SourceActivity(source_type="gmail", created=5, highlights=["Sub1"]),
            ],
            new_connections=3,
            new_topics=0,
        )

        rc = run_digest(json_output=True)
        captured = capsys.readouterr()

        assert rc == 0
        data = json.loads(captured.out)
        assert "sources" in data
        assert "new_connections" in data
        assert "new_topics" in data
        assert "since" in data
        assert "until" in data
        assert data["new_connections"] == 3
        assert len(data["sources"]) == 1
        assert data["sources"][0]["source_type"] == "gmail"

    @patch("worker.cli.digest.DigestQuerier")
    @patch("worker.cli.digest.load_config")
    def test_json_output_is_valid_json(
        self,
        mock_load_config: MagicMock,
        mock_querier_cls: MagicMock,
        capsys: pytest.CaptureFixture,
    ) -> None:
        from worker.cli.digest import run_digest

        mock_load_config.return_value = MagicMock()
        mock_querier = MagicMock()
        mock_querier_cls.return_value.__enter__ = lambda s: mock_querier
        mock_querier_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_querier.query.return_value = DigestResult(
            since="2026-03-18T00:00:00Z",
            until="2026-03-19T00:00:00Z",
        )

        rc = run_digest(json_output=True)
        captured = capsys.readouterr()

        assert rc == 0
        # Should not raise
        json.loads(captured.out)


class TestCliDigestWithSummaryOutput:
    @patch("worker.cli.digest.DigestQuerier")
    @patch("worker.cli.digest.load_config")
    def test_summary_section_appears_in_output(
        self,
        mock_load_config: MagicMock,
        mock_querier_cls: MagicMock,
        capsys: pytest.CaptureFixture,
    ) -> None:
        from worker.cli.digest import run_digest

        mock_load_config.return_value = MagicMock()
        mock_querier = MagicMock()
        mock_querier_cls.return_value.__enter__ = lambda s: mock_querier
        mock_querier_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_querier.query.return_value = DigestResult(
            since="2026-03-18T00:00:00Z",
            until="2026-03-19T00:00:00Z",
            sources=[SourceActivity(source_type="obsidian", modified=1)],
            summary="You modified 1 Obsidian note.",
        )

        # Bypass _generate_summary by passing summarize=False but pre-setting summary
        with patch(
            "worker.cli.digest._generate_summary",
            return_value="You modified 1 Obsidian note.",
        ):
            rc = run_digest(summarize=True)

        captured = capsys.readouterr()
        assert rc == 0
        assert "Summary:" in captured.out


# ---------------------------------------------------------------------------
# MCP tool registration
# ---------------------------------------------------------------------------


class TestMcpDigestTool:
    def test_digest_tool_registered(self) -> None:
        from worker.mcp_server import TOOLS

        names = [t.name for t in TOOLS]
        assert "digest" in names

    def test_digest_schema_has_since_and_summarize(self) -> None:
        from worker.mcp_server import TOOLS

        digest = next(t for t in TOOLS if t.name == "digest")
        props = digest.inputSchema["properties"]
        assert "since" in props
        assert "summarize" in props
