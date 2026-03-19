"""Tests for the timeline query module.

Covers TimelineQuerier (unit, with mocked Neo4j + Qdrant), the
_parse_relative_time helper, the CLI run_timeline() function, and MCP
tool registration.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from worker.query.timeline import (
    TimelineEntry,
    TimelineQuerier,
    TimelineResult,
    VALID_SOURCE_TYPES,
    _parse_relative_time,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_querier(
    neo4j_rows: list[dict[str, Any]] | None = None,
    qdrant_points: list[Any] | None = None,
    neo4j_side_effect: Exception | None = None,
) -> tuple[TimelineQuerier, MagicMock, MagicMock]:
    """Build a TimelineQuerier with mocked Neo4j driver and Qdrant client.

    Returns (querier, mock_driver, mock_qdrant).
    """
    with (
        patch("worker.query.timeline.GraphDatabase") as mock_gdb_cls,
        patch("worker.query.timeline.QdrantClient") as mock_qdrant_cls,
    ):
        mock_driver = MagicMock()
        mock_gdb_cls.driver.return_value = mock_driver

        mock_session = MagicMock()
        mock_session.__enter__ = lambda s: s
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_driver.session.return_value = mock_session

        if neo4j_side_effect is not None:
            mock_session.execute_read.side_effect = neo4j_side_effect
        else:
            rows = neo4j_rows if neo4j_rows is not None else []
            mock_session.execute_read.side_effect = lambda fn: fn(
                MagicMock(**{"run.return_value": MagicMock(**{"data.return_value": rows})})
            )

        mock_qdrant = MagicMock()
        mock_qdrant_cls.return_value = mock_qdrant

        if qdrant_points is not None:
            mock_qdrant.scroll.return_value = (qdrant_points, None)
        else:
            mock_qdrant.scroll.return_value = ([], None)

        from worker.config import Neo4jConfig, QdrantConfig

        querier = TimelineQuerier(
            Neo4jConfig(uri="bolt://localhost:7687", user="neo4j", password="test"),
            QdrantConfig(host="localhost", port=6333, collection="fieldnotes"),
        )
        # Replace internal references with our mocks so tests can inspect calls.
        querier._driver = mock_driver
        querier._qdrant = mock_qdrant

        return querier, mock_driver, mock_qdrant


def _neo4j_row(
    source_id: str,
    label: str,
    title: str,
    ts: str,
    event_type: str = "modified",
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "label": label,
        "title": title,
        "ts": ts,
        "event_type": event_type,
    }


def _qdrant_point(source_id: str, text: str) -> MagicMock:
    point = MagicMock()
    point.payload = {"source_id": source_id, "text": text, "chunk_index": 0}
    return point


# ---------------------------------------------------------------------------
# _parse_relative_time
# ---------------------------------------------------------------------------


class TestParseRelativeTime:
    def test_hours_24(self) -> None:
        now = datetime.now(timezone.utc)
        result = _parse_relative_time("24h")
        assert abs((now - result - timedelta(hours=24)).total_seconds()) < 5

    def test_hours_2(self) -> None:
        now = datetime.now(timezone.utc)
        result = _parse_relative_time("2h")
        assert abs((now - result - timedelta(hours=2)).total_seconds()) < 5

    def test_days_7(self) -> None:
        now = datetime.now(timezone.utc)
        result = _parse_relative_time("7d")
        assert abs((now - result - timedelta(days=7)).total_seconds()) < 5

    def test_weeks_2(self) -> None:
        now = datetime.now(timezone.utc)
        result = _parse_relative_time("2w")
        assert abs((now - result - timedelta(weeks=2)).total_seconds()) < 5

    def test_iso_datetime_utc(self) -> None:
        result = _parse_relative_time("2026-03-01T00:00:00Z")
        assert result == datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc)

    def test_iso_date_only(self) -> None:
        result = _parse_relative_time("2026-03-01")
        assert result == datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc)

    def test_now_returns_current_time(self) -> None:
        before = datetime.now(timezone.utc)
        result = _parse_relative_time("now")
        after = datetime.now(timezone.utc)
        assert before <= result <= after

    def test_invalid_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            _parse_relative_time("not-a-date")

    def test_empty_string_returns_now(self) -> None:
        before = datetime.now(timezone.utc)
        result = _parse_relative_time("")
        after = datetime.now(timezone.utc)
        assert before <= result <= after


# ---------------------------------------------------------------------------
# TimelineQuerier unit tests
# ---------------------------------------------------------------------------


class TestTimelineDefault24h:
    """Default query returns entries sorted by timestamp DESC."""

    @patch("worker.query.timeline.GraphDatabase")
    @patch("worker.query.timeline.QdrantClient")
    def test_three_entries_returned(
        self, mock_qdrant_cls: MagicMock, mock_gdb_cls: MagicMock
    ) -> None:
        now = datetime.now(timezone.utc)
        rows = [
            _neo4j_row("file-1", "File", "Note A", (now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")),
            _neo4j_row("task-1", "Task", "Buy milk", (now - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")),
            _neo4j_row("email-1", "Email", "Hello", (now - timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ")),
        ]

        mock_driver = MagicMock()
        mock_gdb_cls.driver.return_value = mock_driver
        mock_session = MagicMock()
        mock_session.__enter__ = lambda s: s
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_driver.session.return_value = mock_session
        mock_session.execute_read.side_effect = lambda fn: fn(
            MagicMock(**{"run.return_value": MagicMock(**{"data.return_value": rows})})
        )

        mock_qdrant = MagicMock()
        mock_qdrant_cls.return_value = mock_qdrant
        mock_qdrant.scroll.return_value = ([], None)

        from worker.config import Neo4jConfig, QdrantConfig

        querier = TimelineQuerier(
            Neo4jConfig(uri="bolt://localhost:7687", user="neo4j", password="test"),
            QdrantConfig(host="localhost", port=6333, collection="fieldnotes"),
        )

        result = querier.query(since="24h", until="now")

        assert result.error is None
        assert len(result.entries) == 3
        assert result.entries[0].title == "Note A"
        assert result.entries[1].title == "Buy milk"
        assert result.entries[2].title == "Hello"
        assert result.since != ""
        assert result.until != ""

    @patch("worker.query.timeline.GraphDatabase")
    @patch("worker.query.timeline.QdrantClient")
    def test_source_types_mapped_correctly(
        self, mock_qdrant_cls: MagicMock, mock_gdb_cls: MagicMock
    ) -> None:
        now = datetime.now(timezone.utc)
        rows = [
            _neo4j_row("f-1", "File", "A", (now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")),
            _neo4j_row("t-1", "Task", "B", (now - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")),
            _neo4j_row("e-1", "Email", "C", (now - timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ")),
        ]

        mock_driver = MagicMock()
        mock_gdb_cls.driver.return_value = mock_driver
        mock_session = MagicMock()
        mock_session.__enter__ = lambda s: s
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_driver.session.return_value = mock_session
        mock_session.execute_read.side_effect = lambda fn: fn(
            MagicMock(**{"run.return_value": MagicMock(**{"data.return_value": rows})})
        )
        mock_qdrant = MagicMock()
        mock_qdrant_cls.return_value = mock_qdrant
        mock_qdrant.scroll.return_value = ([], None)

        from worker.config import Neo4jConfig, QdrantConfig

        querier = TimelineQuerier(
            Neo4jConfig(uri="bolt://localhost:7687", user="neo4j", password="test"),
            QdrantConfig(host="localhost", port=6333, collection="fieldnotes"),
        )
        result = querier.query()

        assert result.entries[0].source_type == "file"
        assert result.entries[1].source_type == "omnifocus"
        assert result.entries[2].source_type == "gmail"


class TestTimelineSourceFilter:
    """source_type filter routes to a narrower Cypher query."""

    @patch("worker.query.timeline.GraphDatabase")
    @patch("worker.query.timeline.QdrantClient")
    def test_omnifocus_filter_all_entries_have_correct_source_type(
        self, mock_qdrant_cls: MagicMock, mock_gdb_cls: MagicMock
    ) -> None:
        now = datetime.now(timezone.utc)
        rows = [
            _neo4j_row("t-1", "Task", "Task One", (now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")),
            _neo4j_row("t-2", "Task", "Task Two", (now - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")),
        ]

        mock_driver = MagicMock()
        mock_gdb_cls.driver.return_value = mock_driver
        mock_session = MagicMock()
        mock_session.__enter__ = lambda s: s
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_driver.session.return_value = mock_session
        mock_session.execute_read.side_effect = lambda fn: fn(
            MagicMock(**{"run.return_value": MagicMock(**{"data.return_value": rows})})
        )
        mock_qdrant = MagicMock()
        mock_qdrant_cls.return_value = mock_qdrant
        mock_qdrant.scroll.return_value = ([], None)

        from worker.config import Neo4jConfig, QdrantConfig

        querier = TimelineQuerier(
            Neo4jConfig(uri="bolt://localhost:7687", user="neo4j", password="test"),
            QdrantConfig(host="localhost", port=6333, collection="fieldnotes"),
        )
        result = querier.query(source_type="omnifocus")

        assert result.error is None
        assert len(result.entries) == 2
        for entry in result.entries:
            assert entry.source_type == "omnifocus"

    @patch("worker.query.timeline.GraphDatabase")
    @patch("worker.query.timeline.QdrantClient")
    def test_invalid_source_type_returns_error(
        self, mock_qdrant_cls: MagicMock, mock_gdb_cls: MagicMock
    ) -> None:
        mock_gdb_cls.driver.return_value = MagicMock()
        mock_qdrant_cls.return_value = MagicMock()

        from worker.config import Neo4jConfig, QdrantConfig

        querier = TimelineQuerier(
            Neo4jConfig(uri="bolt://localhost:7687", user="neo4j", password="test"),
            QdrantConfig(host="localhost", port=6333, collection="fieldnotes"),
        )
        result = querier.query(source_type="not-a-real-source")

        assert result.error is not None
        assert "not-a-real-source" in result.error


class TestTimelineEmptyRange:
    """Empty Neo4j results produce an empty entry list, not an error."""

    @patch("worker.query.timeline.GraphDatabase")
    @patch("worker.query.timeline.QdrantClient")
    def test_empty_entries_no_error(
        self, mock_qdrant_cls: MagicMock, mock_gdb_cls: MagicMock
    ) -> None:
        mock_driver = MagicMock()
        mock_gdb_cls.driver.return_value = mock_driver
        mock_session = MagicMock()
        mock_session.__enter__ = lambda s: s
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_driver.session.return_value = mock_session
        mock_session.execute_read.side_effect = lambda fn: fn(
            MagicMock(**{"run.return_value": MagicMock(**{"data.return_value": []})})
        )
        mock_qdrant = MagicMock()
        mock_qdrant_cls.return_value = mock_qdrant
        mock_qdrant.scroll.return_value = ([], None)

        from worker.config import Neo4jConfig, QdrantConfig

        querier = TimelineQuerier(
            Neo4jConfig(uri="bolt://localhost:7687", user="neo4j", password="test"),
            QdrantConfig(host="localhost", port=6333, collection="fieldnotes"),
        )
        result = querier.query(since="7d", until="now")

        assert result.error is None
        assert result.entries == []


class TestTimelineLimit:
    """Limit parameter caps returned entries."""

    @patch("worker.query.timeline.GraphDatabase")
    @patch("worker.query.timeline.QdrantClient")
    def test_limit_passed_to_neo4j_query(
        self, mock_qdrant_cls: MagicMock, mock_gdb_cls: MagicMock
    ) -> None:
        now = datetime.now(timezone.utc)
        # Return 100 rows from Neo4j — the Cypher LIMIT is passed as a param.
        rows = [
            _neo4j_row(f"f-{i}", "File", f"File {i}", (now - timedelta(minutes=i)).strftime("%Y-%m-%dT%H:%M:%SZ"))
            for i in range(100)
        ]

        captured: list[dict] = []

        def _execute_read(fn: Any) -> Any:
            mock_tx = MagicMock()

            def _run(cypher: str, **kwargs: Any) -> MagicMock:
                captured.append({"cypher": cypher, "params": kwargs})
                return MagicMock(**{"data.return_value": rows[:kwargs.get("limit", 100)]})

            mock_tx.run.side_effect = _run
            return fn(mock_tx)

        mock_driver = MagicMock()
        mock_gdb_cls.driver.return_value = mock_driver
        mock_session = MagicMock()
        mock_session.__enter__ = lambda s: s
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_driver.session.return_value = mock_session
        mock_session.execute_read.side_effect = _execute_read

        mock_qdrant = MagicMock()
        mock_qdrant_cls.return_value = mock_qdrant
        mock_qdrant.scroll.return_value = ([], None)

        from worker.config import Neo4jConfig, QdrantConfig

        querier = TimelineQuerier(
            Neo4jConfig(uri="bolt://localhost:7687", user="neo4j", password="test"),
            QdrantConfig(host="localhost", port=6333, collection="fieldnotes"),
        )
        result = querier.query(limit=10)

        assert result.error is None
        assert len(result.entries) == 10
        assert captured[0]["params"]["limit"] == 10


class TestTimelineSnippetFromQdrant:
    """Snippets from Qdrant are attached to entries and truncated."""

    @patch("worker.query.timeline.GraphDatabase")
    @patch("worker.query.timeline.QdrantClient")
    def test_snippet_populated_and_truncated(
        self, mock_qdrant_cls: MagicMock, mock_gdb_cls: MagicMock
    ) -> None:
        now = datetime.now(timezone.utc)
        rows = [
            _neo4j_row("file-a", "File", "Doc A", (now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")),
            _neo4j_row("file-b", "File", "Doc B", (now - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")),
        ]
        long_text_a = "A" * 300
        long_text_b = "B" * 50
        qdrant_results = [
            _qdrant_point("file-a", long_text_a),
            _qdrant_point("file-b", long_text_b),
        ]

        mock_driver = MagicMock()
        mock_gdb_cls.driver.return_value = mock_driver
        mock_session = MagicMock()
        mock_session.__enter__ = lambda s: s
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_driver.session.return_value = mock_session
        mock_session.execute_read.side_effect = lambda fn: fn(
            MagicMock(**{"run.return_value": MagicMock(**{"data.return_value": rows})})
        )
        mock_qdrant = MagicMock()
        mock_qdrant_cls.return_value = mock_qdrant
        mock_qdrant.scroll.return_value = (qdrant_results, None)

        from worker.config import Neo4jConfig, QdrantConfig

        querier = TimelineQuerier(
            Neo4jConfig(uri="bolt://localhost:7687", user="neo4j", password="test"),
            QdrantConfig(host="localhost", port=6333, collection="fieldnotes"),
        )
        result = querier.query()

        entry_a = next(e for e in result.entries if e.source_id == "file-a")
        entry_b = next(e for e in result.entries if e.source_id == "file-b")

        assert len(entry_a.snippet) == 200  # truncated to _SNIPPET_MAX
        assert entry_a.snippet == "A" * 200
        assert entry_b.snippet == "B" * 50  # short text preserved as-is


class TestTimelineCompletedTaskEventType:
    """Tasks with status='completed' produce event_type='completed'."""

    @patch("worker.query.timeline.GraphDatabase")
    @patch("worker.query.timeline.QdrantClient")
    def test_completed_task_event_type(
        self, mock_qdrant_cls: MagicMock, mock_gdb_cls: MagicMock
    ) -> None:
        now = datetime.now(timezone.utc)
        rows = [
            _neo4j_row("t-1", "Task", "Finished task", (now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"), event_type="completed"),
        ]

        mock_driver = MagicMock()
        mock_gdb_cls.driver.return_value = mock_driver
        mock_session = MagicMock()
        mock_session.__enter__ = lambda s: s
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_driver.session.return_value = mock_session
        mock_session.execute_read.side_effect = lambda fn: fn(
            MagicMock(**{"run.return_value": MagicMock(**{"data.return_value": rows})})
        )
        mock_qdrant = MagicMock()
        mock_qdrant_cls.return_value = mock_qdrant
        mock_qdrant.scroll.return_value = ([], None)

        from worker.config import Neo4jConfig, QdrantConfig

        querier = TimelineQuerier(
            Neo4jConfig(uri="bolt://localhost:7687", user="neo4j", password="test"),
            QdrantConfig(host="localhost", port=6333, collection="fieldnotes"),
        )
        result = querier.query()

        assert len(result.entries) == 1
        assert result.entries[0].event_type == "completed"


class TestTimelineNeo4jError:
    """Neo4j failures produce an error in the result, not an exception."""

    @patch("worker.query.timeline.GraphDatabase")
    @patch("worker.query.timeline.QdrantClient")
    def test_neo4j_exception_returns_error_result(
        self, mock_qdrant_cls: MagicMock, mock_gdb_cls: MagicMock
    ) -> None:
        mock_driver = MagicMock()
        mock_gdb_cls.driver.return_value = mock_driver
        mock_session = MagicMock()
        mock_session.__enter__ = lambda s: s
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_driver.session.return_value = mock_session
        mock_session.execute_read.side_effect = RuntimeError("Neo4j is down")

        mock_qdrant = MagicMock()
        mock_qdrant_cls.return_value = mock_qdrant

        from worker.config import Neo4jConfig, QdrantConfig

        querier = TimelineQuerier(
            Neo4jConfig(uri="bolt://localhost:7687", user="neo4j", password="test"),
            QdrantConfig(host="localhost", port=6333, collection="fieldnotes"),
        )
        result = querier.query()

        assert result.error is not None
        assert "Neo4j" in result.error


class TestTimelineQdrantFailureNonFatal:
    """Qdrant snippet failures don't fail the whole query."""

    @patch("worker.query.timeline.GraphDatabase")
    @patch("worker.query.timeline.QdrantClient")
    def test_qdrant_failure_returns_entries_without_snippets(
        self, mock_qdrant_cls: MagicMock, mock_gdb_cls: MagicMock
    ) -> None:
        now = datetime.now(timezone.utc)
        rows = [
            _neo4j_row("file-1", "File", "Note", (now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")),
        ]

        mock_driver = MagicMock()
        mock_gdb_cls.driver.return_value = mock_driver
        mock_session = MagicMock()
        mock_session.__enter__ = lambda s: s
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_driver.session.return_value = mock_session
        mock_session.execute_read.side_effect = lambda fn: fn(
            MagicMock(**{"run.return_value": MagicMock(**{"data.return_value": rows})})
        )
        mock_qdrant = MagicMock()
        mock_qdrant_cls.return_value = mock_qdrant
        mock_qdrant.scroll.side_effect = RuntimeError("Qdrant is down")

        from worker.config import Neo4jConfig, QdrantConfig

        querier = TimelineQuerier(
            Neo4jConfig(uri="bolt://localhost:7687", user="neo4j", password="test"),
            QdrantConfig(host="localhost", port=6333, collection="fieldnotes"),
        )
        result = querier.query()

        assert result.error is None
        assert len(result.entries) == 1
        assert result.entries[0].snippet == ""


# ---------------------------------------------------------------------------
# CLI output tests
# ---------------------------------------------------------------------------


class TestCliTimelineHumanOutput:
    @patch("worker.cli.timeline.TimelineQuerier")
    @patch("worker.cli.timeline.load_config")
    def test_human_output_contains_label_and_title(
        self, mock_load_config: MagicMock, mock_querier_cls: MagicMock, capsys: pytest.CaptureFixture
    ) -> None:
        from worker.cli.timeline import run_timeline

        mock_cfg = MagicMock()
        mock_load_config.return_value = mock_cfg

        now = datetime.now(timezone.utc)
        mock_querier = MagicMock()
        mock_querier_cls.return_value.__enter__ = lambda s: mock_querier
        mock_querier_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_querier.query.return_value = TimelineResult(
            entries=[
                TimelineEntry(
                    source_type="file",
                    source_id="f-1",
                    label="File",
                    title="My Important Note",
                    timestamp=(now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    event_type="modified",
                ),
                TimelineEntry(
                    source_type="omnifocus",
                    source_id="t-1",
                    label="Task",
                    title="Buy groceries",
                    timestamp=(now - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    event_type="completed",
                ),
            ],
            since=(now - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            until=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

        rc = run_timeline(since="24h", until="now")
        captured = capsys.readouterr()

        assert rc == 0
        assert "File" in captured.out
        assert "My Important Note" in captured.out
        assert "Task" in captured.out
        assert "Buy groceries" in captured.out

    @patch("worker.cli.timeline.TimelineQuerier")
    @patch("worker.cli.timeline.load_config")
    def test_human_output_has_date_header_for_multiday(
        self, mock_load_config: MagicMock, mock_querier_cls: MagicMock, capsys: pytest.CaptureFixture
    ) -> None:
        from worker.cli.timeline import run_timeline

        mock_cfg = MagicMock()
        mock_load_config.return_value = mock_cfg

        mock_querier = MagicMock()
        mock_querier_cls.return_value.__enter__ = lambda s: mock_querier
        mock_querier_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_querier.query.return_value = TimelineResult(
            entries=[
                TimelineEntry(
                    source_type="file",
                    source_id="f-1",
                    label="File",
                    title="Today note",
                    timestamp="2026-03-19T10:00:00Z",
                    event_type="modified",
                ),
                TimelineEntry(
                    source_type="file",
                    source_id="f-2",
                    label="File",
                    title="Yesterday note",
                    timestamp="2026-03-18T10:00:00Z",
                    event_type="modified",
                ),
            ],
            since="2026-03-18T00:00:00Z",
            until="2026-03-19T23:59:59Z",
        )

        rc = run_timeline(since="2d", until="now")
        captured = capsys.readouterr()

        assert rc == 0
        # Multi-day: day headers should appear
        assert "2026-03-19" in captured.out
        assert "2026-03-18" in captured.out


class TestCliTimelineJsonOutput:
    @patch("worker.cli.timeline.TimelineQuerier")
    @patch("worker.cli.timeline.load_config")
    def test_json_output_is_valid_and_has_required_fields(
        self, mock_load_config: MagicMock, mock_querier_cls: MagicMock, capsys: pytest.CaptureFixture
    ) -> None:
        from worker.cli.timeline import run_timeline

        mock_cfg = MagicMock()
        mock_load_config.return_value = mock_cfg

        now = datetime.now(timezone.utc)
        mock_querier = MagicMock()
        mock_querier_cls.return_value.__enter__ = lambda s: mock_querier
        mock_querier_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_querier.query.return_value = TimelineResult(
            entries=[
                TimelineEntry(
                    source_type="file",
                    source_id="f-1",
                    label="File",
                    title="A note",
                    timestamp=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    event_type="modified",
                    snippet="Short snippet",
                ),
            ],
            since=(now - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            until=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

        rc = run_timeline(since="24h", until="now", json_output=True)
        captured = capsys.readouterr()

        assert rc == 0
        data = json.loads(captured.out)
        assert "entries" in data
        assert "since" in data
        assert "until" in data
        assert len(data["entries"]) == 1
        entry = data["entries"][0]
        assert entry["source_type"] == "file"
        assert entry["title"] == "A note"
        assert entry["snippet"] == "Short snippet"

    @patch("worker.cli.timeline.TimelineQuerier")
    @patch("worker.cli.timeline.load_config")
    def test_error_result_returns_exit_code_1(
        self, mock_load_config: MagicMock, mock_querier_cls: MagicMock, capsys: pytest.CaptureFixture
    ) -> None:
        from worker.cli.timeline import run_timeline

        mock_cfg = MagicMock()
        mock_load_config.return_value = mock_cfg

        mock_querier = MagicMock()
        mock_querier_cls.return_value.__enter__ = lambda s: mock_querier
        mock_querier_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_querier.query.return_value = TimelineResult(
            entries=[],
            error="Neo4j error: connection refused",
        )

        rc = run_timeline(since="24h", until="now")
        captured = capsys.readouterr()

        assert rc == 1
        assert "connection refused" in captured.err


# ---------------------------------------------------------------------------
# MCP tool registration
# ---------------------------------------------------------------------------


class TestMcpTimelineTool:
    def test_timeline_tool_registered(self) -> None:
        from worker.mcp_server import TOOLS

        names = [t.name for t in TOOLS]
        assert "timeline" in names

    def test_timeline_schema_has_required_properties(self) -> None:
        from worker.mcp_server import TOOLS

        tl = next(t for t in TOOLS if t.name == "timeline")
        props = tl.inputSchema["properties"]
        assert "since" in props
        assert "until" in props
        assert "source_type" in props
        assert "limit" in props

    def test_timeline_source_type_enum_matches_valid_source_types(self) -> None:
        from worker.mcp_server import TOOLS

        tl = next(t for t in TOOLS if t.name == "timeline")
        schema_enum = set(tl.inputSchema["properties"]["source_type"]["enum"])
        assert schema_enum == VALID_SOURCE_TYPES
