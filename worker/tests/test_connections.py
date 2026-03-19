"""Unit tests for the connection suggestions module.

All Neo4j and Qdrant interactions are mocked — no running services required.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, call, patch

import pytest

from worker.config import Config, Neo4jConfig, QdrantConfig
from worker.query.connections import (
    ConnectionQuerier,
    ConnectionResult,
    SuggestedConnection,
    _DEFAULT_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cfg() -> tuple[Neo4jConfig, QdrantConfig]:
    neo4j = Neo4jConfig(uri="bolt://localhost:7687", user="neo4j", password="test")
    qdrant = QdrantConfig(host="localhost", port=6333, collection="fieldnotes")
    return neo4j, qdrant


def _make_querier(mock_gdb: MagicMock, mock_qdrant_cls: MagicMock) -> ConnectionQuerier:
    neo4j_cfg, qdrant_cfg = _make_cfg()
    return ConnectionQuerier(neo4j_cfg, qdrant_cfg)


def _qdrant_point(source_id: str, source_type: str, vector: list[float] | None = None) -> MagicMock:
    """Build a fake Qdrant scroll/search hit."""
    point = MagicMock()
    point.payload = {"source_id": source_id, "source_type": source_type}
    point.vector = vector or [0.1, 0.2, 0.3]
    point.score = 0.90
    return point


def _qdrant_hit(source_id: str, score: float, source_type: str = "file") -> MagicMock:
    """Build a fake Qdrant search hit (with .score)."""
    hit = MagicMock()
    hit.payload = {"source_id": source_id, "source_type": source_type}
    hit.score = score
    return hit


def _neo4j_session_returning(records: list[dict]) -> MagicMock:
    """Return a mock Neo4j driver whose session.run().data() returns *records*."""
    session = MagicMock()
    run_result = MagicMock()
    run_result.data.return_value = records
    session.run.return_value = run_result
    session.__enter__ = lambda s: session
    session.__exit__ = MagicMock(return_value=False)
    return session


# ---------------------------------------------------------------------------
# ConnectionQuerier unit tests
# ---------------------------------------------------------------------------


class TestConnectionsFindsUnlinkedSimilarDocs:
    """Mock Qdrant + Neo4j to verify basic unlinked-pair detection."""

    @patch("worker.query.connections.QdrantClient")
    @patch("worker.query.connections.GraphDatabase")
    def test_connections_finds_unlinked_similar_docs(
        self, mock_gdb: MagicMock, mock_qdrant_cls: MagicMock
    ) -> None:
        """2 of 3 similar pairs are unlinked → 2 suggestions returned."""
        querier = _make_querier(mock_gdb, mock_qdrant_cls)

        seed_vector = [1.0, 0.0, 0.0]
        seed_point = _qdrant_point("file://seed", "file", seed_vector)

        # Qdrant: scroll returns the seed; search returns 3 similar docs
        mock_qdrant = mock_qdrant_cls.return_value
        mock_qdrant.scroll.return_value = ([seed_point], None)
        mock_qdrant.search.return_value = [
            _qdrant_hit("file://linked", 0.95),
            _qdrant_hit("file://unlinked1", 0.92),
            _qdrant_hit("file://unlinked2", 0.88),
        ]

        # Neo4j: "linked" pair has edge_count=1; others have 0.
        # Keys must match canonical (alphabetical) order used by ConnectionQuerier.
        # "file://linked" < "file://seed" so a="file://linked", b="file://seed"
        session = _neo4j_session_returning([
            {"a": "file://linked", "b": "file://seed", "edge_count": 1},
            {"a": "file://seed", "b": "file://unlinked1", "edge_count": 0},
            {"a": "file://seed", "b": "file://unlinked2", "edge_count": 0},
        ])
        # second session call is for node info
        node_info_session = _neo4j_session_returning([
            {"sid": "file://seed", "labels": ["File"], "title": "Seed", "source_type": "file"},
            {"sid": "file://unlinked1", "labels": ["File"], "title": "Doc1", "source_type": "file"},
            {"sid": "file://unlinked2", "labels": ["File"], "title": "Doc2", "source_type": "file"},
        ])
        mock_gdb.driver.return_value.session.side_effect = [session, node_info_session]

        result = querier.suggest()

        assert result.error is None
        assert len(result.suggestions) == 2
        sids = {(s.source_a, s.source_b) for s in result.suggestions}
        # The linked pair must NOT appear
        for pair in sids:
            assert "file://linked" not in pair


class TestConnectionsThresholdFiltering:
    """Scores below threshold must be excluded."""

    @patch("worker.query.connections.QdrantClient")
    @patch("worker.query.connections.GraphDatabase")
    def test_connections_threshold_filtering(
        self, mock_gdb: MagicMock, mock_qdrant_cls: MagicMock
    ) -> None:
        querier = _make_querier(mock_gdb, mock_qdrant_cls)

        seed_point = _qdrant_point("file://seed", "file")
        mock_qdrant = mock_qdrant_cls.return_value
        mock_qdrant.scroll.return_value = ([seed_point], None)
        # Qdrant honours score_threshold itself, so we simulate that here:
        # only scores >= 0.80 come back
        mock_qdrant.search.return_value = [
            _qdrant_hit("file://a", 0.95),
            _qdrant_hit("file://b", 0.85),
            # 0.75 and 0.65 would be filtered by Qdrant's score_threshold
        ]

        edge_session = _neo4j_session_returning([
            {"a": "file://seed", "b": "file://a", "edge_count": 0},
            {"a": "file://seed", "b": "file://b", "edge_count": 0},
        ])
        node_session = _neo4j_session_returning([
            {"sid": "file://seed", "labels": ["File"], "title": "Seed", "source_type": "file"},
            {"sid": "file://a", "labels": ["File"], "title": "A", "source_type": "file"},
            {"sid": "file://b", "labels": ["File"], "title": "B", "source_type": "file"},
        ])
        mock_gdb.driver.return_value.session.side_effect = [edge_session, node_session]

        result = querier.suggest(threshold=0.80)

        assert result.error is None
        # Qdrant call must have received the threshold
        mock_qdrant.search.assert_called_once()
        _, kwargs = mock_qdrant.search.call_args
        assert kwargs["score_threshold"] == 0.80
        # All returned suggestions have similarity >= 0.80
        for s in result.suggestions:
            assert s.similarity >= 0.80


class TestConnectionsCrossSourceFilter:
    """cross_source=True keeps only inter-source-type pairs."""

    @patch("worker.query.connections.QdrantClient")
    @patch("worker.query.connections.GraphDatabase")
    def test_connections_cross_source_filter(
        self, mock_gdb: MagicMock, mock_qdrant_cls: MagicMock
    ) -> None:
        querier = _make_querier(mock_gdb, mock_qdrant_cls)

        # Seeds: one File and one Task seed
        seed_file = _qdrant_point("file://note1", "file", [1.0, 0.0, 0.0])
        mock_qdrant = mock_qdrant_cls.return_value
        mock_qdrant.scroll.return_value = ([seed_file], None)
        mock_qdrant.search.return_value = [
            _qdrant_hit("task://t1", 0.93, "omnifocus"),   # cross-source
            _qdrant_hit("file://note2", 0.90, "file"),     # same source
            _qdrant_hit("email://e1", 0.87, "gmail"),      # cross-source
        ]

        edge_session = _neo4j_session_returning([
            {"a": "file://note1", "b": "task://t1", "edge_count": 0},
            {"a": "file://note1", "b": "file://note2", "edge_count": 0},
            {"a": "email://e1", "b": "file://note1", "edge_count": 0},
        ])
        node_session = _neo4j_session_returning([
            {"sid": "file://note1", "labels": ["File"], "title": "Note1", "source_type": "file"},
            {"sid": "task://t1", "labels": ["Task"], "title": "Task1", "source_type": "omnifocus"},
            {"sid": "file://note2", "labels": ["File"], "title": "Note2", "source_type": "file"},
            {"sid": "email://e1", "labels": ["Email"], "title": "Email1", "source_type": "gmail"},
        ])
        mock_gdb.driver.return_value.session.side_effect = [edge_session, node_session]

        result = querier.suggest(cross_source=True)

        assert result.error is None
        for s in result.suggestions:
            assert s.source_type_a != s.source_type_b, (
                f"Same-source pair survived filter: {s.source_type_a}"
            )
        # The file↔file pair must NOT appear
        source_ids = {sid for s in result.suggestions for sid in (s.source_a, s.source_b)}
        assert "file://note2" not in source_ids or not any(
            s.source_a == "file://note1" and s.source_b == "file://note2"
            or s.source_a == "file://note2" and s.source_b == "file://note1"
            for s in result.suggestions
        )


class TestConnectionsSourceIdSeed:
    """When source_id is given, only that document is used as seed."""

    @patch("worker.query.connections.QdrantClient")
    @patch("worker.query.connections.GraphDatabase")
    def test_connections_source_id_seed(
        self, mock_gdb: MagicMock, mock_qdrant_cls: MagicMock
    ) -> None:
        querier = _make_querier(mock_gdb, mock_qdrant_cls)

        seed_vector = [0.5, 0.5, 0.0]
        seed_point = _qdrant_point("obsidian://note1", "obsidian", seed_vector)
        mock_qdrant = mock_qdrant_cls.return_value
        mock_qdrant.scroll.return_value = ([seed_point], None)
        mock_qdrant.search.return_value = [
            _qdrant_hit("file://related", 0.91),
        ]

        edge_session = _neo4j_session_returning([
            {"a": "file://related", "b": "obsidian://note1", "edge_count": 0},
        ])
        node_session = _neo4j_session_returning([
            {"sid": "obsidian://note1", "labels": ["Note"], "title": "Note1", "source_type": "obsidian"},
            {"sid": "file://related", "labels": ["File"], "title": "Related", "source_type": "file"},
        ])
        mock_gdb.driver.return_value.session.side_effect = [edge_session, node_session]

        result = querier.suggest(source_id="obsidian://note1")

        # Qdrant scroll must have been called with a filter on that source_id
        mock_qdrant.scroll.assert_called_once()
        _, kwargs = mock_qdrant.scroll.call_args
        assert kwargs.get("with_vectors") is True

        # Qdrant search uses the seed's own vector
        mock_qdrant.search.assert_called_once()
        search_kwargs = mock_qdrant.search.call_args[1]
        assert search_kwargs["query_vector"] == seed_vector

        assert len(result.suggestions) == 1
        s = result.suggestions[0]
        assert "obsidian://note1" in (s.source_a, s.source_b)


class TestConnectionsSourceTypeFilter:
    """source_type filter restricts which seeds are used."""

    @patch("worker.query.connections.QdrantClient")
    @patch("worker.query.connections.GraphDatabase")
    def test_connections_source_type_filter(
        self, mock_gdb: MagicMock, mock_qdrant_cls: MagicMock
    ) -> None:
        querier = _make_querier(mock_gdb, mock_qdrant_cls)

        task_point = _qdrant_point("task://t1", "omnifocus")
        mock_qdrant = mock_qdrant_cls.return_value
        mock_qdrant.scroll.return_value = ([task_point], None)
        mock_qdrant.search.return_value = []

        mock_gdb.driver.return_value.session.side_effect = []

        querier.suggest(source_type="omnifocus")

        # scroll must include a filter on source_type
        mock_qdrant.scroll.assert_called_once()
        _, kwargs = mock_qdrant.scroll.call_args
        scroll_filter = kwargs.get("scroll_filter")
        assert scroll_filter is not None


class TestConnectionsLimit:
    """limit parameter caps the returned suggestions, sorted DESC by similarity."""

    @patch("worker.query.connections.QdrantClient")
    @patch("worker.query.connections.GraphDatabase")
    def test_connections_limit(
        self, mock_gdb: MagicMock, mock_qdrant_cls: MagicMock
    ) -> None:
        querier = _make_querier(mock_gdb, mock_qdrant_cls)

        seed = _qdrant_point("file://seed", "file")
        mock_qdrant = mock_qdrant_cls.return_value
        mock_qdrant.scroll.return_value = ([seed], None)

        # 8 different hits (all unique source_ids)
        hits = [_qdrant_hit(f"file://doc{i}", 0.90 - i * 0.01) for i in range(8)]
        mock_qdrant.search.return_value = hits

        edge_records = [
            {"a": "file://seed", "b": f"file://doc{i}", "edge_count": 0}
            for i in range(8)
        ]
        node_records = [
            {"sid": "file://seed", "labels": ["File"], "title": "Seed", "source_type": "file"}
        ] + [
            {"sid": f"file://doc{i}", "labels": ["File"], "title": f"Doc{i}", "source_type": "file"}
            for i in range(8)
        ]
        edge_session = _neo4j_session_returning(edge_records)
        node_session = _neo4j_session_returning(node_records)
        mock_gdb.driver.return_value.session.side_effect = [edge_session, node_session]

        result = querier.suggest(limit=5)

        assert len(result.suggestions) == 5
        scores = [s.similarity for s in result.suggestions]
        assert scores == sorted(scores, reverse=True)


class TestConnectionsEmptyGraph:
    """When Qdrant returns no similar docs, result is empty, no error."""

    @patch("worker.query.connections.QdrantClient")
    @patch("worker.query.connections.GraphDatabase")
    def test_connections_empty_graph(
        self, mock_gdb: MagicMock, mock_qdrant_cls: MagicMock
    ) -> None:
        querier = _make_querier(mock_gdb, mock_qdrant_cls)

        seed = _qdrant_point("file://seed", "file")
        mock_qdrant = mock_qdrant_cls.return_value
        mock_qdrant.scroll.return_value = ([seed], None)
        mock_qdrant.search.return_value = []

        # No Neo4j calls expected
        mock_gdb.driver.return_value.session.side_effect = []

        result = querier.suggest()

        assert result.error is None
        assert result.suggestions == []


class TestConnectionsDeduplicatesSameDocChunks:
    """Multiple chunks from the same source_id must not be suggested as connections."""

    @patch("worker.query.connections.QdrantClient")
    @patch("worker.query.connections.GraphDatabase")
    def test_connections_deduplicates_same_doc_chunks(
        self, mock_gdb: MagicMock, mock_qdrant_cls: MagicMock
    ) -> None:
        querier = _make_querier(mock_gdb, mock_qdrant_cls)

        seed = _qdrant_point("file://seed", "file")
        mock_qdrant = mock_qdrant_cls.return_value
        # scroll: seed appears once; but search returns two hits with same source_id
        # (simulating two chunks of the same document)
        mock_qdrant.scroll.return_value = ([seed], None)

        chunk1 = _qdrant_hit("file://other", 0.94)
        chunk2 = _qdrant_hit("file://other", 0.91)  # same source_id, different chunk
        mock_qdrant.search.return_value = [chunk1, chunk2]

        edge_session = _neo4j_session_returning([
            {"a": "file://seed", "b": "file://other", "edge_count": 0},
        ])
        node_session = _neo4j_session_returning([
            {"sid": "file://seed", "labels": ["File"], "title": "Seed", "source_type": "file"},
            {"sid": "file://other", "labels": ["File"], "title": "Other", "source_type": "file"},
        ])
        mock_gdb.driver.return_value.session.side_effect = [edge_session, node_session]

        result = querier.suggest()

        # Only one suggestion for the pair, not two
        assert len(result.suggestions) == 1


class TestConnectionsBatchEdgeCheck:
    """Neo4j must be called exactly once (UNWIND), not per pair."""

    @patch("worker.query.connections.QdrantClient")
    @patch("worker.query.connections.GraphDatabase")
    def test_connections_batch_edge_check(
        self, mock_gdb: MagicMock, mock_qdrant_cls: MagicMock
    ) -> None:
        querier = _make_querier(mock_gdb, mock_qdrant_cls)

        seed = _qdrant_point("file://seed", "file")
        mock_qdrant = mock_qdrant_cls.return_value
        mock_qdrant.scroll.return_value = ([seed], None)

        # 10 candidates
        hits = [_qdrant_hit(f"file://doc{i}", 0.95 - i * 0.01) for i in range(10)]
        mock_qdrant.search.return_value = hits

        edge_records = [
            {"a": "file://seed", "b": f"file://doc{i}", "edge_count": 0}
            for i in range(10)
        ]
        node_records = [
            {"sid": "file://seed", "labels": ["File"], "title": "Seed", "source_type": "file"}
        ] + [
            {"sid": f"file://doc{i}", "labels": ["File"], "title": f"Doc{i}", "source_type": "file"}
            for i in range(10)
        ]
        edge_session = _neo4j_session_returning(edge_records)
        node_session = _neo4j_session_returning(node_records)

        driver = mock_gdb.driver.return_value
        driver.session.side_effect = [edge_session, node_session]

        querier.suggest()

        # Each session.run is called exactly once (UNWIND batch)
        assert edge_session.run.call_count == 1
        # The UNWIND query passes a 'pairs' parameter (list of dicts)
        run_call = edge_session.run.call_args
        assert "pairs" in run_call[1] or (len(run_call[0]) > 1)


class TestConnectionsEnrichment:
    """Node metadata (label, title, source_type) must be set on SuggestedConnection."""

    @patch("worker.query.connections.QdrantClient")
    @patch("worker.query.connections.GraphDatabase")
    def test_connections_enrichment(
        self, mock_gdb: MagicMock, mock_qdrant_cls: MagicMock
    ) -> None:
        querier = _make_querier(mock_gdb, mock_qdrant_cls)

        seed = _qdrant_point("obsidian://n1", "obsidian")
        mock_qdrant = mock_qdrant_cls.return_value
        mock_qdrant.scroll.return_value = ([seed], None)
        mock_qdrant.search.return_value = [_qdrant_hit("task://t1", 0.93, "omnifocus")]

        edge_session = _neo4j_session_returning([
            {"a": "obsidian://n1", "b": "task://t1", "edge_count": 0},
        ])
        node_session = _neo4j_session_returning([
            {"sid": "obsidian://n1", "labels": ["Note"], "title": "My Note", "source_type": "obsidian"},
            {"sid": "task://t1", "labels": ["Task"], "title": "My Task", "source_type": "omnifocus"},
        ])
        mock_gdb.driver.return_value.session.side_effect = [edge_session, node_session]

        result = querier.suggest()

        assert len(result.suggestions) == 1
        s = result.suggestions[0]
        # One side is Note/obsidian, the other is Task/omnifocus
        labels = {s.label_a, s.label_b}
        titles = {s.title_a, s.title_b}
        types = {s.source_type_a, s.source_type_b}
        assert "Note" in labels and "Task" in labels
        assert "My Note" in titles and "My Task" in titles
        assert "obsidian" in types and "omnifocus" in types


# ---------------------------------------------------------------------------
# CLI output tests
# ---------------------------------------------------------------------------


class TestCLIConnectionsHumanOutput:
    """run_connections() human-readable output contains expected fields."""

    @patch("worker.cli.connections.ConnectionQuerier")
    @patch("worker.cli.connections.load_config")
    def test_cli_connections_human_output(
        self,
        mock_load_config: MagicMock,
        mock_querier_cls: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from worker.cli.connections import run_connections

        mock_cfg = MagicMock()
        mock_load_config.return_value = mock_cfg

        mock_querier = MagicMock()
        mock_querier_cls.return_value.__enter__ = lambda s: mock_querier
        mock_querier_cls.return_value.__exit__ = MagicMock(return_value=False)

        mock_querier.suggest.return_value = ConnectionResult(
            suggestions=[
                SuggestedConnection(
                    source_a="file://a.md",
                    source_b="task://t1",
                    label_a="File",
                    label_b="Task",
                    title_a="My Note",
                    title_b="My Task",
                    source_type_a="file",
                    source_type_b="omnifocus",
                    similarity=0.9312,
                )
            ],
            checked=5,
        )

        rc = run_connections()

        assert rc == 0
        out = capsys.readouterr().out
        assert "0.93" in out
        assert "My Note" in out
        assert "My Task" in out
        assert "File" in out
        assert "Task" in out
        # Arrow symbol present
        assert "↔" in out


class TestCLIConnectionsJSONOutput:
    """run_connections() with json_output=True emits valid JSON."""

    @patch("worker.cli.connections.ConnectionQuerier")
    @patch("worker.cli.connections.load_config")
    def test_cli_connections_json_output(
        self,
        mock_load_config: MagicMock,
        mock_querier_cls: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from worker.cli.connections import run_connections

        mock_cfg = MagicMock()
        mock_load_config.return_value = mock_cfg

        mock_querier = MagicMock()
        mock_querier_cls.return_value.__enter__ = lambda s: mock_querier
        mock_querier_cls.return_value.__exit__ = MagicMock(return_value=False)

        mock_querier.suggest.return_value = ConnectionResult(
            suggestions=[
                SuggestedConnection(
                    source_a="file://a",
                    source_b="file://b",
                    label_a="File",
                    label_b="File",
                    title_a="A",
                    title_b="B",
                    source_type_a="file",
                    source_type_b="file",
                    similarity=0.88,
                )
            ],
            checked=3,
        )

        rc = run_connections(json_output=True)

        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "suggestions" in data
        assert "checked" in data
        assert data["checked"] == 3
        assert len(data["suggestions"]) == 1
        s = data["suggestions"][0]
        assert s["similarity"] == 0.88


# ---------------------------------------------------------------------------
# MCP tool registration test
# ---------------------------------------------------------------------------


class TestMCPSuggestConnectionsToolRegistered:
    """suggest_connections must appear in TOOLS with the expected schema fields."""

    def test_mcp_suggest_connections_tool_registered(self) -> None:
        from worker.mcp_server import TOOLS

        names = [t.name for t in TOOLS]
        assert "suggest_connections" in names

        tool = next(t for t in TOOLS if t.name == "suggest_connections")
        props = tool.inputSchema.get("properties", {})
        assert "source_id" in props
        assert "source_type" in props
        assert "threshold" in props
        assert "limit" in props
        assert "cross_source" in props
