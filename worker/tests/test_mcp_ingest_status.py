"""Tests for the MCP ingest_status tool."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from worker.config import Config, Neo4jConfig, QdrantConfig
from worker.mcp_server import FieldnotesServer


def _make_server() -> FieldnotesServer:
    cfg = Config(
        neo4j=Neo4jConfig(uri="bolt://localhost:7687", user="neo4j", password="test"),
        qdrant=QdrantConfig(host="localhost", port=6333, collection="fieldnotes"),
    )
    return FieldnotesServer(cfg)


def _mock_neo4j_session(label_counts: dict[str, int] | None = None,
                         entity_rows: list[dict] | None = None,
                         topic_rows: list[dict] | None = None) -> MagicMock:
    """Build a mock Neo4j session that returns prescribed counts."""
    if label_counts is None:
        label_counts = {}
    if entity_rows is None:
        entity_rows = []
    if topic_rows is None:
        topic_rows = []

    session = MagicMock()

    def _run_query(query, **kwargs):
        result = MagicMock()
        # Label count queries
        if "MATCH (n:" in query:
            for label in ("File", "Email", "Commit", "Entity",
                          "Topic", "Chunk", "Image", "Repository"):
                if f"`{label}`" in query:
                    row = MagicMock()
                    row.__getitem__ = lambda self, key, lb=label: (
                        label_counts.get(lb, 0) if key == "cnt" else None
                    )
                    result.single.return_value = row
                    return result
        # Entity type query
        if "e.type AS type" in query:
            result.data.return_value = entity_rows
            return result
        # Topic source query
        if "t.source AS source" in query:
            result.data.return_value = topic_rows
            return result
        result.single.return_value = MagicMock(
            __getitem__=lambda self, key: 0
        )
        return result

    session.run = _run_query
    return session


def _setup_server_with_mocks(
    mock_session: MagicMock | None = None,
    qdrant_points: int = 100,
    neo4j_error: Exception | None = None,
    qdrant_error: Exception | None = None,
) -> FieldnotesServer:
    """Create a server with mocked internal queriers."""
    server = _make_server()

    # Mock graph querier
    if neo4j_error:
        server._graph_querier = None  # Will trigger RuntimeError in handler
    else:
        mock_driver = MagicMock()
        if mock_session is not None:
            mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
            mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        mock_graph = MagicMock()
        mock_graph._driver = mock_driver

        mock_gq = MagicMock()
        mock_gq._graph = mock_graph
        server._graph_querier = mock_gq

    # Mock vector querier
    if qdrant_error:
        server._vector_querier = None  # Will trigger RuntimeError in handler
    else:
        mock_collection = MagicMock()
        mock_collection.points_count = qdrant_points
        mock_qdrant = MagicMock()
        mock_qdrant.get_collection.return_value = mock_collection
        mock_vq = MagicMock()
        mock_vq._qdrant = mock_qdrant
        server._vector_querier = mock_vq

    return server


class TestIngestStatus:
    @pytest.mark.asyncio
    async def test_healthy_status(self) -> None:
        mock_session = _mock_neo4j_session(
            label_counts={"File": 10, "Email": 5, "Entity": 3},
            entity_rows=[{"type": "Person", "cnt": 2}, {"type": "Org", "cnt": 1}],
            topic_rows=[{"source": "cluster", "cnt": 4}, {"source": "user", "cnt": 2}],
        )
        server = _setup_server_with_mocks(mock_session=mock_session, qdrant_points=100)
        result = await server._call_tool("ingest_status", {})

        data = json.loads(result[0].text)
        assert data["health"]["neo4j"] == "ok"
        assert data["health"]["qdrant"] == "ok"
        assert data["vectors"]["count"] == 100

    @pytest.mark.asyncio
    async def test_neo4j_down(self) -> None:
        server = _setup_server_with_mocks(
            neo4j_error=ConnectionError("Connection refused"),
            qdrant_points=50,
        )
        result = await server._call_tool("ingest_status", {})

        data = json.loads(result[0].text)
        assert "error" in data["health"]["neo4j"]
        assert data["health"]["qdrant"] == "ok"
        assert data["sources"] == {}

    @pytest.mark.asyncio
    async def test_qdrant_down(self) -> None:
        mock_session = _mock_neo4j_session()
        server = _setup_server_with_mocks(
            mock_session=mock_session,
            qdrant_error=ConnectionError("Qdrant unreachable"),
        )
        result = await server._call_tool("ingest_status", {})

        data = json.loads(result[0].text)
        assert data["health"]["neo4j"] == "ok"
        assert "error" in data["health"]["qdrant"]
        assert data["vectors"]["count"] == 0

    @pytest.mark.asyncio
    async def test_both_services_down(self) -> None:
        server = _setup_server_with_mocks(
            neo4j_error=ConnectionError("Neo4j down"),
            qdrant_error=ConnectionError("Qdrant down"),
        )
        result = await server._call_tool("ingest_status", {})

        data = json.loads(result[0].text)
        assert "error" in data["health"]["neo4j"]
        assert "error" in data["health"]["qdrant"]

    @pytest.mark.asyncio
    async def test_neo4j_session_error_caught(self) -> None:
        """Verify errors during Neo4j session queries are caught gracefully."""
        mock_session = MagicMock()
        mock_session.run.side_effect = RuntimeError("query failed")

        server = _setup_server_with_mocks(mock_session=mock_session)
        result = await server._call_tool("ingest_status", {})

        data = json.loads(result[0].text)
        assert "error" in data["health"]["neo4j"]
        # Qdrant should still work
        assert data["health"]["qdrant"] == "ok"
