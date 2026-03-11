"""Tests for the MCP ingest_status tool."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from worker.config import Config, Neo4jConfig, QdrantConfig
from worker.mcp_server import FieldnotesServer


def _make_server() -> FieldnotesServer:
    cfg = Config(
        neo4j=Neo4jConfig(uri="bolt://localhost:7687", user="neo4j", password="test"),
        qdrant=QdrantConfig(host="localhost", port=6333, collection="fieldnotes"),
    )
    return FieldnotesServer(cfg)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


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


class TestIngestStatus:
    @patch("worker.mcp_server.QdrantClient")
    @patch("worker.mcp_server.GraphDatabase")
    def test_healthy_status(
        self,
        mock_graph_db: MagicMock,
        mock_qdrant_cls: MagicMock,
    ) -> None:
        # Mock Neo4j
        mock_session = _mock_neo4j_session(
            label_counts={"File": 10, "Email": 5, "Entity": 3},
            entity_rows=[{"type": "Person", "cnt": 2}, {"type": "Org", "cnt": 1}],
            topic_rows=[{"source": "cluster", "cnt": 4}, {"source": "user", "cnt": 2}],
        )
        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
        mock_graph_db.driver.return_value = mock_driver

        # Mock Qdrant
        mock_collection_info = MagicMock()
        mock_collection_info.points_count = 100
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection_info
        mock_qdrant_cls.return_value = mock_client

        server = _make_server()
        result = _run(server._call_tool("ingest_status", {}))

        data = json.loads(result[0].text)
        assert data["health"]["neo4j"] == "ok"
        assert data["health"]["qdrant"] == "ok"
        assert data["vectors"]["count"] == 100

    @patch("worker.mcp_server.QdrantClient")
    @patch("worker.mcp_server.GraphDatabase")
    def test_neo4j_down(
        self,
        mock_graph_db: MagicMock,
        mock_qdrant_cls: MagicMock,
    ) -> None:
        # Neo4j driver raises on connection
        mock_graph_db.driver.side_effect = ConnectionError("Connection refused")

        # Qdrant works
        mock_collection_info = MagicMock()
        mock_collection_info.points_count = 50
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection_info
        mock_qdrant_cls.return_value = mock_client

        server = _make_server()
        result = _run(server._call_tool("ingest_status", {}))

        data = json.loads(result[0].text)
        assert "error" in data["health"]["neo4j"]
        assert data["health"]["qdrant"] == "ok"
        assert data["sources"] == {}

    @patch("worker.mcp_server.QdrantClient")
    @patch("worker.mcp_server.GraphDatabase")
    def test_qdrant_down(
        self,
        mock_graph_db: MagicMock,
        mock_qdrant_cls: MagicMock,
    ) -> None:
        # Neo4j works
        mock_session = _mock_neo4j_session()
        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
        mock_graph_db.driver.return_value = mock_driver

        # Qdrant fails
        mock_qdrant_cls.side_effect = ConnectionError("Qdrant unreachable")

        server = _make_server()
        result = _run(server._call_tool("ingest_status", {}))

        data = json.loads(result[0].text)
        assert data["health"]["neo4j"] == "ok"
        assert "error" in data["health"]["qdrant"]
        assert data["vectors"]["count"] == 0

    @patch("worker.mcp_server.QdrantClient")
    @patch("worker.mcp_server.GraphDatabase")
    def test_both_services_down(
        self,
        mock_graph_db: MagicMock,
        mock_qdrant_cls: MagicMock,
    ) -> None:
        mock_graph_db.driver.side_effect = ConnectionError("Neo4j down")
        mock_qdrant_cls.side_effect = ConnectionError("Qdrant down")

        server = _make_server()
        result = _run(server._call_tool("ingest_status", {}))

        data = json.loads(result[0].text)
        assert "error" in data["health"]["neo4j"]
        assert "error" in data["health"]["qdrant"]
