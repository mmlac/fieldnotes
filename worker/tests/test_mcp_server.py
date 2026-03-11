"""Tests for MCP server initialization, tool listing, and lifecycle."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from worker.config import Config, Neo4jConfig, QdrantConfig
from worker.mcp_server import FieldnotesServer, TOOLS


# ------------------------------------------------------------------
# Tool definitions
# ------------------------------------------------------------------


class TestToolDefinitions:
    """Verify the TOOLS list has the expected entries and schemas."""

    def test_all_tools_present(self) -> None:
        names = {t.name for t in TOOLS}
        assert names == {"search", "list_topics", "show_topic", "topic_gaps", "ingest_status"}

    def test_search_schema_requires_query(self) -> None:
        search = next(t for t in TOOLS if t.name == "search")
        assert "query" in search.inputSchema["required"]

    def test_search_schema_has_top_k(self) -> None:
        search = next(t for t in TOOLS if t.name == "search")
        assert "top_k" in search.inputSchema["properties"]

    def test_search_schema_has_source_type_enum(self) -> None:
        search = next(t for t in TOOLS if t.name == "search")
        source_type = search.inputSchema["properties"]["source_type"]
        assert "enum" in source_type
        assert "file" in source_type["enum"]

    def test_show_topic_requires_name(self) -> None:
        st = next(t for t in TOOLS if t.name == "show_topic")
        assert "name" in st.inputSchema["required"]

    def test_list_topics_source_enum(self) -> None:
        lt = next(t for t in TOOLS if t.name == "list_topics")
        source = lt.inputSchema["properties"]["source"]
        assert set(source["enum"]) == {"all", "cluster", "user"}

    def test_topic_gaps_no_required(self) -> None:
        tg = next(t for t in TOOLS if t.name == "topic_gaps")
        assert "required" not in tg.inputSchema

    def test_ingest_status_no_required(self) -> None:
        ist = next(t for t in TOOLS if t.name == "ingest_status")
        assert "required" not in ist.inputSchema


# ------------------------------------------------------------------
# Server lifecycle
# ------------------------------------------------------------------


def _make_cfg() -> Config:
    return Config(
        neo4j=Neo4jConfig(uri="bolt://localhost:7687", user="neo4j", password="test"),
        qdrant=QdrantConfig(host="localhost", port=6333, collection="fieldnotes"),
    )


class TestFieldnotesServerInit:
    def test_init_creates_server(self) -> None:
        cfg = _make_cfg()
        server = FieldnotesServer(cfg)
        assert server._graph_querier is None
        assert server._vector_querier is None

    @patch("worker.mcp_server.VectorQuerier")
    @patch("worker.mcp_server.GraphQuerier")
    @patch("worker.mcp_server.ModelRegistry")
    def test_connect_initialises_queriers(
        self,
        mock_registry: MagicMock,
        mock_gq: MagicMock,
        mock_vq: MagicMock,
    ) -> None:
        cfg = _make_cfg()
        server = FieldnotesServer(cfg)
        server._connect()
        mock_registry.assert_called_once_with(cfg)
        mock_gq.assert_called_once()
        mock_vq.assert_called_once()
        assert server._graph_querier is not None
        assert server._vector_querier is not None

    @patch("worker.mcp_server.VectorQuerier")
    @patch("worker.mcp_server.GraphQuerier")
    @patch("worker.mcp_server.ModelRegistry")
    def test_disconnect_closes_queriers(
        self,
        mock_registry: MagicMock,
        mock_gq: MagicMock,
        mock_vq: MagicMock,
    ) -> None:
        cfg = _make_cfg()
        server = FieldnotesServer(cfg)
        server._connect()
        gq = server._graph_querier
        vq = server._vector_querier
        server._disconnect()
        gq.close.assert_called_once()
        vq.close.assert_called_once()
        assert server._graph_querier is None
        assert server._vector_querier is None

    @patch("worker.mcp_server.VectorQuerier")
    @patch("worker.mcp_server.GraphQuerier")
    @patch("worker.mcp_server.ModelRegistry")
    def test_disconnect_idempotent(
        self,
        mock_registry: MagicMock,
        mock_gq: MagicMock,
        mock_vq: MagicMock,
    ) -> None:
        cfg = _make_cfg()
        server = FieldnotesServer(cfg)
        # disconnect without connect should not raise
        server._disconnect()
        assert server._graph_querier is None


class TestListTools:
    def test_list_tools_returns_all(self) -> None:
        cfg = _make_cfg()
        server = FieldnotesServer(cfg)
        result = asyncio.get_event_loop().run_until_complete(server._list_tools())
        assert result == TOOLS
        assert len(result) == 5


class TestCallToolErrors:
    def test_unknown_tool_returns_error(self) -> None:
        cfg = _make_cfg()
        server = FieldnotesServer(cfg)
        result = asyncio.get_event_loop().run_until_complete(
            server._call_tool("nonexistent", {})
        )
        assert len(result) == 1
        assert "error" in result[0].text
        assert "Unknown tool" in result[0].text
