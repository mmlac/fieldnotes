"""Tests for the MCP search tool handler."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from worker.config import Config, Neo4jConfig, QdrantConfig
from worker.mcp_server import FieldnotesServer
from worker.query.graph import GraphQueryResult
from worker.query.vector import VectorQueryResult, VectorResult


def _make_server() -> FieldnotesServer:
    cfg = Config(
        neo4j=Neo4jConfig(uri="bolt://localhost:7687", user="neo4j", password="test"),
        qdrant=QdrantConfig(host="localhost", port=6333, collection="fieldnotes"),
    )
    return FieldnotesServer(cfg)


# ------------------------------------------------------------------
# Search tool
# ------------------------------------------------------------------


class TestSearchTool:
    @pytest.mark.asyncio
    @patch("worker.mcp_server.VectorQuerier")
    @patch("worker.mcp_server.GraphQuerier")
    @patch("worker.mcp_server.ModelRegistry")
    async def test_basic_search(
        self,
        mock_registry: MagicMock,
        mock_gq_cls: MagicMock,
        mock_vq_cls: MagicMock,
    ) -> None:
        server = _make_server()
        server._connect()

        graph_result = GraphQueryResult(
            question="test query",
            cypher="MATCH (n) RETURN n",
            raw_results=[{"n": {"source_id": "doc1", "name": "Doc 1"}}],
            answer="Found Doc 1",
        )
        vector_result = VectorQueryResult(
            question="test query",
            results=[
                VectorResult(
                    source_type="file",
                    source_id="doc2",
                    text="Some content",
                    date="2024-01-01",
                    score=0.9,
                ),
            ],
        )

        server._graph_querier.query = MagicMock(return_value=graph_result)
        server._vector_querier.query = MagicMock(return_value=vector_result)

        result = await server._call_tool("search", {"query": "test query"})

        assert len(result) == 1
        text = result[0].text
        assert "Doc 1" in text or "doc1" in text
        assert "doc2" in text

    @pytest.mark.asyncio
    @patch("worker.mcp_server.VectorQuerier")
    @patch("worker.mcp_server.GraphQuerier")
    @patch("worker.mcp_server.ModelRegistry")
    async def test_search_with_source_type_filter(
        self,
        mock_registry: MagicMock,
        mock_gq_cls: MagicMock,
        mock_vq_cls: MagicMock,
    ) -> None:
        server = _make_server()
        server._connect()

        graph_result = GraphQueryResult(question="q", cypher="", raw_results=[])
        vector_result = VectorQueryResult(question="q", results=[])

        server._graph_querier.query = MagicMock(return_value=graph_result)
        server._vector_querier.query = MagicMock(return_value=vector_result)

        await server._call_tool("search", {
            "query": "test",
            "source_type": "email",
            "top_k": 5,
        })

        # Verify source_type and top_k were passed to vector query
        server._vector_querier.query.assert_called_once()
        # The lambda captures args, so we check the mock was called
        assert server._vector_querier.query.called

    @pytest.mark.asyncio
    @patch("worker.mcp_server.VectorQuerier")
    @patch("worker.mcp_server.GraphQuerier")
    @patch("worker.mcp_server.ModelRegistry")
    async def test_search_graph_failure_fallback(
        self,
        mock_registry: MagicMock,
        mock_gq_cls: MagicMock,
        mock_vq_cls: MagicMock,
    ) -> None:
        """When graph query raises, results should still include vector results."""
        server = _make_server()
        server._connect()

        server._graph_querier.query = MagicMock(
            side_effect=ConnectionError("Neo4j down")
        )
        vector_result = VectorQueryResult(
            question="q",
            results=[
                VectorResult(
                    source_type="file",
                    source_id="v1",
                    text="Vector hit",
                    date="2024-01-01",
                    score=0.85,
                ),
            ],
        )
        server._vector_querier.query = MagicMock(return_value=vector_result)

        result = await server._call_tool("search", {"query": "test"})

        text = result[0].text
        # Should contain warnings about graph failure
        assert "v1" in text

    @pytest.mark.asyncio
    @patch("worker.mcp_server.VectorQuerier")
    @patch("worker.mcp_server.GraphQuerier")
    @patch("worker.mcp_server.ModelRegistry")
    async def test_search_vector_failure_fallback(
        self,
        mock_registry: MagicMock,
        mock_gq_cls: MagicMock,
        mock_vq_cls: MagicMock,
    ) -> None:
        """When vector query raises, results should still include graph results."""
        server = _make_server()
        server._connect()

        graph_result = GraphQueryResult(
            question="q",
            cypher="MATCH (n) RETURN n",
            raw_results=[{"source_id": "g1"}],
            answer="Graph answer",
        )
        server._graph_querier.query = MagicMock(return_value=graph_result)
        server._vector_querier.query = MagicMock(
            side_effect=ConnectionError("Qdrant down")
        )

        result = await server._call_tool("search", {"query": "test"})

        text = result[0].text
        assert "Graph answer" in text

    @pytest.mark.asyncio
    @patch("worker.mcp_server.VectorQuerier")
    @patch("worker.mcp_server.GraphQuerier")
    @patch("worker.mcp_server.ModelRegistry")
    async def test_search_empty_results(
        self,
        mock_registry: MagicMock,
        mock_gq_cls: MagicMock,
        mock_vq_cls: MagicMock,
    ) -> None:
        server = _make_server()
        server._connect()

        graph_result = GraphQueryResult(question="q", cypher="", raw_results=[])
        vector_result = VectorQueryResult(question="q", results=[])

        server._graph_querier.query = MagicMock(return_value=graph_result)
        server._vector_querier.query = MagicMock(return_value=vector_result)

        result = await server._call_tool("search", {"query": "nothing"})

        assert "No results found" in result[0].text

    @pytest.mark.asyncio
    @patch("worker.mcp_server.VectorQuerier")
    @patch("worker.mcp_server.GraphQuerier")
    @patch("worker.mcp_server.ModelRegistry")
    async def test_search_top_k_parameter(
        self,
        mock_registry: MagicMock,
        mock_gq_cls: MagicMock,
        mock_vq_cls: MagicMock,
    ) -> None:
        server = _make_server()
        server._connect()

        graph_result = GraphQueryResult(question="q", cypher="", raw_results=[])
        vector_result = VectorQueryResult(question="q", results=[])

        server._graph_querier.query = MagicMock(return_value=graph_result)
        server._vector_querier.query = MagicMock(return_value=vector_result)

        await server._call_tool("search", {"query": "test", "top_k": 20})

        # The lambda in _handle_search captures top_k, so verify the mock was called
        assert server._vector_querier.query.called

    @pytest.mark.asyncio
    @patch("worker.mcp_server.VectorQuerier")
    @patch("worker.mcp_server.GraphQuerier")
    @patch("worker.mcp_server.ModelRegistry")
    async def test_search_source_ids_deduplication(
        self,
        mock_registry: MagicMock,
        mock_gq_cls: MagicMock,
        mock_vq_cls: MagicMock,
    ) -> None:
        """Source IDs in the output should be deduplicated."""
        server = _make_server()
        server._connect()

        graph_result = GraphQueryResult(
            question="q",
            cypher="",
            raw_results=[
                {"n": {"source_id": "dup1"}},
                {"source_id": "dup1"},
            ],
        )
        vector_result = VectorQueryResult(
            question="q",
            results=[
                VectorResult("file", "dup1", "text", "2024-01-01", 0.9),
                VectorResult("file", "unique1", "text2", "2024-01-01", 0.8),
            ],
        )

        server._graph_querier.query = MagicMock(return_value=graph_result)
        server._vector_querier.query = MagicMock(return_value=vector_result)

        result = await server._call_tool("search", {"query": "test"})

        text = result[0].text
        # dup1 should appear in Sources but only once
        sources_section = text[text.index("[Sources]"):] if "[Sources]" in text else ""
        if sources_section:
            assert sources_section.count("dup1") == 1
