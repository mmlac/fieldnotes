"""Tests for the MCP ask tool handler."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from worker.config import Config, Neo4jConfig, QdrantConfig
from worker.mcp_server import FieldnotesServer
from worker.models.base import CompletionResponse
from worker.query import EMPTY_CORPUS_MESSAGE
from worker.query.graph import GraphQueryResult
from worker.query.vector import VectorQueryResult, VectorResult


@pytest.fixture(autouse=True)
def _non_empty_corpus():
    """Default: corpus is non-empty so queries proceed normally."""
    with patch("worker.mcp_server.is_corpus_empty", return_value=False):
        yield


def _make_server() -> FieldnotesServer:
    cfg = Config(
        neo4j=Neo4jConfig(uri="bolt://localhost:7687", user="neo4j", password="test"),
        qdrant=QdrantConfig(host="localhost", port=6333, collection="fieldnotes"),
    )
    return FieldnotesServer(cfg)


class TestAskTool:
    @pytest.mark.asyncio
    @patch("worker.mcp_server.VectorQuerier")
    @patch("worker.mcp_server.GraphQuerier")
    @patch("worker.mcp_server.ModelRegistry")
    async def test_basic_ask(
        self,
        mock_registry_cls: MagicMock,
        mock_gq_cls: MagicMock,
        mock_vq_cls: MagicMock,
    ) -> None:
        server = _make_server()
        server._connect()

        graph_result = GraphQueryResult(
            question="what is X?",
            cypher="MATCH (n) RETURN n",
            raw_results=[{"n": {"source_id": "doc1", "name": "Doc 1"}}],
            answer="X is a thing",
        )
        vector_result = VectorQueryResult(
            question="what is X?",
            results=[
                VectorResult(
                    source_type="file",
                    source_id="doc2",
                    text="X is explained here",
                    date="2024-01-01",
                    score=0.9,
                ),
            ],
        )

        server._graph_querier.query = MagicMock(return_value=graph_result)
        server._vector_querier.query = MagicMock(return_value=vector_result)

        # Mock the LLM call
        mock_model = MagicMock()
        mock_model.complete.return_value = CompletionResponse(
            text="X is a concept described in doc1 and doc2.",
            input_tokens=100,
            output_tokens=20,
        )
        server._registry.for_role = MagicMock(return_value=mock_model)

        result = await server._call_tool("ask", {"question": "what is X?"})

        assert len(result) == 1
        text = result[0].text
        assert "[Answer]" in text
        assert "X is a concept" in text
        assert "[Sources]" in text
        assert "doc1" in text
        assert "doc2" in text

    @pytest.mark.asyncio
    @patch("worker.mcp_server.VectorQuerier")
    @patch("worker.mcp_server.GraphQuerier")
    @patch("worker.mcp_server.ModelRegistry")
    async def test_ask_no_context(
        self,
        mock_registry_cls: MagicMock,
        mock_gq_cls: MagicMock,
        mock_vq_cls: MagicMock,
    ) -> None:
        """When no context is found, returns a graceful message."""
        server = _make_server()
        server._connect()

        graph_result = GraphQueryResult(question="q", cypher="", raw_results=[])
        vector_result = VectorQueryResult(question="q", results=[])

        server._graph_querier.query = MagicMock(return_value=graph_result)
        server._vector_querier.query = MagicMock(return_value=vector_result)

        result = await server._call_tool("ask", {"question": "unknown topic"})

        text = result[0].text
        assert "[Answer]" in text
        assert "don't have enough information" in text
        assert "[Confidence]" in text

    @pytest.mark.asyncio
    @patch("worker.mcp_server.VectorQuerier")
    @patch("worker.mcp_server.GraphQuerier")
    @patch("worker.mcp_server.ModelRegistry")
    async def test_ask_sparse_context_confidence(
        self,
        mock_registry_cls: MagicMock,
        mock_gq_cls: MagicMock,
        mock_vq_cls: MagicMock,
    ) -> None:
        """When context is sparse (< 3 results), shows low confidence."""
        server = _make_server()
        server._connect()

        graph_result = GraphQueryResult(question="q", cypher="", raw_results=[])
        vector_result = VectorQueryResult(
            question="q",
            results=[
                VectorResult("file", "doc1", "Some text", "2024-01-01", 0.8),
            ],
        )

        server._graph_querier.query = MagicMock(return_value=graph_result)
        server._vector_querier.query = MagicMock(return_value=vector_result)

        mock_model = MagicMock()
        mock_model.complete.return_value = CompletionResponse(
            text="Based on limited info...",
            input_tokens=50,
            output_tokens=10,
        )
        server._registry.for_role = MagicMock(return_value=mock_model)

        result = await server._call_tool("ask", {"question": "sparse query"})

        text = result[0].text
        assert "[Confidence]" in text
        assert "low" in text

    @pytest.mark.asyncio
    @patch("worker.mcp_server.VectorQuerier")
    @patch("worker.mcp_server.GraphQuerier")
    @patch("worker.mcp_server.ModelRegistry")
    async def test_ask_with_source_type_filter(
        self,
        mock_registry_cls: MagicMock,
        mock_gq_cls: MagicMock,
        mock_vq_cls: MagicMock,
    ) -> None:
        server = _make_server()
        server._connect()

        graph_result = GraphQueryResult(question="q", cypher="", raw_results=[])
        vector_result = VectorQueryResult(
            question="q",
            results=[
                VectorResult("email", "e1", "Email content", "2024-01-01", 0.9),
                VectorResult("email", "e2", "More email", "2024-01-02", 0.85),
                VectorResult("email", "e3", "Another email", "2024-01-03", 0.8),
            ],
        )

        server._graph_querier.query = MagicMock(return_value=graph_result)
        server._vector_querier.query = MagicMock(return_value=vector_result)

        mock_model = MagicMock()
        mock_model.complete.return_value = CompletionResponse(
            text="Based on emails...",
            input_tokens=80,
            output_tokens=15,
        )
        server._registry.for_role = MagicMock(return_value=mock_model)

        result = await server._call_tool(
            "ask", {"question": "email stuff", "source_type": "email"}
        )

        text = result[0].text
        assert "[Answer]" in text
        # Should NOT show low confidence with 3 results
        assert "[Confidence]" not in text

    @pytest.mark.asyncio
    @patch("worker.mcp_server.VectorQuerier")
    @patch("worker.mcp_server.GraphQuerier")
    @patch("worker.mcp_server.ModelRegistry")
    async def test_ask_falls_back_to_extraction_role(
        self,
        mock_registry_cls: MagicMock,
        mock_gq_cls: MagicMock,
        mock_vq_cls: MagicMock,
    ) -> None:
        """When 'query' role is not configured, falls back to 'extraction'."""
        server = _make_server()
        server._connect()

        graph_result = GraphQueryResult(
            question="q",
            cypher="",
            raw_results=[{"source_id": "d1"}],
            answer="Answer",
        )
        vector_result = VectorQueryResult(question="q", results=[])

        server._graph_querier.query = MagicMock(return_value=graph_result)
        server._vector_querier.query = MagicMock(return_value=vector_result)

        mock_model = MagicMock()
        mock_model.complete.return_value = CompletionResponse(
            text="Fallback answer",
            input_tokens=50,
            output_tokens=10,
        )

        def _for_role(role: str) -> MagicMock:
            if role == "query":
                raise KeyError("query role not configured")
            return mock_model

        server._registry.for_role = MagicMock(side_effect=_for_role)

        result = await server._call_tool("ask", {"question": "test"})

        text = result[0].text
        assert "Fallback answer" in text
        # Verify extraction role was used as fallback
        assert server._registry.for_role.call_count == 2

    @pytest.mark.asyncio
    @patch("worker.mcp_server.VectorQuerier")
    @patch("worker.mcp_server.GraphQuerier")
    @patch("worker.mcp_server.ModelRegistry")
    async def test_ask_graph_failure_still_works(
        self,
        mock_registry_cls: MagicMock,
        mock_gq_cls: MagicMock,
        mock_vq_cls: MagicMock,
    ) -> None:
        """When graph query fails, ask should still work with vector results."""
        server = _make_server()
        server._connect()

        server._graph_querier.query = MagicMock(
            side_effect=ConnectionError("Neo4j down")
        )
        vector_result = VectorQueryResult(
            question="q",
            results=[
                VectorResult("file", "v1", "Text", "2024-01-01", 0.9),
                VectorResult("file", "v2", "More text", "2024-01-02", 0.85),
                VectorResult("file", "v3", "Even more", "2024-01-03", 0.8),
            ],
        )
        server._vector_querier.query = MagicMock(return_value=vector_result)

        mock_model = MagicMock()
        mock_model.complete.return_value = CompletionResponse(
            text="Answer from vector context only.",
            input_tokens=60,
            output_tokens=12,
        )
        server._registry.for_role = MagicMock(return_value=mock_model)

        result = await server._call_tool("ask", {"question": "test"})

        text = result[0].text
        assert "[Answer]" in text
        assert "Answer from vector context only" in text
        assert "Warnings" in text

    @pytest.mark.asyncio
    @patch("worker.mcp_server.VectorQuerier")
    @patch("worker.mcp_server.GraphQuerier")
    @patch("worker.mcp_server.ModelRegistry")
    async def test_ask_source_ids_deduplicated(
        self,
        mock_registry_cls: MagicMock,
        mock_gq_cls: MagicMock,
        mock_vq_cls: MagicMock,
    ) -> None:
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
                VectorResult("file", "unique2", "text3", "2024-01-02", 0.7),
            ],
        )

        server._graph_querier.query = MagicMock(return_value=graph_result)
        server._vector_querier.query = MagicMock(return_value=vector_result)

        mock_model = MagicMock()
        mock_model.complete.return_value = CompletionResponse(
            text="Synthesized answer.",
            input_tokens=80,
            output_tokens=15,
        )
        server._registry.for_role = MagicMock(return_value=mock_model)

        result = await server._call_tool("ask", {"question": "test"})

        text = result[0].text
        sources_section = text[text.index("[Sources]") :]
        assert sources_section.count("dup1") == 1

    @pytest.mark.asyncio
    @patch("worker.mcp_server.is_corpus_empty", return_value=True)
    @patch("worker.mcp_server.VectorQuerier")
    @patch("worker.mcp_server.GraphQuerier")
    @patch("worker.mcp_server.ModelRegistry")
    async def test_ask_empty_corpus(
        self,
        mock_registry_cls: MagicMock,
        mock_gq_cls: MagicMock,
        mock_vq_cls: MagicMock,
        mock_empty: MagicMock,
    ) -> None:
        """When corpus is empty, return specific guidance message."""
        server = _make_server()
        server._connect()

        result = await server._call_tool("ask", {"question": "anything"})

        text = result[0].text
        assert "[Answer]" in text
        assert EMPTY_CORPUS_MESSAGE in text
        assert "corpus is empty" in text
        # Should NOT have called the actual query methods.
        server._graph_querier.query.assert_not_called()
        server._vector_querier.query.assert_not_called()
