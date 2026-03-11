"""Tests for the graph query NL→Cypher translation layer.

Uses unittest.mock to stub out Neo4j and LangChain so tests run
without running services.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from worker.query.graph import (
    GraphQuerier,
    GraphQueryResult,
    ReadOnlyCypherViolation,
    _validate_cypher_readonly,
    _RegistryLLM,
    _lc_role,
)


# ------------------------------------------------------------------
# Read-only validation
# ------------------------------------------------------------------


class TestCypherReadOnlyValidation:
    """Tests for the _validate_cypher_readonly safety gate."""

    @pytest.mark.parametrize(
        "cypher",
        [
            "MATCH (n:File) RETURN n",
            "MATCH (a)-[:MENTIONS]->(b) RETURN a.name, b.name",
            "MATCH (n) WHERE n.source_id = 'x' RETURN n LIMIT 10",
            "MATCH (n) RETURN count(n)",
            "OPTIONAL MATCH (n)-[r]->(m) RETURN n, r, m",
        ],
    )
    def test_allows_read_queries(self, cypher: str) -> None:
        _validate_cypher_readonly(cypher)  # should not raise

    @pytest.mark.parametrize(
        "cypher",
        [
            "CREATE (n:File {name: 'evil'})",
            "MERGE (n:File {source_id: 'x'})",
            "MATCH (n) DELETE n",
            "MATCH (n) DETACH DELETE n",
            "MATCH (n:File) SET n.name = 'hacked'",
            "MATCH (n) REMOVE n.name",
            "DROP CONSTRAINT constraint_name",
            "match (n) delete n",  # case insensitive
            "MATCH (n) CALL { CREATE (m:X) }",
        ],
    )
    def test_blocks_write_queries(self, cypher: str) -> None:
        with pytest.raises(ReadOnlyCypherViolation):
            _validate_cypher_readonly(cypher)


# ------------------------------------------------------------------
# _lc_role mapping
# ------------------------------------------------------------------


class TestLcRole:
    def test_system_message(self) -> None:
        msg = MagicMock()
        msg.type = "system"
        assert _lc_role(msg) == "system"

    def test_ai_message(self) -> None:
        msg = MagicMock()
        msg.type = "ai"
        assert _lc_role(msg) == "assistant"

    def test_human_message(self) -> None:
        msg = MagicMock()
        msg.type = "human"
        assert _lc_role(msg) == "user"


# ------------------------------------------------------------------
# GraphQueryResult dataclass
# ------------------------------------------------------------------


class TestGraphQueryResult:
    def test_defaults(self) -> None:
        r = GraphQueryResult(question="test?", cypher="MATCH (n) RETURN n")
        assert r.question == "test?"
        assert r.raw_results == []
        assert r.answer == ""
        assert r.error is None

    def test_with_error(self) -> None:
        r = GraphQueryResult(question="q", cypher="", error="boom")
        assert r.error == "boom"


# ------------------------------------------------------------------
# GraphQuerier
# ------------------------------------------------------------------


class TestGraphQuerier:
    """Tests for the GraphQuerier end-to-end flow with mocked deps."""

    @patch("worker.query.graph.GraphCypherQAChain")
    @patch("worker.query.graph.Neo4jGraph")
    def test_query_returns_structured_result(
        self, mock_neo4j_graph_cls: MagicMock, mock_chain_cls: MagicMock
    ) -> None:
        registry = MagicMock()
        resolved = MagicMock()
        resolved.alias = "test-model"
        registry.for_role.return_value = resolved

        mock_graph = MagicMock()
        mock_neo4j_graph_cls.return_value = mock_graph

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "result": "Alice and Bob",
            "intermediate_steps": [
                {"query": "MATCH (n:Entity) RETURN n.name", "context": [{"name": "Alice"}, {"name": "Bob"}]}
            ],
        }
        mock_chain_cls.from_llm.return_value = mock_chain

        querier = GraphQuerier(registry)
        result = querier.query("Who is mentioned?")

        assert result.question == "Who is mentioned?"
        assert result.cypher == "MATCH (n:Entity) RETURN n.name"
        assert result.raw_results == [{"name": "Alice"}, {"name": "Bob"}]
        assert result.answer == "Alice and Bob"
        assert result.error is None

    @patch("worker.query.graph.GraphCypherQAChain")
    @patch("worker.query.graph.Neo4jGraph")
    def test_query_blocks_write_cypher(
        self, mock_neo4j_graph_cls: MagicMock, mock_chain_cls: MagicMock
    ) -> None:
        registry = MagicMock()
        resolved = MagicMock()
        resolved.alias = "test-model"
        registry.for_role.return_value = resolved

        mock_graph = MagicMock()
        mock_neo4j_graph_cls.return_value = mock_graph

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "result": "done",
            "intermediate_steps": [
                {"query": "CREATE (n:File {name: 'evil'})", "context": []}
            ],
        }
        mock_chain_cls.from_llm.return_value = mock_chain

        querier = GraphQuerier(registry)
        with pytest.raises(ReadOnlyCypherViolation):
            querier.query("Do something bad")

    @patch("worker.query.graph.GraphCypherQAChain")
    @patch("worker.query.graph.Neo4jGraph")
    def test_query_handles_chain_exception(
        self, mock_neo4j_graph_cls: MagicMock, mock_chain_cls: MagicMock
    ) -> None:
        registry = MagicMock()
        resolved = MagicMock()
        resolved.alias = "test-model"
        registry.for_role.return_value = resolved

        mock_graph = MagicMock()
        mock_neo4j_graph_cls.return_value = mock_graph

        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = RuntimeError("neo4j down")
        mock_chain_cls.from_llm.return_value = mock_chain

        querier = GraphQuerier(registry)
        result = querier.query("test?")

        assert result.error == "neo4j down"
        assert result.cypher == ""

    @patch("worker.query.graph.GraphCypherQAChain")
    @patch("worker.query.graph.Neo4jGraph")
    def test_refresh_schema(
        self, mock_neo4j_graph_cls: MagicMock, mock_chain_cls: MagicMock
    ) -> None:
        registry = MagicMock()
        resolved = MagicMock()
        resolved.alias = "test-model"
        registry.for_role.return_value = resolved

        mock_graph = MagicMock()
        mock_neo4j_graph_cls.return_value = mock_graph
        mock_chain_cls.from_llm.return_value = MagicMock()

        querier = GraphQuerier(registry)
        # Called once in __init__, then once more on refresh
        mock_graph.refresh_schema.reset_mock()
        querier.refresh_schema()
        mock_graph.refresh_schema.assert_called_once()

    @patch("worker.query.graph.GraphCypherQAChain")
    @patch("worker.query.graph.Neo4jGraph")
    def test_close(
        self, mock_neo4j_graph_cls: MagicMock, mock_chain_cls: MagicMock
    ) -> None:
        registry = MagicMock()
        resolved = MagicMock()
        resolved.alias = "test-model"
        registry.for_role.return_value = resolved

        mock_neo4j_graph_cls.return_value = MagicMock()
        mock_chain_cls.from_llm.return_value = MagicMock()

        querier = GraphQuerier(registry)
        querier.close()
        assert querier._graph is None
