"""Tests for the graph query NL→Cypher translation layer.

Uses unittest.mock to stub out Neo4j and LangChain so tests run
without running services.
"""

from __future__ import annotations

from typing import Any
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
            "FOREACH (n IN [1,2] | CREATE (m:X))",
            "LOAD CSV FROM 'file:///etc/passwd' AS row RETURN row",
            "CALL apoc.periodic.commit('CREATE (n:X)', {})",
            "CALL apoc.periodic.iterate('MATCH (n) RETURN n', 'DELETE n', {})",
            "CALL apoc.cypher.run('CREATE (n:X)', {})",
            "CALL apoc.cypher.doIt('CREATE (n:X)', {})",
            "CALL apoc.create.node(['Label'], {name: 'evil'})",
            "CALL apoc.create.relationship(n, 'REL', {}, m)",
            "CALL dbms.security.createUser('admin', 'pass', false)",
            "CALL db.createLabel('Evil')",
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


def _make_querier(
    mock_neo4j_graph_cls: MagicMock,
    mock_chain_cls: MagicMock,
    *,
    generated_cypher: str = "MATCH (n:Entity) RETURN n.name",
    graph_results: list[dict[str, Any]] | None = None,
    qa_answer: str = "Alice and Bob",
    generation_side_effect: Exception | None = None,
) -> GraphQuerier:
    """Build a GraphQuerier with mocked Neo4j and LangChain sub-chains."""
    registry = MagicMock()
    resolved = MagicMock()
    resolved.alias = "test-model"
    registry.for_role.return_value = resolved

    mock_graph = MagicMock()
    mock_graph.get_structured_schema = {"nodes": [], "relationships": []}

    # Mock the driver for read-only transaction path.
    mock_session = MagicMock()
    mock_records = graph_results if graph_results is not None else [{"name": "Alice"}, {"name": "Bob"}]
    # execute_read calls the work function with a transaction; the tx.run()
    # result must yield record objects with a .data() method.
    mock_record_objs = [MagicMock(**{"data.return_value": r}) for r in mock_records]
    mock_tx_result = MagicMock()
    mock_tx_result.__iter__ = lambda self: iter(mock_record_objs)
    mock_session.__enter__ = lambda self: mock_session
    mock_session.__exit__ = MagicMock(return_value=False)
    mock_session.execute_read.side_effect = lambda fn: fn(MagicMock(**{"run.return_value": mock_tx_result}))
    mock_graph._driver.session.return_value = mock_session

    mock_neo4j_graph_cls.return_value = mock_graph

    mock_chain = MagicMock()
    gen_chain = MagicMock()
    if generation_side_effect:
        gen_chain.run.side_effect = generation_side_effect
    else:
        gen_chain.run.return_value = generated_cypher
    mock_chain.cypher_generation_chain = gen_chain

    qa_chain = MagicMock()
    qa_chain.invoke.return_value = {"text": qa_answer}
    mock_chain.qa_chain = qa_chain
    mock_chain.top_k = 10

    mock_chain_cls.from_llm.return_value = mock_chain

    return GraphQuerier(registry)


class TestGraphQuerier:
    """Tests for the GraphQuerier end-to-end flow with mocked deps."""

    @patch("worker.query.graph.GraphCypherQAChain")
    @patch("worker.query.graph.Neo4jGraph")
    def test_query_returns_structured_result(
        self, mock_neo4j_graph_cls: MagicMock, mock_chain_cls: MagicMock
    ) -> None:
        querier = _make_querier(mock_neo4j_graph_cls, mock_chain_cls)
        result = querier.query("Who is mentioned?")

        assert result.question == "Who is mentioned?"
        assert result.cypher == "MATCH (n:Entity) RETURN n.name"
        assert result.raw_results == [{"name": "Alice"}, {"name": "Bob"}]
        assert result.answer == "Alice and Bob"
        assert result.error is None

    @patch("worker.query.graph.GraphCypherQAChain")
    @patch("worker.query.graph.Neo4jGraph")
    def test_query_blocks_write_cypher_before_execution(
        self, mock_neo4j_graph_cls: MagicMock, mock_chain_cls: MagicMock
    ) -> None:
        querier = _make_querier(
            mock_neo4j_graph_cls, mock_chain_cls,
            generated_cypher="CREATE (n:File {name: 'evil'})",
        )
        with pytest.raises(ReadOnlyCypherViolation):
            querier.query("Do something bad")

        # Crucially, the write query must NOT have been executed against Neo4j.
        mock_neo4j_graph_cls.return_value._driver.session.assert_not_called()

    @patch("worker.query.graph.GraphCypherQAChain")
    @patch("worker.query.graph.Neo4jGraph")
    def test_query_handles_chain_exception(
        self, mock_neo4j_graph_cls: MagicMock, mock_chain_cls: MagicMock
    ) -> None:
        querier = _make_querier(
            mock_neo4j_graph_cls, mock_chain_cls,
            generation_side_effect=RuntimeError("neo4j down"),
        )
        result = querier.query("test?")

        assert result.error == "neo4j down"
        assert result.cypher == ""

    @patch("worker.query.graph.GraphCypherQAChain")
    @patch("worker.query.graph.Neo4jGraph")
    def test_query_uses_readonly_transaction(
        self, mock_neo4j_graph_cls: MagicMock, mock_chain_cls: MagicMock
    ) -> None:
        querier = _make_querier(mock_neo4j_graph_cls, mock_chain_cls)
        result = querier.query("Who is mentioned?")

        # Verify execute_read was called (not a plain query/execute_write).
        mock_graph = mock_neo4j_graph_cls.return_value
        mock_graph._driver.session.assert_called_once()
        # _make_querier sets up mock_session with custom __enter__ that returns itself
        mock_session = mock_graph._driver.session.return_value
        mock_session.execute_read.assert_called_once()

    @patch("worker.query.graph.GraphCypherQAChain")
    @patch("worker.query.graph.Neo4jGraph")
    def test_refresh_schema(
        self, mock_neo4j_graph_cls: MagicMock, mock_chain_cls: MagicMock
    ) -> None:
        querier = _make_querier(mock_neo4j_graph_cls, mock_chain_cls)
        mock_graph = mock_neo4j_graph_cls.return_value
        mock_graph.refresh_schema.reset_mock()
        querier.refresh_schema()
        mock_graph.refresh_schema.assert_called_once()

    @patch("worker.query.graph.GraphCypherQAChain")
    @patch("worker.query.graph.Neo4jGraph")
    def test_close_calls_driver_close(
        self, mock_neo4j_graph_cls: MagicMock, mock_chain_cls: MagicMock
    ) -> None:
        querier = _make_querier(mock_neo4j_graph_cls, mock_chain_cls)
        mock_driver = querier._graph._driver
        querier.close()
        mock_driver.close.assert_called_once()
        assert querier._graph is None

    @patch("worker.query.graph.GraphCypherQAChain")
    @patch("worker.query.graph.Neo4jGraph")
    def test_close_idempotent(
        self, mock_neo4j_graph_cls: MagicMock, mock_chain_cls: MagicMock
    ) -> None:
        querier = _make_querier(mock_neo4j_graph_cls, mock_chain_cls)
        querier.close()
        querier.close()  # second call should not raise
        assert querier._graph is None
