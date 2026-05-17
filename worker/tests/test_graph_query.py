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
    _normalize_cypher_for_validation,
    _validate_cypher_readonly,
    _ensure_limit,
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
            "SHOW DATABASES",
            "SHOW USERS",
            "SHOW CURRENT USER",
            "show databases yield name",
        ],
    )
    def test_blocks_write_queries(self, cypher: str) -> None:
        with pytest.raises(ReadOnlyCypherViolation):
            _validate_cypher_readonly(cypher)

    @pytest.mark.parametrize(
        "cypher",
        [
            # Unicode NBSP (\u00a0) between word boundary to bypass \b
            "MATCH (n)\u00a0DELETE\u00a0n",
            "MATCH (n)\u00a0CREATE (m:X)",
            "\u00a0MERGE\u00a0(n:File {id: 'x'})",
            # Em space (\u2003)
            "MATCH (n)\u2003SET\u2003n.x = 1",
            # Cypher line comment to split keyword
            "MATCH (n) // safe\nDELETE n",
            # Cypher block comment to split keyword
            "MATCH (n) /* comment */ DELETE n",
            # Comment hiding a write after innocent-looking read
            "MATCH (n) RETURN n // \nCREATE (m:Evil)",
        ],
    )
    def test_blocks_unicode_and_comment_bypass(self, cypher: str) -> None:
        """Bypass attempts using Unicode whitespace or Cypher comments."""
        with pytest.raises(ReadOnlyCypherViolation):
            _validate_cypher_readonly(cypher)


class TestCypherNormalization:
    """Tests for the normalization step that defeats bypass techniques."""

    def test_replaces_nbsp_with_space(self) -> None:
        result = _normalize_cypher_for_validation("MATCH\u00a0(n)")
        assert "\u00a0" not in result
        assert "MATCH (n)" == result

    def test_replaces_em_space(self) -> None:
        result = _normalize_cypher_for_validation("A\u2003B")
        assert result == "A B"

    def test_strips_line_comments(self) -> None:
        result = _normalize_cypher_for_validation("MATCH (n) // comment\nRETURN n")
        assert "//" not in result
        assert "RETURN n" in result

    def test_strips_block_comments(self) -> None:
        result = _normalize_cypher_for_validation("MATCH /* evil */ (n)")
        assert "/*" not in result
        assert "MATCH" in result
        assert "(n)" in result

    def test_strips_multiline_block_comment(self) -> None:
        result = _normalize_cypher_for_validation("A /*\nhide\n*/ B")
        assert "hide" not in result
        assert "A" in result and "B" in result


# ------------------------------------------------------------------
# LIMIT enforcement
# ------------------------------------------------------------------


class TestEnsureLimit:
    def test_appends_limit_when_missing(self) -> None:
        assert _ensure_limit("MATCH (n) RETURN n") == "MATCH (n) RETURN n LIMIT 1000"

    def test_preserves_existing_limit(self) -> None:
        q = "MATCH (n) RETURN n LIMIT 10"
        assert _ensure_limit(q) == q

    def test_preserves_existing_limit_case_insensitive(self) -> None:
        q = "MATCH (n) RETURN n limit 50"
        assert _ensure_limit(q) == q

    def test_strips_trailing_semicolon(self) -> None:
        assert _ensure_limit("MATCH (n) RETURN n;") == "MATCH (n) RETURN n LIMIT 1000"

    def test_custom_limit(self) -> None:
        assert (
            _ensure_limit("MATCH (n) RETURN n", limit=5) == "MATCH (n) RETURN n LIMIT 5"
        )


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
    mock_gdb_cls: MagicMock,
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
    mock_graph._database = "neo4j"

    mock_neo4j_graph_cls.return_value = mock_graph

    # Mock the own driver for read-only transaction path.
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_records = (
        graph_results
        if graph_results is not None
        else [{"name": "Alice"}, {"name": "Bob"}]
    )
    mock_record_objs = [MagicMock(**{"data.return_value": r}) for r in mock_records]
    mock_tx_result = MagicMock()
    mock_tx_result.__iter__ = lambda self: iter(mock_record_objs)
    mock_session.__enter__ = lambda self: mock_session
    mock_session.__exit__ = MagicMock(return_value=False)
    mock_session.execute_read.side_effect = lambda fn: fn(
        MagicMock(**{"run.return_value": mock_tx_result})
    )
    mock_driver.session.return_value = mock_session
    mock_gdb_cls.return_value = mock_driver

    mock_chain = MagicMock()
    gen_chain = MagicMock()
    if generation_side_effect:
        gen_chain.invoke.side_effect = generation_side_effect
    else:
        gen_chain.invoke.return_value = {"text": generated_cypher}
    mock_chain.cypher_generation_chain = gen_chain

    qa_chain = MagicMock()
    qa_chain.invoke.return_value = {"text": qa_answer}
    mock_chain.qa_chain = qa_chain
    mock_chain.top_k = 10

    mock_chain_cls.from_llm.return_value = mock_chain

    return GraphQuerier(registry)


class TestGraphQuerier:
    """Tests for the GraphQuerier end-to-end flow with mocked deps."""

    @patch("worker.query.graph.build_driver")
    @patch("worker.query.graph.GraphCypherQAChain")
    @patch("worker.query.graph.Neo4jGraph")
    def test_query_returns_structured_result(
        self,
        mock_neo4j_graph_cls: MagicMock,
        mock_chain_cls: MagicMock,
        mock_gdb_cls: MagicMock,
    ) -> None:
        querier = _make_querier(mock_neo4j_graph_cls, mock_chain_cls, mock_gdb_cls)
        result = querier.query("Who is mentioned?")

        assert result.question == "Who is mentioned?"
        assert result.cypher == "MATCH (n:Entity) RETURN n.name"
        assert result.raw_results == [{"name": "Alice"}, {"name": "Bob"}]
        assert result.answer == "Alice and Bob"
        assert result.error is None

    @patch("worker.query.graph.build_driver")
    @patch("worker.query.graph.GraphCypherQAChain")
    @patch("worker.query.graph.Neo4jGraph")
    def test_query_blocks_write_cypher_before_execution(
        self,
        mock_neo4j_graph_cls: MagicMock,
        mock_chain_cls: MagicMock,
        mock_gdb_cls: MagicMock,
    ) -> None:
        querier = _make_querier(
            mock_neo4j_graph_cls,
            mock_chain_cls,
            mock_gdb_cls,
            generated_cypher="CREATE (n:File {name: 'evil'})",
        )
        with pytest.raises(ReadOnlyCypherViolation):
            querier.query("Do something bad")

        # Crucially, the write query must NOT have been executed.
        mock_gdb_cls.return_value.session.assert_not_called()

    @patch("worker.query.graph.build_driver")
    @patch("worker.query.graph.GraphCypherQAChain")
    @patch("worker.query.graph.Neo4jGraph")
    def test_query_handles_chain_exception(
        self,
        mock_neo4j_graph_cls: MagicMock,
        mock_chain_cls: MagicMock,
        mock_gdb_cls: MagicMock,
    ) -> None:
        querier = _make_querier(
            mock_neo4j_graph_cls,
            mock_chain_cls,
            mock_gdb_cls,
            generation_side_effect=RuntimeError("neo4j down"),
        )
        result = querier.query("test?")

        assert result.error == "neo4j down"
        assert result.cypher == ""

    @patch("worker.query.graph.build_driver")
    @patch("worker.query.graph.GraphCypherQAChain")
    @patch("worker.query.graph.Neo4jGraph")
    def test_query_uses_readonly_transaction(
        self,
        mock_neo4j_graph_cls: MagicMock,
        mock_chain_cls: MagicMock,
        mock_gdb_cls: MagicMock,
    ) -> None:
        querier = _make_querier(mock_neo4j_graph_cls, mock_chain_cls, mock_gdb_cls)
        querier.query("Who is mentioned?")

        # Verify execute_read was called on the own driver (not _graph._driver).
        mock_driver = mock_gdb_cls.return_value
        mock_driver.session.assert_called_once()
        mock_session = mock_driver.session.return_value
        mock_session.execute_read.assert_called_once()

    @patch("worker.query.graph.build_driver")
    @patch("worker.query.graph.GraphCypherQAChain")
    @patch("worker.query.graph.Neo4jGraph")
    def test_query_passes_timeout_to_tx_run(
        self,
        mock_neo4j_graph_cls: MagicMock,
        mock_chain_cls: MagicMock,
        mock_gdb_cls: MagicMock,
    ) -> None:
        querier = _make_querier(mock_neo4j_graph_cls, mock_chain_cls, mock_gdb_cls)
        querier.query("Who is mentioned?")

        # The work function passed to execute_read should call tx.run with timeout.
        mock_driver = mock_gdb_cls.return_value
        mock_session = mock_driver.session.return_value
        # execute_read was called with a function; that function calls tx.run
        # Our mock setup in _make_querier makes execute_read call fn(mock_tx).
        # Verify the mock tx.run was called with timeout kwarg.
        call_args = mock_session.execute_read.call_args
        work_fn = call_args[0][0]
        mock_tx = MagicMock()
        mock_record = MagicMock(**{"data.return_value": {"x": 1}})
        mock_tx.run.return_value = [mock_record]
        work_fn(mock_tx)
        mock_tx.run.assert_called_once()
        _, kwargs = mock_tx.run.call_args
        assert kwargs["timeout"] == 30

    @patch("worker.query.graph.build_driver")
    @patch("worker.query.graph.GraphCypherQAChain")
    @patch("worker.query.graph.Neo4jGraph")
    def test_refresh_schema(
        self,
        mock_neo4j_graph_cls: MagicMock,
        mock_chain_cls: MagicMock,
        mock_gdb_cls: MagicMock,
    ) -> None:
        """``refresh_schema`` now uses APOC-free Cypher (Neo4j Community
        Edition lacks ``apoc.meta.data``), so it queries the graph
        directly via ``db.labels()`` / ``db.relationshipTypes()``
        instead of delegating to ``Neo4jGraph.refresh_schema``.
        """
        querier = _make_querier(mock_neo4j_graph_cls, mock_chain_cls, mock_gdb_cls)
        mock_graph = mock_neo4j_graph_cls.return_value
        mock_graph.query.return_value = []
        mock_graph.query.reset_mock()

        querier.refresh_schema()

        # Verify the APOC-free schema discovery queries fired.
        cyphers = [
            (call.args[0] if call.args else "") for call in mock_graph.query.call_args_list
        ]
        assert any("db.labels()" in c for c in cyphers)
        assert any("db.relationshipTypes()" in c for c in cyphers)

    @patch("worker.query.graph.build_driver")
    @patch("worker.query.graph.GraphCypherQAChain")
    @patch("worker.query.graph.Neo4jGraph")
    def test_close_calls_driver_close(
        self,
        mock_neo4j_graph_cls: MagicMock,
        mock_chain_cls: MagicMock,
        mock_gdb_cls: MagicMock,
    ) -> None:
        querier = _make_querier(mock_neo4j_graph_cls, mock_chain_cls, mock_gdb_cls)
        mock_driver = querier._driver
        querier.close()
        mock_driver.close.assert_called_once()
        assert querier._graph is None
        assert querier._driver is None

    @patch("worker.query.graph.build_driver")
    @patch("worker.query.graph.GraphCypherQAChain")
    @patch("worker.query.graph.Neo4jGraph")
    def test_close_idempotent(
        self,
        mock_neo4j_graph_cls: MagicMock,
        mock_chain_cls: MagicMock,
        mock_gdb_cls: MagicMock,
    ) -> None:
        querier = _make_querier(mock_neo4j_graph_cls, mock_chain_cls, mock_gdb_cls)
        querier.close()
        querier.close()  # second call should not raise
        assert querier._graph is None
        assert querier._driver is None
