"""Tests for transitive SAME_AS closure in writer.py."""

from __future__ import annotations

from unittest.mock import MagicMock

from worker.pipeline.writer import Writer


class TestCloseSameAsTransitive:
    """Tests for close_same_as_transitive() with mocked Neo4j."""

    def _make_writer(self, created_count: int = 0):
        """Create a Writer with mocked Neo4j returning given created count."""
        writer = object.__new__(Writer)
        writer._neo4j_driver = MagicMock()

        mock_session = MagicMock()
        writer._neo4j_driver.session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        writer._neo4j_driver.session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        mock_result = MagicMock()
        mock_result.single.return_value = {"cnt": created_count}
        mock_session.run.return_value = mock_result

        return writer, mock_session

    def test_returns_zero_when_no_new_edges(self):
        writer, _ = self._make_writer(created_count=0)
        assert writer._close_same_as_transitive_neo4j() == 0

    def test_returns_count_of_created_edges(self):
        writer, _ = self._make_writer(created_count=5)
        assert writer._close_same_as_transitive_neo4j() == 5

    def test_runs_correct_cypher_query(self):
        writer, mock_session = self._make_writer(created_count=0)
        writer._close_same_as_transitive_neo4j()

        assert mock_session.run.call_count == 1
        query = mock_session.run.call_args[0][0]
        # Verify key parts of the Cypher query
        assert "SAME_AS*2..4" in query
        assert "id(a) < id(b)" in query
        assert "NOT (a)-[:SAME_AS]-(b)" in query
        assert "MERGE" in query
        assert "transitive_closure" in query

    def test_sets_match_type_metadata(self):
        writer, mock_session = self._make_writer(created_count=1)
        writer._close_same_as_transitive_neo4j()

        query = mock_session.run.call_args[0][0]
        assert "match_type = 'transitive_closure'" in query
        assert "cross_source = true" in query

    def test_public_method_delegates(self):
        """close_same_as_transitive() should call the neo4j implementation."""
        writer, _ = self._make_writer(created_count=3)
        result = writer.close_same_as_transitive()
        assert result == 3

    def test_session_used_as_context_manager(self):
        writer, _ = self._make_writer(created_count=0)
        writer._close_same_as_transitive_neo4j()
        writer._neo4j_driver.session.assert_called_once()


class TestTransitiveClosureLogic:
    """Verify the Cypher pattern covers expected transitive scenarios.

    These tests document the intended behavior of the Cypher query:
    - 2-hop: A↔B, B↔C → A↔C
    - 3-hop: A↔B, B↔C, C↔D → A↔D (and A↔C, B↔D)
    - Already-linked pairs are skipped
    - Direction-agnostic matching (undirected SAME_AS traversal)
    """

    def test_query_uses_undirected_traversal(self):
        """SAME_AS*2..4 without direction arrow matches both directions."""
        writer, mock_session = self._make_writer(created_count=0)
        writer._close_same_as_transitive_neo4j()
        query = mock_session.run.call_args[0][0]
        # The pattern (a)-[:SAME_AS*2..4]-(b) uses undirected matching
        assert "[:SAME_AS*2..4]-" in query
        # No directed arrow → or ←
        assert "->(" not in query.split("MATCH")[1].split("WHERE")[0]

    def test_query_prevents_self_loops(self):
        """id(a) < id(b) prevents creating self-referential edges."""
        writer, mock_session = self._make_writer(created_count=0)
        writer._close_same_as_transitive_neo4j()
        query = mock_session.run.call_args[0][0]
        assert "id(a) < id(b)" in query

    def test_query_skips_existing_edges(self):
        """NOT (a)-[:SAME_AS]-(b) prevents duplicate edges."""
        writer, mock_session = self._make_writer(created_count=0)
        writer._close_same_as_transitive_neo4j()
        query = mock_session.run.call_args[0][0]
        assert "NOT (a)-[:SAME_AS]-(b)" in query

    def _make_writer(self, created_count: int = 0):
        writer = object.__new__(Writer)
        writer._neo4j_driver = MagicMock()

        mock_session = MagicMock()
        writer._neo4j_driver.session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        writer._neo4j_driver.session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        mock_result = MagicMock()
        mock_result.single.return_value = {"cnt": created_count}
        mock_session.run.return_value = mock_result

        return writer, mock_session
