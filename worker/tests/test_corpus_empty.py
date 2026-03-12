"""Tests for the corpus emptiness check."""

from __future__ import annotations

from unittest.mock import MagicMock

from worker.query import is_corpus_empty


class TestIsCorpusEmpty:
    def test_both_empty(self) -> None:
        """Both Qdrant and Neo4j report zero → corpus is empty."""
        gq = MagicMock()
        vq = MagicMock()

        # Qdrant returns 0 points.
        vq._qdrant.get_collection.return_value.points_count = 0

        # Neo4j returns 0 nodes.
        mock_row = {"cnt": 0}
        session_ctx = MagicMock()
        session_ctx.run.return_value.single.return_value = mock_row
        gq._driver.session.return_value.__enter__ = MagicMock(return_value=session_ctx)
        gq._driver.session.return_value.__exit__ = MagicMock(return_value=False)

        assert is_corpus_empty(gq, vq) is True

    def test_qdrant_has_data(self) -> None:
        """Qdrant has points → corpus not empty."""
        gq = MagicMock()
        vq = MagicMock()

        vq._qdrant.get_collection.return_value.points_count = 42

        mock_row = {"cnt": 0}
        session_ctx = MagicMock()
        session_ctx.run.return_value.single.return_value = mock_row
        gq._driver.session.return_value.__enter__ = MagicMock(return_value=session_ctx)
        gq._driver.session.return_value.__exit__ = MagicMock(return_value=False)

        assert is_corpus_empty(gq, vq) is False

    def test_neo4j_has_data(self) -> None:
        """Neo4j has source nodes → corpus not empty."""
        gq = MagicMock()
        vq = MagicMock()

        vq._qdrant.get_collection.return_value.points_count = 0

        mock_row = {"cnt": 5}
        session_ctx = MagicMock()
        session_ctx.run.return_value.single.return_value = mock_row
        gq._driver.session.return_value.__enter__ = MagicMock(return_value=session_ctx)
        gq._driver.session.return_value.__exit__ = MagicMock(return_value=False)

        assert is_corpus_empty(gq, vq) is False

    def test_qdrant_connection_error(self) -> None:
        """Qdrant unreachable, Neo4j empty → still considers corpus empty."""
        gq = MagicMock()
        vq = MagicMock()

        vq._qdrant.get_collection.side_effect = ConnectionError("unreachable")

        mock_row = {"cnt": 0}
        session_ctx = MagicMock()
        session_ctx.run.return_value.single.return_value = mock_row
        gq._driver.session.return_value.__enter__ = MagicMock(return_value=session_ctx)
        gq._driver.session.return_value.__exit__ = MagicMock(return_value=False)

        assert is_corpus_empty(gq, vq) is True

    def test_both_unreachable(self) -> None:
        """Both stores unreachable → returns False (don't falsely claim empty)."""
        gq = MagicMock()
        vq = MagicMock()

        vq._qdrant.get_collection.side_effect = ConnectionError("unreachable")
        gq._driver.session.side_effect = ConnectionError("unreachable")

        assert is_corpus_empty(gq, vq) is False
