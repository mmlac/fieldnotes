"""Tests for the vector query semantic search layer.

Uses unittest.mock to stub out Qdrant and ModelRegistry so tests run
without running services.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


from worker.query.vector import (
    VectorQuerier,
    VectorQueryResult,
    VectorResult,
    DEFAULT_TOP_K,
)


# ------------------------------------------------------------------
# VectorResult / VectorQueryResult dataclasses
# ------------------------------------------------------------------


class TestVectorResult:
    def test_fields(self) -> None:
        r = VectorResult(
            source_type="file",
            source_id="notes/march.md",
            text="some chunk",
            date="2026-03-01",
            score=0.95,
            chunk_index=2,
        )
        assert r.source_type == "file"
        assert r.source_id == "notes/march.md"
        assert r.score == 0.95
        assert r.chunk_index == 2

    def test_default_chunk_index(self) -> None:
        r = VectorResult(
            source_type="file", source_id="x", text="t", date="", score=0.5
        )
        assert r.chunk_index == 0


class TestVectorQueryResult:
    def test_defaults(self) -> None:
        r = VectorQueryResult(question="test?")
        assert r.question == "test?"
        assert r.results == []
        assert r.error is None

    def test_with_error(self) -> None:
        r = VectorQueryResult(question="q", error="connection refused")
        assert r.error == "connection refused"


# ------------------------------------------------------------------
# VectorQuerier
# ------------------------------------------------------------------


def _make_scored_point(
    score: float,
    source_type: str = "file",
    source_id: str = "doc.md",
    text: str = "chunk text",
    date: str = "2026-03-01",
    chunk_index: int = 0,
) -> MagicMock:
    """Create a mock ScoredPoint matching Qdrant's return type."""
    point = MagicMock()
    point.score = score
    point.payload = {
        "source_type": source_type,
        "source_id": source_id,
        "text": text,
        "date": date,
        "chunk_index": chunk_index,
    }
    return point


class TestVectorQuerier:
    """Tests for VectorQuerier with mocked Qdrant and registry."""

    @patch("worker.query.vector.QdrantClient")
    def test_query_returns_ranked_results(self, mock_qdrant_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_qdrant_cls.return_value = mock_client
        mock_client.search.return_value = [
            _make_scored_point(0.95, text="best match"),
            _make_scored_point(0.80, text="second match"),
        ]

        registry = MagicMock()
        resolved = MagicMock()
        embed_resp = MagicMock()
        embed_resp.vectors = [[0.1] * 768]
        resolved.embed.return_value = embed_resp
        registry.for_role.return_value = resolved

        querier = VectorQuerier(registry)
        result = querier.query("machine learning")

        assert result.question == "machine learning"
        assert result.error is None
        assert len(result.results) == 2
        assert result.results[0].score == 0.95
        assert result.results[0].text == "best match"
        assert result.results[1].score == 0.80

        registry.for_role.assert_called_with("embed")
        mock_client.search.assert_called_once()

        call_kwargs = mock_client.search.call_args
        assert call_kwargs.kwargs["collection_name"] == "fieldnotes"
        assert call_kwargs.kwargs["limit"] == DEFAULT_TOP_K
        assert call_kwargs.kwargs["query_filter"] is None

    @patch("worker.query.vector.QdrantClient")
    def test_query_with_source_type_filter(self, mock_qdrant_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_qdrant_cls.return_value = mock_client
        mock_client.search.return_value = [
            _make_scored_point(0.90, source_type="email"),
        ]

        registry = MagicMock()
        resolved = MagicMock()
        embed_resp = MagicMock()
        embed_resp.vectors = [[0.1] * 768]
        resolved.embed.return_value = embed_resp
        registry.for_role.return_value = resolved

        querier = VectorQuerier(registry)
        result = querier.query("emails about project", source_type="email")

        assert len(result.results) == 1
        assert result.results[0].source_type == "email"

        call_kwargs = mock_client.search.call_args
        query_filter = call_kwargs.kwargs["query_filter"]
        assert query_filter is not None
        assert len(query_filter.must) == 1
        assert query_filter.must[0].key == "source_type"

    @patch("worker.query.vector.QdrantClient")
    def test_query_custom_top_k(self, mock_qdrant_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_qdrant_cls.return_value = mock_client
        mock_client.search.return_value = []

        registry = MagicMock()
        resolved = MagicMock()
        embed_resp = MagicMock()
        embed_resp.vectors = [[0.1] * 768]
        resolved.embed.return_value = embed_resp
        registry.for_role.return_value = resolved

        querier = VectorQuerier(registry)
        result = querier.query("test", top_k=5)

        assert result.results == []
        call_kwargs = mock_client.search.call_args
        assert call_kwargs.kwargs["limit"] == 5

    @patch("worker.query.vector.QdrantClient")
    def test_query_handles_exception(self, mock_qdrant_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_qdrant_cls.return_value = mock_client
        mock_client.search.side_effect = RuntimeError("qdrant down")

        registry = MagicMock()
        resolved = MagicMock()
        embed_resp = MagicMock()
        embed_resp.vectors = [[0.1] * 768]
        resolved.embed.return_value = embed_resp
        registry.for_role.return_value = resolved

        querier = VectorQuerier(registry)
        result = querier.query("test?")

        assert result.error == "qdrant down"
        assert result.results == []

    @patch("worker.query.vector.QdrantClient")
    def test_query_handles_embed_failure(self, mock_qdrant_cls: MagicMock) -> None:
        mock_qdrant_cls.return_value = MagicMock()

        registry = MagicMock()
        registry.for_role.side_effect = KeyError("embed")

        querier = VectorQuerier(registry)
        result = querier.query("test?")

        assert result.error is not None
        assert "embed" in result.error

    @patch("worker.query.vector.QdrantClient")
    def test_close(self, mock_qdrant_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_qdrant_cls.return_value = mock_client

        registry = MagicMock()
        querier = VectorQuerier(registry)
        querier.close()

        mock_client.close.assert_called_once()
        assert querier._qdrant is None

    @patch("worker.query.vector.QdrantClient")
    def test_payload_mapping(self, mock_qdrant_cls: MagicMock) -> None:
        """Verify all payload fields are correctly mapped to VectorResult."""
        mock_client = MagicMock()
        mock_qdrant_cls.return_value = mock_client
        mock_client.search.return_value = [
            _make_scored_point(
                0.88,
                source_type="email",
                source_id="inbox/msg42",
                text="important meeting notes",
                date="2026-03-11",
                chunk_index=3,
            ),
        ]

        registry = MagicMock()
        resolved = MagicMock()
        embed_resp = MagicMock()
        embed_resp.vectors = [[0.1] * 768]
        resolved.embed.return_value = embed_resp
        registry.for_role.return_value = resolved

        querier = VectorQuerier(registry)
        result = querier.query("meetings")

        r = result.results[0]
        assert r.source_type == "email"
        assert r.source_id == "inbox/msg42"
        assert r.text == "important meeting notes"
        assert r.date == "2026-03-11"
        assert r.score == 0.88
        assert r.chunk_index == 3
