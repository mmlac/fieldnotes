"""Tests for clustering/cluster.py — UMAP + HDBSCAN clustering core."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from worker.clustering.cluster import (
    ClusterResult,
    CorpusTooSmallError,
    DEFAULT_HDBSCAN_METRIC,
    DEFAULT_MIN_CLUSTER_SIZE,
    DEFAULT_MIN_CORPUS_SIZE,
    DEFAULT_UMAP_DIMS,
    _build_results,
    _hdbscan_cluster,
    _scroll_all_vectors,
    _umap_reduce,
    cluster_embeddings,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_blobs(
    n_points: int = 60,
    n_clusters: int = 3,
    dim: int = 768,
    rng_seed: int = 0,
) -> tuple[list[str], np.ndarray]:
    """Generate synthetic embedding blobs for testing.

    Returns (ids, matrix) where matrix has clearly separable clusters.
    """
    rng = np.random.default_rng(rng_seed)
    points_per = n_points // n_clusters
    ids: list[str] = []
    parts: list[np.ndarray] = []

    for c in range(n_clusters):
        center = rng.standard_normal(dim).astype(np.float32) * 10
        blob = center + rng.standard_normal((points_per, dim)).astype(np.float32) * 0.1
        parts.append(blob)
        ids.extend(f"id-{c}-{i}" for i in range(points_per))

    matrix = np.vstack(parts)
    return ids, matrix


def _make_scroll_point(point_id: str, vector: list[float]) -> MagicMock:
    """Create a mock Qdrant ScoredPoint."""
    p = MagicMock()
    p.id = point_id
    p.vector = vector
    return p


# ---------------------------------------------------------------------------
# _scroll_all_vectors
# ---------------------------------------------------------------------------

class TestScrollAllVectors:
    def test_scrolls_single_page(self) -> None:
        client = MagicMock()
        points = [_make_scroll_point("a", [1.0, 2.0])]
        client.scroll.return_value = (points, None)

        ids, vecs = _scroll_all_vectors(client, "test-coll")

        assert ids == ["a"]
        assert vecs == [[1.0, 2.0]]
        client.scroll.assert_called_once()

    def test_scrolls_multiple_pages(self) -> None:
        client = MagicMock()
        page1 = [_make_scroll_point("a", [1.0])]
        page2 = [_make_scroll_point("b", [2.0])]
        client.scroll.side_effect = [
            (page1, "next-offset"),
            (page2, None),
        ]

        ids, vecs = _scroll_all_vectors(client, "coll")

        assert ids == ["a", "b"]
        assert vecs == [[1.0], [2.0]]
        assert client.scroll.call_count == 2

    def test_empty_collection(self) -> None:
        client = MagicMock()
        client.scroll.return_value = ([], None)

        ids, vecs = _scroll_all_vectors(client, "empty")

        assert ids == []
        assert vecs == []


# ---------------------------------------------------------------------------
# _umap_reduce
# ---------------------------------------------------------------------------

class TestUmapReduce:
    def test_reduces_dimensions(self) -> None:
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((50, 768)).astype(np.float32)

        reduced = _umap_reduce(matrix, n_components=32, n_neighbors=15, metric="cosine")

        assert reduced.shape == (50, 32)

    def test_preserves_row_count(self) -> None:
        rng = np.random.default_rng(0)
        matrix = rng.standard_normal((30, 100)).astype(np.float32)

        reduced = _umap_reduce(matrix, n_components=5, n_neighbors=10, metric="euclidean")

        assert reduced.shape[0] == 30


# ---------------------------------------------------------------------------
# _hdbscan_cluster
# ---------------------------------------------------------------------------

class TestHdbscanCluster:
    def test_returns_label_array(self) -> None:
        _, matrix = _make_blobs(n_points=60, n_clusters=3, dim=32)

        labels = _hdbscan_cluster(matrix, min_cluster_size=10, metric="euclidean")

        assert isinstance(labels, np.ndarray)
        assert len(labels) == 60

    def test_finds_separable_clusters(self) -> None:
        _, matrix = _make_blobs(n_points=90, n_clusters=3, dim=32)

        labels = _hdbscan_cluster(matrix, min_cluster_size=10, metric="euclidean")

        # Should find at least 2 clusters in well-separated data
        real_labels = set(labels) - {-1}
        assert len(real_labels) >= 2


# ---------------------------------------------------------------------------
# _build_results
# ---------------------------------------------------------------------------

class TestBuildResults:
    def test_excludes_noise(self) -> None:
        ids = ["a", "b", "c"]
        matrix = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        labels = np.array([-1, 0, 0])

        results = _build_results(ids, matrix, labels)

        assert len(results) == 1
        assert results[0].cluster_id == 0
        assert results[0].chunk_ids == ["b", "c"]

    def test_all_noise_returns_empty(self) -> None:
        ids = ["a", "b"]
        matrix = np.array([[1.0], [2.0]])
        labels = np.array([-1, -1])

        results = _build_results(ids, matrix, labels)

        assert results == []

    def test_centroid_is_mean_of_original(self) -> None:
        ids = ["a", "b"]
        matrix = np.array([[2.0, 4.0], [4.0, 6.0]])
        labels = np.array([0, 0])

        results = _build_results(ids, matrix, labels)

        assert len(results) == 1
        np.testing.assert_allclose(results[0].centroid, [3.0, 5.0])

    def test_multiple_clusters(self) -> None:
        ids = ["a", "b", "c", "d"]
        matrix = np.array([[1.0], [2.0], [10.0], [11.0]])
        labels = np.array([0, 0, 1, 1])

        results = _build_results(ids, matrix, labels)

        assert len(results) == 2
        assert results[0].chunk_ids == ["a", "b"]
        assert results[1].chunk_ids == ["c", "d"]

    def test_single_cluster(self) -> None:
        ids = ["x", "y", "z"]
        matrix = np.array([[1.0], [2.0], [3.0]])
        labels = np.array([0, 0, 0])

        results = _build_results(ids, matrix, labels)

        assert len(results) == 1
        assert results[0].chunk_ids == ["x", "y", "z"]


# ---------------------------------------------------------------------------
# cluster_embeddings (integration with mocked Qdrant)
# ---------------------------------------------------------------------------

class TestClusterEmbeddings:
    def test_corpus_too_small_raises(self) -> None:
        with patch("worker.clustering.cluster.QdrantClient") as MockClient:
            client = MockClient.return_value
            points = [_make_scroll_point(f"id-{i}", [float(i)]) for i in range(5)]
            client.scroll.return_value = (points, None)

            with pytest.raises(CorpusTooSmallError, match="5 embeddings"):
                cluster_embeddings(min_corpus_size=10)

    def test_end_to_end_with_synthetic_blobs(self) -> None:
        """Full pipeline with mocked Qdrant returning separable blobs."""
        ids, matrix = _make_blobs(n_points=90, n_clusters=3, dim=768)

        with patch("worker.clustering.cluster.QdrantClient") as MockClient:
            client = MockClient.return_value
            # Return all points in one scroll page
            points = [
                _make_scroll_point(ids[i], matrix[i].tolist())
                for i in range(len(ids))
            ]
            client.scroll.return_value = (points, None)

            results = cluster_embeddings(
                min_corpus_size=20,
                min_cluster_size=10,
            )

        # Should find clusters (well-separated blobs)
        assert len(results) >= 2
        for r in results:
            assert isinstance(r, ClusterResult)
            assert len(r.chunk_ids) > 0
            assert len(r.centroid) == 768

    def test_all_noise_returns_empty_list(self) -> None:
        """When HDBSCAN labels everything as noise, return empty list."""
        n = 30
        with (
            patch("worker.clustering.cluster.QdrantClient") as MockClient,
            patch("worker.clustering.cluster._hdbscan_cluster") as mock_hdb,
            patch("worker.clustering.cluster._umap_reduce") as mock_umap,
        ):
            client = MockClient.return_value
            rng = np.random.default_rng(99)
            matrix = rng.standard_normal((n, 768)).astype(np.float32)
            points = [
                _make_scroll_point(f"id-{i}", matrix[i].tolist())
                for i in range(n)
            ]
            client.scroll.return_value = (points, None)
            mock_umap.return_value = rng.standard_normal((n, 32)).astype(np.float32)
            mock_hdb.return_value = np.full(n, -1)

            results = cluster_embeddings(min_corpus_size=20)

        assert results == []

    def test_closes_qdrant_client(self) -> None:
        """Qdrant client is closed even if an error occurs."""
        with patch("worker.clustering.cluster.QdrantClient") as MockClient:
            client = MockClient.return_value
            client.scroll.side_effect = RuntimeError("connection failed")

            with pytest.raises(RuntimeError):
                cluster_embeddings()

            client.close.assert_called_once()
