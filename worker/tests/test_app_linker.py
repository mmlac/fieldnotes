"""Tests for clustering/app_linker.py — Application/Tool to Topic linking."""

from unittest.mock import MagicMock, patch

import numpy as np

from worker.clustering.app_linker import (
    AppNode,
    _build_centroid_matrix,
    _build_description_texts,
    _cosine_similarity,
    _find_matches,
    _write_edges_tx,
    link_apps_to_topics,
)
from worker.clustering.cluster import ClusterResult
from worker.clustering.labeler import LabeledCluster


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _labeled(
    cluster_id: int = 0,
    label: str = "Machine Learning",
    description: str = "Notes about ML techniques.",
) -> LabeledCluster:
    return LabeledCluster(
        cluster_id=cluster_id,
        label=label,
        description=description,
    )


def _cluster_result(
    cluster_id: int = 0,
    centroid: list[float] | None = None,
) -> ClusterResult:
    return ClusterResult(
        cluster_id=cluster_id,
        chunk_ids=["pt-1", "pt-2"],
        centroid=[1.0, 0.0, 0.0] if centroid is None else centroid,
    )


def _app_node(
    source_id: str = "app://com.docker.docker",
    label: str = "Application",
    name: str = "Docker Desktop",
    description: str = "Container runtime and management tool",
) -> AppNode:
    return AppNode(
        source_id=source_id,
        label=label,
        name=name,
        description=description,
    )


# ---------------------------------------------------------------------------
# _build_description_texts
# ---------------------------------------------------------------------------


class TestBuildDescriptionTexts:
    def test_with_description(self):
        apps = [_app_node(description="A cool tool")]
        texts = _build_description_texts(apps)
        assert texts == ["Docker Desktop: A cool tool"]

    def test_without_description(self):
        apps = [_app_node(description="")]
        texts = _build_description_texts(apps)
        assert texts == ["Docker Desktop"]

    def test_empty_list(self):
        assert _build_description_texts([]) == []


# ---------------------------------------------------------------------------
# _build_centroid_matrix
# ---------------------------------------------------------------------------


class TestBuildCentroidMatrix:
    def test_basic(self):
        labeled = [_labeled(cluster_id=0)]
        results = [_cluster_result(cluster_id=0, centroid=[1.0, 2.0, 3.0])]
        matrix, mapping = _build_centroid_matrix(labeled, results)
        assert matrix.shape == (1, 3)
        assert mapping == {0: "Machine Learning"}

    def test_missing_cluster_result(self):
        labeled = [_labeled(cluster_id=99)]
        results = [_cluster_result(cluster_id=0)]
        matrix, mapping = _build_centroid_matrix(labeled, results)
        assert matrix.shape[0] == 0
        assert mapping == {}

    def test_empty_centroid(self):
        labeled = [_labeled(cluster_id=0)]
        results = [_cluster_result(cluster_id=0, centroid=[])]
        matrix, mapping = _build_centroid_matrix(labeled, results)
        assert matrix.shape[0] == 0


# ---------------------------------------------------------------------------
# _cosine_similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self):
        a = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        b = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        sim = _cosine_similarity(a, b)
        assert sim.shape == (1, 1)
        assert abs(float(sim[0, 0]) - 1.0) < 1e-5

    def test_orthogonal_vectors(self):
        a = np.array([[1.0, 0.0]], dtype=np.float32)
        b = np.array([[0.0, 1.0]], dtype=np.float32)
        sim = _cosine_similarity(a, b)
        assert abs(float(sim[0, 0])) < 1e-5

    def test_multiple_vectors(self):
        a = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        b = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        sim = _cosine_similarity(a, b)
        assert sim.shape == (2, 2)
        assert abs(float(sim[0, 0]) - 1.0) < 1e-5
        assert abs(float(sim[1, 1]) - 1.0) < 1e-5
        assert abs(float(sim[0, 1])) < 1e-5


# ---------------------------------------------------------------------------
# _find_matches
# ---------------------------------------------------------------------------


class TestFindMatches:
    def test_above_threshold(self):
        apps = [_app_node()]
        # App vector identical to centroid → cosine sim = 1.0
        app_vectors = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        centroid_matrix = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        cluster_id_to_label = {0: "Containerization"}

        matches = _find_matches(
            apps,
            app_vectors,
            centroid_matrix,
            cluster_id_to_label,
            top_k=3,
            similarity_threshold=0.6,
        )
        assert len(matches) == 1
        assert matches[0][0] == "app://com.docker.docker"
        assert matches[0][1] == "Application"
        assert matches[0][2] == "Containerization"
        assert matches[0][3] > 0.99

    def test_below_threshold(self):
        apps = [_app_node()]
        # Orthogonal vectors → cosine sim ≈ 0
        app_vectors = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        centroid_matrix = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        cluster_id_to_label = {0: "Unrelated"}

        matches = _find_matches(
            apps,
            app_vectors,
            centroid_matrix,
            cluster_id_to_label,
            top_k=3,
            similarity_threshold=0.6,
        )
        assert len(matches) == 0

    def test_top_k_limits(self):
        apps = [_app_node()]
        # App vector similar to all 3 centroids
        app_vectors = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
        centroid_matrix = np.array(
            [
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        cluster_id_to_label = {0: "A", 1: "B", 2: "C"}

        matches = _find_matches(
            apps,
            app_vectors,
            centroid_matrix,
            cluster_id_to_label,
            top_k=2,
            similarity_threshold=0.0,
        )
        assert len(matches) == 2

    def test_empty_inputs(self):
        matches = _find_matches(
            [],
            np.empty((0, 3)),
            np.empty((0, 3)),
            {},
            top_k=3,
            similarity_threshold=0.6,
        )
        assert matches == []


# ---------------------------------------------------------------------------
# _write_edges_tx
# ---------------------------------------------------------------------------


class TestWriteEdgesTx:
    def test_deletes_old_and_creates_new(self):
        tx = MagicMock()
        single_mock = MagicMock()
        single_mock.__getitem__ = MagicMock(return_value=1)
        tx.run.return_value.single.return_value = single_mock

        matches = [
            ("app://com.docker.docker", "Application", "Containerization", 0.85),
        ]
        _write_edges_tx(tx, matches)

        # First call: delete old edges
        delete_call = tx.run.call_args_list[0]
        assert "DELETE r" in delete_call.args[0]
        assert "auto_linked" in delete_call.args[0]

        # Second call: create new edge
        create_call = tx.run.call_args_list[1]
        assert "RELATED_TO_TOPIC" in create_call.args[0]
        assert create_call.kwargs["sid"] == "app://com.docker.docker"
        assert create_call.kwargs["topic_name"] == "Containerization"

    def test_empty_matches_still_deletes(self):
        tx = MagicMock()
        _write_edges_tx(tx, [])
        # Should still delete old auto-linked edges
        assert tx.run.call_count == 1
        assert "DELETE r" in tx.run.call_args.args[0]


# ---------------------------------------------------------------------------
# link_apps_to_topics (integration with mocks)
# ---------------------------------------------------------------------------


class TestLinkAppsToTopics:
    def test_empty_labeled(self):
        result = link_apps_to_topics([], [], MagicMock())
        assert result == 0

    def test_empty_cluster_results(self):
        result = link_apps_to_topics(
            [_labeled()],
            [],
            MagicMock(),
        )
        assert result == 0

    @patch("worker.clustering.app_linker._write_edges")
    @patch("worker.clustering.app_linker._fetch_app_nodes")
    @patch("worker.clustering.app_linker.build_driver")
    def test_no_app_nodes(self, mock_gdb, mock_fetch, mock_write):
        mock_fetch.return_value = []
        mock_driver = MagicMock()
        mock_gdb.return_value = mock_driver

        result = link_apps_to_topics(
            [_labeled()],
            [_cluster_result()],
            MagicMock(),
        )
        assert result == 0
        mock_write.assert_not_called()

    @patch("worker.clustering.app_linker._write_edges")
    @patch("worker.clustering.app_linker._fetch_app_nodes")
    @patch("worker.clustering.app_linker.build_driver")
    def test_full_pipeline(self, mock_gdb, mock_fetch, mock_write):
        mock_driver = MagicMock()
        mock_gdb.return_value = mock_driver
        mock_fetch.return_value = [
            _app_node(description="Container management"),
        ]
        mock_write.return_value = 1

        # Mock embedding model
        registry = MagicMock()
        embed_model = MagicMock()
        embed_resp = MagicMock()
        # Return a vector close to the centroid [1, 0, 0]
        embed_resp.vectors = [[0.9, 0.1, 0.0]]
        embed_model.embed.return_value = embed_resp
        registry.for_role.return_value = embed_model

        labeled = [_labeled(cluster_id=0, label="Containerization")]
        clusters = [_cluster_result(cluster_id=0, centroid=[1.0, 0.0, 0.0])]

        result = link_apps_to_topics(labeled, clusters, registry)

        assert result == 1
        registry.for_role.assert_called_once_with("embed")
        mock_write.assert_called_once()

        # Check matches passed to _write_edges
        matches_arg = mock_write.call_args[0][1]
        assert len(matches_arg) == 1
        assert matches_arg[0][2] == "Containerization"

    @patch("worker.clustering.app_linker._write_edges")
    @patch("worker.clustering.app_linker._fetch_app_nodes")
    @patch("worker.clustering.app_linker.build_driver")
    def test_zero_apps_no_crash(self, mock_gdb, mock_fetch, mock_write):
        """Works with zero apps installed (no crash, no edges)."""
        mock_driver = MagicMock()
        mock_gdb.return_value = mock_driver
        mock_fetch.return_value = []

        result = link_apps_to_topics(
            [_labeled()],
            [_cluster_result()],
            MagicMock(),
        )
        assert result == 0
