"""Tests for clustering/labeler.py — LLM-powered cluster labeling."""

from unittest.mock import MagicMock, patch

import pytest

from worker.clustering.cluster import ClusterResult
from worker.clustering.labeler import (
    LabeledCluster,
    _call_labeling_model,
    _deduplicate_labels,
    _get_central_chunk_texts,
    label_clusters,
)
from worker.models.base import CompletionResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cluster(
    cluster_id: int = 0,
    chunk_ids: list[str] | None = None,
    centroid: list[float] | None = None,
) -> ClusterResult:
    return ClusterResult(
        cluster_id=cluster_id,
        chunk_ids=["c1", "c2", "c3"] if chunk_ids is None else chunk_ids,
        centroid=centroid if centroid is not None else [1.0, 0.0, 0.0],
    )


def _make_qdrant_point(point_id: str, vector: list[float], text: str) -> MagicMock:
    p = MagicMock()
    p.id = point_id
    p.vector = vector
    p.payload = {"text": text, "source_id": "src-1"}
    return p


def _mock_model(label: str = "Test Topic", description: str = "A test topic.") -> MagicMock:
    model = MagicMock()
    import json
    model.complete.return_value = CompletionResponse(
        text=json.dumps({"label": label, "description": description}),
    )
    return model


# ---------------------------------------------------------------------------
# _get_central_chunk_texts
# ---------------------------------------------------------------------------

class TestGetCentralChunkTexts:
    def test_returns_texts_sorted_by_distance(self) -> None:
        centroid = [1.0, 0.0, 0.0]
        cluster = _make_cluster(
            chunk_ids=["near", "far", "mid"],
            centroid=centroid,
        )
        client = MagicMock()
        client.retrieve.return_value = [
            _make_qdrant_point("near", [1.0, 0.1, 0.0], "near text"),
            _make_qdrant_point("far", [0.0, 1.0, 0.0], "far text"),
            _make_qdrant_point("mid", [0.8, 0.2, 0.0], "mid text"),
        ]

        texts = _get_central_chunk_texts(cluster, client, "coll", top_k=2)

        assert len(texts) == 2
        assert texts[0] == "near text"
        assert texts[1] == "mid text"

    def test_respects_top_k(self) -> None:
        cluster = _make_cluster(
            chunk_ids=["a", "b", "c"],
            centroid=[0.0, 0.0],
        )
        client = MagicMock()
        client.retrieve.return_value = [
            _make_qdrant_point("a", [0.1, 0.0], "text a"),
            _make_qdrant_point("b", [0.2, 0.0], "text b"),
            _make_qdrant_point("c", [0.3, 0.0], "text c"),
        ]

        texts = _get_central_chunk_texts(cluster, client, "coll", top_k=1)

        assert len(texts) == 1
        assert texts[0] == "text a"

    def test_empty_chunk_ids(self) -> None:
        cluster = _make_cluster(chunk_ids=[])
        client = MagicMock()

        texts = _get_central_chunk_texts(cluster, client, "coll", top_k=20)

        assert texts == []
        client.retrieve.assert_not_called()

    def test_no_points_returned(self) -> None:
        cluster = _make_cluster()
        client = MagicMock()
        client.retrieve.return_value = []

        texts = _get_central_chunk_texts(cluster, client, "coll", top_k=20)

        assert texts == []

    def test_skips_empty_text(self) -> None:
        cluster = _make_cluster(
            chunk_ids=["a", "b"],
            centroid=[0.0],
        )
        client = MagicMock()
        p_empty = MagicMock()
        p_empty.id = "a"
        p_empty.vector = [0.1]
        p_empty.payload = {"text": "", "source_id": "s"}
        p_valid = _make_qdrant_point("b", [0.2], "valid text")
        client.retrieve.return_value = [p_empty, p_valid]

        texts = _get_central_chunk_texts(cluster, client, "coll", top_k=20)

        assert texts == ["valid text"]


# ---------------------------------------------------------------------------
# _call_labeling_model
# ---------------------------------------------------------------------------

class TestCallLabelingModel:
    def test_parses_valid_json(self) -> None:
        model = _mock_model("Machine Learning", "Notes about ML techniques.")

        label, desc = _call_labeling_model(model, ["chunk 1", "chunk 2"])

        assert label == "Machine Learning"
        assert desc == "Notes about ML techniques."

    def test_combines_texts_with_separator(self) -> None:
        model = _mock_model()

        _call_labeling_model(model, ["text A", "text B"])

        call_args = model.complete.call_args[0][0]
        assert "text A" in call_args.messages[0]["content"]
        assert "text B" in call_args.messages[0]["content"]
        assert "---" in call_args.messages[0]["content"]

    def test_fallback_on_invalid_json(self) -> None:
        model = MagicMock()
        model.complete.return_value = CompletionResponse(text="not json")

        label, desc = _call_labeling_model(model, ["chunk"])

        assert label == "Unknown Topic (cluster_0)"
        assert "could not be parsed" in desc

    def test_fallback_on_missing_keys(self) -> None:
        model = MagicMock()
        model.complete.return_value = CompletionResponse(
            text='{"wrong_key": "value"}'
        )

        label, desc = _call_labeling_model(model, ["chunk"])

        assert label == "Unknown Topic (cluster_0)"

    def test_uses_temperature_zero(self) -> None:
        model = _mock_model()

        _call_labeling_model(model, ["text"])

        req = model.complete.call_args[0][0]
        assert req.temperature == 0.0

    def test_warns_on_word_count_out_of_range(self, caplog) -> None:
        model = _mock_model(
            label="One",
            description="A short topic.",
        )

        label, _ = _call_labeling_model(model, ["text"])

        assert label == "One"
        assert "words" in caplog.text


# ---------------------------------------------------------------------------
# label_clusters (integration with mocks)
# ---------------------------------------------------------------------------

class TestLabelClusters:
    def test_labels_multiple_clusters(self) -> None:
        clusters = [
            _make_cluster(cluster_id=0, chunk_ids=["a", "b"]),
            _make_cluster(cluster_id=1, chunk_ids=["c", "d"]),
        ]

        with patch("worker.clustering.labeler.QdrantClient") as MockClient:
            client = MockClient.return_value
            client.retrieve.return_value = [
                _make_qdrant_point("a", [1.0, 0.0, 0.0], "chunk a"),
                _make_qdrant_point("b", [0.9, 0.1, 0.0], "chunk b"),
            ]

            registry = MagicMock()
            model = _mock_model("Topic Name", "Description here.")
            registry.for_role.return_value = model

            results = label_clusters(clusters, registry)

        assert len(results) == 2
        assert all(isinstance(r, LabeledCluster) for r in results)
        assert results[0].cluster_id == 0
        assert results[0].label == "Topic Name"
        # Second cluster gets deduplicated label since LLM returned the same name
        assert results[1].cluster_id == 1
        assert results[1].label == "Topic Name (#2)"
        registry.for_role.assert_called_once_with("cluster_label")

    def test_empty_clusters_returns_empty(self) -> None:
        registry = MagicMock()

        results = label_clusters([], registry)

        assert results == []
        registry.for_role.assert_not_called()

    def test_closes_qdrant_client(self) -> None:
        clusters = [_make_cluster()]

        with patch("worker.clustering.labeler.QdrantClient") as MockClient:
            client = MockClient.return_value
            client.retrieve.return_value = [
                _make_qdrant_point("c1", [1.0, 0.0, 0.0], "text"),
            ]

            registry = MagicMock()
            registry.for_role.return_value = _mock_model()

            label_clusters(clusters, registry)

            client.close.assert_called_once()

    def test_closes_qdrant_on_error(self) -> None:
        clusters = [_make_cluster()]

        with patch("worker.clustering.labeler.QdrantClient") as MockClient:
            client = MockClient.return_value
            client.retrieve.side_effect = RuntimeError("connection failed")

            registry = MagicMock()
            registry.for_role.return_value = _mock_model()

            with pytest.raises(RuntimeError):
                label_clusters(clusters, registry)

            client.close.assert_called_once()

    def test_fallback_label_when_no_texts(self) -> None:
        clusters = [_make_cluster(chunk_ids=["missing"])]

        with patch("worker.clustering.labeler.QdrantClient") as MockClient:
            client = MockClient.return_value
            client.retrieve.return_value = []

            registry = MagicMock()
            registry.for_role.return_value = _mock_model()

            results = label_clusters(clusters, registry)

        assert len(results) == 1
        assert results[0].label == "Unknown Topic (cluster_0)"
        assert "No chunk texts" in results[0].description


# ---------------------------------------------------------------------------
# _deduplicate_labels
# ---------------------------------------------------------------------------


class TestDeduplicateLabels:
    def test_no_duplicates_unchanged(self) -> None:
        results = [
            LabeledCluster(0, "Topic A", "desc"),
            LabeledCluster(1, "Topic B", "desc"),
        ]
        _deduplicate_labels(results)
        assert results[0].label == "Topic A"
        assert results[1].label == "Topic B"

    def test_duplicate_labels_get_suffix(self) -> None:
        results = [
            LabeledCluster(0, "Machine Learning", "desc"),
            LabeledCluster(1, "Machine Learning", "desc"),
            LabeledCluster(2, "Machine Learning", "desc"),
        ]
        _deduplicate_labels(results)
        assert results[0].label == "Machine Learning"
        assert results[1].label == "Machine Learning (#2)"
        assert results[2].label == "Machine Learning (#3)"

    def test_mixed_unique_and_duplicate(self) -> None:
        results = [
            LabeledCluster(0, "Topic A", "desc"),
            LabeledCluster(1, "Topic B", "desc"),
            LabeledCluster(2, "Topic A", "desc"),
        ]
        _deduplicate_labels(results)
        assert results[0].label == "Topic A"
        assert results[1].label == "Topic B"
        assert results[2].label == "Topic A (#2)"

    def test_empty_list(self) -> None:
        results: list[LabeledCluster] = []
        _deduplicate_labels(results)
        assert results == []
