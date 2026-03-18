"""Tests for clustering/writer.py — Topic node writer with TAGGED edges."""

from unittest.mock import MagicMock, patch

import pytest

from worker.clustering.cluster import ClusterResult
from worker.clustering.labeler import LabeledCluster
from worker.clustering.writer import (
    _create_tagged_edge,
    _create_tagged_edges_batch,
    _delete_cluster_tagged_edges,
    _delete_orphaned_cluster_topics,
    _resolve_chunk_sources,
    _upsert_topic_node,
    _upsert_topic_nodes_batch,
    _write_tx,
    write_clusters,
)


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
    chunk_ids: list[str] | None = None,
) -> ClusterResult:
    return ClusterResult(
        cluster_id=cluster_id,
        chunk_ids=chunk_ids or ["pt-1", "pt-2"],
        centroid=[1.0, 0.0],
    )


def _make_qdrant_point(point_id: str, source_id: str) -> MagicMock:
    p = MagicMock()
    p.id = point_id
    p.payload = {"source_id": source_id, "text": "some text"}
    return p


# ---------------------------------------------------------------------------
# _delete_cluster_tagged_edges
# ---------------------------------------------------------------------------


class TestDeleteClusterTaggedEdges:
    def test_deletes_cluster_tagged_edges(self) -> None:
        tx = MagicMock()
        _delete_cluster_tagged_edges(tx)

        tx.run.assert_called_once()
        query = tx.run.call_args[0][0]
        assert "TAGGED" in query
        assert "source: 'cluster'" in query
        assert "DELETE r" in query

    def test_does_not_reference_tagged_by_user(self) -> None:
        tx = MagicMock()
        _delete_cluster_tagged_edges(tx)

        query = tx.run.call_args[0][0]
        assert "TAGGED_BY_USER" not in query


# ---------------------------------------------------------------------------
# _delete_orphaned_cluster_topics
# ---------------------------------------------------------------------------


class TestDeleteOrphanedClusterTopics:
    def test_deletes_orphaned_topics(self) -> None:
        tx = MagicMock()
        _delete_orphaned_cluster_topics(tx)

        tx.run.assert_called_once()
        query = tx.run.call_args[0][0]
        assert "Topic" in query
        assert "source: 'cluster'" in query
        assert "NOT EXISTS" in query
        assert "DELETE t" in query


# ---------------------------------------------------------------------------
# _upsert_topic_node
# ---------------------------------------------------------------------------


class TestUpsertTopicNode:
    def test_creates_topic_with_correct_params(self) -> None:
        tx = MagicMock()
        cluster = _labeled(label="Data Science", description="Data science notes.")
        _upsert_topic_node(tx, cluster)

        tx.run.assert_called_once()
        query = tx.run.call_args[0][0]
        kwargs = tx.run.call_args[1]
        assert "MERGE" in query
        assert "Topic" in query
        assert kwargs["name"] == "Data Science"
        assert kwargs["description"] == "Data science notes."

    def test_sets_source_cluster(self) -> None:
        tx = MagicMock()
        _upsert_topic_node(tx, _labeled())

        query = tx.run.call_args[0][0]
        assert "source: 'cluster'" in query


# ---------------------------------------------------------------------------
# _create_tagged_edge
# ---------------------------------------------------------------------------


class TestCreateTaggedEdge:
    def test_creates_edge_with_correct_params(self) -> None:
        tx = MagicMock()
        _create_tagged_edge(tx, "notes/a.md", "Machine Learning")

        tx.run.assert_called_once()
        kwargs = tx.run.call_args[1]
        assert kwargs["sid"] == "notes/a.md"
        assert kwargs["name"] == "Machine Learning"

    def test_edge_has_source_cluster(self) -> None:
        tx = MagicMock()
        _create_tagged_edge(tx, "notes/a.md", "ML")

        query = tx.run.call_args[0][0]
        assert "TAGGED" in query
        assert "source: 'cluster'" in query

    def test_uses_merge_for_idempotency(self) -> None:
        tx = MagicMock()
        _create_tagged_edge(tx, "notes/a.md", "ML")

        query = tx.run.call_args[0][0]
        assert "MERGE" in query

    def test_constrains_source_node_labels(self) -> None:
        tx = MagicMock()
        _create_tagged_edge(tx, "notes/a.md", "ML")

        query = tx.run.call_args[0][0]
        assert "s:File" in query
        assert "s:Email" in query
        assert "s:Commit" in query
        assert "s:Image" in query


# ---------------------------------------------------------------------------
# _write_tx (transaction body)
# ---------------------------------------------------------------------------


class TestWriteTx:
    def test_full_transaction_sequence(self) -> None:
        tx = MagicMock()
        clusters = [
            _labeled(cluster_id=0, label="Topic A", description="Desc A"),
            _labeled(cluster_id=1, label="Topic B", description="Desc B"),
        ]
        source_map = {
            0: {"notes/a.md", "notes/b.md"},
            1: {"email/c@d.com"},
        }

        _write_tx(tx, clusters, source_map)

        # 1 delete edges + 1 delete orphans + 1 batch topic upsert + 1 batch tagged edges = 4 calls
        assert tx.run.call_count == 4

    def test_deletes_before_creates(self) -> None:
        tx = MagicMock()
        clusters = [_labeled(cluster_id=0)]
        source_map = {0: {"notes/a.md"}}

        _write_tx(tx, clusters, source_map)

        queries = [c[0][0] for c in tx.run.call_args_list]
        # First two calls should be deletes
        assert "DELETE r" in queries[0]
        assert "DELETE t" in queries[1]
        # Then batch topic upsert
        assert "MERGE" in queries[2]

    def test_handles_empty_source_map(self) -> None:
        tx = MagicMock()
        clusters = [_labeled(cluster_id=0)]
        source_map = {0: set()}

        _write_tx(tx, clusters, source_map)

        # 1 delete edges + 1 delete orphans + 1 batch topic upsert = 3 calls
        # (no tagged edges batch since there are no source_ids)
        assert tx.run.call_count == 3

    def test_handles_missing_cluster_in_source_map(self) -> None:
        tx = MagicMock()
        clusters = [_labeled(cluster_id=99)]
        source_map = {}

        _write_tx(tx, clusters, source_map)

        # 1 delete edges + 1 delete orphans + 1 batch topic upsert = 3 calls
        assert tx.run.call_count == 3


# ---------------------------------------------------------------------------
# _resolve_chunk_sources
# ---------------------------------------------------------------------------


class TestResolveChunkSources:
    def test_resolves_source_ids_from_qdrant(self) -> None:
        chunk_ids_by_cluster = {
            0: ["pt-1", "pt-2"],
            1: ["pt-3"],
        }

        with patch("worker.clustering.writer.QdrantClient") as MockClient:
            client = MockClient.return_value
            client.retrieve.return_value = [
                _make_qdrant_point("pt-1", "notes/a.md"),
                _make_qdrant_point("pt-2", "notes/a.md"),
                _make_qdrant_point("pt-3", "notes/b.md"),
            ]

            from worker.config import QdrantConfig

            result = _resolve_chunk_sources(chunk_ids_by_cluster, QdrantConfig())

        assert result[0] == {"notes/a.md"}
        assert result[1] == {"notes/b.md"}

    def test_deduplicates_source_ids(self) -> None:
        chunk_ids_by_cluster = {0: ["pt-1", "pt-2", "pt-3"]}

        with patch("worker.clustering.writer.QdrantClient") as MockClient:
            client = MockClient.return_value
            client.retrieve.return_value = [
                _make_qdrant_point("pt-1", "notes/a.md"),
                _make_qdrant_point("pt-2", "notes/a.md"),
                _make_qdrant_point("pt-3", "notes/b.md"),
            ]

            from worker.config import QdrantConfig

            result = _resolve_chunk_sources(chunk_ids_by_cluster, QdrantConfig())

        assert result[0] == {"notes/a.md", "notes/b.md"}

    def test_empty_chunk_ids(self) -> None:
        chunk_ids_by_cluster = {0: []}

        with patch("worker.clustering.writer.QdrantClient") as MockClient:
            client = MockClient.return_value

            from worker.config import QdrantConfig

            result = _resolve_chunk_sources(chunk_ids_by_cluster, QdrantConfig())

        assert result[0] == set()
        client.retrieve.assert_not_called()

    def test_closes_qdrant_client(self) -> None:
        chunk_ids_by_cluster = {0: ["pt-1"]}

        with patch("worker.clustering.writer.QdrantClient") as MockClient:
            client = MockClient.return_value
            client.retrieve.return_value = [
                _make_qdrant_point("pt-1", "notes/a.md"),
            ]

            from worker.config import QdrantConfig

            _resolve_chunk_sources(chunk_ids_by_cluster, QdrantConfig())

            client.close.assert_called_once()

    def test_closes_qdrant_on_error(self) -> None:
        chunk_ids_by_cluster = {0: ["pt-1"]}

        with patch("worker.clustering.writer.QdrantClient") as MockClient:
            client = MockClient.return_value
            client.retrieve.side_effect = RuntimeError("connection failed")

            from worker.config import QdrantConfig

            with pytest.raises(RuntimeError):
                _resolve_chunk_sources(chunk_ids_by_cluster, QdrantConfig())

            client.close.assert_called_once()

    def test_skips_points_without_source_id(self) -> None:
        chunk_ids_by_cluster = {0: ["pt-1", "pt-2"]}

        with patch("worker.clustering.writer.QdrantClient") as MockClient:
            client = MockClient.return_value
            p_no_payload = MagicMock()
            p_no_payload.id = "pt-1"
            p_no_payload.payload = None
            client.retrieve.return_value = [
                p_no_payload,
                _make_qdrant_point("pt-2", "notes/a.md"),
            ]

            from worker.config import QdrantConfig

            result = _resolve_chunk_sources(chunk_ids_by_cluster, QdrantConfig())

        assert result[0] == {"notes/a.md"}


# ---------------------------------------------------------------------------
# write_clusters (integration with mocks)
# ---------------------------------------------------------------------------


class TestWriteClusters:
    def test_writes_topics_and_edges(self) -> None:
        labeled = [_labeled(cluster_id=0, label="ML", description="Machine learning")]
        results = [_cluster_result(cluster_id=0, chunk_ids=["pt-1"])]

        with (
            patch("worker.clustering.writer.QdrantClient") as MockQdrant,
            patch("worker.clustering.writer.GraphDatabase") as MockGDB,
        ):
            qdrant = MockQdrant.return_value
            qdrant.retrieve.return_value = [
                _make_qdrant_point("pt-1", "notes/a.md"),
            ]

            driver = MagicMock()
            MockGDB.driver.return_value = driver
            session = MagicMock()
            driver.session.return_value.__enter__ = MagicMock(return_value=session)
            driver.session.return_value.__exit__ = MagicMock(return_value=False)

            write_clusters(labeled, results)

            session.execute_write.assert_called_once()

    def test_empty_clusters_skips_all(self) -> None:
        with (
            patch("worker.clustering.writer.QdrantClient") as MockQdrant,
            patch("worker.clustering.writer.GraphDatabase") as MockGDB,
        ):
            write_clusters([], [])

            MockQdrant.assert_not_called()
            MockGDB.driver.assert_not_called()

    def test_closes_neo4j_driver(self) -> None:
        labeled = [_labeled(cluster_id=0)]
        results = [_cluster_result(cluster_id=0)]

        with (
            patch("worker.clustering.writer.QdrantClient") as MockQdrant,
            patch("worker.clustering.writer.GraphDatabase") as MockGDB,
        ):
            qdrant = MockQdrant.return_value
            qdrant.retrieve.return_value = [
                _make_qdrant_point("pt-1", "notes/a.md"),
            ]

            driver = MagicMock()
            MockGDB.driver.return_value = driver
            session = MagicMock()
            driver.session.return_value.__enter__ = MagicMock(return_value=session)
            driver.session.return_value.__exit__ = MagicMock(return_value=False)

            write_clusters(labeled, results)

            driver.close.assert_called_once()

    def test_closes_neo4j_on_error(self) -> None:
        labeled = [_labeled(cluster_id=0)]
        results = [_cluster_result(cluster_id=0)]

        with (
            patch("worker.clustering.writer.QdrantClient") as MockQdrant,
            patch("worker.clustering.writer.GraphDatabase") as MockGDB,
        ):
            qdrant = MockQdrant.return_value
            qdrant.retrieve.return_value = [
                _make_qdrant_point("pt-1", "notes/a.md"),
            ]

            driver = MagicMock()
            MockGDB.driver.return_value = driver
            driver.session.side_effect = RuntimeError("connection failed")

            with pytest.raises(RuntimeError):
                write_clusters(labeled, results)

            driver.close.assert_called_once()

    def test_multiple_clusters_with_overlapping_sources(self) -> None:
        labeled = [
            _labeled(cluster_id=0, label="ML"),
            _labeled(cluster_id=1, label="Cooking"),
        ]
        results = [
            _cluster_result(cluster_id=0, chunk_ids=["pt-1", "pt-2"]),
            _cluster_result(cluster_id=1, chunk_ids=["pt-2", "pt-3"]),
        ]

        with (
            patch("worker.clustering.writer.QdrantClient") as MockQdrant,
            patch("worker.clustering.writer.GraphDatabase") as MockGDB,
        ):
            qdrant = MockQdrant.return_value
            qdrant.retrieve.return_value = [
                _make_qdrant_point("pt-1", "notes/a.md"),
                _make_qdrant_point("pt-2", "notes/b.md"),
                _make_qdrant_point("pt-3", "notes/c.md"),
            ]

            driver = MagicMock()
            MockGDB.driver.return_value = driver
            session = MagicMock()
            driver.session.return_value.__enter__ = MagicMock(return_value=session)
            driver.session.return_value.__exit__ = MagicMock(return_value=False)

            write_clusters(labeled, results)

            session.execute_write.assert_called_once()
            # Verify the tx function was called with correct source_map
            tx_args = session.execute_write.call_args
            source_map = tx_args[0][2]
            assert source_map[0] == {"notes/a.md", "notes/b.md"}
            assert source_map[1] == {"notes/b.md", "notes/c.md"}


# ---------------------------------------------------------------------------
# _upsert_topic_nodes_batch
# ---------------------------------------------------------------------------


class TestUpsertTopicNodesBatch:
    def test_single_call_for_multiple_clusters(self) -> None:
        tx = MagicMock()
        clusters = [
            _labeled(cluster_id=0, label="ML", description="Machine learning"),
            _labeled(cluster_id=1, label="Cooking", description="Cooking notes"),
        ]
        _upsert_topic_nodes_batch(tx, clusters)

        tx.run.assert_called_once()
        query = tx.run.call_args[0][0]
        kwargs = tx.run.call_args[1]
        assert "UNWIND" in query
        assert "MERGE" in query
        assert "Topic" in query
        assert "source: 'cluster'" in query
        topics = kwargs["topics"]
        assert len(topics) == 2
        assert topics[0]["name"] == "ML"
        assert topics[1]["name"] == "Cooking"

    def test_no_op_for_empty_clusters(self) -> None:
        tx = MagicMock()
        _upsert_topic_nodes_batch(tx, [])
        tx.run.assert_not_called()


# ---------------------------------------------------------------------------
# _create_tagged_edges_batch
# ---------------------------------------------------------------------------


class TestCreateTaggedEdgesBatch:
    def test_single_call_for_multiple_pairs(self) -> None:
        tx = MagicMock()
        pairs = [
            {"sid": "notes/a.md", "name": "ML"},
            {"sid": "notes/b.md", "name": "ML"},
            {"sid": "email/x@y.com", "name": "Cooking"},
        ]
        _create_tagged_edges_batch(tx, pairs)

        tx.run.assert_called_once()
        query = tx.run.call_args[0][0]
        kwargs = tx.run.call_args[1]
        assert "UNWIND" in query
        assert "TAGGED" in query
        assert "source: 'cluster'" in query
        assert "MERGE" in query
        assert kwargs["pairs"] == pairs

    def test_constrains_source_node_labels(self) -> None:
        tx = MagicMock()
        _create_tagged_edges_batch(tx, [{"sid": "f.md", "name": "T"}])
        query = tx.run.call_args[0][0]
        assert "s:File" in query
        assert "s:Email" in query
        assert "s:Commit" in query
        assert "s:Image" in query
