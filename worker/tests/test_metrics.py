"""Tests for metrics.py — index status collector and gauge updates."""

from unittest.mock import MagicMock

from worker.metrics import (
    CHUNKS_TOTAL,
    EDGES_TOTAL,
    ENTITIES_TOTAL,
    QDRANT_COLLECTION_BYTES,
    QDRANT_POINTS_TOTAL,
    SOURCES_TOTAL,
    TOPICS_TOTAL,
    collect_index_status,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _neo4j_records(data: list[dict]) -> list[MagicMock]:
    """Build a list of mock Neo4j records from dicts."""
    records = []
    for d in data:
        rec = MagicMock()
        rec.__getitem__ = lambda self, key, _d=d: _d[key]
        records.append(rec)
    return records


def _mock_session():
    """Create a mock Neo4j session with run() that returns iterable results."""
    session = MagicMock()
    return session


def _single_record(data: dict):
    """Build a mock result whose .single() returns a dict-like record."""
    rec = MagicMock()
    rec.__getitem__ = lambda self, key, _d=data: _d[key]
    result = MagicMock()
    result.single.return_value = rec
    return result


# ---------------------------------------------------------------------------
# collect_index_status — Neo4j
# ---------------------------------------------------------------------------


class TestCollectNeo4j:
    def test_source_counts(self) -> None:
        driver = MagicMock()
        session = MagicMock()
        driver.session.return_value.__enter__ = MagicMock(return_value=session)
        driver.session.return_value.__exit__ = MagicMock(return_value=False)

        # Build results for each session.run() call
        source_records = _neo4j_records(
            [
                {"type": "File", "count": 42},
                {"type": "Email", "count": 10},
            ]
        )
        entity_count = _single_record({"count": 100.0})
        chunk_count = _single_record({"count": 500.0})
        topic_count = _single_record({"count": 5.0})
        edge_records = _neo4j_records(
            [
                {"type": "MENTIONS", "count": 200},
                {"type": "HAS_CHUNK", "count": 500},
            ]
        )
        # Store size query — raise to simulate unavailable
        store_err = MagicMock(side_effect=Exception("JMX not available"))

        session.run.side_effect = [
            source_records,  # source counts
            entity_count,  # Entity count
            chunk_count,  # Chunk count
            topic_count,  # Topic count
            edge_records,  # edge counts
            store_err,  # store size (will fail)
        ]

        qdrant = MagicMock()
        info = MagicMock()
        info.points_count = 999
        info.disk_data_size = 1024000
        qdrant.get_collection.return_value = info

        collect_index_status(driver, qdrant, "fieldnotes")

        # Verify source gauges
        assert SOURCES_TOTAL.labels(source_type="File")._value.get() == 42
        assert SOURCES_TOTAL.labels(source_type="Email")._value.get() == 10

        # Verify entity/chunk/topic gauges
        assert ENTITIES_TOTAL._value.get() == 100
        assert CHUNKS_TOTAL._value.get() == 500
        assert TOPICS_TOTAL._value.get() == 5

        # Verify edge gauges
        assert EDGES_TOTAL.labels(type="MENTIONS")._value.get() == 200
        assert EDGES_TOTAL.labels(type="HAS_CHUNK")._value.get() == 500

    def test_neo4j_failure_does_not_raise(self) -> None:
        driver = MagicMock()
        driver.session.side_effect = RuntimeError("connection failed")

        qdrant = MagicMock()
        info = MagicMock()
        info.points_count = 0
        info.disk_data_size = None
        qdrant.get_collection.return_value = info

        # Should not raise
        collect_index_status(driver, qdrant, "fieldnotes")


# ---------------------------------------------------------------------------
# collect_index_status — Qdrant
# ---------------------------------------------------------------------------


class TestCollectQdrant:
    def test_qdrant_points_and_bytes(self) -> None:
        driver = MagicMock()
        driver.session.side_effect = RuntimeError("skip neo4j")

        qdrant = MagicMock()
        info = MagicMock()
        info.points_count = 12345
        info.disk_data_size = 5_000_000
        qdrant.get_collection.return_value = info

        collect_index_status(driver, qdrant, "test_collection")

        assert QDRANT_POINTS_TOTAL._value.get() == 12345
        assert QDRANT_COLLECTION_BYTES._value.get() == 5_000_000
        qdrant.get_collection.assert_called_once_with("test_collection")

    def test_qdrant_handles_none_disk_size(self) -> None:
        driver = MagicMock()
        driver.session.side_effect = RuntimeError("skip neo4j")

        qdrant = MagicMock()
        info = MagicMock()
        info.points_count = 100
        info.disk_data_size = None
        qdrant.get_collection.return_value = info

        # Should not raise
        collect_index_status(driver, qdrant, "fieldnotes")
        assert QDRANT_POINTS_TOTAL._value.get() == 100

    def test_qdrant_failure_does_not_raise(self) -> None:
        driver = MagicMock()
        driver.session.side_effect = RuntimeError("skip neo4j")

        qdrant = MagicMock()
        qdrant.get_collection.side_effect = ConnectionError("qdrant down")

        # Should not raise
        collect_index_status(driver, qdrant, "fieldnotes")
