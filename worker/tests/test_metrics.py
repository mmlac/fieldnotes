"""Tests for metrics.py — index status collector and gauge updates."""

from unittest.mock import MagicMock

from worker.metrics import (
    CHUNKS_TOTAL,
    EDGES_TOTAL,
    ENTITIES_TOTAL,
    NEO4J_STORE_BYTES,
    QDRANT_POINTS_TOTAL,
    QDRANT_STORE_BYTES,
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

        session.run.side_effect = [
            source_records,  # source counts
            entity_count,  # Entity count
            chunk_count,  # Chunk count
            topic_count,  # Topic count
            edge_records,  # edge counts
        ]

        qdrant = MagicMock()
        info = MagicMock()
        info.points_count = 999
        qdrant.get_collection.return_value = info

        collect_index_status(driver, qdrant, "fieldnotes", data_dir="/nonexistent")

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
        qdrant.get_collection.return_value = info

        # Should not raise
        collect_index_status(driver, qdrant, "fieldnotes", data_dir="/nonexistent")


# ---------------------------------------------------------------------------
# collect_index_status — Qdrant
# ---------------------------------------------------------------------------


class TestCollectQdrant:
    def test_qdrant_points(self) -> None:
        driver = MagicMock()
        driver.session.side_effect = RuntimeError("skip neo4j")

        qdrant = MagicMock()
        info = MagicMock()
        info.points_count = 12345
        qdrant.get_collection.return_value = info

        collect_index_status(driver, qdrant, "test_collection", data_dir="/nonexistent")

        assert QDRANT_POINTS_TOTAL._value.get() == 12345
        qdrant.get_collection.assert_called_once_with("test_collection")

    def test_qdrant_failure_does_not_raise(self) -> None:
        driver = MagicMock()
        driver.session.side_effect = RuntimeError("skip neo4j")

        qdrant = MagicMock()
        info = MagicMock()
        info.points_count = 100
        qdrant.get_collection.return_value = info

        # Should not raise
        collect_index_status(driver, qdrant, "fieldnotes", data_dir="/nonexistent")
        assert QDRANT_POINTS_TOTAL._value.get() == 100

    def test_qdrant_failure_does_not_raise(self) -> None:
        driver = MagicMock()
        driver.session.side_effect = RuntimeError("skip neo4j")

        qdrant = MagicMock()
        qdrant.get_collection.side_effect = ConnectionError("qdrant down")

        # Should not raise
        collect_index_status(driver, qdrant, "fieldnotes", data_dir="/nonexistent")


# ---------------------------------------------------------------------------
# collect_index_status — Store sizes (filesystem)
# ---------------------------------------------------------------------------


class TestCollectStoreSizes:
    def test_store_sizes_from_filesystem(self, tmp_path) -> None:
        # Create fake data dirs with known file sizes
        neo4j_dir = tmp_path / "neo4j"
        neo4j_dir.mkdir()
        (neo4j_dir / "store.db").write_bytes(b"x" * 1024)
        (neo4j_dir / "sub").mkdir()
        (neo4j_dir / "sub" / "index.db").write_bytes(b"y" * 512)

        qdrant_dir = tmp_path / "qdrant"
        qdrant_dir.mkdir()
        (qdrant_dir / "collection.bin").write_bytes(b"z" * 2048)

        driver = MagicMock()
        driver.session.side_effect = RuntimeError("skip")
        qdrant = MagicMock()
        qdrant.get_collection.side_effect = RuntimeError("skip")

        collect_index_status(driver, qdrant, "fieldnotes", data_dir=str(tmp_path))

        assert NEO4J_STORE_BYTES._value.get() == 1024 + 512
        assert QDRANT_STORE_BYTES._value.get() == 2048

    def test_missing_data_dir_does_not_raise(self) -> None:
        driver = MagicMock()
        driver.session.side_effect = RuntimeError("skip")
        qdrant = MagicMock()
        qdrant.get_collection.side_effect = RuntimeError("skip")

        # Should not raise
        collect_index_status(driver, qdrant, "fieldnotes", data_dir="/nonexistent")


# ---------------------------------------------------------------------------
# Initial sync progress helpers
# ---------------------------------------------------------------------------


class TestInitialSyncTracking:
    """Tests for initial_sync_add_items / initial_sync_get_total helpers."""

    def setup_method(self) -> None:
        import worker.metrics as m

        m._initial_sync_total = 0
        m.INITIAL_SYNC_ITEMS_TOTAL.set(0)

    def test_add_items_increments_total(self) -> None:
        from worker.metrics import (
            INITIAL_SYNC_ITEMS_TOTAL,
            initial_sync_add_items,
            initial_sync_get_total,
        )

        initial_sync_add_items(10)
        assert initial_sync_get_total() == 10
        assert INITIAL_SYNC_ITEMS_TOTAL._value.get() == 10

        initial_sync_add_items(5)
        assert initial_sync_get_total() == 15
        assert INITIAL_SYNC_ITEMS_TOTAL._value.get() == 15

    def test_zero_items_is_noop(self) -> None:
        from worker.metrics import initial_sync_add_items, initial_sync_get_total

        initial_sync_add_items(0)
        assert initial_sync_get_total() == 0
