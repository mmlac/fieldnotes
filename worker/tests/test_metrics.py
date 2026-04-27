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


# ---------------------------------------------------------------------------
# Cardinality contract: ``account`` label allowlist
# ---------------------------------------------------------------------------


class TestAccountLabelCardinality:
    """Guard the cardinality contract documented at the top of metrics.py.

    Any metric carrying an ``account`` label must source the value from a
    config-validated account name (bounded by ``_ACCOUNT_NAME_RE`` in
    ``worker.config``).  This test enumerates metric definitions in
    ``metrics.py`` and fails when a new metric appears with the label
    without being added to the allowlist below — forcing a reviewer to
    inspect every call site before unbounded user input can leak in.
    """

    # Metrics explicitly reviewed and confirmed to receive ``account``
    # only from config-validated names.  Add a new entry only after
    # auditing every call site that invokes ``.labels(account=...)``.
    ALLOWED_METRICS_WITH_ACCOUNT_LABEL = frozenset(
        {
            "WORKER_ATTACHMENT_FETCH_FAILURES",
        }
    )

    @staticmethod
    def _metrics_with_account_label() -> set[str]:
        """Return the set of metric variable names in metrics.py whose
        labels list contains ``account``."""
        import ast
        from pathlib import Path

        import worker.metrics as metrics_module

        source = Path(metrics_module.__file__).read_text()
        tree = ast.parse(source)

        metric_factories = {"Counter", "Gauge", "Histogram", "Summary"}
        names: set[str] = set()

        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if not isinstance(node.value, ast.Call):
                continue
            func = node.value.func
            factory_name = (
                func.id if isinstance(func, ast.Name) else
                func.attr if isinstance(func, ast.Attribute) else None
            )
            if factory_name not in metric_factories:
                continue

            labels: list[str] = []
            # Positional labels arg: Counter(name, doc, [labels], ...)
            if len(node.value.args) >= 3 and isinstance(
                node.value.args[2], (ast.List, ast.Tuple)
            ):
                labels = [
                    elt.value
                    for elt in node.value.args[2].elts
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                ]
            # Keyword labels arg: labelnames=[...]
            for kw in node.value.keywords:
                if kw.arg in {"labelnames", "labels"} and isinstance(
                    kw.value, (ast.List, ast.Tuple)
                ):
                    labels = [
                        elt.value
                        for elt in kw.value.elts
                        if isinstance(elt, ast.Constant)
                        and isinstance(elt.value, str)
                    ]

            if "account" not in labels:
                continue

            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)

        return names

    def test_account_label_allowlist_is_exhaustive(self) -> None:
        found = self._metrics_with_account_label()
        new = found - self.ALLOWED_METRICS_WITH_ACCOUNT_LABEL
        assert not new, (
            f"New metric(s) defined with an 'account' label without review: "
            f"{sorted(new)}.  See the 'Cardinality contract' comment block at "
            f"the top of worker/metrics.py.  Audit every .labels(account=...) "
            f"call site to confirm the value is config-derived, then add the "
            f"metric to ALLOWED_METRICS_WITH_ACCOUNT_LABEL in this test."
        )

    def test_account_label_allowlist_has_no_stale_entries(self) -> None:
        found = self._metrics_with_account_label()
        stale = self.ALLOWED_METRICS_WITH_ACCOUNT_LABEL - found
        assert not stale, (
            f"Allowlist references metric(s) that no longer exist or no "
            f"longer carry an 'account' label: {sorted(stale)}.  Remove them "
            f"from ALLOWED_METRICS_WITH_ACCOUNT_LABEL."
        )
