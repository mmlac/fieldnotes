"""Tests for the Neo4j + Qdrant write layer.

Uses unittest.mock to stub out Neo4j and Qdrant clients so tests run
without running services.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from worker.parsers.base import GraphHint, ParsedDocument
from worker.pipeline.chunker import Chunk
from worker.pipeline.writer import (
    ALLOWED_PREDICATES,
    COLLECTION_NAME,
    WriteUnit,
    Writer,
    _chunk_node_id,
    _clean_source_edges,
    _clean_stale_edges,
    _cleanup_orphan_entities,
    _merge_attached_to_edge,
    _merge_depicts_edge,
    _merge_entity_edge,
    _merge_entity_edges_batch,
    _merge_mentions_edge,
    _upsert_chunk,
    _upsert_chunks_batch,
    _upsert_entities_and_depicts_batch,
    _upsert_entities_and_mentions_batch,
    _upsert_entity,
    _upsert_source_node,
    _validate_cypher_identifier,
    _write_graph_hint,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


def _doc(**overrides) -> ParsedDocument:
    """Create a minimal ParsedDocument for testing."""
    defaults = dict(
        source_type="file",
        source_id="notes/test.md",
        operation="created",
        text="Hello world",
        node_label="File",
        node_props={"name": "test.md", "path": "notes/test.md"},
    )
    defaults.update(overrides)
    return ParsedDocument(**defaults)


def _unit(doc=None, **kwargs) -> WriteUnit:
    """Create a WriteUnit with sensible defaults."""
    return WriteUnit(doc=doc or _doc(), **kwargs)


@pytest.fixture
def mock_neo4j():
    with patch("worker.pipeline.writer.GraphDatabase") as mock_gdb:
        driver = MagicMock()
        mock_gdb.driver.return_value = driver
        yield driver


@pytest.fixture
def mock_qdrant():
    with patch("worker.pipeline.writer.QdrantClient") as mock_cls:
        client = MagicMock()
        mock_cls.return_value = client
        # Simulate empty collection list so _ensure_qdrant_collection creates it
        coll = MagicMock()
        coll.name = "other"
        collections_resp = MagicMock()
        collections_resp.collections = [coll]
        client.get_collections.return_value = collections_resp
        yield client


@pytest.fixture
def writer(mock_neo4j, mock_qdrant):
    """Create a Writer with mocked backends."""
    w = Writer()
    return w


# ------------------------------------------------------------------
# WriteUnit / dataclass tests
# ------------------------------------------------------------------


class TestWriteUnit:
    def test_defaults(self):
        doc = _doc()
        unit = WriteUnit(doc=doc)
        assert unit.chunks == []
        assert unit.vectors == []
        assert unit.entities == []
        assert unit.triples == []

    def test_with_data(self):
        doc = _doc()
        chunks = [Chunk(text="hello", index=0)]
        vectors = [[0.1] * 768]
        entities = [{"name": "Neo4j", "type": "Technology"}]
        triples = [{"subject": "Neo4j", "predicate": "RELATED_TO", "object": "Qdrant"}]
        unit = WriteUnit(
            doc=doc,
            chunks=chunks,
            vectors=vectors,
            entities=entities,
            triples=triples,
        )
        assert len(unit.chunks) == 1
        assert len(unit.vectors) == 1


# ------------------------------------------------------------------
# Writer construction
# ------------------------------------------------------------------


class TestWriterInit:
    def test_creates_collection_if_missing(self, mock_neo4j, mock_qdrant):
        """Collection should be created when it doesn't exist."""
        Writer()
        mock_qdrant.create_collection.assert_called_once()

    def test_skips_collection_if_exists(self, mock_neo4j, mock_qdrant):
        """No create_collection when collection already exists."""
        coll = MagicMock()
        coll.name = COLLECTION_NAME
        collections_resp = MagicMock()
        collections_resp.collections = [coll]
        mock_qdrant.get_collections.return_value = collections_resp
        Writer()
        mock_qdrant.create_collection.assert_not_called()


# ------------------------------------------------------------------
# Neo4j transaction helpers
# ------------------------------------------------------------------


class TestNeo4jHelpers:
    def test_upsert_source_node(self):
        tx = MagicMock()
        doc = _doc(node_props={"name": "test.md", "ext": ".md"})
        _upsert_source_node(tx, doc)
        tx.run.assert_called_once()
        args, kwargs = tx.run.call_args
        assert "MERGE" in args[0]
        assert "File" in args[0]
        assert kwargs["source_id"] == "notes/test.md"
        assert kwargs["name"] == "test.md"

    def test_upsert_entity(self):
        tx = MagicMock()
        _upsert_entity(tx, {"name": "Neo4j", "type": "Technology", "confidence": 0.9})
        tx.run.assert_called_once()
        args, kwargs = tx.run.call_args
        assert "MERGE" in args[0]
        assert "Entity" in args[0]
        assert kwargs["name"] == "Neo4j"
        assert kwargs["type"] == "Technology"

    def test_upsert_entity_defaults(self):
        tx = MagicMock()
        _upsert_entity(tx, {"name": "Something"})
        _, kwargs = tx.run.call_args
        assert kwargs["type"] == "Concept"
        assert kwargs["confidence"] == 0.75

    def test_upsert_entity_uses_on_create_on_match(self):
        """Cypher must use ON CREATE / ON MATCH to guard against lower-confidence overwrites."""
        tx = MagicMock()
        _upsert_entity(tx, {"name": "Neo4j", "type": "Technology", "confidence": 0.9})
        args, _ = tx.run.call_args
        cypher = args[0]
        assert "ON CREATE" in cypher
        assert "ON MATCH" in cypher

    def test_upsert_entity_confidence_guard_in_cypher(self):
        """ON MATCH SET should only update when incoming confidence is higher."""
        tx = MagicMock()
        _upsert_entity(tx, {"name": "Alice", "type": "Person", "confidence": 0.5})
        args, kwargs = tx.run.call_args
        cypher = args[0]
        # The Cypher CASE expression must compare $confidence against e.confidence
        assert "$confidence > e.confidence" in cypher
        assert kwargs["confidence"] == 0.5
        assert kwargs["type"] == "Person"

    def test_merge_mentions_edge(self):
        tx = MagicMock()
        _merge_mentions_edge(tx, "notes/test.md", "Neo4j")
        tx.run.assert_called_once()
        args, kwargs = tx.run.call_args
        assert "MENTIONS" in args[0]
        assert kwargs["sid"] == "notes/test.md"
        assert kwargs["name"] == "Neo4j"

    def test_merge_entity_edge(self):
        tx = MagicMock()
        triple = {"subject": "Neo4j", "predicate": "RELATED_TO", "object": "Qdrant"}
        _merge_entity_edge(tx, triple)
        tx.run.assert_called_once()
        args, _ = tx.run.call_args
        assert "RELATED_TO" in args[0]

    def test_merge_entity_edge_allowed_predicate(self):
        """Whitelisted predicates are used as-is."""
        tx = MagicMock()
        triple = {"subject": "Alice", "predicate": "works at", "object": "Acme"}
        _merge_entity_edge(tx, triple)
        args, _ = tx.run.call_args
        assert "WORKS_AT" in args[0]

    def test_merge_entity_edge_unknown_predicate_mapped(self):
        """Unknown predicates are mapped to RELATED_TO."""
        tx = MagicMock()
        triple = {"subject": "A", "predicate": "BAZINGA", "object": "B"}
        _merge_entity_edge(tx, triple)
        args, _ = tx.run.call_args
        assert "RELATED_TO" in args[0]
        assert "BAZINGA" not in args[0]

    def test_allowed_predicates_contains_common_types(self):
        """Sanity check: common relationship types are in the whitelist."""
        for pred in (
            "RELATED_TO",
            "WORKS_AT",
            "PART_OF",
            "KNOWS",
            "USES",
            "CREATED_BY",
        ):
            assert pred in ALLOWED_PREDICATES

    def test_upsert_chunk(self):
        tx = MagicMock()
        chunk = Chunk(text="hello world", index=0)
        _upsert_chunk(tx, "src:chunk:0", "src", chunk)
        tx.run.assert_called_once()
        args, kwargs = tx.run.call_args
        assert "Chunk" in args[0]
        assert "HAS_CHUNK" in args[0]
        assert kwargs["text"] == "hello world"

    def test_write_graph_hint(self):
        tx = MagicMock()
        hint = GraphHint(
            subject_id="notes/a.md",
            subject_label="File",
            predicate="LINKS_TO",
            object_id="notes/b.md",
            object_label="File",
            object_props={},
            confidence=0.95,
        )
        _write_graph_hint(tx, hint)
        assert tx.run.call_count == 3  # subject MERGE, object MERGE, relationship MERGE

    def test_write_graph_hint_with_object_props(self):
        tx = MagicMock()
        hint = GraphHint(
            subject_id="notes/a.md",
            subject_label="File",
            predicate="TAGGED_BY_USER",
            object_id="tag:python",
            object_label="Topic",
            object_props={"source": "user"},
            confidence=1.0,
        )
        _write_graph_hint(tx, hint)
        # Object MERGE should include source property (prefixed with o_)
        obj_call = tx.run.call_args_list[1]
        assert "o_source" in obj_call[1]


# ------------------------------------------------------------------
# _clean_source_edges
# ------------------------------------------------------------------


class TestCleanSourceEdges:
    def test_deletes_mentions_edges(self):
        tx = MagicMock()
        _clean_source_edges(tx, "notes/test.md", "MENTIONS")
        tx.run.assert_called_once()
        args, kwargs = tx.run.call_args
        assert "MENTIONS" in args[0]
        assert "DELETE r" in args[0]
        assert kwargs["sid"] == "notes/test.md"

    def test_rejects_unsafe_edge_type(self):
        tx = MagicMock()
        with pytest.raises(ValueError, match="edge_type"):
            _clean_source_edges(tx, "notes/test.md", "BAD; DROP")
        tx.run.assert_not_called()

    def test_modified_operation_cleans_mentions_after_write(self):
        """Modified sources should have stale MENTIONS edges deleted AFTER new edges are written."""
        doc = _doc(operation="modified")
        unit = _unit(
            doc=doc,
            entities=[{"name": "NewEntity", "type": "Concept"}],
        )
        tx = MagicMock()
        record = MagicMock()
        record.__getitem__ = lambda self, key: 0 if key == "removed" else None
        tx.run.return_value.single.return_value = record
        Writer._write_neo4j_tx(tx, unit)

        queries = [c[0][0] for c in tx.run.call_args_list]
        # The MENTIONS MERGE should appear before the stale edge DELETE
        mentions_idx = next(
            i for i, q in enumerate(queries) if "MENTIONS" in q and "MERGE" in q
        )
        delete_idx = next(
            i for i, q in enumerate(queries) if "DELETE r" in q and "MENTIONS" in q
        )
        assert mentions_idx < delete_idx

    def test_created_operation_no_cleanup(self):
        """Created sources should NOT have MENTIONS edges deleted."""
        doc = _doc(operation="created")
        unit = _unit(
            doc=doc,
            entities=[{"name": "Entity", "type": "Concept"}],
        )
        tx = MagicMock()
        Writer._write_neo4j_tx(tx, unit)

        queries = [c[0][0] for c in tx.run.call_args_list]
        delete_queries = [q for q in queries if "DELETE r" in q]
        assert len(delete_queries) == 0


# ------------------------------------------------------------------
# _clean_stale_edges
# ------------------------------------------------------------------


class TestCleanStaleEdges:
    def test_deletes_edges_not_in_keep_list(self):
        tx = MagicMock()
        _clean_stale_edges(tx, "notes/test.md", "MENTIONS", ["Alice"])
        tx.run.assert_called_once()
        args, kwargs = tx.run.call_args
        assert "MENTIONS" in args[0]
        assert "NOT e.name IN $keep" in args[0]
        assert kwargs["keep"] == ["Alice"]
        assert kwargs["sid"] == "notes/test.md"

    def test_deletes_all_when_keep_empty(self):
        tx = MagicMock()
        _clean_stale_edges(tx, "notes/test.md", "MENTIONS", [])
        tx.run.assert_called_once()
        args, kwargs = tx.run.call_args
        assert "MENTIONS" in args[0]
        assert "DELETE r" in args[0]
        assert "keep" not in kwargs

    def test_rejects_unsafe_edge_type(self):
        tx = MagicMock()
        with pytest.raises(ValueError, match="edge_type"):
            _clean_stale_edges(tx, "notes/test.md", "BAD; DROP", ["Alice"])
        tx.run.assert_not_called()


# ------------------------------------------------------------------
# _cleanup_orphan_entities
# ------------------------------------------------------------------


class TestCleanupOrphanEntities:
    def test_deletes_orphan_entities(self):
        """Should run DETACH DELETE on entities with no incoming edges."""
        tx = MagicMock()
        record = MagicMock()
        record.__getitem__ = lambda self, key: 3 if key == "removed" else None
        tx.run.return_value.single.return_value = record
        removed = _cleanup_orphan_entities(tx, ["Alice", "Neo4j"])
        assert removed == 3
        args, _ = tx.run.call_args
        assert "Entity" in args[0]
        assert "DETACH DELETE" in args[0]
        assert "NOT ()-[]->(e)" in args[0]

    def test_returns_zero_when_no_orphans(self):
        tx = MagicMock()
        record = MagicMock()
        record.__getitem__ = lambda self, key: 0 if key == "removed" else None
        tx.run.return_value.single.return_value = record
        removed = _cleanup_orphan_entities(tx, ["SomeEntity"])
        assert removed == 0

    def test_skips_query_when_empty_candidates(self):
        """Should return 0 without querying when candidate list is empty."""
        tx = MagicMock()
        removed = _cleanup_orphan_entities(tx, [])
        assert removed == 0
        tx.run.assert_not_called()

    def test_modified_operation_triggers_orphan_cleanup(self):
        """Modified sources should run orphan entity cleanup after writes."""
        doc = _doc(operation="modified")
        unit = _unit(
            doc=doc,
            entities=[{"name": "NewEntity", "type": "Concept"}],
        )
        tx = MagicMock()
        record = MagicMock()
        record.__getitem__ = lambda self, key: 0 if key == "removed" else None
        tx.run.return_value.single.return_value = record
        Writer._write_neo4j_tx(tx, unit)

        queries = [c[0][0] for c in tx.run.call_args_list]
        orphan_queries = [q for q in queries if "NOT ()-[]->(e)" in q]
        assert len(orphan_queries) == 1

    def test_created_operation_no_orphan_cleanup(self):
        """Created sources should NOT run orphan entity cleanup."""
        doc = _doc(operation="created")
        unit = _unit(
            doc=doc,
            entities=[{"name": "Entity", "type": "Concept"}],
        )
        tx = MagicMock()
        Writer._write_neo4j_tx(tx, unit)

        queries = [c[0][0] for c in tx.run.call_args_list]
        orphan_queries = [q for q in queries if "NOT ()-[]->(e)" in q]
        assert len(orphan_queries) == 0


# ------------------------------------------------------------------
# chunk_node_id
# ------------------------------------------------------------------


class TestChunkNodeId:
    def test_deterministic(self):
        a = _chunk_node_id("notes/test.md", 0)
        b = _chunk_node_id("notes/test.md", 0)
        assert a == b

    def test_different_for_different_index(self):
        a = _chunk_node_id("notes/test.md", 0)
        b = _chunk_node_id("notes/test.md", 1)
        assert a != b


# ------------------------------------------------------------------
# Writer.fetch_existing_entities
# ------------------------------------------------------------------


class TestFetchExistingEntities:
    def test_returns_entity_dicts(self, writer, mock_neo4j, mock_qdrant):
        """Should query Entity nodes and return list of dicts."""
        record1 = {"name": "Alice", "type": "Person", "confidence": 0.95}
        record2 = {"name": "Neo4j", "type": "Technology", "confidence": 0.8}

        session = MagicMock()
        mock_neo4j.session.return_value.__enter__ = MagicMock(return_value=session)
        mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)
        session.run.return_value = [record1, record2]

        result = writer.fetch_existing_entities()

        assert len(result) == 2
        assert result[0] == {"name": "Alice", "type": "Person", "confidence": 0.95}
        assert result[1] == {"name": "Neo4j", "type": "Technology", "confidence": 0.8}

    def test_returns_empty_when_no_entities(self, writer, mock_neo4j, mock_qdrant):
        """Should return empty list when no Entity nodes exist."""
        session = MagicMock()
        mock_neo4j.session.return_value.__enter__ = MagicMock(return_value=session)
        mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)
        session.run.return_value = []

        result = writer.fetch_existing_entities()

        assert result == []

    def test_defaults_for_null_fields(self, writer, mock_neo4j, mock_qdrant):
        """Should use defaults when type or confidence is None."""
        record = {"name": "Unknown", "type": None, "confidence": None}

        session = MagicMock()
        mock_neo4j.session.return_value.__enter__ = MagicMock(return_value=session)
        mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)
        session.run.return_value = [record]

        result = writer.fetch_existing_entities()

        assert result[0]["type"] == "Concept"
        assert result[0]["confidence"] == 0.75


# ------------------------------------------------------------------
# Writer.write — create/modify
# ------------------------------------------------------------------


class TestWriterWrite:
    def test_write_calls_neo4j_and_qdrant(self, writer, mock_neo4j, mock_qdrant):
        """A non-deletion write should call execute_write and qdrant upsert."""
        unit = _unit(
            chunks=[Chunk(text="hello", index=0)],
            vectors=[[0.1] * 768],
            entities=[{"name": "Test", "type": "Concept"}],
        )
        session = MagicMock()
        mock_neo4j.session.return_value.__enter__ = MagicMock(return_value=session)
        mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

        writer.write(unit)

        session.execute_write.assert_called_once()
        mock_qdrant.upsert.assert_called_once()

    def test_write_no_pre_deletion(self, writer, mock_neo4j, mock_qdrant):
        """Upsert should NOT pre-delete vectors (idempotent via UUID5 IDs)."""
        unit = _unit(
            chunks=[Chunk(text="hello", index=0)],
            vectors=[[0.1] * 768],
        )
        session = MagicMock()
        mock_neo4j.session.return_value.__enter__ = MagicMock(return_value=session)
        mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

        writer.write(unit)

        # delete is called for stale chunk cleanup, not blanket pre-deletion
        delete_calls = mock_qdrant.delete.call_args_list
        for c in delete_calls:
            filt = (
                c[1].get("points_selector") or c[0][0]
                if c[0]
                else c[1].get("points_selector")
            )
            # Stale cleanup filter has a Range condition on chunk_index
            conditions = filt.must
            has_range = any(
                getattr(cond, "range", None) is not None for cond in conditions
            )
            assert has_range, (
                "delete should only be for stale chunk cleanup, not blanket deletion"
            )

    def test_write_skips_qdrant_when_no_vectors(self, writer, mock_neo4j, mock_qdrant):
        """No qdrant upsert when there are no vectors."""
        unit = _unit()
        session = MagicMock()
        mock_neo4j.session.return_value.__enter__ = MagicMock(return_value=session)
        mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

        writer.write(unit)

        mock_qdrant.upsert.assert_not_called()

    def test_qdrant_payload_structure(self, writer, mock_neo4j, mock_qdrant):
        """Verify Qdrant points have the expected payload fields."""
        doc = _doc(source_metadata={"date": "2026-03-10T00:00:00Z"})
        unit = _unit(
            doc=doc,
            chunks=[Chunk(text="chunk text", index=0)],
            vectors=[[0.5] * 768],
        )
        session = MagicMock()
        mock_neo4j.session.return_value.__enter__ = MagicMock(return_value=session)
        mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

        writer.write(unit)

        upsert_call = mock_qdrant.upsert.call_args
        points = upsert_call[1]["points"]
        assert len(points) == 1
        payload = points[0].payload
        assert payload["source_type"] == "file"
        assert payload["source_id"] == "notes/test.md"
        assert payload["chunk_index"] == 0
        assert payload["text"] == "chunk text"
        assert payload["date"] == "2026-03-10T00:00:00Z"

    def test_qdrant_deterministic_ids(self, writer, mock_neo4j, mock_qdrant):
        """Point IDs should be deterministic for the same source_id + chunk_index."""
        unit = _unit(
            chunks=[Chunk(text="a", index=0), Chunk(text="b", index=1)],
            vectors=[[0.1] * 768, [0.2] * 768],
        )
        session = MagicMock()
        mock_neo4j.session.return_value.__enter__ = MagicMock(return_value=session)
        mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

        writer.write(unit)

        points = mock_qdrant.upsert.call_args[1]["points"]
        ids = [p.id for p in points]
        assert len(set(ids)) == 2  # two distinct IDs

    def test_qdrant_retry_on_failure(self, writer, mock_neo4j, mock_qdrant):
        """Qdrant write should be retried on transient failure via tenacity."""
        unit = _unit(
            chunks=[Chunk(text="a", index=0)],
            vectors=[[0.1] * 768],
        )
        session = MagicMock()
        mock_neo4j.session.return_value.__enter__ = MagicMock(return_value=session)
        mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

        # Fail first upsert, succeed on retry
        mock_qdrant.upsert.side_effect = [ConnectionError("timeout"), None]

        writer.write(unit)

        assert mock_qdrant.upsert.call_count == 2

    def test_qdrant_raises_after_max_retries(self, writer, mock_neo4j, mock_qdrant):
        """Should reraise ConnectionError after exhausting tenacity retries."""
        unit = _unit(
            chunks=[Chunk(text="a", index=0)],
            vectors=[[0.1] * 768],
        )
        session = MagicMock()
        mock_neo4j.session.return_value.__enter__ = MagicMock(return_value=session)
        mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

        mock_qdrant.upsert.side_effect = ConnectionError("timeout")

        with pytest.raises(ConnectionError):
            writer.write(unit)

        # tenacity _qdrant_retry uses stop_after_attempt(4)
        assert mock_qdrant.upsert.call_count == 4

    def test_stale_chunk_cleanup_after_upsert(self, writer, mock_neo4j, mock_qdrant):
        """Stale chunks with higher indices should be deleted after upsert."""
        unit = _unit(
            chunks=[Chunk(text="a", index=0), Chunk(text="b", index=1)],
            vectors=[[0.1] * 768, [0.2] * 768],
        )
        session = MagicMock()
        mock_neo4j.session.return_value.__enter__ = MagicMock(return_value=session)
        mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

        writer.write(unit)

        # Verify stale cleanup was called (delete with Range filter)
        mock_qdrant.delete.assert_called_once()
        filt = mock_qdrant.delete.call_args[1]["points_selector"]
        conditions = filt.must
        range_conds = [c for c in conditions if getattr(c, "range", None) is not None]
        assert len(range_conds) == 1
        assert range_conds[0].range.gte == 2  # chunk count

    def test_write_invalidates_entity_cache_when_entities(
        self, writer, mock_neo4j, mock_qdrant
    ):
        """Entity cache should be invalidated after writing entities."""
        unit = _unit(
            entities=[{"name": "Test", "type": "Concept"}],
            chunks=[Chunk(text="hello", index=0)],
            vectors=[[0.1] * 768],
        )
        session = MagicMock()
        mock_neo4j.session.return_value.__enter__ = MagicMock(return_value=session)
        mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

        # Prime the cache
        writer._entity_cache = [{"name": "Old", "type": "Concept", "confidence": 0.75}]
        writer._entity_cache_ts = 9999999999.0  # far future so it's "valid"

        writer.write(unit)

        assert writer._entity_cache is None
        assert writer._entity_cache_ts == 0.0

    def test_write_no_cache_invalidation_without_entities(
        self, writer, mock_neo4j, mock_qdrant
    ):
        """Entity cache should NOT be invalidated when no entities are written."""
        unit = _unit(
            chunks=[Chunk(text="hello", index=0)],
            vectors=[[0.1] * 768],
        )
        session = MagicMock()
        mock_neo4j.session.return_value.__enter__ = MagicMock(return_value=session)
        mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

        # Prime the cache
        cached = [{"name": "Old", "type": "Concept", "confidence": 0.75}]
        writer._entity_cache = cached
        writer._entity_cache_ts = 9999999999.0

        writer.write(unit)

        assert writer._entity_cache is cached  # unchanged


# ------------------------------------------------------------------
# Writer.write — deletion
# ------------------------------------------------------------------


class TestWriterDelete:
    def test_delete_calls_both_stores(self, writer, mock_neo4j, mock_qdrant):
        """A delete operation should remove from both Neo4j and Qdrant."""
        doc = _doc(operation="deleted")
        unit = _unit(doc=doc)

        session = MagicMock()
        mock_neo4j.session.return_value.__enter__ = MagicMock(return_value=session)
        mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

        writer.write(unit)

        session.execute_write.assert_called_once()
        mock_qdrant.delete.assert_called_once()

    def test_delete_neo4j_removes_chunks_then_source(
        self, writer, mock_neo4j, mock_qdrant
    ):
        """The delete tx should remove chunks first, then the source node."""
        tx = MagicMock()
        Writer._delete_neo4j_tx(tx, "notes/test.md")
        assert tx.run.call_count == 2
        # First call removes chunks
        first_query = tx.run.call_args_list[0][0][0]
        assert "HAS_CHUNK" in first_query
        assert "DETACH DELETE" in first_query
        # Second call removes source
        second_query = tx.run.call_args_list[1][0][0]
        assert "DETACH DELETE" in second_query


# ------------------------------------------------------------------
# Writer.write_batch
# ------------------------------------------------------------------


class TestWriterBatch:
    def test_write_batch(self, writer, mock_neo4j, mock_qdrant):
        """write_batch should call write for each unit."""
        session = MagicMock()
        mock_neo4j.session.return_value.__enter__ = MagicMock(return_value=session)
        mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

        units = [_unit() for _ in range(3)]
        writer.write_batch(units)

        assert session.execute_write.call_count == 3


# ------------------------------------------------------------------
# Writer.close
# ------------------------------------------------------------------


class TestWriterClose:
    def test_close(self, writer, mock_neo4j, mock_qdrant):
        writer.close()
        mock_neo4j.close.assert_called_once()
        mock_qdrant.close.assert_called_once()


# ------------------------------------------------------------------
# Cypher injection prevention
# ------------------------------------------------------------------


class TestCypherIdentifierValidation:
    def test_valid_identifiers(self):
        assert _validate_cypher_identifier("File", "test") == "File"
        assert _validate_cypher_identifier("RELATED_TO", "test") == "RELATED_TO"
        assert _validate_cypher_identifier("_private", "test") == "_private"
        assert _validate_cypher_identifier("Node123", "test") == "Node123"

    @pytest.mark.parametrize(
        "bad_value",
        [
            "}) DETACH DELETE n //",
            "File})--(n:Admin",
            "RELATED TO",
            "name; DROP",
            "",
            "123start",
            "has-dash",
            "has.dot",
        ],
    )
    def test_rejects_injection_payloads(self, bad_value):
        with pytest.raises(ValueError, match="Unsafe Cypher identifier"):
            _validate_cypher_identifier(bad_value, "test")

    def test_upsert_source_node_rejects_bad_label(self):
        tx = MagicMock()
        doc = _doc(node_label="}) DETACH DELETE n //")
        with pytest.raises(ValueError, match="node_label"):
            _upsert_source_node(tx, doc)
        tx.run.assert_not_called()

    def test_upsert_source_node_rejects_bad_prop_key(self):
        tx = MagicMock()
        doc = _doc(node_props={"valid": "ok", "bad; DROP": "evil"})
        with pytest.raises(ValueError, match="node_props key"):
            _upsert_source_node(tx, doc)

    def test_merge_entity_edge_maps_bad_predicate_to_related_to(self):
        tx = MagicMock()
        triple = {"subject": "A", "predicate": "REL})--(n:Admin", "object": "B"}
        _merge_entity_edge(tx, triple)
        # Bad predicate should be mapped to RELATED_TO, not raise
        call_args = tx.run.call_args
        assert "RELATED_TO" in call_args[0][0]

    def test_write_graph_hint_rejects_bad_subject_label(self):
        tx = MagicMock()
        hint = GraphHint(
            subject_id="a",
            subject_label="Bad Label",
            predicate="LINKS_TO",
            object_id="b",
            object_label="File",
            object_props={},
            confidence=1.0,
        )
        with pytest.raises(ValueError, match="hint subject_label"):
            _write_graph_hint(tx, hint)

    def test_write_graph_hint_rejects_bad_predicate(self):
        tx = MagicMock()
        hint = GraphHint(
            subject_id="a",
            subject_label="File",
            predicate="BAD; DROP",
            object_id="b",
            object_label="File",
            object_props={},
            confidence=1.0,
        )
        with pytest.raises(ValueError, match="hint predicate"):
            _write_graph_hint(tx, hint)

    def test_write_graph_hint_rejects_bad_object_props_key(self):
        tx = MagicMock()
        hint = GraphHint(
            subject_id="a",
            subject_label="File",
            predicate="LINKS_TO",
            object_id="b",
            object_label="File",
            object_props={"evil}//": "value"},
            confidence=1.0,
        )
        with pytest.raises(ValueError, match="hint object_props key"):
            _write_graph_hint(tx, hint)


# ------------------------------------------------------------------
# Image node: DEPICTS and ATTACHED_TO edges
# ------------------------------------------------------------------


class TestImageNodeWriter:
    def test_merge_depicts_edge(self):
        tx = MagicMock()
        _merge_depicts_edge(tx, "images/photo.png", "Alice")
        tx.run.assert_called_once()
        args, kwargs = tx.run.call_args
        assert "DEPICTS" in args[0]
        assert kwargs["sid"] == "images/photo.png"
        assert kwargs["name"] == "Alice"

    def test_merge_attached_to_edge(self):
        tx = MagicMock()
        _merge_attached_to_edge(tx, "images/photo.png", "notes/daily.md")
        tx.run.assert_called_once()
        args, kwargs = tx.run.call_args
        assert "ATTACHED_TO" in args[0]
        assert "Image" in args[0]
        assert "File" in args[0]
        assert kwargs["img_sid"] == "images/photo.png"
        assert kwargs["parent_sid"] == "notes/daily.md"

    def test_image_node_upsert_with_vision_props(self):
        """Image nodes should be upserted with sha256, vision_processed, parent_source_id."""
        tx = MagicMock()
        doc = _doc(
            node_label="Image",
            source_id="images/photo.png",
            node_props={
                "name": "photo.png",
                "path": "images/photo.png",
                "sha256": "abc123def456",
                "vision_processed": False,
                "parent_source_id": "notes/daily.md",
            },
        )
        _upsert_source_node(tx, doc)
        tx.run.assert_called_once()
        args, kwargs = tx.run.call_args
        assert "MERGE" in args[0]
        assert "Image" in args[0]
        assert kwargs["sha256"] == "abc123def456"
        assert kwargs["vision_processed"] is False
        assert kwargs["parent_source_id"] == "notes/daily.md"

    def test_write_image_creates_attached_to_edge(
        self, writer, mock_neo4j, mock_qdrant
    ):
        """Writing an Image node with parent_source_id should create ATTACHED_TO edge."""
        doc = _doc(
            node_label="Image",
            source_id="images/photo.png",
            node_props={
                "name": "photo.png",
                "path": "images/photo.png",
                "sha256": "abc123",
                "vision_processed": False,
                "parent_source_id": "notes/daily.md",
            },
        )
        unit = _unit(doc=doc)

        session = MagicMock()
        mock_neo4j.session.return_value.__enter__ = MagicMock(return_value=session)
        mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

        writer.write(unit)

        # The tx function is called within execute_write; verify it was called
        session.execute_write.assert_called_once()
        # Extract the tx function and call it manually to inspect queries
        tx = MagicMock()
        tx_fn = session.execute_write.call_args[0][0]
        tx_fn(tx, unit)

        # Find the ATTACHED_TO call among all tx.run calls
        attached_to_calls = [
            c for c in tx.run.call_args_list if "ATTACHED_TO" in str(c)
        ]
        assert len(attached_to_calls) == 1
        args, kwargs = attached_to_calls[0]
        assert kwargs["img_sid"] == "images/photo.png"
        assert kwargs["parent_sid"] == "notes/daily.md"

    def test_write_image_no_attached_to_without_parent(
        self, writer, mock_neo4j, mock_qdrant
    ):
        """Image without parent_source_id should NOT create ATTACHED_TO edge."""
        doc = _doc(
            node_label="Image",
            source_id="images/standalone.png",
            node_props={
                "name": "standalone.png",
                "path": "images/standalone.png",
                "sha256": "def456",
                "vision_processed": False,
            },
        )
        unit = _unit(doc=doc)

        tx = MagicMock()
        Writer._write_neo4j_tx(tx, unit)

        attached_to_calls = [
            c for c in tx.run.call_args_list if "ATTACHED_TO" in str(c)
        ]
        assert len(attached_to_calls) == 0

    def test_write_non_image_no_attached_to(self, writer, mock_neo4j, mock_qdrant):
        """Non-Image nodes should NOT create ATTACHED_TO edge even with parent_source_id."""
        doc = _doc(
            node_label="File",
            node_props={
                "name": "test.md",
                "path": "notes/test.md",
                "parent_source_id": "notes/",
            },
        )
        unit = _unit(doc=doc)

        tx = MagicMock()
        Writer._write_neo4j_tx(tx, unit)

        attached_to_calls = [
            c for c in tx.run.call_args_list if "ATTACHED_TO" in str(c)
        ]
        assert len(attached_to_calls) == 0

    def test_write_image_with_depicts_entities(self, writer, mock_neo4j, mock_qdrant):
        """Image with depicts_entities should create Entity nodes and DEPICTS edges."""
        doc = _doc(
            node_label="Image",
            source_id="images/team.jpg",
            node_props={
                "name": "team.jpg",
                "sha256": "aaa111",
                "vision_processed": False,
                "parent_source_id": "notes/meeting.md",
            },
        )
        depicts = [
            {"name": "Alice", "type": "Person", "confidence": 0.8},
            {"name": "Whiteboard", "type": "Concept", "confidence": 0.8},
        ]
        unit = _unit(doc=doc, depicts_entities=depicts)

        tx = MagicMock()
        Writer._write_neo4j_tx(tx, unit)

        # Should have DEPICTS edges for each entity
        depicts_calls = [c for c in tx.run.call_args_list if "DEPICTS" in str(c)]
        assert len(depicts_calls) == 2

        # Should also have ATTACHED_TO
        attached_calls = [c for c in tx.run.call_args_list if "ATTACHED_TO" in str(c)]
        assert len(attached_calls) == 1


# ------------------------------------------------------------------
# Email graph nodes: Person, Email, Thread and email edges
# ------------------------------------------------------------------


class TestEmailGraphHints:
    """Test _write_graph_hint with email-specific merge keys."""

    def test_person_node_merges_on_email(self):
        """Person nodes should MERGE on the email property, not source_id."""
        tx = MagicMock()
        hint = GraphHint(
            subject_id="person:alice@example.com",
            subject_label="Person",
            predicate="SENT",
            object_id="gmail:msg-123",
            object_label="Email",
            subject_props={"email": "alice@example.com"},
            subject_merge_key="email",
            confidence=1.0,
        )
        _write_graph_hint(tx, hint)

        # Subject MERGE should use email as the merge key
        subj_call = tx.run.call_args_list[0]
        query = subj_call[0][0]
        assert "Person" in query
        assert "{email: $s_merge}" in query
        assert subj_call[1]["s_merge"] == "alice@example.com"
        # source_id should still be SET
        assert subj_call[1]["s_source_id"] == "person:alice@example.com"

    def test_thread_node_merges_on_thread_id(self):
        """Thread nodes should MERGE on thread_id property."""
        tx = MagicMock()
        hint = GraphHint(
            subject_id="gmail:msg-123",
            subject_label="Email",
            predicate="PART_OF",
            object_id="gmail-thread:thread-456",
            object_label="Thread",
            object_props={"thread_id": "thread-456", "subject": "Test"},
            object_merge_key="thread_id",
            confidence=1.0,
        )
        _write_graph_hint(tx, hint)

        # Object MERGE should use thread_id as the merge key
        obj_call = tx.run.call_args_list[1]
        query = obj_call[0][0]
        assert "Thread" in query
        assert "{thread_id: $o_merge}" in query
        assert obj_call[1]["o_merge"] == "thread-456"
        assert obj_call[1]["o_subject"] == "Test"

    def test_to_edge_person_as_object(self):
        """TO edge: Email→Person should merge Person on email."""
        tx = MagicMock()
        hint = GraphHint(
            subject_id="gmail:msg-123",
            subject_label="Email",
            predicate="TO",
            object_id="person:bob@example.com",
            object_label="Person",
            object_props={"email": "bob@example.com"},
            object_merge_key="email",
            confidence=1.0,
        )
        _write_graph_hint(tx, hint)

        # Object (Person) should merge on email
        obj_call = tx.run.call_args_list[1]
        query = obj_call[0][0]
        assert "{email: $o_merge}" in query
        assert obj_call[1]["o_merge"] == "bob@example.com"

        # Edge MATCH should use the correct merge keys
        edge_call = tx.run.call_args_list[2]
        edge_query = edge_call[0][0]
        assert "TO" in edge_query
        assert "{source_id: $s_merge}" in edge_query  # Email uses source_id
        assert "{email: $o_merge}" in edge_query  # Person uses email

    def test_sent_edge_creates_relationship(self):
        """SENT edge: Person→Email should create correct relationship."""
        tx = MagicMock()
        hint = GraphHint(
            subject_id="person:alice@example.com",
            subject_label="Person",
            predicate="SENT",
            object_id="gmail:msg-123",
            object_label="Email",
            subject_props={"email": "alice@example.com"},
            subject_merge_key="email",
            confidence=1.0,
        )
        _write_graph_hint(tx, hint)

        # Verify 3 calls: subject MERGE, object MERGE, edge MERGE
        assert tx.run.call_count == 3
        edge_call = tx.run.call_args_list[2]
        edge_query = edge_call[0][0]
        assert "SENT" in edge_query
        assert "{email: $s_merge}" in edge_query  # Person uses email
        assert "{source_id: $o_merge}" in edge_query  # Email uses source_id

    def test_part_of_edge_creates_relationship(self):
        """PART_OF edge: Email→Thread should create correct relationship."""
        tx = MagicMock()
        hint = GraphHint(
            subject_id="gmail:msg-123",
            subject_label="Email",
            predicate="PART_OF",
            object_id="gmail-thread:thread-456",
            object_label="Thread",
            object_props={"thread_id": "thread-456", "subject": "Test"},
            object_merge_key="thread_id",
            confidence=1.0,
        )
        _write_graph_hint(tx, hint)

        edge_call = tx.run.call_args_list[2]
        edge_query = edge_call[0][0]
        assert "PART_OF" in edge_query
        assert "{source_id: $s_merge}" in edge_query  # Email uses source_id
        assert "{thread_id: $o_merge}" in edge_query  # Thread uses thread_id

    def test_default_merge_key_is_source_id(self):
        """Without custom merge keys, nodes should still MERGE on source_id."""
        tx = MagicMock()
        hint = GraphHint(
            subject_id="notes/a.md",
            subject_label="File",
            predicate="LINKS_TO",
            object_id="notes/b.md",
            object_label="File",
            confidence=0.95,
        )
        _write_graph_hint(tx, hint)

        # Both should use source_id as merge key (default)
        subj_call = tx.run.call_args_list[0]
        assert "{source_id: $s_merge}" in subj_call[0][0]
        obj_call = tx.run.call_args_list[1]
        assert "{source_id: $o_merge}" in obj_call[0][0]

    def test_email_node_written_as_source(self):
        """Email nodes via ParsedDocument should upsert with message_id prop."""
        tx = MagicMock()
        doc = _doc(
            source_type="gmail",
            source_id="gmail:msg-123",
            node_label="Email",
            node_props={
                "message_id": "msg-123",
                "subject": "Test",
                "date": "2026-03-11",
            },
        )
        _upsert_source_node(tx, doc)
        args, kwargs = tx.run.call_args
        assert "MERGE" in args[0]
        assert "Email" in args[0]
        assert kwargs["message_id"] == "msg-123"
        assert kwargs["subject"] == "Test"

    def test_full_email_write_with_hints(self, writer, mock_neo4j, mock_qdrant):
        """End-to-end: writing an Email WriteUnit processes all graph hints."""
        doc = _doc(
            source_type="gmail",
            source_id="gmail:msg-123",
            node_label="Email",
            node_props={
                "message_id": "msg-123",
                "subject": "Test",
                "date": "2026-03-11",
            },
            graph_hints=[
                GraphHint(
                    subject_id="person:alice@example.com",
                    subject_label="Person",
                    predicate="SENT",
                    object_id="gmail:msg-123",
                    object_label="Email",
                    subject_props={"email": "alice@example.com"},
                    subject_merge_key="email",
                    confidence=1.0,
                ),
                GraphHint(
                    subject_id="gmail:msg-123",
                    subject_label="Email",
                    predicate="TO",
                    object_id="person:bob@example.com",
                    object_label="Person",
                    object_props={"email": "bob@example.com"},
                    object_merge_key="email",
                    confidence=1.0,
                ),
                GraphHint(
                    subject_id="gmail:msg-123",
                    subject_label="Email",
                    predicate="PART_OF",
                    object_id="gmail-thread:thread-456",
                    object_label="Thread",
                    object_props={"thread_id": "thread-456", "subject": "Test"},
                    object_merge_key="thread_id",
                    confidence=1.0,
                ),
            ],
        )
        unit = _unit(doc=doc)

        # Execute the transaction function directly
        tx = MagicMock()
        Writer._write_neo4j_tx(tx, unit)

        queries = [c[0][0] for c in tx.run.call_args_list]

        # Should have: source upsert + 3 hints * 3 queries each = 10
        assert len(queries) == 10

        # Verify SENT, TO, PART_OF edges were created
        edge_queries = [
            q for q in queries if "MERGE" in q and ")-[" in q and "]->" in q
        ]
        predicates = set()
        for q in edge_queries:
            for pred in ("SENT", "TO", "PART_OF"):
                if pred in q:
                    predicates.add(pred)
        assert predicates == {"SENT", "TO", "PART_OF"}


class TestMarkVisionProcessed:
    def test_mark_vision_processed(self, writer, mock_neo4j, mock_qdrant):
        """mark_vision_processed should set vision_processed=True on Image node."""
        session = MagicMock()
        mock_neo4j.session.return_value.__enter__ = MagicMock(return_value=session)
        mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

        writer.mark_vision_processed("images/photo.png")

        session.run.assert_called_once()
        args, kwargs = session.run.call_args
        assert "Image" in args[0]
        assert "vision_processed" in args[0]
        assert kwargs["sid"] == "images/photo.png"


# ------------------------------------------------------------------
# Batch helpers
# ------------------------------------------------------------------


class TestUpsertEntitiesAndMentionsBatch:
    def test_single_call_for_multiple_entities(self):
        """Batch should issue one query for N entities + MENTIONS edges."""
        tx = MagicMock()
        entities = [
            {"name": "Alice", "type": "Person", "confidence": 0.9},
            {"name": "Neo4j", "type": "Technology", "confidence": 0.8},
        ]
        _upsert_entities_and_mentions_batch(tx, "notes/test.md", entities)

        tx.run.assert_called_once()
        args, kwargs = tx.run.call_args
        assert "UNWIND" in args[0]
        assert "MENTIONS" in args[0]
        assert "MERGE" in args[0]
        assert kwargs["sid"] == "notes/test.md"
        rows = kwargs["rows"]
        assert len(rows) == 2
        assert rows[0]["name"] == "Alice"
        assert rows[1]["name"] == "Neo4j"

    def test_uses_on_create_on_match(self):
        tx = MagicMock()
        _upsert_entities_and_mentions_batch(tx, "src", [{"name": "X"}])
        args, _ = tx.run.call_args
        assert "ON CREATE" in args[0]
        assert "ON MATCH" in args[0]

    def test_confidence_guard_in_cypher(self):
        tx = MagicMock()
        _upsert_entities_and_mentions_batch(
            tx, "src", [{"name": "X", "confidence": 0.5}]
        )
        args, _ = tx.run.call_args
        assert "row.confidence > e.confidence" in args[0]


class TestUpsertEntitiesAndDepictsBatch:
    def test_single_call_for_multiple_entities(self):
        """Batch should issue one query for N entities + DEPICTS edges."""
        tx = MagicMock()
        entities = [
            {"name": "Cat", "type": "Animal"},
            {"name": "Dog", "type": "Animal"},
        ]
        _upsert_entities_and_depicts_batch(tx, "images/photo.png", entities)

        tx.run.assert_called_once()
        args, kwargs = tx.run.call_args
        assert "UNWIND" in args[0]
        assert "DEPICTS" in args[0]
        assert "MERGE" in args[0]
        assert kwargs["sid"] == "images/photo.png"
        assert len(kwargs["rows"]) == 2


class TestUpsertChunksBatch:
    def test_single_call_for_multiple_chunks(self):
        """Batch should issue one query for N chunks."""
        tx = MagicMock()
        chunks = [Chunk(text="hello", index=0), Chunk(text="world", index=1)]
        _upsert_chunks_batch(tx, "notes/test.md", chunks)

        tx.run.assert_called_once()
        args, kwargs = tx.run.call_args
        assert "UNWIND" in args[0]
        assert "Chunk" in args[0]
        assert "HAS_CHUNK" in args[0]
        assert kwargs["sid"] == "notes/test.md"
        rows = kwargs["rows"]
        assert len(rows) == 2
        assert rows[0]["text"] == "hello"
        assert rows[0]["idx"] == 0
        assert rows[1]["text"] == "world"
        assert rows[1]["idx"] == 1

    def test_chunk_ids_are_deterministic(self):
        tx = MagicMock()
        chunks = [Chunk(text="x", index=3)]
        _upsert_chunks_batch(tx, "src", chunks)
        rows = tx.run.call_args[1]["rows"]
        assert rows[0]["id"] == _chunk_node_id("src", 3)


class TestMergeEntityEdgesBatch:
    def test_single_call_per_predicate(self):
        """Triples with same predicate should be batched into one query."""
        tx = MagicMock()
        triples = [
            {"subject": "Alice", "predicate": "KNOWS", "object": "Bob"},
            {"subject": "Bob", "predicate": "KNOWS", "object": "Carol"},
        ]
        _merge_entity_edges_batch(tx, triples)

        tx.run.assert_called_once()
        args, kwargs = tx.run.call_args
        assert "KNOWS" in args[0]
        assert "UNWIND" in args[0]
        assert len(kwargs["pairs"]) == 2

    def test_two_calls_for_two_predicates(self):
        tx = MagicMock()
        triples = [
            {"subject": "Alice", "predicate": "KNOWS", "object": "Bob"},
            {"subject": "Neo4j", "predicate": "RELATED_TO", "object": "Qdrant"},
        ]
        _merge_entity_edges_batch(tx, triples)
        assert tx.run.call_count == 2

    def test_unknown_predicate_mapped_to_related_to(self):
        tx = MagicMock()
        triples = [{"subject": "A", "predicate": "BAZINGA", "object": "B"}]
        _merge_entity_edges_batch(tx, triples)
        args, _ = tx.run.call_args
        assert "RELATED_TO" in args[0]
        assert "BAZINGA" not in args[0]

    def test_write_neo4j_tx_uses_batch_for_entities(self):
        """_write_neo4j_tx should issue one query for all entity+MENTIONS writes."""
        doc = _doc(operation="created")
        unit = _unit(
            doc=doc,
            entities=[
                {"name": "Alice", "type": "Person"},
                {"name": "Bob", "type": "Person"},
            ],
        )
        tx = MagicMock()
        Writer._write_neo4j_tx(tx, unit)

        queries = [c[0][0] for c in tx.run.call_args_list]
        mentions_queries = [q for q in queries if "MENTIONS" in q]
        # All entities should be handled in a single UNWIND query
        assert len(mentions_queries) == 1
        assert "UNWIND" in mentions_queries[0]

    def test_write_neo4j_tx_uses_batch_for_chunks(self):
        """_write_neo4j_tx should issue one query for all chunk writes."""
        doc = _doc(operation="created")
        unit = _unit(
            doc=doc,
            chunks=[Chunk(text="a", index=0), Chunk(text="b", index=1)],
        )
        tx = MagicMock()
        Writer._write_neo4j_tx(tx, unit)

        queries = [c[0][0] for c in tx.run.call_args_list]
        chunk_queries = [q for q in queries if "HAS_CHUNK" in q]
        # All chunks should be handled in a single UNWIND query
        assert len(chunk_queries) == 1
        assert "UNWIND" in chunk_queries[0]
