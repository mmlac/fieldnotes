"""Tests for the Neo4j + Qdrant write layer.

Uses unittest.mock to stub out Neo4j and Qdrant clients so tests run
without running services.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from worker.parsers.base import GraphHint, ParsedDocument
from worker.pipeline.chunker import Chunk
from worker.pipeline.writer import (
    COLLECTION_NAME,
    VECTOR_SIZE,
    WriteUnit,
    Writer,
    _chunk_node_id,
    _merge_entity_edge,
    _merge_mentions_edge,
    _upsert_chunk,
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
        # Object MERGE should include source property
        obj_call = tx.run.call_args_list[1]
        assert "source" in obj_call[1]


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
            filt = c[1].get("points_selector") or c[0][0] if c[0] else c[1].get("points_selector")
            # Stale cleanup filter has a Range condition on chunk_index
            conditions = filt.must
            has_range = any(
                getattr(cond, "range", None) is not None for cond in conditions
            )
            assert has_range, "delete should only be for stale chunk cleanup, not blanket deletion"

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

    def test_delete_neo4j_removes_chunks_then_source(self, writer, mock_neo4j, mock_qdrant):
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

    @pytest.mark.parametrize("bad_value", [
        "}) DETACH DELETE n //",
        "File})--(n:Admin",
        "RELATED TO",
        "name; DROP",
        "",
        "123start",
        "has-dash",
        "has.dot",
    ])
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

    def test_merge_entity_edge_rejects_bad_predicate(self):
        tx = MagicMock()
        triple = {"subject": "A", "predicate": "REL})--(n:Admin", "object": "B"}
        with pytest.raises(ValueError, match="triple predicate"):
            _merge_entity_edge(tx, triple)
        tx.run.assert_not_called()

    def test_write_graph_hint_rejects_bad_subject_label(self):
        tx = MagicMock()
        hint = GraphHint(
            subject_id="a", subject_label="Bad Label",
            predicate="LINKS_TO", object_id="b", object_label="File",
            object_props={}, confidence=1.0,
        )
        with pytest.raises(ValueError, match="hint subject_label"):
            _write_graph_hint(tx, hint)

    def test_write_graph_hint_rejects_bad_predicate(self):
        tx = MagicMock()
        hint = GraphHint(
            subject_id="a", subject_label="File",
            predicate="BAD; DROP", object_id="b", object_label="File",
            object_props={}, confidence=1.0,
        )
        with pytest.raises(ValueError, match="hint predicate"):
            _write_graph_hint(tx, hint)

    def test_write_graph_hint_rejects_bad_object_props_key(self):
        tx = MagicMock()
        hint = GraphHint(
            subject_id="a", subject_label="File",
            predicate="LINKS_TO", object_id="b", object_label="File",
            object_props={"evil}//": "value"}, confidence=1.0,
        )
        with pytest.raises(ValueError, match="hint object_props key"):
            _write_graph_hint(tx, hint)
