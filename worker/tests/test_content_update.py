"""Integration tests for the content update (modify) flow.

Verifies that the full modify pipeline correctly cleans up stale data
and writes new data atomically. Uses real Neo4j and Qdrant instances;
tests are skipped if either service is unavailable.
"""

from __future__ import annotations

import uuid
from typing import Any, Generator

import pytest
from neo4j import GraphDatabase, Driver
from qdrant_client import QdrantClient

from worker.config import Neo4jConfig, QdrantConfig
from worker.parsers.base import GraphHint, ParsedDocument
from worker.pipeline.chunker import Chunk
from worker.pipeline.writer import VECTOR_SIZE, WriteUnit, Writer

# ------------------------------------------------------------------
# Connection helpers
# ------------------------------------------------------------------

_NEO4J_URI = "bolt://localhost:7687"
_NEO4J_USER = "neo4j"
_NEO4J_PASSWORD = "testpassword"
_QDRANT_HOST = "localhost"
_QDRANT_PORT = 6333
_TEST_COLLECTION = "fieldnotes_test"


def _neo4j_available() -> bool:
    try:
        with GraphDatabase.driver(_NEO4J_URI, auth=(_NEO4J_USER, _NEO4J_PASSWORD)) as d:
            d.verify_connectivity()
        return True
    except Exception:
        return False


def _qdrant_available() -> bool:
    try:
        with QdrantClient(host=_QDRANT_HOST, port=_QDRANT_PORT, timeout=3) as c:
            c.get_collections()
        return True
    except Exception:
        return False


_skip_neo4j = pytest.mark.skipif(
    not _neo4j_available(), reason="Neo4j not available at localhost:7687"
)
_skip_qdrant = pytest.mark.skipif(
    not _qdrant_available(), reason="Qdrant not available at localhost:6333"
)
_skip_services = pytest.mark.skipif(
    not (_neo4j_available() and _qdrant_available()),
    reason="Neo4j and/or Qdrant not available",
)

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


def _unique_id(prefix: str = "test") -> str:
    """Generate a unique source_id for test isolation."""
    return f"{prefix}/{uuid.uuid4().hex[:8]}.md"


def _doc(
    source_id: str, operation: str = "created", **overrides: Any
) -> ParsedDocument:
    defaults = dict(
        source_type="files",
        source_id=source_id,
        operation=operation,
        text="test content",
        node_label="File",
        node_props={"name": source_id.split("/")[-1], "path": source_id},
    )
    defaults.update(overrides)
    return ParsedDocument(**defaults)


def _chunks(texts: list[str]) -> list[Chunk]:
    return [Chunk(text=t, index=i) for i, t in enumerate(texts)]


def _vectors(count: int) -> list[list[float]]:
    return [[0.1] * VECTOR_SIZE for _ in range(count)]


def _entities(*names: str, etype: str = "Concept") -> list[dict[str, Any]]:
    return [{"name": n, "type": etype, "confidence": 0.9} for n in names]


@pytest.fixture
def neo4j_driver() -> Generator[Driver, None, None]:
    driver = GraphDatabase.driver(_NEO4J_URI, auth=(_NEO4J_USER, _NEO4J_PASSWORD))
    yield driver
    driver.close()


@pytest.fixture
def qdrant_client() -> Generator[QdrantClient, None, None]:
    client = QdrantClient(host=_QDRANT_HOST, port=_QDRANT_PORT)
    yield client
    client.close()


@pytest.fixture
def writer() -> Generator[Writer, None, None]:
    neo4j_cfg = Neo4jConfig(uri=_NEO4J_URI, user=_NEO4J_USER, password=_NEO4J_PASSWORD)
    qdrant_cfg = QdrantConfig(
        host=_QDRANT_HOST,
        port=_QDRANT_PORT,
        collection=_TEST_COLLECTION,
        vector_size=VECTOR_SIZE,
    )
    w = Writer(neo4j_cfg=neo4j_cfg, qdrant_cfg=qdrant_cfg)
    yield w
    w.close()


@pytest.fixture(autouse=True)
def _cleanup_neo4j(neo4j_driver: Driver) -> Generator[None, None, None]:
    """Remove all test nodes after each test."""
    yield
    with neo4j_driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")


@pytest.fixture(autouse=True)
def _cleanup_qdrant(qdrant_client: QdrantClient) -> Generator[None, None, None]:
    """Drop the test collection after each test."""
    yield
    try:
        qdrant_client.delete_collection(_TEST_COLLECTION)
    except Exception:
        pass


# ------------------------------------------------------------------
# Neo4j query helpers
# ------------------------------------------------------------------


def _query_edges(driver: Driver, source_id: str, edge_type: str) -> list[str]:
    """Return target entity names for edges of a given type from a source."""
    with driver.session() as session:
        result = session.run(
            f"MATCH (s {{source_id: $sid}})-[:{edge_type}]->(e) RETURN e.name AS name",
            sid=source_id,
        )
        return [r["name"] for r in result]


def _query_hint_edges(driver: Driver, source_id: str) -> list[dict[str, str]]:
    """Return all hint=true edges from a source."""
    with driver.session() as session:
        result = session.run(
            "MATCH (s {source_id: $sid})-[r {hint: true}]->(o) "
            "RETURN type(r) AS rel_type, o.source_id AS target_id",
            sid=source_id,
        )
        return [
            {"rel_type": r["rel_type"], "target_id": r["target_id"]} for r in result
        ]


def _query_chunks(driver: Driver, source_id: str) -> list[str]:
    """Return chunk texts for a source."""
    with driver.session() as session:
        result = session.run(
            "MATCH (s {source_id: $sid})-[:HAS_CHUNK]->(c:Chunk) "
            "RETURN c.text AS text ORDER BY c.chunk_index",
            sid=source_id,
        )
        return [r["text"] for r in result]


def _query_entity_exists(driver: Driver, name: str) -> bool:
    """Check whether an Entity node with the given name exists."""
    with driver.session() as session:
        result = session.run(
            "MATCH (e:Entity {name: $name}) RETURN count(e) AS cnt",
            name=name,
        )
        return result.single()["cnt"] > 0


def _query_qdrant_points(client: QdrantClient, source_id: str) -> int:
    """Count Qdrant points for a source_id."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    result = client.scroll(
        collection_name=_TEST_COLLECTION,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="source_id", match=MatchValue(value=source_id)),
            ]
        ),
        limit=100,
    )
    return len(result[0])


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


@_skip_services
class TestContentUpdate:
    """Integration tests for the source modification cleanup flow."""

    def test_modified_cleans_stale_mentions(
        self,
        writer: Writer,
        neo4j_driver: Driver,
    ) -> None:
        """v1 mentions A,B,C; v2 mentions A,D. B,C edges gone, A,D present."""
        sid = _unique_id()

        # v1: create with entities A, B, C
        v1 = WriteUnit(
            doc=_doc(sid, "created"),
            chunks=_chunks(["chunk one"]),
            vectors=_vectors(1),
            entities=_entities("A", "B", "C"),
        )
        writer.write(v1)

        mentions_v1 = sorted(_query_edges(neo4j_driver, sid, "MENTIONS"))
        assert mentions_v1 == ["A", "B", "C"]

        # v2: modify with entities A, D (B, C removed)
        v2 = WriteUnit(
            doc=_doc(sid, "modified"),
            chunks=_chunks(["chunk two"]),
            vectors=_vectors(1),
            entities=_entities("A", "D"),
        )
        writer.write(v2)

        mentions_v2 = sorted(_query_edges(neo4j_driver, sid, "MENTIONS"))
        assert mentions_v2 == ["A", "D"]

    def test_modified_cleans_stale_hints(
        self,
        writer: Writer,
        neo4j_driver: Driver,
    ) -> None:
        """v1 has wikilinks to X,Y; v2 has wikilink to X only. LINKS_TO Y gone."""
        sid = _unique_id()

        target_x = _unique_id("target_x")
        target_y = _unique_id("target_y")

        hint_x = GraphHint(
            subject_id=sid,
            subject_label="File",
            predicate="LINKS_TO",
            object_id=target_x,
            object_label="File",
        )
        hint_y = GraphHint(
            subject_id=sid,
            subject_label="File",
            predicate="LINKS_TO",
            object_id=target_y,
            object_label="File",
        )

        # v1: create with hints X, Y
        v1 = WriteUnit(
            doc=_doc(sid, "created", graph_hints=[hint_x, hint_y]),
            chunks=_chunks(["chunk one"]),
            vectors=_vectors(1),
        )
        writer.write(v1)

        hints_v1 = _query_hint_edges(neo4j_driver, sid)
        assert len(hints_v1) == 2
        target_ids_v1 = sorted(h["target_id"] for h in hints_v1)
        assert target_ids_v1 == sorted([target_x, target_y])

        # v2: modify with only hint X
        v2 = WriteUnit(
            doc=_doc(sid, "modified", graph_hints=[hint_x]),
            chunks=_chunks(["chunk two"]),
            vectors=_vectors(1),
        )
        writer.write(v2)

        hints_v2 = _query_hint_edges(neo4j_driver, sid)
        assert len(hints_v2) == 1
        assert hints_v2[0]["target_id"] == target_x

    def test_modified_cleans_stale_chunks(
        self,
        writer: Writer,
        neo4j_driver: Driver,
        qdrant_client: QdrantClient,
    ) -> None:
        """v1 has 5 chunks; v2 has 3. Assert exactly 3 Chunk nodes and 3 Qdrant points."""
        sid = _unique_id()

        # v1: 5 chunks
        v1_texts = [f"chunk {i}" for i in range(5)]
        v1 = WriteUnit(
            doc=_doc(sid, "created"),
            chunks=_chunks(v1_texts),
            vectors=_vectors(5),
        )
        writer.write(v1)

        assert len(_query_chunks(neo4j_driver, sid)) == 5
        assert _query_qdrant_points(qdrant_client, sid) == 5

        # v2: 3 chunks
        v2_texts = [f"new chunk {i}" for i in range(3)]
        v2 = WriteUnit(
            doc=_doc(sid, "modified"),
            chunks=_chunks(v2_texts),
            vectors=_vectors(3),
        )
        writer.write(v2)

        neo4j_chunks = _query_chunks(neo4j_driver, sid)
        assert len(neo4j_chunks) == 3
        assert all("new chunk" in t for t in neo4j_chunks)

        assert _query_qdrant_points(qdrant_client, sid) == 3

    def test_orphan_entity_cleanup(
        self,
        writer: Writer,
        neo4j_driver: Driver,
    ) -> None:
        """Entity B only mentioned by this source. After removing B, entity node deleted."""
        sid = _unique_id()

        # v1: create with entities A, B
        v1 = WriteUnit(
            doc=_doc(sid, "created"),
            chunks=_chunks(["chunk"]),
            vectors=_vectors(1),
            entities=_entities("OrphanTestA", "OrphanTestB"),
        )
        writer.write(v1)

        assert _query_entity_exists(neo4j_driver, "OrphanTestA")
        assert _query_entity_exists(neo4j_driver, "OrphanTestB")

        # v2: modify, only mention A (B becomes orphan)
        v2 = WriteUnit(
            doc=_doc(sid, "modified"),
            chunks=_chunks(["chunk v2"]),
            vectors=_vectors(1),
            entities=_entities("OrphanTestA"),
        )
        writer.write(v2)

        assert _query_entity_exists(neo4j_driver, "OrphanTestA")
        assert not _query_entity_exists(neo4j_driver, "OrphanTestB")

    def test_shared_entity_preserved(
        self,
        writer: Writer,
        neo4j_driver: Driver,
    ) -> None:
        """Entity A mentioned by source1 and source2. Modifying source1 to remove A
        should preserve A because source2 still references it."""
        sid1 = _unique_id("source1")
        sid2 = _unique_id("source2")

        # Create source1 mentioning SharedEntity
        v1_s1 = WriteUnit(
            doc=_doc(sid1, "created"),
            chunks=_chunks(["chunk"]),
            vectors=_vectors(1),
            entities=_entities("SharedEntity", "OnlyInSource1"),
        )
        writer.write(v1_s1)

        # Create source2 also mentioning SharedEntity
        v1_s2 = WriteUnit(
            doc=_doc(sid2, "created"),
            chunks=_chunks(["chunk"]),
            vectors=_vectors(1),
            entities=_entities("SharedEntity"),
        )
        writer.write(v1_s2)

        assert _query_entity_exists(neo4j_driver, "SharedEntity")
        assert _query_entity_exists(neo4j_driver, "OnlyInSource1")

        # Modify source1 to remove SharedEntity (and OnlyInSource1)
        v2_s1 = WriteUnit(
            doc=_doc(sid1, "modified"),
            chunks=_chunks(["updated chunk"]),
            vectors=_vectors(1),
            entities=_entities("NewEntity"),
        )
        writer.write(v2_s1)

        # SharedEntity preserved (still referenced by source2)
        assert _query_entity_exists(neo4j_driver, "SharedEntity")
        # OnlyInSource1 is now orphaned → deleted
        assert not _query_entity_exists(neo4j_driver, "OnlyInSource1")
        # NewEntity exists
        assert _query_entity_exists(neo4j_driver, "NewEntity")

    def test_created_operation_no_cleanup(
        self,
        writer: Writer,
        neo4j_driver: Driver,
    ) -> None:
        """'created' operation does not trigger any cleanup logic."""
        sid = _unique_id()

        # Create with entities A, B
        v1 = WriteUnit(
            doc=_doc(sid, "created"),
            chunks=_chunks(["chunk one"]),
            vectors=_vectors(1),
            entities=_entities("CreateTestA", "CreateTestB"),
        )
        writer.write(v1)

        # Create again (simulating duplicate create — no cleanup should happen)
        # The MERGE semantics mean this is idempotent, but cleanup should NOT fire.
        v2 = WriteUnit(
            doc=_doc(sid, "created"),
            chunks=_chunks(["chunk two"]),
            vectors=_vectors(1),
            entities=_entities("CreateTestA", "CreateTestC"),
        )
        writer.write(v2)

        # Both B and C should exist — B was not cleaned up because operation was "created"
        mentions = sorted(_query_edges(neo4j_driver, sid, "MENTIONS"))
        assert "CreateTestA" in mentions
        assert "CreateTestB" in mentions  # Not cleaned up
        assert "CreateTestC" in mentions

    def test_deleted_operation_unchanged(
        self,
        writer: Writer,
        neo4j_driver: Driver,
        qdrant_client: QdrantClient,
    ) -> None:
        """Deletion path removes source node, chunks, and vectors."""
        sid = _unique_id()

        # Create source with chunks and entities
        v1 = WriteUnit(
            doc=_doc(sid, "created"),
            chunks=_chunks(["chunk one", "chunk two"]),
            vectors=_vectors(2),
            entities=_entities("DeleteTestEntity"),
        )
        writer.write(v1)

        assert len(_query_chunks(neo4j_driver, sid)) == 2
        assert _query_qdrant_points(qdrant_client, sid) == 2

        # Delete the source
        delete_unit = WriteUnit(doc=_doc(sid, "deleted"))
        writer.write(delete_unit)

        # Source node and chunks should be gone
        assert len(_query_chunks(neo4j_driver, sid)) == 0
        assert _query_qdrant_points(qdrant_client, sid) == 0

        # Verify source node itself is deleted
        with neo4j_driver.session() as session:
            result = session.run(
                "MATCH (s {source_id: $sid}) RETURN count(s) AS cnt",
                sid=sid,
            )
            assert result.single()["cnt"] == 0
