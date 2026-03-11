"""Neo4j + Qdrant write layer for graph and vector storage.

Handles all writes to both stores in a single logical transaction boundary.
Neo4j writes are ACID within a single transaction; Qdrant writes are
best-effort eventual consistency.

Neo4j writes:
  - Upsert source nodes (File/Email/etc) via MERGE on source_id
  - Create/merge entity nodes
  - Create relationship edges
  - Create Chunk nodes linked via HAS_CHUNK
  - Write GraphHints directly (bypassing LLM)

Qdrant writes:
  - Upsert vectors to 'fieldnotes' collection (768-dim, cosine)
  - Payload: source_type, source_id, chunk_index, text, date

Deletions:
  - Remove Neo4j nodes and relationships for deleted source_ids
  - Remove Qdrant vectors matching the deleted source_id
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from neo4j import GraphDatabase, Driver, Session as Neo4jSession
from neo4j.exceptions import ServiceUnavailable, TransientError
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from worker.config import Neo4jConfig, QdrantConfig
from worker.parsers.base import GraphHint, ParsedDocument
from worker.pipeline.chunker import Chunk

logger = logging.getLogger(__name__)

_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

VECTOR_SIZE = 768  # Default; prefer QdrantConfig.vector_size for runtime use
COLLECTION_NAME = "fieldnotes"

_neo4j_retry = retry(
    retry=retry_if_exception_type((TransientError, ServiceUnavailable, OSError)),
    stop=stop_after_attempt(4),
    wait=wait_exponential_jitter(initial=0.5, max=10),
    before_sleep=lambda rs: logger.warning(
        "Neo4j call failed (%s), retry %d", rs.outcome.exception(), rs.attempt_number
    ),
    reraise=True,
)

_qdrant_retry = retry(
    retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
    stop=stop_after_attempt(4),
    wait=wait_exponential_jitter(initial=0.5, max=10),
    before_sleep=lambda rs: logger.warning(
        "Qdrant call failed (%s), retry %d", rs.outcome.exception(), rs.attempt_number
    ),
    reraise=True,
)


def _validate_cypher_identifier(value: str, context: str) -> str:
    """Validate that a value is safe for use as a Cypher label, type, or property key.

    Neo4j parameters cannot bind labels, relationship types, or property keys,
    so these must be interpolated into query strings. This function ensures only
    safe identifier characters are present to prevent Cypher injection.

    Raises ValueError if the value does not match ^[A-Za-z_][A-Za-z0-9_]*$.
    """
    if not _SAFE_IDENTIFIER_RE.match(value):
        raise ValueError(
            f"Unsafe Cypher identifier in {context}: {value!r}"
        )
    return value


@dataclass
class WriteUnit:
    """A fully processed document ready for storage.

    Produced by the pipeline orchestrator after chunking, embedding, and
    entity extraction. The writer consumes these without knowledge of how
    they were produced.
    """

    doc: ParsedDocument
    chunks: list[Chunk] = field(default_factory=list)
    vectors: list[list[float]] = field(default_factory=list)
    entities: list[dict[str, Any]] = field(default_factory=list)
    triples: list[dict[str, str]] = field(default_factory=list)


class Writer:
    """Writes processed documents to Neo4j and Qdrant."""

    def __init__(
        self,
        neo4j_cfg: Neo4jConfig | None = None,
        qdrant_cfg: QdrantConfig | None = None,
    ) -> None:
        neo4j_cfg = neo4j_cfg or Neo4jConfig()
        qdrant_cfg = qdrant_cfg or QdrantConfig()

        self._neo4j_driver: Driver = GraphDatabase.driver(
            neo4j_cfg.uri,
            auth=(neo4j_cfg.user, neo4j_cfg.password),
        )
        self._qdrant = QdrantClient(
            host=qdrant_cfg.host,
            port=qdrant_cfg.port,
        )
        self._collection = qdrant_cfg.collection or COLLECTION_NAME
        self._vector_size = qdrant_cfg.vector_size or VECTOR_SIZE
        self._ensure_qdrant_collection()

    @_qdrant_retry
    def _ensure_qdrant_collection(self) -> None:
        """Create the Qdrant collection if it does not exist."""
        collections = [
            c.name for c in self._qdrant.get_collections().collections
        ]
        if self._collection not in collections:
            self._qdrant.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._vector_size,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection %r", self._collection)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_existing_entities(self) -> list[dict[str, Any]]:
        """Query Neo4j for all existing Entity nodes.

        Returns a list of dicts with 'name', 'type', and 'confidence' keys,
        suitable for passing to the entity resolver.
        """
        with self._neo4j_driver.session() as session:
            result = session.run(
                "MATCH (e:Entity) RETURN e.name AS name, e.type AS type, "
                "e.confidence AS confidence"
            )
            return [
                {
                    "name": record["name"],
                    "type": record["type"] or "Concept",
                    "confidence": record["confidence"] or 0.75,
                }
                for record in result
            ]

    def write(self, unit: WriteUnit) -> None:
        """Write a single processed document to both stores.

        For deletions, removes all data associated with the source_id.
        For creates/modifications, upserts the source node, entities,
        edges, chunks, and vectors.

        Neo4j is written first (ACID transaction). Qdrant is retried on
        failure to avoid cross-store inconsistency. If Qdrant still fails
        after retries, the error is raised so the caller can re-process
        the document (Neo4j upserts are idempotent).
        """
        doc = unit.doc

        if doc.operation == "deleted":
            self._delete(doc.source_id, doc.source_type)
            return

        self._write_neo4j(unit)
        self._write_qdrant(unit)

        logger.info(
            "Wrote %s %s: %d chunks, %d entities, %d triples, %d hints",
            doc.node_label,
            doc.source_id,
            len(unit.chunks),
            len(unit.entities),
            len(unit.triples),
            len(doc.graph_hints),
        )

    def write_batch(self, units: list[WriteUnit]) -> None:
        """Write multiple units. Each unit is written independently."""
        for unit in units:
            self.write(unit)

    def __enter__(self) -> Writer:
        return self

    def __exit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Release connections."""
        self._neo4j_driver.close()
        self._qdrant.close()

    # ------------------------------------------------------------------
    # Neo4j writes
    # ------------------------------------------------------------------

    @_neo4j_retry
    def _write_neo4j(self, unit: WriteUnit) -> None:
        """Write to Neo4j with retry on transient errors."""
        with self._neo4j_driver.session() as session:
            session.execute_write(self._write_neo4j_tx, unit)

    @staticmethod
    def _write_neo4j_tx(tx: Any, unit: WriteUnit) -> None:
        """Execute all Neo4j writes within a single transaction."""
        doc = unit.doc

        # 1. Upsert source node
        _upsert_source_node(tx, doc)

        # 2. Upsert entity nodes and MENTIONS edges
        for entity in unit.entities:
            _upsert_entity(tx, entity)
            _merge_mentions_edge(tx, doc.source_id, entity["name"])

        # 3. Create relationship triples between entities
        for triple in unit.triples:
            _merge_entity_edge(tx, triple)

        # 4. Create Chunk nodes linked via HAS_CHUNK
        for chunk in unit.chunks:
            chunk_id = _chunk_node_id(doc.source_id, chunk.index)
            _upsert_chunk(tx, chunk_id, doc.source_id, chunk)

        # 5. Write GraphHints directly (bypass LLM)
        for hint in doc.graph_hints:
            _write_graph_hint(tx, hint)

    # ------------------------------------------------------------------
    # Qdrant writes
    # ------------------------------------------------------------------

    @_qdrant_retry
    def _write_qdrant(self, unit: WriteUnit) -> None:
        """Upsert chunk vectors to Qdrant.

        Uses deterministic UUID5 point IDs so upserts are idempotent —
        no pre-deletion required. Stale chunks (from a previous version
        with more chunks) are cleaned up AFTER the upsert succeeds, so
        a crash never leaves the source with zero vectors.
        """
        doc = unit.doc

        if not unit.vectors:
            return

        points = []
        for chunk, vector in zip(unit.chunks, unit.vectors):
            point_id = str(uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{doc.source_id}:{chunk.index}",
            ))
            payload = {
                "source_type": doc.source_type,
                "source_id": doc.source_id,
                "chunk_index": chunk.index,
                "text": chunk.text,
                "date": doc.source_metadata.get("date", ""),
            }
            points.append(PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            ))

        if points:
            self._qdrant.upsert(
                collection_name=self._collection,
                points=points,
            )

        # Clean up stale chunks from previous versions that had more
        # chunks than the current version. This runs AFTER upsert so
        # a crash here leaves extra (not missing) vectors — safe.
        self._delete_stale_qdrant_chunks(doc.source_id, len(unit.chunks))

    def _delete_stale_qdrant_chunks(
        self, source_id: str, current_chunk_count: int
    ) -> None:
        """Remove Qdrant vectors for chunk indices >= current_chunk_count.

        After a document is re-chunked with fewer chunks, old higher-index
        vectors become stale. This is safe to run after upsert — if it
        fails, we have extra vectors but no missing ones.
        """
        self._qdrant.delete(
            collection_name=self._collection,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="source_id",
                        match=MatchValue(value=source_id),
                    ),
                    FieldCondition(
                        key="chunk_index",
                        range=Range(gte=current_chunk_count),
                    ),
                ]
            ),
        )

    # ------------------------------------------------------------------
    # Deletions
    # ------------------------------------------------------------------

    def _delete(self, source_id: str, source_type: str) -> None:
        """Remove all Neo4j nodes and Qdrant vectors for a source_id."""
        self._delete_neo4j(source_id)
        self._delete_qdrant_vectors(source_id)

        logger.info("Deleted %s %s", source_type, source_id)

    @_neo4j_retry
    def _delete_neo4j(self, source_id: str) -> None:
        """Delete Neo4j nodes for a source_id with retry."""
        with self._neo4j_driver.session() as session:
            session.execute_write(self._delete_neo4j_tx, source_id)

    @staticmethod
    def _delete_neo4j_tx(tx: Any, source_id: str) -> None:
        """Remove the source node, its chunks, and all relationships."""
        # Delete chunk nodes linked to this source
        tx.run(
            """
            MATCH (s {source_id: $sid})-[:HAS_CHUNK]->(c:Chunk)
            DETACH DELETE c
            """,
            sid=source_id,
        )
        # Delete the source node itself (and any remaining relationships)
        tx.run(
            """
            MATCH (s {source_id: $sid})
            DETACH DELETE s
            """,
            sid=source_id,
        )

    @_qdrant_retry
    def _delete_qdrant_vectors(self, source_id: str) -> None:
        """Remove all Qdrant vectors matching a source_id."""
        self._qdrant.delete(
            collection_name=self._collection,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="source_id",
                        match=MatchValue(value=source_id),
                    )
                ]
            ),
        )


# ------------------------------------------------------------------
# Neo4j helper functions (called within a transaction)
# ------------------------------------------------------------------


def _upsert_source_node(tx: Any, doc: ParsedDocument) -> None:
    """MERGE a source node on source_id and set its properties."""
    label = _validate_cypher_identifier(doc.node_label, "node_label")
    props = {
        "source_id": doc.source_id,
        "source_type": doc.source_type,
        **doc.node_props,
    }
    for k in props:
        _validate_cypher_identifier(k, "node_props key")
    # Build a dynamic SET clause from node_props
    set_parts = ", ".join(f"s.{k} = ${k}" for k in props)
    query = (
        f"MERGE (s:{label} {{source_id: $source_id}}) "
        f"SET {set_parts}"
    )
    tx.run(query, **props)


def _upsert_entity(tx: Any, entity: dict[str, Any]) -> None:
    """MERGE an Entity node on name, setting type and confidence."""
    tx.run(
        """
        MERGE (e:Entity {name: $name})
        SET e.type = $type,
            e.confidence = $confidence
        """,
        name=entity["name"],
        type=entity.get("type", "Concept"),
        confidence=entity.get("confidence", 0.75),
    )


def _merge_mentions_edge(tx: Any, source_id: str, entity_name: str) -> None:
    """Create a MENTIONS edge from a source node to an Entity."""
    tx.run(
        """
        MATCH (s {source_id: $sid})
        MATCH (e:Entity {name: $name})
        MERGE (s)-[:MENTIONS]->(e)
        """,
        sid=source_id,
        name=entity_name,
    )


def _merge_entity_edge(tx: Any, triple: dict[str, str]) -> None:
    """Create a relationship edge between two entities from a triple."""
    predicate = triple["predicate"].replace(" ", "_").upper()
    _validate_cypher_identifier(predicate, "triple predicate")
    tx.run(
        f"""
        MERGE (s:Entity {{name: $subject}})
        MERGE (o:Entity {{name: $object}})
        MERGE (s)-[:{predicate}]->(o)
        """,
        subject=triple["subject"],
        object=triple["object"],
    )


def _upsert_chunk(
    tx: Any, chunk_id: str, source_id: str, chunk: Chunk
) -> None:
    """Create a Chunk node and link it to the source via HAS_CHUNK."""
    tx.run(
        """
        MERGE (c:Chunk {id: $cid})
        SET c.text = $text,
            c.source_id = $sid,
            c.chunk_index = $idx
        WITH c
        MATCH (s {source_id: $sid})
        MERGE (s)-[:HAS_CHUNK]->(c)
        """,
        cid=chunk_id,
        text=chunk.text,
        sid=source_id,
        idx=chunk.index,
    )


def _write_graph_hint(tx: Any, hint: GraphHint) -> None:
    """Write a pre-extracted graph fact directly to Neo4j."""
    subject_label = _validate_cypher_identifier(hint.subject_label, "hint subject_label")
    object_label = _validate_cypher_identifier(hint.object_label, "hint object_label")
    predicate = hint.predicate.replace(" ", "_").upper()
    _validate_cypher_identifier(predicate, "hint predicate")

    # Ensure subject node exists
    tx.run(
        f"MERGE (s:{subject_label} {{source_id: $sid}})",
        sid=hint.subject_id,
    )
    # Ensure object node exists with its properties
    obj_props = {"source_id": hint.object_id, **hint.object_props}
    for k in obj_props:
        _validate_cypher_identifier(k, "hint object_props key")
    set_parts = ", ".join(f"o.{k} = ${k}" for k in obj_props)
    tx.run(
        f"MERGE (o:{object_label} {{source_id: $oid}}) SET {set_parts}",
        oid=hint.object_id,
        **obj_props,
    )
    # Create the relationship
    tx.run(
        f"""
        MATCH (s:{subject_label} {{source_id: $sid}})
        MATCH (o:{object_label} {{source_id: $oid}})
        MERGE (s)-[:{predicate}]->(o)
        """,
        sid=hint.subject_id,
        oid=hint.object_id,
    )


def _chunk_node_id(source_id: str, chunk_index: int) -> str:
    """Deterministic chunk node ID from source_id and index."""
    return f"{source_id}:chunk:{chunk_index}"
