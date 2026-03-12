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

import hashlib
import logging
import re
import threading
import time
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
from worker.metrics import (
    NEO4J_WRITE_DURATION,
    QDRANT_WRITE_DURATION,
    observe_duration,
)
from worker.parsers.base import GraphHint, ParsedDocument
from worker.pipeline.chunker import Chunk
from worker.pipeline.resolver import (
    CrossSourceMatch,
    resolve_cross_source,
)

logger = logging.getLogger(__name__)

_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

VECTOR_SIZE = 768  # Default; prefer QdrantConfig.vector_size for runtime use
COLLECTION_NAME = "fieldnotes"
ENTITY_CACHE_TTL = 60  # seconds — avoid repeated full scans during batch processing
FULLTEXT_INDEX_NAME = "entity_name_fulltext"
MAX_ENTITY_NAME_LEN = 256

# Allowed relationship types for LLM-generated entity triples.
# Predicates not in this set are mapped to RELATED_TO to prevent
# semantic pollution from unconstrained LLM output.
ALLOWED_PREDICATES: frozenset[str] = frozenset({
    "RELATED_TO",
    "WORKS_AT",
    "WORKS_FOR",
    "WORKS_ON",
    "KNOWS",
    "COLLABORATES_WITH",
    "USES",
    "USED_BY",
    "CREATED_BY",
    "CREATED",
    "PART_OF",
    "BELONGS_TO",
    "CONTAINS",
    "DEPENDS_ON",
    "IS_A",
    "HAS_A",
    "LOCATED_IN",
    "MANAGES",
    "REPORTS_TO",
    "CONTRIBUTED_TO",
    "BASED_ON",
    "DERIVED_FROM",
    "ASSOCIATED_WITH",
    "FOUNDED",
    "FUNDED_BY",
    "PUBLISHED",
    "AUTHORED",
    "EMPLOYED_BY",
    "AFFILIATED_WITH",
    "MEMBER_OF",
    "MENTIONS",
    "REFERENCES",
    "IMPLEMENTS",
    "EXTENDS",
    "INTEGRATES_WITH",
    "SUCCEEDED_BY",
    "PRECEDED_BY",
    "SAME_AS",
    "SIMILAR_TO",
    "OPPOSITE_OF",
    "INFLUENCED_BY",
    "SUPPORTS",
    "CONFLICTS_WITH",
    "ACQUIRED_BY",
    "INVESTED_IN",
    "DEVELOPED_BY",
    "MAINTAINED_BY",
    "OWNED_BY",
    "LED_BY",
    "TAUGHT_BY",
    "ATTENDED",
    "PARTICIPATED_IN",
    "SPOKE_AT",
    "INSTALLED_VIA",
    "CATEGORIZED_AS",
    "PROVIDES",
})

_LUCENE_SPECIAL_RE = re.compile(r'([+\-&|!(){}[\]^"~*?:\\/])')

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
    depicts_entities: list[dict[str, Any]] = field(default_factory=list)


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
        self._entity_cache: list[dict[str, Any]] | None = None
        self._entity_cache_ts: float = 0.0
        self._entity_cache_lock = threading.Lock()
        self._ensure_qdrant_collection()
        self._ensure_entity_fulltext_index()

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

    def mark_vision_processed(self, source_id: str) -> None:
        """Set vision_processed=True on the Image node after vision completes.

        Called by the pipeline after vision extraction, embedding, and writing
        are all finished for an image document.
        """
        self._mark_vision_processed_neo4j(source_id)
        logger.debug("Marked vision_processed=True for %s", source_id)

    @_neo4j_retry
    def _mark_vision_processed_neo4j(self, source_id: str) -> None:
        """Set vision_processed flag in Neo4j with retry."""
        with self._neo4j_driver.session() as session:
            session.run(
                "MATCH (n:Image {source_id: $sid}) "
                "SET n.vision_processed = true",
                sid=source_id,
            )

    def fetch_existing_entities(self) -> list[dict[str, Any]]:
        """Query Neo4j for all existing Entity nodes, with TTL cache.

        Returns a list of dicts with 'name', 'type', and 'confidence' keys,
        suitable for passing to the entity resolver. Results are cached for
        ENTITY_CACHE_TTL seconds to avoid repeated full table scans during
        batch processing.
        """
        with self._entity_cache_lock:
            now = time.monotonic()
            if (
                self._entity_cache is not None
                and (now - self._entity_cache_ts) < ENTITY_CACHE_TTL
            ):
                return self._entity_cache

            with self._neo4j_driver.session() as session:
                result = session.run(
                    "MATCH (e:Entity) RETURN e.name AS name, e.type AS type, "
                    "e.confidence AS confidence"
                )
                entities = [
                    {
                        "name": record["name"],
                        "type": record["type"] or "Concept",
                        "confidence": record["confidence"] or 0.75,
                    }
                    for record in result
                ]

            self._entity_cache = entities
            self._entity_cache_ts = now
            return entities

    def invalidate_entity_cache(self) -> None:
        """Clear the entity cache so the next fetch hits Neo4j."""
        with self._entity_cache_lock:
            self._entity_cache = None
            self._entity_cache_ts = 0.0

    def fetch_candidate_entities(self, names: list[str]) -> list[dict[str, Any]]:
        """Fetch entities similar to the given names using full-text index.

        Uses Neo4j full-text index with Lucene fuzzy queries for pre-filtering,
        reducing the candidate set for fuzzy matching. Falls back to the cached
        full entity list if the full-text query fails.
        """
        if not names:
            return []

        # Build Lucene fuzzy query from extracted entity name tokens
        words: set[str] = set()
        for name in names:
            for word in name.split():
                escaped = _LUCENE_SPECIAL_RE.sub(r"\\\1", word)
                if escaped:
                    words.add(escaped)
        if not words:
            return self.fetch_existing_entities()

        query = " ".join(f"{w}~" for w in words)

        try:
            with self._neo4j_driver.session() as session:
                result = session.run(
                    "CALL db.index.fulltext.queryNodes("
                    "$index_name, $query"
                    ") YIELD node "
                    "RETURN node.name AS name, node.type AS type, "
                    "node.confidence AS confidence",
                    index_name=FULLTEXT_INDEX_NAME,
                    query=query,
                )
                candidates = [
                    {
                        "name": record["name"],
                        "type": record["type"] or "Concept",
                        "confidence": record["confidence"] or 0.75,
                    }
                    for record in result
                ]
            if candidates:
                return candidates
            # No candidates found — fall back to full list (handles edge cases
            # where full-text tokenization misses valid matches)
            return self.fetch_existing_entities()
        except Exception:
            logger.debug(
                "Full-text index query failed, falling back to cached full scan",
                exc_info=True,
            )
            return self.fetch_existing_entities()

    @_neo4j_retry
    def _ensure_entity_fulltext_index(self) -> None:
        """Create full-text index on Entity.name if it doesn't exist."""
        try:
            with self._neo4j_driver.session() as session:
                session.run(
                    "CREATE FULLTEXT INDEX entity_name_fulltext IF NOT EXISTS "
                    "FOR (e:Entity) ON EACH [e.name]"
                )
        except Exception:
            logger.debug(
                "Could not create full-text index (may already exist or "
                "Neo4j version doesn't support IF NOT EXISTS)",
                exc_info=True,
            )

    def reconcile_persons(self) -> int:
        """Scan for Person nodes with matching emails across source types and merge them.

        For each group of Person nodes sharing a normalized email, keeps the
        longest name and creates SAME_AS edges between source-specific IDs.

        Returns the number of Person nodes updated.
        """
        return self._reconcile_persons_neo4j()

    @_neo4j_retry
    def _reconcile_persons_neo4j(self) -> int:
        """Run Person dedup in Neo4j: prefer longer name, add SAME_AS edges."""
        with self._neo4j_driver.session() as session:
            # Update Person names to prefer the longest variant for each email
            result = session.run(
                """
                MATCH (p:Person)
                WHERE p.email IS NOT NULL
                WITH toLower(trim(p.email)) AS norm_email, collect(p) AS persons
                WHERE size(persons) > 1
                WITH norm_email, persons,
                     reduce(best = '', p IN persons |
                         CASE WHEN size(p.name) > size(best) THEN p.name ELSE best END
                     ) AS best_name
                UNWIND persons AS p
                SET p.name = best_name
                RETURN count(p) AS updated
                """
            )
            updated = result.single()["updated"]

            # Create SAME_AS edges between Person nodes with the same email
            session.run(
                """
                MATCH (a:Person), (b:Person)
                WHERE a.email IS NOT NULL AND b.email IS NOT NULL
                  AND toLower(trim(a.email)) = toLower(trim(b.email))
                  AND id(a) < id(b)
                MERGE (a)-[:SAME_AS]->(b)
                """
            )

            if updated > 0:
                logger.info("Reconciled %d Person nodes by email", updated)
            return updated

    def resolve_cross_source_entities(self) -> int:
        """Find and link Entity nodes mentioned across different source types.

        Queries Neo4j for Entity nodes grouped by source type (via MENTIONS edges),
        then runs cross-source fuzzy matching to create SAME_AS edges between
        high-confidence matches.

        Returns the number of SAME_AS edges created.
        """
        return self._resolve_cross_source_neo4j()

    @_neo4j_retry
    def _resolve_cross_source_neo4j(self) -> int:
        """Query entities by source type and create SAME_AS edges for matches."""
        with self._neo4j_driver.session() as session:
            # Collect entities grouped by source type. For Person nodes,
            # also include the email property for email-based matching.
            result = session.run(
                """
                MATCH (s)-[:MENTIONS]->(e:Entity)
                WHERE s.source_type IS NOT NULL
                WITH s.source_type AS src_type, e
                WITH src_type, collect(DISTINCT {
                    name: e.name,
                    type: e.type,
                    confidence: e.confidence
                }) AS entities
                RETURN src_type, entities
                """
            )
            entities_by_source: dict[str, list[dict]] = {}
            for record in result:
                entities_by_source[record["src_type"]] = [
                    {
                        "name": e["name"],
                        "type": e["type"] or "Concept",
                        "confidence": e["confidence"] or 0.75,
                    }
                    for e in record["entities"]
                ]

            # Also gather Person nodes with emails for email-based matching
            person_result = session.run(
                """
                MATCH (p:Person)
                WHERE p.email IS NOT NULL AND p.name IS NOT NULL
                OPTIONAL MATCH (s)-[:MENTIONS]->(e:Entity)
                WHERE toLower(e.name) = toLower(p.name)
                WITH p, s.source_type AS src_type
                WHERE src_type IS NOT NULL
                RETURN src_type, collect(DISTINCT {
                    name: p.name,
                    type: 'Person',
                    confidence: 1.0,
                    email: p.email
                }) AS person_entities
                """
            )
            for record in person_result:
                src_type = record["src_type"]
                person_ents = [
                    {
                        "name": e["name"],
                        "type": "Person",
                        "confidence": e["confidence"],
                        "email": e["email"],
                    }
                    for e in record["person_entities"]
                ]
                if src_type in entities_by_source:
                    # Merge person entities (avoid duplicates by name)
                    existing_names = {
                        e["name"].lower() for e in entities_by_source[src_type]
                    }
                    for pe in person_ents:
                        if pe["name"].lower() not in existing_names:
                            entities_by_source[src_type].append(pe)
                else:
                    entities_by_source[src_type] = person_ents

            if len(entities_by_source) < 2:
                return 0

            # Run cross-source matching
            matches = resolve_cross_source(entities_by_source)

            # Create SAME_AS edges for matches
            created = 0
            for match in matches:
                result = session.run(
                    """
                    MATCH (a:Entity {name: $name_a})
                    MATCH (b:Entity {name: $name_b})
                    WHERE NOT (a)-[:SAME_AS]-(b)
                      AND id(a) <> id(b)
                    MERGE (a)-[r:SAME_AS]->(b)
                    SET r.confidence = $confidence,
                        r.match_type = $match_type,
                        r.cross_source = true
                    RETURN count(r) AS cnt
                    """,
                    name_a=match.entity_a,
                    name_b=match.entity_b,
                    confidence=match.confidence,
                    match_type=match.match_type,
                )
                cnt = result.single()["cnt"]
                created += cnt

            if created > 0:
                logger.info(
                    "Cross-source resolution: created %d SAME_AS edges from %d matches",
                    created,
                    len(matches),
                )
            return created

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
        with observe_duration(NEO4J_WRITE_DURATION):
            with self._neo4j_driver.session() as session:
                session.execute_write(self._write_neo4j_tx, unit)

    @staticmethod
    def _write_neo4j_tx(tx: Any, unit: WriteUnit) -> None:
        """Execute all Neo4j writes within a single transaction."""
        doc = unit.doc

        # 0. Clean stale MENTIONS edges on source modification so only
        #    edges for the current content are present after re-write.
        if doc.operation == "modified":
            _clean_source_edges(tx, doc.source_id, "MENTIONS")

        # 1. Upsert source node
        _upsert_source_node(tx, doc)

        # 2. Upsert entity nodes and MENTIONS edges
        for entity in unit.entities:
            _upsert_entity(tx, entity)
            _merge_mentions_edge(tx, doc.source_id, entity["name"])

        # 3. Create relationship triples between entities
        for triple in unit.triples:
            _merge_entity_edge(tx, triple)

        # 4. Delete stale Chunk nodes on modification (before writing new ones)
        if doc.operation == "modified":
            tx.run(
                "MATCH (s {source_id: $sid})-[:HAS_CHUNK]->(c:Chunk) "
                "DETACH DELETE c",
                sid=doc.source_id,
            )

        # 5. Create Chunk nodes linked via HAS_CHUNK
        for chunk in unit.chunks:
            chunk_id = _chunk_node_id(doc.source_id, chunk.index)
            _upsert_chunk(tx, chunk_id, doc.source_id, chunk)

        # 6. Clean stale hint edges on modification, then write new hints
        if doc.operation == "modified":
            tx.run(
                "MATCH ({source_id: $uri})-[r {hint: true}]->() DELETE r",
                uri=doc.source_id,
            )
        for hint in doc.graph_hints:
            _write_graph_hint(tx, hint)

        # 7. Upsert DEPICTS edges (vision-extracted entities)
        for entity in unit.depicts_entities:
            _upsert_entity(tx, entity)
            _merge_depicts_edge(tx, doc.source_id, entity["name"])

        # 8. Create ATTACHED_TO edge (Image→File for embedded images)
        parent_source_id = doc.node_props.get("parent_source_id")
        if doc.node_label == "Image" and parent_source_id:
            _merge_attached_to_edge(tx, doc.source_id, parent_source_id)

        # 9. Clean up orphaned Entity nodes after source modification.
        #    After deleting stale MENTIONS edges and re-writing current ones,
        #    some Entity nodes may have zero remaining incoming edges.
        if doc.operation == "modified":
            _cleanup_orphan_entities(tx)

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

        with observe_duration(QDRANT_WRITE_DURATION):
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
        **doc.node_props,
        "source_id": doc.source_id,
        "source_type": doc.source_type,
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


def _truncate_entity_name(name: str) -> str:
    """Truncate an entity name to MAX_ENTITY_NAME_LEN with hash disambiguation.

    If the name exceeds the limit, appends a short hash of the full name to
    avoid collisions where two distinct long names share a common prefix.
    """
    if len(name) <= MAX_ENTITY_NAME_LEN:
        return name

    suffix = hashlib.sha256(name.encode()).hexdigest()[:8]
    # Leave room for the hash suffix: "…<truncated>_<8-char-hash>"
    truncated = name[: MAX_ENTITY_NAME_LEN - 9] + "_" + suffix
    logger.warning(
        "Entity name truncated from %d to %d chars (hash=%s): %.80s...",
        len(name),
        len(truncated),
        suffix,
        name[:80],
    )
    return truncated


def _upsert_entity(tx: Any, entity: dict[str, Any]) -> None:
    """MERGE an Entity node on name, setting type and confidence."""
    name = _truncate_entity_name(entity["name"])
    tx.run(
        """
        MERGE (e:Entity {name: $name})
        SET e.type = $type,
            e.confidence = $confidence
        """,
        name=name,
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
        name=_truncate_entity_name(entity_name),
    )


def _merge_depicts_edge(tx: Any, source_id: str, entity_name: str) -> None:
    """Create a DEPICTS edge from a source node to an Entity."""
    tx.run(
        """
        MATCH (s {source_id: $sid})
        MATCH (e:Entity {name: $name})
        MERGE (s)-[:DEPICTS]->(e)
        """,
        sid=source_id,
        name=_truncate_entity_name(entity_name),
    )


def _clean_source_edges(tx: Any, source_id: str, edge_type: str) -> None:
    """Delete all edges of a given type from a source node.

    Used to remove stale edges before re-writing them on source modification,
    ensuring only edges corresponding to the current content remain.
    """
    edge_type = _validate_cypher_identifier(edge_type, "edge_type")
    tx.run(
        f"MATCH (s {{source_id: $sid}})-[r:{edge_type}]->() DELETE r",
        sid=source_id,
    )


def _cleanup_orphan_entities(tx: Any) -> int:
    """Delete Entity nodes with no incoming edges from any source.

    After stale MENTIONS edges are removed and new ones written, some Entity
    nodes may become orphaned — no MENTIONS, LINKS_TO, DEPICTS, or any other
    incoming relationship.  These pollute query results and topic clustering.

    Returns the number of orphaned Entity nodes deleted.
    """
    result = tx.run(
        "MATCH (e:Entity) "
        "WHERE NOT ()-[]->(e) "
        "DETACH DELETE e "
        "RETURN count(e) AS removed"
    )
    removed = result.single()["removed"]
    if removed:
        logger.info("Cleaned up %d orphaned Entity nodes", removed)
    return removed


def _merge_attached_to_edge(tx: Any, image_source_id: str, parent_source_id: str) -> None:
    """Create an ATTACHED_TO edge from an Image node to its parent File node."""
    tx.run(
        """
        MATCH (img:Image {source_id: $img_sid})
        MATCH (f:File {source_id: $parent_sid})
        MERGE (img)-[:ATTACHED_TO]->(f)
        """,
        img_sid=image_source_id,
        parent_sid=parent_source_id,
    )


def _merge_entity_edge(tx: Any, triple: dict[str, str]) -> None:
    """Create a relationship edge between two entities from a triple.

    The predicate is validated against ALLOWED_PREDICATES. Unknown predicates
    are mapped to RELATED_TO to prevent semantic pollution from unconstrained
    LLM output.
    """
    predicate = triple["predicate"].replace(" ", "_").upper()
    try:
        _validate_cypher_identifier(predicate, "triple predicate")
    except ValueError:
        logger.warning(
            "Invalid predicate %r mapped to RELATED_TO (subject=%r, object=%r)",
            triple["predicate"],
            triple["subject"],
            triple["object"],
        )
        predicate = "RELATED_TO"
    if predicate not in ALLOWED_PREDICATES:
        logger.warning(
            "Unknown predicate %r mapped to RELATED_TO (subject=%r, object=%r)",
            predicate,
            triple["subject"],
            triple["object"],
        )
        predicate = "RELATED_TO"
    tx.run(
        f"""
        MERGE (s:Entity {{name: $subject}})
        MERGE (o:Entity {{name: $object}})
        MERGE (s)-[:{predicate}]->(o)
        """,
        subject=_truncate_entity_name(triple["subject"]),
        object=_truncate_entity_name(triple["object"]),
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


def _build_hint_set_parts(
    alias: str, props: dict[str, Any], is_person: bool
) -> str:
    """Build a Cypher SET clause for graph hint node properties.

    For Person nodes, the ``name`` property uses a conditional update that
    keeps the longer (more complete) name — e.g. "M. Smith" won't overwrite
    "Markus Smith".  All other properties are set unconditionally.
    """
    parts: list[str] = []
    for k in props:
        if is_person and k == "name":
            parts.append(
                f"{alias}.name = CASE "
                f"WHEN {alias}.name IS NOT NULL "
                f"AND size({alias}.name) >= size(${alias}_{k}) "
                f"THEN {alias}.name ELSE ${alias}_{k} END"
            )
        else:
            parts.append(f"{alias}.{k} = ${alias}_{k}")
    return ", ".join(parts)


def _write_graph_hint(tx: Any, hint: GraphHint) -> None:
    """Write a pre-extracted graph fact directly to Neo4j.

    Supports custom merge keys per node type. For example, Person nodes
    MERGE on ``email``, Thread nodes on ``thread_id``, while generic nodes
    fall back to ``source_id``. All property values are passed as Cypher
    parameters (no f-string interpolation of values).
    """
    subject_label = _validate_cypher_identifier(hint.subject_label, "hint subject_label")
    object_label = _validate_cypher_identifier(hint.object_label, "hint object_label")
    predicate = hint.predicate.replace(" ", "_").upper()
    _validate_cypher_identifier(predicate, "hint predicate")

    subj_merge_key = _validate_cypher_identifier(
        hint.subject_merge_key, "hint subject_merge_key"
    )
    obj_merge_key = _validate_cypher_identifier(
        hint.object_merge_key, "hint object_merge_key"
    )

    # --- Subject node ---
    subj_props = {"source_id": hint.subject_id, **hint.subject_props}
    for k in subj_props:
        _validate_cypher_identifier(k, "hint subject_props key")
    # Determine merge value: use subject_props if the key exists there,
    # otherwise fall back to source_id (the default merge key).
    subj_merge_val = subj_props.get(subj_merge_key, hint.subject_id)
    subj_set_parts = _build_hint_set_parts(
        "s", subj_props, subject_label == "Person"
    )
    subj_params = {f"s_{k}": v for k, v in subj_props.items()}
    subj_params["s_merge"] = subj_merge_val
    tx.run(
        f"MERGE (s:{subject_label} {{{subj_merge_key}: $s_merge}}) "
        f"SET {subj_set_parts}",
        **subj_params,
    )

    # --- Object node ---
    obj_props = {"source_id": hint.object_id, **hint.object_props}
    for k in obj_props:
        _validate_cypher_identifier(k, "hint object_props key")
    obj_merge_val = obj_props.get(obj_merge_key, hint.object_id)
    obj_set_parts = _build_hint_set_parts(
        "o", obj_props, object_label == "Person"
    )
    obj_params = {f"o_{k}": v for k, v in obj_props.items()}
    obj_params["o_merge"] = obj_merge_val
    tx.run(
        f"MERGE (o:{object_label} {{{obj_merge_key}: $o_merge}}) "
        f"SET {obj_set_parts}",
        **obj_params,
    )

    # --- Relationship edge (tagged hint=true for cleanup on modification) ---
    tx.run(
        f"MATCH (s:{subject_label} {{{subj_merge_key}: $s_merge}}) "
        f"MATCH (o:{object_label} {{{obj_merge_key}: $o_merge}}) "
        f"MERGE (s)-[r:{predicate}]->(o) "
        f"SET r.hint = true",
        s_merge=subj_merge_val,
        o_merge=obj_merge_val,
    )


def _chunk_node_id(source_id: str, chunk_index: int) -> str:
    """Deterministic chunk node ID from source_id and index."""
    return f"{source_id}:chunk:{chunk_index}"
