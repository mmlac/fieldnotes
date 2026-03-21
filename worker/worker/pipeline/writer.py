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

from neo4j import GraphDatabase, Driver
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

from worker.circuit_breaker import CircuitBreaker, CircuitOpenError
from worker.config import Neo4jConfig, QdrantConfig
from worker.log_sanitizer import sanitize_exception
from worker.metrics import (
    CIRCUIT_BREAKER_REJECTIONS,
    NEO4J_WRITE_DURATION,
    QDRANT_WRITE_DURATION,
    observe_duration,
)
from worker.parsers.base import GraphHint, ParsedDocument
from worker.pipeline.chunker import Chunk
from worker.pipeline.resolver import (
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
ALLOWED_PREDICATES: frozenset[str] = frozenset(
    {
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
    }
)

_LUCENE_SPECIAL_RE = re.compile(r'([+\-&|!(){}[\]^"~*?:\\/])')

_neo4j_retry = retry(
    retry=retry_if_exception_type((TransientError, ServiceUnavailable, OSError)),
    stop=stop_after_attempt(4),
    wait=wait_exponential_jitter(initial=0.5, max=10),
    before_sleep=lambda rs: logger.warning(
        "Neo4j call failed (%s), retry %d",
        sanitize_exception(rs.outcome.exception()),
        rs.attempt_number,
    ),
    reraise=True,
)

_qdrant_retry = retry(
    retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
    stop=stop_after_attempt(4),
    wait=wait_exponential_jitter(initial=0.5, max=10),
    before_sleep=lambda rs: logger.warning(
        "Qdrant call failed (%s), retry %d",
        sanitize_exception(rs.outcome.exception()),
        rs.attempt_number,
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
        raise ValueError(f"Unsafe Cypher identifier in {context}: {value!r}")
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
        try:
            self._qdrant = QdrantClient(
                host=qdrant_cfg.host,
                port=qdrant_cfg.port,
            )
            self._collection = qdrant_cfg.collection or COLLECTION_NAME
            self._vector_size = qdrant_cfg.vector_size or VECTOR_SIZE
            self._entity_cache: list[dict[str, Any]] | None = None
            self._entity_cache_ts: float = 0.0
            self._entity_cache_lock = threading.Lock()

            # Circuit breakers for downstream services
            self.neo4j_breaker = CircuitBreaker(
                "neo4j",
                failure_threshold=5,
                recovery_timeout=60.0,
            )
            self.qdrant_breaker = CircuitBreaker(
                "qdrant",
                failure_threshold=5,
                recovery_timeout=60.0,
            )

            self._ensure_qdrant_collection()
            self._ensure_neo4j_schema()
        except Exception:
            self._neo4j_driver.close()
            raise

    @_qdrant_retry
    def _ensure_qdrant_collection(self) -> None:
        """Create the Qdrant collection if it does not exist."""
        collections = [c.name for c in self._qdrant.get_collections().collections]
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
                "MATCH (n:Image {source_id: $sid}) SET n.vision_processed = true",
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
    def _ensure_neo4j_schema(self) -> None:
        """Create all required Neo4j indexes and uniqueness constraints.

        Idempotent — uses IF NOT EXISTS throughout.  Failures are logged at
        WARNING level and swallowed so a single unsupported statement does not
        prevent startup; each statement is executed independently so a failure
        on one does not block the rest.

        Indexes created:
        - Uniqueness constraint on Entity.name (also serves as B-tree index)
        - Uniqueness constraint on Chunk.id
        - B-tree index on source_id for all source node labels
        - B-tree index on Person.email (reconcile_persons and hint MERGE)
        - B-tree index on Thread.thread_id (hint MERGE/MATCH)
        - Fulltext index on Entity.name (fuzzy candidate lookup)
        """
        ddl_statements = [
            # Uniqueness constraints (implicitly create B-tree indexes)
            (
                "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
            ),
            (
                "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS "
                "FOR (c:Chunk) REQUIRE c.id IS UNIQUE"
            ),
            # B-tree indexes on source_id for all source node labels
            "CREATE INDEX file_source_id IF NOT EXISTS FOR (n:File) ON (n.source_id)",
            "CREATE INDEX email_source_id IF NOT EXISTS FOR (n:Email) ON (n.source_id)",
            "CREATE INDEX image_source_id IF NOT EXISTS FOR (n:Image) ON (n.source_id)",
            "CREATE INDEX application_source_id IF NOT EXISTS FOR (n:Application) ON (n.source_id)",
            "CREATE INDEX tool_source_id IF NOT EXISTS FOR (n:Tool) ON (n.source_id)",
            "CREATE INDEX task_source_id IF NOT EXISTS FOR (n:Task) ON (n.source_id)",
            "CREATE INDEX commit_source_id IF NOT EXISTS FOR (n:Commit) ON (n.source_id)",
            # Indexes for hint-based node lookups
            "CREATE INDEX person_email IF NOT EXISTS FOR (n:Person) ON (n.email)",
            "CREATE INDEX thread_thread_id IF NOT EXISTS FOR (n:Thread) ON (n.thread_id)",
            # Fulltext index for fuzzy entity candidate lookup
            (
                "CREATE FULLTEXT INDEX entity_name_fulltext IF NOT EXISTS "
                "FOR (e:Entity) ON EACH [e.name]"
            ),
        ]
        with self._neo4j_driver.session() as session:
            for stmt in ddl_statements:
                try:
                    session.run(stmt)
                except Exception:
                    logger.warning(
                        "Could not apply Neo4j schema statement (skipping): %s",
                        stmt,
                        exc_info=True,
                    )

    @_neo4j_retry
    def _ensure_entity_fulltext_index(self) -> None:
        """Create full-text index on Entity.name if it doesn't exist.

        .. deprecated::
            Superseded by :meth:`_ensure_neo4j_schema`.  Kept for callers
            that reference it directly (e.g. tests and CLI tools).
        """
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

            # Create SAME_AS edges between Person nodes with the same email.
            # Group by email first to avoid a cartesian product across all Person nodes.
            session.run(
                """
                MATCH (p:Person)
                WHERE p.email IS NOT NULL
                WITH toLower(trim(p.email)) AS norm_email, collect(p) AS persons
                WHERE size(persons) > 1
                UNWIND persons AS a
                UNWIND persons AS b
                WITH a, b
                WHERE id(a) < id(b)
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

            # Batch-create SAME_AS edges for all matches in a single UNWIND query
            created = 0
            if matches:
                result = session.run(
                    """
                    UNWIND $matches AS m
                    MATCH (a:Entity {name: m.name_a})
                    MATCH (b:Entity {name: m.name_b})
                    WHERE NOT (a)-[:SAME_AS]-(b)
                      AND id(a) <> id(b)
                    MERGE (a)-[r:SAME_AS]->(b)
                    SET r.confidence = m.confidence,
                        r.match_type = m.match_type,
                        r.cross_source = true
                    RETURN count(r) AS cnt
                    """,
                    matches=[
                        {
                            "name_a": match.entity_a,
                            "name_b": match.entity_b,
                            "confidence": match.confidence,
                            "match_type": match.match_type,
                        }
                        for match in matches
                    ],
                )
                created = result.single()["cnt"]

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

        If a circuit breaker is open, raises CircuitOpenError so the
        pipeline can mark the document for retry on the next run.

        Neo4j is written first (ACID transaction). Qdrant is retried on
        failure to avoid cross-store inconsistency. If Qdrant still fails
        after retries, the error is raised so the caller can re-process
        the document (Neo4j upserts are idempotent).
        """
        doc = unit.doc

        if doc.operation == "deleted":
            self._delete(doc.source_id, doc.source_type)
            return

        if not self.neo4j_breaker.allow_request():
            CIRCUIT_BREAKER_REJECTIONS.labels(service="neo4j").inc()
            raise CircuitOpenError("neo4j")

        try:
            self._write_neo4j(unit)
            self.neo4j_breaker.record_success()
            if unit.entities or unit.depicts_entities:
                self.invalidate_entity_cache()
        except CircuitOpenError:
            raise
        except Exception:
            self.neo4j_breaker.record_failure()
            raise

        if not self.qdrant_breaker.allow_request():
            CIRCUIT_BREAKER_REJECTIONS.labels(service="qdrant").inc()
            raise CircuitOpenError("qdrant")

        try:
            self._write_qdrant(unit)
            self.qdrant_breaker.record_success()
        except CircuitOpenError:
            raise
        except Exception:
            self.qdrant_breaker.record_failure()
            raise

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

    def __exit__(
        self, exc_type: type | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        self.close()

    def close(self) -> None:
        """Release connections.

        Safe to call multiple times — silently ignores already-closed
        resources.
        """
        try:
            self._neo4j_driver.close()
        except Exception:
            logger.debug("Error closing Neo4j driver", exc_info=True)
        try:
            self._qdrant.close()
        except Exception:
            logger.debug("Error closing Qdrant client", exc_info=True)

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
        """Execute all Neo4j writes within a single transaction.

        Two-phase approach: write all new state first, then clean up stale
        data.  This ensures that if any write fails, stale edges are still
        intact and orphan cleanup won't incorrectly delete entity nodes.

        Batch queries (UNWIND) are used for entity upserts, edge merges, and
        chunk upserts to avoid N+1 query patterns.
        """
        doc = unit.doc
        is_modified = doc.operation == "modified"

        # ── Phase 1: Write new state ──────────────────────────────

        # 1. Upsert source node
        _upsert_source_node(tx, doc)

        # 2. Batch upsert entity nodes and MENTIONS edges
        if unit.entities:
            _batch_upsert_entities(tx, unit.entities)
            _batch_merge_entity_edges(
                tx, doc.source_id, [e["name"] for e in unit.entities], "MENTIONS"
            )

        # 3. Create relationship triples between entities
        for triple in unit.triples:
            _merge_entity_edge(tx, triple)

        # 4. Batch create Chunk nodes linked via HAS_CHUNK
        new_chunk_ids: list[str] = []
        if unit.chunks:
            new_chunk_ids = [
                _chunk_node_id(doc.source_id, chunk.index) for chunk in unit.chunks
            ]
            _batch_upsert_chunks(tx, doc.source_id, unit.chunks, new_chunk_ids)

        # 5. Write graph hints
        #    Stale hint edges are removed before writing because hint MERGE
        #    patterns vary per-hint and we cannot selectively identify stale
        #    ones after the fact.  Hint edges don't point to Entity nodes so
        #    this doesn't affect orphan cleanup.
        if is_modified:
            tx.run(
                "MATCH ({source_id: $uri})-[r {hint: true}]->() DELETE r",
                uri=doc.source_id,
            )
        for hint in doc.graph_hints:
            _write_graph_hint(tx, hint)

        # 6. Batch upsert DEPICTS edges (vision-extracted entities)
        if unit.depicts_entities:
            _batch_upsert_entities(tx, unit.depicts_entities)
            _batch_merge_entity_edges(
                tx, doc.source_id, [e["name"] for e in unit.depicts_entities], "DEPICTS"
            )

        # 7. Create ATTACHED_TO edge (Image→File for embedded images)
        parent_source_id = doc.node_props.get("parent_source_id")
        if doc.node_label == "Image" and parent_source_id:
            _merge_attached_to_edge(tx, doc.source_id, parent_source_id)

        # ── Phase 2: Clean up stale data (modified only) ──────────

        if is_modified:
            # 8. Remove MENTIONS edges not pointing to current entities;
            #    collect removed entity names as orphan candidates.
            current_entity_names = [
                _truncate_entity_name(e["name"]) for e in unit.entities
            ]
            stale_mentions = _clean_stale_edges(
                tx, doc.source_id, "MENTIONS", current_entity_names
            )

            # 9. Remove DEPICTS edges not pointing to current depicts entities;
            #    collect removed entity names as orphan candidates.
            current_depicts_names = [
                _truncate_entity_name(e["name"]) for e in unit.depicts_entities
            ]
            stale_depicts = _clean_stale_edges(
                tx, doc.source_id, "DEPICTS", current_depicts_names
            )

            # 10. Remove Chunk nodes not in the current chunk set
            if new_chunk_ids:
                tx.run(
                    "MATCH (s {source_id: $sid})-[:HAS_CHUNK]->(c:Chunk) "
                    "WHERE NOT c.id IN $keep "
                    "DETACH DELETE c",
                    sid=doc.source_id,
                    keep=new_chunk_ids,
                )
            else:
                tx.run(
                    "MATCH (s {source_id: $sid})-[:HAS_CHUNK]->(c:Chunk) "
                    "DETACH DELETE c",
                    sid=doc.source_id,
                )

            # 11. Clean up orphaned Entity nodes — scoped to the entities
            #     whose edges were removed above (no full graph scan).
            _cleanup_orphan_entities(tx, stale_mentions + stale_depicts)

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
                point_id = str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"{doc.source_id}:{chunk.index}",
                    )
                )
                payload = {
                    "source_type": doc.source_type,
                    "source_id": doc.source_id,
                    "chunk_index": chunk.index,
                    "text": chunk.text,
                    "date": doc.source_metadata.get("date", ""),
                }
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload,
                    )
                )

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
        if not self.neo4j_breaker.allow_request():
            CIRCUIT_BREAKER_REJECTIONS.labels(service="neo4j").inc()
            raise CircuitOpenError("neo4j")
        try:
            self._delete_neo4j(source_id)
            self.neo4j_breaker.record_success()
        except CircuitOpenError:
            raise
        except Exception:
            self.neo4j_breaker.record_failure()
            raise

        if not self.qdrant_breaker.allow_request():
            CIRCUIT_BREAKER_REJECTIONS.labels(service="qdrant").inc()
            raise CircuitOpenError("qdrant")
        try:
            self._delete_qdrant_vectors(source_id)
            self.qdrant_breaker.record_success()
        except CircuitOpenError:
            raise
        except Exception:
            self.qdrant_breaker.record_failure()
            raise

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
        f"SET {set_parts}, s.indexed_at = datetime()"
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
    """MERGE an Entity node on name, setting type and confidence.

    On first creation, type and confidence are set unconditionally.
    On subsequent merges, type and confidence are only updated when the
    incoming confidence is strictly higher than the stored value, preventing
    lower-confidence extractions from overwriting higher-confidence ones.
    """
    name = _truncate_entity_name(entity["name"])
    tx.run(
        """
        MERGE (e:Entity {name: $name})
        ON CREATE SET e.type = $type,
                      e.confidence = $confidence
        ON MATCH SET e.type = CASE WHEN $confidence > e.confidence THEN $type ELSE e.type END,
                     e.confidence = CASE WHEN $confidence > e.confidence THEN $confidence ELSE e.confidence END
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
        MERGE (s)-[r:MENTIONS]->(e)
        ON CREATE SET r.created_at = datetime()
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
        MERGE (s)-[r:DEPICTS]->(e)
        ON CREATE SET r.created_at = datetime()
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


def _clean_stale_edges(
    tx: Any,
    source_id: str,
    edge_type: str,
    keep_names: list[str],
) -> list[str]:
    """Delete edges of a given type that point to entities NOT in *keep_names*.

    Unlike :func:`_clean_source_edges` which blanket-deletes all edges first,
    this function is safe to call *after* new edges have been written — it only
    removes edges that are no longer current.  When *keep_names* is empty every
    edge of the given type is removed (the source no longer mentions anything).

    Returns the names of Entity nodes whose edges were deleted — these are
    candidates for orphan cleanup.
    """
    edge_type = _validate_cypher_identifier(edge_type, "edge_type")
    if keep_names:
        result = tx.run(
            f"MATCH (s {{source_id: $sid}})-[r:{edge_type}]->(e:Entity) "
            f"WHERE NOT e.name IN $keep "
            f"DELETE r "
            f"RETURN e.name AS name",
            sid=source_id,
            keep=keep_names,
        )
    else:
        result = tx.run(
            f"MATCH (s {{source_id: $sid}})-[r:{edge_type}]->(e:Entity) "
            f"DELETE r "
            f"RETURN e.name AS name",
            sid=source_id,
        )
    return [record["name"] for record in result]


def _cleanup_orphan_entities(tx: Any, candidate_names: list[str]) -> int:
    """Delete orphaned Entity nodes from *candidate_names*.

    Only checks the specific entities whose stale edges were removed in this
    write cycle, avoiding a full graph scan on every write.

    Returns the number of orphaned Entity nodes deleted.
    """
    if not candidate_names:
        return 0
    result = tx.run(
        "MATCH (e:Entity) "
        "WHERE e.name IN $names AND NOT ()-[]->(e) "
        "DETACH DELETE e "
        "RETURN count(e) AS removed",
        names=candidate_names,
    )
    removed = result.single()["removed"]
    if removed:
        logger.info("Cleaned up %d orphaned Entity nodes", removed)
    return removed


def _merge_attached_to_edge(
    tx: Any, image_source_id: str, parent_source_id: str
) -> None:
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
    # Defense in depth: assert predicate is safe immediately before interpolation.
    # This guard prevents injection if the validation logic above is ever refactored.
    assert predicate in ALLOWED_PREDICATES, (
        f"predicate {predicate!r} not in ALLOWED_PREDICATES"
    )
    _validate_cypher_identifier(predicate, "triple predicate (pre-interpolation)")
    tx.run(
        f"""
        MERGE (s:Entity {{name: $subject}})
        MERGE (o:Entity {{name: $object}})
        MERGE (s)-[:{predicate}]->(o)
        """,
        subject=_truncate_entity_name(triple["subject"]),
        object=_truncate_entity_name(triple["object"]),
    )


def _upsert_chunk(tx: Any, chunk_id: str, source_id: str, chunk: Chunk) -> None:
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


def _build_hint_set_parts(alias: str, props: dict[str, Any], is_person: bool) -> str:
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
    subject_label = _validate_cypher_identifier(
        hint.subject_label, "hint subject_label"
    )
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
    subj_set_parts = _build_hint_set_parts("s", subj_props, subject_label == "Person")
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
    obj_set_parts = _build_hint_set_parts("o", obj_props, object_label == "Person")
    obj_params = {f"o_{k}": v for k, v in obj_props.items()}
    obj_params["o_merge"] = obj_merge_val
    tx.run(
        f"MERGE (o:{object_label} {{{obj_merge_key}: $o_merge}}) SET {obj_set_parts}",
        **obj_params,
    )

    # --- Relationship edge (tagged hint=true for cleanup on modification) ---
    tx.run(
        f"MATCH (s:{subject_label} {{{subj_merge_key}: $s_merge}}) "
        f"MATCH (o:{object_label} {{{obj_merge_key}: $o_merge}}) "
        f"MERGE (s)-[r:{predicate}]->(o) "
        f"ON CREATE SET r.created_at = datetime() "
        f"SET r.hint = true",
        s_merge=subj_merge_val,
        o_merge=obj_merge_val,
    )


def _chunk_node_id(source_id: str, chunk_index: int) -> str:
    """Deterministic chunk node ID from source_id and index."""
    return f"{source_id}:chunk:{chunk_index}"


def _batch_upsert_entities(tx: Any, entities: list[dict[str, Any]]) -> None:
    """Batch MERGE Entity nodes via UNWIND, replacing N individual _upsert_entity calls.

    Applies the same ON CREATE / ON MATCH confidence-guard logic as
    :func:`_upsert_entity`, but in a single round-trip to Neo4j.
    """
    tx.run(
        """
        UNWIND $entities AS e
        MERGE (ent:Entity {name: e.name})
        ON CREATE SET ent.type = e.type,
                      ent.confidence = e.confidence
        ON MATCH SET ent.type = CASE WHEN e.confidence > ent.confidence
                                     THEN e.type ELSE ent.type END,
                     ent.confidence = CASE WHEN e.confidence > ent.confidence
                                          THEN e.confidence ELSE ent.confidence END
        """,
        entities=[
            {
                "name": _truncate_entity_name(e["name"]),
                "type": e.get("type", "Concept"),
                "confidence": e.get("confidence", 0.75),
            }
            for e in entities
        ],
    )


def _batch_merge_entity_edges(
    tx: Any, source_id: str, entity_names: list[str], edge_type: str
) -> None:
    """Batch MERGE MENTIONS or DEPICTS edges via UNWIND.

    Replaces N individual :func:`_merge_mentions_edge` or
    :func:`_merge_depicts_edge` calls with a single round-trip.
    Entity nodes must already exist before calling this function.
    """
    edge_type = _validate_cypher_identifier(edge_type, "edge_type")
    tx.run(
        f"MATCH (s {{source_id: $sid}}) "
        f"WITH s "
        f"UNWIND $names AS name "
        f"MATCH (e:Entity {{name: name}}) "
        f"MERGE (s)-[:{edge_type}]->(e)",
        sid=source_id,
        names=[_truncate_entity_name(n) for n in entity_names],
    )


def _batch_upsert_chunks(
    tx: Any,
    source_id: str,
    chunks: list["Chunk"],
    chunk_ids: list[str],
) -> None:
    """Batch MERGE Chunk nodes and HAS_CHUNK edges via UNWIND.

    Replaces N individual :func:`_upsert_chunk` calls with a single
    round-trip.  The source node must already exist before calling this.
    """
    tx.run(
        """
        MATCH (s {source_id: $sid})
        WITH s
        UNWIND $chunks AS c
        MERGE (chunk:Chunk {id: c.id})
        SET chunk.text = c.text,
            chunk.source_id = $sid,
            chunk.chunk_index = c.index
        MERGE (s)-[:HAS_CHUNK]->(chunk)
        """,
        sid=source_id,
        chunks=[
            {"id": cid, "text": ch.text, "index": ch.index}
            for cid, ch in zip(chunk_ids, chunks)
        ],
    )
