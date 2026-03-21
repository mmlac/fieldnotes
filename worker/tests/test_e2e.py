"""End-to-end integration tests across all sources and MCP tools.

Exercises the full pipeline from parsing through writing for multiple
source types (file, obsidian, email, repository), then verifies MCP
tool responses against real ingested data.

Requires real Neo4j and Qdrant instances; tests are skipped if either
service is unavailable.  Run via::

    pytest tests/test_e2e.py -v
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Generator

import pytest
from neo4j import GraphDatabase, Driver
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from worker.config import Config, Neo4jConfig, QdrantConfig
from worker.parsers.base import GraphHint, ParsedDocument
from worker.parsers.files import FileParser
from worker.parsers.gmail import GmailParser
from worker.parsers.obsidian import ObsidianParser
from worker.parsers.repositories import RepositoryParser
from worker.pipeline.chunker import Chunk
from worker.pipeline.writer import VECTOR_SIZE, WriteUnit, Writer
from worker.mcp_server import FieldnotesServer

# ------------------------------------------------------------------
# Connection helpers
# ------------------------------------------------------------------

_NEO4J_URI = "bolt://localhost:7687"
_NEO4J_USER = "neo4j"
_NEO4J_PASSWORD = "testpassword"
_QDRANT_HOST = "localhost"
_QDRANT_PORT = 6333
_TEST_COLLECTION = "fieldnotes_e2e_test"


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


_skip_services = pytest.mark.skipif(
    not (_neo4j_available() and _qdrant_available()),
    reason="Neo4j and/or Qdrant not available",
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _uid(prefix: str = "e2e") -> str:
    return f"{prefix}/{uuid.uuid4().hex[:8]}"


def _doc(
    source_id: str, operation: str = "created", **overrides: Any
) -> ParsedDocument:
    defaults = dict(
        source_type="files",
        source_id=source_id,
        operation=operation,
        text="test content",
        node_label="File",
        node_props={"name": source_id.rsplit("/", 1)[-1], "path": source_id},
    )
    defaults.update(overrides)
    return ParsedDocument(**defaults)


def _chunks(texts: list[str]) -> list[Chunk]:
    return [Chunk(text=t, index=i) for i, t in enumerate(texts)]


def _vectors(count: int) -> list[list[float]]:
    return [[float(i) * 0.01 + 0.1] * VECTOR_SIZE for i in range(count)]


def _entities(*names: str, etype: str = "Concept") -> list[dict[str, Any]]:
    return [{"name": n, "type": etype, "confidence": 0.9} for n in names]


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def neo4j_driver() -> Generator[Driver, None, None]:
    driver = GraphDatabase.driver(_NEO4J_URI, auth=(_NEO4J_USER, _NEO4J_PASSWORD))
    yield driver
    driver.close()


@pytest.fixture
def qdrant_client_() -> Generator[QdrantClient, None, None]:
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
    yield
    with neo4j_driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")


@pytest.fixture(autouse=True)
def _cleanup_qdrant(qdrant_client_: QdrantClient) -> Generator[None, None, None]:
    yield
    try:
        qdrant_client_.delete_collection(_TEST_COLLECTION)
    except Exception:
        pass


# ------------------------------------------------------------------
# Query helpers
# ------------------------------------------------------------------


def _count_nodes(driver: Driver, label: str) -> int:
    with driver.session() as session:
        result = session.run(f"MATCH (n:`{label}`) RETURN count(n) AS cnt")
        return result.single()["cnt"]


def _query_edges(driver: Driver, source_id: str, edge_type: str) -> list[str]:
    with driver.session() as session:
        result = session.run(
            f"MATCH (s {{source_id: $sid}})-[:{edge_type}]->(e) RETURN e.name AS name",
            sid=source_id,
        )
        return [r["name"] for r in result]


def _query_chunks(driver: Driver, source_id: str) -> list[str]:
    with driver.session() as session:
        result = session.run(
            "MATCH (s {source_id: $sid})-[:HAS_CHUNK]->(c:Chunk) "
            "RETURN c.text AS text ORDER BY c.chunk_index",
            sid=source_id,
        )
        return [r["text"] for r in result]


def _query_hint_edges(
    driver: Driver, source_id: str, predicate: str
) -> list[dict[str, Any]]:
    with driver.session() as session:
        result = session.run(
            "MATCH (s {source_id: $sid})-[r {hint: true}]->(o) "
            "WHERE type(r) = $pred "
            "RETURN type(r) AS rel_type, o.source_id AS target_id, "
            "properties(o) AS props",
            sid=source_id,
            pred=predicate,
        )
        return [dict(r) for r in result]


def _qdrant_count(client: QdrantClient, source_id: str) -> int:
    result = client.scroll(
        collection_name=_TEST_COLLECTION,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="source_id", match=MatchValue(value=source_id)),
            ]
        ),
        limit=1000,
    )
    return len(result[0])


def _qdrant_search(
    client: QdrantClient, vector: list[float], top_k: int = 10
) -> list[Any]:
    return client.search(
        collection_name=_TEST_COLLECTION,
        query_vector=vector,
        limit=top_k,
        with_payload=True,
    )


def _node_exists(driver: Driver, source_id: str) -> bool:
    with driver.session() as session:
        result = session.run(
            "MATCH (n {source_id: $sid}) RETURN count(n) AS cnt",
            sid=source_id,
        )
        return result.single()["cnt"] > 0


def _entity_exists(driver: Driver, name: str) -> bool:
    with driver.session() as session:
        result = session.run(
            "MATCH (e:Entity {name: $name}) RETURN count(e) AS cnt",
            name=name,
        )
        return result.single()["cnt"] > 0


def _person_node(driver: Driver, email: str) -> dict[str, Any] | None:
    with driver.session() as session:
        result = session.run(
            "MATCH (p:Person {email: $email}) RETURN properties(p) AS props",
            email=email,
        )
        record = result.single()
        return dict(record["props"]) if record else None


# ------------------------------------------------------------------
# Test fixtures corpus — known entities and relationships
# ------------------------------------------------------------------

# A small, self-contained test corpus with deterministic entities:
#
# Person: Alice Chen (alice@example.com) — appears in email AND commit
# Person: Bob Zhang (bob@example.com) — email recipient
# File: notes/ml-research.md — mentions "Machine Learning", "Neural Networks"
# Email: email/msg-001 — from Alice, mentions "Neural Networks"
# Commit: commit:abc123 — authored by Alice
# Obsidian: notes/project-alpha.md — links to notes/ml-research.md


CORPUS_FILE_TEXT = (
    "Machine Learning is transforming how we build software. "
    "Neural Networks are particularly effective for NLP tasks. "
    "Alice Chen presented her research findings at the conference. "
) * 10  # repeat to ensure chunking

CORPUS_EMAIL_BODY = (
    "Hi Bob, I wanted to share my latest findings on Neural Networks. "
    "The transformer architecture shows promising results for our project. "
    "Let me know if you have questions. Best, Alice"
)

CORPUS_COMMIT_MSG = (
    "feat: add transformer model training pipeline\n\n"
    "Implements the data preprocessing and model training loop "
    "for the new transformer-based NLP model."
)

CORPUS_OBSIDIAN_TEXT = (
    "---\ntitle: Project Alpha\ntags: [research, ml]\n---\n"
    "# Project Alpha\n\n"
    "This project builds on our [[ml-research]] findings.\n"
    "The goal is to deploy a production transformer model.\n"
) + "Additional context about the project roadmap. " * 15


# ==================================================================
# 1. Full pipeline E2E: ingest multiple source types
# ==================================================================


@_skip_services
class TestFullPipelineE2E:
    """Ingest documents from multiple source types and verify graph + vector state."""

    def test_file_source_creates_nodes_and_vectors(
        self,
        writer: Writer,
        neo4j_driver: Driver,
        qdrant_client_: QdrantClient,
    ) -> None:
        """File source → Source node, Entity nodes, MENTIONS edges, Chunks, Qdrant vectors."""
        sid = "notes/ml-research.md"
        chunks = _chunks(["Machine Learning for NLP", "Neural Networks overview"])
        unit = WriteUnit(
            doc=_doc(sid, "created", text=CORPUS_FILE_TEXT),
            chunks=chunks,
            vectors=_vectors(2),
            entities=_entities(
                "Machine Learning", "Neural Networks", etype="Technology"
            ),
        )
        writer.write(unit)

        # Source node exists
        assert _node_exists(neo4j_driver, sid)

        # Entity nodes exist
        assert _entity_exists(neo4j_driver, "Machine Learning")
        assert _entity_exists(neo4j_driver, "Neural Networks")

        # MENTIONS edges
        mentions = sorted(_query_edges(neo4j_driver, sid, "MENTIONS"))
        assert "Machine Learning" in mentions
        assert "Neural Networks" in mentions

        # Chunks
        neo4j_chunks = _query_chunks(neo4j_driver, sid)
        assert len(neo4j_chunks) == 2

        # Qdrant vectors
        assert _qdrant_count(qdrant_client_, sid) == 2

    def test_email_source_with_graph_hints(
        self,
        writer: Writer,
        neo4j_driver: Driver,
        qdrant_client_: QdrantClient,
    ) -> None:
        """Email source → Email node, Person nodes via graph hints, SENT/TO edges."""
        parser = GmailParser()
        docs = parser.parse(
            {
                "source_id": "email/msg-001",
                "operation": "created",
                "text": CORPUS_EMAIL_BODY,
                "mime_type": "text/plain",
                "meta": {
                    "message_id": "msg-001",
                    "thread_id": "thread-001",
                    "subject": "Neural Networks Research",
                    "date": "2026-03-10",
                    "sender_email": "Alice Chen <alice@example.com>",
                    "recipients": ["Bob Zhang <bob@example.com>"],
                },
            }
        )
        assert len(docs) == 1
        email_doc = docs[0]

        chunks = _chunks(["Neural Networks research findings"])
        unit = WriteUnit(
            doc=email_doc,
            chunks=chunks,
            vectors=_vectors(1),
            entities=_entities("Neural Networks", etype="Technology"),
        )
        writer.write(unit)

        # Email node exists
        assert _node_exists(neo4j_driver, "email/msg-001")

        # Person nodes created via graph hints
        assert _person_node(neo4j_driver, "alice@example.com") is not None
        assert _person_node(neo4j_driver, "bob@example.com") is not None

        # SENT edge from Person to Email
        with neo4j_driver.session() as session:
            result = session.run(
                "MATCH (p:Person {email: 'alice@example.com'})-[:SENT]->(e:Email) "
                "RETURN e.source_id AS sid"
            )
            sent_emails = [r["sid"] for r in result]
        assert "email/msg-001" in sent_emails

        # TO edge from Email to Person
        with neo4j_driver.session() as session:
            result = session.run(
                "MATCH (e:Email {source_id: 'email/msg-001'})-[:TO]->(p:Person) "
                "RETURN p.email AS email"
            )
            recipients = [r["email"] for r in result]
        assert "bob@example.com" in recipients

        # Thread relationship
        with neo4j_driver.session() as session:
            result = session.run(
                "MATCH (e:Email {source_id: 'email/msg-001'})-[:PART_OF]->(t:Thread) "
                "RETURN t.thread_id AS tid"
            )
            thread_ids = [r["tid"] for r in result]
        assert "thread-001" in thread_ids

    def test_repository_commit_source(
        self,
        writer: Writer,
        neo4j_driver: Driver,
    ) -> None:
        """Repository commit → Commit node, Person AUTHORED, PART_OF Repository."""
        parser = RepositoryParser()
        docs = parser.parse(
            {
                "source_id": "commit:abc123",
                "text": CORPUS_COMMIT_MSG,
                "meta": {
                    "sha": "abc123def456",
                    "author_name": "Alice Chen",
                    "author_email": "alice@example.com",
                    "date": "2026-03-09",
                    "repo_name": "ml-pipeline",
                    "repo_path": "/repos/ml-pipeline",
                    "changed_files": ["src/train.py", "src/preprocess.py"],
                },
            }
        )
        assert len(docs) == 1
        commit_doc = docs[0]

        chunks = _chunks(["transformer model training pipeline"])
        unit = WriteUnit(
            doc=commit_doc,
            chunks=chunks,
            vectors=_vectors(1),
            entities=_entities("Transformer", etype="Technology"),
        )
        writer.write(unit)

        # Commit node exists
        assert _node_exists(neo4j_driver, "commit:abc123")

        # AUTHORED edge from Person to Commit
        with neo4j_driver.session() as session:
            result = session.run(
                "MATCH (p:Person)-[:AUTHORED]->(c:Commit {source_id: 'commit:abc123'}) "
                "RETURN p.email AS email"
            )
            authors = [r["email"] for r in result]
        assert "alice@example.com" in authors

        # PART_OF edge from Commit to Repository
        with neo4j_driver.session() as session:
            result = session.run(
                "MATCH (c:Commit {source_id: 'commit:abc123'})-[:PART_OF]->(r:Repository) "
                "RETURN r.name AS name"
            )
            repos = [r["name"] for r in result]
        assert "ml-pipeline" in repos

    def test_obsidian_source_with_wikilinks(
        self,
        writer: Writer,
        neo4j_driver: Driver,
        qdrant_client_: QdrantClient,
    ) -> None:
        """Obsidian note → File node, wikilink LINKS_TO edges via graph hints."""
        parser = ObsidianParser()
        docs = parser.parse(
            {
                "source_id": "notes/project-alpha.md",
                "operation": "created",
                "text": CORPUS_OBSIDIAN_TEXT,
                "meta": {},
            }
        )

        # Should produce at least 1 text doc
        text_docs = [d for d in docs if d.text and d.node_label == "File"]
        assert len(text_docs) >= 1
        obsidian_doc = text_docs[0]

        # Verify wikilink graph hint was extracted
        link_hints = [h for h in obsidian_doc.graph_hints if h.predicate == "LINKS_TO"]
        assert len(link_hints) >= 1
        assert any("ml-research" in h.object_id for h in link_hints)

        chunks = _chunks(["Project Alpha overview"])
        unit = WriteUnit(
            doc=obsidian_doc,
            chunks=chunks,
            vectors=_vectors(1),
        )
        writer.write(unit)

        # Source node exists
        assert _node_exists(neo4j_driver, "notes/project-alpha.md")

        # LINKS_TO edge created via graph hint
        hint_edges = _query_hint_edges(
            neo4j_driver, "notes/project-alpha.md", "LINKS_TO"
        )
        assert len(hint_edges) >= 1

    def test_multi_source_graph_structure(
        self,
        writer: Writer,
        neo4j_driver: Driver,
        qdrant_client_: QdrantClient,
    ) -> None:
        """Ingest file + email + commit from corpus and verify overall graph structure."""
        # 1. File
        file_sid = "notes/ml-research.md"
        writer.write(
            WriteUnit(
                doc=_doc(file_sid, text=CORPUS_FILE_TEXT),
                chunks=_chunks(["ML chunk 1", "ML chunk 2"]),
                vectors=_vectors(2),
                entities=_entities(
                    "Machine Learning", "Neural Networks", etype="Technology"
                ),
            )
        )

        # 2. Email (uses GmailParser for realistic graph hints)
        email_doc = GmailParser().parse(
            {
                "source_id": "email/msg-001",
                "operation": "created",
                "text": CORPUS_EMAIL_BODY,
                "mime_type": "text/plain",
                "meta": {
                    "message_id": "msg-001",
                    "thread_id": "thread-001",
                    "subject": "Neural Networks Research",
                    "date": "2026-03-10",
                    "sender_email": "alice@example.com",
                    "recipients": ["bob@example.com"],
                },
            }
        )[0]
        writer.write(
            WriteUnit(
                doc=email_doc,
                chunks=_chunks(["Email chunk about NNs"]),
                vectors=_vectors(1),
                entities=_entities("Neural Networks", etype="Technology"),
            )
        )

        # 3. Commit
        commit_doc = RepositoryParser().parse(
            {
                "source_id": "commit:abc123",
                "text": CORPUS_COMMIT_MSG,
                "meta": {
                    "sha": "abc123",
                    "author_name": "Alice Chen",
                    "author_email": "alice@example.com",
                    "date": "2026-03-09",
                    "repo_name": "ml-pipeline",
                    "repo_path": "/repos/ml-pipeline",
                    "changed_files": ["src/train.py"],
                },
            }
        )[0]
        writer.write(
            WriteUnit(
                doc=commit_doc,
                chunks=_chunks(["Commit chunk"]),
                vectors=_vectors(1),
                entities=_entities("Transformer", etype="Technology"),
            )
        )

        # Verify graph structure
        assert _count_nodes(neo4j_driver, "File") >= 1
        assert _count_nodes(neo4j_driver, "Email") >= 1
        assert _count_nodes(neo4j_driver, "Commit") >= 1
        assert _count_nodes(neo4j_driver, "Person") >= 1
        assert _count_nodes(neo4j_driver, "Entity") >= 1
        assert _count_nodes(neo4j_driver, "Chunk") >= 4  # 2 + 1 + 1

        # Qdrant has vectors for all sources
        total_vectors = (
            _qdrant_count(qdrant_client_, file_sid)
            + _qdrant_count(qdrant_client_, "email/msg-001")
            + _qdrant_count(qdrant_client_, "commit:abc123")
        )
        assert total_vectors == 4


# ==================================================================
# 2. Cross-source query tests
# ==================================================================


@_skip_services
class TestCrossSourceQueries:
    """Verify that entity resolution links data across source types."""

    def test_same_person_across_email_and_commit(
        self,
        writer: Writer,
        neo4j_driver: Driver,
    ) -> None:
        """Alice appears in email (sender) AND commit (author). Person node linked."""
        # Email from Alice
        email_doc = GmailParser().parse(
            {
                "source_id": "email/msg-alice",
                "operation": "created",
                "text": "Meeting notes from Alice",
                "mime_type": "text/plain",
                "meta": {
                    "message_id": "msg-alice",
                    "thread_id": "thread-alice",
                    "subject": "Meeting Notes",
                    "date": "2026-03-10",
                    "sender_email": "alice@example.com",
                    "recipients": [],
                },
            }
        )[0]
        writer.write(
            WriteUnit(
                doc=email_doc,
                chunks=_chunks(["Alice meeting notes"]),
                vectors=_vectors(1),
            )
        )

        # Commit by Alice
        commit_doc = RepositoryParser().parse(
            {
                "source_id": "commit:alice-commit",
                "text": "fix: update config",
                "meta": {
                    "sha": "aaa111",
                    "author_name": "Alice Chen",
                    "author_email": "alice@example.com",
                    "date": "2026-03-09",
                    "repo_name": "myrepo",
                    "repo_path": "/repos/myrepo",
                    "changed_files": [],
                },
            }
        )[0]
        writer.write(
            WriteUnit(
                doc=commit_doc,
                chunks=_chunks(["Config fix"]),
                vectors=_vectors(1),
            )
        )

        # Reconcile persons
        writer.reconcile_persons()

        # Single Person node with alice@example.com
        alice = _person_node(neo4j_driver, "alice@example.com")
        assert alice is not None

        # Person has both SENT and AUTHORED edges
        with neo4j_driver.session() as session:
            sent = session.run(
                "MATCH (p:Person {email: 'alice@example.com'})-[:SENT]->(e) RETURN count(e) AS cnt"
            ).single()["cnt"]
            authored = session.run(
                "MATCH (p:Person {email: 'alice@example.com'})-[:AUTHORED]->(c) RETURN count(c) AS cnt"
            ).single()["cnt"]
        assert sent >= 1
        assert authored >= 1

    def test_shared_entity_across_sources(
        self,
        writer: Writer,
        neo4j_driver: Driver,
    ) -> None:
        """Same entity 'Neural Networks' mentioned by file AND email."""
        # File mentioning Neural Networks
        writer.write(
            WriteUnit(
                doc=_doc("notes/nn-file.md", text="Neural Networks intro"),
                chunks=_chunks(["NN chunk"]),
                vectors=_vectors(1),
                entities=_entities("Neural Networks", etype="Technology"),
            )
        )

        # Email mentioning Neural Networks
        email_doc = GmailParser().parse(
            {
                "source_id": "email/nn-email",
                "operation": "created",
                "text": "Discussion about Neural Networks",
                "mime_type": "text/plain",
                "meta": {
                    "message_id": "nn-email",
                    "thread_id": "t-nn",
                    "subject": "NN Discussion",
                    "date": "2026-03-10",
                    "sender_email": "researcher@example.com",
                    "recipients": [],
                },
            }
        )[0]
        writer.write(
            WriteUnit(
                doc=email_doc,
                chunks=_chunks(["NN email chunk"]),
                vectors=_vectors(1),
                entities=_entities("Neural Networks", etype="Technology"),
            )
        )

        # Entity exists once (MERGE semantics)
        assert _entity_exists(neo4j_driver, "Neural Networks")

        # Both sources have MENTIONS edges to the same entity
        file_mentions = _query_edges(neo4j_driver, "notes/nn-file.md", "MENTIONS")
        email_mentions = _query_edges(neo4j_driver, "email/nn-email", "MENTIONS")
        assert "Neural Networks" in file_mentions
        assert "Neural Networks" in email_mentions

    def test_cross_source_vector_search(
        self,
        writer: Writer,
        qdrant_client_: QdrantClient,
    ) -> None:
        """Vectors from different source types are all searchable in the same collection."""
        # Write vectors for 3 different source types
        file_vec = [0.9] * VECTOR_SIZE
        email_vec = [0.8] * VECTOR_SIZE
        commit_vec = [0.7] * VECTOR_SIZE

        writer.write(
            WriteUnit(
                doc=_doc("file/test.md", source_type="files", text="File content"),
                chunks=_chunks(["file chunk"]),
                vectors=[file_vec],
            )
        )

        email_doc = ParsedDocument(
            source_type="gmail",
            source_id="email/test-vec",
            operation="created",
            text="Email content",
            node_label="Email",
            node_props={"message_id": "test-vec"},
        )
        writer.write(
            WriteUnit(
                doc=email_doc,
                chunks=_chunks(["email chunk"]),
                vectors=[email_vec],
            )
        )

        commit_doc = ParsedDocument(
            source_type="repositories",
            source_id="commit:testvec",
            operation="created",
            text="Commit content",
            node_label="Commit",
            node_props={"sha": "testvec"},
        )
        writer.write(
            WriteUnit(
                doc=commit_doc,
                chunks=_chunks(["commit chunk"]),
                vectors=[commit_vec],
            )
        )

        # Search with a vector close to the file vector — file should rank first
        search_vec = [0.89] * VECTOR_SIZE
        results = _qdrant_search(qdrant_client_, search_vec, top_k=3)
        assert len(results) == 3

        # All three source types represented
        source_types = {r.payload.get("source_type") for r in results}
        assert "files" in source_types
        assert "gmail" in source_types
        assert "repositories" in source_types


# ==================================================================
# 3. Content update E2E tests
# ==================================================================


@_skip_services
class TestContentUpdateE2E:
    """Full modify cycle: ingest v1 → modify → ingest v2 → verify cleanup."""

    def test_modify_cleans_stale_data_and_writes_new(
        self,
        writer: Writer,
        neo4j_driver: Driver,
        qdrant_client_: QdrantClient,
    ) -> None:
        """v1 has entities A,B and 3 chunks. v2 has entities A,C and 2 chunks.
        B is gone, C is new, chunk count updated."""
        sid = _uid("update-test")

        # v1
        writer.write(
            WriteUnit(
                doc=_doc(sid, "created", text="v1 content"),
                chunks=_chunks(["v1 chunk 0", "v1 chunk 1", "v1 chunk 2"]),
                vectors=_vectors(3),
                entities=_entities("EntityA", "EntityB"),
            )
        )
        assert sorted(_query_edges(neo4j_driver, sid, "MENTIONS")) == [
            "EntityA",
            "EntityB",
        ]
        assert len(_query_chunks(neo4j_driver, sid)) == 3
        assert _qdrant_count(qdrant_client_, sid) == 3

        # v2 — modify
        writer.write(
            WriteUnit(
                doc=_doc(sid, "modified", text="v2 content"),
                chunks=_chunks(["v2 chunk 0", "v2 chunk 1"]),
                vectors=_vectors(2),
                entities=_entities("EntityA", "EntityC"),
            )
        )

        # Stale data cleaned
        mentions = sorted(_query_edges(neo4j_driver, sid, "MENTIONS"))
        assert mentions == ["EntityA", "EntityC"]
        assert len(_query_chunks(neo4j_driver, sid)) == 2
        assert _qdrant_count(qdrant_client_, sid) == 2

        # Orphaned EntityB deleted (only mentioned by this source)
        assert not _entity_exists(neo4j_driver, "EntityB")

    def test_delete_removes_all_data(
        self,
        writer: Writer,
        neo4j_driver: Driver,
        qdrant_client_: QdrantClient,
    ) -> None:
        """Delete operation removes source node, chunks, and vectors."""
        sid = _uid("delete-test")

        writer.write(
            WriteUnit(
                doc=_doc(sid, "created", text="to be deleted"),
                chunks=_chunks(["chunk 0", "chunk 1"]),
                vectors=_vectors(2),
                entities=_entities("DeleteEntity"),
            )
        )
        assert _node_exists(neo4j_driver, sid)
        assert _qdrant_count(qdrant_client_, sid) == 2

        # Delete
        writer.write(WriteUnit(doc=_doc(sid, "deleted")))

        assert not _node_exists(neo4j_driver, sid)
        assert len(_query_chunks(neo4j_driver, sid)) == 0
        assert _qdrant_count(qdrant_client_, sid) == 0

    def test_modify_with_graph_hints_cleans_stale_hints(
        self,
        writer: Writer,
        neo4j_driver: Driver,
    ) -> None:
        """v1 has LINKS_TO hints to X,Y. v2 has LINKS_TO hint to X only. Y gone."""
        sid = _uid("hint-update")
        target_x = _uid("target_x")
        target_y = _uid("target_y")

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

        # v1
        writer.write(
            WriteUnit(
                doc=_doc(sid, "created", graph_hints=[hint_x, hint_y]),
                chunks=_chunks(["hint chunk"]),
                vectors=_vectors(1),
            )
        )
        hints_v1 = _query_hint_edges(neo4j_driver, sid, "LINKS_TO")
        assert len(hints_v1) == 2

        # v2 — only hint_x
        writer.write(
            WriteUnit(
                doc=_doc(sid, "modified", graph_hints=[hint_x]),
                chunks=_chunks(["updated hint chunk"]),
                vectors=_vectors(1),
            )
        )
        hints_v2 = _query_hint_edges(neo4j_driver, sid, "LINKS_TO")
        assert len(hints_v2) == 1


# ==================================================================
# 4. MCP tool E2E tests
# ==================================================================


@_skip_services
class TestMCPToolsE2E:
    """Test MCP tools against real ingested data."""

    def _ingest_corpus(self, writer: Writer, neo4j_driver: Driver) -> None:
        """Ingest the test corpus into Neo4j + Qdrant."""
        # File
        writer.write(
            WriteUnit(
                doc=_doc("notes/ml-research.md", text=CORPUS_FILE_TEXT),
                chunks=_chunks(["ML intro chunk", "NN overview chunk"]),
                vectors=_vectors(2),
                entities=_entities(
                    "Machine Learning", "Neural Networks", etype="Technology"
                ),
            )
        )

        # Email
        email_doc = GmailParser().parse(
            {
                "source_id": "email/msg-001",
                "operation": "created",
                "text": CORPUS_EMAIL_BODY,
                "mime_type": "text/plain",
                "meta": {
                    "message_id": "msg-001",
                    "thread_id": "thread-001",
                    "subject": "Neural Networks Research",
                    "date": "2026-03-10",
                    "sender_email": "alice@example.com",
                    "recipients": ["bob@example.com"],
                },
            }
        )[0]
        writer.write(
            WriteUnit(
                doc=email_doc,
                chunks=_chunks(["NN research email chunk"]),
                vectors=_vectors(1),
                entities=_entities("Neural Networks", etype="Technology"),
            )
        )

        # Commit
        commit_doc = RepositoryParser().parse(
            {
                "source_id": "commit:abc123",
                "text": CORPUS_COMMIT_MSG,
                "meta": {
                    "sha": "abc123",
                    "author_name": "Alice Chen",
                    "author_email": "alice@example.com",
                    "date": "2026-03-09",
                    "repo_name": "ml-pipeline",
                    "repo_path": "/repos/ml-pipeline",
                    "changed_files": ["src/train.py"],
                },
            }
        )[0]
        writer.write(
            WriteUnit(
                doc=commit_doc,
                chunks=_chunks(["transformer training chunk"]),
                vectors=_vectors(1),
                entities=_entities("Transformer", etype="Technology"),
            )
        )

        # Topic nodes (manually inserted for topic tool tests)
        with neo4j_driver.session() as session:
            session.run(
                "CREATE (t:Topic {name: 'Machine Learning', source: 'cluster', "
                "description: 'Research on ML algorithms'})"
            )
            session.run(
                "CREATE (t:Topic {name: 'Project Management', source: 'user', "
                "description: 'User-defined project management topic'})"
            )
            # Tag file with ML topic
            session.run(
                "MATCH (f:File {source_id: 'notes/ml-research.md'}), "
                "(t:Topic {name: 'Machine Learning'}) "
                "CREATE (f)-[:TAGGED]->(t)"
            )

    @pytest.mark.asyncio
    async def test_ingest_status_tool(
        self,
        writer: Writer,
        neo4j_driver: Driver,
        qdrant_client_: QdrantClient,
    ) -> None:
        """ingest_status returns accurate counts matching ingested data."""
        self._ingest_corpus(writer, neo4j_driver)

        cfg = Config(
            neo4j=Neo4jConfig(
                uri=_NEO4J_URI, user=_NEO4J_USER, password=_NEO4J_PASSWORD
            ),
            qdrant=QdrantConfig(
                host=_QDRANT_HOST,
                port=_QDRANT_PORT,
                collection=_TEST_COLLECTION,
            ),
        )
        server = FieldnotesServer(cfg)
        server._connect()
        try:
            result = await server._call_tool("ingest_status", {})
            data = json.loads(result[0].text)

            assert data["health"]["neo4j"] == "ok"
            assert data["health"]["qdrant"] == "ok"

            # Source counts match what we ingested
            assert data["sources"]["file"]["count"] >= 1
            assert data["sources"]["email"]["count"] >= 1
            assert data["sources"]["commit"]["count"] >= 1
            assert data["sources"]["entity"]["count"] >= 1
            assert data["sources"]["chunk"]["count"] >= 4  # 2 + 1 + 1
            assert data["sources"]["topic"]["count"] >= 2

            # Entity types
            assert data["entities"]["total"] >= 1

            # Vector count matches
            assert data["vectors"]["count"] >= 4

            # Topic breakdown
            assert data["topics"].get("cluster", 0) >= 1
            assert data["topics"].get("user", 0) >= 1
        finally:
            server._disconnect()

    def test_list_topics_tool(
        self,
        writer: Writer,
        neo4j_driver: Driver,
    ) -> None:
        """list_topics returns topics after ingestion."""
        self._ingest_corpus(writer, neo4j_driver)

        cfg = Config(
            neo4j=Neo4jConfig(
                uri=_NEO4J_URI, user=_NEO4J_USER, password=_NEO4J_PASSWORD
            ),
        )
        from worker.query.topics import TopicQuerier

        with TopicQuerier(cfg.neo4j) as querier:
            topics = querier.list_topics()
            assert len(topics) >= 2

            names = {t.name for t in topics}
            assert "Machine Learning" in names
            assert "Project Management" in names

            # Check doc counts
            ml_topic = next(t for t in topics if t.name == "Machine Learning")
            assert ml_topic.source == "cluster"
            assert ml_topic.doc_count >= 1

    def test_show_topic_tool(
        self,
        writer: Writer,
        neo4j_driver: Driver,
    ) -> None:
        """show_topic returns topic details with linked documents."""
        self._ingest_corpus(writer, neo4j_driver)

        from worker.query.topics import TopicQuerier

        cfg = Config(
            neo4j=Neo4jConfig(
                uri=_NEO4J_URI, user=_NEO4J_USER, password=_NEO4J_PASSWORD
            ),
        )
        with TopicQuerier(cfg.neo4j) as querier:
            detail = querier.show_topic("Machine Learning")
            assert detail is not None
            assert detail.name == "Machine Learning"
            assert detail.source == "cluster"
            assert len(detail.documents) >= 1
            assert any(
                d["source_id"] == "notes/ml-research.md" for d in detail.documents
            )

    def test_show_topic_not_found(
        self,
        writer: Writer,
        neo4j_driver: Driver,
    ) -> None:
        """show_topic returns None for nonexistent topic."""
        from worker.query.topics import TopicQuerier

        cfg = Config(
            neo4j=Neo4jConfig(
                uri=_NEO4J_URI, user=_NEO4J_USER, password=_NEO4J_PASSWORD
            ),
        )
        with TopicQuerier(cfg.neo4j) as querier:
            detail = querier.show_topic("Nonexistent Topic XYZ")
            assert detail is None

    def test_topic_gaps_tool(
        self,
        writer: Writer,
        neo4j_driver: Driver,
    ) -> None:
        """topic_gaps returns cluster topics without user counterparts."""
        self._ingest_corpus(writer, neo4j_driver)

        from worker.query.topics import TopicQuerier

        cfg = Config(
            neo4j=Neo4jConfig(
                uri=_NEO4J_URI, user=_NEO4J_USER, password=_NEO4J_PASSWORD
            ),
        )
        with TopicQuerier(cfg.neo4j) as querier:
            gaps = querier.topic_gaps()
            # "Machine Learning" is cluster-only, no user counterpart
            assert len(gaps) >= 1
            gap_names = {g.name for g in gaps}
            assert "Machine Learning" in gap_names
            # "Project Management" is user-defined, not a gap
            assert "Project Management" not in gap_names


# ==================================================================
# 5. Parser integration tests (verify parsers produce valid docs)
# ==================================================================


@_skip_services
class TestParserToWriterIntegration:
    """Verify that each parser's output is accepted by the Writer without error."""

    def test_file_parser_output_writable(
        self,
        writer: Writer,
        neo4j_driver: Driver,
    ) -> None:
        parser = FileParser()
        docs = parser.parse(
            {
                "mime_type": "text/markdown",
                "source_id": "notes/parser-test.md",
                "operation": "created",
                "text": "Simple test content for parsing. " * 5,
            }
        )
        assert len(docs) >= 1
        for doc in docs:
            if doc.text:
                writer.write(
                    WriteUnit(
                        doc=doc,
                        chunks=_chunks(["parser test chunk"]),
                        vectors=_vectors(1),
                    )
                )
        assert _node_exists(neo4j_driver, "notes/parser-test.md")

    def test_gmail_parser_output_writable(
        self,
        writer: Writer,
        neo4j_driver: Driver,
    ) -> None:
        parser = GmailParser()
        docs = parser.parse(
            {
                "source_id": "email/parser-test",
                "operation": "created",
                "text": "<html><body><p>HTML email body</p></body></html>",
                "mime_type": "text/html",
                "meta": {
                    "message_id": "parser-test",
                    "thread_id": "t-parser",
                    "subject": "Test Subject",
                    "date": "2026-03-11",
                    "sender_email": "test@example.com",
                    "recipients": ["other@example.com"],
                },
            }
        )
        assert len(docs) == 1
        doc = docs[0]
        # HTML should be stripped
        assert "<html>" not in doc.text
        assert "HTML email body" in doc.text

        writer.write(
            WriteUnit(
                doc=doc,
                chunks=_chunks(["html email chunk"]),
                vectors=_vectors(1),
            )
        )
        assert _node_exists(neo4j_driver, "email/parser-test")

    def test_repository_file_parser_output_writable(
        self,
        writer: Writer,
        neo4j_driver: Driver,
    ) -> None:
        parser = RepositoryParser()
        docs = parser.parse(
            {
                "source_id": "repo:/repos/test:src/main.py",
                "operation": "created",
                "text": "def main():\n    print('hello')\n",
                "mime_type": "text/x-python",
                "meta": {
                    "repo_name": "test-repo",
                    "repo_path": "/repos/test",
                    "relative_path": "src/main.py",
                },
            }
        )
        assert len(docs) == 1
        writer.write(
            WriteUnit(
                doc=docs[0],
                chunks=_chunks(["python code chunk"]),
                vectors=_vectors(1),
            )
        )
        assert _node_exists(neo4j_driver, "repo:/repos/test:src/main.py")

    def test_repository_commit_parser_output_writable(
        self,
        writer: Writer,
        neo4j_driver: Driver,
    ) -> None:
        parser = RepositoryParser()
        docs = parser.parse(
            {
                "source_id": "commit:parser-test-sha",
                "text": "chore: update dependencies",
                "meta": {
                    "sha": "parser-test-sha",
                    "author_name": "Dev",
                    "author_email": "dev@example.com",
                    "date": "2026-03-11",
                    "repo_name": "test-repo",
                    "repo_path": "/repos/test",
                    "changed_files": ["requirements.txt"],
                },
            }
        )
        assert len(docs) == 1
        writer.write(
            WriteUnit(
                doc=docs[0],
                chunks=_chunks(["commit chunk"]),
                vectors=_vectors(1),
            )
        )
        assert _node_exists(neo4j_driver, "commit:parser-test-sha")

    def test_obsidian_parser_output_writable(
        self,
        writer: Writer,
        neo4j_driver: Driver,
    ) -> None:
        parser = ObsidianParser()
        docs = parser.parse(
            {
                "source_id": "vault/parser-test.md",
                "operation": "created",
                "text": "---\ntitle: Test\n---\nContent with [[link]] and more text. "
                * 10,
                "meta": {},
            }
        )
        text_docs = [d for d in docs if d.text and d.node_label == "File"]
        assert len(text_docs) >= 1
        for doc in text_docs:
            writer.write(
                WriteUnit(
                    doc=doc,
                    chunks=_chunks(["obsidian chunk"]),
                    vectors=_vectors(1),
                )
            )
        assert _node_exists(neo4j_driver, "vault/parser-test.md")
