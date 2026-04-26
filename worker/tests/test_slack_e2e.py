"""End-to-end Slack ingestion test.

Drives the full source -> parser -> chunker -> writer pipeline against a
fake Slack ``WebClient`` backed by canned JSON fixtures, with real Neo4j
and Qdrant. The LLM extractor and embedder are intentionally bypassed —
the fixture corpus only needs deterministic graph state and chunk
vectors, not entity extraction quality.

Skipped when Neo4j and/or Qdrant are unavailable (matches the pattern
used by ``tests/test_e2e.py``).
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from typing import Any, Generator
from unittest.mock import MagicMock

import pytest
from neo4j import Driver, GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from worker.config import Neo4jConfig, QdrantConfig
from worker.parsers.slack import SlackParser
from worker.pipeline.chunker import chunk_text
from worker.pipeline.writer import VECTOR_SIZE, WriteUnit, Writer
from worker.sources.slack import SlackSource

# Connection details mirror tests/test_e2e.py so a single docker-compose
# stack serves both suites.
_NEO4J_URI = "bolt://localhost:7687"
_NEO4J_USER = "neo4j"
_NEO4J_PASSWORD = "testpassword"
_QDRANT_HOST = "localhost"
_QDRANT_PORT = 6333
_TEST_COLLECTION = "fieldnotes_slack_e2e_test"

_TEAM_ID = "T-TEST"


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


# ---------------------------------------------------------------------------
# Test queue (module-local; matches the API used by SlackSource)
# ---------------------------------------------------------------------------


class _ListQueue:
    def __init__(self) -> None:
        self.enqueued: list[dict[str, Any]] = []
        self.cursors: dict[str, str] = {}

    def enqueue(
        self,
        event: dict[str, Any],
        cursor_key: str | None = None,
        cursor_value: str | None = None,
    ) -> str:
        self.enqueued.append(event)
        if cursor_key is not None and cursor_value is not None:
            self.cursors[cursor_key] = cursor_value
        return event.get("id", str(uuid.uuid4()))

    def is_enqueued(self, source_id: str) -> bool:
        return any(e.get("source_id") == source_id for e in self.enqueued)

    def load_cursor(self, key: str) -> str | None:
        return self.cursors.get(key)

    def save_cursor(self, key: str, value: str) -> None:
        self.cursors[key] = value


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


# Note: the ``fake_slack_client`` fixture is provided by conftest.py.


# ---------------------------------------------------------------------------
# Helpers — drive the source and write each parsed event
# ---------------------------------------------------------------------------


def _stable_vector(text: str) -> list[float]:
    """Deterministic, low-magnitude vector keyed off the chunk text.

    The pipeline's actual embedder is replaced — we only need vectors
    that round-trip through Qdrant and admit a search assertion."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    # Spread 32 bytes across VECTOR_SIZE slots, normalised to [0, 1].
    floats = [(digest[i % len(digest)] / 255.0) for i in range(VECTOR_SIZE)]
    return floats


def _ingest_once(source: SlackSource, queue: _ListQueue) -> list[dict[str, Any]]:
    """Run a single backfill cycle: discover channels and poll each one.

    Returns the events enqueued during this cycle (not the cumulative
    queue contents)."""
    before = len(queue.enqueued)
    channels = source._discover_conversations()
    team_cursor: dict[str, dict[str, Any]] = {}

    async def _poll_all() -> None:
        for ch in channels:
            await source._poll_conversation(ch, team_cursor, queue)

    asyncio.run(_poll_all())
    return list(queue.enqueued[before:])


def _write_events(writer: Writer, events: list[dict[str, Any]]) -> int:
    """Parse → chunk → write each Slack event. Returns docs written."""
    parser = SlackParser()
    written = 0
    for ev in events:
        for doc in parser.parse(ev):
            chunks = chunk_text(
                doc.text,
                chunk_strategy={
                    "mode": "message_overlap",
                    "overlap_messages": 3,
                },
            )
            if not chunks:
                # Empty / system-only docs still produce a metadata-only
                # text via the parser; skip if chunker yielded nothing.
                continue
            vectors = [_stable_vector(c.text) for c in chunks]
            writer.write(
                WriteUnit(
                    doc=doc,
                    chunks=chunks,
                    vectors=vectors,
                    entities=[],
                )
            )
            written += 1
    return written


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def _count_label(driver: Driver, label: str) -> int:
    with driver.session() as session:
        result = session.run(f"MATCH (n:`{label}`) RETURN count(n) AS cnt")
        return result.single()["cnt"]


def _doc_count_by_kind(driver: Driver, kind: str) -> int:
    """Slack documents are stored under the ``SlackMessage`` label with
    ``has_thread`` distinguishing thread vs window, and
    ``conversation_type='im'`` flagging DMs."""
    with driver.session() as session:
        if kind == "thread":
            result = session.run(
                "MATCH (c:SlackMessage) "
                "WHERE c.has_thread = true RETURN count(c) AS cnt"
            )
        elif kind == "im":
            result = session.run(
                "MATCH (c:SlackMessage) "
                "WHERE c.conversation_type = 'im' RETURN count(c) AS cnt"
            )
        elif kind == "window":
            result = session.run(
                "MATCH (c:SlackMessage) "
                "WHERE c.has_thread = false "
                "AND c.conversation_type <> 'im' "
                "RETURN count(c) AS cnt"
            )
        else:
            raise ValueError(kind)
        return result.single()["cnt"]


def _qdrant_count_for_source(client: QdrantClient, source_id: str) -> int:
    result = client.scroll(
        collection_name=_TEST_COLLECTION,
        scroll_filter=Filter(
            must=[FieldCondition(key="source_id", match=MatchValue(value=source_id))]
        ),
        limit=1000,
    )
    return len(result[0])


def _all_qdrant_points(client: QdrantClient) -> list[Any]:
    result = client.scroll(
        collection_name=_TEST_COLLECTION,
        with_payload=True,
        with_vectors=True,
        limit=10_000,
    )
    return list(result[0])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@_skip_services
class TestSlackEndToEnd:
    def _build_source(self, fake_client: MagicMock) -> SlackSource:
        source = SlackSource()
        source.configure({"poll_interval_seconds": 0})
        source._client = fake_client
        source._team_id = _TEAM_ID
        return source

    def test_full_pipeline_creates_expected_graph_state(
        self,
        writer: Writer,
        neo4j_driver: Driver,
        qdrant_client_: QdrantClient,
        fake_slack_client: MagicMock,
    ) -> None:
        """One backfill cycle: thread + N windows + DM are written; every
        Document has IN_CHANNEL + ≥1 SENT_BY edge; Person nodes for
        alice/bob exist exactly once each; chunks never split a Slack
        message mid-text."""
        source = self._build_source(fake_slack_client)
        queue = _ListQueue()

        events = _ingest_once(source, queue)
        assert events, "fake source produced no events"

        kinds = sorted(ev["meta"]["kind"] for ev in events)
        # 1 thread + ≥2 windows in #engineering (45-minute quiet gap) + 1 DM window.
        assert kinds.count("thread") == 1
        assert kinds.count("window") >= 3

        written = _write_events(writer, events)
        assert written == len(events)

        # --- Document counts -----------------------------------------------
        thread_docs = _doc_count_by_kind(neo4j_driver, "thread")
        window_docs = _doc_count_by_kind(neo4j_driver, "window")
        im_docs = _doc_count_by_kind(neo4j_driver, "im")
        assert thread_docs == 1
        assert window_docs >= 2  # the 45-min gap forces at least 2 windows
        assert im_docs >= 1

        # --- Edge invariants: every Document has IN_CHANNEL + ≥1 SENT_BY ---
        with neo4j_driver.session() as session:
            result = session.run(
                "MATCH (c:SlackMessage) "
                "RETURN c.source_id AS sid, "
                "       size([(c)-[:IN_CHANNEL]->(ch:Channel) | ch]) AS ch_cnt, "
                "       size([(c)-[:SENT_BY]->(p:Person) | p]) AS author_cnt"
            )
            rows = [dict(r) for r in result]
        assert rows, "no Conversation nodes written"
        for row in rows:
            assert row["ch_cnt"] == 1, f"missing IN_CHANNEL on {row['sid']!r}"
            assert row["author_cnt"] >= 1, f"missing SENT_BY on {row['sid']!r}"

        # --- Person dedup: alice and bob each exist exactly once ----------
        # alice@gmail.com appears with both gmail-and-googlemail variants
        # in other tests; here we just assert the count is 2 (alice + bob).
        person_count = _count_label(neo4j_driver, "Person")
        assert person_count == 2, f"expected exactly 2 Person nodes, got {person_count}"

        with neo4j_driver.session() as session:
            emails = sorted(
                r["email"]
                for r in session.run(
                    "MATCH (p:Person) WHERE p.email IS NOT NULL RETURN p.email AS email"
                )
            )
        assert emails == ["alice@gmail.com", "bob@example.com"]

        # --- Qdrant: every Document has ≥1 chunk, no message split mid-text
        all_points = _all_qdrant_points(qdrant_client_)
        assert all_points, "no Qdrant points written"
        for sid in (row["sid"] for row in rows):
            assert _qdrant_count_for_source(qdrant_client_, sid) >= 1, (
                f"{sid!r}: no Qdrant points"
            )

        # No chunk may begin in the middle of a Slack message body — every
        # non-empty chunk for a Slack doc must start with the message
        # header pattern (or be empty whitespace).
        import re

        header_re = re.compile(r"^(?:  )?\[\d{2}:\d{2} UTC\] ")
        slack_chunks = [
            p for p in all_points if p.payload.get("source_type") == "slack"
        ]
        assert slack_chunks
        for p in slack_chunks:
            text = (p.payload.get("text") or "").lstrip("\n")
            if not text.strip():
                continue
            first_line = text.split("\n", 1)[0]
            assert header_re.match(first_line), (
                f"chunk does not start on a message header: {first_line!r}"
            )

    def test_search_surfaces_a_slack_chunk(
        self,
        writer: Writer,
        qdrant_client_: QdrantClient,
        fake_slack_client: MagicMock,
    ) -> None:
        """A vector search keyed off a fixture phrase returns at least
        one Slack chunk in the top-k."""
        source = self._build_source(fake_slack_client)
        queue = _ListQueue()
        events = _ingest_once(source, queue)
        _write_events(writer, events)

        # Pick a fixture phrase known to appear in #engineering chunks
        # and use a stable_vector for it as the query — any chunk whose
        # text hashes to a similar pattern will surface. Because we use
        # the same hash as the indexer, exact text matches the corpus's
        # chunks will rank highly.
        all_points = _all_qdrant_points(qdrant_client_)
        slack_payloads = [
            p.payload for p in all_points if p.payload.get("source_type") == "slack"
        ]
        # Pick any Slack chunk's text as the query target — the search
        # must surface at least one Slack chunk in top-5.
        query_text = next(
            p["text"]
            for p in slack_payloads
            if "engineering deploy" in (p.get("text") or "")
        )
        results = qdrant_client_.search(
            collection_name=_TEST_COLLECTION,
            query_vector=_stable_vector(query_text),
            limit=5,
            with_payload=True,
        )
        slack_hits = [r for r in results if r.payload.get("source_type") == "slack"]
        assert slack_hits, "no Slack chunk surfaced in top-5"

    def test_idempotent_second_run_is_a_no_op(
        self,
        writer: Writer,
        neo4j_driver: Driver,
        qdrant_client_: QdrantClient,
        fake_slack_client: MagicMock,
    ) -> None:
        """Re-running ingest against the same fixture corpus must not
        create new Documents, new Person nodes, or duplicate edges."""
        source = self._build_source(fake_slack_client)
        queue = _ListQueue()

        # First ingest
        first_events = _ingest_once(source, queue)
        _write_events(writer, first_events)

        with neo4j_driver.session() as session:
            doc_count_before = session.run(
                "MATCH (c:SlackMessage) RETURN count(c) AS n"
            ).single()["n"]
            person_count_before = session.run(
                "MATCH (p:Person) RETURN count(p) AS n"
            ).single()["n"]
            edge_count_before = session.run(
                "MATCH ()-[r]->() WHERE type(r) IN ['IN_CHANNEL', 'SENT_BY', 'MENTIONS'] "
                "RETURN count(r) AS n"
            ).single()["n"]
        qdrant_before = len(_all_qdrant_points(qdrant_client_))

        # Reset cursor and re-run with the SAME fixture client. The
        # source emits the same events; the writer must MERGE them
        # idempotently.
        second_source = self._build_source(fake_slack_client)
        second_queue = _ListQueue()
        second_events = _ingest_once(second_source, second_queue)
        _write_events(writer, second_events)

        with neo4j_driver.session() as session:
            doc_count_after = session.run(
                "MATCH (c:SlackMessage) RETURN count(c) AS n"
            ).single()["n"]
            person_count_after = session.run(
                "MATCH (p:Person) RETURN count(p) AS n"
            ).single()["n"]
            edge_count_after = session.run(
                "MATCH ()-[r]->() WHERE type(r) IN ['IN_CHANNEL', 'SENT_BY', 'MENTIONS'] "
                "RETURN count(r) AS n"
            ).single()["n"]
        qdrant_after = len(_all_qdrant_points(qdrant_client_))

        assert doc_count_after == doc_count_before, "re-ingest created new Documents"
        assert person_count_after == person_count_before, (
            "re-ingest duplicated Person nodes"
        )
        assert edge_count_after == edge_count_before, "re-ingest duplicated edges"
        assert qdrant_after == qdrant_before, "re-ingest duplicated Qdrant points"
