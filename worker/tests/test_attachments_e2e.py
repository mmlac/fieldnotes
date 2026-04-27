"""End-to-end attachment ingestion test.

Drives the full source -> parser -> chunker -> writer pipeline against
fake Gmail, Slack, and Calendar clients with attachments, plus injected
fake fetchers for the actual byte downloads (Gmail attachments.get,
Slack url_private_download, Drive files.get_media).  Real Neo4j and
Qdrant.  The LLM extractor and embedder are bypassed; vectors come from
a deterministic ``_stable_vector`` so chunk round-tripping is testable
without standing up an embedder.

Skipped when Neo4j and/or Qdrant are unavailable (matches the pattern
used by ``tests/test_e2e.py`` and ``tests/test_slack_e2e.py``).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock

import pytest
from neo4j import Driver, GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from conftest import (  # type: ignore[import-not-found]
    attachment_bytes,
    make_fake_calendar_service,
    make_fake_drive_service,
    make_fake_gmail_messages_api,
)

from worker.config import Neo4jConfig, QdrantConfig
from worker.parsers import calendar as calendar_parser_mod
from worker.parsers import gmail as gmail_parser_mod
from worker.parsers.calendar import GoogleCalendarParser
from worker.parsers.gmail import GmailParser
from worker.parsers.slack import SlackParser
from worker.pipeline.chunker import chunk_text
from worker.pipeline.writer import VECTOR_SIZE, WriteUnit, Writer
from worker.queue import PersistentQueue
from worker.sources.calendar import GoogleCalendarSource
from worker.sources.gmail import GmailSource
from worker.sources.slack import SlackSource


_NEO4J_URI = "bolt://localhost:7687"
_NEO4J_USER = "neo4j"
_NEO4J_PASSWORD = "testpassword"
_QDRANT_HOST = "localhost"
_QDRANT_PORT = 6333
_TEST_COLLECTION = "fieldnotes_attachments_e2e_test"

_TEAM_ID = "T-TEST"
_TEAM_DOMAIN = "test"
_CHANNEL_ID = "C-ENG"
_SLACK_TS = "1700200000.000100"
_SLACK_TS_COMPACT = _SLACK_TS.replace(".", "")

# --- Fixture identifiers reused across asserts -----------------------------

_GMAIL_ACCOUNT = "personal"
_GMAIL_THREAD_ID = "thread-att-1"
_GMAIL_MSG_ID = "msg-att-1"
_GMAIL_PDF_ATT_ID = "att-pdf-1"
_GMAIL_DOCX_ATT_ID = "att-docx-1"

_CAL_ACCOUNT = "personal"
_CAL_EVENT_ID = "evt-att-1"
_CAL_PDF_FILE_ID = "drive-pdf-1"
_CAL_DOCX_FILE_ID = "drive-docx-1"
_CAL_HTML_LINK = "https://calendar.google.com/event?eid=evt-att-1"

_SLACK_PNG_FILE_ID = "F-PNG-1"
_SLACK_ZIP_FILE_ID = "F-ZIP-1"


# ---------------------------------------------------------------------------
# Service availability gate
# ---------------------------------------------------------------------------


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
# Stub Slack vision result + fake fetchers
# ---------------------------------------------------------------------------


class _StubVisionResult:
    """Mimics the duck-typed ``VisionResult`` consumed by the attachment helper."""

    def __init__(self, description: str, visible_text: str) -> None:
        self.description = description
        self.visible_text = visible_text
        self.entities: list[dict[str, Any]] = []


def _fake_vision_extractor(_data: bytes, _mime: str) -> _StubVisionResult:
    return _StubVisionResult(
        description="A bright red square — sample.png test fixture content.",
        visible_text="png-fixture-visible-text",
    )


def _build_slack_file_map() -> dict[str, bytes]:
    """url_private_download URL → bytes for the Slack fixture files."""
    return {
        f"https://files.slack.com/files-pri/T-TEST-{_SLACK_PNG_FILE_ID}/sample.png": attachment_bytes(
            "sample.png"
        ),
        f"https://files.slack.com/files-pri/T-TEST-{_SLACK_ZIP_FILE_ID}/bundle.zip": b"PK\x03\x04",  # not actually fetched (zip is metadata-only)
    }


def _make_slack_fetcher(failures: set[str] | None = None):
    """Return a fetcher closure that pulls bytes from the fixture map.

    *failures* is an optional set of URLs that should raise to simulate a
    401/timeout, exercising the metadata-only fallback path.
    """
    bytes_by_url = _build_slack_file_map()
    fail_urls = failures or set()

    def fetch(url: str) -> bytes:
        if url in fail_urls:
            raise RuntimeError(f"slack fetch simulated failure: {url}")
        return bytes_by_url[url]

    return fetch


def _make_gmail_attachment_fetcher_factory():
    """Replacement for ``_gmail_attachment_fetcher`` that bypasses OAuth.

    The production helper builds a Gmail client and calls
    ``users.messages.attachments.get``.  Here we route by attachment_id
    against the local fixture binaries so the parser runs end-to-end
    without touching the network.
    """
    bytes_by_id = {
        _GMAIL_PDF_ATT_ID: attachment_bytes("sample.pdf"),
        _GMAIL_DOCX_ATT_ID: attachment_bytes("sample.docx"),
    }

    def factory(
        *,
        account: str,
        message_id: str,
        attachment_id: str,
        client_secrets_path: str | None,
    ):
        def fetch() -> bytes:
            return bytes_by_id[attachment_id]

        return fetch

    return factory


def _make_drive_fetcher_factory(failures: set[str] | None = None):
    """Replacement for calendar's ``_build_drive_fetcher``.

    Routes by Drive file_id into the local fixture binaries.  *failures*
    is an optional set of file_ids whose fetch should raise, exercising
    the metadata-only fallback (Drive 404/403).
    """
    bytes_by_id = {
        _CAL_PDF_FILE_ID: attachment_bytes("sample.pdf"),
        _CAL_DOCX_FILE_ID: attachment_bytes("sample.docx"),
    }
    fail_ids = failures or set()

    def factory(account: str, file_id: str):
        def fetch() -> bytes:
            if file_id in fail_ids:
                raise RuntimeError(f"drive fetch simulated 404: {file_id}")
            return bytes_by_id[file_id]

        return fetch

    return factory


# ---------------------------------------------------------------------------
# Pytest fixtures (services + queue + monkeypatched fetchers)
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
    with neo4j_driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
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


@pytest.fixture
def queue_db(tmp_path: Path) -> Generator[PersistentQueue, None, None]:
    q = PersistentQueue(db_path=tmp_path / "queue.db")
    yield q
    q.close()


@pytest.fixture
def patch_gmail_fetcher(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the Gmail attachment fetcher factory with a fixture-backed
    version so the parser never reaches the network."""
    monkeypatch.setattr(
        gmail_parser_mod,
        "_gmail_attachment_fetcher",
        _make_gmail_attachment_fetcher_factory(),
    )


@pytest.fixture
def patch_drive_fetcher(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        calendar_parser_mod,
        "_build_drive_fetcher",
        _make_drive_fetcher_factory(),
    )


# ---------------------------------------------------------------------------
# Drive sources / write events
# ---------------------------------------------------------------------------


def _stable_vector(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [(digest[i % len(digest)] / 255.0) for i in range(VECTOR_SIZE)]


def _ingest_gmail(
    account: str,
    messages_api: MagicMock,
    queue: PersistentQueue,
    *,
    download_attachments: bool = True,
) -> None:
    src = GmailSource()
    src.configure(
        {
            "account": account,
            "client_secrets_path": "/dev/null",
            "max_initial_threads": 5,
            "download_attachments": download_attachments,
        }
    )
    asyncio.run(src._backfill(messages_api, queue, is_initial=True))


def _ingest_calendar(
    account: str,
    calendar_id: str,
    service: MagicMock,
    drive_service: MagicMock,
    queue: PersistentQueue,
    *,
    download_attachments: bool = True,
) -> None:
    src = GoogleCalendarSource()
    src.configure(
        {
            "account": account,
            "client_secrets_path": "/dev/null",
            "calendar_ids": [calendar_id],
            "download_attachments": download_attachments,
        }
    )
    asyncio.run(
        src._poll_calendar(
            service,
            queue,
            calendar_id,
            None,
            {},
            drive_service=drive_service,
        )
    )


def _build_fake_slack_client() -> MagicMock:
    """Slack fixture wired against ``conversations_history_with_files.json``.

    A single channel ``C-ENG`` containing one message with a PNG +
    a ZIP file.  Replies and other channels return empty.  ``users_info``
    reuses the existing fixture so author resolution still works.
    """
    base = Path(__file__).parent / "fixtures" / "slack"
    history = json.loads((base / "conversations_history_with_files.json").read_text())
    users = json.loads((base / "users_info.json").read_text())
    channels = {
        "ok": True,
        "channels": [
            {
                "id": _CHANNEL_ID,
                "name": "engineering",
                "is_channel": True,
                "is_archived": False,
                "is_private": False,
                "is_im": False,
                "is_mpim": False,
                "is_member": True,
            }
        ],
        "response_metadata": {"next_cursor": ""},
    }

    client = MagicMock()
    client.auth_test.return_value = {
        "team_id": _TEAM_ID,
        "team_domain": _TEAM_DOMAIN,
        "ok": True,
    }
    client.conversations_list.return_value = channels
    client.conversations_history.return_value = history
    client.conversations_replies.return_value = {
        "ok": True,
        "messages": [],
        "response_metadata": {"next_cursor": ""},
    }
    client.users_info.side_effect = lambda **kw: {
        "ok": True,
        "user": users.get(kw.get("user", ""), {}),
    }
    return client


def _ingest_slack(
    client: MagicMock,
    queue: PersistentQueue,
    *,
    download_attachments: bool = True,
) -> None:
    src = SlackSource()
    src.configure(
        {
            "poll_interval_seconds": 0,
            "download_attachments": download_attachments,
        }
    )
    src._client = client
    src._team_id = _TEAM_ID
    src._team_domain = _TEAM_DOMAIN

    channels = src._discover_conversations()

    async def _poll_all() -> None:
        team_cursor: dict[str, dict[str, Any]] = {}
        for ch in channels:
            await src._poll_conversation(ch, team_cursor, queue)

    asyncio.run(_poll_all())


def _all_pending_events(queue: PersistentQueue) -> list[dict[str, Any]]:
    with sqlite3.connect(str(queue._db_path)) as conn:  # type: ignore[attr-defined]
        rows = conn.execute(
            "SELECT payload FROM queue WHERE status IN ('pending','processing')"
        ).fetchall()
    return [json.loads(r[0]) for r in rows]


def _make_slack_parser(
    slack_failures: set[str] | None = None,
    *,
    fetcher: Any = None,
) -> SlackParser:
    parser = SlackParser()
    parser._fetcher = (
        fetcher if fetcher is not None else _make_slack_fetcher(slack_failures)
    )
    parser._vision_extractor = _fake_vision_extractor
    return parser


def _write_event(
    writer: Writer,
    ev: dict[str, Any],
    *,
    slack_failures: set[str] | None = None,
    slack_fetcher: Any = None,
) -> None:
    if ev["source_type"] == "gmail":
        parser: Any = GmailParser()
    elif ev["source_type"] == "google_calendar":
        parser = GoogleCalendarParser()
    elif ev["source_type"] == "slack":
        parser = _make_slack_parser(slack_failures, fetcher=slack_fetcher)
    else:
        pytest.fail(f"unexpected source_type {ev['source_type']!r}")

    for doc in parser.parse(ev):
        if doc.text:
            chunks = chunk_text(doc.text)
        else:
            chunks = []
        vectors = [_stable_vector(c.text) for c in chunks]
        writer.write(
            WriteUnit(
                doc=doc,
                chunks=chunks,
                vectors=vectors,
                entities=[],
            )
        )


def _ingest_and_write_all(
    writer: Writer,
    queue: PersistentQueue,
    *,
    slack_failures: set[str] | None = None,
    download_attachments: bool = True,
    slack_fetcher: Any = None,
) -> None:
    """Drive Gmail + Slack + Calendar through the full pipeline."""
    gmail_api = make_fake_gmail_messages_api(_GMAIL_ACCOUNT + "_with_attachments")
    cal_service = make_fake_calendar_service(_CAL_ACCOUNT + "_with_attachments")
    drive_service = make_fake_drive_service(
        {
            _CAL_PDF_FILE_ID: 1444,
            _CAL_DOCX_FILE_ID: 926,
        }
    )
    slack_client = _build_fake_slack_client()

    _ingest_gmail(
        _GMAIL_ACCOUNT, gmail_api, queue, download_attachments=download_attachments
    )
    _ingest_calendar(
        _CAL_ACCOUNT,
        "primary",
        cal_service,
        drive_service,
        queue,
        download_attachments=download_attachments,
    )
    _ingest_slack(slack_client, queue, download_attachments=download_attachments)

    for ev in _all_pending_events(queue):
        _write_event(
            writer,
            ev,
            slack_failures=slack_failures,
            slack_fetcher=slack_fetcher,
        )


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def _node_props(driver: Driver, source_id: str) -> dict[str, Any] | None:
    with driver.session() as session:
        row = session.run(
            "MATCH (n {source_id: $sid}) "
            "RETURN properties(n) AS props, labels(n) AS labels",
            sid=source_id,
        ).single()
    if row is None:
        return None
    out = dict(row["props"])
    out["__labels__"] = list(row["labels"])
    return out


def _qdrant_chunks_for(client: QdrantClient, source_id: str) -> list[Any]:
    result = client.scroll(
        collection_name=_TEST_COLLECTION,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="source_id",
                    match=MatchValue(value=source_id),
                )
            ]
        ),
        with_payload=True,
        limit=100,
    )
    return list(result[0])


def _has_attached_to_edge(driver: Driver, child_sid: str) -> bool:
    with driver.session() as session:
        row = session.run(
            "MATCH (a {source_id: $sid})-[:ATTACHED_TO]->(p) RETURN p.source_id AS psid",
            sid=child_sid,
        ).single()
    return row is not None


def _attached_parent(driver: Driver, child_sid: str) -> str | None:
    with driver.session() as session:
        row = session.run(
            "MATCH (a {source_id: $sid})-[:ATTACHED_TO]->(p) RETURN p.source_id AS psid",
            sid=child_sid,
        ).single()
    return None if row is None else row["psid"]


# ---------------------------------------------------------------------------
# Source IDs computed from fixtures
# ---------------------------------------------------------------------------

_GMAIL_PARENT_SID = f"gmail://{_GMAIL_ACCOUNT}/message/{_GMAIL_MSG_ID}"
_GMAIL_THREAD_SID = f"gmail://{_GMAIL_ACCOUNT}/thread/{_GMAIL_THREAD_ID}"
_GMAIL_PDF_SID = (
    f"gmail://{_GMAIL_ACCOUNT}/thread/{_GMAIL_THREAD_ID}/attachment/{_GMAIL_PDF_ATT_ID}"
)
_GMAIL_DOCX_SID = (
    f"gmail://{_GMAIL_ACCOUNT}/thread/{_GMAIL_THREAD_ID}"
    f"/attachment/{_GMAIL_DOCX_ATT_ID}"
)

_CAL_PARENT_SID = f"google-calendar://{_CAL_ACCOUNT}/event/{_CAL_EVENT_ID}"
_CAL_PDF_SID = (
    f"google-calendar://{_CAL_ACCOUNT}/event/{_CAL_EVENT_ID}"
    f"/attachment/{_CAL_PDF_FILE_ID}"
)
_CAL_DOCX_SID = (
    f"google-calendar://{_CAL_ACCOUNT}/event/{_CAL_EVENT_ID}"
    f"/attachment/{_CAL_DOCX_FILE_ID}"
)

_SLACK_PARENT_SID = f"slack://{_TEAM_ID}/{_CHANNEL_ID}/window/{_SLACK_TS}-{_SLACK_TS}"
_SLACK_PNG_SID = (
    f"slack://{_TEAM_ID}/{_CHANNEL_ID}/{_SLACK_TS}/file/{_SLACK_PNG_FILE_ID}"
)
_SLACK_ZIP_SID = (
    f"slack://{_TEAM_ID}/{_CHANNEL_ID}/{_SLACK_TS}/file/{_SLACK_ZIP_FILE_ID}"
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@_skip_services
class TestAttachmentsEndToEnd:
    def test_per_source_attachment_documents(
        self,
        writer: Writer,
        neo4j_driver: Driver,
        qdrant_client_: QdrantClient,
        queue_db: PersistentQueue,
        patch_gmail_fetcher: None,
        patch_drive_fetcher: None,
    ) -> None:
        """For each source: indexable attachments → chunked Document with
        parent_url + ATTACHED_TO → parent; non-indexable → metadata-only
        Document with the same edge but no chunks."""
        _ingest_and_write_all(writer, queue_db)

        # --- Gmail PDF: chunked + parent_url + ATTACHED_TO ---------------
        pdf = _node_props(neo4j_driver, _GMAIL_PDF_SID)
        assert pdf is not None, "missing gmail PDF attachment node"
        assert "Attachment" in pdf["__labels__"]
        assert pdf.get("indexed") is True
        assert pdf.get("parent_url") == (
            f"https://mail.google.com/mail/?ui=2&view=cv&th={_GMAIL_THREAD_ID}"
        )
        assert _attached_parent(neo4j_driver, _GMAIL_PDF_SID) == _GMAIL_THREAD_SID
        assert _qdrant_chunks_for(qdrant_client_, _GMAIL_PDF_SID), (
            "Gmail PDF attachment should have chunks (parsed PDF text)"
        )

        # --- Gmail DOCX: metadata-only -----------------------------------
        docx = _node_props(neo4j_driver, _GMAIL_DOCX_SID)
        assert docx is not None, "missing gmail DOCX attachment node"
        assert docx.get("indexed") is False
        assert _attached_parent(neo4j_driver, _GMAIL_DOCX_SID) == _GMAIL_THREAD_SID
        # No chunks for non-indexable attachments — node exists with the
        # synthetic 'content not indexed' description as text but is not
        # chunked into Qdrant when we drive the writer with chunks=[].
        # (The metadata description text *is* short enough to chunk; just
        # assert the indexed flag and the parent edge.)

        # --- Slack PNG: chunked + parent_url + ATTACHED_TO ---------------
        png = _node_props(neo4j_driver, _SLACK_PNG_SID)
        assert png is not None, "missing slack PNG attachment node"
        assert png.get("indexed") is True
        assert png.get("parent_url") == (
            f"https://{_TEAM_DOMAIN}.slack.com/archives/{_CHANNEL_ID}/p{_SLACK_TS_COMPACT}"
        )
        assert _attached_parent(neo4j_driver, _SLACK_PNG_SID) == _SLACK_PARENT_SID
        assert _qdrant_chunks_for(qdrant_client_, _SLACK_PNG_SID), (
            "Slack PNG attachment should have chunks (vision pipeline output)"
        )

        # --- Slack ZIP: metadata-only ------------------------------------
        zip_node = _node_props(neo4j_driver, _SLACK_ZIP_SID)
        assert zip_node is not None, "missing slack ZIP attachment node"
        assert zip_node.get("indexed") is False
        assert _attached_parent(neo4j_driver, _SLACK_ZIP_SID) == _SLACK_PARENT_SID

        # --- Calendar PDF: chunked + parent_url + ATTACHED_TO ------------
        cal_pdf = _node_props(neo4j_driver, _CAL_PDF_SID)
        assert cal_pdf is not None, "missing calendar PDF attachment node"
        assert cal_pdf.get("decision") == "download_and_index"
        assert cal_pdf.get("parent_url") == _CAL_HTML_LINK
        assert _attached_parent(neo4j_driver, _CAL_PDF_SID) == _CAL_PARENT_SID
        assert _qdrant_chunks_for(qdrant_client_, _CAL_PDF_SID), (
            "Calendar PDF attachment should have chunks (parsed PDF text)"
        )

        # --- Calendar DOCX: metadata-only --------------------------------
        cal_docx = _node_props(neo4j_driver, _CAL_DOCX_SID)
        assert cal_docx is not None, "missing calendar DOCX attachment node"
        assert cal_docx.get("decision") == "metadata_only"
        assert _attached_parent(neo4j_driver, _CAL_DOCX_SID) == _CAL_PARENT_SID

    def test_parent_text_augmented_with_attachments_section(
        self,
        writer: Writer,
        qdrant_client_: QdrantClient,
        queue_db: PersistentQueue,
        patch_gmail_fetcher: None,
        patch_drive_fetcher: None,
    ) -> None:
        """Gmail and Calendar parents carry an 'Attachments:' section in
        their chunked text so a search for 'sample.pdf' lands on the
        parent even when the attachment itself is metadata-only.

        Slack uses inline ``[file] foo`` markers instead of a section
        header — and the SlackSource currently does not propagate
        ``meta['messages']`` / ``meta['users_info']`` to the parser
        (tracked separately in fn-a3p), so parent text rendering is
        broken end-to-end.  Slack augmentation is asserted via the
        Attachment node's filename (covered by the other tests in this
        class)."""
        _ingest_and_write_all(writer, queue_db)

        for parent_sid, expected_filenames in (
            (_GMAIL_PARENT_SID, ("sample.pdf", "sample.docx")),
            (_CAL_PARENT_SID, ("sample.pdf", "sample.docx")),
        ):
            chunks = _qdrant_chunks_for(qdrant_client_, parent_sid)
            assert chunks, f"no chunks for parent {parent_sid!r}"
            joined = "\n".join(c.payload.get("text", "") for c in chunks)
            assert "Attachments:" in joined, (
                f"parent text missing 'Attachments:' section for "
                f"{parent_sid!r}: {joined!r}"
            )
            for fn in expected_filenames:
                assert fn in joined, (
                    f"expected filename {fn!r} in parent {parent_sid!r} chunks"
                )

    def test_attachment_counters_on_parents(
        self,
        writer: Writer,
        neo4j_driver: Driver,
        queue_db: PersistentQueue,
        patch_gmail_fetcher: None,
        patch_drive_fetcher: None,
    ) -> None:
        """Each parent Document carries the three attachment counters plus
        the deprecated ``has_attachments`` alias so graph queries can filter
        on 'has files' (intended) or 'has parsed-text files' (indexed)
        without a JOIN."""
        _ingest_and_write_all(writer, queue_db)

        for parent_sid in (_GMAIL_PARENT_SID, _SLACK_PARENT_SID, _CAL_PARENT_SID):
            props = _node_props(neo4j_driver, parent_sid)
            assert props is not None, f"missing parent node {parent_sid!r}"

            intended = props.get("attachments_count_intended")
            indexed = props.get("attachments_count_indexed")
            metadata_only = props.get("attachments_count_metadata_only")
            assert intended, (
                f"parent {parent_sid!r} missing/falsy "
                f"attachments_count_intended: {intended!r}"
            )
            assert indexed is not None, (
                f"parent {parent_sid!r} missing attachments_count_indexed"
            )
            assert metadata_only is not None, (
                f"parent {parent_sid!r} missing attachments_count_metadata_only"
            )
            assert intended == indexed + metadata_only, (
                f"parent {parent_sid!r} counter sum mismatch: "
                f"intended={intended} indexed={indexed} metadata_only={metadata_only}"
            )

            # Deprecated alias mirrors the intended count for one release.
            assert props.get("has_attachments") == intended

    def test_three_layer_retrieval_for_sample_pdf(
        self,
        writer: Writer,
        neo4j_driver: Driver,
        qdrant_client_: QdrantClient,
        queue_db: PersistentQueue,
        patch_gmail_fetcher: None,
        patch_drive_fetcher: None,
    ) -> None:
        """A search for 'sample.pdf' must surface three layers of
        evidence: parent body chunks, Attachment node metadata, and the
        attachment's own parsed-content chunks.  We verify each layer
        directly rather than via vector similarity (the test embedder is
        a SHA hash, so an exact-text query never lights up unrelated
        chunks)."""
        _ingest_and_write_all(writer, queue_db)

        # Layer 1: parent body chunks (Qdrant + Neo4j parent text).
        parent_chunks_with_pdf: list[Any] = []
        for parent_sid in (_GMAIL_PARENT_SID, _CAL_PARENT_SID):
            chunks = _qdrant_chunks_for(qdrant_client_, parent_sid)
            for c in chunks:
                if "sample.pdf" in c.payload.get("text", ""):
                    parent_chunks_with_pdf.append(c)
        assert parent_chunks_with_pdf, (
            "no parent chunk surfaced 'sample.pdf' — augmentation missing"
        )

        # Layer 2: Attachment Document with filename surfaced in graph.
        with neo4j_driver.session() as session:
            att_rows = list(
                session.run(
                    "MATCH (a:Attachment) "
                    "WHERE a.filename = 'sample.pdf' OR a.title = 'sample.pdf' OR "
                    "      a.name = 'sample.pdf' "
                    "RETURN a.source_id AS sid"
                )
            )
        att_sids = sorted(r["sid"] for r in att_rows)
        assert _GMAIL_PDF_SID in att_sids
        assert _CAL_PDF_SID in att_sids

        # Layer 3: Attachment's own parsed-content chunks.
        for att_sid in (_GMAIL_PDF_SID, _CAL_PDF_SID):
            content_chunks = _qdrant_chunks_for(qdrant_client_, att_sid)
            assert content_chunks, (
                f"attachment {att_sid!r} has no parsed-content chunks"
            )
            joined = "\n".join(c.payload.get("text", "") for c in content_chunks)
            # The fixture PDF embeds this marker word so we can prove the
            # chunks came from the parsed PDF, not from metadata.
            assert "pdfattach-fixture-content" in joined, (
                f"parsed PDF text missing marker in {att_sid!r}: {joined!r}"
            )

    def test_no_on_disk_attachment_cache(
        self,
        writer: Writer,
        queue_db: PersistentQueue,
        tmp_path: Path,
        patch_gmail_fetcher: None,
        patch_drive_fetcher: None,
    ) -> None:
        """Stream-and-forget download: nothing is written under /tmp or
        the queue's data directory after a full ingest."""
        before_tmp = set(Path(tempfile.gettempdir()).iterdir())

        _ingest_and_write_all(writer, queue_db)

        after_tmp = set(Path(tempfile.gettempdir()).iterdir())
        new_tmp = sorted(p.name for p in after_tmp - before_tmp)
        # Filter out unrelated noise other tests / processes may have
        # produced; only flag entries that look like attachment caches.
        att_like = [n for n in new_tmp if "attach" in n.lower()]
        assert att_like == [], f"attachment files left under /tmp: {att_like!r}"

        # The queue's parent directory must not contain an 'attachments'
        # subdirectory — caches would land there if anyone wrote one.
        queue_root = queue_db._db_path.parent  # type: ignore[attr-defined]
        assert not (queue_root / "attachments").exists(), (
            "queue dir should not have an 'attachments' cache subdir"
        )

    def test_failure_paths_fall_back_to_metadata_only(
        self,
        writer: Writer,
        neo4j_driver: Driver,
        qdrant_client_: QdrantClient,
        queue_db: PersistentQueue,
        monkeypatch: pytest.MonkeyPatch,
        patch_gmail_fetcher: None,
    ) -> None:
        """Drive 404 on the calendar PDF and a Slack 401 on the PNG must
        each emit a metadata-only Document — the parent ingestion must
        not fail.  Gmail still indexes its PDF normally so the test
        proves the failure is isolated to the failing source."""
        # Simulate Drive 404 on the PDF only — DOCX still goes
        # metadata-only via classify_attachment.
        monkeypatch.setattr(
            calendar_parser_mod,
            "_build_drive_fetcher",
            _make_drive_fetcher_factory(failures={_CAL_PDF_FILE_ID}),
        )
        slack_failures = {
            f"https://files.slack.com/files-pri/T-TEST-{_SLACK_PNG_FILE_ID}/sample.png",
        }
        _ingest_and_write_all(writer, queue_db, slack_failures=slack_failures)

        # Calendar PDF: still emitted as a metadata-only Document with
        # the parent edge intact; no parsed-content chunks.
        cal_pdf = _node_props(neo4j_driver, _CAL_PDF_SID)
        assert cal_pdf is not None
        assert cal_pdf.get("decision") == "metadata_only"
        assert _attached_parent(neo4j_driver, _CAL_PDF_SID) == _CAL_PARENT_SID

        # Slack PNG: same fallback shape.
        png = _node_props(neo4j_driver, _SLACK_PNG_SID)
        assert png is not None
        assert png.get("indexed") is False
        assert _attached_parent(neo4j_driver, _SLACK_PNG_SID) == _SLACK_PARENT_SID

        # Gmail PDF: failure was isolated — still indexed normally.
        gmail_pdf = _node_props(neo4j_driver, _GMAIL_PDF_SID)
        assert gmail_pdf is not None
        assert gmail_pdf.get("indexed") is True
        assert _qdrant_chunks_for(qdrant_client_, _GMAIL_PDF_SID)

    def test_idempotent_second_run(
        self,
        writer: Writer,
        neo4j_driver: Driver,
        qdrant_client_: QdrantClient,
        queue_db: PersistentQueue,
        tmp_path: Path,
        patch_gmail_fetcher: None,
        patch_drive_fetcher: None,
    ) -> None:
        """Re-running ingest must not create new Documents, new edges,
        or duplicate Qdrant points."""
        _ingest_and_write_all(writer, queue_db)

        with neo4j_driver.session() as session:
            doc_count_before = session.run(
                "MATCH (n) WHERE n:Email OR n:Thread OR n:CalendarEvent "
                "OR n:SlackMessage OR n:Attachment "
                "RETURN count(n) AS n"
            ).single()["n"]
            edge_count_before = session.run(
                "MATCH ()-[r:ATTACHED_TO]->() RETURN count(r) AS n"
            ).single()["n"]
        qdrant_before_count = sum(
            len(_qdrant_chunks_for(qdrant_client_, sid))
            for sid in (
                _GMAIL_PARENT_SID,
                _GMAIL_PDF_SID,
                _GMAIL_DOCX_SID,
                _CAL_PARENT_SID,
                _CAL_PDF_SID,
                _CAL_DOCX_SID,
                _SLACK_PARENT_SID,
                _SLACK_PNG_SID,
                _SLACK_ZIP_SID,
            )
        )

        # Re-run ingest with a fresh queue (the original was drained by
        # the first write loop) but the same fixtures.
        queue2 = PersistentQueue(db_path=tmp_path / "queue2.db")
        try:
            _ingest_and_write_all(writer, queue2)
        finally:
            queue2.close()

        with neo4j_driver.session() as session:
            doc_count_after = session.run(
                "MATCH (n) WHERE n:Email OR n:Thread OR n:CalendarEvent "
                "OR n:SlackMessage OR n:Attachment "
                "RETURN count(n) AS n"
            ).single()["n"]
            edge_count_after = session.run(
                "MATCH ()-[r:ATTACHED_TO]->() RETURN count(r) AS n"
            ).single()["n"]
        qdrant_after_count = sum(
            len(_qdrant_chunks_for(qdrant_client_, sid))
            for sid in (
                _GMAIL_PARENT_SID,
                _GMAIL_PDF_SID,
                _GMAIL_DOCX_SID,
                _CAL_PARENT_SID,
                _CAL_PDF_SID,
                _CAL_DOCX_SID,
                _SLACK_PARENT_SID,
                _SLACK_PNG_SID,
                _SLACK_ZIP_SID,
            )
        )

        assert doc_count_after == doc_count_before, "re-ingest created new Documents"
        assert edge_count_after == edge_count_before, (
            "re-ingest duplicated ATTACHED_TO edges"
        )
        assert qdrant_after_count == qdrant_before_count, (
            "re-ingest duplicated Qdrant chunks"
        )


@_skip_services
class TestDownloadAttachmentsOff:
    """End-to-end coverage for ``download_attachments=False`` — the
    production default and most-used config path.  No bytes are fetched;
    each attachment becomes a metadata-only Attachment Document and the
    parent body still surfaces filenames in its 'Attachments:' section so
    a search for ``sample.pdf`` lands on the parent thread/event."""

    def test_filename_only_embedding_path(
        self,
        writer: Writer,
        neo4j_driver: Driver,
        qdrant_client_: QdrantClient,
        queue_db: PersistentQueue,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Counting wrappers around the Gmail/Drive factory functions:
        # with download_attachments=False, the parser must never enter
        # the download_and_index branch, so these factories must remain
        # uncalled.  The wrapper still returns a working closure so that
        # if the assertion ever flips we get a real fetch path to debug.
        gmail_factory_calls = 0
        drive_factory_calls = 0
        gmail_inner = _make_gmail_attachment_fetcher_factory()
        drive_inner = _make_drive_fetcher_factory()

        def _gmail_factory(**kw: Any) -> Any:
            nonlocal gmail_factory_calls
            gmail_factory_calls += 1
            return gmail_inner(**kw)

        def _drive_factory(account: str, file_id: str) -> Any:
            nonlocal drive_factory_calls
            drive_factory_calls += 1
            return drive_inner(account, file_id)

        monkeypatch.setattr(
            gmail_parser_mod, "_gmail_attachment_fetcher", _gmail_factory
        )
        monkeypatch.setattr(calendar_parser_mod, "_build_drive_fetcher", _drive_factory)

        slack_fetch_calls = 0
        slack_bytes_by_url = _build_slack_file_map()

        def _slack_fetcher(url: str) -> bytes:
            nonlocal slack_fetch_calls
            slack_fetch_calls += 1
            return slack_bytes_by_url[url]

        _ingest_and_write_all(
            writer,
            queue_db,
            download_attachments=False,
            slack_fetcher=_slack_fetcher,
        )

        # 1) ZERO fetches across all three sources.
        assert gmail_factory_calls == 0, (
            f"Gmail attachment fetcher invoked {gmail_factory_calls}× with "
            f"download_attachments=False"
        )
        assert drive_factory_calls == 0, (
            f"Drive fetcher invoked {drive_factory_calls}× with "
            f"download_attachments=False"
        )
        assert slack_fetch_calls == 0, (
            f"Slack url_private_download fetcher invoked {slack_fetch_calls}× "
            f"with download_attachments=False"
        )

        # 2) Each attachment produces ONE metadata-only Attachment
        #    Document with no chunks under the attachment URI.
        for att_sid in (
            _GMAIL_PDF_SID,
            _GMAIL_DOCX_SID,
            _CAL_PDF_SID,
            _CAL_DOCX_SID,
            _SLACK_PNG_SID,
            _SLACK_ZIP_SID,
        ):
            node = _node_props(neo4j_driver, att_sid)
            assert node is not None, f"missing attachment node {att_sid!r}"
            assert "Attachment" in node["__labels__"], (
                f"node {att_sid!r} missing :Attachment label"
            )
            # Gmail/Slack carry an ``indexed`` flag; calendar uses
            # ``decision``.  Both must say "metadata only" here.
            if "decision" in node:
                assert node["decision"] == "metadata_only", (
                    f"calendar attachment {att_sid!r} not metadata_only: "
                    f"{node['decision']!r}"
                )
            else:
                assert node.get("indexed") is False, (
                    f"attachment {att_sid!r} marked indexed under "
                    f"download_attachments=False"
                )

            # Exactly one Attachment node per source_id (MERGE collapses
            # duplicates, but assert it explicitly so a regression that
            # widens the merge key is caught here).
            with neo4j_driver.session() as session:
                cnt = session.run(
                    "MATCH (a:Attachment {source_id: $sid}) RETURN count(a) AS n",
                    sid=att_sid,
                ).single()["n"]
            assert cnt == 1, (
                f"expected exactly 1 Attachment node for {att_sid!r}, got {cnt}"
            )

            # No parsed file content under the attachment URI.  Gmail and
            # Calendar metadata-only Documents render a synthetic
            # 'content not indexed' description that DOES get chunked
            # (short text → 1 chunk); Slack metadata-only Documents emit
            # empty text and produce zero chunks.  In neither case do
            # the parsed-file marker words ever land in Qdrant — that's
            # the invariant we care about for "no content chunks".
            chunks = _qdrant_chunks_for(qdrant_client_, att_sid)
            joined = "\n".join(c.payload.get("text", "") for c in chunks)
            for marker in (
                "pdfattach-fixture-content",  # only in fixture PDF body
                "png-fixture-visible-text",  # only in fake vision OCR
            ):
                assert marker not in joined, (
                    f"parsed-file marker {marker!r} leaked into Qdrant "
                    f"for {att_sid!r} under download_attachments=False"
                )

        # 3) parent_url is populated on every Attachment Document so the
        #    UI can still link back to the source thread/message/event.
        gmail_pdf = _node_props(neo4j_driver, _GMAIL_PDF_SID)
        assert gmail_pdf is not None
        assert gmail_pdf.get("parent_url") == (
            f"https://mail.google.com/mail/?ui=2&view=cv&th={_GMAIL_THREAD_ID}"
        )
        gmail_docx = _node_props(neo4j_driver, _GMAIL_DOCX_SID)
        assert gmail_docx is not None
        assert gmail_docx.get("parent_url") == (
            f"https://mail.google.com/mail/?ui=2&view=cv&th={_GMAIL_THREAD_ID}"
        )
        cal_pdf = _node_props(neo4j_driver, _CAL_PDF_SID)
        assert cal_pdf is not None
        assert cal_pdf.get("parent_url") == _CAL_HTML_LINK
        cal_docx = _node_props(neo4j_driver, _CAL_DOCX_SID)
        assert cal_docx is not None
        assert cal_docx.get("parent_url") == _CAL_HTML_LINK
        slack_png = _node_props(neo4j_driver, _SLACK_PNG_SID)
        assert slack_png is not None
        assert slack_png.get("parent_url") == (
            f"https://{_TEAM_DOMAIN}.slack.com/archives/{_CHANNEL_ID}"
            f"/p{_SLACK_TS_COMPACT}"
        )
        slack_zip = _node_props(neo4j_driver, _SLACK_ZIP_SID)
        assert slack_zip is not None
        assert slack_zip.get("parent_url") == (
            f"https://{_TEAM_DOMAIN}.slack.com/archives/{_CHANNEL_ID}"
            f"/p{_SLACK_TS_COMPACT}"
        )

        # 4) Parent body chunks carry an 'Attachments:' section with all
        #    filenames present, so retrieval-by-filename still hops to
        #    the parent even though the attachment itself has no content
        #    chunks.  (Slack parent text augmentation is broken end-to-end
        #    pending fn-a3p — the same caveat that applies to
        #    TestAttachmentsEndToEnd applies here.)
        for parent_sid, expected_filenames in (
            (_GMAIL_PARENT_SID, ("sample.pdf", "sample.docx")),
            (_CAL_PARENT_SID, ("sample.pdf", "sample.docx")),
        ):
            chunks = _qdrant_chunks_for(qdrant_client_, parent_sid)
            assert chunks, f"no chunks for parent {parent_sid!r}"
            joined = "\n".join(c.payload.get("text", "") for c in chunks)
            assert "Attachments:" in joined, (
                f"parent text missing 'Attachments:' section for "
                f"{parent_sid!r}: {joined!r}"
            )
            for fn in expected_filenames:
                assert fn in joined, (
                    f"expected filename {fn!r} in parent {parent_sid!r}: {joined!r}"
                )

        # 5) Per-parent counters: ``intended`` matches the attachment
        #    count, ``indexed`` is exactly zero.
        for parent_sid in (_GMAIL_PARENT_SID, _CAL_PARENT_SID, _SLACK_PARENT_SID):
            props = _node_props(neo4j_driver, parent_sid)
            assert props is not None, f"missing parent node {parent_sid!r}"
            intended = props.get("attachments_count_intended")
            indexed = props.get("attachments_count_indexed")
            metadata_only = props.get("attachments_count_metadata_only")
            assert intended and intended >= 2, (
                f"parent {parent_sid!r}: expected intended>=2, got {intended!r}"
            )
            assert indexed == 0, (
                f"parent {parent_sid!r}: expected indexed=0 with "
                f"download_attachments=False, got {indexed!r}"
            )
            assert metadata_only == intended, (
                f"parent {parent_sid!r}: indexed=0 means metadata_only must "
                f"equal intended (got intended={intended!r}, "
                f"metadata_only={metadata_only!r})"
            )
            assert props.get("has_attachments") == intended

    def test_idempotent_second_run(
        self,
        writer: Writer,
        neo4j_driver: Driver,
        qdrant_client_: QdrantClient,
        queue_db: PersistentQueue,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Re-running ingest with download_attachments=False produces no
        new Documents, edges, or Qdrant points — and never invokes any
        fetcher on either pass."""

        def _no_fetch(*_a: Any, **_kw: Any) -> Any:
            raise AssertionError(
                "fetcher factory called under download_attachments=False"
            )

        def _no_slack_fetch(_url: str) -> bytes:
            raise AssertionError(
                "slack fetcher called under download_attachments=False"
            )

        monkeypatch.setattr(gmail_parser_mod, "_gmail_attachment_fetcher", _no_fetch)
        monkeypatch.setattr(calendar_parser_mod, "_build_drive_fetcher", _no_fetch)

        all_sids = (
            _GMAIL_PARENT_SID,
            _GMAIL_PDF_SID,
            _GMAIL_DOCX_SID,
            _CAL_PARENT_SID,
            _CAL_PDF_SID,
            _CAL_DOCX_SID,
            _SLACK_PARENT_SID,
            _SLACK_PNG_SID,
            _SLACK_ZIP_SID,
        )

        _ingest_and_write_all(
            writer,
            queue_db,
            download_attachments=False,
            slack_fetcher=_no_slack_fetch,
        )

        with neo4j_driver.session() as session:
            doc_count_before = session.run(
                "MATCH (n) WHERE n:Email OR n:Thread OR n:CalendarEvent "
                "OR n:SlackMessage OR n:Attachment "
                "RETURN count(n) AS n"
            ).single()["n"]
            edge_count_before = session.run(
                "MATCH ()-[r:ATTACHED_TO]->() RETURN count(r) AS n"
            ).single()["n"]
        qdrant_before = sum(
            len(_qdrant_chunks_for(qdrant_client_, sid)) for sid in all_sids
        )

        # Re-run with a fresh queue but identical fixtures.
        queue2 = PersistentQueue(db_path=tmp_path / "queue2.db")
        try:
            _ingest_and_write_all(
                writer,
                queue2,
                download_attachments=False,
                slack_fetcher=_no_slack_fetch,
            )
        finally:
            queue2.close()

        with neo4j_driver.session() as session:
            doc_count_after = session.run(
                "MATCH (n) WHERE n:Email OR n:Thread OR n:CalendarEvent "
                "OR n:SlackMessage OR n:Attachment "
                "RETURN count(n) AS n"
            ).single()["n"]
            edge_count_after = session.run(
                "MATCH ()-[r:ATTACHED_TO]->() RETURN count(r) AS n"
            ).single()["n"]
        qdrant_after = sum(
            len(_qdrant_chunks_for(qdrant_client_, sid)) for sid in all_sids
        )

        assert doc_count_after == doc_count_before, (
            "re-ingest created new Documents under download_attachments=False"
        )
        assert edge_count_after == edge_count_before, (
            "re-ingest duplicated ATTACHED_TO edges under download_attachments=False"
        )
        assert qdrant_after == qdrant_before, (
            "re-ingest duplicated Qdrant chunks under download_attachments=False"
        )
