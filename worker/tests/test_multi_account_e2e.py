"""End-to-end multi-account Gmail + Calendar ingestion test.

Drives the full source -> parser -> writer pipeline against fake Gmail
and Google Calendar clients backed by canned JSON fixtures, with real
Neo4j and Qdrant.  Two Gmail accounts (``personal``, ``work``) and two
Calendar accounts (``personal``, ``shared``) plus a ``[me]`` self-identity
block exercise the multi-account guarantees in one run:

* Per-account doc URIs (``gmail://<account>/...``,
  ``google-calendar://<account>/...``) keep colliding ``thread_id`` /
  ``calendar_id`` values from collapsing across accounts.
* Cross-account email-keyed Person merge: ``alice@example.com`` resolves
  to a single Person node regardless of which account she appears in.
* ``[me]`` survivor: ``me@personal.com`` and ``me@work.com`` collapse via
  ``is_self = true`` + ``SAME_AS`` edge.
* Per-account cursors live in ``queue.db.cursors`` (no JSON cursor
  files), keyed by ``gmail:<account>`` / ``calendar:<account>``.
* Doctor reports per-account auth status and the [me] block.
* The ``fieldnotes migrate gmail-multiaccount`` retag rewrites
  legacy-shape source_id values; subsequent ingest is consistent.
* Idempotent re-ingest produces no new docs / Person / edges.

Skipped when Neo4j and/or Qdrant are unavailable (matches the pattern
used by ``tests/test_e2e.py`` and ``tests/test_slack_e2e.py``).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import pytest
from neo4j import Driver, GraphDatabase
from qdrant_client import QdrantClient

from worker.cli import migrate as migrate_cli
from worker.config import (
    CalendarAccountConfig,
    GmailAccountConfig,
    MeConfig,
    Neo4jConfig,
    QdrantConfig,
)
from worker.doctor import (
    check_calendar_accounts,
    check_gmail_accounts,
    check_me,
)
from worker.parsers.calendar import GoogleCalendarParser
from worker.parsers.gmail import GmailParser
from worker.pipeline.chunker import chunk_text
from worker.pipeline.writer import VECTOR_SIZE, WriteUnit, Writer
from worker.queue import PersistentQueue
from worker.sources.calendar import GoogleCalendarSource
from worker.sources.gmail import GmailSource


# Connection details mirror tests/test_slack_e2e.py so a single
# docker-compose stack serves all e2e suites.
_NEO4J_URI = "bolt://localhost:7687"
_NEO4J_USER = "neo4j"
_NEO4J_PASSWORD = "testpassword"
_QDRANT_HOST = "localhost"
_QDRANT_PORT = 6333
_TEST_COLLECTION = "fieldnotes_multi_account_e2e_test"


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


@pytest.fixture
def _cleanup_neo4j(neo4j_driver: Driver) -> Generator[None, None, None]:
    # Pre-clean too: tests share the same DB and a stray node from a
    # previous run would skew the per-account assertions.
    with neo4j_driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    yield
    with neo4j_driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")


@pytest.fixture
def _cleanup_qdrant(qdrant_client_: QdrantClient) -> Generator[None, None, None]:
    yield
    try:
        qdrant_client_.delete_collection(_TEST_COLLECTION)
    except Exception:
        pass


# E2E tests that hit Neo4j + Qdrant must depend on these via usefixtures
# so the doctor test (which does not touch the services) is not gated by
# fixture availability.
_uses_db_cleanup = pytest.mark.usefixtures("_cleanup_neo4j", "_cleanup_qdrant")


@pytest.fixture
def queue_db(tmp_path: Path) -> Generator[PersistentQueue, None, None]:
    """Real ``PersistentQueue`` backed by a tmp ``queue.db`` so cursor
    asserts run against the production schema (cursors table)."""
    q = PersistentQueue(db_path=tmp_path / "queue.db")
    yield q
    q.close()


# ---------------------------------------------------------------------------
# Drive sources / write events
# ---------------------------------------------------------------------------


def _stable_vector(text: str) -> list[float]:
    """Deterministic, low-magnitude vector keyed off chunk text.

    Matches ``test_slack_e2e._stable_vector`` so the embedder can be
    bypassed without losing deterministic Qdrant round-tripping."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [(digest[i % len(digest)] / 255.0) for i in range(VECTOR_SIZE)]


def _ingest_gmail(
    account: str,
    messages_api: MagicMock,
    queue: PersistentQueue,
) -> None:
    """Drive ``GmailSource._backfill`` for *account*."""
    src = GmailSource()
    src.configure(
        {
            "account": account,
            "client_secrets_path": "/dev/null",
            "max_initial_threads": 50,
        }
    )
    # _backfill consumes messages_api directly — no live OAuth required.
    asyncio.run(src._backfill(messages_api, queue, is_initial=True))


def _all_pending_events(queue: PersistentQueue) -> list[dict[str, Any]]:
    """Read all pending payloads from the queue without consuming them."""
    with sqlite3.connect(str(queue._db_path)) as conn:  # type: ignore[attr-defined]
        rows = conn.execute(
            "SELECT payload FROM queue WHERE status IN ('pending','processing')"
        ).fetchall()
    return [json.loads(r[0]) for r in rows]


def _ingest_calendar(
    account: str,
    calendar_id: str,
    service: MagicMock,
    queue: PersistentQueue,
) -> None:
    """Drive ``GoogleCalendarSource._poll_calendar`` for *account*."""
    src = GoogleCalendarSource()
    src.configure(
        {
            "account": account,
            "client_secrets_path": "/dev/null",
            "calendar_ids": [calendar_id],
        }
    )
    asyncio.run(src._poll_calendar(service, queue, calendar_id, None, {}))


def _write_event(writer: Writer, event: dict[str, Any]) -> None:
    """Parse an IngestEvent → write to Neo4j + Qdrant.

    Bypasses the LLM extractor and embedder; uses ``_stable_vector`` so
    Qdrant round-trips deterministically.
    """
    if event["source_type"] == "gmail":
        parser: Any = GmailParser()
    elif event["source_type"] == "google_calendar":
        parser = GoogleCalendarParser()
    else:
        pytest.fail(f"unexpected source_type {event['source_type']!r}")

    for doc in parser.parse(event):
        chunks = chunk_text(doc.text) if doc.text else []
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
    fake_gmail_clients: dict[str, MagicMock],
    fake_calendar_services: dict[str, MagicMock],
) -> None:
    """Run the multi-account ingest end-to-end into Neo4j + Qdrant."""
    for account in ("personal", "work"):
        _ingest_gmail(account, fake_gmail_clients[account], queue)
    for account in ("personal", "shared"):
        _ingest_calendar(account, "primary", fake_calendar_services[account], queue)
    for event in _all_pending_events(queue):
        _write_event(writer, event)
    # Cross-source Person resolution + self-identity collapse.
    writer.reconcile_persons()
    writer.reconcile_persons_by_name()
    writer.close_same_as_transitive()
    writer.reconcile_self_person(MeConfig(emails=["me@personal.com", "me@work.com"]))


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def _count_label(driver: Driver, label: str) -> int:
    with driver.session() as session:
        result = session.run(f"MATCH (n:`{label}`) RETURN count(n) AS cnt")
        return result.single()["cnt"]


def _persons_by_email(driver: Driver) -> dict[str, list[dict[str, Any]]]:
    """Map email → list of Person property dicts (1 entry per node)."""
    with driver.session() as session:
        result = session.run(
            "MATCH (p:Person) WHERE p.email IS NOT NULL "
            "RETURN p.email AS email, p.is_self AS is_self, "
            "       p.display_name AS display_name, "
            "       elementId(p) AS eid"
        )
        out: dict[str, list[dict[str, Any]]] = {}
        for r in result:
            out.setdefault(r["email"], []).append(dict(r))
    return out


def _accounts_for_person(
    driver: Driver, email: str, predicates: tuple[str, ...]
) -> set[str]:
    """Collect distinct ``account`` values stamped on edges touching the
    Person matching *email*, restricted to the named predicates.
    """
    accounts: set[str] = set()
    with driver.session() as session:
        for pred in predicates:
            result = session.run(
                f"MATCH (p:Person {{email: $email}})<-[r:{pred}]-(n) "
                f"RETURN r.account AS account",
                email=email,
            )
            for r in result:
                acc = r["account"]
                if isinstance(acc, str) and acc:
                    accounts.add(acc)
            result = session.run(
                f"MATCH (p:Person {{email: $email}})-[r:{pred}]->(n) "
                f"RETURN r.account AS account",
                email=email,
            )
            for r in result:
                acc = r["account"]
                if isinstance(acc, str) and acc:
                    accounts.add(acc)
    return accounts


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@_skip_services
@_uses_db_cleanup
class TestMultiAccountEndToEnd:
    def test_per_account_doc_uris_are_distinct(
        self,
        writer: Writer,
        neo4j_driver: Driver,
        queue_db: PersistentQueue,
        fake_gmail_clients: dict[str, MagicMock],
        fake_calendar_services: dict[str, MagicMock],
    ) -> None:
        """Two Gmail accounts with the same thread_id and two Calendar
        accounts with the same calendar_id ('primary') produce distinct
        nodes — namespacing by account in the doc URI prevents collision.
        """
        _ingest_and_write_all(
            writer, queue_db, fake_gmail_clients, fake_calendar_services
        )

        # Colliding gmail thread_id 't-shared': one Thread node per account.
        with neo4j_driver.session() as session:
            shared_threads = sorted(
                r["sid"]
                for r in session.run(
                    "MATCH (t:Thread {thread_id: 't-shared'}) RETURN t.source_id AS sid"
                )
            )
        assert shared_threads == [
            "gmail://personal/thread/t-shared",
            "gmail://work/thread/t-shared",
        ]

        # Colliding calendar_id 'primary': two distinct CalendarEvent nodes.
        with neo4j_driver.session() as session:
            primary_event_sids = sorted(
                r["sid"]
                for r in session.run(
                    "MATCH (c:CalendarEvent) "
                    "WHERE c.calendar_id ENDS WITH '/primary' "
                    "RETURN c.source_id AS sid"
                )
            )
        assert primary_event_sids == [
            "google-calendar://personal/event/evt-personal-1",
            "google-calendar://shared/event/evt-shared-1",
        ]

    def test_alice_is_one_person_across_accounts(
        self,
        writer: Writer,
        neo4j_driver: Driver,
        queue_db: PersistentQueue,
        fake_gmail_clients: dict[str, MagicMock],
        fake_calendar_services: dict[str, MagicMock],
    ) -> None:
        """``alice@example.com`` appears in both Gmail accounts and one
        Calendar account; she must collapse to a single Person node."""
        _ingest_and_write_all(
            writer, queue_db, fake_gmail_clients, fake_calendar_services
        )

        persons = _persons_by_email(neo4j_driver)
        assert "alice@example.com" in persons
        assert len(persons["alice@example.com"]) == 1, (
            f"expected 1 alice Person, got "
            f"{len(persons['alice@example.com'])}: "
            f"{persons['alice@example.com']!r}"
        )

        # Alice has edges from both Gmail accounts (and the personal calendar).
        alice_accounts = _accounts_for_person(
            neo4j_driver, "alice@example.com", ("SENT", "TO", "ATTENDED_BY")
        )
        assert "personal" in alice_accounts
        assert "work" in alice_accounts

    def test_bob_only_in_personal_account(
        self,
        writer: Writer,
        neo4j_driver: Driver,
        queue_db: PersistentQueue,
        fake_gmail_clients: dict[str, MagicMock],
        fake_calendar_services: dict[str, MagicMock],
    ) -> None:
        """Bob appears only in the personal Gmail thread — his Person
        must exist exactly once and be wired only to personal-account docs.
        """
        _ingest_and_write_all(
            writer, queue_db, fake_gmail_clients, fake_calendar_services
        )

        persons = _persons_by_email(neo4j_driver)
        assert len(persons.get("bob@example.com", [])) == 1

        bob_accounts = _accounts_for_person(
            neo4j_driver, "bob@example.com", ("SENT", "TO", "ATTENDED_BY")
        )
        assert bob_accounts == {"personal"}

    def test_self_person_survivor_with_same_as(
        self,
        writer: Writer,
        neo4j_driver: Driver,
        queue_db: PersistentQueue,
        fake_gmail_clients: dict[str, MagicMock],
        fake_calendar_services: dict[str, MagicMock],
    ) -> None:
        """[me] reconciliation flags both self-emails ``is_self = true``
        and joins them with a ``SAME_AS`` edge tagged ``self_identity``.
        """
        _ingest_and_write_all(
            writer, queue_db, fake_gmail_clients, fake_calendar_services
        )

        persons = _persons_by_email(neo4j_driver)
        for email in ("me@personal.com", "me@work.com"):
            entries = persons.get(email, [])
            assert len(entries) == 1, (
                f"expected exactly 1 Person for {email}, got "
                f"{len(entries)}: {entries!r}"
            )
            assert entries[0]["is_self"] is True

        with neo4j_driver.session() as session:
            edges = list(
                session.run(
                    "MATCH (a:Person {email: 'me@personal.com'}) "
                    "MATCH (b:Person {email: 'me@work.com'}) "
                    "MATCH (a)-[r:SAME_AS]-(b) "
                    "RETURN r.match_type AS mt, r.confidence AS conf, "
                    "       r.cross_source AS xs"
                )
            )
        assert edges, "expected SAME_AS edge between the two self Persons"
        assert any(e["mt"] == "self_identity" and e["xs"] is True for e in edges), (
            f"expected self_identity SAME_AS edge, got {edges!r}"
        )

    def test_cursor_state_in_queue_db(
        self,
        writer: Writer,
        queue_db: PersistentQueue,
        fake_gmail_clients: dict[str, MagicMock],
        fake_calendar_services: dict[str, MagicMock],
    ) -> None:
        """Per-account cursors must live in ``queue.db.cursors`` under the
        keys ``gmail:<account>`` / ``calendar:<account>``.  No JSON cursor
        files should be created during the test (asserted by tmp_path
        cleanliness — the queue creates its own DB and nothing else).
        """
        _ingest_and_write_all(
            writer, queue_db, fake_gmail_clients, fake_calendar_services
        )

        expected = {
            "gmail:personal",
            "gmail:work",
            "calendar:personal",
            "calendar:shared",
        }
        for key in expected:
            raw = queue_db.load_cursor(key)
            assert raw is not None, f"missing cursor key {key!r}"
            # Each cursor stores valid JSON.
            try:
                json.loads(raw)
            except json.JSONDecodeError as exc:
                pytest.fail(f"cursor {key!r} is not valid JSON: {exc!s}")

        # The queue dir should not contain any *_cursor*.json files —
        # those are deprecated artefacts that the multi-account migration
        # removes.
        queue_dir = queue_db._db_path.parent  # type: ignore[attr-defined]
        leaked = sorted(
            p.name
            for p in queue_dir.iterdir()
            if p.suffix == ".json" and "cursor" in p.name
        )
        assert leaked == [], f"unexpected JSON cursor files in queue dir: {leaked!r}"

    def test_idempotent_second_run_is_a_no_op(
        self,
        writer: Writer,
        neo4j_driver: Driver,
        qdrant_client_: QdrantClient,
        queue_db: PersistentQueue,
        fake_gmail_clients: dict[str, MagicMock],
        fake_calendar_services: dict[str, MagicMock],
    ) -> None:
        """Re-running the multi-account ingest must not create new
        Documents, Person nodes, or duplicate edges."""
        _ingest_and_write_all(
            writer, queue_db, fake_gmail_clients, fake_calendar_services
        )

        with neo4j_driver.session() as session:
            doc_count = session.run(
                "MATCH (n) WHERE n:Email OR n:Thread OR n:CalendarEvent "
                "RETURN count(n) AS n"
            ).single()["n"]
            person_count = session.run(
                "MATCH (p:Person) RETURN count(p) AS n"
            ).single()["n"]
            edge_count = session.run(
                "MATCH ()-[r]->() WHERE type(r) IN "
                "['SENT','TO','PART_OF','ATTENDED_BY','ORGANIZED_BY','SAME_AS'] "
                "RETURN count(r) AS n"
            ).single()["n"]

        # Re-run the entire multi-account ingest with fresh source
        # instances + a new queue + fresh fake clients (the previous
        # MagicMocks have memoised side_effects on iterators in some
        # cases; rebuild them).
        from conftest import (  # type: ignore[import-not-found]
            make_fake_calendar_service,
            make_fake_gmail_messages_api,
        )

        gmail_clients_2 = {
            "personal": make_fake_gmail_messages_api("personal"),
            "work": make_fake_gmail_messages_api("work"),
        }
        cal_services_2 = {
            "personal": make_fake_calendar_service("personal"),
            "shared": make_fake_calendar_service("shared"),
        }
        queue2 = PersistentQueue(
            db_path=queue_db._db_path.parent / "queue2.db"  # type: ignore[attr-defined]
        )
        try:
            _ingest_and_write_all(writer, queue2, gmail_clients_2, cal_services_2)
        finally:
            queue2.close()

        with neo4j_driver.session() as session:
            doc_count_after = session.run(
                "MATCH (n) WHERE n:Email OR n:Thread OR n:CalendarEvent "
                "RETURN count(n) AS n"
            ).single()["n"]
            person_count_after = session.run(
                "MATCH (p:Person) RETURN count(p) AS n"
            ).single()["n"]
            edge_count_after = session.run(
                "MATCH ()-[r]->() WHERE type(r) IN "
                "['SENT','TO','PART_OF','ATTENDED_BY','ORGANIZED_BY','SAME_AS'] "
                "RETURN count(r) AS n"
            ).single()["n"]

        assert doc_count_after == doc_count, "re-ingest created new Documents"
        assert person_count_after == person_count, "re-ingest duplicated Person nodes"
        assert edge_count_after == edge_count, "re-ingest duplicated edges"


class TestDoctorMultiAccount:
    """Doctor reporting does not require live Neo4j / Qdrant services."""

    def test_doctor_reports_per_account_status(
        self,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        """``doctor`` prints ``Gmail [<account>]: OK``,
        ``Calendar [<account>]: OK``, and ``[me]: 2 emails`` when
        every per-account auth probe and the [me] block are healthy.
        """
        # Create a stub client_secrets file so the doctor's existence
        # check passes without us having to wire OAuth.
        secrets = tmp_path / "credentials.json"
        secrets.write_text("{}")

        gmail_accounts = {
            "personal": GmailAccountConfig(
                name="personal", client_secrets_path=str(secrets)
            ),
            "work": GmailAccountConfig(name="work", client_secrets_path=str(secrets)),
        }
        cal_accounts = {
            "personal": CalendarAccountConfig(
                name="personal", client_secrets_path=str(secrets)
            ),
            "shared": CalendarAccountConfig(
                name="shared", client_secrets_path=str(secrets)
            ),
        }
        me = MeConfig(emails=["me@personal.com", "me@work.com"])

        # Patch the shared OAuth probe so each per-account check returns
        # OK without needing a real token file on disk.
        with patch(
            "worker.doctor._check_google_auth",
            side_effect=lambda label, account, *a, **kw: (
                print(f"  ✓ {label} [{account}]: OK") or 0
            ),
        ):
            errors = (
                check_gmail_accounts(gmail_accounts)
                + check_calendar_accounts(cal_accounts)
                + check_me(me)
            )
        assert errors == 0
        captured = capsys.readouterr().out
        for expected in (
            "Gmail [personal]: OK",
            "Gmail [work]: OK",
            "Calendar [personal]: OK",
            "Calendar [shared]: OK",
            "[me]: 2 email(s)",
        ):
            assert expected in captured, (
                f"missing {expected!r} in doctor output:\n{captured}"
            )


@_skip_services
class TestMigrationScenario:
    """Migrating old-shape artefacts requires a live Neo4j."""

    @pytest.fixture(autouse=True)
    def _clean_neo4j(self, neo4j_driver: Driver) -> Generator[None, None, None]:
        with neo4j_driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        yield
        with neo4j_driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def test_migration_retags_old_shape_under_account(
        self,
        neo4j_driver: Driver,
        tmp_path: Path,
    ) -> None:
        """Pre-populate Neo4j and a queue.db with old-shape (no account
        segment) artefacts.  Run the migration mutators against
        ``account='default'``.  Old prefixes are gone, new prefixes
        are present, and the ``cursors`` table key is renamed.
        """
        # 1. Seed Neo4j with old-shape Documents + a Chunk + a fallback
        #    Person whose source_id is itself an old-shape doc URI.
        with neo4j_driver.session() as session:
            session.run(
                "CREATE (:Email {source_id: 'gmail://message/old-1', "
                "                subject: 'Pre-migration'})"
            )
            session.run(
                "CREATE (:Thread {source_id: 'gmail://thread/old-1', "
                "                 thread_id: 'old-1'})"
            )
            session.run(
                "CREATE (:CalendarEvent {source_id: "
                "  'google-calendar://event/old-evt-1', summary: 'old'})"
            )
            session.run(
                "CREATE (:Chunk {id: 'gmail://message/old-1:chunk:0', "
                "                text: 'hi'})"
            )
            session.run(
                "CREATE (:Person {source_id: "
                "  'google-calendar://event/old-evt-1/attendee/0', "
                "                 name: 'No-Email Bob'})"
            )

        # 2. Seed queue.db with old-shape rows and old cursor keys.
        queue_db_path = tmp_path / "queue.db"
        with sqlite3.connect(str(queue_db_path)) as conn:
            conn.executescript(
                """
                CREATE TABLE queue (
                    id TEXT PRIMARY KEY,
                    source_type TEXT,
                    source_id TEXT,
                    operation TEXT,
                    status TEXT,
                    payload TEXT,
                    blob_path TEXT,
                    enqueued_at TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    attempts INT DEFAULT 0,
                    error TEXT
                );
                CREATE TABLE cursors (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT
                );
                """
            )
            conn.execute(
                "INSERT INTO queue "
                "(id, source_type, source_id, operation, status, payload) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    "q-1",
                    "gmail",
                    "gmail://message/old-1",
                    "created",
                    "pending",
                    json.dumps(
                        {
                            "id": "q-1",
                            "source_id": "gmail://message/old-1",
                            "source_type": "gmail",
                            "operation": "created",
                        }
                    ),
                ),
            )
            conn.execute(
                "INSERT INTO cursors (key, value, updated_at) VALUES "
                "('gmail', '{\"history_id\": \"42\"}', '2026-04-26')"
            )
            conn.commit()

        # 3. Run the queue + Neo4j mutators against account='default'.
        q_result = migrate_cli.migrate_queue(queue_db_path, "default")
        assert q_result["rows"] >= 1
        assert q_result["cursors"] == 1

        with neo4j_driver.session() as session:
            n_result = migrate_cli.migrate_neo4j(session, "default")
        assert n_result["documents"] >= 3
        assert n_result["chunks"] >= 1
        assert n_result["persons"] >= 1

        # 4. queue.db: cursors row renamed; no rows still on old prefix.
        with sqlite3.connect(str(queue_db_path)) as conn:
            new_cursors = sorted(
                r[0] for r in conn.execute("SELECT key FROM cursors").fetchall()
            )
            old_rows = conn.execute(
                "SELECT count(*) FROM queue WHERE source_id LIKE 'gmail://message/%' "
                "AND source_id NOT LIKE 'gmail://default/%'"
            ).fetchone()[0]
        assert new_cursors == ["gmail:default"]
        assert old_rows == 0

        # 5. Neo4j: every previously-old node now has a new-shape source_id.
        with neo4j_driver.session() as session:
            stuck = session.run(
                "MATCH (n) WHERE n.source_id IS NOT NULL "
                "AND (n.source_id STARTS WITH 'gmail://message/' "
                "     OR n.source_id STARTS WITH 'gmail://thread/' "
                "     OR n.source_id STARTS WITH 'google-calendar://event/' "
                "     OR n.source_id STARTS WITH 'google-calendar://series/') "
                "RETURN count(n) AS c"
            ).single()["c"]
        assert stuck == 0, "old-shape source_id values remained after migrate"

        with neo4j_driver.session() as session:
            new_email = session.run(
                "MATCH (n:Email {source_id: 'gmail://default/message/old-1'}) "
                "RETURN n.source_id AS sid"
            ).single()
            new_chunk = session.run(
                "MATCH (c:Chunk {id: 'gmail://default/message/old-1:chunk:0'}) "
                "RETURN c.id AS id"
            ).single()
            new_person = session.run(
                "MATCH (p:Person {source_id: "
                "  'google-calendar://default/event/old-evt-1/attendee/0'}) "
                "RETURN p.source_id AS sid"
            ).single()
        assert new_email is not None
        assert new_chunk is not None
        assert new_person is not None
