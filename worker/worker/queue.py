"""SQLite-backed persistent queue for the ingestion pipeline.

Replaces the in-memory ``asyncio.Queue`` with a durable queue that survives
restarts.  Cursor state is co-located in the same database so that enqueue
and cursor-update can execute in a single transaction — eliminating the need
for ``_on_indexed`` callbacks and sidecar files.

The database uses WAL mode for safe concurrent access from multiple threads
(e.g. watchdog observer threads and the main event loop).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS queue (
    id          TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,
    source_id   TEXT NOT NULL,
    operation   TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    payload     TEXT NOT NULL,
    blob_path   TEXT,
    enqueued_at TEXT NOT NULL,
    started_at  TEXT,
    attempts    INTEGER NOT NULL DEFAULT 0,
    error       TEXT
);

CREATE INDEX IF NOT EXISTS idx_queue_status ON queue(status);
CREATE INDEX IF NOT EXISTS idx_queue_source_id ON queue(source_id, status);
CREATE INDEX IF NOT EXISTS idx_queue_enqueued ON queue(enqueued_at);

CREATE TABLE IF NOT EXISTS cursors (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

DEFAULT_MAX_RETRIES = 3


@dataclass(frozen=True)
class CursorUpdate:
    """Describes a cursor write to execute atomically with a queue insert."""

    key: str
    value: str  # JSON-serialized cursor data


class PersistentQueue:
    """SQLite-backed durable queue with co-located cursor storage.

    Thread-safe.  Uses WAL mode so readers never block writers.
    """

    def __init__(
        self,
        db_path: Path,
        blob_dir: Path | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self._db_path = db_path
        self._blob_dir = blob_dir or db_path.parent / "queue_blobs"
        self._max_retries = max_retries
        self._lock = threading.Lock()

        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(db_path),
            check_same_thread=False,
            isolation_level=None,  # autocommit; we manage transactions explicitly
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------
    # Enqueue
    # ------------------------------------------------------------------

    def enqueue(
        self,
        event: dict[str, Any],
        cursor_key: str | None = None,
        cursor_value: str | None = None,
    ) -> str:
        """Insert an event into the queue.

        If *cursor_key* and *cursor_value* are provided, the cursor table
        is updated in the **same transaction** so the two writes are atomic.

        Skips the insert (no-op) if the same ``source_id`` is already
        pending or processing — this prevents double-enqueue even when
        two threads race past an external ``is_enqueued()`` check.

        Returns the queue row ID (or the existing row's ID if skipped).
        """
        event_id = event.get("id") or str(uuid.uuid4())
        source_type = event.get("source_type", "")
        source_id = event.get("source_id", "")
        operation = event.get("operation", "")
        enqueued_at = event.get("enqueued_at") or _now_iso()

        # Prepare payload (strip binary data and legacy callbacks).
        payload_event = dict(event)
        raw_bytes = payload_event.pop("raw_bytes", None)
        payload_event.pop("_on_indexed", None)
        payload = json.dumps(payload_event, default=str)

        blob_path: str | None = None

        with self._lock:
            # Atomic source_id dedup check inside the lock — prevents the
            # race where two threads both pass is_enqueued() then both insert.
            existing = self._conn.execute(
                "SELECT id FROM queue WHERE source_id = ? "
                "AND status IN ('pending', 'processing') LIMIT 1",
                (source_id,),
            ).fetchone()
            if existing:
                # Already enqueued — still apply cursor update if requested.
                if cursor_key is not None and cursor_value is not None:
                    self._conn.execute(
                        "INSERT OR REPLACE INTO cursors (key, value, updated_at) "
                        "VALUES (?, ?, ?)",
                        (cursor_key, cursor_value, _now_iso()),
                    )
                return existing[0]

            self._conn.execute("BEGIN IMMEDIATE")
            try:
                # Write blob AFTER acquiring the transaction lock so a crash
                # between blob write and INSERT leaves the DB clean (the
                # orphaned blob is cleaned up on rollback).
                if raw_bytes is not None:
                    self._blob_dir.mkdir(parents=True, exist_ok=True)
                    blob_file = self._blob_dir / event_id
                    blob_file.write_bytes(raw_bytes)
                    blob_path = str(blob_file)

                self._conn.execute(
                    "INSERT INTO queue "
                    "(id, source_type, source_id, operation, status, payload, "
                    " blob_path, enqueued_at) "
                    "VALUES (?, ?, ?, ?, 'pending', ?, ?, ?)",
                    (event_id, source_type, source_id, operation,
                     payload, blob_path, enqueued_at),
                )
                if cursor_key is not None and cursor_value is not None:
                    self._conn.execute(
                        "INSERT OR REPLACE INTO cursors (key, value, updated_at) "
                        "VALUES (?, ?, ?)",
                        (cursor_key, cursor_value, _now_iso()),
                    )
                self._conn.execute("COMMIT")
            except BaseException:
                try:
                    self._conn.execute("ROLLBACK")
                except Exception:
                    pass
                # Clean up blob on failure.
                if blob_path is not None:
                    with contextlib.suppress(OSError):
                        os.unlink(blob_path)
                raise

        return event_id

    # ------------------------------------------------------------------
    # Dedup
    # ------------------------------------------------------------------

    def is_enqueued(self, source_id: str) -> bool:
        """Return True if *source_id* is already pending or processing."""
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM queue WHERE source_id = ? "
                "AND status IN ('pending', 'processing') LIMIT 1",
                (source_id,),
            ).fetchone()
        return row is not None

    # ------------------------------------------------------------------
    # Claim / Complete / Fail
    # ------------------------------------------------------------------

    def claim(self) -> dict[str, Any] | None:
        """Atomically claim the oldest pending item.

        Returns the event dict with an extra ``_queue_id`` key, or *None*
        if the queue is empty.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT id, payload, blob_path FROM queue "
                "WHERE status = 'pending' ORDER BY enqueued_at ASC LIMIT 1",
            ).fetchone()
            if row is None:
                return None
            queue_id, payload_json, blob_path = row
            self._conn.execute(
                "UPDATE queue SET status = 'processing', "
                "started_at = ?, attempts = attempts + 1 WHERE id = ?",
                (_now_iso(), queue_id),
            )

        event = json.loads(payload_json)
        event["_queue_id"] = queue_id

        # Rehydrate binary data if present.
        if blob_path is not None:
            blob_file = Path(blob_path)
            if blob_file.exists():
                event["raw_bytes"] = blob_file.read_bytes()

        return event

    def complete(self, queue_id: str) -> None:
        """Remove a successfully processed item from the queue."""
        with self._lock:
            row = self._conn.execute(
                "SELECT blob_path FROM queue WHERE id = ?", (queue_id,)
            ).fetchone()
            self._conn.execute("DELETE FROM queue WHERE id = ?", (queue_id,))

        # Clean up blob file.
        if row and row[0]:
            with contextlib.suppress(OSError):
                os.unlink(row[0])

    def fail(self, queue_id: str, error: str | None = None) -> None:
        """Record a processing failure.

        If attempts < max_retries, reset status to ``'pending'`` for retry.
        Otherwise mark as ``'failed'``.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT attempts FROM queue WHERE id = ?", (queue_id,)
            ).fetchone()
            if row is None:
                return
            attempts = row[0]
            if attempts < self._max_retries:
                self._conn.execute(
                    "UPDATE queue SET status = 'pending', error = ? WHERE id = ?",
                    (error, queue_id),
                )
            else:
                self._conn.execute(
                    "UPDATE queue SET status = 'failed', error = ? WHERE id = ?",
                    (error, queue_id),
                )

    # ------------------------------------------------------------------
    # Cursors
    # ------------------------------------------------------------------

    def load_cursor(self, key: str) -> str | None:
        """Load a cursor value by key. Returns the JSON string or None."""
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM cursors WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row else None

    def save_cursor(self, key: str, value: str) -> None:
        """Save a cursor value outside of an enqueue transaction."""
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO cursors (key, value, updated_at) "
                "VALUES (?, ?, ?)",
                (key, value, _now_iso()),
            )

    # ------------------------------------------------------------------
    # Recovery
    # ------------------------------------------------------------------

    def recover(self) -> int:
        """Reset any ``'processing'`` items back to ``'pending'``.

        Call this at startup to recover items interrupted by a crash.
        Returns the number of recovered items.
        """
        with self._lock:
            cur = self._conn.execute(
                "UPDATE queue SET status = 'pending' WHERE status = 'processing'"
            )
        count = cur.rowcount
        if count:
            logger.info("Recovered %d interrupted queue item(s)", count)
        return count

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def depth(self) -> int:
        """Total number of pending + processing items."""
        with self._lock:
            row = self._conn.execute(
                "SELECT count(*) FROM queue WHERE status IN ('pending', 'processing')"
            ).fetchone()
        return row[0] if row else 0

    def stats(self) -> dict[str, dict[str, int]]:
        """Return counts grouped by (source_type, status).

        Returns ``{source_type: {status: count, ...}, ...}``.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT source_type, status, count(*) "
                "FROM queue GROUP BY source_type, status"
            ).fetchall()
        result: dict[str, dict[str, int]] = {}
        for source_type, status, count in rows:
            result.setdefault(source_type, {})[status] = count
        return result

    def summary(self) -> dict[str, int]:
        """Return total counts by status (pending, processing, failed)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT status, count(*) FROM queue GROUP BY status"
            ).fetchall()
        return {status: count for status, count in rows}

    def list_items(
        self,
        *,
        status: str | None = None,
        source_type: str | None = None,
        limit: int = 50,
        offset: int = 0,
        order: str = "asc",
    ) -> list[dict[str, Any]]:
        """Return queue items for CLI inspection."""
        clauses = []
        params: list[Any] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if source_type is not None:
            clauses.append("source_type = ?")
            params.append(source_type)

        where = ""
        if clauses:
            where = "WHERE " + " AND ".join(clauses)

        direction = "ASC" if order == "asc" else "DESC"
        sql = (
            f"SELECT id, source_type, source_id, operation, status, "
            f"enqueued_at, started_at, attempts, error "
            f"FROM queue {where} ORDER BY enqueued_at {direction} "
            f"LIMIT ? OFFSET ?"
        )
        params.extend([limit, offset])

        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()

        return [
            {
                "id": r[0],
                "source_type": r[1],
                "source_id": r[2],
                "operation": r[3],
                "status": r[4],
                "enqueued_at": r[5],
                "started_at": r[6],
                "attempts": r[7],
                "error": r[8],
            }
            for r in rows
        ]

    def retry_failed(self) -> int:
        """Reset all ``'failed'`` items to ``'pending'``."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE queue SET status = 'pending', attempts = 0, error = NULL "
                "WHERE status = 'failed'"
            )
        return cur.rowcount

    def purge(self, status: str = "failed") -> int:
        """Delete all items with the given status. Returns count deleted."""
        with self._lock:
            # Clean up blobs first.
            rows = self._conn.execute(
                "SELECT blob_path FROM queue WHERE status = ? AND blob_path IS NOT NULL",
                (status,),
            ).fetchall()
            for (blob_path,) in rows:
                with contextlib.suppress(OSError):
                    os.unlink(blob_path)
            cur = self._conn.execute(
                "DELETE FROM queue WHERE status = ?", (status,)
            )
        return cur.rowcount

    # ------------------------------------------------------------------
    # Migration
    # ------------------------------------------------------------------

    def migrate_cursor_files(self, data_dir: Path, state_dir: Path | None = None) -> int:
        """Import old per-source cursor JSON files into the cursors table.

        Renames imported files to ``*.migrated``.  Returns the number of
        files imported.
        """
        state_dir = state_dir or data_dir

        _CURSOR_MAP: dict[str, Path] = {
            "files": data_dir / "file_cursor.json",
            "obsidian": data_dir / "obsidian_cursor.json",
            "gmail": data_dir / "gmail_cursor.json",
            "calendar": data_dir / "calendar_cursor.json",
            "repositories": data_dir / "repo_cursor.json",
            "omnifocus": state_dir / "omnifocus.json",
        }

        imported = 0
        for key, path in _CURSOR_MAP.items():
            if not path.exists():
                continue
            # Skip if already imported.
            if self.load_cursor(key) is not None:
                continue
            try:
                raw = path.read_text()
                # Validate it's valid JSON.
                json.loads(raw)
                self.save_cursor(key, raw)
                path.rename(path.with_suffix(path.suffix + ".migrated"))
                logger.info("Migrated cursor file %s → cursors[%s]", path, key)
                imported += 1
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to migrate cursor %s: %s", path, exc)

        # Clean up sidecar files.
        for pattern in ("*_processed.json", "repo_*_processed.json"):
            for sidecar in data_dir.glob(pattern):
                with contextlib.suppress(OSError):
                    sidecar.rename(sidecar.with_suffix(sidecar.suffix + ".migrated"))

        return imported

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        with contextlib.suppress(Exception):
            self._conn.close()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
