"""Shared watchdog handler base and utilities for source watchers.

Provides the common event-handling logic used by both the plain filesystem
source and the Obsidian vault source, eliminating near-identical code in
each handler subclass.
"""

from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from watchdog.events import (
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileSystemEvent,
    FileSystemEventHandler,
)

from worker.metrics import (
    INITIAL_SCAN_DEDUP_DROPPED,
    SOURCE_WATCHER_EVENTS,
    WATCHER_LAST_EVENT_TIMESTAMP,
)

from .cursor import FileEntry

logger = logging.getLogger(__name__)

_SHA256_CHUNK_SIZE = 64 * 1024  # 64 KiB read chunks for streaming hash

# Default 100 MiB — files above this are logged and skipped to prevent OOM.
DEFAULT_MAX_FILE_SIZE = 100 * 1024 * 1024

# Image extensions that pass through include_extensions filters so standalone
# images reach the vision pipeline even when the user restricts file types.
IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".svg",
        ".webp",
        ".bmp",
        ".tiff",
        ".tif",
        ".heic",
        ".heif",
    }
)


def streaming_sha256(path: Path, max_size: int) -> tuple[str, int] | None:
    """Compute SHA-256 by streaming *path* in chunks.

    Returns ``(hex_digest, size_bytes)`` or ``None`` if the file exceeds
    *max_size*.
    """
    h = hashlib.sha256()
    total = 0
    with path.open("rb") as f:
        while True:
            chunk = f.read(_SHA256_CHUNK_SIZE)
            if not chunk:
                break
            total += len(chunk)
            if total > max_size:
                return None
            h.update(chunk)
    return h.hexdigest(), total


def _read_file_atomic(path: Path, max_size: int) -> tuple[bytes, int] | None:
    """Read file into memory in a single pass, enforcing *max_size*.

    Returns ``(data, mtime_ns)`` or ``None`` if the file exceeds *max_size*.
    The caller gets a consistent snapshot: hash and text are derived from
    the same bytes, eliminating the TOCTOU race of separate reads.
    """
    fd = path.open("rb")
    try:
        data = fd.read(max_size + 1)
        # fstat on the open fd instead of path.stat() after close —
        # this eliminates the TOCTOU window where the file could be
        # deleted or replaced between read and stat.
        mtime_ns = os.fstat(fd.fileno()).st_mtime_ns
    finally:
        fd.close()
    if len(data) > max_size:
        return None
    return data, mtime_ns


def _sha256_of(data: bytes) -> str:
    """Return hex SHA-256 digest of an in-memory buffer."""
    return hashlib.sha256(data).hexdigest()


def guess_mime(path: str) -> str:
    """Return a basic MIME type based on file extension."""
    ext = Path(path).suffix.lower()
    mime_map = {
        ".md": "text/markdown",
        ".txt": "text/plain",
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".svg": "image/svg+xml",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
        ".heic": "image/heic",
        ".heif": "image/heif",
        ".json": "application/json",
        ".yaml": "text/yaml",
        ".yml": "text/yaml",
        ".toml": "text/toml",
        ".html": "text/html",
        ".csv": "text/csv",
        ".canvas": "application/json",
        ".pages": "application/x-iwork-pages",
        ".key": "application/x-iwork-keynote",
    }
    return mime_map.get(ext, "application/octet-stream")


class BaseHandler(FileSystemEventHandler):
    """Watchdog handler with shared filtering, hashing, and dispatch logic.

    Subclasses must set ``_source_type`` and may override ``_extra_skip``
    and ``_extra_meta`` to customise behaviour.
    """

    _source_type: str = ""

    def __init__(
        self,
        queue: asyncio.Queue[dict[str, Any]],
        loop: asyncio.AbstractEventLoop,
        include_extensions: set[str] | None,
        exclude_patterns: list[str],
        max_file_size: int = DEFAULT_MAX_FILE_SIZE,
    ) -> None:
        self._queue = queue
        self._loop = loop
        self._include_extensions = include_extensions
        self._exclude_patterns = exclude_patterns
        self._max_file_size = max_file_size
        self._cursor: dict[str, FileEntry] = {}
        self._cursor_lock = threading.Lock()
        # Dedup window state — guarded by _dedup_lock since watchdog
        # dispatches from its observer thread while the timer fires
        # on the event loop thread.
        self._dedup_set: set[tuple[str, str]] = set()
        self._dedup_lock = threading.Lock()
        self._dedup_deadline: float = 0.0

    # -- Dedup window -------------------------------------------------------

    def set_dedup_window(
        self,
        scan_results: set[tuple[str, str]],
        duration_seconds: float = 5.0,
    ) -> None:
        """Arm the post-scan dedup window.

        *scan_results* is a set of ``(path, sha256)`` pairs from the initial
        scan.  For *duration_seconds* after this call, any watchdog event
        whose ``(path, sha256)`` matches an entry in the set is silently
        dropped (it was already emitted by the scan).

        After the window expires the set is cleared automatically via an
        ``asyncio`` timer on the event loop.
        """
        with self._dedup_lock:
            self._dedup_set = scan_results
            self._dedup_deadline = time.monotonic() + duration_seconds
        self._loop.call_later(duration_seconds, self._clear_dedup_window)

    def _clear_dedup_window(self) -> None:
        with self._dedup_lock:
            self._dedup_set = set()
            self._dedup_deadline = 0.0

    def _is_dedup_duplicate(self, path: str, sha256: str | None) -> bool:
        """Check if an event should be dropped by the dedup window."""
        if sha256 is None:
            return False
        with self._dedup_lock:
            if not self._dedup_set:
                return False
            if time.monotonic() > self._dedup_deadline:
                # Window expired but timer hasn't fired yet — clear now.
                self._dedup_set = set()
                self._dedup_deadline = 0.0
                return False
            return (path, sha256) in self._dedup_set

    # -- Filtering ----------------------------------------------------------

    def _extra_skip(self, path: str) -> bool:
        """Hook for subclass-specific skip logic (e.g. .obsidian/)."""
        return False

    def _should_skip(self, path: str) -> bool:
        if self._extra_skip(path):
            return True
        p = Path(path)
        suffix = p.suffix.lower()
        if self._include_extensions and suffix not in self._include_extensions:
            # Image extensions always pass through so standalone images
            # reach the vision pipeline even with a restricted extension list.
            if suffix not in IMAGE_EXTENSIONS:
                return True
        for pattern in self._exclude_patterns:
            if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(p.name, pattern):
                return True
        return False

    # -- Event mapping ------------------------------------------------------

    @staticmethod
    def _operation(event: FileSystemEvent) -> str | None:
        if isinstance(event, FileCreatedEvent):
            return "created"
        if isinstance(event, FileModifiedEvent):
            return "modified"
        if isinstance(event, FileDeletedEvent):
            return "deleted"
        return None

    # -- Metadata -----------------------------------------------------------

    def _extra_meta(self, src_path: str) -> dict[str, Any]:
        """Hook for subclass-specific metadata (e.g. vault info)."""
        return {}

    # -- Event building -----------------------------------------------------

    def _build_event(self, event: FileSystemEvent) -> dict[str, Any] | None:
        if event.is_directory:
            return None
        op = self._operation(event)
        if op is None:
            return None
        src_path = str(event.src_path)
        if self._should_skip(src_path):
            return None

        now = datetime.now(timezone.utc).isoformat()
        meta = self._extra_meta(src_path)
        ingest: dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "source_type": self._source_type,
            "source_id": src_path,
            "operation": op,
            "mime_type": guess_mime(src_path),
            "meta": meta,
            "enqueued_at": now,
        }

        p = Path(src_path)
        if op != "deleted" and p.is_file():
            try:
                result = _read_file_atomic(p, self._max_file_size)
                if result is None:
                    logger.warning(
                        "Skipping %s — exceeds max_file_size (%d bytes)",
                        src_path,
                        self._max_file_size,
                    )
                    return None
                data, mtime_ns = result
                ingest["source_modified_at"] = datetime.fromtimestamp(
                    mtime_ns / 1e9, tz=timezone.utc
                ).isoformat()
                ingest["meta"]["sha256"] = _sha256_of(data)
                ingest["meta"]["size_bytes"] = len(data)

                # Decode text content for text MIME types so downstream
                # parsers (FileParser, ObsidianParser) receive the file body.
                mime = ingest["mime_type"]
                if mime.startswith("text/"):
                    ingest["text"] = data.decode("utf-8", errors="replace")
                elif (
                    mime.startswith("image/")
                    or mime == "application/pdf"
                    or mime.startswith("application/x-iwork-")
                ):
                    ingest["raw_bytes"] = data
            except OSError:
                logger.warning(
                    "Failed to read %s, emitting event without content hash", src_path
                )
                ingest["source_modified_at"] = now
        else:
            ingest["source_modified_at"] = now

        return ingest

    # -- Cursor tracking ----------------------------------------------------

    def get_cursor_snapshot(self) -> dict[str, FileEntry]:
        """Return a thread-safe copy of the in-memory cursor."""
        with self._cursor_lock:
            return dict(self._cursor)

    def set_cursor(self, cursor: dict[str, FileEntry]) -> None:
        """Replace the in-memory cursor (e.g. from a loaded checkpoint)."""
        with self._cursor_lock:
            self._cursor = dict(cursor)

    def _update_cursor(self, ingest: dict[str, Any]) -> None:
        """Update in-memory cursor from a dispatched event."""
        src_path = ingest["source_id"]
        op = ingest["operation"]
        with self._cursor_lock:
            if op == "deleted":
                self._cursor.pop(src_path, None)
            else:
                meta = ingest.get("meta", {})
                sha256 = meta.get("sha256", "")
                size = meta.get("size_bytes", 0)
                # mtime_ns from source_modified_at (ISO string -> ns)
                modified_at = ingest.get("source_modified_at", "")
                mtime_ns = 0
                if modified_at:
                    try:
                        from datetime import datetime

                        dt = datetime.fromisoformat(modified_at)
                        mtime_ns = int(dt.timestamp() * 1e9)
                    except (ValueError, OSError):
                        pass
                if sha256:
                    self._cursor[src_path] = FileEntry(
                        sha256=sha256,
                        mtime_ns=mtime_ns,
                        size=size,
                    )

    # -- Dispatch -----------------------------------------------------------

    def on_created(self, event: FileSystemEvent) -> None:
        self._dispatch(event)

    def on_modified(self, event: FileSystemEvent) -> None:
        self._dispatch(event)

    def on_deleted(self, event: FileSystemEvent) -> None:
        self._dispatch(event)

    def _dispatch(self, event: FileSystemEvent) -> None:
        ingest = self._build_event(event)
        if ingest is not None:
            # Check dedup window before enqueuing or updating cursor.
            # Updating the cursor with dedup-dropped events would corrupt
            # mtime_ns values (the ISO-string round-trip is lossy) and
            # cause false "modified" detections on subsequent scans.
            sha256 = ingest.get("meta", {}).get("sha256")
            src_path = ingest.get("source_id", "")
            if self._is_dedup_duplicate(src_path, sha256):
                logger.info("Dedup: skipping %s (already scanned)", src_path)
                INITIAL_SCAN_DEDUP_DROPPED.labels(
                    source_type=self._source_type,
                ).inc()
                return

            # Skip redundant deletions: if a file isn't in the handler's
            # cursor the initial scan already reported its removal.  macOS
            # FSEvents can replay recent deletions when a new observer
            # starts, so this avoids duplicate "deleted" events.
            if ingest["operation"] == "deleted":
                with self._cursor_lock:
                    if src_path not in self._cursor:
                        return

            self._update_cursor(ingest)

            SOURCE_WATCHER_EVENTS.labels(
                source_type=self._source_type,
                event_type=ingest["operation"],
            ).inc()
            WATCHER_LAST_EVENT_TIMESTAMP.labels(
                source_type=self._source_type,
            ).set_to_current_time()
            asyncio.run_coroutine_threadsafe(self._queue.put(ingest), self._loop)
