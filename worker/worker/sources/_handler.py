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

from worker.metrics import SOURCE_WATCHER_EVENTS, WATCHER_LAST_EVENT_TIMESTAMP

logger = logging.getLogger(__name__)

_SHA256_CHUNK_SIZE = 64 * 1024  # 64 KiB read chunks for streaming hash

# Default 100 MiB — files above this are logged and skipped to prevent OOM.
DEFAULT_MAX_FILE_SIZE = 100 * 1024 * 1024


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
    finally:
        # Grab mtime from the same fd before closing to minimise window.
        mtime_ns = path.stat().st_mtime_ns
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
        ".json": "application/json",
        ".yaml": "text/yaml",
        ".yml": "text/yaml",
        ".toml": "text/toml",
        ".html": "text/html",
        ".csv": "text/csv",
        ".canvas": "application/json",
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

    # -- Filtering ----------------------------------------------------------

    def _extra_skip(self, path: str) -> bool:
        """Hook for subclass-specific skip logic (e.g. .obsidian/)."""
        return False

    def _should_skip(self, path: str) -> bool:
        if self._extra_skip(path):
            return True
        p = Path(path)
        if self._include_extensions and p.suffix.lower() not in self._include_extensions:
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
                if ingest["mime_type"].startswith("text/"):
                    ingest["text"] = data.decode("utf-8", errors="replace")
            except OSError:
                logger.warning("Failed to read %s, emitting event without content hash", src_path)
                ingest["source_modified_at"] = now
        else:
            ingest["source_modified_at"] = now

        return ingest

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
            SOURCE_WATCHER_EVENTS.labels(
                source_type=self._source_type,
                event_type=ingest["operation"],
            ).inc()
            WATCHER_LAST_EVENT_TIMESTAMP.labels(
                source_type=self._source_type,
            ).set_to_current_time()
            asyncio.run_coroutine_threadsafe(self._queue.put(ingest), self._loop)
