"""Filesystem source using watchdog for real-time file monitoring.

Watches configured directories and emits IngestEvent dicts for
created, modified, and deleted files. Supports extension filtering
and exclude patterns.
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
from watchdog.observers import Observer

from .base import PythonSource

logger = logging.getLogger(__name__)


_SHA256_CHUNK_SIZE = 64 * 1024  # 64 KiB read chunks for streaming hash

# Default 100 MiB — files above this are logged and skipped to prevent OOM.
DEFAULT_MAX_FILE_SIZE = 100 * 1024 * 1024


def _streaming_sha256(path: Path, max_size: int) -> tuple[str, int] | None:
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


class _Handler(FileSystemEventHandler):
    """Watchdog handler that filters events and pushes to an asyncio queue."""

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

    def _should_skip(self, path: str) -> bool:
        p = Path(path)
        if self._include_extensions and p.suffix.lower() not in self._include_extensions:
            return True
        for pattern in self._exclude_patterns:
            if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(p.name, pattern):
                return True
        return False

    def _operation(self, event: FileSystemEvent) -> str | None:
        if isinstance(event, FileCreatedEvent):
            return "created"
        if isinstance(event, FileModifiedEvent):
            return "modified"
        if isinstance(event, FileDeletedEvent):
            return "deleted"
        return None

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
        ingest: dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "source_type": "files",
            "source_id": src_path,
            "operation": op,
            "mime_type": _guess_mime(src_path),
            "meta": {},
            "enqueued_at": now,
        }

        p = Path(src_path)
        if op != "deleted" and p.is_file():
            try:
                stat = p.stat()
                ingest["source_modified_at"] = datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat()
                result = _streaming_sha256(p, self._max_file_size)
                if result is None:
                    logger.warning(
                        "Skipping %s — exceeds max_file_size (%d bytes)",
                        src_path,
                        self._max_file_size,
                    )
                    return None
                digest, size = result
                ingest["meta"]["sha256"] = digest
                ingest["meta"]["size_bytes"] = size
            except OSError:
                logger.warning("Failed to read %s, emitting event without content hash", src_path)
                ingest["source_modified_at"] = now
        else:
            ingest["source_modified_at"] = now

        return ingest

    def on_created(self, event: FileSystemEvent) -> None:
        self._dispatch(event)

    def on_modified(self, event: FileSystemEvent) -> None:
        self._dispatch(event)

    def on_deleted(self, event: FileSystemEvent) -> None:
        self._dispatch(event)

    def _dispatch(self, event: FileSystemEvent) -> None:
        ingest = self._build_event(event)
        if ingest is not None:
            asyncio.run_coroutine_threadsafe(self._queue.put(ingest), self._loop)


class FileSource(PythonSource):
    """Watches filesystem directories using watchdog.

    Config keys (from ``[sources.files]``):
        watch_paths: list[str]       — directories to monitor (required)
        include_extensions: list[str] — e.g. [".md", ".txt"] (optional, default: all)
        exclude_patterns: list[str]  — glob patterns to skip (optional)
        recursive: bool              — watch subdirectories (default: true)
    """

    def __init__(self) -> None:
        self._watch_paths: list[Path] = []
        self._include_extensions: set[str] | None = None
        self._exclude_patterns: list[str] = []
        self._recursive: bool = True
        self._max_file_size: int = DEFAULT_MAX_FILE_SIZE

    def name(self) -> str:
        return "files"

    def configure(self, cfg: dict[str, Any]) -> None:
        raw_paths = cfg.get("watch_paths")
        if not raw_paths:
            raise ValueError("FileSource requires 'watch_paths' in config")
        self._watch_paths = [Path(p).expanduser().resolve() for p in raw_paths]

        exts = cfg.get("include_extensions")
        if exts:
            self._include_extensions = {
                ext if ext.startswith(".") else f".{ext}" for ext in exts
            }

        self._exclude_patterns = cfg.get("exclude_patterns", [])
        self._recursive = cfg.get("recursive", True)
        self._max_file_size = cfg.get("max_file_size", DEFAULT_MAX_FILE_SIZE)

    async def start(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        loop = asyncio.get_running_loop()
        handler = _Handler(
            queue=queue,
            loop=loop,
            include_extensions=self._include_extensions,
            exclude_patterns=self._exclude_patterns,
            max_file_size=self._max_file_size,
        )
        observer = Observer()
        for watch_path in self._watch_paths:
            if not watch_path.is_dir():
                logger.warning("Watch path does not exist, skipping: %s", watch_path)
                continue
            observer.schedule(handler, str(watch_path), recursive=self._recursive)
            logger.info("Watching %s (recursive=%s)", watch_path, self._recursive)

        observer.start()
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            observer.stop()
            observer.join()
            raise


def _guess_mime(path: str) -> str:
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
    }
    return mime_map.get(ext, "application/octet-stream")
