"""Filesystem source using watchdog for real-time file monitoring.

Watches configured directories and emits IngestEvent dicts for
created, modified, and deleted files. Supports extension filtering
and exclude patterns.
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from watchdog.observers import Observer

from worker.metrics import (
    INITIAL_SCAN_DURATION_SECONDS,
    INITIAL_SCAN_FILES_TOTAL,
    WATCHER_ACTIVE,
)

from ._handler import (
    DEFAULT_MAX_FILE_SIZE,
    BaseHandler,
    _read_file_atomic,
    _sha256_of,
    guess_mime,
    streaming_sha256,
)
from .base import PythonSource
from .cursor import CursorDiff, FileEntry, diff_cursor, load_cursor, save_cursor

# Re-export for backwards compatibility with existing imports.
_streaming_sha256 = streaming_sha256

logger = logging.getLogger(__name__)

_DEFAULT_CURSOR_PATH = Path("~/.fieldnotes/data/file_cursor.json")


class _Handler(BaseHandler):
    """Watchdog handler for plain filesystem watching."""

    _source_type = "files"


class FileSource(PythonSource):
    """Watches filesystem directories using watchdog.

    Config keys (from ``[sources.files]``):
        watch_paths: list[str]       — directories to monitor (required)
        include_extensions: list[str] — e.g. [".md", ".txt"] (optional, default: all)
        exclude_patterns: list[str]  — glob patterns to skip (optional)
        recursive: bool              — watch subdirectories (default: true)
        cursor_path: str             — cursor persistence file (optional)
    """

    def __init__(self) -> None:
        self._watch_paths: list[Path] = []
        self._include_extensions: set[str] | None = None
        self._exclude_patterns: list[str] = []
        self._recursive: bool = True
        self._max_file_size: int = DEFAULT_MAX_FILE_SIZE
        self._cursor_path: Path = _DEFAULT_CURSOR_PATH.expanduser()

    def name(self) -> str:
        return "files"

    def configure(self, cfg: dict[str, Any]) -> None:
        raw_paths = cfg.get("watch_paths")
        if not raw_paths:
            raise ValueError("FileSource requires 'watch_paths' in config")
        self._watch_paths = []
        for p in raw_paths:
            path = Path(p).expanduser()
            if path.is_symlink():
                logger.warning(
                    "Watch path is a symlink, skipping for safety: %s", path
                )
                continue
            self._watch_paths.append(path.resolve())

        exts = cfg.get("include_extensions")
        if exts:
            self._include_extensions = {
                ext if ext.startswith(".") else f".{ext}" for ext in exts
            }

        self._exclude_patterns = cfg.get("exclude_patterns", [])
        self._recursive = cfg.get("recursive", True)
        self._max_file_size = cfg.get("max_file_size", DEFAULT_MAX_FILE_SIZE)

        cursor = cfg.get("cursor_path")
        if cursor:
            self._cursor_path = Path(cursor).expanduser().resolve()

    def _should_skip(self, path: str) -> bool:
        """Apply the same filtering rules as the watchdog handler."""
        p = Path(path)
        if self._include_extensions and p.suffix.lower() not in self._include_extensions:
            return True
        for pattern in self._exclude_patterns:
            if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(p.name, pattern):
                return True
        return False

    def _scan_directories(self) -> dict[str, FileEntry]:
        """Walk all watch_paths and build a map of path → FileEntry."""
        current: dict[str, FileEntry] = {}
        for watch_path in self._watch_paths:
            if not watch_path.is_dir():
                continue
            if self._recursive:
                iterator = watch_path.rglob("*")
            else:
                iterator = watch_path.glob("*")
            for file_path in iterator:
                if not file_path.is_file():
                    continue
                abs_path = str(file_path)
                if self._should_skip(abs_path):
                    continue
                try:
                    result = _read_file_atomic(file_path, self._max_file_size)
                    if result is None:
                        logger.debug(
                            "Skipping %s during scan — exceeds max_file_size", abs_path
                        )
                        continue
                    data, mtime_ns = result
                    sha256 = _sha256_of(data)
                    current[abs_path] = FileEntry(
                        sha256=sha256, mtime_ns=mtime_ns, size=len(data)
                    )
                except OSError:
                    logger.warning("Failed to read %s during initial scan, skipping", abs_path)
        return current

    def _build_scan_event(
        self, file_path: str, operation: str, entry: FileEntry | None
    ) -> dict[str, Any]:
        """Build an IngestEvent dict for a file discovered during initial scan."""
        now = datetime.now(timezone.utc).isoformat()
        ingest: dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "source_type": "files",
            "source_id": file_path,
            "operation": operation,
            "mime_type": guess_mime(file_path),
            "meta": {},
            "enqueued_at": now,
        }

        if operation != "deleted" and entry is not None:
            ingest["source_modified_at"] = datetime.fromtimestamp(
                entry.mtime_ns / 1e9, tz=timezone.utc
            ).isoformat()
            ingest["meta"]["sha256"] = entry.sha256
            ingest["meta"]["size_bytes"] = entry.size

            p = Path(file_path)
            if ingest["mime_type"].startswith("text/") and p.is_file():
                try:
                    result = _read_file_atomic(p, self._max_file_size)
                    if result is not None:
                        data, _ = result
                        ingest["text"] = data.decode("utf-8", errors="replace")
                except OSError:
                    pass
        else:
            ingest["source_modified_at"] = now

        return ingest

    async def _initial_scan(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        """Walk directories, diff against cursor, and emit events."""
        scan_start = time.monotonic()

        loop = asyncio.get_running_loop()
        current = await loop.run_in_executor(None, self._scan_directories)

        stored = load_cursor(self._cursor_path)
        diff: CursorDiff = diff_cursor(current, stored)

        counts = {"new": 0, "modified": 0, "deleted": 0, "unchanged": 0}

        for file_path in diff.new:
            event = self._build_scan_event(file_path, "created", current[file_path])
            await queue.put(event)
            counts["new"] += 1

        for file_path in diff.modified:
            event = self._build_scan_event(file_path, "modified", current[file_path])
            await queue.put(event)
            counts["modified"] += 1

        for file_path in diff.deleted:
            event = self._build_scan_event(file_path, "deleted", None)
            await queue.put(event)
            counts["deleted"] += 1

        counts["unchanged"] = len(current) - counts["new"] - counts["modified"]

        save_cursor(self._cursor_path, current)

        scan_duration = time.monotonic() - scan_start
        INITIAL_SCAN_DURATION_SECONDS.labels(source_type="files").set(scan_duration)
        INITIAL_SCAN_FILES_TOTAL.labels(source_type="files", result="new").inc(counts["new"])
        INITIAL_SCAN_FILES_TOTAL.labels(source_type="files", result="modified").inc(counts["modified"])
        INITIAL_SCAN_FILES_TOTAL.labels(source_type="files", result="deleted").inc(counts["deleted"])
        INITIAL_SCAN_FILES_TOTAL.labels(source_type="files", result="unchanged").inc(counts["unchanged"])

        for watch_path in self._watch_paths:
            logger.info(
                "Initial scan of %s: %d new, %d modified, %d deleted, %d skipped",
                watch_path,
                counts["new"],
                counts["modified"],
                counts["deleted"],
                counts["unchanged"],
            )

    async def start(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        # Initial scan BEFORE watchdog to avoid duplicate events
        await self._initial_scan(queue)

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
            if watch_path.is_symlink():
                logger.warning(
                    "Watch path is a symlink, skipping for safety: %s", watch_path
                )
                continue
            observer.schedule(handler, str(watch_path), recursive=self._recursive)
            logger.info("Watching %s (recursive=%s)", watch_path, self._recursive)

        observer.start()
        WATCHER_ACTIVE.labels(source_type="files").set(1)
        try:
            while True:
                await asyncio.sleep(1)
        finally:
            WATCHER_ACTIVE.labels(source_type="files").set(0)
            observer.stop()
            observer.join()
