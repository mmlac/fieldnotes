"""Filesystem source using watchdog for real-time file monitoring.

Watches configured directories and emits IngestEvent dicts for
created, modified, and deleted files. Supports extension filtering
and exclude patterns.
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from watchdog.observers import Observer

from worker.metrics import (
    INITIAL_SCAN_DURATION_SECONDS,
    INITIAL_SCAN_FILES_TOTAL,
    WATCHER_ACTIVE,
    initial_sync_add_items,
    initial_sync_source_done,
)

from ._handler import (
    DEFAULT_MAX_FILE_SIZE,
    IMAGE_EXTENSIONS,
    BaseHandler,
    _read_file_atomic,
    _sha256_of,
    guess_mime,
    streaming_sha256,
)
from .base import PythonSource
from .cursor import (
    CursorDiff,
    FileEntry,
    deserialize_file_cursor,
    diff_cursor,
    serialize_file_cursor,
)

if TYPE_CHECKING:
    from worker.queue import PersistentQueue

# Re-export for backwards compatibility with existing imports.
_streaming_sha256 = streaming_sha256

logger = logging.getLogger(__name__)


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
    """

    def __init__(self) -> None:
        self._watch_paths: list[Path] = []
        self._include_extensions: set[str] | None = None
        self._exclude_patterns: list[str] = []
        self._index_only_patterns: list[str] = []
        self._recursive: bool = True
        self._max_file_size: int = DEFAULT_MAX_FILE_SIZE

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
                logger.warning("Watch path is a symlink, skipping for safety: %s", path)
                continue
            self._watch_paths.append(path.resolve())

        exts = cfg.get("include_extensions")
        if exts:
            self._include_extensions = {
                ext if ext.startswith(".") else f".{ext}" for ext in exts
            }

        self._exclude_patterns = cfg.get("exclude_patterns", [])
        self._index_only_patterns = cfg.get("index_only_patterns", [])
        self._recursive = cfg.get("recursive", True)
        self._max_file_size = cfg.get("max_file_size", DEFAULT_MAX_FILE_SIZE)

    def _should_skip(self, path: str) -> bool:
        """Apply the same filtering rules as the watchdog handler."""
        p = Path(path)
        suffix = p.suffix.lower()
        if self._include_extensions and suffix not in self._include_extensions:
            if suffix not in IMAGE_EXTENSIONS:
                return True
        for pattern in self._exclude_patterns:
            if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(p.name, pattern):
                return True
            # Check if any parent directory component matches the pattern,
            # so excluding "Photos Library.photoslibrary" skips all files
            # inside that directory.
            if any(fnmatch.fnmatch(part, pattern) for part in p.parts):
                return True
        return False

    def _is_index_only(self, path: str) -> bool:
        """Check if *path* matches any ``index_only_patterns``."""
        if not self._index_only_patterns:
            return False
        p = Path(path)
        for pattern in self._index_only_patterns:
            if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(p.name, pattern):
                return True
            if any(fnmatch.fnmatch(part, pattern) for part in p.parts):
                return True
        return False

    def _scan_directories(self) -> dict[str, FileEntry]:
        """Walk all watch_paths and build a map of path → FileEntry.

        A failure on one watch_path (missing directory, permission denied,
        unreadable subtree) MUST NOT stop the scan of the others.  Each
        path is wrapped in its own try/except, and the per-directory walk
        uses ``os.walk`` with an ``onerror`` callback so unreadable
        subdirs are logged-and-skipped instead of aborting iteration.
        """
        current: dict[str, FileEntry] = {}
        for watch_path in self._watch_paths:
            try:
                self._scan_one_watch_path(watch_path, current)
            except Exception:
                # Last-resort backstop — anything that escapes the
                # per-directory error handling below should still leave
                # the remaining watch_paths scannable.
                logger.exception(
                    "Initial scan of %s failed, skipping the rest of this path",
                    watch_path,
                )
        return current

    def _scan_one_watch_path(
        self, watch_path: Path, current: dict[str, FileEntry]
    ) -> None:
        """Scan a single watch_path and merge its results into *current*.

        Resilient to:
          - watch_path missing or not a directory
          - watch_path stat raising (broken symlink, mount unavailable)
          - subdirectories that raise PermissionError or OSError mid-walk
          - individual files that disappear or become unreadable mid-walk
        """
        try:
            if not watch_path.exists():
                logger.warning(
                    "Watch path does not exist, skipping: %s", watch_path
                )
                return
            if not watch_path.is_dir():
                logger.warning(
                    "Watch path is not a directory, skipping: %s", watch_path
                )
                return
        except OSError as exc:
            logger.warning(
                "Cannot stat watch path %s, skipping: %s", watch_path, exc
            )
            return

        def _on_walk_error(exc: OSError) -> None:
            # os.walk's onerror is called when listing a directory fails.
            # Log and continue — the walker will skip the unreadable dir
            # but keep going through siblings.
            logger.warning(
                "Cannot read directory during scan (%s), skipping: %s",
                exc.__class__.__name__,
                exc,
            )

        if self._recursive:
            walker = os.walk(str(watch_path), onerror=_on_walk_error)
        else:
            # Non-recursive: emit just the top-level entries.
            try:
                top_entries = list(watch_path.iterdir())
            except OSError as exc:
                logger.warning(
                    "Cannot list %s, skipping: %s", watch_path, exc
                )
                return
            top_files = [e.name for e in top_entries if not e.is_dir()]
            walker = [(str(watch_path), [], top_files)]

        for dirpath, _dirnames, filenames in walker:
            for fname in filenames:
                file_path = Path(dirpath) / fname
                abs_path = str(file_path)
                # Per-file try/except so a single bad file (vanished
                # mid-scan, permissions, dangling symlink) can never
                # poison the rest of the directory walk.
                try:
                    if not file_path.is_file():
                        continue
                    if self._should_skip(abs_path):
                        continue
                    if self._is_index_only(abs_path):
                        # Index metadata only — no content hash, no body.
                        stat = file_path.stat()
                        current[abs_path] = FileEntry(
                            sha256="",
                            mtime_ns=stat.st_mtime_ns,
                            size=stat.st_size,
                        )
                        continue
                    result = _read_file_atomic(file_path, self._max_file_size)
                    if result is None:
                        # File exceeds max_file_size — index metadata only
                        # (no content hash, no body for parsing).
                        stat = file_path.stat()
                        current[abs_path] = FileEntry(
                            sha256="",
                            mtime_ns=stat.st_mtime_ns,
                            size=stat.st_size,
                        )
                        continue
                    data, mtime_ns = result
                    sha256 = _sha256_of(data)
                    current[abs_path] = FileEntry(
                        sha256=sha256, mtime_ns=mtime_ns, size=len(data)
                    )
                except OSError as exc:
                    logger.warning(
                        "Failed to read %s during initial scan (%s), skipping",
                        abs_path,
                        exc,
                    )

    def _build_scan_event(
        self, file_path: str, operation: str, entry: FileEntry | None
    ) -> dict[str, Any]:
        """Build an IngestEvent dict for a file discovered during initial scan."""
        now = datetime.now(timezone.utc).isoformat()
        index_only = self._is_index_only(file_path)
        ingest: dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "source_type": "files",
            "source_id": file_path,
            "operation": operation,
            "mime_type": guess_mime(file_path),
            "meta": {},
            "enqueued_at": now,
        }
        if index_only:
            ingest["meta"]["index_only"] = True

        if operation != "deleted" and entry is not None:
            ingest["source_modified_at"] = datetime.fromtimestamp(
                entry.mtime_ns / 1e9, tz=timezone.utc
            ).isoformat()
            ingest["meta"]["sha256"] = entry.sha256
            ingest["meta"]["size_bytes"] = entry.size

            if not index_only:
                p = Path(file_path)
                mime = ingest["mime_type"]
                if p.is_file():
                    try:
                        result = _read_file_atomic(p, self._max_file_size)
                        if result is not None:
                            data, _ = result
                            if mime.startswith("text/"):
                                ingest["text"] = data.decode("utf-8", errors="replace")
                            elif mime.startswith("image/") or mime == "application/pdf":
                                ingest["raw_bytes"] = data
                    except OSError:
                        pass
        else:
            ingest["source_modified_at"] = now

        ingest["initial_scan"] = True
        return ingest

    async def _initial_scan(
        self, queue: PersistentQueue
    ) -> tuple[set[tuple[str, str]], dict[str, FileEntry]]:
        """Walk directories, diff against cursor, and enqueue events.

        Returns a tuple of ``(dedup_pairs, current_entries)`` where
        *dedup_pairs* is the set of ``(path, sha256)`` for the post-scan
        dedup window and *current_entries* is the scan result dict.
        """
        scan_start = time.monotonic()

        loop = asyncio.get_running_loop()
        current = await loop.run_in_executor(None, self._scan_directories)

        stored = deserialize_file_cursor(queue.load_cursor("files"))
        diff: CursorDiff = diff_cursor(current, stored)

        counts = {"new": 0, "modified": 0, "deleted": 0, "unchanged": 0}

        actionable = len(diff.new) + len(diff.modified) + len(diff.deleted)
        if actionable:
            initial_sync_add_items(actionable)

        for file_path in diff.new:
            event = self._build_scan_event(file_path, "created", current[file_path])
            queue.enqueue(event)
            counts["new"] += 1

        for file_path in diff.modified:
            event = self._build_scan_event(file_path, "modified", current[file_path])
            queue.enqueue(event)
            counts["modified"] += 1

        for file_path in diff.deleted:
            event = self._build_scan_event(file_path, "deleted", None)
            queue.enqueue(event)
            counts["deleted"] += 1

        # Persist the scan cursor atomically after all events are enqueued.
        queue.save_cursor("files", serialize_file_cursor(current))

        counts["unchanged"] = len(current) - counts["new"] - counts["modified"]

        scan_duration = time.monotonic() - scan_start
        INITIAL_SCAN_DURATION_SECONDS.labels(source_type="files").set(scan_duration)
        INITIAL_SCAN_FILES_TOTAL.labels(source_type="files", result="new").inc(
            counts["new"]
        )
        INITIAL_SCAN_FILES_TOTAL.labels(source_type="files", result="modified").inc(
            counts["modified"]
        )
        INITIAL_SCAN_FILES_TOTAL.labels(source_type="files", result="deleted").inc(
            counts["deleted"]
        )
        INITIAL_SCAN_FILES_TOTAL.labels(source_type="files", result="unchanged").inc(
            counts["unchanged"]
        )

        for watch_path in self._watch_paths:
            logger.info(
                "Initial scan of %s: %d new, %d modified, %d deleted, %d skipped",
                watch_path,
                counts["new"],
                counts["modified"],
                counts["deleted"],
                counts["unchanged"],
            )

        return {(path, entry.sha256) for path, entry in current.items()}, current

    async def start(self, queue: PersistentQueue, **_kwargs) -> None:
        # Initial scan BEFORE watchdog to avoid duplicate events.
        scan_pairs, scan_current = await self._initial_scan(queue)
        initial_sync_source_done()

        loop = asyncio.get_running_loop()
        handler = _Handler(
            queue=queue,
            loop=loop,
            include_extensions=self._include_extensions,
            exclude_patterns=self._exclude_patterns,
            max_file_size=self._max_file_size,
            cursor_key="files",
            index_only_patterns=self._index_only_patterns,
        )

        # Give the handler the post-scan state (not the stale stored
        # cursor) so that FSEvents replays of already-handled deletions
        # are suppressed by the dedup check in _dispatch().
        handler.set_cursor(scan_current)
        handler.set_dedup_window(scan_pairs)

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
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            raise
        finally:
            WATCHER_ACTIVE.labels(source_type="files").set(0)
            observer.stop()
            observer.join()
