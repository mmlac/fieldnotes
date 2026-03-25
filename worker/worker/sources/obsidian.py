"""Obsidian vault source using watchdog for real-time file monitoring.

Detects Obsidian vaults by the presence of ``.obsidian/`` directories under
configured vault paths. Watches vault files and emits IngestEvent dicts with
vault-aware metadata (vault_path, vault_name, relative_path).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING, Any

from watchdog.observers import Observer

from worker.metrics import (
    INITIAL_SCAN_DURATION_SECONDS,
    INITIAL_SCAN_FILES_TOTAL,
    OBSIDIAN_VAULTS_DISCOVERED,
    WATCHER_ACTIVE,
    initial_sync_add_items,
    initial_sync_source_done,
)

from ._handler import (
    DEFAULT_MAX_FILE_SIZE,
    BaseHandler,
    _read_file_atomic,
    _sha256_of,
    guess_mime,
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

logger = logging.getLogger(__name__)


def discover_vaults(search_paths: list[Path]) -> list[Path]:
    """Find Obsidian vaults by looking for ``.obsidian/`` directories.

    Returns resolved paths to vault root directories (the parent of
    ``.obsidian/``).
    """
    vaults: list[Path] = []
    for base in search_paths:
        if not base.is_dir():
            logger.warning("Vault search path does not exist, skipping: %s", base)
            continue
        if base.is_symlink():
            logger.warning(
                "Vault search path is a symlink, skipping for safety: %s", base
            )
            continue
        # Direct check: is base itself a vault?
        if (base / ".obsidian").is_dir():
            vaults.append(base)
            continue
        # One-level scan for vaults under base
        for child in sorted(base.iterdir()):
            if child.is_symlink():
                logger.warning(
                    "Skipping symlinked directory during vault discovery: %s",
                    child,
                )
                continue
            if child.is_dir() and (child / ".obsidian").is_dir():
                vaults.append(child)
    return vaults


def _should_skip_scan(
    path: Path,
    vault_path: Path,
    include_extensions: set[str] | None,
    exclude_patterns: list[str],
) -> bool:
    """Apply the same filtering as _VaultHandler._should_skip for scan files."""
    # Skip .obsidian/ config directory
    try:
        path.relative_to(vault_path / ".obsidian")
        return True
    except ValueError:
        pass

    if include_extensions and path.suffix.lower() not in include_extensions:
        return True

    path_str = str(path)
    for pattern in exclude_patterns:
        if fnmatch(path_str, pattern) or fnmatch(path.name, pattern):
            return True
        if any(fnmatch(part, pattern) for part in path.parts):
            return True

    return False


def _scan_vault(
    vault_path: Path,
    include_extensions: set[str] | None,
    exclude_patterns: list[str],
    max_file_size: int,
) -> dict[str, FileEntry]:
    """Walk a vault directory and build a cursor of file entries."""
    entries: dict[str, FileEntry] = {}
    for root, dirs, files in os.walk(vault_path):
        root_path = Path(root)
        # Prune .obsidian directory from walk
        if ".obsidian" in dirs:
            dirs.remove(".obsidian")
        for fname in files:
            fpath = root_path / fname
            if fpath.is_symlink():
                continue
            if _should_skip_scan(
                fpath, vault_path, include_extensions, exclude_patterns
            ):
                continue
            try:
                result = _read_file_atomic(fpath, max_file_size)
                if result is None:
                    logger.debug("Skipping %s — exceeds max_file_size", fpath)
                    continue
                data, mtime_ns = result
                sha256 = _sha256_of(data)
                entries[str(fpath)] = FileEntry(
                    sha256=sha256, mtime_ns=mtime_ns, size=len(data)
                )
            except OSError:
                logger.debug("Failed to read %s during scan, skipping", fpath)
    return entries


def _build_scan_event(
    file_path: str,
    operation: str,
    entry: FileEntry | None,
    vault_path: Path,
    vault_name: str,
    now: str,
    *,
    categories_key: str = "categories",
) -> dict[str, Any]:
    """Build a single IngestEvent dict for a scan result."""
    try:
        relative_path = str(Path(file_path).relative_to(vault_path))
    except ValueError:
        relative_path = file_path

    meta: dict[str, Any] = {
        "vault_path": str(vault_path),
        "vault_name": vault_name,
        "relative_path": relative_path,
        "categories_key": categories_key,
    }

    ingest: dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "source_type": "obsidian",
        "source_id": file_path,
        "operation": operation,
        "mime_type": guess_mime(file_path),
        "meta": meta,
        "enqueued_at": now,
    }

    if operation != "deleted" and entry is not None:
        ingest["source_modified_at"] = datetime.fromtimestamp(
            entry.mtime_ns / 1e9, tz=timezone.utc
        ).isoformat()
        meta["sha256"] = entry.sha256
        meta["size_bytes"] = entry.size

        # Read text content for text MIME types
        p = Path(file_path)
        if ingest["mime_type"].startswith("text/") and p.is_file():
            try:
                ingest["text"] = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                pass
    else:
        ingest["source_modified_at"] = now

    ingest["initial_scan"] = True
    return ingest


class _VaultHandler(BaseHandler):
    """Watchdog handler that emits vault-aware IngestEvent dicts."""

    _source_type = "obsidian"

    def __init__(
        self,
        queue: PersistentQueue,
        loop: asyncio.AbstractEventLoop,
        vault_path: Path,
        vault_name: str,
        include_extensions: set[str] | None,
        exclude_patterns: list[str],
        max_file_size: int = DEFAULT_MAX_FILE_SIZE,
        categories_key: str = "categories",
        cursor_key: str | None = None,
    ) -> None:
        super().__init__(
            queue=queue,
            loop=loop,
            include_extensions=include_extensions,
            exclude_patterns=exclude_patterns,
            max_file_size=max_file_size,
            cursor_key=cursor_key,
        )
        self._vault_path = vault_path
        self._vault_name = vault_name
        self._categories_key = categories_key

    def _extra_skip(self, path: str) -> bool:
        """Skip files inside .obsidian/ config directory."""
        try:
            Path(path).relative_to(self._vault_path / ".obsidian")
            return True
        except ValueError:
            return False

    def _extra_meta(self, src_path: str) -> dict[str, Any]:
        try:
            relative_path = str(Path(src_path).relative_to(self._vault_path))
        except ValueError:
            relative_path = src_path
        return {
            "vault_path": str(self._vault_path),
            "vault_name": self._vault_name,
            "relative_path": relative_path,
            "categories_key": self._categories_key,
        }


class ObsidianSource(PythonSource):
    """Watches Obsidian vaults using watchdog.

    Config keys (from ``[sources.obsidian]``):
        vault_paths: list[str]       — directories to scan for vaults (required)
        include_extensions: list[str] — e.g. [".md", ".canvas"] (optional, default: all)
        exclude_patterns: list[str]  — glob patterns to skip (optional)
        recursive: bool              — watch subdirectories (default: true)
    """

    def __init__(self) -> None:
        self._vault_paths: list[Path] = []
        self._include_extensions: set[str] | None = None
        self._exclude_patterns: list[str] = []
        self._recursive: bool = True
        self._max_file_size: int = DEFAULT_MAX_FILE_SIZE
        self._categories_key: str = "categories"

    def name(self) -> str:
        return "obsidian"

    def configure(self, cfg: dict[str, Any]) -> None:
        raw_paths = cfg.get("vault_paths")
        if not raw_paths:
            raise ValueError("ObsidianSource requires 'vault_paths' in config")
        self._vault_paths = []
        for p in raw_paths:
            path = Path(p).expanduser()
            if path.is_symlink():
                logger.warning(
                    "Vault search path is a symlink, skipping for safety: %s", path
                )
                continue
            self._vault_paths.append(path.resolve())

        exts = cfg.get("include_extensions")
        if exts:
            self._include_extensions = {
                ext if ext.startswith(".") else f".{ext}" for ext in exts
            }

        self._exclude_patterns = cfg.get("exclude_patterns", [])
        self._recursive = cfg.get("recursive", True)
        self._max_file_size = cfg.get("max_file_size", DEFAULT_MAX_FILE_SIZE)

        cats_key = cfg.get("categories_key")
        if cats_key:
            self._categories_key = cats_key

    async def start(self, queue: PersistentQueue) -> None:
        loop = asyncio.get_running_loop()
        vaults = discover_vaults(self._vault_paths)
        OBSIDIAN_VAULTS_DISCOVERED.set(len(vaults))
        if not vaults:
            logger.warning("No Obsidian vaults found under configured vault_paths")
            # Still run the loop so we can be cleanly cancelled
            try:
                while True:
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                raise

        # -- Initial scan: walk all vaults before starting watchdog --
        scan_start = time.monotonic()
        stored_cursor = deserialize_file_cursor(queue.load_cursor("obsidian"))
        current_cursor: dict[str, FileEntry] = {}
        _stale_vault_deletes = 0
        handled_stored_paths: set[str] = set()
        total_new = 0
        total_modified = 0
        total_deleted = 0

        for vault in vaults:
            vault_name = vault.name
            vault_entries = _scan_vault(
                vault,
                self._include_extensions,
                self._exclude_patterns,
                self._max_file_size,
            )
            current_cursor.update(vault_entries)

            # Filter stored cursor to entries belonging to this vault
            vault_prefix = str(vault) + os.sep
            vault_stored = {
                k: v for k, v in stored_cursor.items() if k.startswith(vault_prefix)
            }
            handled_stored_paths.update(vault_stored)

            diff = diff_cursor(vault_entries, vault_stored)

            vault_actionable = len(diff.new) + len(diff.modified) + len(diff.deleted)
            if vault_actionable:
                initial_sync_add_items(vault_actionable)

            now = datetime.now(timezone.utc).isoformat()
            for file_path in diff.new:
                event = _build_scan_event(
                    file_path, "created", vault_entries[file_path],
                    vault, vault_name, now,
                    categories_key=self._categories_key,
                )
                queue.enqueue(event)

            for file_path in diff.modified:
                event = _build_scan_event(
                    file_path, "modified", vault_entries[file_path],
                    vault, vault_name, now,
                    categories_key=self._categories_key,
                )
                queue.enqueue(event)

            for file_path in diff.deleted:
                event = _build_scan_event(
                    file_path, "deleted", None,
                    vault, vault_name, now,
                    categories_key=self._categories_key,
                )
                queue.enqueue(event)
                # Remove from current_cursor so deleted files aren't
                # re-emitted on the next scan.
                current_cursor.pop(file_path, None)

            n_new = len(diff.new)
            n_mod = len(diff.modified)
            n_del = len(diff.deleted)
            total_new += n_new
            total_modified += n_mod
            total_deleted += n_del

            if n_new or n_mod or n_del:
                logger.info(
                    "Initial scan of vault %s: %d new, %d modified, %d deleted",
                    vault_name,
                    n_new,
                    n_mod,
                    n_del,
                )

        # Emit deleted events for files from vaults that no longer exist
        for stored_path in stored_cursor:
            if stored_path in handled_stored_paths:
                continue
            # This path wasn't under any current vault — vault was removed
            parent = Path(stored_path).parent
            vault_path_str = str(parent)
            vault_name = parent.name

            now = datetime.now(timezone.utc).isoformat()
            try:
                relative_path = str(Path(stored_path).relative_to(vault_path_str))
            except ValueError:
                relative_path = stored_path

            ingest: dict[str, Any] = {
                "id": str(uuid.uuid4()),
                "source_type": "obsidian",
                "source_id": stored_path,
                "operation": "deleted",
                "mime_type": guess_mime(stored_path),
                "meta": {
                    "vault_path": vault_path_str,
                    "vault_name": vault_name,
                    "relative_path": relative_path,
                },
                "enqueued_at": now,
                "source_modified_at": now,
                "initial_scan": True,
            }
            queue.enqueue(ingest)
            total_deleted += 1
            _stale_vault_deletes += 1

        if _stale_vault_deletes:
            initial_sync_add_items(_stale_vault_deletes)

        # Persist the scan cursor after all events are enqueued.
        queue.save_cursor("obsidian", serialize_file_cursor(current_cursor))

        scan_duration = time.monotonic() - scan_start
        total_files = len(current_cursor)
        total_unchanged = total_files - total_new - total_modified
        INITIAL_SCAN_DURATION_SECONDS.labels(source_type="obsidian").set(scan_duration)
        INITIAL_SCAN_FILES_TOTAL.labels(source_type="obsidian", result="new").inc(
            total_new
        )
        INITIAL_SCAN_FILES_TOTAL.labels(source_type="obsidian", result="modified").inc(
            total_modified
        )
        INITIAL_SCAN_FILES_TOTAL.labels(source_type="obsidian", result="deleted").inc(
            total_deleted
        )
        INITIAL_SCAN_FILES_TOTAL.labels(source_type="obsidian", result="unchanged").inc(
            total_unchanged
        )
        logger.info(
            "Initial scan complete: %d files in %.2fs (%d new, %d modified, %d deleted)",
            total_files,
            scan_duration,
            total_new,
            total_modified,
            total_deleted,
        )
        initial_sync_source_done()

        # Build dedup set from all scanned files
        scan_pairs: set[tuple[str, str]] = {
            (path, entry.sha256) for path, entry in current_cursor.items()
        }

        # -- Start watchdog observer --
        observer = Observer()
        handlers: list[_VaultHandler] = []
        for vault in vaults:
            vault_name = vault.name
            handler = _VaultHandler(
                queue=queue,
                loop=loop,
                vault_path=vault,
                vault_name=vault_name,
                include_extensions=self._include_extensions,
                exclude_patterns=self._exclude_patterns,
                max_file_size=self._max_file_size,
                categories_key=self._categories_key,
                cursor_key="obsidian",
            )
            # Seed handler with the just-scanned current state so that
            # _save_checkpoint (called on cancel or interval) persists the
            # fresh scan data rather than overwriting it with the stale
            # stored_cursor that predates this run.
            vault_prefix = str(vault) + os.sep
            vault_current = {
                k: v for k, v in current_cursor.items() if k.startswith(vault_prefix)
            }
            handler.set_cursor(vault_current)
            # Filter dedup set to entries belonging to this vault
            vault_dedup = {(p, s) for p, s in scan_pairs if p.startswith(vault_prefix)}
            handler.set_dedup_window(vault_dedup)
            handlers.append(handler)
            observer.schedule(handler, str(vault), recursive=self._recursive)
            logger.info(
                "Watching Obsidian vault '%s' at %s (recursive=%s)",
                vault_name,
                vault,
                self._recursive,
            )

        observer.start()
        WATCHER_ACTIVE.labels(source_type="obsidian").set(1)
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            raise
        finally:
            WATCHER_ACTIVE.labels(source_type="obsidian").set(0)
            observer.stop()
            observer.join()
