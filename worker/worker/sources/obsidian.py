"""Obsidian vault source using watchdog for real-time file monitoring.

Detects Obsidian vaults by the presence of ``.obsidian/`` directories under
configured vault paths. Watches vault files and emits IngestEvent dicts with
vault-aware metadata (vault_path, vault_name, relative_path).
"""

from __future__ import annotations

import asyncio
import fnmatch
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
from .files import DEFAULT_MAX_FILE_SIZE, _streaming_sha256

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
        # Direct check: is base itself a vault?
        if (base / ".obsidian").is_dir():
            vaults.append(base)
            continue
        # One-level scan for vaults under base
        for child in sorted(base.iterdir()):
            if child.is_dir() and (child / ".obsidian").is_dir():
                vaults.append(child)
    return vaults


class _VaultHandler(FileSystemEventHandler):
    """Watchdog handler that emits vault-aware IngestEvent dicts."""

    def __init__(
        self,
        queue: asyncio.Queue[dict[str, Any]],
        loop: asyncio.AbstractEventLoop,
        vault_path: Path,
        vault_name: str,
        include_extensions: set[str] | None,
        exclude_patterns: list[str],
        max_file_size: int = DEFAULT_MAX_FILE_SIZE,
    ) -> None:
        self._queue = queue
        self._loop = loop
        self._vault_path = vault_path
        self._vault_name = vault_name
        self._include_extensions = include_extensions
        self._exclude_patterns = exclude_patterns
        self._max_file_size = max_file_size

    def _should_skip(self, path: str) -> bool:
        p = Path(path)
        # Always skip files inside .obsidian/ config directory
        try:
            p.relative_to(self._vault_path / ".obsidian")
            return True
        except ValueError:
            pass
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

        try:
            relative_path = str(Path(src_path).relative_to(self._vault_path))
        except ValueError:
            relative_path = src_path

        now = datetime.now(timezone.utc).isoformat()
        ingest: dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "source_type": "obsidian",
            "source_id": src_path,
            "operation": op,
            "mime_type": _guess_mime(src_path),
            "meta": {
                "vault_path": str(self._vault_path),
                "vault_name": self._vault_name,
                "relative_path": relative_path,
            },
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

    def name(self) -> str:
        return "obsidian"

    def configure(self, cfg: dict[str, Any]) -> None:
        raw_paths = cfg.get("vault_paths")
        if not raw_paths:
            raise ValueError("ObsidianSource requires 'vault_paths' in config")
        self._vault_paths = [Path(p).expanduser().resolve() for p in raw_paths]

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
        vaults = discover_vaults(self._vault_paths)
        if not vaults:
            logger.warning("No Obsidian vaults found under configured vault_paths")
            # Still run the loop so we can be cleanly cancelled
            try:
                while True:
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                raise

        observer = Observer()
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
            )
            observer.schedule(handler, str(vault), recursive=self._recursive)
            logger.info(
                "Watching Obsidian vault '%s' at %s (recursive=%s)",
                vault_name,
                vault,
                self._recursive,
            )

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
        ".canvas": "application/json",
    }
    return mime_map.get(ext, "application/octet-stream")
