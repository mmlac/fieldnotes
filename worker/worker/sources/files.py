"""Filesystem source using watchdog for real-time file monitoring.

Watches configured directories and emits IngestEvent dicts for
created, modified, and deleted files. Supports extension filtering
and exclude patterns.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from watchdog.observers import Observer

from ._handler import DEFAULT_MAX_FILE_SIZE, BaseHandler, streaming_sha256
from .base import PythonSource

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
