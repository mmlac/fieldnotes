"""Obsidian vault source using watchdog for real-time file monitoring.

Detects Obsidian vaults by the presence of ``.obsidian/`` directories under
configured vault paths. Watches vault files and emits IngestEvent dicts with
vault-aware metadata (vault_path, vault_name, relative_path).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from watchdog.observers import Observer

from ._handler import DEFAULT_MAX_FILE_SIZE, BaseHandler
from .base import PythonSource

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


class _VaultHandler(BaseHandler):
    """Watchdog handler that emits vault-aware IngestEvent dicts."""

    _source_type = "obsidian"

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
        super().__init__(
            queue=queue,
            loop=loop,
            include_extensions=include_extensions,
            exclude_patterns=exclude_patterns,
            max_file_size=max_file_size,
        )
        self._vault_path = vault_path
        self._vault_name = vault_name

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
        finally:
            observer.stop()
            observer.join()
