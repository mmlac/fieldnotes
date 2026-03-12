"""macOS application scanner source adapter.

Discovers installed .app bundles in standard macOS directories, parses
``Info.plist`` metadata, and emits IngestEvent dicts with change detection.
Automatically disables on non-macOS platforms.

Config section ``[sources.macos_apps]``::

    enabled = true
    scan_dirs = ["/Applications", "~/Applications"]
    poll_interval_seconds = 86400
    state_path = "~/.fieldnotes/state/apps.json"
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import platform
import plistlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from worker.metrics import (
    SOURCE_WATCHER_EVENTS,
    WATCHER_ACTIVE,
    WATCHER_LAST_EVENT_TIMESTAMP,
)

from .base import PythonSource

logger = logging.getLogger(__name__)

DEFAULT_POLL_INTERVAL = 86400  # 24 hours
DEFAULT_SCAN_DIRS = ["/Applications", "~/Applications"]
DEFAULT_STATE_PATH = Path.home() / ".fieldnotes" / "state" / "apps.json"

_SOURCE_TYPE = "macos_apps"


def _is_macos() -> bool:
    return platform.system() == "Darwin"


def _discover_apps(scan_dirs: list[Path]) -> list[Path]:
    """Find .app bundles in *scan_dirs*, recursing one level into subdirs."""
    apps: list[Path] = []
    for base in scan_dirs:
        if not base.is_dir():
            logger.debug("Scan directory does not exist, skipping: %s", base)
            continue
        for child in sorted(base.iterdir()):
            if child.is_symlink():
                logger.debug("Skipping symlinked entry: %s", child)
                continue
            if child.suffix == ".app" and child.is_dir():
                apps.append(child)
            elif child.is_dir() and not child.suffix == ".app":
                # Recurse one level (e.g. /Applications/Utilities/)
                for grandchild in sorted(child.iterdir()):
                    if grandchild.is_symlink():
                        logger.debug("Skipping symlinked entry: %s", grandchild)
                        continue
                    if grandchild.suffix == ".app" and grandchild.is_dir():
                        apps.append(grandchild)
    return apps


def _parse_info_plist(app_path: Path) -> dict[str, Any] | None:
    """Parse Contents/Info.plist from an .app bundle.

    Returns extracted metadata dict, or None if the plist cannot be read.
    """
    plist_path = app_path / "Contents" / "Info.plist"
    if not plist_path.is_file():
        return None
    try:
        with open(plist_path, "rb") as f:
            plist = plistlib.load(f)
    except Exception:
        logger.warning("Failed to parse Info.plist for %s", app_path.name)
        return None

    display_name = (
        plist.get("CFBundleDisplayName")
        or plist.get("CFBundleName")
        or app_path.stem
    )
    bundle_id = plist.get("CFBundleIdentifier", "")
    version = plist.get("CFBundleShortVersionString", "")
    category = plist.get("LSApplicationCategoryType", "")

    return {
        "name": display_name,
        "bundle_id": bundle_id,
        "version": version,
        "path": str(app_path),
        "category": category,
    }


def _hash_plist(app_path: Path) -> str | None:
    """Compute SHA-256 of the Info.plist file for change detection."""
    plist_path = app_path / "Contents" / "Info.plist"
    if not plist_path.is_file():
        return None
    try:
        return hashlib.sha256(plist_path.read_bytes()).hexdigest()
    except OSError:
        return None


def _load_state(path: Path) -> dict[str, str]:
    """Load previous scan state: mapping of bundle_id → plist SHA-256."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        logger.warning("Failed to read app state file %s, starting fresh", path)
        return {}


def _save_state(path: Path, state: dict[str, str]) -> None:
    """Persist scan state to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state))
    path.chmod(0o600)


def _build_event(
    metadata: dict[str, Any],
    operation: str,
) -> dict[str, Any]:
    """Build an IngestEvent dict for a macOS application."""
    bundle_id = metadata.get("bundle_id") or metadata.get("name", "unknown")
    source_id = f"app://{bundle_id}"
    now = datetime.now(timezone.utc).isoformat()

    return {
        "id": str(uuid.uuid4()),
        "source_type": _SOURCE_TYPE,
        "source_id": source_id,
        "operation": operation,
        "mime_type": "application/x-apple-app",
        "meta": metadata,
        "enqueued_at": now,
        "source_modified_at": now,
    }


class MacOSAppsSource(PythonSource):
    """Scans macOS .app bundles and emits IngestEvent dicts.

    Config keys (from ``[sources.macos_apps]``):
        enabled: bool             — enable scanning (default: true on macOS)
        scan_dirs: list[str]      — directories to scan (default: /Applications, ~/Applications)
        poll_interval_seconds: int — scan interval (default: 86400)
        state_path: str           — state persistence file (optional)
    """

    def __init__(self) -> None:
        self._scan_dirs: list[Path] = []
        self._poll_interval: int = DEFAULT_POLL_INTERVAL
        self._state_path: Path = DEFAULT_STATE_PATH
        self._enabled: bool = _is_macos()

    def name(self) -> str:
        return "macos_apps"

    def configure(self, cfg: dict[str, Any]) -> None:
        if "enabled" in cfg:
            self._enabled = bool(cfg["enabled"])
        elif not _is_macos():
            self._enabled = False

        raw_dirs = cfg.get("scan_dirs", DEFAULT_SCAN_DIRS)
        self._scan_dirs = []
        for d in raw_dirs:
            path = Path(d).expanduser()
            if path.is_symlink():
                logger.warning(
                    "Scan directory is a symlink, skipping for safety: %s", path
                )
                continue
            self._scan_dirs.append(path.resolve())

        self._poll_interval = int(
            cfg.get("poll_interval_seconds", DEFAULT_POLL_INTERVAL)
        )

        state = cfg.get("state_path")
        if state:
            self._state_path = Path(state).expanduser().resolve()

    async def start(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        if not self._enabled:
            logger.info("macOS apps source disabled (not macOS or explicitly disabled)")
            try:
                while True:
                    await asyncio.sleep(3600)
            except asyncio.CancelledError:
                raise
            return

        state = _load_state(self._state_path)

        WATCHER_ACTIVE.labels(source_type=_SOURCE_TYPE).set(1)
        try:
            while True:
                await self._scan(state, queue)
                _save_state(self._state_path, state)
                await asyncio.sleep(self._poll_interval)
        except asyncio.CancelledError:
            WATCHER_ACTIVE.labels(source_type=_SOURCE_TYPE).set(0)
            raise

    async def _scan(
        self,
        state: dict[str, str],
        queue: asyncio.Queue[dict[str, Any]],
    ) -> None:
        """Perform one scan cycle, emitting events for changes."""
        loop = asyncio.get_running_loop()
        apps = await loop.run_in_executor(None, _discover_apps, self._scan_dirs)

        current: dict[str, str] = {}  # bundle_id → hash
        count = 0

        for app_path in apps:
            metadata = await loop.run_in_executor(None, _parse_info_plist, app_path)
            if metadata is None:
                continue

            plist_hash = await loop.run_in_executor(None, _hash_plist, app_path)
            if plist_hash is None:
                continue

            bundle_id = metadata.get("bundle_id") or metadata.get("name", "unknown")
            current[bundle_id] = plist_hash

            prev_hash = state.get(bundle_id)
            if prev_hash is None:
                operation = "created"
            elif prev_hash != plist_hash:
                operation = "modified"
            else:
                continue  # unchanged

            event = _build_event(metadata, operation)
            await queue.put(event)
            SOURCE_WATCHER_EVENTS.labels(
                source_type=_SOURCE_TYPE, event_type=operation,
            ).inc()
            WATCHER_LAST_EVENT_TIMESTAMP.labels(
                source_type=_SOURCE_TYPE,
            ).set_to_current_time()
            count += 1

        # Detect removed apps
        for bundle_id in set(state) - set(current):
            event = _build_event(
                {"bundle_id": bundle_id, "name": bundle_id, "path": "", "version": "", "category": ""},
                "deleted",
            )
            await queue.put(event)
            SOURCE_WATCHER_EVENTS.labels(
                source_type=_SOURCE_TYPE, event_type="deleted",
            ).inc()
            WATCHER_LAST_EVENT_TIMESTAMP.labels(
                source_type=_SOURCE_TYPE,
            ).set_to_current_time()
            count += 1

        # Update state to reflect current scan
        state.clear()
        state.update(current)

        if count:
            logger.info(
                "macOS app scan complete: %d event(s) emitted (%d apps discovered)",
                count, len(current),
            )
        else:
            logger.debug(
                "macOS app scan complete: no changes (%d apps discovered)",
                len(current),
            )
