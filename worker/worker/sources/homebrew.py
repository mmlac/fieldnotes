"""Homebrew source adapter with polling-based discovery.

Discovers installed Homebrew formulae and casks via ``brew info --json=v2
--installed``, emitting one IngestEvent per package.  Tracks installs,
uninstalls, and version changes across scans using a local state file.

Config section ``[sources.homebrew]``::

    enabled = true
    poll_interval_seconds = 21600
    state_path = "~/.fieldnotes/state/brew.json"
    include_system = false
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from worker.queue import PersistentQueue

from worker.metrics import (
    SOURCE_WATCHER_EVENTS,
    WATCHER_ACTIVE,
    WATCHER_LAST_EVENT_TIMESTAMP,
    initial_sync_source_done,
)

from worker.log_sanitizer import redact_home_path

from .base import PythonSource
from .cursor import save_json_atomic

logger = logging.getLogger(__name__)

DEFAULT_POLL_INTERVAL = 21600  # 6 hours
DEFAULT_STATE_PATH = Path.home() / ".fieldnotes" / "state" / "brew.json"
BREW_TIMEOUT = 120  # seconds — some systems are very slow


def _find_brew() -> str | None:
    """Return the path to the brew binary, or None if not found."""
    # Check well-known locations first (faster than shutil.which)
    for candidate in ("/opt/homebrew/bin/brew", "/usr/local/bin/brew"):
        if Path(candidate).is_file():
            return candidate
    return shutil.which("brew")


def _load_state(path: Path) -> dict[str, dict[str, Any]]:
    """Load previous scan state.

    Returns a mapping of ``source_id -> {name, version, kind, ...}``.
    """
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        logger.warning(
            "Failed to read brew state file %s, starting fresh",
            redact_home_path(str(path)),
        )
        return {}


def _save_state(path: Path, state: dict[str, dict[str, Any]]) -> None:
    """Persist scan state to disk."""
    save_json_atomic(path, state)


def _collect_installed(brew_path: str) -> dict[str, Any]:
    """Run ``brew info --json=v2 --installed`` and return the parsed JSON."""
    result = subprocess.run(
        [brew_path, "info", "--json=v2", "--installed"],
        capture_output=True,
        text=True,
        timeout=BREW_TIMEOUT,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"brew info failed (rc={result.returncode}): {result.stderr[:500]}"
        )
    return json.loads(result.stdout)


def _brew_prefix(brew_path: str) -> str | None:
    """Return the Homebrew prefix directory."""
    try:
        result = subprocess.run(
            [brew_path, "--prefix"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, OSError):
        pass
    return None


def _formula_binaries(prefix: str | None, formula_name: str) -> list[str]:
    """Return binary names linked by a formula into the Homebrew bin dir."""
    if prefix is None:
        return []
    bin_dir = Path(prefix) / "bin"
    if not bin_dir.is_dir():
        return []
    cellar = Path(prefix) / "Cellar" / formula_name
    if not cellar.is_dir():
        return []
    bins: list[str] = []
    for link in bin_dir.iterdir():
        try:
            if not link.is_symlink():
                continue
            target = link.resolve(strict=True)
            if cellar in target.parents or target.parent == cellar:
                bins.append(link.name)
        except OSError:
            continue
    return sorted(bins)


def _build_formula_snapshot(
    f: dict[str, Any], prefix: str | None
) -> tuple[str, dict[str, Any]]:
    """Build a (source_id, snapshot) pair for a formula."""
    name = f["name"]
    source_id = f"brew://formula/{name}"
    installed = f.get("installed", [])
    version = (
        installed[0]["version"]
        if installed
        else f.get("versions", {}).get("stable", "")
    )
    snapshot: dict[str, Any] = {
        "name": name,
        "kind": "formula",
        "version": version,
        "description": f.get("desc", ""),
        "homepage": f.get("homepage", ""),
        "tap": f.get("tap", ""),
        "binaries": _formula_binaries(prefix, name),
        "installed_paths": [
            entry.get("installed_as_dependency", False) for entry in installed
        ],
    }
    return source_id, snapshot


def _build_cask_snapshot(c: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Build a (source_id, snapshot) pair for a cask."""
    token = c["token"]
    source_id = f"brew://cask/{token}"
    snapshot: dict[str, Any] = {
        "name": c.get("name", [token])[0] if isinstance(c.get("name"), list) else token,
        "kind": "cask",
        "version": c.get("version", ""),
        "description": c.get("desc", ""),
        "homepage": c.get("homepage", ""),
        "tap": c.get("tap", ""),
        "bundle_id": (c.get("artifacts") or [{}])[0].get("app", [""])[0]
        if c.get("artifacts")
        else "",
        "installed_via_brew": True,
    }
    return source_id, snapshot


def _build_event(
    source_id: str,
    snapshot: dict[str, Any],
    operation: str,
) -> dict[str, Any]:
    """Build an IngestEvent dict for a Homebrew package."""
    now = datetime.now(timezone.utc).isoformat()
    kind = snapshot["kind"]
    name = snapshot["name"]

    # Build a human-readable text summary
    lines = [f"{name} ({kind})"]
    if snapshot.get("description"):
        lines.append(snapshot["description"])
    if snapshot.get("version"):
        lines.append(f"Version: {snapshot['version']}")
    if snapshot.get("homepage"):
        lines.append(f"Homepage: {snapshot['homepage']}")
    if snapshot.get("binaries"):
        lines.append(f"Binaries: {', '.join(snapshot['binaries'])}")

    meta: dict[str, Any] = {
        "package_name": name,
        "package_kind": kind,
        "version": snapshot.get("version", ""),
        "tap": snapshot.get("tap", ""),
        "homepage": snapshot.get("homepage", ""),
    }
    if snapshot.get("binaries"):
        meta["binaries"] = snapshot["binaries"]
    if snapshot.get("bundle_id"):
        meta["bundle_id"] = snapshot["bundle_id"]
    if snapshot.get("installed_via_brew"):
        meta["installed_via_brew"] = True

    return {
        "id": str(uuid.uuid4()),
        "source_type": "homebrew",
        "source_id": source_id,
        "operation": operation,
        "mime_type": "text/plain",
        "text": "\n".join(lines),
        "meta": meta,
        "source_modified_at": now,
        "enqueued_at": now,
    }


def _diff_snapshots(
    prev: dict[str, dict[str, Any]],
    curr: dict[str, dict[str, Any]],
) -> list[tuple[str, dict[str, Any], str]]:
    """Compute (source_id, snapshot, operation) triples from state diff."""
    events: list[tuple[str, dict[str, Any], str]] = []
    prev_ids = set(prev.keys())
    curr_ids = set(curr.keys())

    for sid in curr_ids - prev_ids:
        events.append((sid, curr[sid], "created"))

    for sid in prev_ids - curr_ids:
        events.append((sid, prev[sid], "deleted"))

    for sid in prev_ids & curr_ids:
        if prev[sid].get("version") != curr[sid].get("version"):
            events.append((sid, curr[sid], "modified"))

    return events


class HomebrewSource(PythonSource):
    """Polls Homebrew for installed formulae and casks, emits IngestEvent dicts.

    Config keys (from ``[sources.homebrew]``):
        enabled: bool                     — enable this source (default: true on macOS)
        poll_interval_seconds: int        — scan interval (default: 21600)
        state_path: str                   — state persistence file
        include_system: bool              — include macOS-bundled formulae (default: false)
    """

    def __init__(self) -> None:
        self._poll_interval: int = DEFAULT_POLL_INTERVAL
        self._state_path: Path = DEFAULT_STATE_PATH
        self._include_system: bool = False

    def name(self) -> str:
        return "homebrew"

    def configure(self, cfg: dict[str, Any]) -> None:
        self._poll_interval = int(
            cfg.get("poll_interval_seconds", DEFAULT_POLL_INTERVAL)
        )

        state = cfg.get("state_path")
        if state:
            self._state_path = Path(state).expanduser().resolve()

        self._include_system = bool(cfg.get("include_system", False))

    async def start(self, queue: PersistentQueue) -> None:
        brew_path = _find_brew()
        if brew_path is None:
            logger.info("Homebrew not found — homebrew source skipped")
            initial_sync_source_done()
            return

        logger.info("Homebrew found at %s", brew_path)
        prev_state = _load_state(self._state_path)

        WATCHER_ACTIVE.labels(source_type="homebrew").set(1)
        first_cycle = True
        try:
            while True:
                await self._scan(brew_path, prev_state, queue)
                if first_cycle:
                    initial_sync_source_done()
                    first_cycle = False
                await asyncio.sleep(self._poll_interval)
        except asyncio.CancelledError:
            WATCHER_ACTIVE.labels(source_type="homebrew").set(0)
            raise

    async def _scan(
        self,
        brew_path: str,
        prev_state: dict[str, dict[str, Any]],
        queue: PersistentQueue,
    ) -> None:
        """Run one scan cycle."""
        loop = asyncio.get_running_loop()

        try:
            data = await loop.run_in_executor(None, _collect_installed, brew_path)
        except (subprocess.TimeoutExpired, RuntimeError, json.JSONDecodeError) as exc:
            logger.error("Failed to collect Homebrew info: %s", exc)
            return

        prefix = await loop.run_in_executor(None, _brew_prefix, brew_path)

        # Build current snapshot
        curr_state: dict[str, dict[str, Any]] = {}

        for f in data.get("formulae", []):
            # Skip system formulae unless configured
            if not self._include_system:
                installed = f.get("installed", [])
                if installed and installed[0].get("installed_as_dependency"):
                    continue
            sid, snap = _build_formula_snapshot(f, prefix)
            curr_state[sid] = snap

        for c in data.get("casks", []):
            sid, snap = _build_cask_snapshot(c)
            curr_state[sid] = snap

        # Diff and emit events
        changes = _diff_snapshots(prev_state, curr_state)

        if not prev_state and curr_state:
            # Initial scan — emit all as created
            changes = [(sid, snap, "created") for sid, snap in curr_state.items()]

        events: list[dict[str, Any]] = []
        new_count = 0
        removed_count = 0
        updated_count = 0
        for source_id, snapshot, operation in changes:
            event = _build_event(source_id, snapshot, operation)
            events.append(event)
            SOURCE_WATCHER_EVENTS.labels(
                source_type="homebrew",
                event_type=operation,
            ).inc()
            WATCHER_LAST_EVENT_TIMESTAMP.labels(
                source_type="homebrew",
            ).set_to_current_time()
            if operation == "created":
                new_count += 1
            elif operation == "deleted":
                removed_count += 1
            elif operation == "modified":
                updated_count += 1

        logger.info(
            "App scan: %d new, %d removed, %d updated, %d total",
            new_count,
            removed_count,
            updated_count,
            len(curr_state),
        )

        # Update in-memory state immediately (for next poll cycle)
        prev_state.clear()
        prev_state.update(curr_state)

        if not events:
            _save_state(self._state_path, curr_state)
            return

        cursor_json = json.dumps(curr_state)
        for i, ev in enumerate(events):
            is_last = i == len(events) - 1
            queue.enqueue(
                ev,
                cursor_key="homebrew" if is_last else None,
                cursor_value=cursor_json if is_last else None,
            )
        # Also persist to legacy state file
        _save_state(self._state_path, curr_state)
