"""OmniFocus task scanner source adapter.

Extracts tasks from OmniFocus via Omni Automation (JXA) on macOS, emitting
IngestEvent dicts with full task metadata including tags, status, and
project hierarchy.  Change detection uses a local JSON state file keyed
by OmniFocus task id.

Config section ``[sources.omnifocus]``::

    enabled = true
    poll_interval_seconds = 300
    state_path = "~/.fieldnotes/state/omnifocus.json"
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from worker.metrics import (
    SOURCE_WATCHER_EVENTS,
    WATCHER_ACTIVE,
    WATCHER_LAST_EVENT_TIMESTAMP,
    initial_sync_source_done,
)

from .base import PythonSource
from .cursor import _ProgressTracker, save_json_atomic

logger = logging.getLogger(__name__)

DEFAULT_POLL_INTERVAL = 300  # 5 minutes
DEFAULT_STATE_PATH = Path.home() / ".fieldnotes" / "state" / "omnifocus.json"
_PROCESSED_SIDECAR = Path.home() / ".fieldnotes" / "state" / "omnifocus_processed.json"

_SOURCE_TYPE = "omnifocus"

# ── JXA script ────────────────────────────────────────────────────
# Runs inside OmniFocus's JavaScriptCore context via osascript.
# Returns a JSON array of task objects with all relevant fields.
_JXA_SCRIPT = r"""
ObjC.import("stdlib");
var app = Application("OmniFocus");
app.includeStandardAdditions = true;

var doc = app.defaultDocument;
var tasks = doc.flattenedTasks();
var result = [];

for (var i = 0; i < tasks.length; i++) {
    var t = tasks[i];

    var tagNames = [];
    var tags = t.tags();
    for (var j = 0; j < tags.length; j++) {
        tagNames.push(tags[j].name());
    }

    var projectName = "";
    try {
        var proj = t.containingProject();
        if (proj) projectName = proj.name();
    } catch(e) {}

    var parentName = "";
    var parentId = "";
    try {
        var par = t.parentTask();
        if (par) {
            parentName = par.name();
            parentId = par.id();
        }
    } catch(e) {}

    var status;
    if (t.completed()) {
        status = "completed";
    } else if (t.dropped()) {
        status = "dropped";
    } else {
        var ds = t.deferDate();
        if (ds && ds > new Date()) {
            status = "deferred";
        } else if (t.blocked()) {
            status = "blocked";
        } else {
            status = "active";
        }
    }

    var obj = {
        id: t.id(),
        name: t.name(),
        note: t.note() || "",
        status: status,
        flagged: t.flagged(),
        tags: tagNames,
        project: projectName,
        parent_task: parentName,
        parent_task_id: parentId,
        creation_date: t.creationDate() ? t.creationDate().toISOString() : null,
        modification_date: t.modificationDate() ? t.modificationDate().toISOString() : null,
        completion_date: t.completionDate() ? t.completionDate().toISOString() : null,
        due_date: t.dueDate() ? t.dueDate().toISOString() : null,
        defer_date: t.deferDate() ? t.deferDate().toISOString() : null
    };
    result.push(obj);
}

JSON.stringify(result);
"""


def _is_macos() -> bool:
    return platform.system() == "Darwin"


def _fetch_tasks() -> list[dict[str, Any]]:
    """Run JXA via osascript and return parsed task list.

    Raises ``RuntimeError`` on non-zero exit or invalid JSON.
    """
    result = subprocess.run(
        ["osascript", "-l", "JavaScript", "-e", _JXA_SCRIPT],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"osascript failed (exit {result.returncode}): {result.stderr.strip()}"
        )
    raw = result.stdout.strip()
    if not raw:
        return []
    tasks: list[dict[str, Any]] = json.loads(raw)
    if not isinstance(tasks, list):
        raise RuntimeError("JXA script returned non-list JSON")
    return tasks


def _task_hash(task: dict[str, Any]) -> str:
    """Deterministic hash of mutable task fields for change detection."""
    fields = json.dumps(
        {
            "name": task.get("name", ""),
            "note": task.get("note", ""),
            "status": task.get("status", ""),
            "flagged": task.get("flagged", False),
            "tags": sorted(task.get("tags", [])),
            "project": task.get("project", ""),
            "parent_task": task.get("parent_task", ""),
            "parent_task_id": task.get("parent_task_id", ""),
            "due_date": task.get("due_date"),
            "defer_date": task.get("defer_date"),
            "completion_date": task.get("completion_date"),
        },
        sort_keys=True,
    )
    return hashlib.sha256(fields.encode()).hexdigest()


def _load_state(path: Path) -> dict[str, str]:
    """Load previous scan state: mapping of task_id → field_hash."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        logger.warning("Failed to read OmniFocus state file %s, starting fresh", path)
        return {}


def _save_state(path: Path, state: dict[str, str]) -> None:
    """Persist scan state to disk."""
    save_json_atomic(path, state)


def _build_event(
    task: dict[str, Any],
    operation: str,
) -> dict[str, Any]:
    """Build an IngestEvent dict for an OmniFocus task."""
    task_id = task.get("id", "unknown")
    source_id = f"omnifocus://{task_id}"
    now = datetime.now(timezone.utc).isoformat()

    return {
        "id": str(uuid.uuid4()),
        "source_type": _SOURCE_TYPE,
        "source_id": source_id,
        "operation": operation,
        "mime_type": "text/plain",
        "meta": task,
        "enqueued_at": now,
        "source_modified_at": task.get("modification_date") or now,
    }


class OmniFocusSource(PythonSource):
    """Polls OmniFocus for tasks and emits IngestEvent dicts.

    Config keys (from ``[sources.omnifocus]``):
        enabled: bool               — enable polling (default: true on macOS)
        poll_interval_seconds: int  — poll interval (default: 300)
        state_path: str             — state persistence file (optional)
    """

    def __init__(self) -> None:
        self._poll_interval: int = DEFAULT_POLL_INTERVAL
        self._state_path: Path = DEFAULT_STATE_PATH
        self._enabled: bool = _is_macos()

    def name(self) -> str:
        return "omnifocus"

    def configure(self, cfg: dict[str, Any]) -> None:
        self._poll_interval = int(
            cfg.get("poll_interval_seconds", DEFAULT_POLL_INTERVAL)
        )
        if "enabled" in cfg:
            self._enabled = bool(cfg["enabled"])

        state = cfg.get("state_path")
        if state:
            self._state_path = Path(state).expanduser().resolve()

    async def start(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        if not self._enabled:
            logger.info("OmniFocus source skipped (disabled)")
            initial_sync_source_done()
            return

        state = _load_state(self._state_path)

        WATCHER_ACTIVE.labels(source_type=_SOURCE_TYPE).set(1)
        first_cycle = True
        try:
            while True:
                await self._poll(state, queue)
                if first_cycle:
                    initial_sync_source_done()
                    first_cycle = False
                await asyncio.sleep(self._poll_interval)
        except asyncio.CancelledError:
            WATCHER_ACTIVE.labels(source_type=_SOURCE_TYPE).set(0)
            raise

    async def _poll(
        self,
        state: dict[str, str],
        queue: asyncio.Queue[dict[str, Any]],
    ) -> None:
        """Perform one poll cycle, emitting events for changes."""
        loop = asyncio.get_running_loop()
        try:
            tasks = await loop.run_in_executor(None, _fetch_tasks)
        except Exception:
            logger.error("Failed to fetch OmniFocus tasks", exc_info=True)
            return

        current: dict[str, str] = {}  # task_id → hash
        events: list[dict[str, Any]] = []

        for task in tasks:
            task_id = task.get("id")
            if not task_id:
                continue

            h = _task_hash(task)
            current[task_id] = h

            prev_hash = state.get(task_id)
            if prev_hash is None:
                operation = "created"
            elif prev_hash != h:
                operation = "modified"
            else:
                continue  # unchanged

            event = _build_event(task, operation)
            events.append(event)
            SOURCE_WATCHER_EVENTS.labels(
                source_type=_SOURCE_TYPE,
                event_type=operation,
            ).inc()
            WATCHER_LAST_EVENT_TIMESTAMP.labels(
                source_type=_SOURCE_TYPE,
            ).set_to_current_time()

        # Detect deleted tasks (present in old state, absent now)
        for task_id in set(state) - set(current):
            event = _build_event(
                {
                    "id": task_id,
                    "name": "",
                    "note": "",
                    "status": "deleted",
                    "tags": [],
                },
                "deleted",
            )
            events.append(event)
            SOURCE_WATCHER_EVENTS.labels(
                source_type=_SOURCE_TYPE,
                event_type="deleted",
            ).inc()
            WATCHER_LAST_EVENT_TIMESTAMP.labels(
                source_type=_SOURCE_TYPE,
            ).set_to_current_time()

        # Update in-memory state immediately (for next poll cycle)
        state.clear()
        state.update(current)

        if events:
            logger.info(
                "OmniFocus poll complete: %d event(s) emitted (%d tasks)",
                len(events),
                len(current),
            )
        else:
            logger.debug(
                "OmniFocus poll complete: no changes (%d tasks)",
                len(current),
            )

        if not events:
            _save_state(self._state_path, current)
            return

        # Defer disk save until all events processed through the pipeline
        state_path = self._state_path

        def _save() -> None:
            _save_state(state_path, current)

        tracker = _ProgressTracker(
            total=len(events),
            sidecar_path=_PROCESSED_SIDECAR,
            on_all_done=_save,
        )
        for ev in events:
            sid = ev["source_id"]
            ev["_on_indexed"] = lambda s=sid: tracker.ack(s)
            await queue.put(ev)
