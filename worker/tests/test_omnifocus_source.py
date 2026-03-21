"""Tests for the OmniFocus source adapter."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from worker.sources.omnifocus import (
    DEFAULT_POLL_INTERVAL,
    OmniFocusSource,
    _build_event,
    _load_state,
    _save_state,
    _task_hash,
)


# ── helpers ────────────────────────────────────────────────────────


def _make_task(
    task_id: str = "task-1",
    name: str = "Buy groceries",
    note: str = "",
    status: str = "active",
    flagged: bool = False,
    tags: list[str] | None = None,
    project: str = "",
    parent_task: str = "",
    **kwargs: Any,
) -> dict[str, Any]:
    """Create a minimal OmniFocus task dict."""
    return {
        "id": task_id,
        "name": name,
        "note": note,
        "status": status,
        "flagged": flagged,
        "tags": tags or [],
        "project": project,
        "parent_task": parent_task,
        "creation_date": kwargs.get("creation_date"),
        "modification_date": kwargs.get("modification_date"),
        "completion_date": kwargs.get("completion_date"),
        "due_date": kwargs.get("due_date"),
        "defer_date": kwargs.get("defer_date"),
    }


async def _collect_events(
    queue: asyncio.Queue[dict[str, Any]], timeout: float = 2.0
) -> list[dict[str, Any]]:
    """Drain all events from *queue* until *timeout* elapses."""
    events: list[dict[str, Any]] = []
    try:
        while True:
            ev = await asyncio.wait_for(queue.get(), timeout=timeout)
            events.append(ev)
    except (asyncio.TimeoutError, TimeoutError):
        pass
    return events


# ── _task_hash ────────────────────────────────────────────────────


class TestTaskHash:
    def test_deterministic(self) -> None:
        t = _make_task()
        assert _task_hash(t) == _task_hash(t)

    def test_differs_on_name_change(self) -> None:
        t1 = _make_task(name="a")
        t2 = _make_task(name="b")
        assert _task_hash(t1) != _task_hash(t2)

    def test_differs_on_status_change(self) -> None:
        t1 = _make_task(status="active")
        t2 = _make_task(status="completed")
        assert _task_hash(t1) != _task_hash(t2)

    def test_differs_on_tag_change(self) -> None:
        t1 = _make_task(tags=["work"])
        t2 = _make_task(tags=["personal"])
        assert _task_hash(t1) != _task_hash(t2)

    def test_tag_order_independent(self) -> None:
        t1 = _make_task(tags=["a", "b"])
        t2 = _make_task(tags=["b", "a"])
        assert _task_hash(t1) == _task_hash(t2)

    def test_sha256_hex(self) -> None:
        h = _task_hash(_make_task())
        assert len(h) == 64


# ── state persistence ─────────────────────────────────────────────


class TestStatePersistence:
    def test_load_missing_file(self, tmp_path: Path) -> None:
        assert _load_state(tmp_path / "missing.json") == {}

    def test_save_and_load(self, tmp_path: Path) -> None:
        p = tmp_path / "state.json"
        _save_state(p, {"task-1": "abc123"})
        loaded = _load_state(p)
        assert loaded == {"task-1": "abc123"}

    def test_load_corrupt_json(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text("{{{not json")
        assert _load_state(p) == {}

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        p = tmp_path / "sub" / "dir" / "state.json"
        _save_state(p, {"x": "y"})
        assert p.exists()

    def test_save_sets_restrictive_permissions(self, tmp_path: Path) -> None:
        p = tmp_path / "state.json"
        _save_state(p, {"task-1": "abc"})
        assert p.stat().st_mode & 0o777 == 0o600


# ── _build_event ──────────────────────────────────────────────────


class TestBuildEvent:
    def test_event_structure(self) -> None:
        task = _make_task(task_id="abc-123", name="Test task")
        event = _build_event(task, "created")
        assert event["source_type"] == "omnifocus"
        assert event["source_id"] == "omnifocus://abc-123"
        assert event["operation"] == "created"
        assert event["meta"] == task
        assert "id" in event
        assert "enqueued_at" in event

    def test_deleted_event(self) -> None:
        task = _make_task(task_id="del-1")
        event = _build_event(task, "deleted")
        assert event["operation"] == "deleted"
        assert event["source_id"] == "omnifocus://del-1"


# ── OmniFocusSource configure ───────────────────────────────────


class TestOmniFocusSourceConfigure:
    def test_name(self) -> None:
        assert OmniFocusSource().name() == "omnifocus"

    def test_poll_interval(self) -> None:
        s = OmniFocusSource()
        s.configure({"poll_interval_seconds": 600})
        assert s._poll_interval == 600

    def test_default_poll_interval(self) -> None:
        s = OmniFocusSource()
        s.configure({})
        assert s._poll_interval == DEFAULT_POLL_INTERVAL

    def test_explicit_enabled(self) -> None:
        s = OmniFocusSource()
        s.configure({"enabled": True})
        assert s._enabled is True

    def test_explicit_disabled(self) -> None:
        s = OmniFocusSource()
        s.configure({"enabled": False})
        assert s._enabled is False

    @patch("worker.sources.omnifocus._is_macos", return_value=False)
    def test_disabled_on_non_macos(self, _mock: Any) -> None:
        s = OmniFocusSource()
        s.configure({})
        assert s._enabled is False

    def test_custom_state_path(self, tmp_path: Path) -> None:
        s = OmniFocusSource()
        s.configure({"state_path": str(tmp_path / "of.json")})
        assert s._state_path == tmp_path / "of.json"


# ── OmniFocusSource start (integration) ──────────────────────────


@pytest.mark.asyncio
async def test_initial_poll_emits_created_events(tmp_path: Path) -> None:
    """Initial poll should emit 'created' events for all tasks."""
    tasks = [
        _make_task(task_id="t1", name="Task 1"),
        _make_task(task_id="t2", name="Task 2", tags=["People/Boss"]),
    ]

    s = OmniFocusSource()
    s.configure(
        {
            "enabled": True,
            "state_path": str(tmp_path / "state.json"),
            "poll_interval_seconds": 3600,
        }
    )

    with patch("worker.sources.omnifocus._fetch_tasks", return_value=tasks):
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        task = asyncio.create_task(s.start(q))
        events = await _collect_events(q)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert len(events) == 2
    ops = {e["operation"] for e in events}
    assert ops == {"created"}
    source_ids = {e["source_id"] for e in events}
    assert "omnifocus://t1" in source_ids
    assert "omnifocus://t2" in source_ids


@pytest.mark.asyncio
async def test_modified_task_emits_modified_event(tmp_path: Path) -> None:
    """Status change should emit a 'modified' event."""
    task_v1 = _make_task(task_id="t1", name="Task 1", status="active")
    task_v2 = _make_task(task_id="t1", name="Task 1", status="completed")

    state_path = tmp_path / "state.json"
    # Pre-seed state with v1 hash
    _save_state(state_path, {"t1": _task_hash(task_v1)})

    s = OmniFocusSource()
    s.configure(
        {
            "enabled": True,
            "state_path": str(state_path),
            "poll_interval_seconds": 3600,
        }
    )

    with patch("worker.sources.omnifocus._fetch_tasks", return_value=[task_v2]):
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        task = asyncio.create_task(s.start(q))
        events = await _collect_events(q)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert len(events) == 1
    assert events[0]["operation"] == "modified"
    assert events[0]["meta"]["status"] == "completed"


@pytest.mark.asyncio
async def test_deleted_task_emits_deleted_event(tmp_path: Path) -> None:
    """Task disappearing from OmniFocus should emit a 'deleted' event."""
    task = _make_task(task_id="gone")
    state_path = tmp_path / "state.json"
    _save_state(state_path, {"gone": _task_hash(task)})

    s = OmniFocusSource()
    s.configure(
        {
            "enabled": True,
            "state_path": str(state_path),
            "poll_interval_seconds": 3600,
        }
    )

    with patch("worker.sources.omnifocus._fetch_tasks", return_value=[]):
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        t = asyncio.create_task(s.start(q))
        events = await _collect_events(q)
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass

    assert len(events) == 1
    assert events[0]["operation"] == "deleted"
    assert events[0]["source_id"] == "omnifocus://gone"


@pytest.mark.asyncio
async def test_unchanged_tasks_emit_nothing(tmp_path: Path) -> None:
    """Unchanged tasks should not produce events."""
    task = _make_task(task_id="same")
    state_path = tmp_path / "state.json"
    _save_state(state_path, {"same": _task_hash(task)})

    s = OmniFocusSource()
    s.configure(
        {
            "enabled": True,
            "state_path": str(state_path),
            "poll_interval_seconds": 3600,
        }
    )

    with patch("worker.sources.omnifocus._fetch_tasks", return_value=[task]):
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        t = asyncio.create_task(s.start(q))
        events = await _collect_events(q, timeout=1.0)
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass

    assert len(events) == 0


@pytest.mark.asyncio
async def test_disabled_source_emits_nothing(tmp_path: Path) -> None:
    """Disabled source should not poll or emit."""
    s = OmniFocusSource()
    s.configure({"enabled": False})

    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    t = asyncio.create_task(s.start(q))
    events = await _collect_events(q, timeout=0.5)
    t.cancel()
    try:
        await t
    except asyncio.CancelledError:
        pass

    assert len(events) == 0


@pytest.mark.asyncio
async def test_fetch_failure_does_not_crash(tmp_path: Path) -> None:
    """JXA failure should log error but keep polling."""
    s = OmniFocusSource()
    s.configure(
        {
            "enabled": True,
            "state_path": str(tmp_path / "state.json"),
            "poll_interval_seconds": 3600,
        }
    )

    with patch(
        "worker.sources.omnifocus._fetch_tasks",
        side_effect=RuntimeError("osascript failed"),
    ):
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        t = asyncio.create_task(s.start(q))
        events = await _collect_events(q, timeout=1.0)
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass

    assert len(events) == 0
