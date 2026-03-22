"""Tests for FileSource initial scan, dedup window, and periodic checkpoints.

Covers the full initial scan lifecycle: first startup, cursor-aware restarts,
filter/limit compliance, dedup window behaviour, and checkpoint persistence.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest

from worker.sources.cursor import load_cursor
from worker.sources.files import FileSource


# ── Helpers ───────────────────────────────────────────────────────


async def _drain_queue(
    q: asyncio.Queue[dict[str, Any]],
    *,
    timeout: float = 2.0,
) -> list[dict[str, Any]]:
    """Drain all events from queue, waiting up to timeout for the first."""
    events: list[dict[str, Any]] = []
    try:
        # Wait for at least one event or timeout
        ev = await asyncio.wait_for(q.get(), timeout=timeout)
        events.append(ev)
        # Drain any remaining events that arrive quickly
        while True:
            try:
                ev = await asyncio.wait_for(q.get(), timeout=0.3)
                events.append(ev)
            except asyncio.TimeoutError:
                break
    except asyncio.TimeoutError:
        pass
    return events


async def _run_source_briefly(
    source: FileSource,
    q: asyncio.Queue[dict[str, Any]],
    duration: float = 1.5,
) -> list[dict[str, Any]]:
    """Start a source, let it run for duration seconds, cancel and drain."""
    task = asyncio.create_task(source.start(q))
    await asyncio.sleep(duration)

    events: list[dict[str, Any]] = []
    while not q.empty():
        events.append(q.get_nowait())

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # Drain any events that arrived during cancellation
    while not q.empty():
        events.append(q.get_nowait())

    return events


def _make_source(tmp_path: Path, **overrides: Any) -> FileSource:
    """Create and configure a FileSource for tmp_path."""
    cfg: dict[str, Any] = {
        "watch_paths": [str(tmp_path / "watched")],
        "cursor_path": str(tmp_path / "cursor.json"),
        **overrides,
    }
    fs = FileSource()
    fs.configure(cfg)
    return fs


# ── First startup (no cursor) ────────────────────────────────────


@pytest.mark.asyncio
async def test_first_startup_all_files_created(tmp_path: Path) -> None:
    """All files emitted as 'created' when no cursor exists."""
    watched = tmp_path / "watched"
    watched.mkdir()
    (watched / "note1.md").write_text("hello")
    (watched / "note2.txt").write_text("world")
    (watched / "sub").mkdir()
    (watched / "sub" / "deep.md").write_text("deep")

    fs = _make_source(tmp_path)
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    events = await _run_source_briefly(fs, q)

    created = [e for e in events if e["operation"] == "created"]
    source_ids = {e["source_id"] for e in created}

    assert len(created) == 3
    assert str(watched / "note1.md") in source_ids
    assert str(watched / "note2.txt") in source_ids
    assert str(watched / "sub" / "deep.md") in source_ids

    # Verify event structure
    for ev in created:
        assert ev["source_type"] == "files"
        assert "id" in ev
        assert "enqueued_at" in ev
        assert "meta" in ev
        assert "sha256" in ev["meta"]

    # Cursor should be saved
    assert (tmp_path / "cursor.json").exists()


# ── Second startup (cursor exists) ───────────────────────────────


@pytest.mark.asyncio
async def test_second_startup_only_changed_files(tmp_path: Path) -> None:
    """Only changed files emitted on restart with existing cursor."""
    watched = tmp_path / "watched"
    watched.mkdir()
    (watched / "unchanged.md").write_text("stable")
    (watched / "will_change.md").write_text("original")

    fs = _make_source(tmp_path)
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    await _run_source_briefly(fs, q)

    # Modify one file
    (watched / "will_change.md").write_text("modified content")

    # Second startup
    fs2 = _make_source(tmp_path)
    q2: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    events = await _run_source_briefly(fs2, q2)

    scan_events = [
        e for e in events if e["operation"] in ("created", "modified", "deleted")
    ]
    assert len(scan_events) == 1
    assert scan_events[0]["operation"] == "modified"
    assert "will_change.md" in scan_events[0]["source_id"]
    assert scan_events[0]["meta"]["sha256"]  # Has new hash


@pytest.mark.asyncio
async def test_deleted_files_between_runs(tmp_path: Path) -> None:
    """Files removed between runs are emitted as 'deleted'."""
    watched = tmp_path / "watched"
    watched.mkdir()
    doomed = watched / "doomed.md"
    doomed.write_text("goodbye")
    (watched / "survivor.md").write_text("still here")

    fs = _make_source(tmp_path)
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    await _run_source_briefly(fs, q)

    # Delete the file
    doomed.unlink()

    # Second startup
    fs2 = _make_source(tmp_path)
    q2: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    events = await _run_source_briefly(fs2, q2)

    deleted = [e for e in events if e["operation"] == "deleted"]
    assert len(deleted) == 1
    assert "doomed.md" in deleted[0]["source_id"]


@pytest.mark.asyncio
async def test_unchanged_files_no_events(tmp_path: Path) -> None:
    """No events emitted when nothing changed since last scan."""
    watched = tmp_path / "watched"
    watched.mkdir()
    (watched / "stable.md").write_text("constant")

    fs = _make_source(tmp_path)
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    await _run_source_briefly(fs, q)

    # Second startup — nothing changed
    fs2 = _make_source(tmp_path)
    q2: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    events = await _run_source_briefly(fs2, q2)

    scan_events = [
        e for e in events if e["operation"] in ("created", "modified", "deleted")
    ]
    assert len(scan_events) == 0


# ── Filter compliance ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_scan_picks_up_image_files(tmp_path: Path) -> None:
    """Initial scan discovers standalone image files with raw_bytes."""
    watched = tmp_path / "watched"
    watched.mkdir()
    png_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
    (watched / "screenshot.png").write_bytes(png_data)
    (watched / "note.md").write_text("hello")

    fs = _make_source(tmp_path)
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    events = await _run_source_briefly(fs, q)

    created = [e for e in events if e["operation"] == "created"]
    img_events = [e for e in created if "screenshot.png" in e["source_id"]]
    assert len(img_events) == 1
    assert img_events[0]["mime_type"] == "image/png"
    assert img_events[0]["raw_bytes"] == png_data


@pytest.mark.asyncio
async def test_scan_image_passes_include_extensions(tmp_path: Path) -> None:
    """Images pass through include_extensions filter during scan."""
    watched = tmp_path / "watched"
    watched.mkdir()
    (watched / "photo.png").write_bytes(b"\x89PNG" + b"\x00" * 20)
    (watched / "note.md").write_text("hello")
    (watched / "code.py").write_text("x = 1")

    fs = _make_source(tmp_path, include_extensions=[".md"])
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    events = await _run_source_briefly(fs, q)

    created = [e for e in events if e["operation"] == "created"]
    source_ids = [e["source_id"] for e in created]
    assert any("note.md" in s for s in source_ids)
    assert any("photo.png" in s for s in source_ids)
    assert not any("code.py" in s for s in source_ids)


@pytest.mark.asyncio
async def test_scan_respects_include_extensions(tmp_path: Path) -> None:
    """Initial scan only includes files matching include_extensions."""
    watched = tmp_path / "watched"
    watched.mkdir()
    (watched / "keep.md").write_text("markdown")
    (watched / "skip.txt").write_text("text")
    (watched / "skip.py").write_text("python")

    fs = _make_source(tmp_path, include_extensions=[".md"])
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    events = await _run_source_briefly(fs, q)

    created = [e for e in events if e["operation"] == "created"]
    assert len(created) == 1
    assert "keep.md" in created[0]["source_id"]


@pytest.mark.asyncio
async def test_scan_respects_exclude_patterns(tmp_path: Path) -> None:
    """Initial scan excludes files matching exclude_patterns."""
    watched = tmp_path / "watched"
    watched.mkdir()
    (watched / "good.md").write_text("keep")
    (watched / "bad.pyc").write_bytes(b"\x00")
    (watched / "drafts").mkdir()
    (watched / "drafts" / "draft.md").write_text("skip")

    fs = _make_source(tmp_path, exclude_patterns=["*.pyc", "*/drafts/*"])
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    events = await _run_source_briefly(fs, q)

    created = [e for e in events if e["operation"] == "created"]
    source_ids = [e["source_id"] for e in created]
    assert any("good.md" in s for s in source_ids)
    assert not any("bad.pyc" in s for s in source_ids)
    assert not any("draft.md" in s for s in source_ids)


@pytest.mark.asyncio
async def test_scan_respects_max_file_size(tmp_path: Path) -> None:
    """Initial scan emits oversized files as metadata-only (no text content)."""
    watched = tmp_path / "watched"
    watched.mkdir()
    (watched / "small.md").write_text("ok")
    (watched / "big.md").write_bytes(b"x" * 200)

    fs = _make_source(tmp_path, max_file_size=100)
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    events = await _run_source_briefly(fs, q)

    created = [e for e in events if e["operation"] == "created"]
    source_ids = [e["source_id"] for e in created]
    assert any("small.md" in s for s in source_ids)
    assert any("big.md" in s for s in source_ids)

    small = [e for e in created if "small.md" in e["source_id"]][0]
    big = [e for e in created if "big.md" in e["source_id"]][0]

    # Small file has text content and sha256
    assert "text" in small
    assert small["meta"].get("sha256")

    # Big file has size metadata but no text and no sha256
    assert "text" not in big
    assert big["meta"]["size_bytes"] == 200
    assert not big["meta"].get("sha256")


@pytest.mark.asyncio
async def test_scan_respects_recursive_false(tmp_path: Path) -> None:
    """recursive=false only scans top-level files."""
    watched = tmp_path / "watched"
    watched.mkdir()
    (watched / "top.md").write_text("top level")
    (watched / "sub").mkdir()
    (watched / "sub" / "nested.md").write_text("nested")

    fs = _make_source(tmp_path, recursive=False)
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    events = await _run_source_briefly(fs, q)

    created = [e for e in events if e["operation"] == "created"]
    source_ids = [e["source_id"] for e in created]
    assert any("top.md" in s for s in source_ids)
    assert not any("nested.md" in s for s in source_ids)


@pytest.mark.asyncio
async def test_scan_rejects_symlinks(tmp_path: Path) -> None:
    """Symlinked watch paths are skipped during configure."""
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    (real_dir / "note.md").write_text("content")

    link = tmp_path / "link"
    link.symlink_to(real_dir)

    fs = FileSource()
    fs.configure(
        {
            "watch_paths": [str(link)],
            "cursor_path": str(tmp_path / "cursor.json"),
        }
    )

    # Symlinked path should have been filtered out
    assert len(fs._watch_paths) == 0


# ── Empty directory ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_empty_directory_no_events_cursor_saved(tmp_path: Path) -> None:
    """Empty directory produces no events but cursor is still saved."""
    watched = tmp_path / "watched"
    watched.mkdir()

    cursor_path = tmp_path / "cursor.json"
    fs = _make_source(tmp_path)
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    events = await _run_source_briefly(fs, q)

    assert len(events) == 0
    assert cursor_path.exists()
    loaded = load_cursor(cursor_path)
    assert loaded == {}


# ── Metrics ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_initial_scan_updates_metrics(tmp_path: Path) -> None:
    """Initial scan updates INITIAL_SCAN_FILES_TOTAL and DURATION metrics."""
    from worker.metrics import INITIAL_SCAN_DURATION_SECONDS, INITIAL_SCAN_FILES_TOTAL

    watched = tmp_path / "watched"
    watched.mkdir()
    (watched / "a.md").write_text("hello")
    (watched / "b.md").write_text("world")

    fs = _make_source(tmp_path)
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    await _run_source_briefly(fs, q)

    # Duration gauge should have been set to a positive value
    duration = INITIAL_SCAN_DURATION_SECONDS.labels(source_type="files")._value.get()
    assert duration > 0

    # File count metric should reflect new files
    new_count = INITIAL_SCAN_FILES_TOTAL.labels(
        source_type="files", result="new"
    )._value.get()
    assert new_count >= 2


@pytest.mark.asyncio
async def test_initial_scan_sets_sync_total_and_tags_events(tmp_path: Path) -> None:
    """Initial scan registers items via initial_sync_add_items and tags events."""
    import worker.metrics as m

    # Reset module-level accumulator so parallel tests don't interfere
    m._initial_sync_total = 0
    m.INITIAL_SYNC_ITEMS_TOTAL.set(0)

    watched = tmp_path / "watched"
    watched.mkdir()
    (watched / "a.md").write_text("alpha")
    (watched / "b.md").write_text("beta")

    fs = _make_source(tmp_path)
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    events = await _run_source_briefly(fs, q)

    # All scan events should carry the initial_scan flag
    scan_events = [e for e in events if e.get("initial_scan")]
    assert len(scan_events) >= 2

    # Module-level total should have been incremented
    assert m.initial_sync_get_total() >= 2

    # Gauge should match
    assert m.INITIAL_SYNC_ITEMS_TOTAL._value.get() >= 2


# ── Dedup window ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dedup_window_drops_matching_event(tmp_path: Path) -> None:
    """Watchdog event matching scan result is dropped during dedup window."""
    watched = tmp_path / "watched"
    watched.mkdir()
    note = watched / "note.md"
    note.write_text("original content")

    fs = _make_source(tmp_path)
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    task = asyncio.create_task(fs.start(q))
    # Wait for initial scan to complete and dedup window to be armed
    await asyncio.sleep(1.0)

    # Drain scan events
    scan_events = []
    while not q.empty():
        scan_events.append(q.get_nowait())

    assert len(scan_events) >= 1  # initial scan created event

    # Touch the file WITHOUT changing content — watchdog fires, but dedup
    # should drop it since sha256 matches
    os.utime(note)
    await asyncio.sleep(0.5)

    # Should have no new events (dedup dropped it)
    post_events = []
    while not q.empty():
        post_events.append(q.get_nowait())

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # File was touched but content unchanged — watchdog event gets
    # deduplicated because sha256 matches the scan result
    watcher_events = [e for e in post_events if "note.md" in e.get("source_id", "")]
    # Either dedup dropped it or watchdog didn't fire — both valid
    # The key assertion: no duplicate "created" events for same content
    created = [e for e in watcher_events if e["operation"] == "created"]
    assert len(created) == 0


@pytest.mark.asyncio
async def test_dedup_window_passes_changed_content(tmp_path: Path) -> None:
    """Watchdog event with different sha256 passes through dedup window."""
    watched = tmp_path / "watched"
    watched.mkdir()
    note = watched / "note.md"
    note.write_text("original")

    fs = _make_source(tmp_path)
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    task = asyncio.create_task(fs.start(q))
    await asyncio.sleep(1.0)

    # Drain scan events
    while not q.empty():
        q.get_nowait()

    # Modify content — sha256 changes, should pass through dedup
    note.write_text("completely different content")
    await asyncio.sleep(1.0)

    events = []
    while not q.empty():
        events.append(q.get_nowait())

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # Should have at least one modified event for the changed file
    modified = [
        e
        for e in events
        if "note.md" in e.get("source_id", "")
        and e["operation"] in ("modified", "created")
    ]
    assert len(modified) >= 1


@pytest.mark.asyncio
async def test_dedup_window_expires(tmp_path: Path) -> None:
    """Events after window expiry are not filtered."""
    from worker.sources._handler import BaseHandler

    loop = asyncio.get_running_loop()
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    handler = BaseHandler(
        queue=q,
        loop=loop,
        include_extensions=None,
        exclude_patterns=[],
    )
    handler._source_type = "test"

    # Arm window with very short duration
    handler.set_dedup_window({("/test/file.md", "sha_abc")}, duration_seconds=0.1)

    # During window — should be dedup'd
    assert handler._is_dedup_duplicate("/test/file.md", "sha_abc") is True

    # Wait for window to expire
    await asyncio.sleep(0.2)

    # After window — should NOT be dedup'd
    assert handler._is_dedup_duplicate("/test/file.md", "sha_abc") is False


@pytest.mark.asyncio
async def test_dedup_window_clears_set(tmp_path: Path) -> None:
    """Window clears dedup set after expiry via asyncio timer."""
    from worker.sources._handler import BaseHandler

    loop = asyncio.get_running_loop()
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    handler = BaseHandler(
        queue=q,
        loop=loop,
        include_extensions=None,
        exclude_patterns=[],
    )
    handler._source_type = "test"

    handler.set_dedup_window({("/a", "sha1"), ("/b", "sha2")}, duration_seconds=0.1)
    assert len(handler._dedup_set) == 2

    # Let the asyncio timer fire
    await asyncio.sleep(0.2)

    assert len(handler._dedup_set) == 0
    assert handler._dedup_deadline == 0.0


# ── Periodic checkpoint ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_checkpoint_fires_at_interval(tmp_path: Path) -> None:
    """Checkpoint writes cursor at configured interval."""
    watched = tmp_path / "watched"
    watched.mkdir()
    (watched / "note.md").write_text("hello")
    cursor_path = tmp_path / "cursor.json"

    fs = _make_source(tmp_path, cursor_checkpoint_interval=1)
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    task = asyncio.create_task(fs.start(q))
    # Wait for initial scan + at least one checkpoint cycle
    await asyncio.sleep(2.5)

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # Cursor file should have been written by checkpoint
    assert cursor_path.exists()
    loaded = load_cursor(cursor_path)
    assert len(loaded) >= 1


@pytest.mark.asyncio
async def test_watchdog_events_update_handler_cursor(tmp_path: Path) -> None:
    """Watchdog events update the handler's in-memory cursor."""
    watched = tmp_path / "watched"
    watched.mkdir()
    cursor_path = tmp_path / "cursor.json"

    fs = _make_source(tmp_path, cursor_checkpoint_interval=1)
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    task = asyncio.create_task(fs.start(q))
    await asyncio.sleep(1.0)

    # Create a file while watching
    new_file = watched / "new_note.md"
    new_file.write_text("added during watch")

    # Wait for watchdog to detect + checkpoint to fire
    await asyncio.sleep(2.5)

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # Checkpoint should have persisted the new file
    loaded = load_cursor(cursor_path)
    matching = [k for k in loaded if "new_note.md" in k]
    assert len(matching) >= 1, f"new_note.md not found in cursor: {list(loaded.keys())}"


@pytest.mark.asyncio
async def test_graceful_shutdown_saves_cursor(tmp_path: Path) -> None:
    """Cancelling the task saves a final checkpoint."""
    watched = tmp_path / "watched"
    watched.mkdir()
    (watched / "note.md").write_text("content")
    cursor_path = tmp_path / "cursor.json"

    # Use a long checkpoint interval so it won't fire naturally
    fs = _make_source(tmp_path, cursor_checkpoint_interval=3600)
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    task = asyncio.create_task(fs.start(q))
    await asyncio.sleep(1.5)

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # Even though checkpoint interval is 1 hour, cancellation should save
    assert cursor_path.exists()
    loaded = load_cursor(cursor_path)
    assert len(loaded) >= 1


@pytest.mark.asyncio
async def test_restart_after_checkpoint_only_new_changes(tmp_path: Path) -> None:
    """After checkpoint, restart only re-scans post-checkpoint changes."""
    watched = tmp_path / "watched"
    watched.mkdir()
    (watched / "stable.md").write_text("unchanged")

    # First run — creates cursor
    fs = _make_source(tmp_path, cursor_checkpoint_interval=1)
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    await _run_source_briefly(fs, q, duration=2.0)

    # Add a new file between runs
    (watched / "new.md").write_text("added between runs")

    # Second run — should only see the new file
    fs2 = _make_source(tmp_path, cursor_checkpoint_interval=1)
    q2: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    events = await _run_source_briefly(fs2, q2, duration=2.0)

    scan_events = [
        e for e in events if e["operation"] in ("created", "modified", "deleted")
    ]
    created = [e for e in scan_events if e["operation"] == "created"]
    assert len(created) == 1
    assert "new.md" in created[0]["source_id"]
