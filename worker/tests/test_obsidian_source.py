"""Tests for the Obsidian vault source shim."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from worker.sources.obsidian import ObsidianSource, _scan_vault, discover_vaults


# ── discover_vaults ─────────────────────────────────────────────────


def test_discover_vaults_finds_direct_vault(tmp_path: Path):
    (tmp_path / ".obsidian").mkdir()
    vaults = discover_vaults([tmp_path])
    assert vaults == [tmp_path]


def test_discover_vaults_finds_nested_vaults(tmp_path: Path):
    (tmp_path / "vault_a" / ".obsidian").mkdir(parents=True)
    (tmp_path / "vault_b" / ".obsidian").mkdir(parents=True)
    (tmp_path / "not_a_vault").mkdir()
    vaults = discover_vaults([tmp_path])
    names = [v.name for v in vaults]
    assert "vault_a" in names
    assert "vault_b" in names
    assert "not_a_vault" not in names


def test_discover_vaults_skips_missing_path(tmp_path: Path):
    missing = tmp_path / "nonexistent"
    vaults = discover_vaults([missing])
    assert vaults == []


# ── ObsidianSource configure ───────────────────────────────────────


def test_obsidian_source_name():
    s = ObsidianSource()
    assert s.name() == "obsidian"


def test_obsidian_source_requires_vault_paths():
    s = ObsidianSource()
    with pytest.raises(ValueError, match="vault_paths"):
        s.configure({})


def test_obsidian_source_configure_basic(tmp_path: Path):
    s = ObsidianSource()
    s.configure({"vault_paths": [str(tmp_path)]})
    assert s._vault_paths == [tmp_path]
    assert s._include_extensions is None
    assert s._exclude_patterns == []
    assert s._recursive is True


def test_obsidian_source_configure_extensions(tmp_path: Path):
    s = ObsidianSource()
    s.configure(
        {
            "vault_paths": [str(tmp_path)],
            "include_extensions": [".md", "canvas"],
        }
    )
    assert s._include_extensions == {".md", ".canvas"}


# ── ObsidianSource watcher integration ─────────────────────────────


@pytest.mark.asyncio
async def test_obsidian_source_detects_create(tmp_path: Path):
    vault = tmp_path / "my_vault"
    vault.mkdir()
    (vault / ".obsidian").mkdir()

    s = ObsidianSource()
    s.configure(
        {
            "vault_paths": [str(tmp_path)],
            "cursor_path": str(tmp_path / "cursor.json"),
        }
    )
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    task = asyncio.create_task(s.start(q))
    await asyncio.sleep(0.5)

    test_file = vault / "note.md"
    test_file.write_text("# Hello")

    events = []
    try:
        for _ in range(10):
            await asyncio.sleep(0.3)
            while not q.empty():
                events.append(q.get_nowait())
            if any("note.md" in e["source_id"] for e in events):
                break
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    note_events = [e for e in events if "note.md" in e["source_id"]]
    assert len(note_events) >= 1
    ev = note_events[0]
    assert ev["source_type"] == "obsidian"
    assert ev["operation"] in ("created", "modified")
    assert ev["mime_type"] == "text/markdown"
    assert ev["meta"]["vault_name"] == "my_vault"
    assert ev["meta"]["vault_path"] == str(vault)
    assert ev["meta"]["relative_path"] == "note.md"


@pytest.mark.asyncio
async def test_obsidian_source_skips_dotobsidian_files(tmp_path: Path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / ".obsidian").mkdir()

    s = ObsidianSource()
    s.configure(
        {
            "vault_paths": [str(tmp_path)],
            "cursor_path": str(tmp_path / "cursor.json"),
        }
    )
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    task = asyncio.create_task(s.start(q))
    await asyncio.sleep(0.5)

    # Write to .obsidian config — should be skipped
    (vault / ".obsidian" / "app.json").write_text("{}")
    await asyncio.sleep(0.5)

    # Write a real note — should be captured
    (vault / "real.md").write_text("content")

    events = []
    try:
        for _ in range(10):
            await asyncio.sleep(0.3)
            while not q.empty():
                events.append(q.get_nowait())
            if any("real.md" in e["source_id"] for e in events):
                break
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    source_ids = [e["source_id"] for e in events]
    assert any("real.md" in sid for sid in source_ids)
    assert not any(".obsidian" in sid for sid in source_ids)


@pytest.mark.asyncio
async def test_obsidian_source_no_vaults_found(tmp_path: Path):
    """Source should start and be cancellable even with no vaults."""
    s = ObsidianSource()
    s.configure(
        {
            "vault_paths": [str(tmp_path)],
            "cursor_path": str(tmp_path / "cursor.json"),
        }
    )
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    task = asyncio.create_task(s.start(q))
    await asyncio.sleep(0.3)

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert q.empty()


# ── _scan_vault ────────────────────────────────────────────────────


def test_scan_vault_finds_files(tmp_path: Path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / ".obsidian").mkdir()
    (vault / "note.md").write_text("# Hello")
    (vault / "sub").mkdir()
    (vault / "sub" / "deep.md").write_text("# Deep")

    entries = _scan_vault(vault, None, [], 100 * 1024 * 1024)
    assert len(entries) == 2
    assert str(vault / "note.md") in entries
    assert str(vault / "sub" / "deep.md") in entries
    for entry in entries.values():
        assert entry.sha256
        assert entry.mtime_ns > 0
        assert entry.size > 0


def test_scan_vault_skips_dotobsidian(tmp_path: Path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / ".obsidian").mkdir()
    (vault / ".obsidian" / "app.json").write_text("{}")
    (vault / "note.md").write_text("content")

    entries = _scan_vault(vault, None, [], 100 * 1024 * 1024)
    paths = list(entries.keys())
    assert len(paths) == 1
    assert "note.md" in paths[0]
    assert ".obsidian" not in paths[0]


def test_scan_vault_respects_include_extensions(tmp_path: Path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / ".obsidian").mkdir()
    (vault / "note.md").write_text("markdown")
    (vault / "image.png").write_bytes(b"\x89PNG")
    (vault / "data.json").write_text("{}")

    entries = _scan_vault(vault, {".md"}, [], 100 * 1024 * 1024)
    assert len(entries) == 1
    assert str(vault / "note.md") in entries


def test_scan_vault_respects_exclude_patterns(tmp_path: Path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / ".obsidian").mkdir()
    (vault / "note.md").write_text("keep")
    (vault / "draft.md").write_text("skip")

    entries = _scan_vault(vault, None, ["*draft*"], 100 * 1024 * 1024)
    assert len(entries) == 1
    assert str(vault / "note.md") in entries


def test_scan_vault_skips_oversized_files(tmp_path: Path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / ".obsidian").mkdir()
    (vault / "small.md").write_text("ok")
    (vault / "big.md").write_text("x" * 200)

    entries = _scan_vault(vault, None, [], 100)  # max 100 bytes
    assert len(entries) == 1
    assert str(vault / "small.md") in entries


# ── Initial scan integration ──────────────────────────────────────


@pytest.mark.asyncio
async def test_initial_scan_first_startup(tmp_path: Path):
    """All files should be emitted as 'created' on first startup."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / ".obsidian").mkdir()
    (vault / "note1.md").write_text("# Note 1")
    (vault / "note2.md").write_text("# Note 2")

    cursor_path = tmp_path / "cursor.json"

    s = ObsidianSource()
    s.configure(
        {
            "vault_paths": [str(tmp_path)],
            "cursor_path": str(cursor_path),
        }
    )
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    task = asyncio.create_task(s.start(q))
    # Give the scan time to complete and events to be enqueued
    await asyncio.sleep(1.0)

    events = []
    while not q.empty():
        events.append(q.get_nowait())

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # Filter to scan events (created from initial scan)
    created = [e for e in events if e["operation"] == "created"]
    assert len(created) == 2
    source_ids = {e["source_id"] for e in created}
    assert str(vault / "note1.md") in source_ids
    assert str(vault / "note2.md") in source_ids

    # Verify vault metadata
    for ev in created:
        assert ev["source_type"] == "obsidian"
        assert ev["meta"]["vault_name"] == "vault"
        assert ev["meta"]["vault_path"] == str(vault)
        assert "relative_path" in ev["meta"]
        assert ev["meta"]["sha256"]

    # Verify cursor was saved
    assert cursor_path.exists()


@pytest.mark.asyncio
async def test_initial_scan_detects_modifications(tmp_path: Path):
    """Modified files should be detected on subsequent startup."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / ".obsidian").mkdir()

    note = vault / "note.md"
    note.write_text("original")

    cursor_path = tmp_path / "cursor.json"

    # First scan — builds cursor
    s = ObsidianSource()
    s.configure(
        {
            "vault_paths": [str(tmp_path)],
            "cursor_path": str(cursor_path),
        }
    )
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    task = asyncio.create_task(s.start(q))
    await asyncio.sleep(1.0)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # Drain first scan events
    while not q.empty():
        q.get_nowait()

    # Modify the file
    note.write_text("modified content")

    # Second scan — should detect modification
    s2 = ObsidianSource()
    s2.configure(
        {
            "vault_paths": [str(tmp_path)],
            "cursor_path": str(cursor_path),
        }
    )
    q2: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    task2 = asyncio.create_task(s2.start(q2))
    await asyncio.sleep(1.0)
    task2.cancel()
    try:
        await task2
    except asyncio.CancelledError:
        pass

    events = []
    while not q2.empty():
        events.append(q2.get_nowait())

    modified = [e for e in events if e["operation"] == "modified"]
    assert len(modified) >= 1
    assert str(vault / "note.md") in modified[0]["source_id"]


@pytest.mark.asyncio
async def test_initial_scan_detects_deletions(tmp_path: Path):
    """Deleted files should be detected on subsequent startup."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / ".obsidian").mkdir()

    note = vault / "note.md"
    note.write_text("will be deleted")

    cursor_path = tmp_path / "cursor.json"

    # First scan
    s = ObsidianSource()
    s.configure(
        {
            "vault_paths": [str(tmp_path)],
            "cursor_path": str(cursor_path),
        }
    )
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    task = asyncio.create_task(s.start(q))
    await asyncio.sleep(1.0)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # Delete the file
    note.unlink()

    # Second scan — should detect deletion
    s2 = ObsidianSource()
    s2.configure(
        {
            "vault_paths": [str(tmp_path)],
            "cursor_path": str(cursor_path),
        }
    )
    q2: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    task2 = asyncio.create_task(s2.start(q2))
    await asyncio.sleep(1.0)
    task2.cancel()
    try:
        await task2
    except asyncio.CancelledError:
        pass

    events = []
    while not q2.empty():
        events.append(q2.get_nowait())

    deleted = [e for e in events if e["operation"] == "deleted"]
    assert len(deleted) >= 1
    assert str(vault / "note.md") in deleted[0]["source_id"]


@pytest.mark.asyncio
async def test_initial_scan_no_events_when_unchanged(tmp_path: Path):
    """No events should be emitted when nothing changed since last scan."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / ".obsidian").mkdir()
    (vault / "note.md").write_text("stable")

    cursor_path = tmp_path / "cursor.json"

    # First scan
    s = ObsidianSource()
    s.configure(
        {
            "vault_paths": [str(tmp_path)],
            "cursor_path": str(cursor_path),
        }
    )
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    task = asyncio.create_task(s.start(q))
    await asyncio.sleep(1.0)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # Drain
    while not q.empty():
        q.get_nowait()

    # Second scan — nothing changed
    s2 = ObsidianSource()
    s2.configure(
        {
            "vault_paths": [str(tmp_path)],
            "cursor_path": str(cursor_path),
        }
    )
    q2: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    task2 = asyncio.create_task(s2.start(q2))
    await asyncio.sleep(1.0)
    task2.cancel()
    try:
        await task2
    except asyncio.CancelledError:
        pass

    events = []
    while not q2.empty():
        events.append(q2.get_nowait())

    # No scan events should be emitted (only watcher events if any)
    scan_events = [
        e for e in events if e["operation"] in ("created", "modified", "deleted")
    ]
    assert len(scan_events) == 0


@pytest.mark.asyncio
async def test_initial_scan_cursor_path_config(tmp_path: Path):
    """cursor_path config should be respected."""
    s = ObsidianSource()
    custom_path = tmp_path / "custom" / "cursor.json"
    s.configure(
        {
            "vault_paths": [str(tmp_path)],
            "cursor_path": str(custom_path),
        }
    )
    assert s._cursor_path == custom_path
