"""Tests for ObsidianSource initial scan, multi-vault, and vault removal.

Covers vault-specific scan behaviour: multiple vaults scanned independently,
vault removal detection, .obsidian/ skipping, and vault metadata in events.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest

from worker.sources.cursor import FileEntry, load_cursor, save_cursor
from worker.sources.obsidian import ObsidianSource


# ── Helpers ───────────────────────────────────────────────────────


async def _run_source_briefly(
    source: ObsidianSource,
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

    # Drain any remaining
    while not q.empty():
        events.append(q.get_nowait())

    return events


def _make_vault(base: Path, name: str, files: dict[str, str] | None = None) -> Path:
    """Create an Obsidian vault with optional files."""
    vault = base / name
    vault.mkdir(parents=True, exist_ok=True)
    (vault / ".obsidian").mkdir(exist_ok=True)
    if files:
        for fname, content in files.items():
            fpath = vault / fname
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text(content)
    return vault


def _make_source(tmp_path: Path, **overrides: Any) -> ObsidianSource:
    """Create and configure an ObsidianSource."""
    cfg: dict[str, Any] = {
        "vault_paths": [str(tmp_path / "vaults")],
        "cursor_path": str(tmp_path / "cursor.json"),
        **overrides,
    }
    s = ObsidianSource()
    s.configure(cfg)
    return s


# ── Multiple vaults ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_multiple_vaults_scanned_independently(tmp_path: Path) -> None:
    """Each vault is scanned independently with correct metadata."""
    vaults_dir = tmp_path / "vaults"
    vaults_dir.mkdir()

    vault_a = _make_vault(vaults_dir, "vault_a", {"note_a.md": "hello from A"})
    vault_b = _make_vault(vaults_dir, "vault_b", {"note_b.md": "hello from B"})

    s = _make_source(tmp_path)
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    events = await _run_source_briefly(s, q)

    created = [e for e in events if e["operation"] == "created"]
    assert len(created) == 2

    # Check vault_a event
    a_events = [e for e in created if "note_a.md" in e["source_id"]]
    assert len(a_events) == 1
    assert a_events[0]["meta"]["vault_name"] == "vault_a"
    assert a_events[0]["meta"]["vault_path"] == str(vault_a)
    assert a_events[0]["meta"]["relative_path"] == "note_a.md"

    # Check vault_b event
    b_events = [e for e in created if "note_b.md" in e["source_id"]]
    assert len(b_events) == 1
    assert b_events[0]["meta"]["vault_name"] == "vault_b"
    assert b_events[0]["meta"]["vault_path"] == str(vault_b)
    assert b_events[0]["meta"]["relative_path"] == "note_b.md"


@pytest.mark.asyncio
async def test_multiple_vaults_independent_cursors(tmp_path: Path) -> None:
    """Modifications in one vault don't cause events in another."""
    vaults_dir = tmp_path / "vaults"
    vaults_dir.mkdir()

    vault_a = _make_vault(vaults_dir, "vault_a", {"note_a.md": "stable"})
    vault_b = _make_vault(vaults_dir, "vault_b", {"note_b.md": "will change"})

    # Build cursor manually to simulate a previous run — avoids relying on
    # the shutdown checkpoint which may overwrite the scan cursor.
    from worker.sources._handler import _read_file_atomic, _sha256_of

    cursor: dict[str, FileEntry] = {}
    for vault in [vault_a, vault_b]:
        for fpath in vault.rglob("*"):
            if fpath.is_file() and ".obsidian" not in fpath.parts:
                result = _read_file_atomic(fpath, 100 * 1024 * 1024)
                if result:
                    data, mtime_ns = result
                    cursor[str(fpath)] = FileEntry(
                        _sha256_of(data), mtime_ns, len(data)
                    )
    save_cursor(tmp_path / "cursor.json", cursor)

    # Modify only vault_b
    (vault_b / "note_b.md").write_text("changed content")

    # Scan with existing cursor — should only see vault_b modification
    s = _make_source(tmp_path)
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    events = await _run_source_briefly(s, q)

    scan_events = [
        e for e in events if e["operation"] in ("created", "modified", "deleted")
    ]
    assert len(scan_events) == 1
    assert scan_events[0]["operation"] == "modified"
    assert "note_b.md" in scan_events[0]["source_id"]
    assert scan_events[0]["meta"]["vault_name"] == "vault_b"


# ── Vault removed between runs ──────────────────────────────────


@pytest.mark.asyncio
async def test_vault_removed_emits_deleted_events(tmp_path: Path) -> None:
    """Files from a removed vault are emitted as 'deleted'."""
    import shutil

    from worker.sources._handler import _read_file_atomic, _sha256_of

    vaults_dir = tmp_path / "vaults"
    vaults_dir.mkdir()

    vault_a = _make_vault(vaults_dir, "vault_a", {"note_a.md": "keep"})
    vault_b = _make_vault(
        vaults_dir,
        "vault_b",
        {
            "note_b1.md": "remove",
            "note_b2.md": "remove too",
        },
    )

    # Build cursor manually to simulate previous run
    cursor: dict[str, FileEntry] = {}
    for vault in [vault_a, vault_b]:
        for fpath in vault.rglob("*"):
            if fpath.is_file() and ".obsidian" not in fpath.parts:
                result = _read_file_atomic(fpath, 100 * 1024 * 1024)
                if result:
                    data, mtime_ns = result
                    cursor[str(fpath)] = FileEntry(
                        _sha256_of(data), mtime_ns, len(data)
                    )
    save_cursor(tmp_path / "cursor.json", cursor)

    # Remove vault_b entirely
    shutil.rmtree(vault_b)

    # Scan with existing cursor — vault_b files should be 'deleted'
    s = _make_source(tmp_path)
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    events = await _run_source_briefly(s, q)

    deleted = [e for e in events if e["operation"] == "deleted"]
    deleted_ids = {e["source_id"] for e in deleted}

    assert len(deleted) == 2
    assert any("note_b1.md" in d for d in deleted_ids)
    assert any("note_b2.md" in d for d in deleted_ids)

    # vault_a should have no events (unchanged)
    a_events = [e for e in events if "vault_a" in e.get("source_id", "")]
    scan_a = [
        e for e in a_events if e["operation"] in ("created", "modified", "deleted")
    ]
    assert len(scan_a) == 0


# ── .obsidian/ skipping ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_scan_skips_dot_obsidian_directory(tmp_path: Path) -> None:
    """Files inside .obsidian/ are excluded from initial scan."""
    vaults_dir = tmp_path / "vaults"
    vaults_dir.mkdir()

    vault = _make_vault(vaults_dir, "vault", {"note.md": "content"})
    # Add config files to .obsidian/
    (vault / ".obsidian" / "app.json").write_text('{"key": "val"}')
    (vault / ".obsidian" / "plugins").mkdir()
    (vault / ".obsidian" / "plugins" / "plugin.json").write_text("{}")

    s = _make_source(tmp_path)
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    events = await _run_source_briefly(s, q)

    created = [e for e in events if e["operation"] == "created"]
    assert len(created) == 1
    assert "note.md" in created[0]["source_id"]

    # No .obsidian files should appear
    all_ids = [e["source_id"] for e in events]
    assert not any(".obsidian" in sid for sid in all_ids)


# ── Vault metadata ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_vault_metadata_in_events(tmp_path: Path) -> None:
    """Scan events contain vault_path, vault_name, relative_path, sha256."""
    vaults_dir = tmp_path / "vaults"
    vaults_dir.mkdir()

    vault = _make_vault(
        vaults_dir,
        "my_vault",
        {
            "top.md": "top level note",
            "sub/nested.md": "nested note",
        },
    )

    s = _make_source(tmp_path)
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    events = await _run_source_briefly(s, q)

    created = [e for e in events if e["operation"] == "created"]
    assert len(created) == 2

    for ev in created:
        assert ev["source_type"] == "obsidian"
        meta = ev["meta"]
        assert meta["vault_name"] == "my_vault"
        assert meta["vault_path"] == str(vault)
        assert "relative_path" in meta
        assert meta["sha256"]
        assert meta["size_bytes"] > 0

    # Verify relative paths
    rel_paths = {e["meta"]["relative_path"] for e in created}
    assert "top.md" in rel_paths
    assert str(Path("sub") / "nested.md") in rel_paths


# ── Dedup window for obsidian ─────────────────────────────────────


@pytest.mark.asyncio
async def test_obsidian_dedup_window(tmp_path: Path) -> None:
    """ObsidianSource arms dedup window after scan to prevent duplicates."""
    vaults_dir = tmp_path / "vaults"
    vaults_dir.mkdir()

    vault = _make_vault(vaults_dir, "vault", {"note.md": "stable content"})

    s = _make_source(tmp_path)
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    task = asyncio.create_task(s.start(q))
    await asyncio.sleep(1.0)

    # Drain scan events
    scan_events = []
    while not q.empty():
        scan_events.append(q.get_nowait())

    created = [e for e in scan_events if e["operation"] == "created"]
    assert len(created) == 1

    # Touch file without changing content (triggers watchdog but should be dedup'd)
    os.utime(vault / "note.md")
    await asyncio.sleep(0.5)

    post_events = []
    while not q.empty():
        post_events.append(q.get_nowait())

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # No duplicate created events for unchanged content
    note_created = [
        e
        for e in post_events
        if "note.md" in e.get("source_id", "") and e["operation"] == "created"
    ]
    assert len(note_created) == 0


# ── Periodic checkpoint for obsidian ──────────────────────────────


@pytest.mark.asyncio
async def test_obsidian_checkpoint_merges_vault_cursors(tmp_path: Path) -> None:
    """Checkpoint merges all vault handler cursors into one file."""
    vaults_dir = tmp_path / "vaults"
    vaults_dir.mkdir()

    vault_a = _make_vault(vaults_dir, "vault_a", {"a.md": "hello"})
    vault_b = _make_vault(vaults_dir, "vault_b", {"b.md": "world"})

    cursor_path = tmp_path / "cursor.json"
    s = _make_source(tmp_path, cursor_checkpoint_interval=1)
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    task = asyncio.create_task(s.start(q))
    # Wait for initial scan to complete
    await asyncio.sleep(1.0)

    # Ack initial scan events so cursor is populated
    while not q.empty():
        ev = q.get_nowait()
        cb = ev.get("_on_indexed")
        if cb:
            cb()

    # Create files in both vaults so watchdog events update handler cursors
    (vault_a / "new_a.md").write_text("new in A")
    (vault_b / "new_b.md").write_text("new in B")

    # Wait for watchdog to fire, then ack the new events
    await asyncio.sleep(1.5)
    while not q.empty():
        ev = q.get_nowait()
        cb = ev.get("_on_indexed")
        if cb:
            cb()

    # Wait for checkpoint interval
    await asyncio.sleep(1.5)

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # Cursor should contain entries from both vaults (via watchdog events)
    loaded = load_cursor(cursor_path)
    keys = list(loaded.keys())
    assert any("vault_a" in k for k in keys), f"vault_a not in cursor: {keys}"
    assert any("vault_b" in k for k in keys), f"vault_b not in cursor: {keys}"


@pytest.mark.asyncio
async def test_obsidian_graceful_shutdown_checkpoint(tmp_path: Path) -> None:
    """Cancelling ObsidianSource saves final merged checkpoint."""
    vaults_dir = tmp_path / "vaults"
    vaults_dir.mkdir()

    vault = _make_vault(vaults_dir, "vault", {"note.md": "content"})
    cursor_path = tmp_path / "cursor.json"

    # Long interval so checkpoint doesn't fire naturally
    s = _make_source(tmp_path, cursor_checkpoint_interval=3600)
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    task = asyncio.create_task(s.start(q))
    await asyncio.sleep(1.0)

    # Ack initial scan events
    while not q.empty():
        ev = q.get_nowait()
        cb = ev.get("_on_indexed")
        if cb:
            cb()

    # Create a file so watchdog updates handler cursor
    (vault / "new.md").write_text("new content")
    await asyncio.sleep(1.0)

    # Ack watchdog events
    while not q.empty():
        ev = q.get_nowait()
        cb = ev.get("_on_indexed")
        if cb:
            cb()

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # Even though checkpoint interval is 1 hour, cancellation saves handler cursor
    assert cursor_path.exists()
    loaded = load_cursor(cursor_path)
    assert len(loaded) >= 1


# ── Initial sync tracking ────────────────────────────────────────


@pytest.mark.asyncio
async def test_obsidian_scan_sets_sync_total_and_tags_events(tmp_path: Path) -> None:
    """ObsidianSource scan registers items and tags events with initial_scan."""
    import worker.metrics as m

    m._initial_sync_total = 0
    m.INITIAL_SYNC_ITEMS_TOTAL.set(0)

    vaults_dir = tmp_path / "vaults"
    _make_vault(vaults_dir, "v1", {"a.md": "alpha", "b.md": "beta"})

    src = _make_source(tmp_path)
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    events = await _run_source_briefly(src, q)

    scan_events = [e for e in events if e.get("initial_scan")]
    assert len(scan_events) >= 2

    assert m.initial_sync_get_total() >= 2
    assert m.INITIAL_SYNC_ITEMS_TOTAL._value.get() >= 2
