"""Tests for cursor persistence: atomic writes, large cursors, and fsync.

Complements test_cursor.py with tests for atomicity guarantees, concurrent
access safety, and performance with large cursor files.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from unittest.mock import patch

from worker.sources.cursor import FileEntry, diff_cursor, load_cursor, save_cursor


# ── Atomic write ──────────────────────────────────────────────────


def test_atomic_write_concurrent_reads(tmp_path: Path) -> None:
    """Concurrent reads during write see old or new data, never partial."""
    cursor_path = tmp_path / "cursor.json"

    # Write initial cursor
    old_cursor = {f"/file_{i}": FileEntry(f"old_{i}", i, i) for i in range(100)}
    save_cursor(cursor_path, old_cursor)

    # Prepare new cursor
    new_cursor = {f"/file_{i}": FileEntry(f"new_{i}", i * 2, i * 2) for i in range(100)}

    results: list[dict[str, FileEntry]] = []
    errors: list[Exception] = []
    barrier = threading.Barrier(2)

    def reader() -> None:
        barrier.wait()
        for _ in range(50):
            try:
                loaded = load_cursor(cursor_path)
                results.append(loaded)
            except Exception as exc:
                errors.append(exc)

    t = threading.Thread(target=reader)
    t.start()
    barrier.wait()
    save_cursor(cursor_path, new_cursor)
    t.join()

    assert not errors, f"Reader raised exceptions: {errors}"

    # Each read should return a complete cursor (old or new), never partial
    for loaded in results:
        if not loaded:
            continue  # File might not exist momentarily during atomic rename
        # All entries should share the same prefix (old_ or new_), never mixed
        prefixes = set()
        for entry in loaded.values():
            prefixes.add(entry.sha256.split("_")[0])
        assert len(prefixes) == 1, (
            f"Got mixed prefixes {prefixes} — partial write was visible"
        )


def test_fsync_called_on_write(tmp_path: Path) -> None:
    """os.fsync is called during save_cursor."""
    cursor_path = tmp_path / "cursor.json"
    cursor = {"/a": FileEntry("sha", 1, 1)}

    with patch("worker.sources.cursor.os.fsync") as mock_fsync:
        save_cursor(cursor_path, cursor)
        assert mock_fsync.called, "os.fsync was not called during save"


def test_temp_file_cleaned_on_error(tmp_path: Path) -> None:
    """If writing fails, the temp file is cleaned up."""
    cursor_path = tmp_path / "cursor.json"

    # Write initial data so cursor_path exists
    save_cursor(cursor_path, {"/a": FileEntry("sha", 1, 1)})

    # Force json.dumps to fail
    with patch("worker.sources.cursor.json.dumps", side_effect=RuntimeError("boom")):
        try:
            save_cursor(cursor_path, {"/b": FileEntry("sha2", 2, 2)})
        except RuntimeError:
            pass

    # Original file should still be intact
    loaded = load_cursor(cursor_path)
    assert "/a" in loaded

    # No temp files left behind
    temps = list(tmp_path.glob(".cursor_*.tmp"))
    assert temps == [], f"Temp files left behind: {temps}"


# ── Large cursor ──────────────────────────────────────────────────


def test_large_cursor_save_load_roundtrip(tmp_path: Path) -> None:
    """10k+ entry cursor survives save/load roundtrip."""
    cursor_path = tmp_path / "cursor.json"
    n = 10_000
    cursor = {
        f"/path/to/file_{i}.md": FileEntry(f"sha256_{i}", i * 1000, i) for i in range(n)
    }

    save_cursor(cursor_path, cursor)
    loaded = load_cursor(cursor_path)

    assert len(loaded) == n
    # Spot-check a few entries
    assert loaded["/path/to/file_0.md"] == FileEntry("sha256_0", 0, 0)
    assert loaded["/path/to/file_9999.md"] == FileEntry("sha256_9999", 9999000, 9999)


def test_large_cursor_performance(tmp_path: Path) -> None:
    """10k+ entry cursor loads and saves within reasonable time."""
    cursor_path = tmp_path / "cursor.json"
    n = 10_000
    cursor = {f"/path/file_{i}": FileEntry(f"sha_{i}", i, i) for i in range(n)}

    start = time.monotonic()
    save_cursor(cursor_path, cursor)
    save_duration = time.monotonic() - start

    start = time.monotonic()
    loaded = load_cursor(cursor_path)
    load_duration = time.monotonic() - start

    assert len(loaded) == n
    # Should complete well under 5 seconds even on slow CI
    assert save_duration < 5.0, f"Save took {save_duration:.2f}s"
    assert load_duration < 5.0, f"Load took {load_duration:.2f}s"


# ── diff with large data sets ─────────────────────────────────────


def test_diff_cursor_large_with_mixed_changes() -> None:
    """Large cursor diff with realistic mixed changes."""
    n = 10_000
    stored = {f"/file_{i}": FileEntry(f"sha_{i}", i, i) for i in range(n)}

    # Simulate: 500 new, 500 modified, 500 deleted, 8500 unchanged
    current = {}
    for i in range(500, n):  # skip first 500 (deleted)
        if i < 1000:
            # modified: different sha256
            current[f"/file_{i}"] = FileEntry(f"sha_{i}_mod", i + 1, i)
        else:
            # unchanged
            current[f"/file_{i}"] = FileEntry(f"sha_{i}", i, i)

    # 500 new files
    for i in range(n, n + 500):
        current[f"/file_{i}"] = FileEntry(f"sha_{i}", i, i)

    result = diff_cursor(current, stored)
    assert len(result.new) == 500
    assert len(result.modified) == 500
    assert len(result.deleted) == 500


# ── Permissions ───────────────────────────────────────────────────


def test_save_cursor_permissions_0o600(tmp_path: Path) -> None:
    """Cursor file gets 0o600 permissions on every save."""
    cursor_path = tmp_path / "cursor.json"

    # First save
    save_cursor(cursor_path, {"/a": FileEntry("sha", 1, 1)})
    mode = os.stat(cursor_path).st_mode & 0o777
    assert mode == 0o600

    # Second save (overwrite) should also set 0o600
    save_cursor(cursor_path, {"/b": FileEntry("sha2", 2, 2)})
    mode = os.stat(cursor_path).st_mode & 0o777
    assert mode == 0o600
