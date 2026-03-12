"""Tests for the shared cursor persistence module."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

from worker.sources.cursor import (
    CursorDiff,
    FileEntry,
    diff_cursor,
    load_cursor,
    save_cursor,
)


# ── load_cursor ─────────────────────────────────────────────────────


def test_load_cursor_missing_file(tmp_path: Path) -> None:
    """Returns empty dict when cursor file does not exist."""
    result = load_cursor(tmp_path / "nonexistent.json")
    assert result == {}


def test_load_cursor_valid(tmp_path: Path) -> None:
    """Loads well-formed cursor data."""
    cursor_path = tmp_path / "cursor.json"
    data = {
        "/home/user/notes/foo.md": {"sha256": "abc123", "mtime_ns": 1000, "size": 42},
        "/home/user/notes/bar.md": {"sha256": "def456", "mtime_ns": 2000, "size": 99},
    }
    cursor_path.write_text(json.dumps(data))

    result = load_cursor(cursor_path)
    assert len(result) == 2
    assert result["/home/user/notes/foo.md"] == FileEntry("abc123", 1000, 42)
    assert result["/home/user/notes/bar.md"] == FileEntry("def456", 2000, 99)


def test_load_cursor_corrupt_json(tmp_path: Path) -> None:
    """Returns empty dict on corrupt JSON (with warning)."""
    cursor_path = tmp_path / "cursor.json"
    cursor_path.write_text("{not valid json")

    result = load_cursor(cursor_path)
    assert result == {}


def test_load_cursor_non_dict(tmp_path: Path) -> None:
    """Returns empty dict when top-level value is not a dict."""
    cursor_path = tmp_path / "cursor.json"
    cursor_path.write_text(json.dumps([1, 2, 3]))

    result = load_cursor(cursor_path)
    assert result == {}


def test_load_cursor_empty_file(tmp_path: Path) -> None:
    """Returns empty dict on empty file."""
    cursor_path = tmp_path / "cursor.json"
    cursor_path.write_text("")

    result = load_cursor(cursor_path)
    assert result == {}


# ── save_cursor ─────────────────────────────────────────────────────


def test_save_cursor_roundtrip(tmp_path: Path) -> None:
    """Data survives a save/load roundtrip."""
    cursor_path = tmp_path / "cursor.json"
    entries = {
        "/a/b.md": FileEntry("aaa", 100, 10),
        "/c/d.txt": FileEntry("bbb", 200, 20),
    }

    save_cursor(cursor_path, entries)
    loaded = load_cursor(cursor_path)

    assert loaded == entries


def test_save_cursor_permissions(tmp_path: Path) -> None:
    """Cursor file is created with 0o600 permissions."""
    cursor_path = tmp_path / "cursor.json"
    save_cursor(cursor_path, {"/x": FileEntry("sha", 1, 1)})

    mode = stat.S_IMODE(os.stat(cursor_path).st_mode)
    assert mode == 0o600


def test_save_cursor_creates_parent_dirs(tmp_path: Path) -> None:
    """Parent directories are created automatically."""
    cursor_path = tmp_path / "sub" / "dir" / "cursor.json"
    save_cursor(cursor_path, {"/x": FileEntry("sha", 1, 1)})

    assert cursor_path.exists()


def test_save_cursor_overwrites_existing(tmp_path: Path) -> None:
    """Overwriting an existing cursor file works."""
    cursor_path = tmp_path / "cursor.json"

    save_cursor(cursor_path, {"/a": FileEntry("old", 1, 1)})
    save_cursor(cursor_path, {"/b": FileEntry("new", 2, 2)})

    loaded = load_cursor(cursor_path)
    assert list(loaded.keys()) == ["/b"]
    assert loaded["/b"].sha256 == "new"


# ── diff_cursor ─────────────────────────────────────────────────────


def test_diff_cursor_empty_both() -> None:
    """Empty current and stored yields empty diff."""
    result = diff_cursor({}, {})
    assert result == CursorDiff(set(), set(), set())


def test_diff_cursor_all_new() -> None:
    """All paths in current but not in stored are new."""
    current = {
        "/a": FileEntry("sha1", 100, 10),
        "/b": FileEntry("sha2", 200, 20),
    }
    result = diff_cursor(current, {})
    assert result.new == {"/a", "/b"}
    assert result.modified == set()
    assert result.deleted == set()


def test_diff_cursor_all_deleted() -> None:
    """All paths in stored but not in current are deleted."""
    stored = {
        "/a": FileEntry("sha1", 100, 10),
        "/b": FileEntry("sha2", 200, 20),
    }
    result = diff_cursor({}, stored)
    assert result.new == set()
    assert result.modified == set()
    assert result.deleted == {"/a", "/b"}


def test_diff_cursor_modified_sha256() -> None:
    """Path with changed sha256 is modified."""
    stored = {"/a": FileEntry("old_sha", 100, 10)}
    current = {"/a": FileEntry("new_sha", 100, 10)}

    result = diff_cursor(current, stored)
    assert result.modified == {"/a"}
    assert result.new == set()
    assert result.deleted == set()


def test_diff_cursor_modified_mtime() -> None:
    """Path with changed mtime_ns is modified."""
    stored = {"/a": FileEntry("sha", 100, 10)}
    current = {"/a": FileEntry("sha", 200, 10)}

    result = diff_cursor(current, stored)
    assert result.modified == {"/a"}


def test_diff_cursor_unchanged() -> None:
    """Identical entries produce no diff."""
    entries = {"/a": FileEntry("sha", 100, 10)}
    result = diff_cursor(entries, dict(entries))
    assert result == CursorDiff(set(), set(), set())


def test_diff_cursor_size_change_only() -> None:
    """Size change alone (without sha256/mtime change) is NOT modified."""
    stored = {"/a": FileEntry("sha", 100, 10)}
    current = {"/a": FileEntry("sha", 100, 99)}

    result = diff_cursor(current, stored)
    assert result.modified == set()


def test_diff_cursor_mixed() -> None:
    """Mix of new, modified, deleted, and unchanged paths."""
    stored = {
        "/unchanged": FileEntry("sha1", 100, 10),
        "/modified": FileEntry("old", 200, 20),
        "/deleted": FileEntry("sha3", 300, 30),
    }
    current = {
        "/unchanged": FileEntry("sha1", 100, 10),
        "/modified": FileEntry("new", 200, 20),
        "/new_file": FileEntry("sha4", 400, 40),
    }

    result = diff_cursor(current, stored)
    assert result.new == {"/new_file"}
    assert result.modified == {"/modified"}
    assert result.deleted == {"/deleted"}


def test_diff_cursor_large(tmp_path: Path) -> None:
    """Handles 100k+ entries without error."""
    n = 100_000
    stored = {f"/file_{i}": FileEntry(f"sha_{i}", i, i) for i in range(n)}
    # Modify 100, delete 100, add 100
    current = dict(stored)
    for i in range(100):
        current[f"/file_{i}"] = FileEntry(f"sha_{i}_mod", i, i)
    for i in range(n - 100, n):
        del current[f"/file_{i}"]
    for i in range(n, n + 100):
        current[f"/file_{i}"] = FileEntry(f"sha_{i}", i, i)

    result = diff_cursor(current, stored)
    assert len(result.new) == 100
    assert len(result.modified) == 100
    assert len(result.deleted) == 100
