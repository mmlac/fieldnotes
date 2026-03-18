"""Shared cursor persistence for file-based sources.

Stores per-file SHA256 hashes keyed by absolute path so that file and obsidian
sources can detect new, modified, and deleted files across restarts.

Cursor files use atomic writes (tempfile + os.replace) with fsync for
durability and chmod 0o600 for security.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, NamedTuple

logger = logging.getLogger(__name__)


class FileEntry(NamedTuple):
    """Per-file cursor entry."""

    sha256: str
    mtime_ns: int
    size: int


class CursorDiff(NamedTuple):
    """Result of diffing current scan results against a stored cursor."""

    new: set[str]
    modified: set[str]
    deleted: set[str]


def load_cursor(path: Path) -> dict[str, FileEntry]:
    """Load a cursor file from *path*.

    Returns an empty dict if the file is missing or corrupt (logs a warning
    on corrupt data).
    """
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict):
            logger.warning("Cursor file %s has unexpected format, starting fresh", path)
            return {}
        result: dict[str, FileEntry] = {}
        for file_path, entry in raw.items():
            if isinstance(entry, dict):
                result[file_path] = FileEntry(
                    sha256=entry.get("sha256", ""),
                    mtime_ns=int(entry.get("mtime_ns", 0)),
                    size=int(entry.get("size", 0)),
                )
        return result
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        logger.warning("Failed to read cursor file %s (%s), starting fresh", path, exc)
        return {}


def save_json_atomic(path: Path, data: Any) -> None:
    """Persist *data* as JSON to *path* atomically.

    Writes to a temporary file in the same directory, fsyncs, then renames
    into place.  On POSIX ``os.replace`` is atomic so a crash mid-write
    cannot leave a partially-written state file.  File permissions are set
    to 0o600 (owner read/write only).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=".state_",
        suffix=".tmp",
        delete=False,
    )
    try:
        fd.write(json.dumps(data))
        fd.flush()
        os.fsync(fd.fileno())
        fd.close()
        os.chmod(fd.name, 0o600)
        os.replace(fd.name, path)
    except BaseException:
        fd.close()
        with contextlib.suppress(OSError):
            os.unlink(fd.name)
        raise


def save_cursor(path: Path, cursor: dict[str, FileEntry]) -> None:
    """Persist *cursor* to *path* atomically.

    Writes to a temporary file in the same directory, fsyncs, then renames
    into place.  On POSIX ``os.replace`` is atomic so a crash mid-write
    cannot leave a partially-written cursor file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Serialise FileEntry namedtuples to plain dicts for JSON.
    serialisable: dict[str, dict[str, Any]] = {
        fp: {"sha256": e.sha256, "mtime_ns": e.mtime_ns, "size": e.size}
        for fp, e in cursor.items()
    }

    fd = tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=".cursor_",
        suffix=".tmp",
        delete=False,
    )
    try:
        fd.write(json.dumps(serialisable))
        fd.flush()
        os.fsync(fd.fileno())
        fd.close()
        os.chmod(fd.name, 0o600)
        os.replace(fd.name, path)
    except BaseException:
        fd.close()
        with contextlib.suppress(OSError):
            os.unlink(fd.name)
        raise


def diff_cursor(
    current: dict[str, FileEntry],
    stored: dict[str, FileEntry],
) -> CursorDiff:
    """Compare *current* scan results against *stored* cursor.

    Returns sets of (new, modified, deleted) absolute paths.

    * **New**: path in *current* but not in *stored*.
    * **Modified**: path in both but sha256 or mtime_ns differs.
    * **Deleted**: path in *stored* but not in *current*.
    """
    current_keys = set(current)
    stored_keys = set(stored)

    new = current_keys - stored_keys
    deleted = stored_keys - current_keys

    modified: set[str] = set()
    for path in current_keys & stored_keys:
        cur = current[path]
        old = stored[path]
        if cur.sha256 != old.sha256 or cur.mtime_ns != old.mtime_ns:
            modified.add(path)

    return CursorDiff(new=new, modified=modified, deleted=deleted)
