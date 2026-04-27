"""Symlink-safe, atomic token persistence helpers.

OAuth token files at ``~/.fieldnotes/data/<source>_token*.json`` carry
long-lived secrets.  The naive write sequence
``Path.touch / Path.write_text / Path.chmod`` follows symlinks and leaves
a window between write and chmod, so an attacker with write access to the
parent dir could swap the target for a symlink and capture (or redirect)
the secret.

These helpers avoid both pitfalls:

* writes use ``O_NOFOLLOW`` on the tmp file, refuse symlinked targets, and
  ``rename`` atomically — the file is never partially written under its
  final name and ``chmod`` is never racing the write;
* reads use ``O_NOFOLLOW`` so a swapped-in symlink raises rather than
  silently delivering an attacker's file content.

Both functions assume the parent directory is owned by the current uid;
they refuse to operate otherwise.  Callers (gmail/calendar/slack auth)
route every ``_save_token``/``_load_token`` call through here so the
symlink/atomicity invariants hold across all three sources.
"""

from __future__ import annotations

import errno
import os
import stat
from pathlib import Path


class TokenIOError(OSError):
    """Raised when a token file path fails the symlink/ownership checks."""


def _check_parent(parent: Path) -> None:
    st = os.lstat(parent)
    if stat.S_ISLNK(st.st_mode):
        raise TokenIOError(f"refusing to use {parent}: parent is a symlink")
    if st.st_uid != os.getuid():
        raise TokenIOError(f"refusing to use {parent}: not owned by current uid")


def write_token_atomic(path: Path, payload: str) -> None:
    """Write *payload* to *path* with mode 0o600, atomically and symlink-safe.

    Refuses (typed ``TokenIOError``) when the parent dir is a symlink, the
    parent is owned by another uid, or *path* itself is already a symlink.
    The write goes to ``<path>.tmp`` with ``O_NOFOLLOW`` and is renamed
    over *path* only after fsync, so a crash mid-write leaves the prior
    token intact.
    """
    parent = path.parent
    _check_parent(parent)

    tmp = path.with_name(path.name + ".tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW
    try:
        fd = os.open(tmp, flags, 0o600)
    except OSError as e:
        if e.errno in (errno.ELOOP, errno.EEXIST):
            raise TokenIOError(f"refusing to write {tmp}: target is a symlink") from e
        raise
    try:
        os.write(fd, payload.encode("utf-8"))
        os.fsync(fd)
        # Tighten mode in case umask widened the just-created tmp file.
        os.fchmod(fd, 0o600)
    finally:
        os.close(fd)

    if path.is_symlink():
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise TokenIOError(f"refusing to overwrite {path}: target is a symlink")

    os.rename(tmp, path)


def read_token_safe(path: Path) -> str | None:
    """Read *path* refusing to follow symlinks.

    Returns ``None`` when the file does not exist; raises ``TokenIOError``
    when the path (or any component the kernel resolves with
    ``O_NOFOLLOW``) is a symlink.
    """
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return None
    except OSError as e:
        if e.errno == errno.ELOOP:
            raise TokenIOError(f"refusing to read {path}: target is a symlink") from e
        raise
    try:
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks).decode("utf-8")
    finally:
        os.close(fd)
