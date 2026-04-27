"""Glob-pattern matcher for filesystem paths.

Used by both the ingest pipeline (``worker.sources.files`` /
``worker.sources._handler``) and the ``fieldnotes queue retag`` CLI
(``worker.cli.queue``) so the same patterns produce the same matches.

Two semantics this helper guarantees:

1. **Multi-segment patterns match contiguous path segments anywhere in
   the path.**  A pattern like ``"Library/Uni"`` matches any path whose
   segments include ``Library`` immediately followed by ``Uni``
   (e.g. ``/home/u/Documents/Library/Uni/syllabus.pdf``).  A bare
   ``fnmatch`` does not treat ``/`` specially, so the previous
   per-segment matcher silently failed on these patterns (fn-pju).

2. **Unicode normalization (NFC) is applied to both pattern and
   path before comparison.**  macOS HFS+/APFS stores filenames as NFD
   while config files typed by humans usually arrive as NFC; comparing
   them codepoint-for-codepoint silently fails for accented characters
   like ``ü`` / ``ö`` / ``ä`` (fn-pju).

Single-segment patterns retain the legacy three-way fallback
(full path, basename, any path component) so existing configs keep
working unchanged.
"""

from __future__ import annotations

import fnmatch
import pathlib
import unicodedata
from collections.abc import Iterable


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def matches_any(path: str, patterns: Iterable[str]) -> bool:
    """Return True if *path* matches any glob in *patterns*.

    See module docstring for matching rules.
    """
    norm_path = _nfc(path)
    p = pathlib.PurePosixPath(norm_path)
    parts = tuple(_nfc(part) for part in p.parts)
    name = parts[-1] if parts else ""

    for raw_pattern in patterns:
        pattern = _nfc(raw_pattern)
        stripped = pattern.strip("/")
        if not stripped:
            continue

        if "/" in stripped:
            pat_parts = stripped.split("/")
            n = len(pat_parts)
            if n > len(parts):
                continue
            for i in range(len(parts) - n + 1):
                if all(
                    fnmatch.fnmatchcase(parts[i + j], pat_parts[j]) for j in range(n)
                ):
                    return True
        else:
            # Strip leading/trailing slashes so directory-style patterns
            # like ``"node_modules/"`` match the bare segment name.
            single = stripped
            if fnmatch.fnmatch(norm_path, single):
                return True
            if fnmatch.fnmatch(name, single):
                return True
            if any(fnmatch.fnmatch(part, single) for part in parts):
                return True

    return False
