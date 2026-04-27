"""Tests for ``worker.parsers._pattern_match.matches_any`` (fn-pju).

Covers:

- Single-segment globs (extension, basename, segment-anywhere).
- Multi-segment patterns matching contiguous segments anywhere in the path.
- Multi-segment strict ordering (no skipping segments).
- Unicode normalization (NFC ↔ NFD) for accented characters and emoji.
- Leading/trailing slash normalization on patterns.
- Regression tests for existing single-segment configs that must keep working.
- Integration assertions that the two production call sites route through the
  shared helper.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path
from unittest.mock import patch

from worker.parsers._pattern_match import matches_any


# ---------------------------------------------------------------------------
# Single-segment patterns
# ---------------------------------------------------------------------------


def test_simple_glob() -> None:
    assert matches_any("foo.pdf", ["*.pdf"]) is True
    assert matches_any("foo.txt", ["*.pdf"]) is False


def test_basename_only() -> None:
    assert matches_any("/a/b/c/foo.pdf", ["*.pdf"]) is True


def test_segment_match() -> None:
    assert matches_any("/Users/x/notes/attachments/img.png", ["attachments"]) is True


# ---------------------------------------------------------------------------
# Multi-segment patterns
# ---------------------------------------------------------------------------


def test_multi_segment_anywhere() -> None:
    pattern = "Library/Uni"
    assert matches_any("/home/u/Documents/Library/Uni/syllabus.pdf", [pattern]) is True
    assert matches_any("/Library/Uni", [pattern]) is True
    assert matches_any("Library/Uni/x", [pattern]) is True
    assert matches_any("Library/Uni", [pattern]) is True


def test_multi_segment_strict_order() -> None:
    pattern = "Library/Uni"
    # Adjacent-but-wrong: 'Library' next to 'notUni' (literal mismatch).
    assert matches_any("/home/u/Library/notUni/x.pdf", [pattern]) is False
    # Non-contiguous: 'Library' and 'Uni' separated by another segment.
    assert matches_any("/Library/foo/Uni/x.pdf", [pattern]) is False


# ---------------------------------------------------------------------------
# Unicode normalization
# ---------------------------------------------------------------------------


def test_unicode_nfc_pattern_nfc_path() -> None:
    nfc_pattern = unicodedata.normalize("NFC", "Bücher")
    nfc_path = unicodedata.normalize("NFC", "/home/u/Bücher/x.pdf")
    assert matches_any(nfc_path, [nfc_pattern]) is True


def test_unicode_nfc_pattern_nfd_path() -> None:
    nfc_pattern = unicodedata.normalize("NFC", "Bücher")
    nfd_path = unicodedata.normalize("NFD", "/home/u/Bücher/x.pdf")
    # Sanity-check that the strings really are different on the wire.
    assert nfc_pattern != unicodedata.normalize("NFD", nfc_pattern)
    assert matches_any(nfd_path, [nfc_pattern]) is True


def test_unicode_nfd_pattern_nfc_path() -> None:
    nfd_pattern = unicodedata.normalize("NFD", "Bücher")
    nfc_path = unicodedata.normalize("NFC", "/home/u/Bücher/x.pdf")
    assert matches_any(nfc_path, [nfd_pattern]) is True


def test_german_umlauts_in_segment() -> None:
    assert matches_any("/home/u/Bücher/x.pdf", ["Bücher"]) is True


def test_emoji_in_path() -> None:
    assert matches_any("/launch/🚀/notes", ["🚀"]) is True


def test_combined_segment_unicode() -> None:
    nfc_pattern = unicodedata.normalize("NFC", "Schule/Bücher")
    nfd_path = unicodedata.normalize("NFD", "/x/Schule/Bücher/y.pdf")
    nfc_path = unicodedata.normalize("NFC", "/x/Schule/Bücher/y.pdf")
    assert matches_any(nfd_path, [nfc_pattern]) is True
    nfd_pattern = unicodedata.normalize("NFD", "Schule/Bücher")
    assert matches_any(nfc_path, [nfd_pattern]) is True


# ---------------------------------------------------------------------------
# Slash normalization
# ---------------------------------------------------------------------------


def test_leading_trailing_slashes_normalized() -> None:
    target_paths = [
        "/home/u/Documents/Library/Uni/syllabus.pdf",
        "/Library/Uni",
        "Library/Uni/x",
        "Library/Uni",
    ]
    for variant in ("Library/Uni", "/Library/Uni", "/Library/Uni/", "Library/Uni/"):
        for tp in target_paths:
            assert matches_any(tp, [variant]) is True, f"{variant!r} on {tp!r}"


# ---------------------------------------------------------------------------
# Regression: existing single-segment configs must keep working unchanged.
# ---------------------------------------------------------------------------


def test_existing_extension_pattern_unchanged() -> None:
    # Per the README index_only example.
    assert matches_any("/Volumes/Boot Camp.iso", ["*.iso"]) is True
    assert matches_any("/x/y/installer.dmg", ["*.dmg"]) is True
    assert matches_any("/x/y/notes.pdf", ["*.iso", "*.dmg"]) is False


def test_exclude_node_modules_unchanged() -> None:
    # The most common exclude pattern in the wild — still skipped at any depth.
    assert (
        matches_any("/repo/web/node_modules/react/index.js", ["node_modules/"]) is True
    )
    assert (
        matches_any("/repo/web/node_modules/react/index.js", ["node_modules"]) is True
    )
    assert matches_any("/repo/web/src/index.js", ["node_modules"]) is False


# ---------------------------------------------------------------------------
# Integration: production call sites must route through the helper.
# ---------------------------------------------------------------------------


def test_files_source_uses_helper() -> None:
    """``FileSource._should_skip`` and ``_is_index_only`` route through ``matches_any``."""
    from worker.sources.files import FileSource

    src = FileSource()
    src.configure(
        {
            "watch_paths": [str(Path.cwd())],
            "exclude_patterns": ["node_modules"],
            "index_only_patterns": ["*.iso"],
        }
    )

    with patch("worker.sources.files.matches_any", wraps=matches_any) as spy:
        assert src._should_skip("/repo/node_modules/react/index.js") is True
        assert src._is_index_only("/Volumes/Boot Camp.iso") is True
        # Both call sites must have invoked the helper.
        assert spy.call_count >= 2
        called_patterns = [call.args[1] for call in spy.call_args_list]
        assert ["node_modules"] in called_patterns
        assert ["*.iso"] in called_patterns


def test_queue_retag_uses_helper() -> None:
    """``cli.queue._matches_any_pattern`` routes through ``matches_any``."""
    from worker.cli import queue as queue_cli

    with patch("worker.cli.queue.matches_any", wraps=matches_any) as spy:
        assert (
            queue_cli._matches_any_pattern(
                "/home/u/Documents/Library/Uni/syllabus.pdf", ["Library/Uni"]
            )
            is True
        )
        spy.assert_called_once()
