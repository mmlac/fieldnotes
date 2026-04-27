"""Tests for ``worker.parsers._safe_filename.sanitize_for_inline``."""

from __future__ import annotations

import pytest

from worker.parsers._safe_filename import sanitize_for_inline


class TestStripsNewlines:
    @pytest.mark.parametrize(
        "raw",
        [
            "evil.pdf\nFrom: ceo@x",
            "evil.pdf\rFrom: ceo@x",
            "evil.pdf\r\nFrom: ceo@x",
        ],
    )
    def test_no_line_breaks_remain(self, raw: str) -> None:
        out = sanitize_for_inline(raw)
        assert "\n" not in out
        assert "\r" not in out
        # The visible payload must still be preserved (just on one line).
        assert "evil.pdf" in out
        assert "From: ceo@x" in out


class TestStripsUnicodeLineSeparators:
    @pytest.mark.parametrize(
        "sep",
        [
            " ",  # LINE SEPARATOR
            " ",  # PARAGRAPH SEPARATOR
            "",  # NEXT LINE
        ],
    )
    def test_no_unicode_line_seps_remain(self, sep: str) -> None:
        out = sanitize_for_inline(f"safe.pdf{sep}injected")
        assert sep not in out
        assert "\n" not in out
        assert "safe.pdf" in out
        assert "injected" in out


class TestStripsControlChars:
    def test_all_ascii_control_chars_replaced(self) -> None:
        # \x00 through \x1f plus \x7f.
        controls = "".join(chr(c) for c in range(0x00, 0x20)) + "\x7f"
        out = sanitize_for_inline(f"a{controls}b")
        for c in controls:
            assert c not in out
        assert "a" in out and "b" in out

    def test_vertical_tab_and_form_feed_replaced(self) -> None:
        out = sanitize_for_inline("a\x0bb\x0cc")
        assert "\x0b" not in out
        assert "\x0c" not in out


class TestTruncation:
    def test_truncates_to_default_max(self) -> None:
        raw = "x" * 500
        out = sanitize_for_inline(raw)
        assert len(out) <= 200
        assert out.endswith("…")

    def test_truncates_to_custom_max(self) -> None:
        raw = "y" * 50
        out = sanitize_for_inline(raw, max_len=10)
        assert len(out) == 10
        assert out.endswith("…")

    def test_short_name_not_truncated(self) -> None:
        out = sanitize_for_inline("short.pdf")
        assert out == "short.pdf"


class TestMarkdownLeaderQuoting:
    @pytest.mark.parametrize("leader", ["-", "#", "*", ">", "[", "!", "|"])
    def test_markdown_leaders_get_backticked(self, leader: str) -> None:
        out = sanitize_for_inline(f"{leader} report.pdf")
        assert out.startswith("`") and out.endswith("`")

    def test_normal_filename_not_quoted(self) -> None:
        assert sanitize_for_inline("report.pdf") == "report.pdf"

    def test_existing_backticks_preserved_not_doubled(self) -> None:
        # If the cleaned name already starts and ends with backticks,
        # we should not wrap it again — calling sanitize twice must be
        # a no-op (idempotency, see TestIdempotent).
        first = sanitize_for_inline("- evil.pdf")
        twice = sanitize_for_inline(first)
        assert first == twice


class TestIdempotent:
    @pytest.mark.parametrize(
        "raw",
        [
            "plain.pdf",
            "evil.pdf\nFrom: x",
            "safe.pdf injected",
            "x" * 500,
            "- list-leader.pdf",
            "# heading.pdf",
            "",
        ],
    )
    def test_double_sanitize_equals_single(self, raw: str) -> None:
        once = sanitize_for_inline(raw)
        twice = sanitize_for_inline(once)
        assert once == twice


class TestEdgeCases:
    def test_empty_string(self) -> None:
        assert sanitize_for_inline("") == ""

    def test_only_whitespace_collapses_to_empty(self) -> None:
        assert sanitize_for_inline("   \t  ") == ""

    def test_only_control_chars_collapses_to_empty(self) -> None:
        assert sanitize_for_inline("\x00\x01\x02") == ""

    def test_collapses_runs_of_whitespace(self) -> None:
        out = sanitize_for_inline("a    b\t\tc")
        assert out == "a b c"
