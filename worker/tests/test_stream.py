"""Tests for the streaming answer renderer."""

from __future__ import annotations

import io
import json

from worker.cli.stream import (
    format_sources,
    render_json,
    render_no_stream,
    render_stream,
    _apply_inline_formatting,
    _format_header,
    _is_citation,
)
from worker.models.base import StreamChunk


class TestRenderStream:
    """Test token-by-token streaming render."""

    def test_basic_streaming(self) -> None:
        chunks = [
            StreamChunk(text="Hello "),
            StreamChunk(text="world!"),
            StreamChunk(input_tokens=10, output_tokens=5, done=True),
        ]
        out = io.StringIO()
        result = render_stream(iter(chunks), file=out)

        assert result.text == "Hello world!"
        assert result.input_tokens == 10
        assert result.output_tokens == 5
        assert not result.interrupted

    def test_empty_stream(self) -> None:
        chunks = [StreamChunk(done=True)]
        out = io.StringIO()
        result = render_stream(iter(chunks), file=out)

        assert result.text == ""
        assert not result.interrupted

    def test_multiline_stream(self) -> None:
        chunks = [
            StreamChunk(text="Line 1\n"),
            StreamChunk(text="Line 2\n"),
            StreamChunk(done=True),
        ]
        out = io.StringIO()
        result = render_stream(iter(chunks), file=out)

        assert result.text == "Line 1\nLine 2\n"

    def test_collects_all_text(self) -> None:
        chunks = [
            StreamChunk(text="a"),
            StreamChunk(text="b"),
            StreamChunk(text="c"),
            StreamChunk(done=True),
        ]
        out = io.StringIO()
        result = render_stream(iter(chunks), file=out)
        assert result.text == "abc"


class TestRenderNoStream:
    """Test non-streaming render."""

    def test_basic_text(self) -> None:
        out = io.StringIO()
        render_no_stream("Hello world", file=out)
        assert "Hello world" in out.getvalue()

    def test_code_block(self) -> None:
        out = io.StringIO()
        render_no_stream("```python\nprint('hi')\n```", file=out)
        output = out.getvalue()
        # Code should be rendered with green ANSI
        assert "print('hi')" in output

    def test_headers(self) -> None:
        out = io.StringIO()
        render_no_stream("# Title\n## Subtitle", file=out)
        output = out.getvalue()
        assert "Title" in output
        assert "Subtitle" in output


class TestRenderJson:
    """Test JSON output mode."""

    def test_basic_json(self) -> None:
        out = io.StringIO()
        render_json(
            "What is X?",
            "X is Y.",
            ["source-1", "source-2"],
            elapsed=1.234,
            input_tokens=100,
            output_tokens=50,
            file=out,
        )
        data = json.loads(out.getvalue())
        assert data["question"] == "What is X?"
        assert data["answer"] == "X is Y."
        assert data["sources"] == ["source-1", "source-2"]
        assert data["timing"]["elapsed_seconds"] == 1.234
        assert data["tokens"]["input"] == 100
        assert data["tokens"]["output"] == 50

    def test_empty_sources(self) -> None:
        out = io.StringIO()
        render_json("Q?", "A.", [], file=out)
        data = json.loads(out.getvalue())
        assert data["sources"] == []


class TestInlineFormatting:
    """Test inline markdown formatting."""

    def test_bold(self) -> None:
        result = _apply_inline_formatting("This is **bold** text")
        assert "bold" in result
        assert "**" not in result

    def test_italic(self) -> None:
        result = _apply_inline_formatting("This is *italic* text")
        assert "italic" in result
        assert result.count("*") == 0

    def test_inline_code(self) -> None:
        result = _apply_inline_formatting("Use `code` here")
        assert "code" in result
        assert "`" not in result

    def test_citation_formatting(self) -> None:
        result = _apply_inline_formatting("See [email://msg-abc123] for details")
        # Citation should be dimmed
        assert "\033[2m" in result
        assert "email://msg-abc123" in result


class TestCitationDetection:
    """Test citation heuristics."""

    def test_uri_citation(self) -> None:
        assert _is_citation("email://msg-abc123")
        assert _is_citation("file:///path/to/doc")

    def test_path_citation(self) -> None:
        assert _is_citation("notes/2024/topic")

    def test_msg_prefix(self) -> None:
        assert _is_citation("msg-abc123456")

    def test_short_text_not_citation(self) -> None:
        assert not _is_citation("abc")
        assert not _is_citation("TODO")


class TestFormatSources:
    """Test source list formatting."""

    def test_formats_source_list(self) -> None:
        result = format_sources(["source-1", "source-2"])
        assert "Sources" in result
        assert "source-1" in result
        assert "source-2" in result

    def test_file_paths_get_uri(self) -> None:
        result = format_sources(["/home/user/doc.md"])
        assert "file:///home/user/doc.md" in result

    def test_empty_list(self) -> None:
        assert format_sources([]) == ""


class TestFormatHeader:
    """Test header formatting."""

    def test_h1(self) -> None:
        result = _format_header("# Title")
        assert "Title" in result
        assert "\033[1m" in result  # bold

    def test_h2(self) -> None:
        result = _format_header("## Subtitle")
        assert "Subtitle" in result
        assert "\033[1m" in result

    def test_h3(self) -> None:
        result = _format_header("### Section")
        assert "Section" in result
        assert "\033[4m" in result  # underline
