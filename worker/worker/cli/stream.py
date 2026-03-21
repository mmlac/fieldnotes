"""Streaming answer renderer: token-by-token output with rich formatting.

Renders LLM streaming responses to the terminal with:
- Token-by-token output as they arrive
- Inline citation formatting (dimmed [source_id] markers)
- Basic markdown rendering (bold, italic, headers, code blocks)
- Ctrl+C interruption handling
- Non-streaming and JSON output fallbacks
"""

from __future__ import annotations

import json
import re
import signal
import sys
from collections.abc import Iterator
from dataclasses import dataclass

from worker.models.base import StreamChunk

# ANSI escape codes
_BOLD = "\033[1m"
_DIM = "\033[2m"
_ITALIC = "\033[3m"
_UNDERLINE = "\033[4m"
_CYAN = "\033[36m"
_YELLOW = "\033[33m"
_GREEN = "\033[32m"
_MAGENTA = "\033[35m"
_RESET = "\033[0m"

# Citation pattern: [source_id] where source_id contains :// or looks like a hash/path
_CITATION_RE = re.compile(r"\[([^\]]{4,80})\]")


@dataclass
class StreamResult:
    """Accumulated result from a streaming render."""

    text: str = ""
    interrupted: bool = False
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class _RenderState:
    """Mutable state for the streaming renderer."""

    buffer: str = ""
    in_code_block: bool = False
    code_fence: str = ""
    collected_text: str = ""
    line_start: bool = True
    interrupted: bool = False
    input_tokens: int = 0
    output_tokens: int = 0


def _is_citation(text: str) -> bool:
    """Heuristic: looks like a source_id rather than normal bracketed text."""
    return (
        "://" in text
        or "/" in text
        or len(text) > 20
        or text.startswith("msg-")
        or text.startswith("file-")
        or text.startswith("source-")
    )


def _format_citation(source_id: str) -> str:
    """Format a citation marker with dimmed styling."""
    return f"{_DIM}[{source_id}]{_RESET}"


def _format_header(line: str) -> str:
    """Format a markdown header line."""
    stripped = line.lstrip("#")
    level = len(line) - len(stripped)
    text = stripped.strip()
    if level == 1:
        return f"\n{_BOLD}{_CYAN}{text}{_RESET}\n"
    elif level == 2:
        return f"\n{_BOLD}{text}{_RESET}\n"
    elif level >= 3:
        return f"\n{_UNDERLINE}{text}{_RESET}\n"
    return line


def _apply_inline_formatting(line: str) -> str:
    """Apply inline markdown formatting (bold, italic, code, citations)."""
    # Inline code (before bold/italic to avoid conflicts)
    line = re.sub(r"`([^`]+)`", rf"{_GREEN}\1{_RESET}", line)
    # Bold + italic
    line = re.sub(r"\*\*\*(.+?)\*\*\*", rf"{_BOLD}{_ITALIC}\1{_RESET}", line)
    # Bold
    line = re.sub(r"\*\*(.+?)\*\*", rf"{_BOLD}\1{_RESET}", line)
    # Italic
    line = re.sub(r"\*(.+?)\*", rf"{_ITALIC}\1{_RESET}", line)

    # Citations
    def _cite_repl(m: re.Match[str]) -> str:
        inner = m.group(1)
        if _is_citation(inner):
            return _format_citation(inner)
        return m.group(0)

    line = _CITATION_RE.sub(_cite_repl, line)
    return line


def _format_complete_line(line: str, in_code_block: bool) -> str:
    """Format a complete line of text."""
    if in_code_block:
        return line

    # Headers
    if line.startswith("#"):
        return _format_header(line)

    # Bullet points
    if re.match(r"^(\s*[-*+]\s)", line):
        return _apply_inline_formatting(line)

    return _apply_inline_formatting(line)


def render_stream(
    chunks: Iterator[StreamChunk],
    *,
    file: object | None = None,
) -> StreamResult:
    """Render a stream of chunks to the terminal with formatting.

    Writes tokens as they arrive. Handles Ctrl+C by stopping iteration
    and returning what was received so far.

    Args:
        chunks: Iterator of StreamChunk from a streaming completion.
        file: Output file object (default: sys.stdout).

    Returns:
        StreamResult with the accumulated text and token counts.
    """
    out = file or sys.stdout
    state = _RenderState()

    # Set up interrupt handling
    _old_handler = signal.getsignal(signal.SIGINT)

    def _interrupt(signum: int, frame: object) -> None:
        state.interrupted = True

    signal.signal(signal.SIGINT, _interrupt)

    try:
        for chunk in chunks:
            if state.interrupted:
                break

            if chunk.done:
                state.input_tokens = chunk.input_tokens
                state.output_tokens = chunk.output_tokens
                continue

            if not chunk.text:
                continue

            state.collected_text += chunk.text
            _render_token(chunk.text, state, out)

        # Flush any remaining buffer
        if state.buffer:
            _flush_buffer(state, out)

        if state.interrupted:
            out.write(f"\n{_DIM}(interrupted){_RESET}\n")
            out.flush()

    finally:
        signal.signal(signal.SIGINT, _old_handler)

    return StreamResult(
        text=state.collected_text,
        interrupted=state.interrupted,
        input_tokens=state.input_tokens,
        output_tokens=state.output_tokens,
    )


def _render_token(token: str, state: _RenderState, out: object) -> None:
    """Process a single token and render formatted output."""
    state.buffer += token

    # Process complete lines from the buffer
    while "\n" in state.buffer:
        line, state.buffer = state.buffer.split("\n", 1)
        _render_line(line, state, out)
        out.write("\n")
        state.line_start = True

    # If buffer doesn't contain potential formatting markers, flush it
    # Keep buffering if we might be in the middle of a markdown construct
    if state.buffer and not _might_be_partial(state.buffer):
        _flush_buffer(state, out)


def _might_be_partial(text: str) -> bool:
    """Check if text might be a partial markdown construct worth buffering."""
    # Partial code fence
    if text.endswith("`") or text.endswith("``"):
        return True
    # Partial bold/italic marker
    if text.endswith("*"):
        return True
    # Partial citation opening
    if "[" in text and "]" not in text:
        return True
    return False


def _render_line(line: str, state: _RenderState, out: object) -> None:
    """Render a complete line with formatting."""
    # Code fence toggling
    if line.startswith("```"):
        if not state.in_code_block:
            state.in_code_block = True
            state.code_fence = line
            lang = line[3:].strip()
            label = f" {lang}" if lang else ""
            out.write(f"{_DIM}---{label}{_RESET}")
        else:
            state.in_code_block = False
            state.code_fence = ""
            out.write(f"{_DIM}---{_RESET}")
        out.flush()
        return

    if state.in_code_block:
        out.write(f"{_GREEN}{line}{_RESET}")
    else:
        formatted = _format_complete_line(line, state.in_code_block)
        out.write(formatted)
    out.flush()


def _flush_buffer(state: _RenderState, out: object) -> None:
    """Flush the buffer, applying inline formatting if not in a code block."""
    text = state.buffer
    state.buffer = ""

    if not text:
        return

    if state.in_code_block:
        out.write(f"{_GREEN}{text}{_RESET}")
    else:
        out.write(_apply_inline_formatting(text))
    out.flush()


def render_no_stream(text: str, *, file: object | None = None) -> None:
    """Render a complete response with formatting (non-streaming mode)."""
    out = file or sys.stdout
    lines = text.split("\n")
    in_code = False

    for line in lines:
        if line.startswith("```"):
            if not in_code:
                in_code = True
                lang = line[3:].strip()
                label = f" {lang}" if lang else ""
                out.write(f"{_DIM}---{label}{_RESET}\n")
            else:
                in_code = False
                out.write(f"{_DIM}---{_RESET}\n")
            continue

        if in_code:
            out.write(f"{_GREEN}{line}{_RESET}\n")
        else:
            out.write(_format_complete_line(line, False) + "\n")

    out.flush()


def render_json(
    question: str,
    answer: str,
    sources: list[str],
    *,
    elapsed: float = 0.0,
    input_tokens: int = 0,
    output_tokens: int = 0,
    file: object | None = None,
) -> None:
    """Output structured JSON for scripting."""
    out = file or sys.stdout
    data = {
        "question": question,
        "answer": answer,
        "sources": sources,
        "timing": {"elapsed_seconds": round(elapsed, 3)},
        "tokens": {
            "input": input_tokens,
            "output": output_tokens,
        },
    }
    out.write(json.dumps(data, indent=2))
    out.write("\n")
    out.flush()


def format_sources(source_ids: list[str]) -> str:
    """Format a source list for display after the answer."""
    if not source_ids:
        return ""
    lines = [f"{_DIM}[Sources]{_RESET}"]
    for sid in source_ids:
        # Make file:// URIs for local paths
        if sid.startswith("/"):
            lines.append(f"  {_DIM}file://{sid}{_RESET}")
        else:
            lines.append(f"  {_DIM}{sid}{_RESET}")
    return "\n".join(lines)
