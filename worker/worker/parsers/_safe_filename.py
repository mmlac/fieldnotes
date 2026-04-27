"""Sanitize attachment filenames before interpolating them into parent body text.

Author-controlled filenames flow into chunked body text via the
``Attachments:`` / ``[file] …`` augmentation in the Gmail, Slack, and Calendar
parsers.  A crafted filename that embeds line breaks (``\\n``, ``\\r``,
U+2028, U+2029, …) or markdown structure can corrupt the chunker's
line-based parsing and inject planted text into the indexed corpus
(see fn-s3e).

``sanitize_for_inline`` produces a single-line, length-bounded rendering
suitable for inline interpolation.  The original (unsanitized) filename
remains the canonical value on the Attachment Document — the helper only
guards body-text rendering.
"""

from __future__ import annotations

# U+0000 .. U+001F (excluding nothing) plus DEL.  These all have line-break
# or terminal-control side effects when rendered.
_ASCII_CONTROL = {chr(c) for c in range(0x00, 0x20)} | {chr(0x7F)}

# Unicode characters that act as line/paragraph separators in many
# downstream renderers (notably the chunker, which splits on more than
# just ``\\n``).
_UNICODE_LINE_SEPS = {
    " ",  # LINE SEPARATOR
    " ",  # PARAGRAPH SEPARATOR
    "",  # VERTICAL TAB (also in _ASCII_CONTROL but listed for clarity)
    "",  # FORM FEED         (ditto)
    "",  # NEXT LINE (NEL)
}

_BAD_CHARS = _ASCII_CONTROL | _UNICODE_LINE_SEPS

# Leading characters that look like markdown structure.  Filenames that
# begin with these should be wrapped in backticks so the chunker /
# downstream renderers do not treat them as headings, list items, or
# link/reference targets.
_MARKDOWN_LEADERS = ("-", "#", "*", ">", "[", "]", "`", "|", "!")

DEFAULT_MAX_LEN = 200
_TRUNCATION_SUFFIX = "…"


def sanitize_for_inline(name: str, *, max_len: int = DEFAULT_MAX_LEN) -> str:
    """Return *name* in a form safe to interpolate into chunkable body text.

    Steps (in order):

    1. Replace ASCII control characters (``\\x00``-``\\x1f``, ``\\x7f``)
       with a single space.
    2. Replace Unicode line/paragraph separators (U+2028, U+2029, NEL,
       VT, FF) with a single space.
    3. Collapse runs of whitespace and trim.
    4. Truncate to ``max_len`` characters, appending ``…`` if truncated.
    5. If the result starts with a markdown structural character, wrap
       the whole filename in backticks so it renders as inline code.

    The function is idempotent: ``sanitize_for_inline(sanitize_for_inline(x)) ==
    sanitize_for_inline(x)`` for all ``x``.
    """
    if not name:
        return ""

    out_chars: list[str] = []
    for ch in name:
        if ch in _BAD_CHARS:
            out_chars.append(" ")
        else:
            out_chars.append(ch)
    cleaned = "".join(out_chars)

    cleaned = " ".join(cleaned.split())

    if not cleaned:
        return ""

    if max_len > 0 and len(cleaned) > max_len:
        cleaned = cleaned[: max_len - 1].rstrip() + _TRUNCATION_SUFFIX

    if cleaned.startswith(_MARKDOWN_LEADERS) and not (
        cleaned.startswith("`") and cleaned.endswith("`")
    ):
        cleaned = f"`{cleaned}`"

    return cleaned
