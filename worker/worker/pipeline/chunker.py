"""Sentence-aware text splitter with overlap.

Splits text into chunks of ~512 tokens with 64-token overlap, using sentence
boundaries as preferred split points. Chunks shorter than 100 tokens are merged
with adjacent chunks. No model dependencies — uses whitespace tokenisation.

Optional Slack-aware mode: when ``chunk_strategy={"mode": "message_overlap",
"overlap_messages": N}`` is passed, the chunker treats lines that match the
Slack parser's per-message header (``[HH:MM UTC] author: text``, optionally
indented for thread replies) as message boundaries and splits at the natural
gap nearest the target chunk size. Adjacent chunks share the last N whole
messages of the previous chunk as overlap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# Sentence-ending punctuation followed by whitespace or end-of-string.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# Slack message header: optional 2-space indent (thread reply) + [HH:MM UTC].
# Anchored to start-of-line via re.MULTILINE.
_SLACK_MESSAGE_RE = re.compile(r"^(?:  )?\[\d{2}:\d{2} UTC\]", re.MULTILINE)

DEFAULT_CHUNK_SIZE = 512
DEFAULT_OVERLAP = 64
MIN_CHUNK_TOKENS = 100


@dataclass
class Chunk:
    """A text chunk with its positional index."""

    text: str
    index: int


def _tokenize(text: str) -> list[str]:
    """Split text into whitespace-delimited tokens."""
    return text.split()


def _detokenize(tokens: list[str]) -> str:
    """Join tokens back into text."""
    return " ".join(tokens)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using punctuation boundaries.

    Falls back to the full text as a single sentence if no boundaries found.
    """
    parts = _SENTENCE_RE.split(text)
    return [s for s in parts if s.strip()]


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    min_chunk_tokens: int = MIN_CHUNK_TOKENS,
    chunk_strategy: dict[str, Any] | None = None,
) -> list[Chunk]:
    """Split *text* into overlapping chunks using sentence boundaries.

    Parameters
    ----------
    text:
        The input text to split.
    chunk_size:
        Target chunk size in whitespace tokens (~512).
    overlap:
        Number of overlapping tokens between consecutive chunks (~64).
    min_chunk_tokens:
        Chunks shorter than this are merged with the next chunk.
    chunk_strategy:
        Optional per-document strategy hint. Currently supported:
        ``{"mode": "message_overlap", "overlap_messages": N}`` — split at
        Slack message boundaries with whole-message overlap. Any other
        value (or None) uses the default sentence-aware path.

    Returns
    -------
    list[Chunk]
        Ordered chunks with their positional indices.
    """
    if not text or not text.strip():
        return []

    if chunk_strategy and chunk_strategy.get("mode") == "message_overlap":
        return _chunk_by_message_overlap(
            text,
            chunk_size=chunk_size,
            overlap_messages=int(chunk_strategy.get("overlap_messages", 3)),
        )

    sentences = _split_sentences(text)
    # Tokenise each sentence.
    sentence_tokens: list[list[str]] = [_tokenize(s) for s in sentences]
    # Drop empty sentences.
    sentence_tokens = [t for t in sentence_tokens if t]

    if not sentence_tokens:
        return []

    chunks: list[list[str]] = []
    current: list[str] = []

    for tokens in sentence_tokens:
        # If adding this sentence would exceed the target, flush current chunk
        # (but only if current is non-empty and already meets the minimum).
        if current and len(current) + len(tokens) > chunk_size:
            chunks.append(current)
            # Start new chunk with overlap from the end of the previous one.
            if overlap > 0 and len(current) >= overlap:
                current = current[-overlap:] + tokens
            else:
                current = list(tokens)
        else:
            current.extend(tokens)

    # Flush remaining tokens.
    if current:
        chunks.append(current)

    # Merge short trailing chunk into the previous one.
    chunks = _merge_short_chunks(chunks, min_chunk_tokens, chunk_size)

    return [Chunk(text=_detokenize(tokens), index=i) for i, tokens in enumerate(chunks)]


def _merge_short_chunks(
    chunks: list[list[str]], min_tokens: int, chunk_size: int
) -> list[list[str]]:
    """Merge any chunk shorter than *min_tokens* with an adjacent chunk.

    Merged chunks are capped at 1.5× *chunk_size* to prevent unbounded growth
    when many consecutive short chunks appear.
    """
    if not chunks:
        return chunks

    max_merged = int(chunk_size * 1.5)
    merged: list[list[str]] = [chunks[0]]

    for chunk in chunks[1:]:
        if len(chunk) < min_tokens and len(merged[-1]) + len(chunk) <= max_merged:
            # Merge into the previous chunk only if it won't exceed the cap.
            merged[-1].extend(chunk)
        else:
            merged.append(chunk)

    # If the first chunk ended up too short after everything, merge forward.
    if len(merged) > 1 and len(merged[0]) < min_tokens:
        if len(merged[0]) + len(merged[1]) <= max_merged:
            merged[1] = merged[0] + merged[1]
            merged.pop(0)

    return merged


def _split_slack_messages(text: str) -> list[str]:
    """Split Slack-formatted text into per-message blocks.

    Each block runs from one ``[HH:MM UTC] ...`` header line up to (but not
    including) the next header line. Trailing newlines are stripped. If the
    text contains no recognizable headers, the entire body is treated as a
    single message so the caller never silently drops content.
    """
    matches = list(_SLACK_MESSAGE_RE.finditer(text))
    if not matches:
        stripped = text.strip("\n")
        return [stripped] if stripped else []

    messages: list[str] = []
    # Any prose before the first header (rare but possible) becomes its own
    # leading message so we don't drop it.
    first_start = matches[0].start()
    preamble = text[:first_start].strip("\n")
    if preamble.strip():
        messages.append(preamble)

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end].rstrip("\n")
        if block.strip():
            messages.append(block)
    return messages


def _chunk_by_message_overlap(
    text: str,
    chunk_size: int,
    overlap_messages: int,
) -> list[Chunk]:
    """Split *text* on Slack message boundaries with whole-message overlap.

    Greedy fill: include messages until adding the next would push the chunk
    further from ``chunk_size`` than stopping would. A message is never split
    in half; if a single message exceeds ``chunk_size`` it becomes its own
    chunk. Adjacent chunks share the last ``overlap_messages`` whole
    messages of the prior chunk.
    """
    if overlap_messages < 0:
        overlap_messages = 0

    messages = _split_slack_messages(text)
    if not messages:
        return []

    token_counts = [len(_tokenize(m)) for m in messages]
    n = len(messages)

    chunks: list[list[str]] = []
    start = 0
    while start < n:
        end = start  # inclusive
        cum = token_counts[start]
        while end + 1 < n:
            nxt = token_counts[end + 1]
            if cum + nxt <= chunk_size:
                end += 1
                cum += nxt
                continue
            # Adding next would overshoot — pick whichever boundary is closer.
            undershoot = chunk_size - cum
            overshoot = (cum + nxt) - chunk_size
            if overshoot < undershoot:
                end += 1
                cum += nxt
            break

        chunks.append(messages[start : end + 1])

        if end + 1 >= n:
            break

        # Next chunk shares the last `overlap_messages` of this one.
        next_start = end + 1 - overlap_messages
        # Force progress: the next chunk must include at least one new message
        # that the current chunk did not.
        if next_start <= start:
            next_start = start + 1
        start = next_start

    return [Chunk(text="\n".join(msgs), index=i) for i, msgs in enumerate(chunks)]
