"""Sentence-aware text splitter with overlap.

Splits text into chunks of ~512 tokens with 64-token overlap, using sentence
boundaries as preferred split points. Chunks shorter than 100 tokens are merged
with adjacent chunks. No model dependencies — uses whitespace tokenisation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Sentence-ending punctuation followed by whitespace or end-of-string.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

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

    Returns
    -------
    list[Chunk]
        Ordered chunks with their positional indices.
    """
    if not text or not text.strip():
        return []

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

    return [
        Chunk(text=_detokenize(tokens), index=i)
        for i, tokens in enumerate(chunks)
    ]


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
