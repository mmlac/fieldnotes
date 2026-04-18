"""Embedder — generates 768-dim vectors from text chunks.

Uses ModelRegistry.for_role('embed') to resolve the embedding model
(nomic-embed-text via Ollama by default). Batches embedding calls for
efficiency. Returns list of (chunk_text, vector) tuples.
"""

from __future__ import annotations

import logging

from ..models.base import EmbedRequest
from ..models.resolver import ModelRegistry

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 64

# Safety limit for text sent to the embedding API.  nomic-embed-text has a
# 2048-token context window.  The chars-per-subword-token ratio varies: ~4
# for plain English, but as low as ~2 for URLs, code, or non-Latin text.
# Using 2048 × 3 = 6144, rounded down to 6000 for safety.  This only fires
# for edge cases — the chunker already targets ~512 whitespace tokens.
_MAX_EMBED_CHARS = 6000

# Last-resort truncation when even _MAX_EMBED_CHARS overflows the model's
# context (e.g. dense non-Latin text at ~2 chars/token).  500 chars is
# safely under any embedding model's limit and still captures the gist.
_FALLBACK_CHARS = 500


def _is_context_overflow(exc: BaseException) -> bool:
    """Return True if *exc* is an HTTP 400 from Ollama about context length."""
    try:
        import httpx

        if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 400:
            return True
    except ImportError:
        pass
    return False


def _truncate_for_embed(text: str, max_chars: int = _MAX_EMBED_CHARS) -> str:
    """Truncate *text* to *max_chars* at a word boundary if it exceeds the limit."""
    if len(text) <= max_chars:
        return text
    # Cut at the last whitespace before the limit to avoid splitting a word.
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > max_chars // 2:
        truncated = truncated[:last_space]
    logger.warning(
        "Truncated chunk from %d to %d chars for embedding (context limit)",
        len(text),
        len(truncated),
    )
    return truncated


def embed_chunks(
    chunks: list[str],
    registry: ModelRegistry,
    *,
    role: str = "embed",
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[tuple[str, list[float]]]:
    """Generate embedding vectors for a list of text chunks.

    Parameters
    ----------
    chunks:
        Plain text strings to embed.
    registry:
        ModelRegistry instance with a configured embedding role.
    role:
        Pipeline role name to resolve the embedding model.
    batch_size:
        Number of texts per embedding API call. The underlying Ollama
        provider iterates one text at a time, but batching here keeps
        request grouping and logging manageable.

    Returns
    -------
    list[tuple[str, list[float]]]
        (chunk_text, vector) pairs in input order.
    """
    if not chunks:
        return []

    model = registry.for_role(role)
    results: list[tuple[str, list[float]]] = []
    skipped = 0

    for start in range(0, len(chunks), batch_size):
        batch = [_truncate_for_embed(c) for c in chunks[start : start + batch_size]]
        try:
            resp = model.embed(EmbedRequest(texts=batch))
        except Exception as batch_exc:
            if not _is_context_overflow(batch_exc):
                raise
            # A chunk in this batch exceeds the model's context window.
            # Fall back to embedding one at a time so only the offending
            # chunk is skipped — the rest of the document still gets vectors.
            logger.warning(
                "Batch embed failed (context overflow) — retrying %d chunks individually",
                len(batch),
            )
            for text in batch:
                try:
                    single = model.embed(EmbedRequest(texts=[text]))
                    results.append((text, single.vectors[0]))
                except Exception as single_exc:
                    if not _is_context_overflow(single_exc):
                        raise
                    # Aggressively truncate and retry so the chunk still
                    # gets a vector (even if imprecise) and the downstream
                    # vector count matches the chunk count.
                    fallback = text[:_FALLBACK_CHARS]
                    try:
                        single = model.embed(EmbedRequest(texts=[fallback]))
                        results.append((text, single.vectors[0]))
                    except Exception:
                        skipped += 1
                        logger.warning(
                            "Skipping chunk (%d chars) — exceeds embedding "
                            "context limit even after truncation",
                            len(text),
                        )
            continue

        if len(resp.vectors) != len(batch):
            raise ValueError(f"Expected {len(batch)} vectors, got {len(resp.vectors)}")

        for text, vec in zip(batch, resp.vectors):
            results.append((text, vec))

        logger.debug(
            "Embedded batch %d–%d (%d tokens)",
            start,
            start + len(batch),
            resp.input_tokens,
        )

    if skipped:
        logger.warning(
            "Skipped %d/%d chunks due to context overflow — "
            "document indexed with partial vectors",
            skipped,
            len(chunks),
        )
    logger.info(
        "Embedded %d/%d chunks via %s/%s",
        len(results),
        len(chunks),
        model.provider.provider_type,
        model.model,
    )
    return results
