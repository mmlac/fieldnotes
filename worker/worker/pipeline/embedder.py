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
# 2048-token context window; at ~4 chars per subword token that's ~8192 chars.
# We truncate at 8000 to leave margin.  This only fires for edge cases —
# the chunker already targets ~512 whitespace tokens (~2000 chars).
_MAX_EMBED_CHARS = 8000


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

    for start in range(0, len(chunks), batch_size):
        batch = [_truncate_for_embed(c) for c in chunks[start : start + batch_size]]
        resp = model.embed(EmbedRequest(texts=batch))

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

    logger.info(
        "Embedded %d chunks via %s/%s",
        len(results),
        model.provider.provider_type,
        model.model,
    )
    return results
