"""Second-stage reranker for hybrid search.

Vector search returns the top-K most-similar chunks by embedding cosine
distance.  A cross-encoder (e.g. ``BAAI/bge-reranker-v2-m3``) scores
each ``(query, chunk_text)`` pair *together* and produces a much sharper
relevance signal than dense embeddings alone — at the cost of one model
call per pair.  The reranker takes a larger candidate pool (``top_k_pre``)
and trims to a smaller, better-ordered set (``top_k_post``).

Two implementations:

- :class:`NullReranker` — pass-through, used when reranking is disabled
  or no ``rerank`` role is configured.
- :class:`CrossEncoderReranker` — wraps a :class:`ResolvedModel` whose
  provider implements ``rerank()`` (currently the
  ``sentence_transformers`` provider).

The :class:`Reranker` Protocol is intentionally narrow so additional
backends (Cohere, Voyage, TEI) can slot in as new provider types
without touching the call sites.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from worker.config import RerankerConfig
from worker.metrics import (
    RERANKER_DROPPED_BELOW_THRESHOLD,
    RERANKER_DURATION_SECONDS,
    RERANKER_REORDER_DISTANCE,
)
from worker.models.resolver import ModelRegistry, ResolvedModel
from worker.query.vector import VectorResult

logger = logging.getLogger(__name__)


@dataclass
class RerankCandidate:
    """A passage to rerank, paired with its original ranking metadata.

    ``payload`` carries the original :class:`VectorResult` so the caller
    can swap reordered candidates back into the hybrid context without
    losing source metadata.
    """

    text: str
    payload: VectorResult
    original_score: float
    rerank_score: float = 0.0


class Reranker(Protocol):
    """Pluggable second-stage reranker."""

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        top_k: int,
    ) -> list[RerankCandidate]: ...


class NullReranker:
    """Pass-through reranker — keeps original order, trims to ``top_k``."""

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        top_k: int,
    ) -> list[RerankCandidate]:
        return candidates[:top_k]


class CrossEncoderReranker:
    """Reranker backed by a cross-encoder ``ResolvedModel``.

    On each call:

    1. Score every candidate's text against the query in batches of
       ``batch_size``.
    2. Drop candidates whose score is below ``score_threshold``.
    3. Sort by descending rerank score, return the top ``top_k``.
    4. Record reorder distance + duration metrics.

    On any provider failure the original candidate order is preserved
    (truncated to ``top_k``) and a warning is logged.  A failing reranker
    must never break search.
    """

    def __init__(
        self,
        model: ResolvedModel,
        *,
        score_threshold: float = 0.0,
        batch_size: int = 32,
    ) -> None:
        self._model = model
        self._score_threshold = score_threshold
        self._batch_size = max(1, batch_size)

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        top_k: int,
    ) -> list[RerankCandidate]:
        if not candidates or top_k <= 0:
            return []

        passages = [c.text for c in candidates]

        import time

        start = time.monotonic()
        try:
            scores: list[float] = []
            for i in range(0, len(passages), self._batch_size):
                batch = passages[i : i + self._batch_size]
                scores.extend(self._model.rerank(query, batch))
        except Exception:
            elapsed = time.monotonic() - start
            RERANKER_DURATION_SECONDS.labels(backend="cross_encoder").observe(elapsed)
            logger.exception(
                "Reranker failed (%d candidates) — falling back to original order",
                len(candidates),
            )
            return candidates[:top_k]

        elapsed = time.monotonic() - start
        RERANKER_DURATION_SECONDS.labels(backend="cross_encoder").observe(elapsed)

        if len(scores) != len(candidates):
            logger.error(
                "Reranker returned %d scores for %d candidates — falling back",
                len(scores),
                len(candidates),
            )
            return candidates[:top_k]

        for cand, score in zip(candidates, scores):
            cand.rerank_score = float(score)

        kept = [c for c in candidates if c.rerank_score >= self._score_threshold]
        dropped = len(candidates) - len(kept)
        if dropped:
            RERANKER_DROPPED_BELOW_THRESHOLD.inc(dropped)

        kept.sort(key=lambda c: c.rerank_score, reverse=True)
        result = kept[:top_k]

        if result:
            # How far did the new top-1 sit in the original order?
            # 0 means rerank agreed with the embedding-based ranking.
            top_payload_id = result[0].payload.source_id, result[0].payload.chunk_index
            for original_index, c in enumerate(candidates):
                if (c.payload.source_id, c.payload.chunk_index) == top_payload_id:
                    RERANKER_REORDER_DISTANCE.observe(original_index)
                    break

        logger.info(
            "Reranked %d candidates → %d kept in %.3fs (top score=%.3f)",
            len(candidates),
            len(result),
            elapsed,
            result[0].rerank_score if result else 0.0,
        )
        return result


def build_reranker(
    cfg: RerankerConfig,
    registry: ModelRegistry,
) -> Reranker:
    """Construct the appropriate reranker for the running config.

    Returns a :class:`NullReranker` when reranking is disabled or no
    ``rerank`` role is bound — callers don't need to branch on whether
    the feature is on.
    """
    if not cfg.enabled:
        return NullReranker()
    try:
        model = registry.for_role("rerank")
    except KeyError:
        logger.info(
            "Reranker enabled but no model is bound to the 'rerank' role — "
            "search will use unranked vector results"
        )
        return NullReranker()
    return CrossEncoderReranker(
        model,
        score_threshold=cfg.score_threshold,
        batch_size=cfg.batch_size,
    )


def candidates_from_vector_results(
    results: list[VectorResult],
) -> list[RerankCandidate]:
    """Build rerank candidates from a vector search result list."""
    return [
        RerankCandidate(
            text=r.text,
            payload=r,
            original_score=r.score,
        )
        for r in results
    ]
