"""Vector search: Qdrant semantic search with query embedding.

Embeds a natural-language query via the 'embed' role model, then
performs top-k cosine similarity search against the 'fieldnotes'
Qdrant collection. Returns ranked chunks with source metadata
(source_type, source_id, text, date). Supports optional source_type
filtering and configurable top_k.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from worker.config import QdrantConfig
from worker.models.base import EmbedRequest
from worker.models.resolver import ModelRegistry

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 10


@dataclass
class VectorResult:
    """A single ranked chunk from similarity search."""

    source_type: str
    source_id: str
    text: str
    date: str
    score: float
    chunk_index: int = 0


@dataclass
class VectorQueryResult:
    """Structured result from a vector similarity search."""

    question: str
    results: list[VectorResult] = field(default_factory=list)
    error: str | None = None


class VectorQuerier:
    """Semantic similarity search over the Qdrant fieldnotes collection.

    Usage::

        querier = VectorQuerier(registry, qdrant_cfg)
        result = querier.query("What did I write about machine learning?")
        for r in result.results:
            print(r.score, r.text)
        querier.close()
    """

    def __init__(
        self,
        registry: ModelRegistry,
        qdrant_cfg: QdrantConfig | None = None,
    ) -> None:
        qdrant_cfg = qdrant_cfg or QdrantConfig()
        self._registry = registry
        self._collection = qdrant_cfg.collection
        self._qdrant = QdrantClient(host=qdrant_cfg.host, port=qdrant_cfg.port)

    def query(
        self,
        question: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        source_type: str | None = None,
    ) -> VectorQueryResult:
        """Embed *question* and return the closest chunks from Qdrant.

        Parameters
        ----------
        question:
            Natural-language query text to embed and search.
        top_k:
            Maximum number of results to return.
        source_type:
            If set, filter results to only this source type (e.g. "file").
        """
        try:
            # Embed the query text.
            model = self._registry.for_role("embed")
            resp = model.embed(EmbedRequest(texts=[question]))
            query_vector = resp.vectors[0]

            # Build optional filter.
            query_filter: Filter | None = None
            if source_type is not None:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="source_type",
                            match=MatchValue(value=source_type),
                        )
                    ]
                )

            # Search Qdrant.
            hits = self._qdrant.search(
                collection_name=self._collection,
                query_vector=query_vector,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )

            results = [
                VectorResult(
                    source_type=hit.payload.get("source_type", ""),
                    source_id=hit.payload.get("source_id", ""),
                    text=hit.payload.get("text", ""),
                    date=hit.payload.get("date", ""),
                    score=hit.score,
                    chunk_index=hit.payload.get("chunk_index", 0),
                )
                for hit in hits
            ]

            logger.info(
                "Vector search returned %d results for: %.60s",
                len(results),
                question,
            )

            return VectorQueryResult(question=question, results=results)

        except Exception as exc:
            logger.exception("Vector search failed for: %s", question)
            return VectorQueryResult(question=question, error=str(exc))

    def close(self) -> None:
        """Release the Qdrant connection."""
        self._qdrant.close()
        self._qdrant = None  # type: ignore[assignment]
