"""Connection suggestions: surface semantically similar but unlinked documents.

Finds pairs of documents that are close in vector space but have no direct
edge in the Neo4j graph.  Useful for discovering latent relationships across
source types (e.g. an Obsidian note related to an OmniFocus task).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from neo4j import Driver
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from worker.config import Neo4jConfig, QdrantConfig
from worker.neo4j_driver import build_driver

logger = logging.getLogger(__name__)

# Default parameters
_DEFAULT_THRESHOLD = 0.82
_DEFAULT_LIMIT = 20
_DEFAULT_SEED_COUNT = 20
_DEFAULT_VECTOR_TOP_K = 10


@dataclass
class SuggestedConnection:
    """A pair of semantically similar documents with no graph edge."""

    source_a: str
    source_b: str
    label_a: str
    label_b: str
    title_a: str
    title_b: str
    source_type_a: str
    source_type_b: str
    similarity: float
    reason: str = "high vector similarity, no graph edge"


@dataclass
class ConnectionResult:
    """Result from a connection suggestion query."""

    suggestions: list[SuggestedConnection] = field(default_factory=list)
    checked: int = 0
    error: str | None = None


class ConnectionQuerier:
    """Find semantically similar but unlinked document pairs.

    Usage::

        querier = ConnectionQuerier(neo4j_cfg, qdrant_cfg)
        result = querier.suggest(limit=20, threshold=0.82)
        for s in result.suggestions:
            print(s.similarity, s.title_a, "<->", s.title_b)
        querier.close()
    """

    def __init__(
        self,
        neo4j_cfg: Neo4jConfig | None = None,
        qdrant_cfg: QdrantConfig | None = None,
    ) -> None:
        neo4j_cfg = neo4j_cfg or Neo4jConfig()
        qdrant_cfg = qdrant_cfg or QdrantConfig()
        self._driver: Driver = build_driver(neo4j_cfg.uri, neo4j_cfg.user, neo4j_cfg.password)
        self._qdrant = QdrantClient(host=qdrant_cfg.host, port=qdrant_cfg.port)
        self._collection = qdrant_cfg.collection

    def suggest(
        self,
        *,
        source_id: str | None = None,
        source_type: str | None = None,
        threshold: float = _DEFAULT_THRESHOLD,
        limit: int = _DEFAULT_LIMIT,
        cross_source: bool = False,
    ) -> ConnectionResult:
        """Find document pairs that are similar but not connected.

        Parameters
        ----------
        source_id:
            If set, use this single document as the only seed.
        source_type:
            If set, seed from documents of this source type.
        threshold:
            Minimum cosine similarity score to consider (0–1).
        limit:
            Maximum number of suggestions to return.
        cross_source:
            If True, only return pairs where source types differ.
        """
        try:
            # Step 1: Select seed documents
            seeds = self._select_seeds(source_id=source_id, source_type=source_type)
            if not seeds:
                return ConnectionResult(suggestions=[], checked=0)

            # Step 2: For each seed, find similar vectors from Qdrant
            candidate_pairs: dict[tuple[str, str], float] = {}
            for seed in seeds:
                seed_sid = seed.get("source_id", "")
                seed_vector = seed.get("vector")
                if not seed_sid or not seed_vector:
                    continue
                hits = self._similar_vectors(
                    seed_vector,
                    exclude_source_id=seed_sid,
                    source_type_filter=None,
                    threshold=threshold,
                )
                for hit_sid, hit_score in hits:
                    # Canonical pair key: alphabetical order to deduplicate
                    pair = (
                        (seed_sid, hit_sid)
                        if seed_sid < hit_sid
                        else (hit_sid, seed_sid)
                    )
                    # Keep the best similarity score for this pair
                    if hit_score > candidate_pairs.get(pair, -1):
                        candidate_pairs[pair] = hit_score

            checked = len(candidate_pairs)
            if not candidate_pairs:
                return ConnectionResult(suggestions=[], checked=0)

            # Step 3: Batch-check Neo4j for existing edges
            pairs_list = [{"a": a, "b": b} for (a, b) in candidate_pairs]
            edge_counts = self._batch_edge_check(pairs_list)

            # Step 4: Filter to unconnected pairs
            unconnected: list[tuple[str, str, float]] = []
            for (a, b), score in candidate_pairs.items():
                pair_key = (a, b) if a < b else (b, a)
                if edge_counts.get((pair_key[0], pair_key[1]), 0) == 0:
                    unconnected.append((a, b, score))

            if not unconnected:
                return ConnectionResult(suggestions=[], checked=checked)

            # Step 5: Enrich with Neo4j node metadata
            all_source_ids = {sid for (a, b, _) in unconnected for sid in (a, b)}
            node_info = self._fetch_node_info(list(all_source_ids))

            # Step 6: Build suggestions, apply cross_source filter, sort
            suggestions: list[SuggestedConnection] = []
            for a, b, score in unconnected:
                info_a = node_info.get(a, {})
                info_b = node_info.get(b, {})
                st_a = info_a.get("source_type", "")
                st_b = info_b.get("source_type", "")

                if cross_source and st_a == st_b:
                    continue

                suggestions.append(
                    SuggestedConnection(
                        source_a=a,
                        source_b=b,
                        label_a=info_a.get("label", "Document"),
                        label_b=info_b.get("label", "Document"),
                        title_a=info_a.get("title", a),
                        title_b=info_b.get("title", b),
                        source_type_a=st_a,
                        source_type_b=st_b,
                        similarity=round(score, 4),
                    )
                )

            suggestions.sort(key=lambda s: s.similarity, reverse=True)
            return ConnectionResult(
                suggestions=suggestions[:limit],
                checked=checked,
            )

        except Exception as exc:
            logger.exception("ConnectionQuerier.suggest failed")
            return ConnectionResult(error=str(exc))

    # -- internal helpers --------------------------------------------------

    def _select_seeds(
        self,
        *,
        source_id: str | None,
        source_type: str | None,
    ) -> list[dict[str, Any]]:
        """Return seed documents with their Qdrant vectors."""
        if source_id is not None:
            # Single document seed
            results, _ = self._qdrant.scroll(
                collection_name=self._collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="source_id",
                            match=MatchValue(value=source_id),
                        )
                    ]
                ),
                limit=1,
                with_vectors=True,
                with_payload=True,
            )
            return [
                {
                    "source_id": r.payload.get("source_id", ""),
                    "vector": r.vector,
                }
                for r in results
                if r.vector
            ]

        # Sample recent documents
        scroll_filter: Filter | None = None
        if source_type is not None:
            scroll_filter = Filter(
                must=[
                    FieldCondition(
                        key="source_type",
                        match=MatchValue(value=source_type),
                    )
                ]
            )

        results, _ = self._qdrant.scroll(
            collection_name=self._collection,
            scroll_filter=scroll_filter,
            limit=_DEFAULT_SEED_COUNT,
            with_vectors=True,
            with_payload=True,
        )

        # Deduplicate by source_id: keep first chunk per document
        seen: set[str] = set()
        seeds: list[dict[str, Any]] = []
        for r in results:
            sid = r.payload.get("source_id", "")
            if sid and sid not in seen and r.vector:
                seen.add(sid)
                seeds.append({"source_id": sid, "vector": r.vector})

        return seeds

    def _similar_vectors(
        self,
        vector: list[float],
        *,
        exclude_source_id: str,
        source_type_filter: str | None,
        threshold: float,
    ) -> list[tuple[str, float]]:
        """Return (source_id, score) pairs for vectors similar to *vector*.

        Excludes chunks belonging to the seed document itself.
        """
        qfilter: Filter | None = None
        if source_type_filter is not None:
            qfilter = Filter(
                must=[
                    FieldCondition(
                        key="source_type",
                        match=MatchValue(value=source_type_filter),
                    )
                ]
            )

        hits = self._qdrant.search(
            collection_name=self._collection,
            query_vector=vector,
            limit=_DEFAULT_VECTOR_TOP_K + 5,  # slight over-fetch to cover exclusions
            query_filter=qfilter,
            with_payload=True,
            score_threshold=threshold,
        )

        # Collect unique source_ids, skip the seed itself
        seen: set[str] = set()
        results: list[tuple[str, float]] = []
        for hit in hits:
            sid = hit.payload.get("source_id", "")
            if not sid or sid == exclude_source_id or sid in seen:
                continue
            seen.add(sid)
            results.append((sid, hit.score))
            if len(results) >= _DEFAULT_VECTOR_TOP_K:
                break

        return results

    def _batch_edge_check(
        self,
        pairs: list[dict[str, str]],
    ) -> dict[tuple[str, str], int]:
        """Return edge counts for all (a, b) pairs in one Cypher query."""
        if not pairs:
            return {}

        with self._driver.session() as session:
            records = session.run(
                """
                UNWIND $pairs AS pair
                OPTIONAL MATCH (a {source_id: pair.a})-[r]-(b {source_id: pair.b})
                RETURN pair.a AS a, pair.b AS b, count(r) AS edge_count
                """,
                pairs=pairs,
            ).data()

        return {(r["a"], r["b"]): r["edge_count"] for r in records}

    def _fetch_node_info(
        self,
        source_ids: list[str],
    ) -> dict[str, dict[str, str]]:
        """Return label, title, and source_type for a list of source_ids."""
        if not source_ids:
            return {}

        with self._driver.session() as session:
            records = session.run(
                """
                UNWIND $ids AS sid
                MATCH (n {source_id: sid})
                RETURN sid,
                       labels(n) AS labels,
                       COALESCE(n.name, n.title, n.source_id, sid) AS title,
                       COALESCE(n.source_type, '') AS source_type
                LIMIT $limit
                """,
                ids=source_ids,
                limit=len(source_ids) * 2,
            ).data()

        info: dict[str, dict[str, str]] = {}
        for r in records:
            sid = r["sid"]
            if sid in info:
                continue
            node_labels = [lb for lb in (r["labels"] or []) if lb not in ("Chunk",)]
            info[sid] = {
                "label": node_labels[0] if node_labels else "Document",
                "title": r["title"] or sid,
                "source_type": r["source_type"],
            }

        return info

    def close(self) -> None:
        """Release Neo4j and Qdrant connections."""
        self._driver.close()
        self._qdrant.close()

    def __enter__(self) -> ConnectionQuerier:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
