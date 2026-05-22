"""Hybrid query merge: combine graph + vector results with dedup.

Merges results from graph query (high precision) and vector search
(high recall), deduplicates by source_id, and formats the merged
context as a structured prompt fragment with [Graph context] and
[Semantic context] sections for LLM consumption.

Optionally post-filters vector results by ``date_window`` (drops
out-of-window results) and ``require_journal_folder`` (drops results
whose source_id doesn't match any configured journal-folder pattern).
These exist because, without them, "Summarize my journal entries of
the last 7 days" pulls in random PDFs and old emails that just happen
to score well on semantic similarity to "journal".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime

from worker.query.graph import GraphQueryResult
from worker.query.reranker import (
    NullReranker,
    Reranker,
    candidates_from_vector_results,
)
from worker.query.vector import VectorQueryResult, VectorResult

logger = logging.getLogger(__name__)


def _parse_iso_date(value: str) -> date | None:
    """Best-effort parse of VectorResult.date into a calendar date.

    Accepts ``YYYY-MM-DD`` and ``YYYY-MM-DDTHH:MM:SS[Z|+offset]`` forms.
    Returns None when the value is empty or unparseable — caller treats
    that as "unknown date, keep" to avoid losing data.
    """
    if not value:
        return None
    head = value.split("T", 1)[0]
    try:
        return datetime.strptime(head, "%Y-%m-%d").date()
    except ValueError:
        return None


def _filter_vector_results(
    results: list[VectorResult],
    *,
    date_window: tuple[date, date] | None,
    require_journal_folder: list[str] | None,
) -> tuple[list[VectorResult], int, int]:
    """Drop results that fall outside the date window or journal-folder set.

    Returns ``(kept, dropped_by_date, dropped_by_folder)``. A result with
    no parseable date is kept (the filter is best-effort strict — we don't
    drop on missing data).
    """
    dropped_date = 0
    dropped_folder = 0
    kept: list[VectorResult] = []
    for r in results:
        if date_window is not None:
            d = _parse_iso_date(r.date)
            if d is not None and not (date_window[0] <= d <= date_window[1]):
                dropped_date += 1
                continue
        if require_journal_folder:
            sid_lower = r.source_id.lower()
            matched = any(p.lower() in sid_lower for p in require_journal_folder)
            if not matched:
                dropped_folder += 1
                continue
        kept.append(r)
    return kept, dropped_date, dropped_folder


@dataclass
class HybridResult:
    """Merged result from graph and vector queries."""

    question: str
    graph_results: list[dict[str, object]] = field(default_factory=list)
    vector_results: list[VectorResult] = field(default_factory=list)
    context: str = ""
    errors: list[str] = field(default_factory=list)


def _extract_source_ids_from_graph(raw_results: list[dict[str, object]]) -> set[str]:
    """Extract source_id values from graph query raw results."""
    ids: set[str] = set()
    for row in raw_results:
        for value in row.values():
            if isinstance(value, dict):
                sid = value.get("source_id")
                if sid:
                    ids.add(str(sid))
            elif isinstance(value, str):
                # Some graph results return source_id as a direct value.
                pass
        # Also check top-level source_id.
        sid = row.get("source_id")
        if sid:
            ids.add(str(sid))
    return ids


def _format_graph_section(
    graph: GraphQueryResult,
) -> str:
    """Format graph results as a prompt section."""
    if graph.error or not graph.raw_results:
        return ""

    lines: list[str] = ["[Graph context]"]

    if graph.answer:
        lines.append(graph.answer)

    for row in graph.raw_results:
        parts = []
        for key, value in row.items():
            if isinstance(value, dict):
                # Node properties — flatten to readable form.
                props = ", ".join(f"{k}: {v}" for k, v in value.items())
                parts.append(f"{key}({props})")
            else:
                parts.append(f"{key}: {value}")
        if parts:
            lines.append("- " + "; ".join(parts))

    return "\n".join(lines)


def _format_vector_section(results: list[VectorResult]) -> str:
    """Format vector results as a prompt section."""
    if not results:
        return ""

    lines: list[str] = ["[Semantic context]"]
    for r in results:
        header = f"- [{r.source_type}] {r.source_id}"
        if r.date:
            header += f" ({r.date})"
        lines.append(header)
        lines.append(f"  {r.text}")

    return "\n".join(lines)


def merge(
    question: str,
    graph: GraphQueryResult,
    vector: VectorQueryResult,
    *,
    reranker: Reranker | None = None,
    top_k_post: int | None = None,
    date_window: tuple[date, date] | None = None,
    require_journal_folder: list[str] | None = None,
) -> HybridResult:
    """Merge graph and vector results with dedup by source_id.

    Graph results are ranked first (higher precision). Vector results
    fill gaps (higher recall) after removing duplicates already covered
    by graph results.  When *reranker* is provided, the deduped vector
    list is re-scored and trimmed to *top_k_post*.

    Parameters
    ----------
    question:
        The original natural-language query.
    graph:
        Result from :class:`GraphQuerier.query`.
    vector:
        Result from :class:`VectorQuerier.query`.
    reranker:
        Optional second-stage reranker.  When omitted the vector list is
        returned in original (embedding-cosine) order.  Graph results
        are never reranked — they're the precision lane.
    top_k_post:
        How many vector results to keep after reranking.  Ignored when
        *reranker* is None or :class:`NullReranker`.  Defaults to the
        full deduped list.
    date_window:
        Optional ``(start, end)`` inclusive calendar-date window. Vector
        results whose ``date`` parses to a value outside the window are
        dropped before reranking. Results with no parseable date are
        kept (best-effort strict). Graph results are never filtered here;
        the Cypher should have done that.
    require_journal_folder:
        Optional list of path substrings (case-insensitive). When set,
        vector results whose ``source_id`` doesn't contain any of these
        substrings are dropped. Typically driven by
        :func:`worker.query._question_time.mentions_journal` +
        ``[retrieval] journal_folder_patterns``.

    Returns
    -------
    HybridResult
        Merged, deduplicated results with a formatted context string.
    """
    errors: list[str] = []
    if graph.error:
        errors.append(f"graph: {graph.error}")
    if vector.error:
        errors.append(f"vector: {vector.error}")

    # Collect source_ids already covered by graph results.
    graph_source_ids = _extract_source_ids_from_graph(graph.raw_results)

    # Deduplicate vector results: keep only those not in graph results.
    deduped_vector: list[VectorResult] = [
        r for r in vector.results if r.source_id not in graph_source_ids
    ]

    if date_window is not None or require_journal_folder:
        deduped_vector, dropped_date, dropped_folder = _filter_vector_results(
            deduped_vector,
            date_window=date_window,
            require_journal_folder=require_journal_folder,
        )
        if dropped_date or dropped_folder:
            logger.info(
                "Vector post-filter: dropped %d by date, %d by journal-folder",
                dropped_date,
                dropped_folder,
            )

    pre_rerank_count = len(deduped_vector)
    if reranker is not None and not isinstance(reranker, NullReranker) and deduped_vector:
        candidates = candidates_from_vector_results(deduped_vector)
        keep_n = top_k_post if top_k_post is not None else len(candidates)
        reranked = reranker.rerank(question, candidates, keep_n)
        deduped_vector = [c.payload for c in reranked]

    logger.info(
        "Hybrid merge: %d graph rows, %d vector results (%d after dedup, %d after rerank)",
        len(graph.raw_results),
        len(vector.results),
        pre_rerank_count,
        len(deduped_vector),
    )

    # Build formatted context.
    sections: list[str] = []
    graph_section = _format_graph_section(graph)
    if graph_section:
        sections.append(graph_section)
    vector_section = _format_vector_section(deduped_vector)
    if vector_section:
        sections.append(vector_section)

    context = "\n\n".join(sections)

    return HybridResult(
        question=question,
        graph_results=graph.raw_results,
        vector_results=deduped_vector,
        context=context,
        errors=errors,
    )
