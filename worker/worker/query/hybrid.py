"""Hybrid query merge: combine graph + vector results with dedup.

Merges results from graph query (high precision) and vector search
(high recall), deduplicates by source_id, and formats the merged
context as a structured prompt fragment with [Graph context] and
[Semantic context] sections for LLM consumption.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from worker.query.graph import GraphQueryResult
from worker.query.vector import VectorQueryResult, VectorResult

logger = logging.getLogger(__name__)


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
) -> HybridResult:
    """Merge graph and vector results with dedup by source_id.

    Graph results are ranked first (higher precision). Vector results
    fill gaps (higher recall) after removing duplicates already covered
    by graph results.

    Parameters
    ----------
    question:
        The original natural-language query.
    graph:
        Result from :class:`GraphQuerier.query`.
    vector:
        Result from :class:`VectorQuerier.query`.

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

    logger.info(
        "Hybrid merge: %d graph rows, %d vector results (%d after dedup)",
        len(graph.raw_results),
        len(vector.results),
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
