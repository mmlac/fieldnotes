"""Query utilities."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

EMPTY_CORPUS_MESSAGE = (
    "Your knowledge graph is empty. Run `fieldnotes serve --daemon` to start indexing."
)


def is_corpus_empty(
    graph_querier,
    vector_querier,
) -> bool:
    """Return *True* when both Neo4j and Qdrant contain zero indexed data.

    Checks Qdrant point count and Neo4j source-node count.  If either store
    has data the corpus is considered non-empty.  Connection errors are logged
    and treated as "unknown" (returns *False* to avoid false positives).
    """
    qdrant_count: int | None = None
    neo4j_count: int | None = None

    # --- Qdrant ---
    try:
        info = vector_querier._qdrant.get_collection(vector_querier._collection)
        qdrant_count = int(info.points_count or 0)
    except Exception:
        logger.debug("Corpus-empty check: Qdrant query failed", exc_info=True)

    # --- Neo4j ---
    try:
        with graph_querier._driver.session() as session:
            row = session.run(
                "MATCH (n) WHERE n:File OR n:Email OR n:ObsidianNote "
                "OR n:Repository OR n:Chunk "
                "RETURN count(n) AS cnt"
            ).single()
            neo4j_count = int(row["cnt"]) if row else 0
    except Exception:
        logger.debug("Corpus-empty check: Neo4j query failed", exc_info=True)

    # If we couldn't reach either store, don't claim empty.
    if qdrant_count is None and neo4j_count is None:
        return False

    # Corpus is empty only when every reachable store reports zero.
    if qdrant_count is not None and qdrant_count > 0:
        return False
    if neo4j_count is not None and neo4j_count > 0:
        return False

    return True
