"""Link Application/Tool nodes to Topic clusters via RELATED_TO_TOPIC edges.

After clustering assigns topic labels, this module:
  1. Queries Neo4j for all Application and Tool nodes with descriptions
  2. Embeds their description text using the same embedding model as chunks
  3. Computes cosine similarity against each topic cluster centroid
  4. Creates RELATED_TO_TOPIC edges for matches above a similarity threshold

Cleanup: on each run, deletes all existing auto_linked RELATED_TO_TOPIC edges
before creating new ones, so stale links don't accumulate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from neo4j import GraphDatabase, Driver
from neo4j.exceptions import ServiceUnavailable, TransientError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from worker.clustering.cluster import ClusterResult
from worker.clustering.labeler import LabeledCluster
from worker.config import Neo4jConfig
from worker.models.base import EmbedRequest
from worker.models.resolver import ModelRegistry

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 3
DEFAULT_SIMILARITY_THRESHOLD = 0.6

_neo4j_retry = retry(
    retry=retry_if_exception_type((TransientError, ServiceUnavailable, OSError)),
    stop=stop_after_attempt(4),
    wait=wait_exponential_jitter(initial=0.5, max=10),
    before_sleep=lambda rs: logger.warning(
        "Neo4j call failed (%s), retry %d", rs.outcome.exception(), rs.attempt_number
    ),
    reraise=True,
)


@dataclass
class AppNode:
    """An Application or Tool node from Neo4j."""

    source_id: str
    label: str  # "Application" or "Tool"
    name: str
    description: str


def link_apps_to_topics(
    labeled: list[LabeledCluster],
    cluster_results: list[ClusterResult],
    registry: ModelRegistry,
    neo4j_cfg: Neo4jConfig | None = None,
    *,
    top_k: int = DEFAULT_TOP_K,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> int:
    """Link Application/Tool nodes to Topic clusters.

    Parameters
    ----------
    labeled:
        Labeled clusters from the labeling step.
    cluster_results:
        Original cluster results with centroids.
    registry:
        Model registry for resolving the embedding model.
    neo4j_cfg:
        Neo4j connection settings.
    top_k:
        Maximum number of topics to link per app/tool.
    similarity_threshold:
        Minimum cosine similarity to create an edge (0-1).

    Returns
    -------
    int
        Number of RELATED_TO_TOPIC edges created.
    """
    if not labeled or not cluster_results:
        return 0

    neo4j_cfg = neo4j_cfg or Neo4jConfig()

    driver = GraphDatabase.driver(
        neo4j_cfg.uri,
        auth=(neo4j_cfg.user, neo4j_cfg.password),
    )
    try:
        apps = _fetch_app_nodes(driver)
        if not apps:
            logger.info("No Application/Tool nodes found, skipping app-topic linking")
            return 0

        # Build description texts for embedding
        texts = _build_description_texts(apps)

        # Embed descriptions
        model = registry.for_role("embed")
        resp = model.embed(EmbedRequest(texts=texts))
        app_vectors = np.array(resp.vectors, dtype=np.float32)

        # Build centroid matrix and label mapping
        centroid_matrix, cluster_id_to_label = _build_centroid_matrix(
            labeled,
            cluster_results,
        )

        # Compute similarities and find matches
        matches = _find_matches(
            apps,
            app_vectors,
            centroid_matrix,
            cluster_id_to_label,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )

        # Write edges to Neo4j
        edge_count = _write_edges(driver, matches)

        logger.info(
            "Linked %d Application/Tool nodes to topics with %d RELATED_TO_TOPIC edges",
            len({m[0] for m in matches}),
            edge_count,
        )
        return edge_count
    finally:
        driver.close()


@_neo4j_retry
def _fetch_app_nodes(driver: Driver) -> list[AppNode]:
    """Fetch all Application and Tool nodes from Neo4j."""
    with driver.session() as session:
        result = session.run(
            """
            MATCH (n)
            WHERE (n:Application OR n:Tool)
              AND n.source_id IS NOT NULL
            RETURN n.source_id AS source_id,
                   labels(n) AS labels,
                   n.name AS name,
                   coalesce(n.description, '') AS description
            """
        )
        nodes: list[AppNode] = []
        for record in result:
            node_labels = record["labels"]
            label = "Application" if "Application" in node_labels else "Tool"
            nodes.append(
                AppNode(
                    source_id=record["source_id"],
                    label=label,
                    name=record["name"] or "",
                    description=record["description"],
                )
            )
        return nodes


def _build_description_texts(apps: list[AppNode]) -> list[str]:
    """Build text representations for embedding."""
    texts: list[str] = []
    for app in apps:
        if app.description:
            texts.append(f"{app.name}: {app.description}")
        else:
            texts.append(app.name)
    return texts


def _build_centroid_matrix(
    labeled: list[LabeledCluster],
    cluster_results: list[ClusterResult],
) -> tuple[np.ndarray, dict[int, str]]:
    """Build a matrix of cluster centroids and a mapping of cluster_id → topic label."""
    cr_by_id = {cr.cluster_id: cr for cr in cluster_results}
    cluster_id_to_label: dict[int, str] = {}
    centroids: list[list[float]] = []

    for lc in labeled:
        cr = cr_by_id.get(lc.cluster_id)
        if cr is None or not cr.centroid:
            continue
        cluster_id_to_label[lc.cluster_id] = lc.label
        centroids.append(cr.centroid)

    if not centroids:
        return np.empty((0, 0), dtype=np.float32), {}

    return np.array(centroids, dtype=np.float32), cluster_id_to_label


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between rows of a and b.

    Parameters
    ----------
    a: shape (N, D) — app embeddings
    b: shape (M, D) — centroid embeddings

    Returns
    -------
    np.ndarray shape (N, M)
    """
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return a_norm @ b_norm.T


def _find_matches(
    apps: list[AppNode],
    app_vectors: np.ndarray,
    centroid_matrix: np.ndarray,
    cluster_id_to_label: dict[int, str],
    *,
    top_k: int,
    similarity_threshold: float,
) -> list[tuple[str, str, str, float]]:
    """Find (source_id, node_label, topic_name, similarity) matches.

    Returns only matches above the similarity threshold, limited to top_k per app.
    """
    if centroid_matrix.size == 0 or app_vectors.size == 0:
        return []

    # cluster_id_to_label is ordered by insertion (Python 3.7+), same as centroid_matrix rows
    topic_labels = list(cluster_id_to_label.values())

    sim_matrix = _cosine_similarity(app_vectors, centroid_matrix)

    matches: list[tuple[str, str, str, float]] = []
    for i, app in enumerate(apps):
        sims = sim_matrix[i]
        # Get top-K indices sorted by similarity descending
        top_indices = np.argsort(sims)[::-1][:top_k]

        for idx in top_indices:
            score = float(sims[idx])
            if score >= similarity_threshold:
                matches.append(
                    (
                        app.source_id,
                        app.label,
                        topic_labels[idx],
                        score,
                    )
                )

    return matches


@_neo4j_retry
def _write_edges(
    driver: Driver,
    matches: list[tuple[str, str, str, float]],
) -> int:
    """Delete old auto-linked edges and write new RELATED_TO_TOPIC edges."""
    with driver.session() as session:
        return session.execute_write(_write_edges_tx, matches)


def _write_edges_tx(
    tx: Any,
    matches: list[tuple[str, str, str, float]],
) -> int:
    """Execute edge writes in a single transaction."""
    # 1. Delete all existing auto-linked RELATED_TO_TOPIC edges
    tx.run(
        """
        MATCH (a)-[r:RELATED_TO_TOPIC {auto_linked: true}]->(t:Topic)
        DELETE r
        """
    )

    # 2. Create new edges
    created = 0
    for source_id, node_label, topic_name, similarity in matches:
        result = tx.run(
            """
            MATCH (a {source_id: $sid})
            WHERE a:Application OR a:Tool
            MATCH (t:Topic {name: $topic_name, source: 'cluster'})
            MERGE (a)-[r:RELATED_TO_TOPIC]->(t)
            SET r.confidence = $similarity,
                r.source = 'clustering',
                r.auto_linked = true,
                r.updated_at = datetime()
            RETURN count(r) AS cnt
            """,
            sid=source_id,
            topic_name=topic_name,
            similarity=similarity,
        )
        created += result.single()["cnt"]

    return created
