"""Write clustering results to Neo4j as Topic nodes with TAGGED edges.

For each labeled cluster:
  1. Create/update a Topic node (source='cluster', name=label, description)
  2. Resolve chunk IDs to source_ids via Qdrant payload lookup
  3. Create TAGGED edges from source nodes (File/Email/etc) to the Topic

Idempotency: before writing, deletes all existing TAGGED edges where
source='cluster' and cleans up orphaned cluster-derived Topic nodes.
NEVER touches TAGGED_BY_USER edges (source='user').
"""

from __future__ import annotations

import logging
from typing import Any

from neo4j import GraphDatabase, Driver
from neo4j.exceptions import ServiceUnavailable, TransientError
from qdrant_client import QdrantClient
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from worker.clustering.cluster import ClusterResult
from worker.clustering.labeler import LabeledCluster
from worker.config import Neo4jConfig, QdrantConfig

logger = logging.getLogger(__name__)

_SCROLL_BATCH = 256

_neo4j_retry = retry(
    retry=retry_if_exception_type((TransientError, ServiceUnavailable, OSError)),
    stop=stop_after_attempt(4),
    wait=wait_exponential_jitter(initial=0.5, max=10),
    before_sleep=lambda rs: logger.warning(
        "Neo4j call failed (%s), retry %d", rs.outcome.exception(), rs.attempt_number
    ),
    reraise=True,
)


def write_clusters(
    labeled: list[LabeledCluster],
    cluster_results: list[ClusterResult],
    neo4j_cfg: Neo4jConfig | None = None,
    qdrant_cfg: QdrantConfig | None = None,
) -> None:
    """Write labeled clusters to Neo4j as Topic nodes with TAGGED edges.

    Parameters
    ----------
    labeled:
        Labeled clusters from the labeling step.
    cluster_results:
        Original cluster results (used for chunk_id → source_id resolution).
        Must correspond 1:1 with *labeled* by cluster_id.
    neo4j_cfg:
        Neo4j connection settings. Uses defaults if None.
    qdrant_cfg:
        Qdrant connection settings (for resolving chunk IDs to source IDs).
        Uses defaults if None.
    """
    if not labeled:
        logger.info("No clusters to write")
        return

    neo4j_cfg = neo4j_cfg or Neo4jConfig()
    qdrant_cfg = qdrant_cfg or QdrantConfig()

    # Build cluster_id → chunk_ids mapping from ClusterResults
    chunk_ids_by_cluster = {cr.cluster_id: cr.chunk_ids for cr in cluster_results}

    # Resolve chunk IDs to source IDs via Qdrant
    source_map = _resolve_chunk_sources(chunk_ids_by_cluster, qdrant_cfg)

    driver = GraphDatabase.driver(
        neo4j_cfg.uri,
        auth=(neo4j_cfg.user, neo4j_cfg.password),
    )
    try:
        _write_to_neo4j(driver, labeled, source_map)
    finally:
        driver.close()

    topic_count = len(labeled)
    edge_count = sum(len(sids) for sids in source_map.values())
    logger.info(
        "Wrote %d cluster-derived topics with %d TAGGED edges",
        topic_count,
        edge_count,
    )


def _resolve_chunk_sources(
    chunk_ids_by_cluster: dict[int, list[str]],
    qdrant_cfg: QdrantConfig,
) -> dict[int, set[str]]:
    """Map cluster_id → set of unique source_ids via Qdrant point lookup.

    Parameters
    ----------
    chunk_ids_by_cluster:
        Mapping of cluster_id → list of Qdrant point IDs.
    qdrant_cfg:
        Qdrant connection settings.

    Returns
    -------
    dict[int, set[str]]
        Mapping of cluster_id → unique source_ids found in that cluster.
    """
    all_ids: list[str] = []
    for chunk_ids in chunk_ids_by_cluster.values():
        all_ids.extend(chunk_ids)

    if not all_ids:
        return {cid: set() for cid in chunk_ids_by_cluster}

    client = QdrantClient(host=qdrant_cfg.host, port=qdrant_cfg.port)
    try:
        id_to_source: dict[str, str] = {}
        for i in range(0, len(all_ids), _SCROLL_BATCH):
            batch = all_ids[i : i + _SCROLL_BATCH]
            points = client.retrieve(
                collection_name=qdrant_cfg.collection,
                ids=batch,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                sid = point.payload.get("source_id", "") if point.payload else ""
                if sid:
                    id_to_source[str(point.id)] = sid
    finally:
        client.close()

    result: dict[int, set[str]] = {}
    for cluster_id, chunk_ids in chunk_ids_by_cluster.items():
        source_ids = set()
        for cid in chunk_ids:
            sid = id_to_source.get(cid)
            if sid:
                source_ids.add(sid)
        result[cluster_id] = source_ids

    return result


@_neo4j_retry
def _write_to_neo4j(
    driver: Driver,
    clusters: list[LabeledCluster],
    source_map: dict[int, set[str]],
) -> None:
    """Write all cluster topics and edges in a single transaction."""
    with driver.session() as session:
        session.execute_write(_write_tx, clusters, source_map)


def _write_tx(
    tx: Any,
    clusters: list[LabeledCluster],
    source_map: dict[int, set[str]],
) -> None:
    """Execute all Neo4j writes within a single transaction.

    1. Delete existing TAGGED edges where source='cluster'
    2. Delete orphaned cluster-derived Topic nodes
    3. Batch-create Topic nodes for all clusters (single UNWIND query)
    4. Batch-create TAGGED edges for all source→topic pairs (single UNWIND query)
    """
    _delete_cluster_tagged_edges(tx)
    _delete_orphaned_cluster_topics(tx)

    if clusters:
        _batch_upsert_topic_nodes(tx, clusters)

    edge_data = [
        {"sid": source_id, "name": cluster.label}
        for cluster in clusters
        for source_id in source_map.get(cluster.cluster_id, set())
    ]
    if edge_data:
        _batch_create_tagged_edges(tx, edge_data)


def _delete_cluster_tagged_edges(tx: Any) -> None:
    """Delete all TAGGED relationships where source='cluster'.

    NEVER touches TAGGED_BY_USER edges or any TAGGED edges
    with source != 'cluster'.
    """
    tx.run(
        """
        MATCH ()-[r:TAGGED {source: 'cluster'}]->()
        DELETE r
        """
    )


def _delete_orphaned_cluster_topics(tx: Any) -> None:
    """Delete Topic nodes with source='cluster' that have no remaining edges."""
    tx.run(
        """
        MATCH (t:Topic {source: 'cluster'})
        WHERE NOT EXISTS { (t)--() }
        DELETE t
        """
    )


def _batch_upsert_topic_nodes(tx: Any, clusters: list[LabeledCluster]) -> None:
    """Batch MERGE Topic nodes for all clusters in a single UNWIND query."""
    tx.run(
        """
        UNWIND $topics AS t
        MERGE (topic:Topic {name: t.name, source: 'cluster'})
        SET topic.description = t.description
        """,
        topics=[{"name": c.label, "description": c.description} for c in clusters],
    )


def _batch_create_tagged_edges(tx: Any, edge_data: list[dict]) -> None:
    """Batch MERGE TAGGED edges for all (source_id, topic_name) pairs in a single UNWIND query."""
    tx.run(
        """
        UNWIND $edges AS e
        MATCH (s {source_id: e.sid})
        WHERE s:File OR s:Email OR s:Commit OR s:Image
        MATCH (t:Topic {name: e.name, source: 'cluster'})
        MERGE (s)-[:TAGGED {source: 'cluster'}]->(t)
        """,
        edges=edge_data,
    )


def _upsert_topic_node(tx: Any, cluster: LabeledCluster) -> None:
    """Create or update a Topic node for a labeled cluster."""
    tx.run(
        """
        MERGE (t:Topic {name: $name, source: 'cluster'})
        SET t.description = $description
        """,
        name=cluster.label,
        description=cluster.description,
    )


def _create_tagged_edge(tx: Any, source_id: str, topic_name: str) -> None:
    """Create a TAGGED edge from a source node to a Topic.

    The WHERE clause restricts the label-less source_id scan to known
    source node labels, preventing a full graph scan.
    """
    tx.run(
        """
        MATCH (s {source_id: $sid})
        WHERE s:File OR s:Email OR s:Commit OR s:Image
        MATCH (t:Topic {name: $name, source: 'cluster'})
        MERGE (s)-[:TAGGED {source: 'cluster'}]->(t)
        """,
        sid=source_id,
        name=topic_name,
    )
