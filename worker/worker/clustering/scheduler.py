"""Cron-based clustering scheduler with corpus size gate.

Runs the full clustering pipeline (UMAP → HDBSCAN → label → write Topics) on a
cron schedule. Before each run, checks the corpus size in Qdrant and skips if
below the configured minimum.

Integrates with the main event loop as an asyncio background task.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from croniter import croniter
from qdrant_client import QdrantClient

from worker.clustering.app_linker import link_apps_to_topics
from worker.clustering.cluster import (
    CorpusTooSmallError,
    cluster_embeddings,
)
from worker.clustering.labeler import label_clusters
from worker.clustering.writer import write_clusters
from worker.config import ClusteringConfig, Neo4jConfig, QdrantConfig
from worker.models.resolver import ModelRegistry

logger = logging.getLogger(__name__)


def _corpus_size(qdrant_cfg: QdrantConfig) -> int:
    """Return the number of points in the Qdrant collection."""
    client = QdrantClient(host=qdrant_cfg.host, port=qdrant_cfg.port)
    try:
        info = client.get_collection(qdrant_cfg.collection)
        return info.points_count or 0
    finally:
        client.close()


def run_clustering_pipeline(
    registry: ModelRegistry,
    clustering_cfg: ClusteringConfig,
    qdrant_cfg: QdrantConfig,
    neo4j_cfg: Neo4jConfig,
) -> bool:
    """Execute the full clustering pipeline once.

    Returns True if clustering ran, False if skipped (corpus too small).
    Raises on unexpected errors.
    """
    # Gate: check corpus size before doing expensive work
    size = _corpus_size(qdrant_cfg)
    if size < clustering_cfg.min_corpus_size:
        logger.info(
            "Clustering skipped: corpus has %d points, need at least %d",
            size,
            clustering_cfg.min_corpus_size,
        )
        return False

    logger.info("Starting clustering pipeline (%d points in corpus)", size)

    try:
        clusters = cluster_embeddings(
            qdrant_cfg,
            min_corpus_size=clustering_cfg.min_corpus_size,
            max_vectors=clustering_cfg.max_vectors,
        )
    except CorpusTooSmallError:
        # Race: points removed between check and clustering
        logger.warning("Corpus shrank below minimum during clustering, skipping")
        return False

    if not clusters:
        logger.info("Clustering produced no clusters (all noise), skipping write")
        return True

    labeled = label_clusters(clusters, registry, qdrant_cfg)
    write_clusters(labeled, clusters, neo4j_cfg, qdrant_cfg)

    # Link Application/Tool nodes to topic clusters
    try:
        edge_count = link_apps_to_topics(
            labeled, clusters, registry, neo4j_cfg,
        )
        logger.info("App-topic linking: %d RELATED_TO_TOPIC edges created", edge_count)
    except Exception:
        logger.exception("App-topic linking failed, clustering results still saved")

    logger.info("Clustering pipeline complete: %d topics written", len(labeled))
    return True


def _seconds_until_next(cron_expr: str) -> float:
    """Return seconds from now until the next cron trigger."""
    now = datetime.now(timezone.utc)
    cron = croniter(cron_expr, now)
    next_dt = cron.get_next(datetime)
    delta = (next_dt - now).total_seconds()
    return max(delta, 0.0)


async def clustering_loop(
    registry: ModelRegistry,
    clustering_cfg: ClusteringConfig,
    qdrant_cfg: QdrantConfig,
    neo4j_cfg: Neo4jConfig,
) -> None:
    """Background asyncio task: run clustering on the configured cron schedule.

    Loops forever, sleeping until the next cron trigger, then running the
    pipeline. Safe to cancel — exits cleanly on CancelledError.
    """
    logger.info(
        "Clustering scheduler started (cron=%r, min_corpus=%d)",
        clustering_cfg.cron,
        clustering_cfg.min_corpus_size,
    )

    min_delay = clustering_cfg.min_interval_seconds

    while True:
        delay = _seconds_until_next(clustering_cfg.cron)
        delay = max(delay, min_delay)
        logger.debug("Next clustering run in %.0f seconds", delay)

        await asyncio.sleep(delay)

        try:
            # Run synchronous pipeline in executor to avoid blocking the event loop
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                run_clustering_pipeline,
                registry,
                clustering_cfg,
                qdrant_cfg,
                neo4j_cfg,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Clustering pipeline failed, will retry at next cron tick")
