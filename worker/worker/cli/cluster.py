"""CLI handler for ``fieldnotes cluster`` — manual clustering run."""

from __future__ import annotations

import sys
from pathlib import Path

from worker.clustering.cluster import (
    CorpusTooSmallError,
    cluster_embeddings,
)
from worker.clustering.labeler import label_clusters
from worker.clustering.writer import write_clusters
from worker.clustering.app_linker import link_apps_to_topics
from worker.config import load_config
from worker.models.resolver import ModelRegistry

# Qdrant timeout for the corpus-size check (seconds).
_QDRANT_TIMEOUT_S = 30


def _corpus_size(qdrant_cfg) -> int:
    """Return the number of points in the Qdrant collection."""
    from qdrant_client import QdrantClient

    client = QdrantClient(
        host=qdrant_cfg.host,
        port=qdrant_cfg.port,
        timeout=_QDRANT_TIMEOUT_S,
    )
    try:
        info = client.get_collection(qdrant_cfg.collection)
        return info.points_count or 0
    finally:
        client.close()


def run_cluster(
    *,
    config_path: Path | None = None,
    min_cluster_size: int | None = None,
    force: bool = False,
) -> int:
    """Run the full clustering pipeline synchronously with progress output.

    Returns an exit code (0 = success, 1 = error).
    """
    cfg = load_config(config_path)
    registry = ModelRegistry(cfg)
    qdrant_cfg = cfg.qdrant
    neo4j_cfg = cfg.neo4j
    clustering_cfg = cfg.clustering

    min_corpus = clustering_cfg.min_corpus_size

    # 1. Check corpus size
    size = _corpus_size(qdrant_cfg)
    if size < min_corpus and not force:
        print(
            f"Corpus has {size} vectors, need at least {min_corpus}. "
            f"Use --force to run anyway.",
        )
        return 0

    # 2. Fetch vectors and cluster
    print(f"Fetching vectors from Qdrant... ({size} vectors)")

    cluster_kwargs: dict = dict(
        max_vectors=clustering_cfg.max_vectors,
    )
    if force:
        cluster_kwargs["min_corpus_size"] = 0
    else:
        cluster_kwargs["min_corpus_size"] = min_corpus
    if min_cluster_size is not None:
        cluster_kwargs["min_cluster_size"] = min_cluster_size

    try:
        print("Running UMAP reduction...")
        print("Clustering with HDBSCAN...")
        clusters = cluster_embeddings(qdrant_cfg, **cluster_kwargs)
    except CorpusTooSmallError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not clusters:
        print("No clusters found (all points classified as noise).")
        return 0

    total_assigned = sum(len(c.chunk_ids) for c in clusters)

    # 3. Label clusters
    print(f"Labeling {len(clusters)} clusters...")
    labeled = label_clusters(clusters, registry, qdrant_cfg)

    # 4. Write to Neo4j
    print("Writing topics to Neo4j...")
    write_clusters(labeled, clusters, neo4j_cfg, qdrant_cfg)

    # 5. App-topic linking
    print("Linking apps to topics...")
    try:
        link_apps_to_topics(labeled, clusters, registry, neo4j_cfg)
    except Exception:
        print(
            "warning: app-topic linking failed (clustering results saved)",
            file=sys.stderr,
        )

    # 6. Summary
    orphaned = size - total_assigned
    print(
        f"Done: {len(labeled)} topics discovered, "
        f"{total_assigned} documents assigned, "
        f"{orphaned} orphaned"
    )
    return 0
