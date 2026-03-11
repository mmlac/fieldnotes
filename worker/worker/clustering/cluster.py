"""Core clustering: UMAP dimensionality reduction + HDBSCAN.

Pulls all chunk embeddings from Qdrant, reduces from 768→32 dims via UMAP,
then clusters with HDBSCAN. Returns cluster assignments as a list of
(cluster_id, chunk_ids, centroid_vector) tuples.

Edge cases handled:
  - Corpus too small: raises if fewer than ``min_corpus_size`` points
  - All noise: returns empty cluster list
  - Single cluster: returned normally (list of length 1)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import umap
from sklearn.cluster import HDBSCAN

from qdrant_client import QdrantClient

from worker.config import QdrantConfig

logger = logging.getLogger(__name__)

# Defaults
DEFAULT_UMAP_DIMS = 32
DEFAULT_UMAP_NEIGHBORS = 15
DEFAULT_UMAP_METRIC = "cosine"
DEFAULT_MIN_CLUSTER_SIZE = 10
DEFAULT_HDBSCAN_METRIC = "euclidean"
DEFAULT_MIN_CORPUS_SIZE = 20
DEFAULT_MAX_VECTORS = 500_000
_SCROLL_BATCH = 256


class CorpusTooSmallError(Exception):
    """Raised when the corpus has fewer points than ``min_corpus_size``."""


@dataclass
class ClusterResult:
    """A single discovered cluster."""

    cluster_id: int
    chunk_ids: list[str] = field(default_factory=list)
    centroid: list[float] = field(default_factory=list)


def cluster_embeddings(
    qdrant_cfg: QdrantConfig | None = None,
    *,
    umap_dims: int = DEFAULT_UMAP_DIMS,
    umap_neighbors: int = DEFAULT_UMAP_NEIGHBORS,
    umap_metric: str = DEFAULT_UMAP_METRIC,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    hdbscan_metric: str = DEFAULT_HDBSCAN_METRIC,
    min_corpus_size: int = DEFAULT_MIN_CORPUS_SIZE,
    max_vectors: int = DEFAULT_MAX_VECTORS,
) -> list[ClusterResult]:
    """Run the full clustering pipeline.

    1. Scroll chunk embeddings from Qdrant (bounded by ``max_vectors``)
    2. UMAP reduce 768 → ``umap_dims``
    3. HDBSCAN cluster
    4. Compute centroids and return assignments

    Parameters
    ----------
    qdrant_cfg:
        Qdrant connection and collection settings. Uses defaults if None.
    umap_dims:
        Target dimensionality for UMAP reduction.
    umap_neighbors:
        UMAP n_neighbors parameter (local vs global structure trade-off).
    umap_metric:
        Distance metric for UMAP (applied to the original high-dim space).
    min_cluster_size:
        HDBSCAN minimum cluster size.
    hdbscan_metric:
        Distance metric for HDBSCAN (applied in the reduced space).
    min_corpus_size:
        Minimum number of embeddings required. Raises CorpusTooSmallError
        if the corpus is smaller.
    max_vectors:
        Upper bound on the number of vectors to load. When the collection
        exceeds this limit, a random sample of ``max_vectors`` points is
        used instead of the full collection.

    Returns
    -------
    list[ClusterResult]
        One entry per discovered cluster (noise points excluded).
        Empty list if HDBSCAN assigns all points to noise.
    """
    qdrant_cfg = qdrant_cfg or QdrantConfig()
    client = QdrantClient(host=qdrant_cfg.host, port=qdrant_cfg.port)

    try:
        ids, matrix = _scroll_vectors(
            client, qdrant_cfg.collection, max_vectors=max_vectors,
        )
    finally:
        client.close()

    if len(ids) < min_corpus_size:
        raise CorpusTooSmallError(
            f"Corpus has {len(ids)} embeddings, need at least {min_corpus_size}"
        )

    logger.info(
        "Clustering %d embeddings (%d dims → %d dims)",
        matrix.shape[0],
        matrix.shape[1],
        umap_dims,
    )

    reduced = _umap_reduce(
        matrix,
        n_components=umap_dims,
        n_neighbors=umap_neighbors,
        metric=umap_metric,
    )

    labels = _hdbscan_cluster(
        reduced,
        min_cluster_size=min_cluster_size,
        metric=hdbscan_metric,
    )

    results = _build_results(ids, matrix, labels)
    logger.info(
        "Found %d clusters (%d noise points out of %d total)",
        len(results),
        int(np.sum(labels == -1)),
        len(labels),
    )
    return results


def _scroll_vectors(
    client: QdrantClient,
    collection: str,
    *,
    max_vectors: int = DEFAULT_MAX_VECTORS,
) -> tuple[list[str], np.ndarray]:
    """Scroll vectors from a Qdrant collection with a bounded memory footprint.

    Vectors are accumulated directly into a numpy array (one batch at a time)
    to avoid keeping both a Python list and a numpy copy in memory.

    When the collection contains more than *max_vectors* points, a random
    sample of *max_vectors* rows is returned so memory stays bounded.
    """
    ids: list[str] = []
    batches: list[np.ndarray] = []

    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
            limit=_SCROLL_BATCH,
            offset=offset,
            with_vectors=True,
            with_payload=False,
        )

        if points:
            ids.extend(str(p.id) for p in points)
            batches.append(
                np.array([p.vector for p in points], dtype=np.float32)
            )

        if next_offset is None:
            break
        offset = next_offset

    if not batches:
        logger.debug("Scrolled 0 vectors from %r", collection)
        return [], np.empty((0, 0), dtype=np.float32)

    matrix = np.vstack(batches)
    del batches  # free intermediate batch list immediately

    total = matrix.shape[0]
    logger.debug("Scrolled %d vectors from %r", total, collection)

    if total > max_vectors:
        logger.info(
            "Collection %r has %d vectors, sampling %d",
            collection, total, max_vectors,
        )
        rng = np.random.default_rng(42)
        indices = rng.choice(total, size=max_vectors, replace=False)
        indices.sort()  # preserve original ordering
        matrix = matrix[indices]
        ids = [ids[i] for i in indices]

    return ids, matrix


def _umap_reduce(
    matrix: np.ndarray,
    *,
    n_components: int,
    n_neighbors: int,
    metric: str,
) -> np.ndarray:
    """Reduce dimensionality via UMAP."""
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        metric=metric,
        random_state=42,
    )
    return reducer.fit_transform(matrix)


def _hdbscan_cluster(
    reduced: np.ndarray,
    *,
    min_cluster_size: int,
    metric: str,
) -> np.ndarray:
    """Cluster the reduced embeddings with HDBSCAN. Returns label array."""
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric=metric,
    )
    clusterer.fit(reduced)
    return clusterer.labels_


def _build_results(
    ids: list[str],
    original_matrix: np.ndarray,
    labels: np.ndarray,
) -> list[ClusterResult]:
    """Group IDs by cluster label and compute centroids in original space.

    Noise points (label == -1) are excluded. Centroids are computed from
    the original high-dimensional vectors, not the UMAP-reduced space.
    """
    unique_labels = sorted(set(labels))
    results: list[ClusterResult] = []

    for label in unique_labels:
        if label == -1:
            continue

        mask = labels == label
        cluster_ids = [ids[i] for i, m in enumerate(mask) if m]
        centroid = original_matrix[mask].mean(axis=0).tolist()

        results.append(
            ClusterResult(
                cluster_id=int(label),
                chunk_ids=cluster_ids,
                centroid=centroid,
            )
        )

    return results
