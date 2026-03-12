"""LLM-powered cluster labeling — topic names from central chunks.

For each cluster produced by the clustering core:
  1. Retrieve chunk texts from Qdrant for the cluster's chunk IDs
  2. Select the 20 most central chunks (by L2 distance to centroid)
  3. Send their text to the 'cluster_label' role model
  4. Parse structured JSON response (label + description)

Returns a list of LabeledCluster(cluster_id, label, description).
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass

import numpy as np
from qdrant_client import QdrantClient

from worker.clustering.cluster import ClusterResult
from worker.config import QdrantConfig
from worker.models.base import CompletionRequest
from worker.models.resolver import ModelRegistry, ResolvedModel

logger = logging.getLogger(__name__)

CLUSTER_LABEL_ROLE = "cluster_label"
DEFAULT_TOP_K = 20
LLM_TIMEOUT = 120.0  # seconds

SYSTEM_PROMPT = """\
You are a topic labeling system. Given a set of text chunks that belong to the \
same cluster, produce a short topic label and a one-sentence description.

Respond with a JSON object (no markdown fencing) containing exactly two keys:
  "label": a 2-4 word topic label
  "description": a single sentence describing the topic"""


@dataclass
class LabeledCluster:
    """A cluster with an LLM-assigned topic label."""

    cluster_id: int
    label: str
    description: str


def label_clusters(
    clusters: list[ClusterResult],
    registry: ModelRegistry,
    qdrant_cfg: QdrantConfig | None = None,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> list[LabeledCluster]:
    """Label each cluster by sending central chunks to the LLM.

    Parameters
    ----------
    clusters:
        Cluster results from the clustering core.
    registry:
        Model registry for resolving the 'cluster_label' role.
    qdrant_cfg:
        Qdrant connection settings. Uses defaults if None.
    top_k:
        Number of most-central chunks to send to the LLM per cluster.

    Returns
    -------
    list[LabeledCluster]
        One labeled cluster per input cluster.
    """
    if not clusters:
        return []

    model = registry.for_role(CLUSTER_LABEL_ROLE)
    qdrant_cfg = qdrant_cfg or QdrantConfig()
    client = QdrantClient(host=qdrant_cfg.host, port=qdrant_cfg.port)

    try:
        results: list[LabeledCluster] = []
        for cluster in clusters:
            labeled = _label_single_cluster(
                cluster, model, client, qdrant_cfg.collection, top_k
            )
            results.append(labeled)
        _deduplicate_labels(results)
        return results
    finally:
        client.close()


def _label_single_cluster(
    cluster: ClusterResult,
    model: ResolvedModel,
    client: QdrantClient,
    collection: str,
    top_k: int,
) -> LabeledCluster:
    """Label a single cluster."""
    texts = _get_central_chunk_texts(
        cluster, client, collection, top_k
    )

    if not texts:
        logger.warning(
            "Cluster %d: no chunk texts retrieved, using fallback label",
            cluster.cluster_id,
        )
        return LabeledCluster(
            cluster_id=cluster.cluster_id,
            label="Unknown Topic",
            description="No chunk texts available for labeling.",
        )

    label, description = _call_labeling_model(model, texts)

    return LabeledCluster(
        cluster_id=cluster.cluster_id,
        label=label,
        description=description,
    )


def _get_central_chunk_texts(
    cluster: ClusterResult,
    client: QdrantClient,
    collection: str,
    top_k: int,
) -> list[str]:
    """Retrieve the top_k most central chunk texts for a cluster.

    Fetches vectors and payloads for all chunk IDs in the cluster,
    ranks by L2 distance to the cluster centroid, and returns the
    texts of the closest chunks.
    """
    if not cluster.chunk_ids:
        return []

    points = client.retrieve(
        collection_name=collection,
        ids=cluster.chunk_ids,
        with_vectors=True,
        with_payload=True,
    )

    if not points:
        return []

    centroid = np.array(cluster.centroid, dtype=np.float32)

    # Compute distances and sort by proximity to centroid
    scored: list[tuple[float, str]] = []
    for point in points:
        vec = np.array(point.vector, dtype=np.float32)
        dist = float(np.linalg.norm(vec - centroid))
        text = point.payload.get("text", "") if point.payload else ""
        if text:
            scored.append((dist, text))

    scored.sort(key=lambda x: x[0])
    return [text for _, text in scored[:top_k]]


def _call_labeling_model(
    model: ResolvedModel,
    texts: list[str],
) -> tuple[str, str]:
    """Send chunk texts to the LLM and parse the label response.

    Returns (label, description). Falls back to defaults on parse failure.
    """
    combined = "\n\n---\n\n".join(texts)
    req = CompletionRequest(
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": combined}],
        temperature=0.0,
        timeout=LLM_TIMEOUT,
    )

    resp = model.complete(req, task="label_topic")

    try:
        data = json.loads(resp.text)
        label = str(data["label"]).strip()
        description = str(data["description"]).strip()

        # Validate label is 2-4 words
        word_count = len(label.split())
        if word_count < 2 or word_count > 4:
            logger.warning(
                "Label %r has %d words (expected 2-4), using as-is",
                label,
                word_count,
            )

        return label, description
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.error("Failed to parse labeling response: %s", exc)
        return "Unknown Topic", "LLM response could not be parsed."


def _deduplicate_labels(results: list[LabeledCluster]) -> None:
    """Append disambiguator suffix to duplicate labels in-place.

    If the LLM assigns the same label to multiple clusters, the writer
    would MERGE them into a single Topic node.  Detect duplicates and
    append " (#N)" so each cluster gets a unique Topic.
    """
    counts: Counter[str] = Counter(r.label for r in results)
    dupes = {label for label, n in counts.items() if n > 1}
    if not dupes:
        return

    seen: dict[str, int] = {}
    for result in results:
        if result.label not in dupes:
            continue
        idx = seen.get(result.label, 0) + 1
        seen[result.label] = idx
        if idx > 1:
            old = result.label
            result.label = f"{old} (#{idx})"
            logger.warning(
                "Duplicate label %r for cluster %d, renamed to %r",
                old,
                result.cluster_id,
                result.label,
            )
