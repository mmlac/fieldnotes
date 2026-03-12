"""Centralized Prometheus metrics for the fieldnotes worker.

Defines all metric objects, a background push client for Pushgateway,
and a timing context manager for histograms.

Usage::

    from worker.metrics import (
        DOCUMENTS_PROCESSED, PIPELINE_DURATION, observe_duration, init_metrics,
    )

    DOCUMENTS_PROCESSED.labels(source_type="markdown", operation="ingest").inc()

    with observe_duration(PIPELINE_DURATION, stage="chunking"):
        do_chunking()
"""

from __future__ import annotations

import atexit
import logging
import threading
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    push_to_gateway,
)

if TYPE_CHECKING:
    from worker.config import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry — private registry to avoid default platform/process metrics
# ---------------------------------------------------------------------------

REGISTRY = CollectorRegistry()

JOB_NAME = "fieldnotes_worker"

# ---------------------------------------------------------------------------
# Histogram bucket definitions
# ---------------------------------------------------------------------------

DURATION_BUCKETS = (
    0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120,
)

# ---------------------------------------------------------------------------
# Index status gauges
# ---------------------------------------------------------------------------

SOURCES_TOTAL = Gauge(
    "sources_total",
    "Total number of configured sources",
    ["source_type"],
    registry=REGISTRY,
)

ENTITIES_TOTAL = Gauge(
    "entities_total",
    "Total entities in the knowledge graph",
    registry=REGISTRY,
)

CHUNKS_TOTAL = Gauge(
    "chunks_total",
    "Total chunks in the vector store",
    registry=REGISTRY,
)

QDRANT_POINTS_TOTAL = Gauge(
    "qdrant_points_total",
    "Total points stored in Qdrant",
    registry=REGISTRY,
)

TOPICS_TOTAL = Gauge(
    "topics_total",
    "Total topics discovered by clustering",
    registry=REGISTRY,
)

EDGES_TOTAL = Gauge(
    "edges_total",
    "Total edges in the knowledge graph",
    ["type"],
    registry=REGISTRY,
)

NEO4J_STORE_BYTES = Gauge(
    "neo4j_store_bytes",
    "Neo4j store size in bytes",
    registry=REGISTRY,
)

QDRANT_COLLECTION_BYTES = Gauge(
    "qdrant_collection_bytes",
    "Qdrant collection size in bytes",
    registry=REGISTRY,
)

# ---------------------------------------------------------------------------
# Pipeline counters
# ---------------------------------------------------------------------------

DOCUMENTS_PROCESSED = Counter(
    "documents_processed_total",
    "Total documents processed",
    ["source_type", "operation"],
    registry=REGISTRY,
)

DOCUMENTS_FAILED = Counter(
    "documents_failed_total",
    "Total documents that failed processing",
    ["source_type", "stage"],
    registry=REGISTRY,
)

CHUNKS_EMBEDDED = Counter(
    "chunks_embedded_total",
    "Total chunks embedded",
    registry=REGISTRY,
)

ENTITIES_EXTRACTED = Counter(
    "entities_extracted_total",
    "Total entities extracted from documents",
    registry=REGISTRY,
)

ENTITIES_RESOLVED = Counter(
    "entities_resolved_total",
    "Total entities resolved via cross-source resolution",
    registry=REGISTRY,
)

# ---------------------------------------------------------------------------
# Pipeline histograms
# ---------------------------------------------------------------------------

PIPELINE_DURATION = Histogram(
    "pipeline_duration_seconds",
    "Duration of individual pipeline stages",
    ["stage"],
    buckets=DURATION_BUCKETS,
    registry=REGISTRY,
)

PIPELINE_TOTAL_DURATION = Histogram(
    "pipeline_total_duration_seconds",
    "Total end-to-end pipeline duration",
    buckets=DURATION_BUCKETS,
    registry=REGISTRY,
)

LLM_REQUEST_DURATION = Histogram(
    "llm_request_duration_seconds",
    "Duration of LLM API requests",
    ["model", "task"],
    buckets=DURATION_BUCKETS,
    registry=REGISTRY,
)

LLM_TOKENS = Counter(
    "llm_tokens_total",
    "Total LLM tokens consumed",
    ["model", "task", "direction"],
    registry=REGISTRY,
)

NEO4J_WRITE_DURATION = Histogram(
    "neo4j_write_duration_seconds",
    "Duration of Neo4j write operations",
    buckets=DURATION_BUCKETS,
    registry=REGISTRY,
)

QDRANT_WRITE_DURATION = Histogram(
    "qdrant_write_duration_seconds",
    "Duration of Qdrant write operations",
    buckets=DURATION_BUCKETS,
    registry=REGISTRY,
)

EMBEDDING_DURATION = Histogram(
    "embedding_duration_seconds",
    "Duration of embedding generation",
    ["model"],
    buckets=DURATION_BUCKETS,
    registry=REGISTRY,
)

# ---------------------------------------------------------------------------
# Operational gauges / counters
# ---------------------------------------------------------------------------

QUEUE_DEPTH = Gauge(
    "queue_depth",
    "Current document processing queue depth",
    registry=REGISTRY,
)

WORKER_UPTIME = Gauge(
    "worker_uptime_seconds",
    "Worker uptime in seconds",
    registry=REGISTRY,
)

VISION_QUEUE_DEPTH = Gauge(
    "vision_queue_depth",
    "Current vision processing queue depth",
    registry=REGISTRY,
)

CLUSTERING_LAST_RUN = Gauge(
    "clustering_last_run_timestamp",
    "Unix timestamp of the last clustering run",
    registry=REGISTRY,
)

SOURCE_WATCHER_EVENTS = Counter(
    "source_watcher_events_total",
    "Total source watcher events",
    ["source_type", "event_type"],
    registry=REGISTRY,
)

# ---------------------------------------------------------------------------
# Timing context manager
# ---------------------------------------------------------------------------


@contextmanager
def observe_duration(histogram: Histogram, **labels: str) -> Iterator[None]:
    """Time a block and observe the duration on *histogram*.

    Usage::

        with observe_duration(PIPELINE_DURATION, stage="chunking"):
            run_chunking()
    """
    start = time.monotonic()
    try:
        yield
    finally:
        elapsed = time.monotonic() - start
        if labels:
            histogram.labels(**labels).observe(elapsed)
        else:
            histogram.observe(elapsed)


# ---------------------------------------------------------------------------
# Push client
# ---------------------------------------------------------------------------

_push_thread: threading.Thread | None = None
_push_stop = threading.Event()

DEFAULT_PUSH_INTERVAL = 15.0
DEFAULT_PUSHGATEWAY_URL = "http://localhost:9091"


def _push_loop(gateway: str, interval: float) -> None:
    """Background loop that pushes metrics to Pushgateway."""
    while not _push_stop.wait(timeout=interval):
        try:
            push_to_gateway(gateway, job=JOB_NAME, registry=REGISTRY)
        except Exception:
            logger.debug("Failed to push metrics to %s", gateway, exc_info=True)


def _final_push(gateway: str) -> None:
    """Push metrics one last time on shutdown."""
    try:
        push_to_gateway(gateway, job=JOB_NAME, registry=REGISTRY)
    except Exception:
        logger.debug("Final metrics push failed", exc_info=True)


def init_metrics(config: Config) -> None:
    """Initialize the metrics push client.

    Reads ``metrics.pushgateway_url`` and ``metrics.push_interval`` from the
    config.  If the pushgateway URL is empty or not configured, metrics are
    still collected in-process but never pushed (no-op mode).
    """
    global _push_thread  # noqa: PLW0603

    metrics_settings = {}
    if hasattr(config, "sources"):
        # Metrics config may live under a dedicated section; fall back to
        # sensible defaults when not present.
        pass

    # Try to read from a [metrics] section in the raw config, if wired up.
    gateway = getattr(config, "_metrics_pushgateway_url", "") or ""
    interval = getattr(config, "_metrics_push_interval", DEFAULT_PUSH_INTERVAL)

    # Allow env-var override (convenient for containers)
    import os

    gateway = os.environ.get("FIELDNOTES_PUSHGATEWAY_URL", gateway)
    if not gateway:
        logger.info("Pushgateway URL not configured — metrics push disabled")
        return

    interval_env = os.environ.get("FIELDNOTES_METRICS_PUSH_INTERVAL")
    if interval_env:
        try:
            interval = float(interval_env)
        except ValueError:
            pass

    logger.info("Starting metrics push to %s every %.0fs", gateway, interval)

    _push_stop.clear()
    _push_thread = threading.Thread(
        target=_push_loop,
        args=(gateway, interval),
        daemon=True,
        name="metrics-push",
    )
    _push_thread.start()

    atexit.register(_final_push, gateway)
