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
from typing import TYPE_CHECKING, Any, Iterator

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    push_to_gateway,
)

if TYPE_CHECKING:
    from neo4j import Driver
    from qdrant_client import QdrantClient
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
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1,
    2.5,
    5,
    10,
    30,
    60,
    120,
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

EMBEDDING_BATCH_SIZE = Histogram(
    "embedding_batch_size",
    "Number of texts per embedding call",
    ["model"],
    buckets=(1, 2, 5, 10, 25, 50, 100, 250, 500),
    registry=REGISTRY,
)

LLM_ERRORS = Counter(
    "llm_errors_total",
    "Total LLM API errors",
    ["model", "task", "error_type"],
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

WATCHER_ACTIVE = Gauge(
    "fieldnotes_watcher_active",
    "Whether a source watcher is currently running (1) or stopped (0)",
    ["source_type"],
    registry=REGISTRY,
)

WATCHER_LAST_EVENT_TIMESTAMP = Gauge(
    "fieldnotes_watcher_last_event_timestamp",
    "Unix timestamp of the last event emitted by a source watcher",
    ["source_type"],
    registry=REGISTRY,
)

INITIAL_SCAN_FILES_TOTAL = Counter(
    "fieldnotes_initial_scan_files_total",
    "Total files processed during initial directory scan",
    ["source_type", "result"],
    registry=REGISTRY,
)

INITIAL_SCAN_DURATION_SECONDS = Gauge(
    "fieldnotes_initial_scan_duration_seconds",
    "Duration of initial directory scan in seconds",
    ["source_type"],
    registry=REGISTRY,
)

INITIAL_SCAN_DEDUP_DROPPED = Counter(
    "fieldnotes_initial_scan_dedup_dropped_total",
    "Watchdog events dropped by the post-scan dedup window",
    ["source_type"],
    registry=REGISTRY,
)

# -- Initial sync progress (ETA tracking) --

INITIAL_SYNC_ITEMS_TOTAL = Gauge(
    "fieldnotes_initial_sync_items_total",
    "Total items to process during initial sync",
    registry=REGISTRY,
)

INITIAL_SYNC_ITEMS_PROCESSED = Gauge(
    "fieldnotes_initial_sync_items_processed",
    "Items processed so far during initial sync",
    registry=REGISTRY,
)

INITIAL_SYNC_ETA_SECONDS = Gauge(
    "fieldnotes_initial_sync_eta_seconds",
    "Estimated seconds remaining for initial sync to complete",
    registry=REGISTRY,
)

# Module-level accumulator so sources can register totals and main loops
# can read them without reaching into Gauge internals.
_initial_sync_total: int = 0


def initial_sync_add_items(count: int) -> None:
    """Register *count* items discovered during an initial scan."""
    global _initial_sync_total
    _initial_sync_total += count
    INITIAL_SYNC_ITEMS_TOTAL.set(_initial_sync_total)


def initial_sync_get_total() -> int:
    """Return the current total of registered initial-sync items."""
    return _initial_sync_total


OBSIDIAN_VAULTS_DISCOVERED = Gauge(
    "fieldnotes_obsidian_vaults_discovered",
    "Number of Obsidian vaults discovered and being watched",
    registry=REGISTRY,
)

CURSOR_CHECKPOINT_WRITES = Counter(
    "fieldnotes_cursor_checkpoint_writes_total",
    "Total cursor checkpoint writes to disk",
    ["source_type"],
    registry=REGISTRY,
)

METADATA_ONLY_FILES = Counter(
    "fieldnotes_metadata_only_files_total",
    "Files indexed with metadata only (no content parsing)",
    ["source_type"],
    registry=REGISTRY,
)

IWORK_EXTRACTION_DURATION_SECONDS = Histogram(
    "fieldnotes_iwork_extraction_duration_seconds",
    "Duration of iWork file text extraction via osascript",
    ["app"],
    buckets=DURATION_BUCKETS,
    registry=REGISTRY,
)

GMAIL_POLL_DURATION = Histogram(
    "fieldnotes_gmail_poll_duration_seconds",
    "Duration of Gmail poll cycles",
    buckets=DURATION_BUCKETS,
    registry=REGISTRY,
)

RATE_LIMIT_WAITS = Counter(
    "rate_limit_waits_total",
    "Total times a request had to wait for a rate limit slot",
    ["provider"],
    registry=REGISTRY,
)

RATE_LIMIT_REJECTIONS = Counter(
    "rate_limit_rejections_total",
    "Total requests rejected by rate limiter (timeout)",
    ["provider"],
    registry=REGISTRY,
)

TOKEN_BUDGET_USED = Gauge(
    "token_budget_used_total",
    "Total tokens consumed against the daily budget",
    registry=REGISTRY,
)

TOKEN_BUDGET_REJECTIONS = Counter(
    "token_budget_rejections_total",
    "Total requests rejected due to exhausted token budget",
    registry=REGISTRY,
)

CONCURRENCY_LIMIT_WAITS = Counter(
    "concurrency_limit_waits_total",
    "Total times a request waited for a concurrency slot",
    registry=REGISTRY,
)

CIRCUIT_BREAKER_STATE = Gauge(
    "circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=open, 2=half_open)",
    ["service"],
    registry=REGISTRY,
)

CIRCUIT_BREAKER_REJECTIONS = Counter(
    "circuit_breaker_rejections_total",
    "Total requests rejected by circuit breaker",
    ["service"],
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
# Index status collector
# ---------------------------------------------------------------------------

DEFAULT_COLLECT_INTERVAL = 60.0


def collect_index_status(
    neo4j_driver: Driver,
    qdrant_client: QdrantClient,
    collection_name: str,
) -> None:
    """Query Neo4j and Qdrant for current index statistics and update gauges.

    All Neo4j queries use read-only (auto-commit) sessions.  Failures are
    logged and swallowed so the collector never crashes the worker.
    """
    _collect_neo4j(neo4j_driver)
    _collect_qdrant(qdrant_client, collection_name)


def _collect_neo4j(driver: Driver) -> None:
    """Read index stats from Neo4j."""
    try:
        with driver.session() as session:
            # Source counts by type
            result = session.run(
                "MATCH (s) "
                "WHERE s:File OR s:Email OR s:Repository OR s:ObsidianNote "
                "RETURN labels(s)[0] AS type, count(s) AS count"
            )
            for record in result:
                SOURCES_TOTAL.labels(source_type=record["type"]).set(record["count"])

            # Entity, Chunk, Topic counts
            for label, gauge in [
                ("Entity", ENTITIES_TOTAL),
                ("Chunk", CHUNKS_TOTAL),
                ("Topic", TOPICS_TOTAL),
            ]:
                result = session.run(f"MATCH (n:{label}) RETURN count(n) AS count")
                gauge.set(result.single()["count"])

            # Edge counts by type
            result = session.run(
                "MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS count"
            )
            for record in result:
                EDGES_TOTAL.labels(type=record["type"]).set(record["count"])

            # Store size (not available in all Neo4j editions)
            try:
                result = session.run(
                    "CALL dbms.queryJmx("
                    "'org.neo4j:instance=kernel#0,name=Store sizes'"
                    ") YIELD attributes "
                    "RETURN attributes"
                )
                rec = result.single()
                if rec:
                    attrs: dict[str, Any] = rec["attributes"]
                    total = attrs.get("TotalStoreSize", {})
                    val = total.get("value")
                    if val is not None:
                        NEO4J_STORE_BYTES.set(int(val))
            except Exception:
                logger.debug("Neo4j store size metric unavailable", exc_info=True)

    except Exception:
        logger.warning("Failed to collect Neo4j index status", exc_info=True)


def _collect_qdrant(client: QdrantClient, collection_name: str) -> None:
    """Read index stats from Qdrant."""
    try:
        info = client.get_collection(collection_name)
        QDRANT_POINTS_TOTAL.set(info.points_count or 0)
        if info.disk_data_size:
            QDRANT_COLLECTION_BYTES.set(info.disk_data_size)
    except Exception:
        logger.warning("Failed to collect Qdrant index status", exc_info=True)


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

    # Read from the [metrics] config section.
    metrics_cfg = getattr(config, "metrics", None)
    gateway = getattr(metrics_cfg, "pushgateway_url", "") if metrics_cfg else ""
    interval = getattr(metrics_cfg, "push_interval", DEFAULT_PUSH_INTERVAL) if metrics_cfg else DEFAULT_PUSH_INTERVAL

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
