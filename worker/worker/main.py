"""Standalone entrypoint for the fieldnotes Phase 1 worker.

Loads config.toml, initializes the model registry, connects to Neo4j and
Qdrant (with connection health checks), initializes the Pipeline, starts
Python source shims (file watcher, obsidian), and wires source events
through the parser registry into pipeline.process().

Usage:
    python -m worker.main
    python -m worker.main --config /path/to/config.toml
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase
from qdrant_client import QdrantClient

from worker.clustering.scheduler import clustering_loop
from worker.config import Config, load_config
from worker.metrics import (
    DEFAULT_COLLECT_INTERVAL,
    QUEUE_DEPTH,
    WORKER_UPTIME,
    collect_index_status,
    init_metrics,
)
from worker.models.resolver import ModelRegistry
from worker.pipeline import Pipeline
from worker.pipeline.writer import Writer
from worker.sources.files import FileSource
from worker.sources.gmail import GmailSource
from worker.sources.obsidian import ObsidianSource
from worker.sources.repositories import RepositorySource
from worker.sources.homebrew import HomebrewSource
from worker.sources.macos_apps import MacOSAppsSource
from worker.sources.base import PythonSource

# Importing parsers triggers @register decorators
import worker.parsers  # noqa: F401
# Importing providers triggers @register decorators
import worker.models.providers  # noqa: F401
from worker.parsers.registry import get as get_parser

logger = logging.getLogger("worker")

# All available source classes, keyed by their .name() value
SOURCE_CLASSES: dict[str, type[PythonSource]] = {
    "files": FileSource,
    "gmail": GmailSource,
    "obsidian": ObsidianSource,
    "repositories": RepositorySource,
    "macos_apps": MacOSAppsSource,
    "homebrew": HomebrewSource,
}


def _setup_logging(level: str) -> None:
    """Configure root logger from config.core.log_level."""
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


def _check_neo4j(cfg: Config) -> None:
    """Verify Neo4j is reachable."""
    driver = GraphDatabase.driver(
        cfg.neo4j.uri,
        auth=(cfg.neo4j.user, cfg.neo4j.password),
    )
    try:
        driver.verify_connectivity()
        logger.info("Neo4j connection OK (%s)", cfg.neo4j.uri)
    finally:
        driver.close()


def _check_qdrant(cfg: Config) -> None:
    """Verify Qdrant is reachable."""
    client = QdrantClient(host=cfg.qdrant.host, port=cfg.qdrant.port)
    try:
        client.get_collections()
        logger.info("Qdrant connection OK (%s:%d)", cfg.qdrant.host, cfg.qdrant.port)
    finally:
        client.close()


def _build_sources(cfg: Config) -> list[PythonSource]:
    """Instantiate and configure sources from [sources.*] config sections."""
    sources: list[PythonSource] = []
    for name, source_cfg in cfg.sources.items():
        cls = SOURCE_CLASSES.get(name)
        if cls is None:
            logger.warning("Unknown source type %r in config, skipping", name)
            continue
        source = cls()
        source.configure(source_cfg.settings)
        sources.append(source)
        logger.info("Configured source: %s", name)
    return sources


async def _index_status_loop(
    writer: Writer,
    collection_name: str,
    start_time: float,
    interval: float = DEFAULT_COLLECT_INTERVAL,
) -> None:
    """Background task: collect index status gauges periodically."""
    logger.info("Index status collector started (interval=%.0fs)", interval)
    loop = asyncio.get_running_loop()
    while True:
        try:
            WORKER_UPTIME.set(time.monotonic() - start_time)
            await loop.run_in_executor(
                None,
                collect_index_status,
                writer._neo4j_driver,
                writer._qdrant,
                collection_name,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Index status collection failed", exc_info=True)
        await asyncio.sleep(interval)


async def _run(cfg: Config) -> None:
    """Main async loop: start sources, consume events, run pipeline."""
    # Initialize model registry
    registry = ModelRegistry(cfg)
    logger.info("Model registry initialized")

    # Initialize writer (connects to Neo4j + Qdrant)
    writer = Writer(neo4j_cfg=cfg.neo4j, qdrant_cfg=cfg.qdrant)
    logger.info("Writer initialized")

    # Record startup time for uptime gauge
    start_time = time.monotonic()
    WORKER_UPTIME.set(0)

    # Initialize pipeline
    pipeline = Pipeline(registry=registry, writer=writer)
    logger.info("Pipeline initialized")

    # Build sources
    sources = _build_sources(cfg)
    if not sources:
        logger.error("No sources configured — nothing to watch")
        pipeline.close()
        return

    # Start background tasks
    background_tasks: list[asyncio.Task[None]] = []

    # Index status collector — first collection on startup, then every 60s
    collection_name = cfg.qdrant.collection or "fieldnotes"
    status_task = asyncio.create_task(
        _index_status_loop(writer, collection_name, start_time),
        name="index-status-collector",
    )
    background_tasks.append(status_task)
    logger.info("Index status collector enabled")

    # Start clustering scheduler as a background task
    if cfg.clustering.enabled:
        cluster_task = asyncio.create_task(
            clustering_loop(registry, cfg.clustering, cfg.qdrant, cfg.neo4j),
            name="clustering-scheduler",
        )
        background_tasks.append(cluster_task)
        logger.info("Clustering scheduler enabled (cron=%s)", cfg.clustering.cron)
    else:
        logger.info("Clustering scheduler disabled")

    # Shared event queue for all sources — bounded to apply backpressure
    # when the consumer falls behind (prevents unbounded memory growth on
    # burst file-change events like git checkout).
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=4096)

    # Start source tasks
    source_tasks: list[asyncio.Task[None]] = []
    for source in sources:
        task = asyncio.create_task(source.start(queue), name=f"source:{source.name()}")
        source_tasks.append(task)
    logger.info("Started %d source(s)", len(source_tasks))

    # Set up graceful shutdown
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _signal_handler() -> None:
        logger.info("Shutdown signal received")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    # Main event loop: consume from queue, parse, process
    try:
        while not stop_event.is_set():
            QUEUE_DEPTH.set(queue.qsize())
            try:
                event = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            QUEUE_DEPTH.set(queue.qsize())
            source_type = event.get("source_type", "")
            source_id = event.get("source_id", "")
            operation = event.get("operation", "")

            try:
                parser = get_parser(source_type)
                parsed_docs = parser.parse(event)
                for doc in parsed_docs:
                    await loop.run_in_executor(None, pipeline.process, doc)
            except Exception:
                logger.exception(
                    "Failed to process event %s %s (%s)",
                    source_type, source_id, operation,
                )
    finally:
        _DRAIN_TIMEOUT = 10  # seconds to wait for in-progress work

        # Cancel background tasks (clustering scheduler)
        for task in background_tasks:
            task.cancel()
        try:
            await asyncio.wait_for(
                asyncio.gather(*background_tasks, return_exceptions=True),
                timeout=_DRAIN_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning("Background tasks did not finish within %ds", _DRAIN_TIMEOUT)

        # Cancel all source tasks
        logger.info("Shutting down sources...")
        for task in source_tasks:
            task.cancel()
        try:
            await asyncio.wait_for(
                asyncio.gather(*source_tasks, return_exceptions=True),
                timeout=_DRAIN_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning("Source tasks did not finish within %ds", _DRAIN_TIMEOUT)

        # Close pipeline (releases Neo4j + Qdrant connections)
        pipeline.close()
        logger.info("Shutdown complete")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="fieldnotes Phase 1 worker",
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=None,
        help="Path to config.toml (default: ~/.fieldnotes/config.toml)",
    )
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    _setup_logging(cfg.core.log_level)

    logger.info("fieldnotes worker starting")

    # Initialize metrics push client
    init_metrics(cfg)

    # Health checks with retry (services may still be starting)
    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            _check_neo4j(cfg)
            break
        except Exception as exc:
            if attempt == max_retries:
                logger.error("Neo4j health check failed after %d attempts: %s", max_retries, exc)
                sys.exit(1)
            delay = 2 ** attempt  # 2, 4, 8, 16s — ~30s total
            logger.warning("Neo4j health check failed (attempt %d/%d): %s — retrying in %ds", attempt, max_retries, exc, delay)
            time.sleep(delay)

    for attempt in range(1, max_retries + 1):
        try:
            _check_qdrant(cfg)
            break
        except Exception as exc:
            if attempt == max_retries:
                logger.error("Qdrant health check failed after %d attempts: %s", max_retries, exc)
                sys.exit(1)
            delay = 2 ** attempt
            logger.warning("Qdrant health check failed (attempt %d/%d): %s — retrying in %ds", attempt, max_retries, exc, delay)
            time.sleep(delay)

    # Run the async event loop
    asyncio.run(_run(cfg))


if __name__ == "__main__":
    main()
