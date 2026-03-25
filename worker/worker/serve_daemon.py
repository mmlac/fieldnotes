"""Daemon mode: background ingest pipeline.

``fieldnotes serve --daemon`` runs the source-watching ingest pipeline as a
long-running background service.  It consumes source events and writes to
Neo4j/Qdrant.  All log output goes to stderr so it can be captured by the
platform service manager (launchd, systemd) without conflicting with other
I/O channels.

The MCP server (``fieldnotes serve --mcp``) must be run as a separate process.
Running MCP over stdio inside the daemon is intentionally unsupported: when the
daemon is managed by launchd or systemd both stdout and stderr are redirected to
a log file, so any JSON-RPC output written to stdout by ``stdio_server`` would
corrupt that log file and no MCP client can connect via stdin anyway.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import signal
import sys
import time
from pathlib import Path

from worker.config import Config, load_config
from worker.main import (
    _build_sources,
    _check_neo4j,
    _check_qdrant,
    _index_status_loop,
    _setup_logging,
)

logger = logging.getLogger(__name__)


async def _run_daemon(cfg: Config) -> None:
    """Run the ingest pipeline as a background service."""
    from typing import Any

    from worker.clustering.scheduler import clustering_loop
    from worker.metrics import (
        INITIAL_SYNC_ETA_SECONDS,
        INITIAL_SYNC_ITEMS_PROCESSED,
        QUEUE_DEPTH,
        initial_sync_all_sources_done,
        initial_sync_get_total,
        initial_sync_register_source,
    )
    from worker.models.resolver import ModelRegistry
    from worker.pipeline import Pipeline
    from worker.pipeline.writer import Writer
    from worker.parsers.registry import get as get_parser

    # Ensure provider registration side-effects run.
    import worker.models.providers  # noqa: F401
    import worker.parsers  # noqa: F401

    registry = ModelRegistry(cfg)
    writer = Writer(neo4j_cfg=cfg.neo4j, qdrant_cfg=cfg.qdrant)
    pipeline = Pipeline(registry=registry, writer=writer)

    sources = _build_sources(cfg)
    if not sources:
        logger.warning("No sources configured — ingest pipeline will be idle")

    # Record startup time for uptime gauge
    start_time = time.monotonic()

    # Background tasks
    background_tasks: list[asyncio.Task[None]] = []

    # Index status collector — first collection on startup, then every 60s
    collection_name = cfg.qdrant.collection or "fieldnotes"
    status_task = asyncio.create_task(
        _index_status_loop(writer, collection_name, start_time),
        name="index-status-collector",
    )
    background_tasks.append(status_task)
    logger.info("Index status collector enabled")

    if cfg.clustering.enabled:
        cluster_task = asyncio.create_task(
            clustering_loop(registry, cfg.clustering, cfg.qdrant, cfg.neo4j),
            name="clustering-scheduler",
        )
        background_tasks.append(cluster_task)

    # Source event queue
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=4096)

    source_tasks: list[asyncio.Task[None]] = []
    for source in sources:
        initial_sync_register_source()
        task = asyncio.create_task(source.start(queue), name=f"source:{source.name()}")
        source_tasks.append(task)

    # Startup summary
    source_names = [s.name() for s in sources]
    logger.info("Started %d source(s): %s", len(source_tasks), ", ".join(source_names))

    provider_names = [f"{n} ({p.type})" for n, p in cfg.providers.items()]
    if provider_names:
        logger.info("Providers: %s", ", ".join(provider_names))

    key_roles = ["embed", "extract", "completion", "query"]
    role_info = []
    for role in key_roles:
        alias = cfg.roles.get(role)
        if alias and alias in cfg.models:
            m = cfg.models[alias]
            role_info.append(f"{role}={m.model}")
    if role_info:
        logger.info("Model roles: %s", ", ".join(role_info))

    # Health-check endpoint (off by default)
    if cfg.health.enabled:
        from worker.health import HealthServer

        health_server = HealthServer(
            cfg,
            queue=queue,
            start_time=start_time,
            neo4j_driver=writer._neo4j_driver,
            qdrant_client=writer._qdrant,
        )
        health_task = asyncio.create_task(health_server.run(), name="health-server")
        background_tasks.append(health_task)
        logger.info(
            "Health endpoint enabled on %s:%d", cfg.health.bind, cfg.health.port
        )

    # Graceful shutdown
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _signal_handler() -> None:
        logger.info("Shutdown signal received")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    # Main ingest loop
    try:
        _sync_processed = 0
        _sync_times: collections.deque[float] = collections.deque(maxlen=50)
        _last_depth_log = 0.0

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
            is_initial = event.get("initial_scan", False)

            t0 = time.monotonic()
            try:
                parser = get_parser(source_type)
                parsed_docs = parser.parse(event)
                for doc in parsed_docs:
                    if stop_event.is_set():
                        break
                    await loop.run_in_executor(None, pipeline.process, doc)
                    QUEUE_DEPTH.set(queue.qsize())
                else:
                    # All docs processed successfully — acknowledge to source
                    on_indexed = event.get("_on_indexed")
                    if on_indexed:
                        on_indexed()
            except Exception:
                logger.exception(
                    "Failed to process event %s %s (%s)",
                    source_type,
                    source_id,
                    operation,
                )
                # Still acknowledge the event so one poisoned item does
                # not block cursor progress for the entire batch.  The
                # event has been consumed from the queue and will not be
                # retried, so we must advance the tracker regardless.
                on_indexed = event.get("_on_indexed")
                if on_indexed:
                    on_indexed()

            if is_initial:
                elapsed = time.monotonic() - t0
                _sync_processed += 1
                _sync_times.append(elapsed)
                INITIAL_SYNC_ITEMS_PROCESSED.set(_sync_processed)
                remaining = initial_sync_get_total() - _sync_processed
                if remaining > 0 and _sync_times:
                    avg = sum(_sync_times) / len(_sync_times)
                    INITIAL_SYNC_ETA_SECONDS.set(remaining * avg)
                elif not initial_sync_all_sources_done():
                    # Sources still scanning — don't report complete yet
                    INITIAL_SYNC_ETA_SECONDS.set(-1)
                else:
                    INITIAL_SYNC_ETA_SECONDS.set(0)

            # Periodic queue depth log (every 60s)
            now = time.monotonic()
            if now - _last_depth_log >= 60.0:
                _last_depth_log = now
                logger.info("Queue depth: %d", queue.qsize())
    finally:
        _DRAIN_TIMEOUT = 10  # seconds to wait for in-progress work

        for task in background_tasks:
            task.cancel()
        for task in source_tasks:
            task.cancel()
        try:
            await asyncio.wait_for(
                asyncio.gather(
                    *background_tasks,
                    *source_tasks,
                    return_exceptions=True,
                ),
                timeout=_DRAIN_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Tasks did not finish within %ds — forcing shutdown", _DRAIN_TIMEOUT
            )
        pipeline.close()
        logger.info("Daemon shutdown complete")


_LOG_FILE = Path.home() / ".fieldnotes" / "data" / "daemon.log"


def run_daemon(config_path: Path | None = None) -> None:
    """Entry point for ``fieldnotes serve --daemon``."""
    cfg = load_config(config_path)
    _setup_logging(cfg.core.log_level, log_file=_LOG_FILE)

    logger.info("fieldnotes daemon starting")

    from worker.metrics import init_metrics

    init_metrics(cfg)

    # Health checks
    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            _check_neo4j(cfg)
            break
        except Exception as exc:
            if attempt == max_retries:
                logger.error(
                    "Neo4j check failed after %d attempts: %s", max_retries, exc
                )
                sys.exit(1)
            time.sleep(2**attempt)

    for attempt in range(1, max_retries + 1):
        try:
            _check_qdrant(cfg)
            break
        except Exception as exc:
            if attempt == max_retries:
                logger.error(
                    "Qdrant check failed after %d attempts: %s", max_retries, exc
                )
                sys.exit(1)
            time.sleep(2**attempt)

    try:
        asyncio.run(_run_daemon(cfg))
    except KeyboardInterrupt:
        logger.info("Interrupted")
