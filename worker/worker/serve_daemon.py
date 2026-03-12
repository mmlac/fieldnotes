"""Combined daemon mode: ingest pipeline + MCP server.

``fieldnotes serve --daemon`` starts both the source-watching ingest pipeline
(from ``worker.main``) and the MCP server (from ``worker.mcp_server``) in a
single process.  The MCP server listens on a Unix domain socket so it can be
reached programmatically while stdio remains free for logging.

The ingest pipeline consumes source events and writes to Neo4j/Qdrant.
The MCP server answers tool calls from Claude Desktop or other MCP clients.
"""

from __future__ import annotations

import asyncio
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
    _setup_logging,
)

logger = logging.getLogger(__name__)


async def _run_daemon(cfg: Config) -> None:
    """Run ingest pipeline and MCP server concurrently."""
    from typing import Any

    from worker.clustering.scheduler import clustering_loop
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
        logger.warning("No sources configured — running MCP server only")

    # Background tasks
    background_tasks: list[asyncio.Task[None]] = []

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
        task = asyncio.create_task(source.start(queue), name=f"source:{source.name()}")
        source_tasks.append(task)
    logger.info("Started %d source(s)", len(source_tasks))

    # Health-check endpoint (off by default)
    if cfg.health.enabled:
        from worker.health import HealthServer

        health_server = HealthServer(cfg, queue=queue, start_time=time.monotonic())
        health_task = asyncio.create_task(health_server.run(), name="health-server")
        background_tasks.append(health_task)
        logger.info("Health endpoint enabled on %s:%d", cfg.health.bind, cfg.health.port)

    # MCP server on Unix socket
    async def _run_mcp() -> None:
        from worker.mcp_server import FieldnotesServer

        server = FieldnotesServer(cfg)
        await server.run()

    mcp_task = asyncio.create_task(_run_mcp(), name="mcp-server")
    logger.info("MCP server started (stdio transport)")

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
        while not stop_event.is_set():
            try:
                event = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            source_type = event.get("source_type", "")
            source_id = event.get("source_id", "")
            operation = event.get("operation", "")

            try:
                parser = get_parser(source_type)
                parsed_docs = parser.parse(event)
                for doc in parsed_docs:
                    pipeline.process(doc)
            except Exception:
                logger.exception(
                    "Failed to process event %s %s (%s)",
                    source_type, source_id, operation,
                )
    finally:
        _DRAIN_TIMEOUT = 10  # seconds to wait for in-progress work

        mcp_task.cancel()
        for task in background_tasks:
            task.cancel()
        for task in source_tasks:
            task.cancel()
        try:
            await asyncio.wait_for(
                asyncio.gather(
                    mcp_task, *background_tasks, *source_tasks,
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


def run_daemon(config_path: Path | None = None) -> None:
    """Entry point for ``fieldnotes serve --daemon``."""
    cfg = load_config(config_path)
    _setup_logging(cfg.core.log_level)

    logger.info("fieldnotes daemon starting")

    # Health checks
    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            _check_neo4j(cfg)
            break
        except Exception as exc:
            if attempt == max_retries:
                logger.error("Neo4j check failed after %d attempts: %s", max_retries, exc)
                sys.exit(1)
            time.sleep(2 ** attempt)

    for attempt in range(1, max_retries + 1):
        try:
            _check_qdrant(cfg)
            break
        except Exception as exc:
            if attempt == max_retries:
                logger.error("Qdrant check failed after %d attempts: %s", max_retries, exc)
                sys.exit(1)
            time.sleep(2 ** attempt)

    asyncio.run(_run_daemon(cfg))
