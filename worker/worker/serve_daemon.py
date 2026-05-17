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
    _build_sighup_handler,
    _check_neo4j,
    _check_qdrant,
    _index_status_loop,
    _setup_logging,
)

logger = logging.getLogger(__name__)


async def _run_daemon(cfg: Config, *, config_path: Path | None = None, progress_enabled: bool = False) -> None:
    """Run the ingest pipeline as a background service."""
    from worker.clustering.scheduler import clustering_loop
    from worker.metrics import (
        INITIAL_SYNC_ITEMS_PROCESSED,
        QUEUE_DEPTH,
        QUEUE_FINISH_ETA_SECONDS,
        SOURCE_TASK_FAILED,
        initial_sync_all_sources_done,
        initial_sync_register_source,
    )
    from worker.circuit_breaker import CircuitOpenError, State, get_breaker
    from worker.models.resolver import ModelRegistry
    from worker.pipeline import Pipeline
    from worker.pipeline.progress import (
        NullProgressReporter,
        ProgressReporter,
        RichProgressReporter,
    )
    from worker.pipeline.writer import Writer
    from worker.parsers.registry import get as get_parser

    # Ensure provider registration side-effects run.
    import worker.models.providers  # noqa: F401
    import worker.parsers  # noqa: F401

    registry = ModelRegistry(cfg)
    writer = Writer(neo4j_cfg=cfg.neo4j, qdrant_cfg=cfg.qdrant)
    progress: ProgressReporter = (
        RichProgressReporter() if progress_enabled else NullProgressReporter()
    )
    pipeline = Pipeline(
        registry=registry, writer=writer, progress=progress, me_config=cfg.me
    )

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

    # Persistent event queue (SQLite-backed).  The writer-backed
    # ``indexed_check`` is wired into the queue itself so any "created"
    # event for a source_id already chunked in Neo4j is dropped before
    # it consumes a queue slot — protecting against re-emission after
    # cursor loss, daemon restart, or fresh source initialisation.
    from worker.queue import PersistentQueue

    indexed_check = writer.indexed_source_ids

    data_dir = Path.home() / ".fieldnotes" / "data"
    queue = PersistentQueue(
        db_path=data_dir / "queue.db",
        indexed_check=indexed_check,
    )
    recovered = queue.recover()
    if recovered:
        logger.info(
            "Recovered %d interrupted queue item(s) from previous run", recovered
        )

    def _on_source_task_done(source_name: str, task: asyncio.Task[None]) -> None:
        """Surface silent source-task crashes.

        ``asyncio.create_task`` parks unretrieved exceptions inside the
        Task object — they only appear at shutdown when the task is
        finalised, which is far too late to debug a startup failure.
        This callback fires the moment a source task finishes (cleanly,
        cancelled, or with an exception) and logs any non-cancellation
        exception with a full traceback.  It also bumps a Prometheus
        counter so the dashboard can show "source X is dead" instead of
        a silently-stopped gauge.
        """
        if task.cancelled():
            return
        exc = task.exception()
        if exc is None:
            # Source returned normally — unusual for a long-running
            # watcher loop, worth a warning.
            logger.warning(
                "Source task %s exited cleanly without being cancelled — "
                "this source will no longer emit events",
                source_name,
            )
            SOURCE_TASK_FAILED.labels(source_type=source_name).inc()
            return
        logger.error(
            "Source task %s crashed and will no longer emit events: %s",
            source_name,
            exc,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        SOURCE_TASK_FAILED.labels(source_type=source_name).inc()

    source_tasks: list[asyncio.Task[None]] = []
    for source in sources:
        initial_sync_register_source()
        source_name = source.name()
        task = asyncio.create_task(
            source.start(queue, indexed_check=indexed_check),
            name=f"source:{source_name}",
        )
        task.add_done_callback(
            lambda t, name=source_name: _on_source_task_done(name, t)
        )
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

    if hasattr(signal, "SIGHUP"):
        loop.add_signal_handler(
            signal.SIGHUP, _build_sighup_handler(pipeline, config_path)
        )

    # Main ingest loop — claim/complete/fail against PersistentQueue
    try:
        _sync_processed = 0
        # Rolling window of per-item processing durations.  Drives the
        # queue-finish ETA: ETA ≈ avg(window) × queue.depth().
        _proc_times: collections.deque[float] = collections.deque(maxlen=50)
        _last_depth_log = 0.0

        while not stop_event.is_set():
            depth_now = queue.depth()
            QUEUE_DEPTH.set(depth_now)
            progress.queue_depth(depth_now)
            event = await loop.run_in_executor(None, queue.claim)
            if event is None:
                # Queue empty — ETA is 0 by definition.
                QUEUE_FINISH_ETA_SECONDS.set(0)
                await asyncio.sleep(0.5)
                continue

            QUEUE_DEPTH.set(queue.depth())
            queue_id = event.pop("_queue_id")
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
                    process_task = loop.run_in_executor(None, pipeline.process, doc)
                    stop_task = asyncio.ensure_future(stop_event.wait())
                    done, _pending = await asyncio.wait(
                        [process_task, stop_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if stop_task in done:
                        process_task.cancel()
                        break
                    stop_task.cancel()
                    process_task.result()  # re-raise if failed
                    QUEUE_DEPTH.set(queue.depth())
                else:
                    await loop.run_in_executor(None, pipeline.reconcile_self_if_configured)
                    queue.complete(queue_id)
                if stop_event.is_set():
                    # Interrupted mid-document — item stays 'processing',
                    # recover() will reclaim it on next startup.
                    break
            except CircuitOpenError as exc:
                # Circuit breaker is open — put the item back as pending
                # (without incrementing attempts) and sleep until recovery.
                # This prevents burning through the queue and permanently
                # failing items that would succeed once the service recovers.
                logger.warning(
                    "Circuit breaker open (%s) — returning %s %s to queue, "
                    "pausing processing",
                    exc.breaker_name,
                    source_type,
                    source_id,
                )
                with queue._lock:
                    queue._conn.execute(
                        "UPDATE queue SET status = 'pending' WHERE id = ?",
                        (queue_id,),
                    )
                breaker = get_breaker(exc.breaker_name)
                if breaker and breaker.state == State.OPEN:
                    remaining = breaker.recovery_timeout - (
                        time.monotonic() - breaker._opened_at
                    )
                    wait = max(1.0, min(remaining, 30.0))
                else:
                    wait = 10.0
                logger.info("Sleeping %.0fs for circuit breaker recovery", wait)
                await asyncio.sleep(wait)
            except Exception:
                logger.exception(
                    "Failed to process event %s %s (%s)",
                    source_type,
                    source_id,
                    operation,
                )
                queue.fail(queue_id, str(sys.exc_info()[1]))

            # Track processing time for ALL items (not just initial sync)
            # so the queue-finish ETA is meaningful at steady state too.
            elapsed = time.monotonic() - t0
            _proc_times.append(elapsed)

            if is_initial:
                _sync_processed += 1
                INITIAL_SYNC_ITEMS_PROCESSED.set(_sync_processed)

            # Queue-finish ETA: rolling avg processing time × current depth.
            # Sources may still be scanning when the queue briefly empties,
            # so emit -1 ("indeterminate") rather than 0 in that window.
            depth_now = queue.depth()
            if _proc_times and depth_now > 0:
                avg = sum(_proc_times) / len(_proc_times)
                QUEUE_FINISH_ETA_SECONDS.set(avg * depth_now)
            elif depth_now == 0 and not initial_sync_all_sources_done():
                QUEUE_FINISH_ETA_SECONDS.set(-1)
            else:
                QUEUE_FINISH_ETA_SECONDS.set(0)

            # Periodic queue depth log (every 60s)
            now = time.monotonic()
            if now - _last_depth_log >= 60.0:
                _last_depth_log = now
                logger.info("Queue depth: %d", queue.depth())
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
        queue.close()
        progress.stop()
        logger.info("Daemon shutdown complete")


_LOG_FILE = Path.home() / ".fieldnotes" / "data" / "daemon.log"


def _resolve_progress_enabled(progress: bool | None) -> bool:
    """Decide whether the live Rich progress display should be active.

    ``True``/``False`` from the caller is honoured verbatim so users can
    force-enable progress in piped contexts (e.g. tmux) or suppress it
    in interactive ones.  When unspecified, fall back to TTY detection
    on stderr — matching the channel the progress display writes to.
    """
    if progress is not None:
        return progress
    return sys.stderr.isatty()


def run_daemon(
    config_path: Path | None = None,
    *,
    progress: bool | None = None,
) -> None:
    """Entry point for ``fieldnotes serve --daemon``."""
    cfg = load_config(config_path)
    _setup_logging(cfg.core.log_level, log_file=_LOG_FILE)

    logger.info("fieldnotes daemon starting")

    progress_enabled = _resolve_progress_enabled(progress)

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

    # Wait for Ollama readiness (if configured) — same retry pattern as
    # Neo4j / Qdrant above.  Without this, the first batch of queue items
    # fails and trips the circuit breaker for 120 seconds.
    for name, provider in cfg.providers.items():
        if provider.type != "ollama":
            continue
        base_url = provider.settings.get("base_url", "http://localhost:11434")
        logger.info("Waiting for Ollama (%s) at %s ...", name, base_url)
        for attempt in range(1, max_retries + 1):
            try:
                import urllib.request

                req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status == 200:
                        logger.info("Ollama (%s) ready", name)
                        break
                    raise ConnectionError(f"HTTP {resp.status}")
            except Exception as exc:
                if attempt == max_retries:
                    logger.error(
                        "Ollama (%s) not reachable after %d attempts: %s",
                        name,
                        max_retries,
                        exc,
                    )
                    sys.exit(1)
                logger.debug(
                    "Ollama (%s) attempt %d/%d failed: %s",
                    name,
                    attempt,
                    max_retries,
                    exc,
                )
                time.sleep(2**attempt)

    try:
        asyncio.run(_run_daemon(cfg, config_path=config_path, progress_enabled=progress_enabled))
    except KeyboardInterrupt:
        logger.info("Interrupted")
