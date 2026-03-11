"""Async worker queue for vision processing with SHA256 deduplication.

Vision inference is 2-5s per image, so it must not block the text pipeline.
This module provides an asyncio-based worker queue that:

  1. Accepts ParsedDocuments with image_bytes
  2. Filters by file size and skip patterns (icons, avatars, etc.)
  3. Deduplicates via SHA256 — skips images already processed
  4. Processes images through the vision extraction pipeline
  5. Feeds results back to the pipeline for embedding + writing

Dedup strategy:
  - Compute SHA256 of image_bytes
  - Check Neo4j for an existing node with matching sha256 and vision_processed=true
  - Skip if already processed; otherwise process and set the flag after write
"""

from __future__ import annotations

import asyncio
import collections
import hashlib
import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from worker.config import VisionConfig
from worker.parsers.base import ParsedDocument

logger = logging.getLogger(__name__)


@dataclass
class VisionResult:
    """Output from vision processing of a single image."""

    source_id: str
    sha256: str
    text: str
    entities: list[dict[str, Any]] = field(default_factory=list)
    triples: list[dict[str, str]] = field(default_factory=list)


class VisionQueueStats:
    """Thread-safe running statistics for the vision queue.

    Counters are guarded by a lock so they can be safely incremented
    from both the asyncio event-loop and ``run_in_executor`` threads.
    """

    __slots__ = (
        "_lock",
        "submitted",
        "processed",
        "skipped_dedup",
        "skipped_size",
        "skipped_pattern",
        "failed",
    )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.submitted = 0
        self.processed = 0
        self.skipped_dedup = 0
        self.skipped_size = 0
        self.skipped_pattern = 0
        self.failed = 0

    def increment(self, counter: str, n: int = 1) -> None:
        """Atomically increment *counter* by *n*."""
        with self._lock:
            setattr(self, counter, getattr(self, counter) + n)


_SEEN_HASHES_MAX = 100_000


class VisionQueue:
    """Async worker queue for vision processing with SHA256 dedup.

    Parameters
    ----------
    config:
        Vision configuration (concurrency, size limits, skip patterns).
    dedup_checker:
        Callable that takes a SHA256 hex string and returns True if
        the image has already been processed. Typically checks Neo4j.
    process_fn:
        Callable that takes a ParsedDocument (with image_bytes) and
        returns a VisionResult. This is the actual vision inference.
    result_callback:
        Callable that receives a VisionResult after successful processing.
        Typically feeds results back to the pipeline for embedding + writing.
    """

    def __init__(
        self,
        config: VisionConfig,
        dedup_checker: Callable[[str], bool],
        process_fn: Callable[[ParsedDocument], VisionResult],
        result_callback: Callable[[VisionResult], None],
    ) -> None:
        self._config = config
        self._dedup_checker = dedup_checker
        self._process_fn = process_fn
        self._result_callback = result_callback

        self._queue: asyncio.Queue[ParsedDocument] = asyncio.Queue(
            maxsize=config.queue_size,
        )
        self._workers: list[asyncio.Task[None]] = []
        self._stats = VisionQueueStats()
        # LRU-bounded set: OrderedDict used as an ordered set (values ignored).
        # Evicts oldest entries when the cap is reached.
        self._seen_hashes: collections.OrderedDict[str, None] = (
            collections.OrderedDict()
        )
        self._skip_patterns = [
            re.compile(pat, re.IGNORECASE) for pat in config.skip_patterns
        ]
        self._running = False

    @property
    def stats(self) -> VisionQueueStats:
        return self._stats

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    async def start(self) -> None:
        """Start the worker pool."""
        if self._running:
            return
        self._running = True
        for i in range(self._config.concurrency):
            task = asyncio.create_task(
                self._worker(i), name=f"vision-worker-{i}"
            )
            self._workers.append(task)
        logger.info(
            "Vision queue started: %d workers, queue_size=%d",
            self._config.concurrency,
            self._config.queue_size,
        )

    async def stop(self) -> None:
        """Stop all workers gracefully, processing remaining items."""
        if not self._running:
            return
        self._running = False

        # Wait for remaining items to drain
        if not self._queue.empty():
            logger.info(
                "Vision queue draining %d remaining items...", self._queue.qsize()
            )
            await self._queue.join()

        for task in self._workers:
            task.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

        logger.info(
            "Vision queue stopped: %d processed, %d dedup-skipped, %d failed",
            self._stats.processed,
            self._stats.skipped_dedup,
            self._stats.failed,
        )

    async def submit(self, doc: ParsedDocument) -> bool:
        """Submit a document with image_bytes for vision processing.

        Returns True if the document was accepted (queued or will be processed),
        False if it was filtered out (size, pattern, dedup).
        """
        if not doc.image_bytes:
            return False

        self._stats.increment("submitted")

        # Size filtering
        size_bytes = len(doc.image_bytes)
        min_bytes = self._config.min_file_size_kb * 1024
        max_bytes = self._config.max_file_size_mb * 1024 * 1024

        if size_bytes < min_bytes:
            logger.debug(
                "Vision skip %s: too small (%d bytes < %d KB)",
                doc.source_id,
                size_bytes,
                self._config.min_file_size_kb,
            )
            self._stats.increment("skipped_size")
            return False

        if size_bytes > max_bytes:
            logger.debug(
                "Vision skip %s: too large (%d bytes > %d MB)",
                doc.source_id,
                size_bytes,
                self._config.max_file_size_mb,
            )
            self._stats.increment("skipped_size")
            return False

        # Skip pattern filtering (check source_id for icon/avatar patterns)
        if self._matches_skip_pattern(doc.source_id):
            logger.debug(
                "Vision skip %s: matches skip pattern", doc.source_id
            )
            self._stats.increment("skipped_pattern")
            return False

        # SHA256 dedup — in-memory check first, then external
        sha256 = _compute_sha256(doc.image_bytes)

        if sha256 in self._seen_hashes:
            self._seen_hashes.move_to_end(sha256)
            logger.debug(
                "Vision skip %s: SHA256 already seen in session", doc.source_id
            )
            self._stats.increment("skipped_dedup")
            return False

        # Blocking dedup checker (e.g. Neo4j query) — run off the event loop.
        loop = asyncio.get_running_loop()
        already_processed = await loop.run_in_executor(
            None, self._dedup_checker, sha256
        )
        if already_processed:
            logger.debug(
                "Vision skip %s: SHA256 %s already processed", doc.source_id, sha256[:12]
            )
            self._record_seen(sha256)
            self._stats.increment("skipped_dedup")
            return False

        self._record_seen(sha256)

        await self._queue.put(doc)
        logger.debug(
            "Vision queued %s (sha256=%s, %d bytes)",
            doc.source_id,
            sha256[:12],
            size_bytes,
        )
        return True

    def _record_seen(self, sha256: str) -> None:
        """Add *sha256* to the LRU seen-set, evicting the oldest if full."""
        self._seen_hashes[sha256] = None
        if len(self._seen_hashes) > _SEEN_HASHES_MAX:
            self._seen_hashes.popitem(last=False)

    def _matches_skip_pattern(self, source_id: str) -> bool:
        """Check if the source_id matches any skip pattern."""
        for pattern in self._skip_patterns:
            if pattern.search(source_id):
                return True
        return False

    async def _worker(self, worker_id: int) -> None:
        """Worker loop: dequeue and process images."""
        logger.debug("Vision worker %d started", worker_id)
        while self._running:
            try:
                doc = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self._process_fn, doc
                )
                self._result_callback(result)
                self._stats.increment("processed")
                logger.debug(
                    "Vision worker %d processed %s", worker_id, doc.source_id
                )
            except Exception:
                self._stats.increment("failed")
                logger.exception(
                    "Vision worker %d failed processing %s",
                    worker_id,
                    doc.source_id,
                )
            finally:
                self._queue.task_done()

    async def __aenter__(self) -> VisionQueue:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        await self.stop()


def _compute_sha256(data: bytes) -> str:
    """Compute SHA256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def check_vision_processed_neo4j(
    neo4j_session_factory: Callable[[], Any],
) -> Callable[[str], bool]:
    """Create a dedup checker that queries Neo4j for vision_processed flag.

    Parameters
    ----------
    neo4j_session_factory:
        Callable that returns a Neo4j session (e.g. driver.session).

    Returns
    -------
    Callable[[str], bool]
        A function that takes a SHA256 hex string and returns True if
        a node with that sha256 has vision_processed=true.
    """

    def checker(sha256: str) -> bool:
        with neo4j_session_factory() as session:
            result = session.run(
                "MATCH (n {sha256: $sha256, vision_processed: true}) "
                "RETURN count(n) > 0 AS exists",
                sha256=sha256,
            )
            record = result.single()
            return bool(record and record["exists"])

    return checker
