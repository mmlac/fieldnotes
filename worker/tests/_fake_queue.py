"""Shared pytest fixtures and helpers for fieldnotes tests.

The :class:`FakeQueue` here is a drop-in replacement for
:class:`worker.queue.PersistentQueue` that's used by source-adapter tests.
It mimics the production queue's enqueue/cursor API while exposing an
asyncio-style ``get()``/``qsize()`` interface so tests can drain emitted
events without needing a real SQLite file.
"""

from __future__ import annotations

import asyncio
from typing import Any


class FakeQueue:
    """In-memory test double for :class:`worker.queue.PersistentQueue`.

    Provides the subset of the PersistentQueue interface that source
    adapters use (``enqueue``, ``is_enqueued``, ``load_cursor``,
    ``save_cursor``) and the asyncio-Queue interface that tests use to
    drain events (``get``, ``get_nowait``, ``put``, ``qsize``, ``empty``).

    Test-only conveniences:

    * ``enqueued`` — list of every event ever enqueued, in order.
    * ``cursors`` — dict of cursor key → value last persisted.
    * ``indexed_check`` — optionally pre-set to a callable that returns
      a set of "already indexed" source_ids; mirrors the production
      pre-filter so tests can exercise the dedup path.
    """

    def __init__(
        self,
        *,
        indexed_check: Any = None,
    ) -> None:
        self._q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._cursors: dict[str, str] = {}
        self._enqueued_ids: set[str] = set()
        self._enqueued: list[dict[str, Any]] = []
        self.indexed_check = indexed_check

    # ------------------------------------------------------------------
    # PersistentQueue-shaped API
    # ------------------------------------------------------------------

    def enqueue(
        self,
        event: dict[str, Any],
        cursor_key: str | None = None,
        cursor_value: str | None = None,
    ) -> str:
        sid = event.get("source_id", "")
        operation = event.get("operation", "")

        # Pre-filter mirroring PersistentQueue: drop already-indexed
        # "created" events; modified/deleted always pass through.
        if (
            operation == "created"
            and sid
            and self.indexed_check is not None
        ):
            try:
                if sid in self.indexed_check([sid]):
                    if cursor_key is not None and cursor_value is not None:
                        self._cursors[cursor_key] = cursor_value
                    return f"skipped:{sid}"
            except Exception:
                pass  # fall through and enqueue normally

        if sid in self._enqueued_ids:
            # Same source_id is already pending — silently dedup, like
            # PersistentQueue's atomic check inside its lock.
            if cursor_key is not None and cursor_value is not None:
                self._cursors[cursor_key] = cursor_value
            return event.get("id", "")

        self._q.put_nowait(event)
        self._enqueued.append(event)
        if sid:
            self._enqueued_ids.add(sid)
        if cursor_key is not None and cursor_value is not None:
            self._cursors[cursor_key] = cursor_value
        return event.get("id", "")

    def is_enqueued(self, source_id: str) -> bool:
        return source_id in self._enqueued_ids

    def load_cursor(self, key: str) -> str | None:
        return self._cursors.get(key)

    def save_cursor(self, key: str, value: str) -> None:
        self._cursors[key] = value

    # ------------------------------------------------------------------
    # asyncio.Queue-shaped API used by drain helpers in tests
    # ------------------------------------------------------------------

    async def get(self) -> dict[str, Any]:
        item = await self._q.get()
        # Mirror PersistentQueue.complete() — once an item is consumed,
        # its source_id is no longer "in flight" so a future enqueue of
        # the same source_id should succeed (e.g. a re-scan that wants
        # to emit a "modified" event for the same path).
        sid = item.get("source_id", "")
        if sid:
            self._enqueued_ids.discard(sid)
        return item

    def get_nowait(self) -> dict[str, Any]:
        item = self._q.get_nowait()
        sid = item.get("source_id", "")
        if sid:
            self._enqueued_ids.discard(sid)
        return item

    async def put(self, item: dict[str, Any]) -> None:
        # Mirror an asyncio.Queue.put: don't dedup, just push.
        await self._q.put(item)
        self._enqueued.append(item)
        sid = item.get("source_id", "")
        if sid:
            self._enqueued_ids.add(sid)

    def qsize(self) -> int:
        return self._q.qsize()

    def empty(self) -> bool:
        return self._q.empty()

    def close(self) -> None:
        # No-op stub mirroring PersistentQueue.close so production code
        # paths that close the queue at shutdown can run unchanged.
        pass

    @property
    def cursors(self) -> dict[str, str]:
        return self._cursors

    @property
    def enqueued(self) -> list[dict[str, Any]]:
        return self._enqueued
