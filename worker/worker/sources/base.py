"""Base interface for Python source shims.

Mirrors the Go Source interface (daemon/internal/sources/source.go) so that
Python-native sources can emit IngestEvent dicts directly into the worker's
internal queue without going through the Go daemon.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from worker.queue import PersistentQueue


IndexedCheck = Callable[[list[str]], set[str]]
"""Callable returning the subset of *source_ids* already indexed in Neo4j.

Implemented by :meth:`worker.pipeline.writer.Writer.indexed_source_ids`.
Sources with content-immutable items (gmail messages, git commits, etc.)
use this to skip the expensive fetch/parse path for items the graph
already has chunks for.
"""


class PythonSource(ABC):
    """Base class for all Python-side source adapters.

    Subclasses implement name(), configure(), and start() to mirror the
    Go Source interface. Events are enqueued via PersistentQueue as plain
    dicts matching the IngestEvent JSON schema.
    """

    @abstractmethod
    def name(self) -> str:
        """Stable identifier for this source type.

        Must match the ``source_type`` field in emitted events and the
        ``[sources.<name>]`` config section.
        """
        ...

    @abstractmethod
    def configure(self, cfg: dict[str, Any]) -> None:
        """Initialise the source from its ``[sources.<name>]`` settings.

        Called once at startup before ``start()``.
        """
        ...

    @abstractmethod
    async def start(
        self,
        queue: PersistentQueue,
        *,
        indexed_check: IndexedCheck | None = None,
    ) -> None:
        """Begin watching and enqueue IngestEvent dicts via *queue*.

        Must run until cancelled (via ``asyncio.CancelledError``).

        Parameters
        ----------
        indexed_check:
            Optional pre-filter callable.  Sources with immutable item IDs
            (gmail messages, git commits, etc.) should call this with a
            batch of candidate source_ids and skip any that come back as
            already indexed — bypassing the per-item fetch entirely.
            Sources that don't benefit (or that handle dedup locally)
            may ignore this parameter.
        """
        ...
