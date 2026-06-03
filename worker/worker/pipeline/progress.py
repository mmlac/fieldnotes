"""Pipeline progress reporting.

Two implementations:
- :class:`NullProgressReporter` — no-op, used when output is not a TTY or
  the user passed ``--no-progress``.
- :class:`RichProgressReporter` — live multi-bar display backed by
  ``rich.progress.Progress``.  Renders one persistent bar for queue
  depth plus a transient bar per file currently being processed.

The reporter is wired through :class:`worker.pipeline.Pipeline` so that
all stages (chunk, embed, extract, resolve, write) can update the
per-file bar without owning a console themselves.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from threading import Lock
from typing import Protocol


class ProgressReporter(Protocol):
    """Pipeline progress sink.

    All methods must be safe to call from worker threads and must be
    cheap when no display is active so callers can invoke them
    unconditionally.
    """

    def start_file(self, source_id: str, label: str, total_chunks: int) -> None: ...

    def set_stage(self, source_id: str, stage: str) -> None: ...

    def advance(self, source_id: str, n: int = 1) -> None: ...

    def finish_file(self, source_id: str) -> None: ...

    def queue_depth(self, depth: int) -> None: ...

    def stop(self) -> None: ...


class NullProgressReporter:
    """No-op reporter used when the live progress display is disabled."""

    def start_file(self, source_id: str, label: str, total_chunks: int) -> None:
        return

    def set_stage(self, source_id: str, stage: str) -> None:
        return

    def advance(self, source_id: str, n: int = 1) -> None:
        return

    def finish_file(self, source_id: str) -> None:
        return

    def queue_depth(self, depth: int) -> None:
        return

    def stop(self) -> None:
        return


class RichProgressReporter:
    """Live multi-bar reporter backed by ``rich.progress.Progress``.

    Bars are added/removed under a lock so concurrent ingest threads do
    not corrupt the live display.  On construction the reporter swaps
    any ``StreamHandler`` writing to stderr/stdout for a
    ``rich.logging.RichHandler`` bound to the same console; this keeps
    log lines from tearing the live region.  File handlers are left in
    place so persisted logs are unaffected.  ``stop()`` restores the
    original handlers.
    """

    _MAX_LABEL_LEN = 60

    def __init__(self) -> None:
        from rich.console import Console
        from rich.logging import RichHandler
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        self._console = Console(file=sys.stderr)
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.fields[label]}[/]"),
            TextColumn("[dim]{task.fields[stage]}[/]"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("ETA"),
            TimeRemainingColumn(),
            console=self._console,
            transient=False,
            expand=True,
            refresh_per_second=8,
        )
        self._lock = Lock()
        self._tasks: dict[str, int] = {}
        self._queue_task = self._progress.add_task(
            "queue",
            total=None,
            label="queue",
            stage="depth: 0",
            visible=False,
        )

        self._installed_handler: logging.Handler | None = None
        self._removed_handlers: list[logging.Handler] = []
        root = logging.getLogger()
        for handler in list(root.handlers):
            stream = getattr(handler, "stream", None)
            if isinstance(handler, logging.StreamHandler) and stream in (
                sys.stderr,
                sys.stdout,
            ):
                root.removeHandler(handler)
                self._removed_handlers.append(handler)
        rich_handler = RichHandler(
            console=self._console,
            show_path=False,
            show_time=True,
            rich_tracebacks=True,
        )
        if self._removed_handlers:
            rich_handler.setLevel(self._removed_handlers[0].level)
        root.addHandler(rich_handler)
        self._installed_handler = rich_handler

        self._progress.start()

    def _truncate(self, label: str) -> str:
        if len(label) <= self._MAX_LABEL_LEN:
            return label
        # Keep the tail (filename) — it's usually the most informative part.
        return "…" + label[-(self._MAX_LABEL_LEN - 1) :]

    def start_file(self, source_id: str, label: str, total_chunks: int) -> None:
        with self._lock:
            if source_id in self._tasks:
                return
            tid = self._progress.add_task(
                source_id,
                total=total_chunks,
                label=self._truncate(label),
                stage="queued",
            )
            self._tasks[source_id] = tid

    def set_stage(self, source_id: str, stage: str) -> None:
        with self._lock:
            tid = self._tasks.get(source_id)
            if tid is None:
                return
            self._progress.update(tid, stage=stage)

    def advance(self, source_id: str, n: int = 1) -> None:
        with self._lock:
            tid = self._tasks.get(source_id)
            if tid is None:
                return
            self._progress.advance(tid, n)

    def finish_file(self, source_id: str) -> None:
        with self._lock:
            tid = self._tasks.pop(source_id, None)
            if tid is not None:
                self._progress.remove_task(tid)

    def queue_depth(self, depth: int) -> None:
        with self._lock:
            self._progress.update(
                self._queue_task,
                stage=f"depth: {depth}",
                visible=depth > 0,
            )

    def stop(self) -> None:
        try:
            self._progress.stop()
        finally:
            root = logging.getLogger()
            if self._installed_handler is not None:
                try:
                    root.removeHandler(self._installed_handler)
                except ValueError:
                    pass
                self._installed_handler = None
            for handler in self._removed_handlers:
                root.addHandler(handler)
            self._removed_handlers.clear()


def resolve_progress_enabled(progress: bool | None) -> bool:
    """Decide whether a live Rich progress display should be active.

    ``True``/``False`` from the caller is honoured verbatim so users can
    force-enable progress in piped contexts (e.g. tmux) or suppress it in
    interactive ones.  When unspecified, fall back to TTY detection on
    stderr — matching the channel the progress display writes to.
    """
    if progress is not None:
        return progress
    return sys.stderr.isatty()


@contextmanager
def phase_progress(
    description: str, total: int
) -> Iterator[Callable[[int, int], None]]:
    """Show a single determinate progress bar for one synchronous phase.

    Styled to match :class:`RichProgressReporter` (spinner, label, bar,
    M-of-N, elapsed, ETA) and rendered on stderr.  Yields an ``advance``
    callback taking ``(completed, total)``; callers invoke it after each
    unit of work to move the bar.  Used by long synchronous CLI phases
    such as ``fieldnotes cluster`` labeling, which would otherwise grind
    silently with no feedback.
    """
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}[/]"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("ETA"),
        TimeRemainingColumn(),
        console=Console(file=sys.stderr),
        transient=False,
        # Rich estimates ETA from speed samples within this trailing window
        # (default 30s) and evicts older ones. When a single unit of work
        # takes longer than the window — e.g. a per-cluster LLM call — every
        # update leaves just one sample, so speed (and ETA) never computes.
        # Keep all samples so ETA is a stable whole-run average.
        speed_estimate_period=float("inf"),
    )
    with progress:
        task = progress.add_task(description, total=total)

        def advance(completed: int, _total: int | None = None) -> None:
            progress.update(task, completed=completed)

        yield advance
