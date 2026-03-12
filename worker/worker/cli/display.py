"""Retrieval progress display: tree, spinners, and verbose output.

Renders a progress tree showing search steps and source counts before
the answer streams, with animated spinners during retrieval and LLM
generation. Degrades gracefully to plain text when the terminal doesn't
support rich output or when ``NO_COLOR`` is set.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from worker.query.graph import GraphQueryResult
    from worker.query.hybrid import HybridResult
    from worker.query.vector import VectorQueryResult


def _use_rich() -> bool:
    """Return True if rich output should be used."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    if not hasattr(sys.stderr, "isatty") or not sys.stderr.isatty():
        return False
    try:
        import rich  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Source grouping
# ---------------------------------------------------------------------------

_SOURCE_TYPE_LABELS = {
    "obsidian": "notes",
    "note": "notes",
    "email": "emails",
    "commit": "commits",
    "file": "files",
    "repository": "repositories",
    "calendar": "calendar events",
    "browser_history": "browser history items",
    "application": "applications",
}


@dataclass
class _SourceGroup:
    """A group of results from one source type."""
    source_type: str
    count: int
    identifiers: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return _SOURCE_TYPE_LABELS.get(self.source_type, self.source_type)


def _group_sources(hybrid: HybridResult) -> list[_SourceGroup]:
    """Group hybrid results by source type with representative identifiers."""
    type_counts: Counter[str] = Counter()
    type_ids: dict[str, list[str]] = {}

    # Graph results
    for row in hybrid.graph_results:
        for value in row.values():
            if isinstance(value, dict):
                stype = value.get("source_type") or value.get("type")
                sid = value.get("source_id") or value.get("email") or value.get("name")
                if stype:
                    stype = str(stype)
                    type_counts[stype] += 1
                    if sid:
                        type_ids.setdefault(stype, []).append(str(sid))

    # Vector results
    for vr in hybrid.vector_results:
        stype = vr.source_type
        type_counts[stype] += 1
        if vr.source_id:
            type_ids.setdefault(stype, []).append(vr.source_id)

    groups = []
    for stype, count in type_counts.most_common():
        ids = type_ids.get(stype, [])
        # Deduplicate and keep up to 3 representative identifiers
        seen: set[str] = set()
        unique: list[str] = []
        for sid in ids:
            if sid not in seen:
                seen.add(sid)
                unique.append(sid)
                if len(unique) >= 3:
                    break
        groups.append(_SourceGroup(source_type=stype, count=count, identifiers=unique))

    return groups


# ---------------------------------------------------------------------------
# Plain-text fallback
# ---------------------------------------------------------------------------

def _format_tree_plain(total: int, groups: list[_SourceGroup]) -> str:
    """Format the progress tree using Unicode box-drawing characters."""
    lines = [f"Searching... {total} source{'s' if total != 1 else ''} found"]
    for i, g in enumerate(groups):
        is_last = i == len(groups) - 1
        prefix = "\u2514\u2500" if is_last else "\u251c\u2500"
        desc = f"{g.count} {g.label}"
        if g.identifiers:
            desc += f" ({', '.join(g.identifiers)})"
        lines.append(f"{prefix} {desc}")
    return "\n".join(lines)


def _format_verbose_plain(
    *,
    graph_result: GraphQueryResult | None = None,
    vector_result: VectorQueryResult | None = None,
    hybrid: HybridResult | None = None,
) -> str:
    """Format verbose details as plain text."""
    parts: list[str] = []
    if graph_result and graph_result.cypher:
        parts.append(f"[Cypher] {graph_result.cypher}")
    if vector_result and vector_result.results:
        scores = [f"{r.score:.3f}" for r in vector_result.results[:5]]
        parts.append(f"[Vector scores] {', '.join(scores)}")
    if hybrid and hybrid.context:
        ctx_len = len(hybrid.context)
        parts.append(f"[Context] {ctx_len:,} chars")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Rich output
# ---------------------------------------------------------------------------

def _print_tree_rich(total: int, groups: list[_SourceGroup]) -> None:
    """Print the progress tree using rich."""
    from rich.console import Console
    from rich.tree import Tree

    console = Console(stderr=True)
    label = f"[bold]Searching...[/bold] {total} source{'s' if total != 1 else ''} found"
    tree = Tree(label)
    for g in groups:
        desc = f"[cyan]{g.count}[/cyan] {g.label}"
        if g.identifiers:
            ids_str = ", ".join(f"[dim]{i}[/dim]" for i in g.identifiers)
            desc += f" ({ids_str})"
        tree.add(desc)
    console.print(tree)


def _print_verbose_rich(
    *,
    graph_result: GraphQueryResult | None = None,
    vector_result: VectorQueryResult | None = None,
    hybrid: HybridResult | None = None,
) -> None:
    """Print verbose details using rich."""
    from rich.console import Console
    from rich.panel import Panel

    console = Console(stderr=True)
    parts: list[str] = []
    if graph_result and graph_result.cypher:
        parts.append(f"[bold]Cypher:[/bold] {graph_result.cypher}")
    if vector_result and vector_result.results:
        scores = [f"{r.score:.3f}" for r in vector_result.results[:5]]
        parts.append(f"[bold]Top vector scores:[/bold] {', '.join(scores)}")
    if hybrid and hybrid.context:
        ctx_len = len(hybrid.context)
        parts.append(f"[bold]Context:[/bold] {ctx_len:,} chars")
    if parts:
        console.print(Panel("\n".join(parts), title="[dim]verbose[/dim]", border_style="dim"))


# ---------------------------------------------------------------------------
# Spinner context manager
# ---------------------------------------------------------------------------

@contextmanager
def spinner(message: str = "Searching...") -> Iterator[None]:
    """Show an animated spinner while the block executes.

    Falls back to a static message on non-rich terminals.
    """
    if _use_rich():
        from rich.console import Console
        from rich.spinner import Spinner
        from rich.live import Live

        console = Console(stderr=True)
        sp = Spinner("dots", text=message, style="bold")
        with Live(sp, console=console, transient=True):
            yield
    else:
        print(message, file=sys.stderr, end="", flush=True)
        try:
            yield
        finally:
            print(" done.", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def display_progress(
    hybrid: HybridResult,
    *,
    graph_result: GraphQueryResult | None = None,
    vector_result: VectorQueryResult | None = None,
    verbose: bool = False,
) -> None:
    """Display the retrieval progress tree and optional verbose details.

    Called after hybrid search completes but before LLM generation.
    Outputs to stderr so it doesn't interfere with the answer on stdout.
    """
    groups = _group_sources(hybrid)
    total = sum(g.count for g in groups)

    if not groups:
        return

    if _use_rich():
        _print_tree_rich(total, groups)
        if verbose:
            _print_verbose_rich(
                graph_result=graph_result,
                vector_result=vector_result,
                hybrid=hybrid,
            )
    else:
        print(_format_tree_plain(total, groups), file=sys.stderr, flush=True)
        if verbose:
            verbose_text = _format_verbose_plain(
                graph_result=graph_result,
                vector_result=vector_result,
                hybrid=hybrid,
            )
            if verbose_text:
                print(verbose_text, file=sys.stderr, flush=True)
