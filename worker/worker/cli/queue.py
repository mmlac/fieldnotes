"""CLI handler for ``fieldnotes queue`` — inspect and manage the ingestion queue."""

from __future__ import annotations

import fnmatch
import json
import sys
from pathlib import Path
from typing import Any

from worker.config import load_config
from worker.queue import PersistentQueue


def _queue_path(config_path: Path | None) -> Path:
    """Resolve the queue database path from config."""
    cfg = load_config(config_path)
    data_dir = Path(cfg.core.data_dir).expanduser()
    return data_dir / "queue.db"


def _open_queue(config_path: Path | None) -> PersistentQueue:
    db = _queue_path(config_path)
    if not db.exists():
        print(
            "error: queue database not found — has the daemon run yet?", file=sys.stderr
        )
        raise SystemExit(1)
    return PersistentQueue(db)


def _format_item(item: dict) -> str:
    """Format a single queue item for human display."""
    status_colors = {
        "pending": "\033[33m",  # yellow
        "processing": "\033[36m",  # cyan
        "failed": "\033[31m",  # red
    }
    color = status_colors.get(item["status"], "")
    reset = "\033[0m" if color else ""

    source_id = item["source_id"]
    if len(source_id) > 50:
        source_id = source_id[:47] + "..."

    flags = ""
    if item.get("index_only"):
        flags = " \033[35m[index-only]\033[0m"

    line = (
        f"  {color}{item['status']:<11}{reset} "
        f"\033[36m{item['source_type']:<14}\033[0m "
        f"\033[1m{source_id}\033[0m "
        f"({item['operation']}){flags}"
    )
    if item.get("error"):
        err = item["error"]
        if len(err) > 60:
            err = err[:57] + "..."
        line += f"\n{'':>28}\033[31m{err}\033[0m"
    return line


def run_queue_summary(
    *, config_path: Path | None = None, json_output: bool = False
) -> int:
    """Show summary counts of queue items by status."""
    q = _open_queue(config_path)
    try:
        summary = q.summary()
        stats = q.stats()
    finally:
        q.close()

    if json_output:
        print(json.dumps({"summary": summary, "by_source": stats}, indent=2))
        return 0

    if not summary:
        print("Queue is empty.")
        return 0

    total = sum(summary.values())
    print(f"\033[1mIngestion Queue\033[0m  ({total} total)")
    for status, count in sorted(summary.items()):
        print(f"  {status:<12} {count}")

    if stats:
        print("\n\033[1mBy Source\033[0m")
        for source_type, status_counts in sorted(stats.items()):
            parts = ", ".join(f"{s}: {c}" for s, c in sorted(status_counts.items()))
            print(f"  {source_type:<14} {parts}")

    return 0


def run_queue_list(
    *,
    n: int = 20,
    order: str = "asc",
    status: str | None = None,
    source_type: str | None = None,
    config_path: Path | None = None,
    json_output: bool = False,
) -> int:
    """List queue items (top = oldest first, tail = newest first)."""
    q = _open_queue(config_path)
    try:
        items = q.list_items(
            status=status,
            source_type=source_type,
            limit=n,
            order=order,
        )
    finally:
        q.close()

    if json_output:
        print(json.dumps(items, indent=2))
        return 0

    if not items:
        print("No items found.")
        return 0

    label = "oldest" if order == "asc" else "newest"
    print(f"\033[1mQueue — {label} {len(items)} items\033[0m\n")
    for item in items:
        print(_format_item(item))

    return 0


def run_queue_retry(*, config_path: Path | None = None) -> int:
    """Reset all failed items to pending for retry."""
    q = _open_queue(config_path)
    try:
        count = q.retry_failed()
    finally:
        q.close()

    if count:
        print(f"Reset {count} failed item(s) to pending.")
    else:
        print("No failed items to retry.")
    return 0


def run_queue_purge(
    *,
    status: str = "failed",
    config_path: Path | None = None,
) -> int:
    """Delete all items with the given status."""
    q = _open_queue(config_path)
    try:
        count = q.purge(status=status)
    finally:
        q.close()

    if count:
        print(f"Purged {count} {status} item(s).")
    else:
        print(f"No {status} items to purge.")
    return 0


def run_queue_migrate(*, config_path: Path | None = None) -> int:
    """Run one-time migration of old cursor JSON files into the queue database."""
    cfg = load_config(config_path)
    data_dir = Path(cfg.core.data_dir).expanduser()
    db_path = data_dir / "queue.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    q = PersistentQueue(db_path)
    try:
        count = q.migrate_cursor_files(data_dir)
    finally:
        q.close()

    if count:
        print(f"Migrated {count} cursor file(s) into queue database.")
    else:
        print("No cursor files to migrate (already migrated or none found).")
    return 0


# ------------------------------------------------------------------
# retag
# ------------------------------------------------------------------


def _matches_any_pattern(path: str, patterns: list[str]) -> bool:
    """Check if *path* matches any glob pattern (full path, basename, or component)."""
    p = Path(path)
    for pattern in patterns:
        if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(p.name, pattern):
            return True
        if any(fnmatch.fnmatch(part, pattern) for part in p.parts):
            return True
    return False


# Source types whose queued events carry a file path that can be
# re-evaluated against exclude / index_only patterns.
_RETAGGABLE_SOURCES = frozenset({"files", "obsidian", "repositories"})


def _resolve_file_path(event: dict[str, Any]) -> str | None:
    """Extract the matchable file path from an event, per source_type."""
    source_type = event.get("source_type", "")
    if source_type not in _RETAGGABLE_SOURCES:
        return None
    if source_type == "repositories":
        # Repositories use relative_path for pattern matching.
        # Skip commit events — only file events are retaggable.
        source_id = event.get("source_id", "")
        if source_id.startswith("commit:"):
            return None
        return event.get("meta", {}).get("relative_path")
    # files and obsidian use source_id (absolute path)
    return event.get("source_id")


def _load_source_patterns(
    cfg: Any,
) -> dict[str, dict[str, list[str]]]:
    """Load exclude_patterns and index_only_patterns from source configs.

    Returns ``{source_type: {"exclude": [...], "index_only": [...], "include": [...]}}``.
    """
    result: dict[str, dict[str, list[str]]] = {}
    for name, source_cfg in cfg.sources.items():
        s = source_cfg.settings
        result[name] = {
            "exclude": s.get("exclude_patterns", []),
            "index_only": s.get("index_only_patterns", []),
            "include": s.get("include_patterns", []),
        }
    return result


def _evaluate_event(
    event: dict[str, Any],
    patterns: dict[str, dict[str, list[str]]],
) -> str:
    """Re-evaluate a queued event against current config patterns.

    Returns one of:
      - ``"exclude"``    — file now matches exclude_patterns, should be removed
      - ``"index_only"`` — file now matches index_only_patterns
      - ``"normal"``     — file should be fully processed
    """
    source_type = event.get("source_type", "")
    file_path = _resolve_file_path(event)
    if not file_path:
        return "normal"

    source_patterns = patterns.get(source_type, {})
    exclude = source_patterns.get("exclude", [])
    index_only = source_patterns.get("index_only", [])

    if exclude and _matches_any_pattern(file_path, exclude):
        return "exclude"
    if index_only and _matches_any_pattern(file_path, index_only):
        return "index_only"
    return "normal"


def run_queue_retag(
    *,
    config_path: Path | None = None,
    dry_run: bool = False,
) -> int:
    """Re-evaluate queued items against current config patterns.

    For each pending/failed item:
      - Now excluded → removed from queue
      - Now matches index_only_patterns → flag set, content stripped
      - Was index_only but no longer matches → flag cleared
    """
    cfg = load_config(config_path)
    patterns = _load_source_patterns(cfg)
    q = _open_queue(config_path)

    try:
        items = q.iter_actionable()
    except Exception as exc:
        q.close()
        print(f"error: {exc}", file=sys.stderr)
        return 1

    removed = 0
    tagged_index_only = 0
    cleared_index_only = 0
    skipped = 0

    try:
        for queue_id, blob_path, event in items:
            source_type = event.get("source_type", "")
            file_path = _resolve_file_path(event)

            # Skip non-file sources (commits, apps, etc.)
            if source_type not in patterns or not file_path:
                skipped += 1
                continue

            verdict = _evaluate_event(event, patterns)
            meta = event.get("meta", {})
            was_index_only = meta.get("index_only", False)

            if verdict == "exclude":
                if dry_run:
                    print(f"  would remove  {source_type}  {file_path}")
                else:
                    q.remove(queue_id)
                removed += 1

            elif verdict == "index_only" and not was_index_only:
                # Tag as index_only and strip content from payload
                meta["index_only"] = True
                event["meta"] = meta
                event.pop("text", None)
                event.pop("raw_bytes", None)
                if dry_run:
                    print(f"  would tag     {source_type}  {file_path}")
                else:
                    q.update_payload(queue_id, event)
                    # Also remove the blob if content was stored on disk
                    if blob_path:
                        import contextlib
                        import os

                        with contextlib.suppress(OSError):
                            os.unlink(blob_path)
                        q._conn.execute(
                            "UPDATE queue SET blob_path = NULL WHERE id = ?",
                            (queue_id,),
                        )
                tagged_index_only += 1

            elif verdict == "normal" and was_index_only:
                # Was index_only but no longer matches — clear the flag.
                # Content is gone; file will need a re-scan to get full
                # indexing (source will re-emit on next cycle).
                meta.pop("index_only", None)
                event["meta"] = meta
                if dry_run:
                    print(f"  would untag   {source_type}  {file_path}")
                else:
                    q.update_payload(queue_id, event)
                cleared_index_only += 1

            else:
                skipped += 1
    finally:
        q.close()

    # Summary
    prefix = "[dry run] " if dry_run else ""
    changes = removed + tagged_index_only + cleared_index_only
    if changes == 0:
        print(f"{prefix}No changes — all queued items match current config.")
    else:
        parts = []
        if removed:
            parts.append(f"{removed} removed (now excluded)")
        if tagged_index_only:
            parts.append(f"{tagged_index_only} tagged index-only")
        if cleared_index_only:
            parts.append(f"{cleared_index_only} cleared index-only (content needs re-scan)")
        print(f"{prefix}Retagged: {', '.join(parts)}.")

    return 0
