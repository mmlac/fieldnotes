"""CLI handler for ``fieldnotes queue`` — inspect and manage the ingestion queue."""

from __future__ import annotations

import json
import sys
from pathlib import Path

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

    line = (
        f"  {color}{item['status']:<11}{reset} "
        f"\033[36m{item['source_type']:<14}\033[0m "
        f"\033[1m{source_id}\033[0m "
        f"({item['operation']})"
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
