"""CLI handler for ``fieldnotes timeline`` — temporal search across all sources."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from worker.config import load_config
from worker.query.timeline import TimelineEntry, TimelineQuerier, TimelineResult


def _fmt_timestamp(ts: str) -> str:
    """Format an ISO timestamp to 'YYYY-MM-DD HH:MM' for display."""
    if not ts:
        return "           "
    # Normalise to strip trailing 'Z' and fractional seconds.
    ts = ts.rstrip("Z").split(".")[0]
    if "T" in ts:
        date_part, time_part = ts.split("T", 1)
        return f"{date_part} {time_part[:5]}"
    return ts[:16]


def _group_by_day(
    entries: list[TimelineEntry],
) -> list[tuple[str | None, list[TimelineEntry]]]:
    """Return entries grouped by calendar day, in order.

    Each group is (day_str | None, [entries]).  day_str is 'YYYY-MM-DD'.
    """
    groups: list[tuple[str | None, list[TimelineEntry]]] = []
    current_day: str | None = None
    current_group: list[TimelineEntry] = []

    for entry in entries:
        ts = entry.timestamp.rstrip("Z").split(".")[0]
        day = ts.split("T")[0] if "T" in ts else ts[:10]
        if day != current_day:
            if current_group:
                groups.append((current_day, current_group))
            current_day = day
            current_group = [entry]
        else:
            current_group.append(entry)

    if current_group:
        groups.append((current_day, current_group))

    return groups


def _format_human(result: TimelineResult) -> str:
    """Format timeline entries for human-readable terminal output."""
    lines: list[str] = []

    since_short = result.since[:10] if result.since else "?"
    until_short = result.until[:10] if result.until else "now"
    lines.append(f"\033[1mTimeline: {since_short} \u2192 {until_short}\033[0m")

    if not result.entries:
        lines.append("\n  No activity found in this time range.")
        return "\n".join(lines)

    # Determine if entries span multiple days.
    days = _group_by_day(result.entries)
    multi_day = len(days) > 1

    for day, group in days:
        if multi_day and day:
            lines.append(f"\n\033[36m{day}\033[0m")

        for entry in group:
            ts_display = _fmt_timestamp(entry.timestamp)
            label = f"[{entry.label}]"
            event = f"({entry.event_type})"
            # Truncate long titles.
            title = entry.title
            if len(title) > 70:
                title = title[:67] + "..."

            lines.append(
                f"  \033[33m{ts_display}\033[0m  "
                f"\033[36m{label:<14}\033[0m "
                f"\033[1m{title}\033[0m {event}"
            )
            if entry.snippet:
                snippet = entry.snippet.replace("\n", " ").strip()
                if len(snippet) > 80:
                    snippet = snippet[:77] + "..."
                lines.append(f"                             {snippet}")

    lines.append(f"\n  {len(result.entries)} entries")
    return "\n".join(lines)


def _format_json(result: TimelineResult) -> str:
    """Format timeline result as structured JSON."""
    return json.dumps(
        {
            "since": result.since,
            "until": result.until,
            "entries": [
                {
                    "source_type": e.source_type,
                    "source_id": e.source_id,
                    "label": e.label,
                    "title": e.title,
                    "timestamp": e.timestamp,
                    "event_type": e.event_type,
                    "snippet": e.snippet,
                }
                for e in result.entries
            ],
            "count": len(result.entries),
            "error": result.error,
        },
        indent=2,
        default=str,
    )


def run_timeline(
    *,
    since: str = "24h",
    until: str = "now",
    source_type: str | None = None,
    limit: int = 50,
    json_output: bool = False,
    config_path: Path | None = None,
) -> int:
    """Execute a timeline query and print results. Returns exit code."""
    cfg = load_config(config_path)

    with TimelineQuerier(cfg.neo4j, cfg.qdrant) as querier:
        result = querier.query(
            since=since,
            until=until,
            source_type=source_type,
            limit=limit,
        )

    if result.error:
        print(f"error: {result.error}", file=sys.stderr)
        return 1

    if json_output:
        print(_format_json(result))
    else:
        print(_format_human(result))

    return 0
