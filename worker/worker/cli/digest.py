"""CLI handler for ``fieldnotes digest`` — daily activity summary."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from worker.config import load_config
from worker.query.digest import DigestQuerier, DigestResult, SourceActivity

logger = logging.getLogger(__name__)

# Human-readable labels for source types
_SOURCE_LABELS: dict[str, str] = {
    "obsidian": "Obsidian",
    "omnifocus": "OmniFocus",
    "gmail": "Gmail",
    "repositories": "Repositories",
    "apps": "Apps",
    "file": "Files",
}


def _format_source_line(activity: SourceActivity) -> list[str]:
    """Format a SourceActivity as one or more display lines."""
    label = _SOURCE_LABELS.get(activity.source_type, activity.source_type.title())
    label_col = f"{label:<14}"

    parts: list[str] = []
    # OmniFocus distinguishes completed from modified
    completed = getattr(activity, "_completed", 0)
    created = activity.created
    modified = (
        activity.modified - completed
        if activity.source_type == "omnifocus"
        else activity.modified
    )

    if activity.source_type == "omnifocus":
        if completed:
            parts.append(f"{completed} completed")
        if created:
            parts.append(f"{created} new")
        if modified:
            parts.append(f"{modified} modified")
    else:
        if created:
            parts.append(f"{created} new")
        if modified:
            parts.append(f"{modified} modified")

    count_str = (
        ", ".join(parts) if parts else f"{activity.created + activity.modified} items"
    )

    lines = [f"  \033[36m{label_col}\033[0m {count_str}"]
    if activity.highlights:
        titles = ", ".join(activity.highlights)
        # Wrap if too long
        if len(titles) > 70:
            titles = titles[:67] + "..."
        lines.append(f"                 {titles}")
    return lines


def _format_human(result: DigestResult) -> str:
    """Format digest for human-readable terminal output."""
    lines: list[str] = []

    since_short = result.since[:10] if result.since else "?"
    until_short = result.until[:10] if result.until else "now"
    lines.append(f"\033[1mDigest: {since_short} \u2192 {until_short}\033[0m")
    lines.append("")

    if not result.sources and result.new_connections == 0 and result.new_topics == 0:
        lines.append("  No activity found in this time range.")
        return "\n".join(lines)

    for activity in result.sources:
        lines.extend(_format_source_line(activity))

    if result.new_connections > 0:
        lines.append(
            f"  \033[36m{'Cross-source':<14}\033[0m {result.new_connections} new connections discovered"
        )

    if result.new_topics > 0:
        lines.append(
            f"  \033[36m{'Topics':<14}\033[0m {result.new_topics} new topic(s) discovered"
        )

    if result.summary:
        lines.append("")
        lines.append(f"\033[1mSummary:\033[0m {result.summary}")

    return "\n".join(lines)


def _format_json(result: DigestResult) -> str:
    """Format digest result as structured JSON."""
    sources = []
    for a in result.sources:
        completed = getattr(a, "_completed", 0)
        entry: dict = {
            "source_type": a.source_type,
            "created": a.created,
            "modified": a.modified,
            "highlights": a.highlights,
        }
        if completed:
            entry["completed"] = completed
        sources.append(entry)

    return json.dumps(
        {
            "since": result.since,
            "until": result.until,
            "sources": sources,
            "new_connections": result.new_connections,
            "new_topics": result.new_topics,
            "summary": result.summary,
            "error": result.error,
        },
        indent=2,
        default=str,
    )


def run_digest(
    *,
    since: str = "24h",
    until: str = "now",
    summarize: bool = False,
    json_output: bool = False,
    config_path: Path | None = None,
) -> int:
    """Execute a digest query and print results. Returns exit code."""
    cfg = load_config(config_path)

    with DigestQuerier(cfg.neo4j) as querier:
        result = querier.query(since=since, until=until)

    if result.error:
        print(f"error: {result.error}", file=sys.stderr)
        return 1

    if summarize:
        result.summary = _generate_summary(result, cfg)

    if json_output:
        print(_format_json(result))
    else:
        print(_format_human(result))

    return 0


def _generate_summary(result: DigestResult, cfg: object) -> str | None:
    """Generate a 2-3 sentence LLM summary of the digest."""
    try:
        from worker.models.base import CompletionRequest
        from worker.models.resolver import ModelRegistry
        import worker.models.providers.ollama  # noqa: F401

        registry = ModelRegistry(cfg)  # type: ignore[arg-type]
        model = registry.for_role("completion")

        # Build a plain-text digest for the LLM
        lines = [f"Activity digest for {result.since[:10]} to {result.until[:10]}:"]
        for a in result.sources:
            completed = getattr(a, "_completed", 0)
            label = a.source_type
            lines.append(
                f"  {label}: {a.created} new, {a.modified} modified"
                + (f", {completed} completed" if completed else "")
            )
            if a.highlights:
                lines.append(f"    Examples: {', '.join(a.highlights)}")
        if result.new_connections:
            lines.append(f"  Cross-source connections: {result.new_connections} new")
        if result.new_topics:
            lines.append(f"  New topics: {result.new_topics}")
        digest_text = "\n".join(lines)

        req = CompletionRequest(
            system=(
                "Summarize this activity digest in 2-3 sentences for the user. "
                "Be specific about what changed and any notable patterns. "
                "Do not use markdown formatting."
            ),
            messages=[{"role": "user", "content": digest_text}],
            max_tokens=256,
            temperature=0.3,
        )
        resp = model.provider.complete(model.model_id, req)
        return resp.text.strip() or None
    except Exception:
        logger.debug("Digest: summary generation failed", exc_info=True)
        return None
