"""CLI handler for ``fieldnotes connections`` — surface unlinked similar docs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from worker.config import load_config
from worker.query.connections import ConnectionQuerier


def run_connections(
    *,
    config_path: Path | None = None,
    source_id: str | None = None,
    source_type: str | None = None,
    threshold: float = 0.82,
    limit: int = 20,
    cross_source: bool = False,
    json_output: bool = False,
) -> int:
    """Run connection suggestion query and print results.

    Returns an exit code (0 = success, 1 = error).
    """
    cfg = load_config(config_path)

    with ConnectionQuerier(cfg.neo4j, cfg.qdrant) as querier:
        result = querier.suggest(
            source_id=source_id,
            source_type=source_type,
            threshold=threshold,
            limit=limit,
            cross_source=cross_source,
        )

    if result.error:
        print(f"error: {result.error}", file=sys.stderr)
        return 1

    if json_output:
        data = {
            "checked": result.checked,
            "suggestions": [
                {
                    "similarity": s.similarity,
                    "source_a": s.source_a,
                    "source_b": s.source_b,
                    "label_a": s.label_a,
                    "label_b": s.label_b,
                    "title_a": s.title_a,
                    "title_b": s.title_b,
                    "source_type_a": s.source_type_a,
                    "source_type_b": s.source_type_b,
                    "reason": s.reason,
                }
                for s in result.suggestions
            ],
        }
        print(json.dumps(data, indent=2))
        return 0

    count = len(result.suggestions)
    print(
        f"Suggested connections ({count} found, {result.checked} pairs checked):\n"
    )

    if not result.suggestions:
        print("  No unlinked similar documents found.")
        return 0

    for s in result.suggestions:
        score_str = f"\033[33m{s.similarity:.2f}\033[0m"
        label_a = f"\033[36m[{s.label_a}]\033[0m"
        label_b = f"\033[36m[{s.label_b}]\033[0m"
        print(f"  {score_str}  {label_a} {s.title_a}")
        print(f"    \033[90m↔\033[0m   {label_b} {s.title_b}")
        print(f"        \033[90m{s.reason}\033[0m")
        print()

    return 0
