"""CLI entry point: ``fieldnotes search <query>`` and ``fieldnotes topics``.

Loads config, connects to Neo4j + Qdrant via ModelRegistry, runs a
hybrid query (graph + vector), and prints formatted results to stdout.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from worker.config import load_config
from worker.models.resolver import ModelRegistry

# Ensure provider registration side-effects run.
import worker.models.providers.ollama  # noqa: F401

from worker.query.graph import GraphQuerier
from worker.query.hybrid import merge
from worker.query.vector import VectorQuerier


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fieldnotes",
        description="Personal knowledge graph — search your notes.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=None,
        help="Path to config.toml (default: ~/.fieldnotes/config.toml)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    sub = parser.add_subparsers(dest="command")

    search_p = sub.add_parser("search", help="Hybrid search across your notes")
    search_p.add_argument("query", nargs="+", help="Natural-language search query")
    search_p.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=10,
        help="Max vector results (default: 10)",
    )

    # ── topics ──────────────────────────────────────────────────────
    topics_p = sub.add_parser("topics", help="Browse and inspect topics")
    topics_p.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Machine-readable JSON output",
    )
    topics_sub = topics_p.add_subparsers(dest="topics_command")

    topics_sub.add_parser("list", help="List all topics with document counts")

    show_p = topics_sub.add_parser("show", help="Show topic details + linked docs")
    show_p.add_argument("name", help="Topic name")

    topics_sub.add_parser(
        "gaps",
        help="Cluster-discovered topics missing from user taxonomy",
    )

    return parser


def _run_search(
    query: str,
    *,
    config_path: Path | None,
    top_k: int,
) -> int:
    """Execute hybrid search and print results. Returns exit code."""
    cfg = load_config(config_path)
    registry = ModelRegistry(cfg)

    graph_querier = GraphQuerier(registry, cfg.neo4j)
    vector_querier = VectorQuerier(registry, cfg.qdrant)

    try:
        graph_result = graph_querier.query(query)
        vector_result = vector_querier.query(query, top_k=top_k)
        hybrid = merge(query, graph_result, vector_result)

        if hybrid.errors:
            for err in hybrid.errors:
                print(f"warning: {err}", file=sys.stderr)

        if not hybrid.context:
            print("No results found.")
            return 0

        print(hybrid.context)
        return 0
    finally:
        graph_querier.close()
        vector_querier.close()


def _run_topics(
    args: argparse.Namespace,
    *,
    config_path: Path | None,
) -> int:
    """Execute a topics subcommand. Returns exit code."""
    from worker.query.topics import (
        TopicQuerier,
        format_topic_detail,
        format_topic_gaps,
        format_topics_list,
    )

    cfg = load_config(config_path)
    use_json = args.json_output

    with TopicQuerier(cfg.neo4j) as querier:
        if args.topics_command == "list":
            topics = querier.list_topics()
            print(format_topics_list(topics, use_json=use_json))
        elif args.topics_command == "show":
            detail = querier.show_topic(args.name)
            print(format_topic_detail(detail, use_json=use_json))
        elif args.topics_command == "gaps":
            gaps = querier.topic_gaps()
            print(format_topic_gaps(gaps, use_json=use_json))
        else:
            # No subcommand given — show topics help
            print("Usage: fieldnotes topics {list,show,gaps}", file=sys.stderr)
            return 1

    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(name)s %(levelname)s: %(message)s",
    )

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "search":
        query = " ".join(args.query)
        return _run_search(
            query,
            config_path=args.config,
            top_k=args.top_k,
        )

    if args.command == "topics":
        return _run_topics(args, config_path=args.config)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
