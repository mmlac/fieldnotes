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

    # ── serve ───────────────────────────────────────────────────────
    serve_p = sub.add_parser("serve", help="Run fieldnotes as a server")
    serve_p.add_argument(
        "--mcp",
        action="store_true",
        help="Start MCP server over stdio transport",
    )
    serve_p.add_argument(
        "--daemon",
        action="store_true",
        help="Run ingest pipeline and MCP server as a background daemon",
    )

    # ── daemon ─────────────────────────────────────────────────────
    daemon_p = sub.add_parser("daemon", help="Manage the fieldnotes background daemon")
    daemon_sub = daemon_p.add_subparsers(dest="daemon_command")
    daemon_sub.add_parser("install", help="Install and start the daemon service")
    daemon_sub.add_parser("uninstall", help="Stop and remove the daemon service")
    daemon_sub.add_parser("status", help="Show daemon status")
    daemon_sub.add_parser("start", help="Start the daemon service")
    daemon_sub.add_parser("stop", help="Stop the daemon service")

    # ── init ───────────────────────────────────────────────────────
    sub.add_parser(
        "init",
        help="Create ~/.fieldnotes/ directory and generate default config",
    )

    # ── setup-claude ─────────────────────────────────────────────────
    sub.add_parser(
        "setup-claude",
        help="Configure fieldnotes as an MCP server for Claude Desktop",
    )

    # ── setup-gastown ────────────────────────────────────────────────
    gastown_p = sub.add_parser(
        "setup-gastown",
        help="Configure fieldnotes as MCP server in a GasTown rig",
    )
    gastown_p.add_argument(
        "--rig-root",
        type=Path,
        default=None,
        help="GasTown rig root (auto-detected if omitted)",
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

    if args.command == "init":
        from worker.init import init

        return init()

    if args.command == "setup-claude":
        from worker.setup import setup_claude

        return setup_claude(config_path=args.config)

    if args.command == "serve":
        if args.daemon:
            from worker.serve_daemon import run_daemon

            try:
                run_daemon(config_path=args.config)
                return 0
            except Exception as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
        if not args.mcp:
            print(
                "error: specify --mcp or --daemon to start the server",
                file=sys.stderr,
            )
            return 1
        from worker.mcp_server import run_server

        try:
            run_server(config_path=args.config)
            return 0
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    if args.command == "daemon":
        from worker import daemon

        if args.daemon_command == "install":
            return daemon.install()
        if args.daemon_command == "uninstall":
            return daemon.uninstall()
        if args.daemon_command == "status":
            return daemon.status()
        if args.daemon_command == "start":
            return daemon.start()
        if args.daemon_command == "stop":
            return daemon.stop()
        print("Usage: fieldnotes daemon {install,uninstall,status,start,stop}",
              file=sys.stderr)
        return 1

    if args.command == "setup-gastown":
        from worker.gastown import setup_gastown

        try:
            return setup_gastown(
                config_path=args.config,
                rig_root=args.rig_root,
            )
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    if args.command == "search":
        if args.top_k < 1:
            print("error: --top-k must be a positive integer", file=sys.stderr)
            return 1
        query = " ".join(args.query)
        try:
            return _run_search(
                query,
                config_path=args.config,
                top_k=args.top_k,
            )
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    if args.command == "topics":
        try:
            return _run_topics(args, config_path=args.config)
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    print(f"error: unknown command {args.command!r}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
