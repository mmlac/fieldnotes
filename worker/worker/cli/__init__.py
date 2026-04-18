"""CLI entry point for the ``fieldnotes`` command.

Subcommands: search, ask, topics, serve, service, init, setup-claude, etc.
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
    serve_progress = serve_p.add_mutually_exclusive_group()
    serve_progress.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        default=None,
        help="Force the live progress display on (default: auto-detect TTY)",
    )
    serve_progress.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="Disable the live progress display",
    )

    # ── service ────────────────────────────────────────────────────
    service_p = sub.add_parser(
        "service", help="Manage the fieldnotes background service"
    )
    service_sub = service_p.add_subparsers(dest="service_command")
    service_sub.add_parser("install", help="Install and start the service")
    service_sub.add_parser("uninstall", help="Stop and remove the service")
    service_sub.add_parser("status", help="Show service status")
    service_sub.add_parser("start", help="Start the service")
    service_sub.add_parser("stop", help="Stop the service")

    # ── daemon (deprecated alias) ─────────────────────────────────
    daemon_p = sub.add_parser(
        "daemon", help="(deprecated: use 'service') Manage the background daemon"
    )
    daemon_sub = daemon_p.add_subparsers(dest="daemon_command")
    daemon_sub.add_parser("install", help="Install and start the daemon service")
    daemon_sub.add_parser("uninstall", help="Stop and remove the daemon service")
    daemon_sub.add_parser("status", help="Show daemon status")
    daemon_sub.add_parser("start", help="Start the daemon service")
    daemon_sub.add_parser("stop", help="Stop the daemon service")

    # ── init ───────────────────────────────────────────────────────
    init_p = sub.add_parser(
        "init",
        help="Create ~/.fieldnotes/ directory and generate default config",
    )
    init_p.add_argument(
        "--with-docker",
        action="store_true",
        dest="with_docker",
        help="Generate .env file and start Docker infrastructure",
    )
    init_p.add_argument(
        "--non-interactive",
        action="store_true",
        dest="non_interactive",
        help="Skip interactive prompts (use defaults)",
    )
    init_p.add_argument(
        "--compose-file",
        type=Path,
        default=None,
        dest="compose_file",
        metavar="PATH",
        help="Use a custom docker-compose.yml instead of the bundled one",
    )

    # ── update ─────────────────────────────────────────────────────
    sub.add_parser(
        "update",
        help="Update infrastructure files from the installed package",
    )

    # ── doctor ─────────────────────────────────────────────────────
    sub.add_parser(
        "doctor",
        help="Run pre-flight checks on config, infrastructure, and models",
    )

    # ── up / stop / down ───────────────────────────────────────────
    for cmd_name, cmd_help in [
        ("up", "Start Docker infrastructure (docker compose up -d)"),
        ("stop", "Stop Docker containers without removing them"),
        ("down", "Tear down Docker infrastructure (docker compose down)"),
    ]:
        p = sub.add_parser(cmd_name, help=cmd_help)
        p.add_argument(
            "--compose-file",
            type=Path,
            default=None,
            dest="compose_file",
            metavar="PATH",
            help="Path to a custom docker-compose.yml",
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

    # ── ask ─────────────────────────────────────────────────────────
    ask_p = sub.add_parser(
        "ask",
        help="Ask a question against your knowledge graph",
    )
    ask_p.add_argument(
        "question",
        nargs="*",
        default=None,
        help="Question to ask (omit for interactive REPL mode)",
    )
    ask_p.add_argument(
        "--verbose",
        action="store_true",
        dest="ask_verbose",
        help="Show query details (Cypher, vector scores, context size)",
    )
    ask_p.add_argument(
        "--resume",
        nargs="?",
        const="",
        default=None,
        metavar="ID",
        help="Resume a previous conversation (omit ID for most recent)",
    )
    ask_p.add_argument(
        "--history",
        action="store_true",
        dest="show_history",
        help="List past conversations",
    )
    ask_p.add_argument(
        "--no-stream",
        action="store_true",
        dest="no_stream",
        help="Disable streaming — collect full response before printing",
    )
    ask_p.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output structured JSON (question, answer, sources, timing)",
    )

    # ── timeline ────────────────────────────────────────────────────
    timeline_p = sub.add_parser(
        "timeline",
        help="Show activity across all indexed sources within a time range",
    )
    timeline_p.add_argument(
        "--since",
        default="24h",
        help="Start time (ISO 8601 or relative: '24h', '7d', '2w'). Default: 24h",
    )
    timeline_p.add_argument(
        "--until",
        default="now",
        help="End time (ISO 8601 or relative). Default: now",
    )
    timeline_p.add_argument(
        "--source",
        default=None,
        dest="source_type",
        metavar="SOURCE",
        help="Filter to one source: obsidian, omnifocus, gmail, file, repositories, apps",
    )
    timeline_p.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum entries to return (default: 50)",
    )
    timeline_p.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Machine-readable JSON output",
    )

    # ── connections ──────────────────────────────────────────────────
    connections_p = sub.add_parser(
        "connections",
        help="Surface semantically similar but unlinked documents",
    )
    connections_p.add_argument(
        "--source-id",
        default=None,
        help="Focus on a specific document by source_id",
    )
    connections_p.add_argument(
        "--source",
        dest="source_type",
        default=None,
        help="Seed from a specific source type (e.g. file, obsidian)",
    )
    connections_p.add_argument(
        "--threshold",
        type=float,
        default=0.82,
        help="Minimum cosine similarity (default: 0.82)",
    )
    connections_p.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Max suggestions (default: 20)",
    )
    connections_p.add_argument(
        "--cross-source",
        action="store_true",
        help="Only show connections between different source types",
    )
    connections_p.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Machine-readable JSON output",
    )

    # ── digest ──────────────────────────────────────────────────────
    digest_p = sub.add_parser(
        "digest",
        help="Summarize recent activity across all indexed sources",
    )
    digest_p.add_argument(
        "--since",
        default="24h",
        help="Start time (ISO 8601 or relative: '24h', '7d', '2w'). Default: 24h",
    )
    digest_p.add_argument(
        "--until",
        default="now",
        help="End time (ISO 8601 or relative). Default: now",
    )
    digest_p.add_argument(
        "--summarize",
        action="store_true",
        help="Generate an LLM summary paragraph",
    )
    digest_p.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Machine-readable JSON output",
    )

    # ── cluster ─────────────────────────────────────────────────────
    cluster_p = sub.add_parser(
        "cluster",
        help="Run the clustering pipeline manually",
    )
    cluster_p.add_argument(
        "--min-cluster-size",
        type=int,
        default=None,
        help="Override HDBSCAN min_cluster_size (default from config)",
    )
    cluster_p.add_argument(
        "--force",
        action="store_true",
        help="Run even if corpus is below min_corpus_size",
    )

    # ── backup ─────────────────────────────────────────────────────
    backup_p = sub.add_parser(
        "backup",
        help="Create or list backups of fieldnotes data and databases",
    )
    backup_p.add_argument(
        "--keep",
        type=int,
        default=None,
        metavar="N",
        help="Only keep the N most recent backups, delete older ones",
    )
    backup_sub = backup_p.add_subparsers(dest="backup_command")
    create_p = backup_sub.add_parser(
        "create", help="Create a backup (default when no subcommand)"
    )
    create_p.add_argument(
        "--keep",
        type=int,
        default=None,
        metavar="N",
        help="Only keep the N most recent backups, delete older ones",
    )
    backup_sub.add_parser("list", help="List existing backups")
    schedule_p = backup_sub.add_parser(
        "schedule", help="Install or remove a daily scheduled backup"
    )
    schedule_p.add_argument(
        "--remove",
        action="store_true",
        help="Remove the scheduled backup instead of installing it",
    )
    schedule_p.add_argument(
        "--keep",
        type=int,
        default=None,
        metavar="N",
        help="Only keep the N most recent backups per scheduled run",
    )

    # ── restore ────────────────────────────────────────────────────
    restore_p = sub.add_parser(
        "restore",
        help="Restore fieldnotes data and databases from a backup",
    )
    restore_p.add_argument(
        "backup_name",
        help="Backup filename (from 'fieldnotes backup list') or full path",
    )

    # ── queue ──────────────────────────────────────────────────────
    queue_p = sub.add_parser("queue", help="Inspect and manage the ingestion queue")
    queue_p.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Machine-readable JSON output",
    )
    queue_sub = queue_p.add_subparsers(dest="queue_command")

    top_p = queue_sub.add_parser("top", help="Show oldest N items (first in queue)")
    top_p.add_argument(
        "n", nargs="?", type=int, default=20, help="Number of items (default: 20)"
    )
    top_p.add_argument(
        "--status", default=None, help="Filter by status: pending, processing, failed"
    )
    top_p.add_argument(
        "--source", default=None, dest="source_type", help="Filter by source type"
    )

    tail_p = queue_sub.add_parser("tail", help="Show newest N items (last in queue)")
    tail_p.add_argument(
        "n", nargs="?", type=int, default=20, help="Number of items (default: 20)"
    )
    tail_p.add_argument(
        "--status", default=None, help="Filter by status: pending, processing, failed"
    )
    tail_p.add_argument(
        "--source", default=None, dest="source_type", help="Filter by source type"
    )

    queue_sub.add_parser("retry", help="Reset all failed items to pending")

    purge_p = queue_sub.add_parser("purge", help="Delete items by status")
    purge_p.add_argument(
        "--status",
        default="failed",
        help="Status to purge (default: failed)",
    )

    queue_sub.add_parser("migrate", help="Import old cursor JSON files into queue DB")

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
        stream=sys.stderr,
    )

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "init":
        from worker.init import init

        return init(
            with_docker=args.with_docker,
            non_interactive=args.non_interactive,
            compose_file=args.compose_file,
        )

    if args.command == "update":
        from worker.init import update_infrastructure

        return update_infrastructure()

    if args.command == "doctor":
        from worker.doctor import doctor

        return doctor(config_path=args.config)

    if args.command in ("up", "stop", "down"):
        from worker.infra import infra_down, infra_stop, infra_up

        fn = {"up": infra_up, "stop": infra_stop, "down": infra_down}[args.command]
        return fn(compose_file=args.compose_file)

    if args.command == "setup-claude":
        from worker.setup import setup_claude

        return setup_claude(config_path=args.config)

    if args.command == "serve":
        if args.daemon:
            from worker.serve_daemon import run_daemon

            try:
                run_daemon(config_path=args.config, progress=args.progress)
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

    if args.command == "service":
        from worker import service

        if args.service_command == "install":
            return service.install()
        if args.service_command == "uninstall":
            return service.uninstall()
        if args.service_command == "status":
            return service.status()
        if args.service_command == "start":
            return service.start()
        if args.service_command == "stop":
            return service.stop()
        print(
            "Usage: fieldnotes service {install,uninstall,status,start,stop}",
            file=sys.stderr,
        )
        return 1

    if args.command == "daemon":
        from worker import service

        if args.daemon_command == "install":
            return service.install()
        if args.daemon_command == "uninstall":
            return service.uninstall()
        if args.daemon_command == "status":
            return service.status()
        if args.daemon_command == "start":
            return service.start()
        if args.daemon_command == "stop":
            return service.stop()
        print(
            "Usage: fieldnotes daemon {install,uninstall,status,start,stop}",
            file=sys.stderr,
        )
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

    if args.command == "ask":
        if args.show_history:
            from worker.cli.ask import run_history

            return run_history()

        from worker.cli.ask import run_ask

        question = " ".join(args.question) if args.question else None
        # Auto-disable streaming when stdout is not a TTY (piped/redirected).
        stream = not args.no_stream and not args.json_output
        if stream and not sys.stdout.isatty():
            stream = False
        try:
            return run_ask(
                question,
                config_path=args.config,
                verbose=args.ask_verbose,
                resume_id=args.resume,
                stream=stream,
                json_output=args.json_output,
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

    if args.command == "timeline":
        from worker.cli.timeline import run_timeline

        try:
            return run_timeline(
                since=args.since,
                until=args.until,
                source_type=args.source_type,
                limit=args.limit,
                json_output=args.json_output,
                config_path=args.config,
            )
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    if args.command == "connections":
        from worker.cli.connections import run_connections

        try:
            return run_connections(
                config_path=args.config,
                source_id=args.source_id,
                source_type=args.source_type,
                threshold=args.threshold,
                limit=args.limit,
                cross_source=args.cross_source,
                json_output=args.json_output,
            )
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    if args.command == "digest":
        from worker.cli.digest import run_digest

        try:
            return run_digest(
                since=args.since,
                until=args.until,
                summarize=args.summarize,
                json_output=args.json_output,
                config_path=args.config,
            )
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    if args.command == "queue":
        from worker.cli.queue import (
            run_queue_list,
            run_queue_migrate,
            run_queue_purge,
            run_queue_retry,
            run_queue_summary,
        )

        try:
            if args.queue_command == "top":
                return run_queue_list(
                    n=args.n,
                    order="asc",
                    status=args.status,
                    source_type=args.source_type,
                    config_path=args.config,
                    json_output=args.json_output,
                )
            if args.queue_command == "tail":
                return run_queue_list(
                    n=args.n,
                    order="desc",
                    status=args.status,
                    source_type=args.source_type,
                    config_path=args.config,
                    json_output=args.json_output,
                )
            if args.queue_command == "retry":
                return run_queue_retry(config_path=args.config)
            if args.queue_command == "purge":
                return run_queue_purge(
                    status=args.status,
                    config_path=args.config,
                )
            if args.queue_command == "migrate":
                return run_queue_migrate(config_path=args.config)
            # No subcommand — show summary
            return run_queue_summary(
                config_path=args.config,
                json_output=args.json_output,
            )
        except SystemExit:
            return 1
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    if args.command == "cluster":
        from worker.cli.cluster import run_cluster

        try:
            return run_cluster(
                config_path=args.config,
                min_cluster_size=args.min_cluster_size,
                force=args.force,
            )
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    if args.command == "backup":
        from worker.backup import backup as do_backup
        from worker.backup import list_backups, schedule_backup

        if args.backup_command == "list":
            return list_backups()
        if args.backup_command == "schedule":
            return schedule_backup(remove=args.remove, keep=args.keep)
        # No subcommand or "create" both mean: create a backup.
        keep = getattr(args, "keep", None)
        return do_backup(keep=keep)

    if args.command == "restore":
        from pathlib import Path as _Path

        from worker.backup import restore

        return restore(_Path(args.backup_name))

    print(f"error: unknown command {args.command!r}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
