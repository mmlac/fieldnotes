"""``fieldnotes reindex-references`` — backfill REFERENCES edges for the existing corpus.

Walks CalendarEvent, Email, SlackMessage, and File (obsidian) nodes in Neo4j,
re-extracts REFERENCES hints from their stored chunk text, and upserts the
resulting edges. Idempotent — re-running produces the same edges.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase

from worker.config import load_config
from worker.parsers.base import extract_source_link_hints
from worker.pipeline.writer import _write_graph_hint

logger = logging.getLogger(__name__)

SUPPORTED_LABELS: tuple[str, ...] = (
    "CalendarEvent",
    "Email",
    "SlackMessage",
    "ObsidianNote",
)

# (neo4j_label, source_type_filter_or_None, hint_subject_label)
_LABEL_SPEC: dict[str, tuple[str, str | None, str]] = {
    "CalendarEvent": ("CalendarEvent", None, "CalendarEvent"),
    "Email": ("Email", None, "Email"),
    "SlackMessage": ("SlackMessage", None, "SlackMessage"),
    "ObsidianNote": ("File", "obsidian", "File"),
}


def _write_hints_tx(tx: Any, hints: list) -> None:
    for hint in hints:
        _write_graph_hint(tx, hint)


def _fetch_nodes(
    session: Any,
    neo4j_label: str,
    source_type_filter: str | None,
) -> list[tuple[str, str]]:
    """Return (source_id, combined_chunk_text) pairs for nodes of the given label."""
    if source_type_filter:
        query = (
            f"MATCH (n:{neo4j_label} {{source_type: $st}}) "
            "OPTIONAL MATCH (n)-[:HAS_CHUNK]->(c:Chunk) "
            "WITH n, c ORDER BY c.chunk_index ASC "
            "RETURN n.source_id AS source_id, collect(c.text) AS chunks"
        )
        result = session.run(query, st=source_type_filter)
    else:
        query = (
            f"MATCH (n:{neo4j_label}) "
            "OPTIONAL MATCH (n)-[:HAS_CHUNK]->(c:Chunk) "
            "WITH n, c ORDER BY c.chunk_index ASC "
            "RETURN n.source_id AS source_id, collect(c.text) AS chunks"
        )
        result = session.run(query)

    rows: list[tuple[str, str]] = []
    for record in result:
        source_id = record["source_id"]
        if not source_id:
            continue
        chunks: list[str] = record["chunks"] or []
        text = "\n".join(c for c in chunks if c)
        rows.append((source_id, text))
    return rows


def _configure_vault_map(cfg: Any) -> None:
    """Attempt to discover obsidian vaults from config and set the global vault map."""
    try:
        from worker.sources.obsidian import discover_vaults
        from worker.parsers.base import configure_obsidian_vaults

        vault_paths_raw: list[str] = []
        for name, source_cfg in cfg.sources.items():
            if "obsidian" in name.lower():
                vp = source_cfg.settings.get("vault_paths", [])
                if isinstance(vp, list):
                    vault_paths_raw.extend(vp)

        if not vault_paths_raw:
            return

        search_paths = [Path(p).expanduser() for p in vault_paths_raw]
        vaults = discover_vaults(search_paths)
        if vaults:
            configure_obsidian_vaults({vault.name: str(vault) for vault in vaults})
            logger.debug("Configured %d obsidian vault(s) for URL resolution", len(vaults))
    except Exception as exc:
        logger.debug("Could not configure obsidian vaults: %s", exc)


def run_reindex_references(
    *,
    config_path: Path | None = None,
    dry_run: bool = False,
    label: str | None = None,
) -> int:
    """Backfill REFERENCES edges for existing corpus nodes.

    Returns an exit code (0 = success, 1 = error).
    """
    if label and label not in _LABEL_SPEC:
        print(
            f"error: unknown label {label!r}. Choose from: {', '.join(SUPPORTED_LABELS)}",
            file=sys.stderr,
        )
        return 1

    cfg = load_config(config_path)
    _configure_vault_map(cfg)

    targets = [label] if label else list(_LABEL_SPEC)

    driver = GraphDatabase.driver(
        cfg.neo4j.uri,
        auth=(cfg.neo4j.user, cfg.neo4j.password),
    )

    total_nodes = 0
    total_edges = 0

    try:
        with driver.session() as session:
            for cli_label in targets:
                neo4j_label, source_type_filter, subject_label = _LABEL_SPEC[cli_label]
                nodes = _fetch_nodes(session, neo4j_label, source_type_filter)

                for source_id, text in nodes:
                    total_nodes += 1
                    if not text:
                        continue
                    hints = extract_source_link_hints(text, source_id, subject_label)
                    if not hints:
                        continue
                    total_edges += len(hints)
                    if not dry_run:
                        session.execute_write(_write_hints_tx, hints)
    finally:
        driver.close()

    if dry_run:
        print(
            f"Dry run — would create {total_edges} REFERENCES edge(s) "
            f"across {total_nodes} node(s)."
        )
    else:
        print(
            f"Created {total_edges} REFERENCES edge(s) across {total_nodes} node(s)."
        )

    return 0
