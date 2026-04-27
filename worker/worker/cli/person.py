"""CLI handler for ``fieldnotes person`` — render a Person profile.

Wires :mod:`worker.query.person` (fn-364.1) into a Rich-rendered
terminal view and a stable ``--json`` schema.  Resolution flow:

* ``<identifier>`` is interpreted as email / ``slack-user:<team>/<user>``
  / fuzzy-name by the query layer's :func:`find_person`.
* ``--search`` forces fuzzy-name lookup even for inputs that would
  otherwise route to email matching (so ``--search "alice@home"`` is a
  fragment, not a malformed address).
* ``--self`` resolves to the canonical ``Person {is_self: true}``;
  errors clearly when the ``[me]`` block is not configured.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from neo4j import Driver, GraphDatabase

from worker.config import Neo4jConfig, load_config
from worker.query._time import parse_relative_time
from worker.query.person import (
    FileMention,
    IdentityMember,
    Interaction,
    OpenTask,
    Person,
    PersonProfile,
    RelatedPerson,
    TopicCount,
    files_mentioning,
    identity_cluster,
    open_tasks,
    recent_interactions,
    related_people,
    top_topics,
)
from worker.query.person import (
    _cluster_ids,
    _find_by_fuzzy_name,
    _find_person_tx,
)


_DEFAULT_SINCE = "30d"
_DEFAULT_LIMIT = 10


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------


def _open_driver(neo4j_cfg: Neo4jConfig) -> Driver:
    return GraphDatabase.driver(
        neo4j_cfg.uri, auth=(neo4j_cfg.user, neo4j_cfg.password)
    )


def _resolve_self(driver: Driver) -> Person | None:
    """Find the canonical Person flagged ``is_self = true``.

    The ``[me]`` block must be configured for this to make sense; the
    caller checks that and surfaces a clear error before we get here.
    """
    from worker.query.person import _canonical_for_id

    def _tx(tx: Any) -> Person | None:
        row = tx.run(
            "MATCH (p:Person {is_self: true}) RETURN id(p) AS id LIMIT 1"
        ).single()
        if not row:
            return None
        return _canonical_for_id(tx, int(row["id"]))

    with driver.session() as session:
        return session.execute_read(_tx)


def _resolve_identifier(
    driver: Driver,
    identifier: str,
    *,
    force_fuzzy: bool,
) -> Person | list[Person] | None:
    """Run the fn-364.1 resolver, optionally forcing fuzzy-name lookup."""
    if force_fuzzy:
        with driver.session() as session:
            return session.execute_read(_find_by_fuzzy_name, identifier)
    with driver.session() as session:
        return session.execute_read(_find_person_tx, identifier)


# ---------------------------------------------------------------------------
# Profile assembly
# ---------------------------------------------------------------------------


def _build_profile(
    driver: Driver,
    person: Person,
    *,
    since: datetime,
    limit: int,
) -> PersonProfile:
    profile = PersonProfile(person=person)
    profile.recent_interactions = recent_interactions(
        person.id, since, limit=limit, driver=driver
    )
    profile.top_topics = top_topics(person.id, k=limit, driver=driver)
    profile.related_people = related_people(person.id, k=limit, driver=driver)
    profile.open_tasks = open_tasks(person.id, driver=driver)
    profile.files = files_mentioning(person.id, k=limit, driver=driver)
    profile.identity_cluster = identity_cluster(person.id, driver=driver)
    return profile


def _source_count(profile: PersonProfile) -> int:
    """Number of distinct source types present across all sections."""
    sources: set[str] = set()
    for ix in profile.recent_interactions:
        if ix.source_type:
            sources.add(ix.source_type)
    if profile.open_tasks:
        sources.add("omnifocus")
    for f in profile.files:
        if f.source:
            sources.add(f.source)
    return len(sources)


def _sources_present(profile: PersonProfile) -> list[str]:
    sources: set[str] = set()
    for ix in profile.recent_interactions:
        if ix.source_type:
            sources.add(ix.source_type)
    if profile.open_tasks:
        sources.add("omnifocus")
    for f in profile.files:
        if f.source:
            sources.add(f.source)
    return sorted(sources)


def _last_seen(profile: PersonProfile) -> str | None:
    if not profile.recent_interactions:
        return None
    return profile.recent_interactions[0].timestamp


def _total_interactions(driver: Driver, person_id: int) -> int:
    """Count every doc-edge from any cluster member, regardless of date."""
    with driver.session() as session:
        cluster = session.execute_read(_cluster_ids, person_id)
        if not cluster:
            return 0
        rows = session.execute_read(
            lambda tx: tx.run(
                """
                MATCH (p:Person) WHERE id(p) IN $cluster
                MATCH (d)-[r]-(p)
                WHERE NOT d:Person AND type(r) <> 'SAME_AS'
                RETURN count(DISTINCT d) AS total
                """,
                cluster=cluster,
            ).data()
        )
    return int(rows[0]["total"]) if rows else 0


def _same_as_aliases(profile: PersonProfile) -> list[str]:
    return [m.member for m in profile.identity_cluster if m.member]


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


def _interaction_dict(ix: Interaction) -> dict[str, Any]:
    return {
        "timestamp": ix.timestamp,
        "source_type": ix.source_type,
        "title": ix.title,
        "snippet": ix.snippet,
        "edge_kind": ix.edge_kind,
    }


def _topic_dict(t: TopicCount) -> dict[str, Any]:
    return {"topic_name": t.topic_name, "doc_count": t.doc_count}


def _related_dict(r: RelatedPerson) -> dict[str, Any]:
    return {"name": r.name, "email": r.email, "shared_count": r.shared_count}


def _task_dict(t: OpenTask) -> dict[str, Any]:
    return {
        "title": t.title,
        "project": t.project,
        "tags": list(t.tags),
        "due": t.due,
        "defer": t.defer,
        "flagged": t.flagged,
    }


def _file_dict(f: FileMention) -> dict[str, Any]:
    return {"path": f.path, "mtime": f.mtime, "source": f.source}


def _cluster_dict(m: IdentityMember) -> dict[str, Any]:
    return asdict(m)


def _emit_json(
    identifier: str,
    profile: PersonProfile,
    total_interactions: int,
    is_self: bool,
) -> str:
    person = profile.person
    payload = {
        "identifier": identifier,
        "resolved": {
            "name": person.name or "",
            "email": person.email or "",
            "is_self": bool(is_self),
        },
        "sources_present": _sources_present(profile),
        "last_seen": _last_seen(profile),
        "total_interactions": total_interactions,
        "recent_interactions": [
            _interaction_dict(ix) for ix in profile.recent_interactions
        ],
        "top_topics": [_topic_dict(t) for t in profile.top_topics],
        "related_people": [_related_dict(r) for r in profile.related_people],
        "open_tasks": [_task_dict(t) for t in profile.open_tasks],
        "files_mentioning": [_file_dict(f) for f in profile.files],
        "identity_cluster": [_cluster_dict(m) for m in profile.identity_cluster],
    }
    return json.dumps(payload, indent=2, default=str)


# ---------------------------------------------------------------------------
# Rich output
# ---------------------------------------------------------------------------


def _fmt_ts(ts: str | None) -> str:
    if not ts:
        return "—"
    s = ts.rstrip("Z").split(".")[0]
    if "T" in s:
        date, time = s.split("T", 1)
        return f"{date} {time[:5]}"
    return s[:16]


def _emit_rich(
    profile: PersonProfile,
    total_interactions: int,
    since_label: str,
    limit: int,
    is_self: bool,
) -> None:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    person = profile.person
    aliases = _same_as_aliases(profile)
    src_count = _source_count(profile)

    # 1. Header
    title_bits: list[str] = []
    if person.name:
        title_bits.append(f"[bold]{person.name}[/bold]")
    if person.email:
        title_bits.append(person.email)
    if is_self:
        title_bits.append("[dim](self)[/dim]")
    header_lines: list[str] = [" · ".join(title_bits) or "(unnamed)"]
    if aliases:
        header_lines.append("Aliases: " + ", ".join(aliases))
    header_lines.append(
        f"Sources: {src_count}    "
        f"Last seen: {_fmt_ts(_last_seen(profile))}    "
        f"Total interactions: {total_interactions}"
    )
    console.print(Panel("\n".join(header_lines), title="Person", expand=False))

    # 2. Recent interactions
    if profile.recent_interactions:
        t = Table(title=f"Recent interactions (last {since_label})")
        t.add_column("When")
        t.add_column("Source")
        t.add_column("Edge")
        t.add_column("Title", overflow="fold")
        for ix in profile.recent_interactions:
            t.add_row(
                _fmt_ts(ix.timestamp),
                ix.source_type,
                ix.edge_kind,
                ix.title,
            )
        console.print(t)
    else:
        console.print(f"[dim]No recent interactions in last {since_label}[/dim]")

    # 3. Top topics
    if profile.top_topics:
        t = Table(title="Top topics")
        t.add_column("Topic")
        t.add_column("Docs", justify="right")
        for tp in profile.top_topics:
            t.add_row(tp.topic_name, str(tp.doc_count))
        console.print(t)
    else:
        console.print(f"[dim]No top topics in last {since_label}[/dim]")

    # 4. Related people
    if profile.related_people:
        t = Table(title="Related people")
        t.add_column("Name")
        t.add_column("Email")
        t.add_column("Shared", justify="right")
        for r in profile.related_people:
            t.add_row(r.name or "—", r.email or "—", str(r.shared_count))
        console.print(t)
    else:
        console.print(f"[dim]No related people in last {since_label}[/dim]")

    # 5. Open tasks
    if profile.open_tasks:
        t = Table(title="Open tasks (OmniFocus)")
        t.add_column("Title", overflow="fold")
        t.add_column("Project")
        t.add_column("Tags")
        t.add_column("Due")
        t.add_column("Flagged")
        for ot in profile.open_tasks:
            t.add_row(
                ot.title,
                ot.project or "—",
                ", ".join(ot.tags) if ot.tags else "—",
                _fmt_ts(ot.due) if ot.due else "—",
                "yes" if ot.flagged else "no",
            )
        console.print(t)
    else:
        console.print(f"[dim]No open tasks in last {since_label}[/dim]")

    # 6. Files mentioning
    if profile.files:
        t = Table(title="Files mentioning")
        t.add_column("Path", overflow="fold")
        t.add_column("Modified")
        t.add_column("Source")
        for f in profile.files:
            t.add_row(f.path, _fmt_ts(f.mtime), f.source or "—")
        console.print(t)
    else:
        console.print(f"[dim]No files mentioning in last {since_label}[/dim]")

    # 7. Identity cluster
    if profile.identity_cluster:
        t = Table(title="Identity cluster")
        t.add_column("Member")
        t.add_column("Match type")
        t.add_column("Confidence", justify="right")
        t.add_column("Cross source")
        for m in profile.identity_cluster:
            t.add_row(
                m.member,
                m.match_type or "—",
                f"{m.confidence:.2f}" if m.confidence is not None else "—",
                "yes" if m.cross_source else ("no" if m.cross_source is False else "—"),
            )
        console.print(t)
    else:
        console.print("[dim]No identity cluster (no SAME_AS edges)[/dim]")
    _ = limit  # currently informational; row caps applied at query time


def _print_disambiguation(
    candidates: list[Person],
    driver: Driver,
    *,
    json_output: bool,
) -> None:
    """Print a candidate table for ambiguous fuzzy-name matches."""
    rows: list[dict[str, Any]] = []
    for cand in candidates:
        last = None
        try:
            recent = recent_interactions(
                cand.id, since=datetime.fromtimestamp(0), limit=1, driver=driver
            )
            if recent:
                last = recent[0].timestamp
        except Exception:
            last = None
        rows.append(
            {
                "name": cand.name or "",
                "email": cand.email or "",
                "last_seen": last,
            }
        )

    if json_output:
        print(
            json.dumps(
                {"error": "ambiguous", "candidates": rows},
                indent=2,
                default=str,
            )
        )
        return

    from rich.console import Console
    from rich.table import Table

    console = Console(stderr=True)
    console.print(
        "[yellow]Ambiguous match — multiple people fit.  "
        "Re-run with a more specific identifier:[/yellow]"
    )
    t = Table()
    t.add_column("Name")
    t.add_column("Email")
    t.add_column("Last seen")
    for r in rows:
        t.add_row(r["name"], r["email"], _fmt_ts(r["last_seen"]))
    console.print(t)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_person(
    *,
    identifier: str | None,
    since: str = _DEFAULT_SINCE,
    limit: int = _DEFAULT_LIMIT,
    use_self: bool = False,
    search: str | None = None,
    json_output: bool = False,
    config_path: Path | None = None,
) -> int:
    """Resolve a Person and render their profile.  Returns exit code."""
    cfg = load_config(config_path)

    if use_self:
        if cfg.me is None or not cfg.me.emails:
            print(
                "error: --self requires a [me] block with at least one email "
                "in your config (~/.fieldnotes/config.toml)",
                file=sys.stderr,
            )
            return 2
        effective_identifier = "[self]"
    elif search is not None:
        if not search.strip():
            print("error: --search requires a non-empty name", file=sys.stderr)
            return 2
        effective_identifier = search
    elif identifier is None or not identifier.strip():
        print(
            "error: provide an identifier (email, slack-user:<team>/<user>, "
            "or name fragment) or use --self / --search",
            file=sys.stderr,
        )
        return 2
    else:
        effective_identifier = identifier

    try:
        since_dt = parse_relative_time(since)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if int(limit) <= 0:
        print("error: --limit must be a positive integer", file=sys.stderr)
        return 2

    driver = _open_driver(cfg.neo4j)
    try:
        if use_self:
            resolved: Person | list[Person] | None = _resolve_self(driver)
        else:
            resolved = _resolve_identifier(
                driver, effective_identifier, force_fuzzy=search is not None
            )

        if resolved is None:
            msg = (
                "No Person flagged is_self=true in the graph"
                if use_self
                else f"No Person found for {effective_identifier!r}"
            )
            if json_output:
                print(json.dumps({"error": msg}, indent=2))
            else:
                print(f"error: {msg}", file=sys.stderr)
            return 1

        if isinstance(resolved, list):
            _print_disambiguation(resolved, driver, json_output=json_output)
            return 1

        person = resolved
        profile = _build_profile(driver, person, since=since_dt, limit=int(limit))
        total = _total_interactions(driver, person.id)

        if json_output:
            print(
                _emit_json(
                    effective_identifier,
                    profile,
                    total_interactions=total,
                    is_self=use_self,
                )
            )
        else:
            _emit_rich(
                profile,
                total_interactions=total,
                since_label=since,
                limit=int(limit),
                is_self=use_self,
            )
        return 0
    finally:
        driver.close()
