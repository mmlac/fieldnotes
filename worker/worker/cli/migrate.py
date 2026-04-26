"""``fieldnotes migrate gmail-multiaccount`` — one-shot retag of legacy
Gmail+Calendar artifacts under a chosen account label.

Rewrites old-shape doc URIs (no account segment) to new-shape
(``gmail://<account>/...`` / ``google-calendar://<account>/...``) across:

  1. SQLite persistent queue (``~/.fieldnotes/data/queue.db``):
     ``queue.source_id`` for rows in any status, ``queue.payload`` JSON
     embedded source_id, and ``cursors.key`` rows for ``gmail`` /
     ``calendar``.
  2. Neo4j: ``Document``-style source nodes' ``source_id`` property,
     derivative ``Chunk.id`` (``{source_id}:chunk:N``), and fallback
     ``Person`` nodes whose ``source_id`` is itself a doc URI
     (``google-calendar://event/X/attendee/Y``).
  3. Qdrant: chunk ``payload.source_id`` (vectors preserved in place).
  4. Token files: rename ``gmail_token.json`` →
     ``gmail_token-<account>.json`` and same for ``calendar_token.json``.
  5. Legacy JSON cursor files (``gmail_cursor*.json``,
     ``calendar_cursor*.json``): deleted (deprecated since cursors moved
     into ``queue.db.cursors``).
  6. ``config.toml``: rewrite legacy ``[sources.gmail]`` /
     ``[sources.google_calendar]`` flat sections into account-keyed form.
     Original is backed up to
     ``config.toml.backup-<UTC-timestamp>``.

Refuses to run while the daemon is alive (``queue.db`` is shared and
concurrent writes race even under WAL).  ``--force-running`` overrides
with a strong warning.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────

_ACCOUNT_RE = re.compile(r"^[a-z][a-z0-9_-]{0,30}$")

# Legacy doc-URI prefixes (no account segment).  Order matters for
# detection-only — substitution uses the full prefix string and is
# unambiguous because the account-namespaced form always has a slash
# *before* the kind segment (e.g. ``gmail://<account>/thread/``).
_OLD_PREFIXES: tuple[str, ...] = (
    "gmail://thread/",
    "gmail://message/",
    "google-calendar://event/",
    "google-calendar://series/",
)


# ── Data ──────────────────────────────────────────────────────────────


@dataclass
class MigrationCounts:
    """Counts surfaced in the dry-run summary and final report."""

    queue_rows: int = 0
    cursor_rows: int = 0
    documents: int = 0
    chunks: int = 0
    fallback_persons: int = 0
    qdrant_points: int = 0
    files_renamed: list[str] = field(default_factory=list)
    files_deleted: list[str] = field(default_factory=list)
    config_rewritten: bool = False


# ── Daemon detection ──────────────────────────────────────────────────


def detect_daemon() -> tuple[bool, str | None]:
    """Detect a running fieldnotes daemon.

    Returns ``(is_running, hint)`` where *hint* is a human-readable
    description of how it was detected.  Defaults to PID file → pgrep.
    """
    pid_file = Path.home() / ".fieldnotes" / "daemon.pid"
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
        except (ValueError, OSError):
            pid = 0
        if pid > 0:
            try:
                os.kill(pid, 0)
                return True, f"PID file {pid_file} → PID {pid} alive"
            except ProcessLookupError:
                pass
            except PermissionError:
                # Process exists, owned by another user.
                return True, f"PID file {pid_file} → PID {pid} alive (other user)"

    try:
        result = subprocess.run(
            ["pgrep", "-f", "fieldnotes serve"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False, None

    if result.returncode == 0 and result.stdout.strip():
        # Filter out our own PID and any pgrep helper.
        own_pid = str(os.getpid())
        pids = [
            p for p in result.stdout.strip().splitlines() if p and p != own_pid
        ]
        if pids:
            return True, f"pgrep -f 'fieldnotes serve' → PID(s) {','.join(pids)}"

    return False, None


# ── Account-label resolution ──────────────────────────────────────────


def resolve_account(
    account_arg: str | None,
    *,
    yes: bool,
    input_fn: Callable[[str], str] = input,
) -> str:
    """Resolve account label from --account flag, prompt, or default.

    Validates the label against ``^[a-z][a-z0-9_-]{0,30}$``.  Raises
    ``ValueError`` on an invalid ``--account``.
    """
    if account_arg is not None:
        if not _ACCOUNT_RE.match(account_arg):
            raise ValueError(
                f"--account {account_arg!r} is invalid: must match "
                f"^[a-z][a-z0-9_-]{{0,30}}$"
            )
        return account_arg

    if yes:
        return "default"

    while True:
        try:
            resp = input_fn("Account label [default]: ").strip()
        except EOFError:
            return "default"
        if not resp:
            return "default"
        if _ACCOUNT_RE.match(resp):
            return resp
        print(
            f"Invalid account label {resp!r}. Must match "
            f"^[a-z][a-z0-9_-]{{0,30}}$",
            file=sys.stderr,
        )


# ── source_id rewriting ───────────────────────────────────────────────


def rewrite_source_id(old_sid: str, account: str) -> str | None:
    """Return the new-shape source_id for *old_sid*, or *None* if it
    doesn't match a legacy prefix.

    Examples:
      ``gmail://thread/abc`` → ``gmail://<account>/thread/abc``
      ``google-calendar://event/X/attendee/3`` →
      ``google-calendar://<account>/event/X/attendee/3``
    """
    for prefix in _OLD_PREFIXES:
        if old_sid.startswith(prefix):
            scheme, rest = prefix.split("://", 1)
            kind = rest.rstrip("/")  # 'thread', 'message', 'event', 'series'
            tail = old_sid[len(prefix):]
            return f"{scheme}://{account}/{kind}/{tail}"
    return None


# ── Detection (read-only counts) ──────────────────────────────────────


def detect_queue_counts(db_path: Path) -> tuple[int, int]:
    """Return ``(queue_rows, cursor_rows)`` needing migration."""
    if not db_path.exists():
        return 0, 0
    conn = sqlite3.connect(str(db_path))
    try:
        like_clause = " OR ".join(["source_id LIKE ?"] * len(_OLD_PREFIXES))
        params = [f"{p}%" for p in _OLD_PREFIXES]
        row_count = conn.execute(
            f"SELECT count(*) FROM queue WHERE {like_clause}", params
        ).fetchone()[0]
        try:
            cursor_count = conn.execute(
                "SELECT count(*) FROM cursors WHERE key IN ('gmail', 'calendar')"
            ).fetchone()[0]
        except sqlite3.OperationalError:
            cursor_count = 0
        return row_count, cursor_count
    finally:
        conn.close()


def detect_neo4j_counts(session: Any) -> tuple[int, int, int]:
    """Return ``(documents, chunks, fallback_persons)`` to migrate.

    ``session`` is a Neo4j session (or a duck-typed mock with ``.run``).
    """
    doc_clauses = " OR ".join(
        f"n.source_id STARTS WITH '{p}'" for p in _OLD_PREFIXES
    )
    chunk_clauses = " OR ".join(
        f"c.id STARTS WITH '{p}'" for p in _OLD_PREFIXES
    )
    person_clauses = " OR ".join(
        f"p.source_id STARTS WITH '{p}'" for p in _OLD_PREFIXES
    )

    docs = _scalar(
        session,
        f"MATCH (n) WHERE n.source_id IS NOT NULL AND ({doc_clauses}) "
        f"RETURN count(n)",
    )
    chunks = _scalar(
        session,
        f"MATCH (c:Chunk) WHERE c.id IS NOT NULL AND ({chunk_clauses}) "
        f"RETURN count(c)",
    )
    persons = _scalar(
        session,
        f"MATCH (p:Person) WHERE p.source_id IS NOT NULL AND ({person_clauses}) "
        f"RETURN count(p)",
    )
    return docs, chunks, persons


def detect_qdrant_count(qdrant: Any, collection: str) -> int:
    """Return number of Qdrant points whose payload.source_id needs
    migration.

    Counts via scroll because the qdrant-client ``count`` API doesn't
    support starts-with on string fields portably.  For a 50K-point
    corpus this is one round-trip pull.
    """
    return sum(1 for _ in _iter_old_shape_points(qdrant, collection, batch=2048))


# ── Mutators ──────────────────────────────────────────────────────────


def migrate_queue(db_path: Path, account: str) -> dict[str, int]:
    """Retag ``queue.source_id`` + payload JSON + ``cursors`` rows.

    Returns ``{"rows": N, "cursors": M}``.  No-op if the DB is absent.
    """
    if not db_path.exists():
        return {"rows": 0, "cursors": 0}

    conn = sqlite3.connect(str(db_path))
    try:
        # Match the writer-side WAL settings so we don't deadlock with a
        # running daemon if --force-running is used.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("BEGIN IMMEDIATE")
        try:
            row_count = 0

            # 1. Update queue.source_id for each old-shape prefix.
            for prefix in _OLD_PREFIXES:
                scheme, rest = prefix.split("://", 1)
                kind = rest.rstrip("/")
                cur = conn.execute(
                    "UPDATE queue SET source_id = ? || substr(source_id, ?) "
                    "WHERE source_id LIKE ?",
                    (
                        f"{scheme}://{account}/{kind}/",
                        len(prefix) + 1,  # SQL substr is 1-indexed
                        f"{prefix}%",
                    ),
                )
                row_count += cur.rowcount

            # 2. Rewrite source_id embedded in queue.payload JSON.
            #    JSON format is opaque to SQL; decode in Python.
            old_payload_rows = conn.execute(
                "SELECT id, payload FROM queue WHERE payload IS NOT NULL"
            ).fetchall()
            for queue_id, payload_json in old_payload_rows:
                try:
                    payload = json.loads(payload_json)
                except (TypeError, json.JSONDecodeError):
                    continue
                if not isinstance(payload, dict):
                    continue
                old_sid = payload.get("source_id")
                if not isinstance(old_sid, str):
                    continue
                new_sid = rewrite_source_id(old_sid, account)
                if new_sid is None:
                    continue
                payload["source_id"] = new_sid
                conn.execute(
                    "UPDATE queue SET payload = ? WHERE id = ?",
                    (json.dumps(payload, default=str), queue_id),
                )

            # 3. Update cursor keys from 'gmail' → 'gmail:<account>' and
            #    'calendar' → 'calendar:<account>'.
            cursor_count = 0
            for old_key, new_key in [
                ("gmail", f"gmail:{account}"),
                ("calendar", f"calendar:{account}"),
            ]:
                cur = conn.execute(
                    "UPDATE cursors SET key = ? WHERE key = ?",
                    (new_key, old_key),
                )
                cursor_count += cur.rowcount

            conn.execute("COMMIT")
            return {"rows": row_count, "cursors": cursor_count}
        except BaseException:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.close()


def migrate_neo4j(session: Any, account: str) -> dict[str, int]:
    """Rewrite source_id-shaped properties in Neo4j.

    Updates source nodes, ``Chunk.id``, and fallback ``Person`` nodes.
    Each kind runs in its own transaction.

    Returns ``{"documents": D, "chunks": C, "persons": P}``.
    """
    docs = 0
    chunks = 0
    persons = 0

    for prefix in _OLD_PREFIXES:
        scheme, rest = prefix.split("://", 1)
        kind = rest.rstrip("/")
        new_prefix = f"{scheme}://{account}/{kind}/"

        # Source nodes (Email/Thread/CalendarEvent/CalendarSeries/etc.).
        result = session.run(
            "MATCH (n) WHERE n.source_id IS NOT NULL "
            "AND n.source_id STARTS WITH $old_prefix "
            "SET n.source_id = $new_prefix + substring(n.source_id, $cut) "
            "RETURN count(n) AS c",
            old_prefix=prefix,
            new_prefix=new_prefix,
            cut=len(prefix),
        )
        docs += _consume_count(result)

        # Chunk nodes (id = {source_id}:chunk:{idx}; embeds source_id).
        result = session.run(
            "MATCH (c:Chunk) WHERE c.id IS NOT NULL "
            "AND c.id STARTS WITH $old_prefix "
            "SET c.id = $new_prefix + substring(c.id, $cut) "
            "RETURN count(c) AS c",
            old_prefix=prefix,
            new_prefix=new_prefix,
            cut=len(prefix),
        )
        chunks += _consume_count(result)

        # Fallback Person nodes (source_id is itself a doc URI for the
        # display-name-only attendee fallback).  Re-MERGE so any same-key
        # node already present collapses with this one.
        result = session.run(
            "MATCH (p:Person) WHERE p.source_id IS NOT NULL "
            "AND p.source_id STARTS WITH $old_prefix "
            "SET p.source_id = $new_prefix + substring(p.source_id, $cut) "
            "RETURN count(p) AS c",
            old_prefix=prefix,
            new_prefix=new_prefix,
            cut=len(prefix),
        )
        persons += _consume_count(result)

    return {"documents": docs, "chunks": chunks, "persons": persons}


def migrate_qdrant(
    qdrant: Any, collection: str, account: str, *, batch: int = 256
) -> dict[str, int]:
    """Rewrite ``payload.source_id`` for every old-shape Qdrant point.

    Vectors and point IDs are preserved (set_payload only touches
    payload).  Returns ``{"points": N}``.
    """
    count = 0
    for point in _iter_old_shape_points(qdrant, collection, batch=batch):
        old_sid = point.payload.get("source_id") if point.payload else None
        if not isinstance(old_sid, str):
            continue
        new_sid = rewrite_source_id(old_sid, account)
        if new_sid is None:
            continue
        qdrant.set_payload(
            collection_name=collection,
            payload={"source_id": new_sid},
            points=[point.id],
        )
        count += 1
    return {"points": count}


def migrate_files(data_dir: Path, account: str) -> dict[str, list[str]]:
    """Rename token files and delete deprecated cursor JSON files.

    Token files: ``gmail_token.json`` → ``gmail_token-<account>.json``,
    same for ``calendar_token.json``.  Missing files are logged but not
    treated as errors.

    Cursor files (deleted, deprecated since cursors moved to
    ``queue.db.cursors``):
      - ``gmail_cursor*.json``
      - ``calendar_cursor*.json``
    """
    renamed: list[str] = []
    deleted: list[str] = []

    rename_map = {
        "gmail_token.json": f"gmail_token-{account}.json",
        "calendar_token.json": f"calendar_token-{account}.json",
    }
    for src_name, dst_name in rename_map.items():
        src = data_dir / src_name
        dst = data_dir / dst_name
        if not src.exists():
            logger.info("Skip rename: %s missing", src)
            continue
        if dst.exists():
            logger.warning(
                "Skip rename: destination %s already exists", dst
            )
            continue
        src.rename(dst)
        renamed.append(f"{src_name} → {dst_name}")
        logger.info("Renamed %s → %s", src, dst)

    # Cursor files: deprecated, delete all variants.
    for pattern in ("gmail_cursor*.json", "calendar_cursor*.json"):
        for path in sorted(data_dir.glob(pattern)):
            try:
                path.unlink()
            except OSError as exc:
                logger.warning("Failed to delete %s: %s", path, exc)
                continue
            deleted.append(path.name)
            logger.info("Deleted deprecated cursor file %s", path)

    return {"renamed": renamed, "deleted": deleted}


def migrate_config(config_path: Path, account: str) -> bool:
    """Rewrite ``config.toml`` to wrap legacy flat ``[sources.gmail]`` /
    ``[sources.google_calendar]`` sections into account-keyed form.

    Backup the original to ``config.toml.backup-<UTC-timestamp>`` and
    write atomically.  No-op (returns False) if the config is already in
    multi-account shape or the file does not exist.
    """
    if not config_path.exists():
        logger.info("Config %s does not exist; skipping rewrite", config_path)
        return False

    import tomlkit

    original = config_path.read_text()
    doc = tomlkit.parse(original)

    sources = doc.get("sources")
    if sources is None:
        return False

    rewrote = False
    for section_name in ("gmail", "google_calendar"):
        section = sources.get(section_name)
        if section is None:
            continue
        # Multi-account form: every direct child is itself a table.
        # Flat form: at least one direct child is a scalar.
        scalar_keys = [
            k for k, v in section.items() if not _is_table_like(v)
        ]
        if not scalar_keys:
            # Already account-keyed.
            continue
        # Wrap the flat section into [sources.<section_name>.<account>].
        new_inner = tomlkit.table()
        for key, value in list(section.items()):
            new_inner[key] = value
        wrapper = tomlkit.table(is_super_table=True)
        wrapper[account] = new_inner
        sources[section_name] = wrapper
        rewrote = True

    if not rewrote:
        return False

    # Backup.
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = config_path.with_name(f"{config_path.name}.backup-{ts}")
    backup_path.write_text(original)

    # Atomic write: temp file in same directory, then rename.
    tmp_path = config_path.with_name(f"{config_path.name}.tmp-{ts}")
    tmp_path.write_text(tomlkit.dumps(doc))
    tmp_path.replace(config_path)

    logger.info(
        "Rewrote %s (backup: %s)",
        config_path.name,
        backup_path.name,
    )
    return True


# ── Interleaved-migration check ───────────────────────────────────────


def detect_interleaved(
    session: Any,
    account: str,
    db_path: Path | None = None,
) -> str | None:
    """Return a non-empty error message if a partial new-shape migration
    for *account* already exists, else *None*.

    The caller refuses the migration when this returns a message.  The
    check covers Neo4j (Documents with the new prefix) and the queue DB
    (cursor key already promoted).
    """
    new_prefixes = [
        f"gmail://{account}/thread/",
        f"gmail://{account}/message/",
        f"google-calendar://{account}/event/",
        f"google-calendar://{account}/series/",
    ]

    where = " OR ".join(
        f"n.source_id STARTS WITH '{p}'" for p in new_prefixes
    )
    docs = _scalar(
        session,
        f"MATCH (n) WHERE n.source_id IS NOT NULL AND ({where}) "
        f"RETURN count(n)",
    )
    if docs > 0:
        return (
            f"Refused: {docs} Document(s) already have the new-shape prefix "
            f"for account={account!r}.  Interleaved migration detected — "
            f"manual review required."
        )

    if db_path is not None and db_path.exists():
        conn = sqlite3.connect(str(db_path))
        try:
            try:
                rows = conn.execute(
                    "SELECT key FROM cursors WHERE key IN (?, ?)",
                    (f"gmail:{account}", f"calendar:{account}"),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []
        finally:
            conn.close()
        if rows:
            keys = ", ".join(r[0] for r in rows)
            return (
                f"Refused: cursor key(s) {{{keys}}} already exist for "
                f"account={account!r}.  Interleaved migration detected — "
                f"manual review required."
            )

    return None


# ── Public orchestrator ───────────────────────────────────────────────


def run_migrate_gmail_multiaccount(
    *,
    config_path: Path | None = None,
    account: str | None = None,
    yes: bool = False,
    dry_run: bool = False,
    force_running: bool = False,
    daemon_detector: Callable[[], tuple[bool, str | None]] = detect_daemon,
    neo4j_session_factory: Callable[[], Any] | None = None,
    qdrant_factory: Callable[[], Any] | None = None,
    qdrant_collection: str | None = None,
    data_dir: Path | None = None,
    out: Any = None,
) -> int:
    """Run the migrate command.  Returns an exit code (0 = success)."""
    out = out or sys.stdout

    # 1. Daemon precondition.
    is_running, hint = daemon_detector()
    if is_running:
        if not force_running:
            print(
                f"fieldnotes daemon is running ({hint}). "
                f"Stop it first: `fieldnotes service stop`. "
                f"Re-run migrate after.",
                file=sys.stderr,
            )
            return 2
        logger.warning(
            "DAEMON STILL RUNNING (%s) — proceeding anyway because "
            "--force-running was passed.  queue.db writes may race.",
            hint,
        )

    # 2. Resolve the account label.
    try:
        account_label = resolve_account(account, yes=yes)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # 3. Resolve paths.  Read raw TOML rather than ``load_config`` —
    #    a flat-shape config is the very thing this migration fixes,
    #    so ``load_config``'s strict validation would refuse to load.
    if data_dir is None:
        data_dir = _raw_data_dir(config_path) or (
            Path.home() / ".fieldnotes" / "data"
        )
    data_dir = Path(data_dir).expanduser()
    queue_db = data_dir / "queue.db"

    config_file = (
        config_path.expanduser()
        if config_path is not None
        else Path.home() / ".fieldnotes" / "config.toml"
    )

    if qdrant_collection is None:
        qdrant_collection = _raw_qdrant_collection(config_path) or "fieldnotes"

    # 4. Detect old-shape state.
    queue_rows, cursor_rows = detect_queue_counts(queue_db)

    docs = chunks = persons = 0
    qdrant_points = 0
    if neo4j_session_factory is not None:
        with neo4j_session_factory() as session:
            docs, chunks, persons = detect_neo4j_counts(session)
    qdrant = qdrant_factory() if qdrant_factory is not None else None
    if qdrant is not None:
        qdrant_points = detect_qdrant_count(qdrant, qdrant_collection)

    # Print summary.
    print(f"Migration target: account={account_label!r}", file=out)
    print(f"  queue.db rows:       {queue_rows}", file=out)
    print(f"  queue.db cursors:    {cursor_rows}", file=out)
    print(f"  Neo4j Documents:     {docs}", file=out)
    print(f"  Neo4j Chunks:        {chunks}", file=out)
    print(f"  Fallback Persons:    {persons}", file=out)
    print(f"  Qdrant points:       {qdrant_points}", file=out)

    if dry_run:
        print("Dry run — no changes made.", file=out)
        return 0

    # 5. Interleaved-migration check.
    if neo4j_session_factory is not None:
        with neo4j_session_factory() as session:
            err = detect_interleaved(session, account_label, db_path=queue_db)
        if err is not None:
            print(err, file=sys.stderr)
            return 3

    # 6. Confirmation.
    if not yes:
        try:
            resp = input(
                f"Proceed with migration to account={account_label!r}? [y/N]: "
            ).strip().lower()
        except EOFError:
            resp = ""
        if resp not in ("y", "yes"):
            print("Aborted.", file=out)
            return 1

    counts = MigrationCounts()

    # 7a. Queue first (so a daemon started during the brief migrate
    #     window cannot pick up old-shape work).
    q = migrate_queue(queue_db, account_label)
    counts.queue_rows = q["rows"]
    counts.cursor_rows = q["cursors"]

    # 7b. Neo4j.
    if neo4j_session_factory is not None:
        with neo4j_session_factory() as session:
            n = migrate_neo4j(session, account_label)
        counts.documents = n["documents"]
        counts.chunks = n["chunks"]
        counts.fallback_persons = n["persons"]

    # 7c. Qdrant.
    if qdrant is not None:
        qd = migrate_qdrant(qdrant, qdrant_collection, account_label)
        counts.qdrant_points = qd["points"]

    # 7d. Files.
    f = migrate_files(data_dir, account_label)
    counts.files_renamed = f["renamed"]
    counts.files_deleted = f["deleted"]

    # 7e. Config rewrite.
    counts.config_rewritten = migrate_config(config_file, account_label)

    # 8. Final summary.
    print(
        f"Retagged {counts.queue_rows} queue rows, "
        f"{counts.documents} Documents, {counts.qdrant_points} Qdrant points, "
        f"{counts.fallback_persons} fallback Person nodes, "
        f"{counts.cursor_rows} cursor entries under "
        f"account={account_label!r}.",
        file=out,
    )
    if counts.files_renamed:
        print(
            f"Renamed token file(s): {', '.join(counts.files_renamed)}",
            file=out,
        )
    if counts.files_deleted:
        print(
            f"Deleted deprecated cursor file(s): "
            f"{', '.join(counts.files_deleted)}",
            file=out,
        )
    if counts.config_rewritten:
        print(
            f"Rewrote {config_file.name} (legacy section wrapped under "
            f"[sources.gmail.{account_label}] / "
            f"[sources.google_calendar.{account_label}]).  Original backup "
            f"saved alongside.  Restart the daemon.",
            file=out,
        )
    else:
        print(
            f"Update your config to [sources.gmail.{account_label}] and "
            f"[sources.google_calendar.{account_label}], then restart the "
            f"daemon.",
            file=out,
        )
    return 0


# ── Internals ─────────────────────────────────────────────────────────


def _scalar(session: Any, query: str, **params: Any) -> int:
    """Run *query* and return the first scalar of the first row, or 0."""
    result = session.run(query, **params)
    record = _first_record(result)
    if record is None:
        return 0
    # Neo4j Record supports indexing and .values().
    try:
        value = record[0]
    except (TypeError, KeyError, IndexError):
        try:
            value = list(record.values())[0]
        except Exception:
            return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _first_record(result: Any) -> Any:
    """Return the first record from a result, robust to Neo4j and mocks."""
    if hasattr(result, "single"):
        try:
            rec = result.single()
            if rec is not None:
                return rec
        except Exception:
            pass
    try:
        return next(iter(result))
    except StopIteration:
        return None
    except TypeError:
        return None


def _consume_count(result: Any) -> int:
    """Pull the ``count(...) AS c`` scalar from a SET-returning result."""
    record = _first_record(result)
    if record is None:
        return 0
    try:
        return int(record["c"])
    except (KeyError, TypeError, ValueError):
        try:
            return int(record[0])
        except Exception:
            return 0


def _raw_data_dir(config_path: Path | None) -> Path | None:
    """Pull ``[core] data_dir`` out of a config.toml without invoking
    ``load_config`` (which would reject a flat multi-account shape — the
    very thing this migration fixes).
    """
    path = (
        config_path.expanduser()
        if config_path is not None
        else Path.home() / ".fieldnotes" / "config.toml"
    )
    if not path.exists():
        return None
    import tomllib

    try:
        raw = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return None
    val = raw.get("core", {}).get("data_dir")
    if isinstance(val, str):
        return Path(val).expanduser()
    return None


def _raw_qdrant_collection(config_path: Path | None) -> str | None:
    """Pull ``[qdrant] collection`` out of a config.toml without
    triggering ``MigrationRequiredError``.
    """
    path = (
        config_path.expanduser()
        if config_path is not None
        else Path.home() / ".fieldnotes" / "config.toml"
    )
    if not path.exists():
        return None
    import tomllib

    try:
        raw = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return None
    val = raw.get("qdrant", {}).get("collection")
    return val if isinstance(val, str) else None


def _is_table_like(value: Any) -> bool:
    """Return True if *value* looks like a TOML table (has dict-style
    access).  Used to distinguish ``[sources.gmail]`` (flat scalars)
    from ``[sources.gmail.<account>]`` (each child is a sub-table).
    """
    return hasattr(value, "items") and not isinstance(value, (str, bytes, list))


def _iter_old_shape_points(
    qdrant: Any, collection: str, *, batch: int = 256
):
    """Iterate Qdrant points whose payload.source_id has an old-shape
    prefix.  Tolerates qdrant-client's two-shape return from ``scroll``.
    """
    next_offset: Any = None
    while True:
        result = qdrant.scroll(
            collection_name=collection,
            limit=batch,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )
        # qdrant_client.scroll returns (points, next_offset) tuple.
        if isinstance(result, tuple) and len(result) == 2:
            points, next_offset = result
        else:
            points = list(result)
            next_offset = None

        if not points:
            return

        for point in points:
            payload = getattr(point, "payload", None) or {}
            sid = payload.get("source_id") if isinstance(payload, dict) else None
            if not isinstance(sid, str):
                continue
            if any(sid.startswith(p) for p in _OLD_PREFIXES):
                yield point

        if next_offset is None:
            return
