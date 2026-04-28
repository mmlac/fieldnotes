"""CLI handler for ``fieldnotes itinerary`` — daily agenda renderer.

Wires :mod:`worker.query.itinerary` (fn-wbc.1) into a Rich-rendered terminal
view and a stable ``--json`` schema.  Default-on per-meeting LLM briefs
go through :mod:`worker.query.itinerary_brief` and
:mod:`worker.cli.itinerary_brief_prompt` (fn-wbc.4); ``--brief`` opts out.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any

from neo4j import Driver, GraphDatabase

from worker.cli.itinerary_brief_prompt import build_event_brief_request
from worker.config import Config, Neo4jConfig, load_config
from worker.query.itinerary import (
    EventWithLinks,
    Itinerary,
    NoteHit,
    OpenTask,
    PersonRef,
    ThreadHit,
    _local_tz,
    _resolve_day,
    get_itinerary,
)
from worker.query.itinerary_brief import assemble_event_brief

logger = logging.getLogger(__name__)


_DEFAULT_DAY = "today"
_DEFAULT_HORIZON = "30d"
_MAX_TASKS = 2
_MAX_NOTES = 2
_MAX_ATTENDEES_INLINE = 3
_BRIEF_ROLE = "completion"

_HORIZON_RE = re.compile(r"^(\d+)\s*(h|d|w|m)$", re.IGNORECASE)


class BriefError(RuntimeError):
    """Raised when the per-meeting brief generator can't run."""


def _open_driver(neo4j_cfg: Neo4jConfig) -> Driver:
    return GraphDatabase.driver(
        neo4j_cfg.uri, auth=(neo4j_cfg.user, neo4j_cfg.password)
    )


def _parse_horizon(s: str) -> timedelta:
    """Parse a relative horizon string like ``30d`` / ``24h`` / ``2w``."""
    text = (s or "").strip().lower()
    m = _HORIZON_RE.match(text)
    if not m:
        raise ValueError(
            f"horizon must be a relative value like '30d', '24h', '2w' (got {s!r})"
        )
    n = int(m.group(1))
    unit = m.group(2)
    if unit == "h":
        return timedelta(hours=n)
    if unit == "d":
        return timedelta(days=n)
    if unit == "w":
        return timedelta(weeks=n)
    return timedelta(days=n * 30)  # 'm'


def _build_registry(cfg: Config) -> Any | None:
    """Best-effort embedding registry; returns ``None`` if not configured."""
    try:
        from worker.models.resolver import ModelRegistry
        import worker.models.providers.ollama  # noqa: F401

        reg = ModelRegistry(cfg)
        # Probe for the embed role; absent role → skip notes silently.
        try:
            reg.for_role("embed")
        except Exception:  # noqa: BLE001
            return None
        return reg
    except Exception:  # noqa: BLE001 - any failure means "skip notes"
        logger.debug("itinerary: failed to build embed registry", exc_info=True)
        return None


def _resolve_completion(cfg: Config, registry: Any | None) -> Any:
    """Resolve the ``completion`` role, fail-fast on missing config.

    *registry* may be a pre-built ``ModelRegistry`` (or test stub).  If
    ``None``, a fresh ``ModelRegistry`` is constructed from *cfg*.
    Failure to construct or to resolve the role surfaces as
    :class:`BriefError`.
    """
    reg = registry
    if reg is None:
        try:
            from worker.models.resolver import ModelRegistry
            import worker.models.providers.ollama  # noqa: F401

            reg = ModelRegistry(cfg)
        except Exception as exc:  # noqa: BLE001
            raise BriefError(
                f"itinerary brief requires the {_BRIEF_ROLE!r} role in "
                f"[models.roles]; run 'fieldnotes doctor' to verify model "
                f"configuration ({exc})"
            ) from exc
    try:
        return reg.for_role(_BRIEF_ROLE)
    except KeyError as exc:
        raise BriefError(
            f"itinerary brief requires the {_BRIEF_ROLE!r} role in "
            f"[models.roles]; run 'fieldnotes doctor' to verify model "
            f"configuration ({exc})"
        ) from exc


def generate_event_briefs(
    itinerary: Itinerary,
    *,
    driver: Driver,
    completion_model: Any,
) -> dict[int, str]:
    """Generate one LLM brief per event, sequentially.

    Returns a mapping ``event_id → brief text``.  Per-meeting fidelity is
    a hard requirement (no batching).
    """
    out: dict[int, str] = {}
    for ew in itinerary.events:
        prebrief = assemble_event_brief(ew, driver=driver)
        request = build_event_brief_request(prebrief)
        response = completion_model.complete(request, task="itinerary_brief")
        out[ew.event.id] = response.text.strip()
    return out


# ---------------------------------------------------------------------------
# Thread enrichment: fetch last_from for an email/slack ThreadHit
# ---------------------------------------------------------------------------


@dataclass
class _ThreadDetail:
    last_from: str | None = None


def _fetch_thread_detail(driver: Driver, hit: ThreadHit) -> _ThreadDetail:
    """Look up the sender of the most-recent email or slack message."""
    if hit.kind == "email":
        cypher = """
        MATCH (em:Email)-[:PART_OF]->(t:Thread {source_id: $sid})
        OPTIONAL MATCH (sender:Person)-[:SENT]->(em)
        WITH em, sender ORDER BY em.date DESC LIMIT 1
        RETURN sender.email AS email, sender.name AS name
        """
    else:  # slack
        cypher = """
        MATCH (s:SlackMessage {source_id: $sid})
        OPTIONAL MATCH (s)-[:SENT_BY]->(p:Person)
        RETURN p.email AS email, p.name AS name LIMIT 1
        """
    try:
        with driver.session() as session:
            row = session.execute_read(
                lambda tx: tx.run(cypher, sid=hit.source_id).single()
            )
    except Exception:  # noqa: BLE001
        return _ThreadDetail()
    if row is None:
        return _ThreadDetail()
    return _ThreadDetail(last_from=row.get("email") or row.get("name"))


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt_local_time(ts: str, tz: tzinfo) -> str:
    """Render a stored timestamp as ``HH:MM`` in *tz*; all-day → ``all-day``."""
    if not ts:
        return "??:??"
    if "T" not in ts:
        return "all-day"
    iso = ts.replace("Z", "+00:00") if ts.endswith("Z") else ts
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        return ts[:5]
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(tz).strftime("%H:%M")


def _fmt_time_range(ev: EventWithLinks, tz: tzinfo) -> str:
    start = _fmt_local_time(ev.event.start_ts, tz)
    end = _fmt_local_time(ev.event.end_ts, tz)
    if start == end == "all-day":
        return "all-day"
    return f"{start}–{end}"


def _fmt_attendee(ref: PersonRef) -> str:
    if ref.name and ref.email:
        return f"{ref.name} <{ref.email}>"
    return ref.name or ref.email or "(unknown)"


def _fmt_attendee_list(attendees: list[PersonRef]) -> str:
    if not attendees:
        return "-"
    if len(attendees) <= _MAX_ATTENDEES_INLINE:
        return ", ".join(_fmt_attendee(a) for a in attendees)
    head = ", ".join(_fmt_attendee(a) for a in attendees[:_MAX_ATTENDEES_INLINE])
    extra = len(attendees) - _MAX_ATTENDEES_INLINE
    return f"{head}, +{extra} others"


def _fmt_mtime(ts: str | None) -> str:
    if not ts:
        return "?"
    return ts.split("T")[0] if "T" in ts else ts[:10]


def _fmt_thread_ts(ts: str) -> str:
    """Slack stores Unix seconds; emails store ISO."""
    if not ts:
        return "?"
    if "T" in ts:
        return ts.split("T")[0]
    try:
        epoch = float(ts)
        return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d")
    except ValueError:
        return ts


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


def _person_json(ref: PersonRef | None) -> dict[str, Any] | None:
    if ref is None:
        return None
    return {"name": ref.name, "email": ref.email}


def _task_json(t: OpenTask) -> dict[str, Any]:
    return {
        "title": t.title,
        "project": t.project,
        "tags": list(t.tags),
        "due": t.due,
        "defer": t.defer,
        "flagged": t.flagged,
        "source_id": t.source_id,
    }


def _note_json(n: NoteHit) -> dict[str, Any]:
    return {
        "source_id": n.source_id,
        "title": n.title,
        "mtime": n.mtime,
        "attendee_overlap": n.attendee_overlap,
    }


def _thread_json(hit: ThreadHit | None, last_from: str | None) -> dict[str, Any] | None:
    if hit is None:
        return None
    return {
        "kind": hit.kind,
        "source_id": hit.source_id,
        "title": hit.title,
        "last_ts": hit.last_ts,
        "last_from": last_from,
    }


def _emit_json(
    itinerary: Itinerary,
    thread_details: dict[int, _ThreadDetail],
    briefs: dict[int, str],
) -> str:
    payload = {
        "day": itinerary.day.strftime("%Y-%m-%d"),
        "timezone": itinerary.timezone,
        "events": [
            {
                "event_id": str(ew.event.id),
                "source_id": ew.event.source_id,
                "title": ew.event.title,
                "start": ew.event.start_ts,
                "end": ew.event.end_ts,
                "account": ew.event.account,
                "calendar_id": ew.event.calendar_id,
                "organizer": _person_json(ew.event.organizer),
                "attendees": [
                    {"name": a.name, "email": a.email} for a in ew.event.attendees
                ],
                "location": ew.event.location,
                "html_link": ew.event.html_link,
                "linked": {
                    "tasks": [_task_json(t) for t in ew.tasks[:_MAX_TASKS]],
                    "notes": [_note_json(n) for n in ew.notes[:_MAX_NOTES]],
                    "thread": _thread_json(
                        ew.thread,
                        thread_details.get(ew.event.id, _ThreadDetail()).last_from,
                    ),
                },
                "next_brief": briefs.get(ew.event.id),
            }
            for ew in itinerary.events
        ],
    }
    return json.dumps(payload, indent=2, default=str)


# ---------------------------------------------------------------------------
# Rich output
# ---------------------------------------------------------------------------


def _emit_rich(
    itinerary: Itinerary,
    thread_details: dict[int, _ThreadDetail],
    briefs: dict[int, str],
    tz: tzinfo,
) -> None:
    from rich.console import Console

    console = Console()

    if not itinerary.events:
        weekday = itinerary.day.strftime("%a")
        console.print(f"[bold]Itinerary - {weekday} {itinerary.day.isoformat()}[/bold]")
        console.print("No events scheduled.")
        return

    accounts: set[str] = set()
    for ew in itinerary.events:
        if ew.event.account:
            accounts.add(ew.event.account)
    n_events = len(itinerary.events)
    n_calendars = len(accounts)
    weekday = itinerary.day.strftime("%a")
    console.print(
        f"[bold]Itinerary - {weekday} {itinerary.day.isoformat()} "
        f"({n_events} events, {n_calendars} calendars)[/bold]"
    )

    for ew in itinerary.events:
        console.print()
        time_range = _fmt_time_range(ew, tz)
        title = ew.event.title or "(untitled)"
        account = ew.event.account or "-"
        console.print(f"[bold]{time_range}[/bold]  {title}  [dim]\\[{account}][/dim]")
        console.print(f"  Attendees: {_fmt_attendee_list(ew.event.attendees)}")

        # Per-meeting LLM brief (fn-wbc.4): one line, '▸ ' marker.
        brief_text = briefs.get(ew.event.id)
        if brief_text:
            console.print(f"  ▸ {brief_text}")

        # Tasks
        tasks = ew.tasks[:_MAX_TASKS]
        if tasks:
            console.print("  Tasks:")
            for t in tasks:
                flag = "[yellow]![/yellow]" if t.flagged else " "
                due = f" — due {_fmt_mtime(t.due)}" if t.due else ""
                proj = f" ({t.project})" if t.project else ""
                console.print(f"    {flag} {t.title}{proj}{due}")
        else:
            console.print("  Tasks: -")

        # Notes
        notes = ew.notes[:_MAX_NOTES]
        if notes:
            console.print("  Notes:")
            for n in notes:
                console.print(f"    - {n.source_id} (mtime: {_fmt_mtime(n.mtime)})")
        else:
            console.print("  Notes: -")

        # Threads (max 1)
        if ew.thread is not None:
            detail = thread_details.get(ew.event.id, _ThreadDetail())
            from_part = f" — last_from: {detail.last_from}" if detail.last_from else ""
            ts = _fmt_thread_ts(ew.thread.last_ts)
            console.print(
                f"  Thread: \\[{ew.thread.kind}] {ew.thread.title}{from_part} — {ts}"
            )
        else:
            console.print("  Thread: -")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _list_accounts(cfg: Config) -> list[str]:
    return sorted(cfg.google_calendar.keys())


def run_itinerary(
    *,
    day: str = _DEFAULT_DAY,
    account: str | None = None,
    brief: bool = False,
    horizon: str = _DEFAULT_HORIZON,
    json_output: bool = False,
    config_path: Path | None = None,
    registry: Any | None = None,
) -> int:
    """Render the daily itinerary.  Returns process exit code.

    Default-on per-meeting LLM brief: when *brief* is ``False`` (the
    default), one ``completion``-role call per event populates
    ``next_brief``.  When *brief* is ``True``, no LLM calls are made and
    ``next_brief`` stays ``None`` on every event.
    """
    cfg = load_config(config_path)
    tz = _local_tz(None)

    # Validate --day before opening any driver.
    try:
        target_day = _resolve_day(day, tz)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Validate --account against configured accounts (when configured).
    if account is not None:
        valid = _list_accounts(cfg)
        if valid and account not in valid:
            print(
                f"error: unknown account {account!r}. "
                f"Configured accounts: {', '.join(valid) if valid else '(none)'}",
                file=sys.stderr,
            )
            return 2

    # Validate --horizon.
    try:
        horizon_td = _parse_horizon(horizon)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    reg = registry if registry is not None else _build_registry(cfg)

    # Resolve completion role early (fail-fast, before any LLM call) when
    # the per-meeting brief is enabled.
    completion_model: Any = None
    if not brief:
        try:
            completion_model = _resolve_completion(cfg, reg)
        except BriefError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    itinerary = get_itinerary(
        target_day,
        account=account,
        timezone_=tz,
        horizon=horizon_td,
        registry=reg,
        qdrant_cfg=cfg.qdrant,
        neo4j_cfg=cfg.neo4j,
    )

    thread_details: dict[int, _ThreadDetail] = {}
    briefs: dict[int, str] = {}
    needs_driver = any(ew.thread is not None for ew in itinerary.events) or (
        completion_model is not None and itinerary.events
    )
    if needs_driver:
        drv = _open_driver(cfg.neo4j)
        try:
            for ew in itinerary.events:
                if ew.thread is not None:
                    thread_details[ew.event.id] = _fetch_thread_detail(drv, ew.thread)
            if completion_model is not None:
                briefs = generate_event_briefs(
                    itinerary, driver=drv, completion_model=completion_model
                )
        finally:
            drv.close()

    if json_output:
        print(_emit_json(itinerary, thread_details, briefs))
    else:
        _emit_rich(itinerary, thread_details, briefs, tz)
    return 0
