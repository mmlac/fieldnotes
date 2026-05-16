"""Itinerary queries: read-only Cypher + vector search for the daily agenda.

Read-only queries that aggregate calendar events with linked OmniFocus
tasks, vector-similar notes, and recent email/Slack conversations into
an Itinerary view.  CLI/MCP/LLM landing on top of this module are
separate beads (fn-wbc.2/3/4).

All Cypher templates are read-only — zero ``MERGE``/``CREATE``/``DELETE``
clauses anywhere.  The module mirrors the conventions in
:mod:`worker.query.person`: dataclass results, optional ``driver`` /
``neo4j_cfg`` injection, and SAME_AS cluster expansion when matching
attendees to mentions across sources.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone, tzinfo
from typing import Any

from neo4j import Driver

from worker.config import Neo4jConfig, QdrantConfig
from worker.neo4j_driver import build_driver

logger = logging.getLogger(__name__)


_DEFAULT_EVENT_LIMIT = 100
_DEFAULT_K_TASKS = 2
_DEFAULT_K_NOTES = 2
_DEFAULT_HORIZON = timedelta(days=30)
_NOTES_PREFETCH_MULTIPLIER = 5
_NOTES_PREFETCH_MIN = 10


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PersonRef:
    """Reference to a Person node — used for organizers and attendees."""

    id: str
    email: str | None = None
    name: str | None = None
    is_self: bool = False


@dataclass
class Event:
    """One row from :func:`events_for_day`."""

    id: str
    source_id: str
    title: str
    description: str | None
    start_ts: str
    end_ts: str
    location: str | None
    account: str | None
    calendar_id: str | None
    html_link: str | None
    organizer: PersonRef | None
    attendees: list[PersonRef] = field(default_factory=list)
    is_self_only: bool = False


@dataclass
class OpenTask:
    """One row from :func:`linked_tasks_for_event`.

    Mirrors the shape used by :class:`worker.query.person.OpenTask` so
    downstream renderers can treat the two interchangeably.
    """

    title: str
    project: str | None = None
    tags: list[str] = field(default_factory=list)
    due: str | None = None
    defer: str | None = None
    flagged: bool = False
    source_id: str | None = None


@dataclass
class NoteHit:
    """One row from :func:`linked_notes_for_event`."""

    source_id: str
    title: str
    snippet: str
    mtime: str | None
    attendee_overlap: bool
    score: float = 0.0


@dataclass
class ThreadHit:
    """Result of :func:`recent_thread_with_attendees`."""

    kind: str  # "email" | "slack"
    source_id: str
    title: str
    last_ts: str
    participants: list[str] = field(default_factory=list)


@dataclass
class EventWithLinks:
    """One element of :class:`Itinerary`.events."""

    event: Event
    tasks: list[OpenTask] = field(default_factory=list)
    notes: list[NoteHit] = field(default_factory=list)
    thread: ThreadHit | None = None


@dataclass
class Itinerary:
    """Aggregated daily agenda returned by :func:`get_itinerary`."""

    day: date
    timezone: str
    events: list[EventWithLinks] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Driver helpers
# ---------------------------------------------------------------------------


def _open_driver(neo4j_cfg: Neo4jConfig | None) -> Driver:
    cfg = neo4j_cfg or Neo4jConfig()
    return build_driver(cfg.uri, cfg.user, cfg.password)


def _local_tz(tz: tzinfo | None) -> tzinfo:
    """Resolve *tz* to the user's local timezone if not provided."""
    if tz is not None:
        return tz
    local = datetime.now().astimezone().tzinfo
    return local or timezone.utc


def _resolve_day(value: date | str, tz: tzinfo) -> date:
    """Resolve *value* to a :class:`datetime.date`.

    Accepts an existing ``date`` (returned unchanged), the strings
    ``"today"`` and ``"tomorrow"`` (relative to *tz*), or an ISO date
    string ``YYYY-MM-DD``.  Anything else raises ``ValueError``.
    """
    if isinstance(value, datetime):
        return value.astimezone(tz).date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        raise ValueError("day must be 'today', 'tomorrow', or YYYY-MM-DD")
    lower = text.lower()
    if lower == "today":
        return datetime.now(tz=tz).date()
    if lower == "tomorrow":
        return (datetime.now(tz=tz) + timedelta(days=1)).date()
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(
            f"day must be 'today', 'tomorrow', or YYYY-MM-DD (got {value!r})"
        ) from exc


def _epoch(ts: str) -> float | None:
    """Best-effort epoch-seconds parse for the timestamp shapes parsers
    write: RFC3339 datetimes (``2026-04-27T12:00:00Z`` or with offset),
    plain dates (``2026-04-27`` — treated as UTC midnight), and Slack
    Unix-seconds (``"1722956123.456"``).
    """
    if not ts:
        return None
    if "T" not in ts and "-" in ts:
        try:
            d = datetime.strptime(ts, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            return d.timestamp()
        except ValueError:
            pass
    if ts[0].isdigit() and "T" not in ts and "-" not in ts[1:]:
        try:
            return float(ts)
        except ValueError:
            return None
    iso = ts.replace("Z", "+00:00") if ts.endswith("Z") else ts
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


# ---------------------------------------------------------------------------
# events_for_day
# ---------------------------------------------------------------------------


# Lex bounds widen the day window by ±1 calendar day so:
#   - All-day events stored as plain "YYYY-MM-DD" still sort into range.
#   - Timed events whose stored offset differs from the user's TZ aren't
#     missed at the day boundary (e.g. a 23:00 PST event stored as
#     07:00Z next-day UTC).
# Python then filters precisely against the local-day [start, end) window.
_EVENTS_FOR_DAY_CYPHER = """\
MATCH (e:CalendarEvent)
WHERE e.start_time >= $lo
  AND e.start_time < $hi
  AND ($account IS NULL OR e.account = $account)
OPTIONAL MATCH (e)-[:ORGANIZED_BY]->(org:Person)
OPTIONAL MATCH (e)-[:ATTENDED_BY]->(att:Person)
WITH e,
     org,
     collect(DISTINCT {
       id: att.source_id,
       email: att.email,
       name: att.name,
       is_self: coalesce(att.is_self, false)
     }) AS attendees
RETURN e.source_id AS id,
       e.source_id AS source_id,
       e.summary AS title,
       e.description AS description,
       e.start_time AS start_ts,
       e.end_time AS end_ts,
       e.location AS location,
       e.account AS account,
       e.calendar_id AS calendar_id,
       e.html_link AS html_link,
       CASE WHEN org IS NULL THEN NULL ELSE {
         id: org.source_id,
         email: org.email,
         name: org.name,
         is_self: coalesce(org.is_self, false)
       } END AS organizer,
       attendees
ORDER BY e.start_time ASC, e.source_id ASC
LIMIT $limit
"""


def _person_ref(row: dict[str, Any] | None) -> PersonRef | None:
    if row is None or row.get("id") is None:
        return None
    return PersonRef(
        id=str(row["id"]),
        email=row.get("email"),
        name=row.get("name"),
        is_self=bool(row.get("is_self") or False),
    )


def _local_window_iso(day: date, tz: tzinfo) -> tuple[str, str]:
    """Return the [start, end) window of *day* in *tz* as RFC3339 UTC strings."""
    start_local = datetime.combine(day, datetime.min.time(), tzinfo=tz)
    end_local = start_local + timedelta(days=1)
    start_utc = start_local.astimezone(timezone.utc)
    end_utc = end_local.astimezone(timezone.utc)
    return (
        start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        end_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def _row_starts_in_window(start_ts: str, end_ts: str, day: date, tz: tzinfo) -> bool:
    """Decide whether a stored event falls inside the local-day window.

    Handles three shapes:
      * RFC3339 timed event — parse and compare in *tz*.
      * All-day event stored as ``YYYY-MM-DD`` — match by date directly.
      * Anything unparseable — fall back to ``False`` (drop quietly).
    """
    if not start_ts:
        return False
    # All-day "YYYY-MM-DD" form: Google stores these as start=day,
    # end=day+1.  We just match the date prefix to the requested day.
    if "T" not in start_ts and len(start_ts) >= 10:
        try:
            ev_day = datetime.strptime(start_ts[:10], "%Y-%m-%d").date()
            return ev_day == day
        except ValueError:
            return False
    epoch_start = _epoch(start_ts)
    if epoch_start is None:
        return False
    start_local = datetime.combine(day, datetime.min.time(), tzinfo=tz)
    end_local = start_local + timedelta(days=1)
    return start_local.timestamp() <= epoch_start < end_local.timestamp()


def events_for_day(
    day: date | str = "today",
    account: str | None = None,
    timezone_: tzinfo | None = None,
    *,
    limit: int = _DEFAULT_EVENT_LIMIT,
    neo4j_cfg: Neo4jConfig | None = None,
    driver: Driver | None = None,
) -> list[Event]:
    """Return CalendarEvent rows whose start lands on *day* in *timezone_*.

    *day* may be a :class:`datetime.date`, ``"today"``/``"tomorrow"``, or an
    ISO ``YYYY-MM-DD`` string (see :func:`_resolve_day`).  *account*, when
    set, filters to a single ``google_calendar.<account>`` source.
    Returns ``[]`` (not ``None``) on empty days.

    The Cypher template uses a wide ±1-day lex range so all-day strings
    and timed events with foreign offsets are caught; Python then filters
    each row precisely against the local-day window.  This keeps the
    query bounded (single MATCH + LIMIT) while staying TZ-correct.
    """
    tz = _local_tz(timezone_)
    target_day = _resolve_day(day, tz)
    lo_iso, hi_iso = _local_window_iso(target_day, tz)
    # Widen the lex bounds by 1 day on each side; precise filtering happens below.
    lo_lex = (target_day - timedelta(days=1)).strftime("%Y-%m-%d")
    hi_lex = (target_day + timedelta(days=2)).strftime("%Y-%m-%d")

    own_driver = driver is None
    drv = driver or _open_driver(neo4j_cfg)
    try:
        with drv.session() as session:
            rows = session.execute_read(
                lambda tx: tx.run(
                    _EVENTS_FOR_DAY_CYPHER,
                    lo=lo_lex,
                    hi=hi_lex,
                    account=account,
                    limit=int(limit),
                ).data()
            )
    finally:
        if own_driver:
            drv.close()

    out: list[Event] = []
    for r in rows:
        start_ts = str(r.get("start_ts") or "")
        end_ts = str(r.get("end_ts") or "")
        if not _row_starts_in_window(start_ts, end_ts, target_day, tz):
            continue
        organizer = _person_ref(r.get("organizer"))
        attendees_raw = r.get("attendees") or []
        attendees: list[PersonRef] = []
        for a in attendees_raw:
            ref = _person_ref(a)
            if ref is None:
                continue
            if ref.is_self:
                continue
            attendees.append(ref)
        # is_self_only: organizer is self (or absent) AND no non-self attendees.
        organizer_is_self = bool(organizer and organizer.is_self)
        is_self_only = organizer_is_self and not attendees
        out.append(
            Event(
                id=str(r.get("id") or ""),
                source_id=str(r.get("source_id") or ""),
                title=str(r.get("title") or ""),
                description=r.get("description"),
                start_ts=start_ts,
                end_ts=end_ts,
                location=r.get("location"),
                account=r.get("account"),
                calendar_id=r.get("calendar_id"),
                html_link=r.get("html_link"),
                organizer=organizer,
                attendees=attendees,
                is_self_only=is_self_only,
            )
        )

    # The Cypher ORDER BY is by raw string; re-sort in epoch space so
    # all-day events ("YYYY-MM-DD") and timed events compare correctly.
    out.sort(key=lambda e: (_epoch(e.start_ts) or float("inf"), e.source_id))
    # Keep the silently-narrow argument honest: the lo/hi pre-filter is
    # 'wide' lex, so the docstring's "default limit applies after the
    # local-day filter" matches behaviour.
    return out[: max(0, int(limit))]


# ---------------------------------------------------------------------------
# Cluster helpers (mirrors worker.query.person._cluster_ids on demand)
# ---------------------------------------------------------------------------


def _cluster_ids(tx: Any, person_id: str) -> list[str]:
    """All source_ids in *person_id*'s SAME_AS cluster (including itself)."""
    rows = tx.run(
        """
        MATCH (seed:Person) WHERE seed.source_id = $pid
        OPTIONAL MATCH (seed)-[:SAME_AS*0..]-(p:Person)
        WITH collect(DISTINCT p) + seed AS members
        UNWIND members AS m
        WITH DISTINCT m
        RETURN m.source_id AS id
        """,
        pid=person_id,
    ).data()
    return [str(r["id"]) for r in rows if r.get("id") is not None]


def _attendee_cluster_ids(tx: Any, event_id: str) -> list[list[str]]:
    """Return one SAME_AS cluster (list of Person source_ids) per non-self attendee."""
    rows = tx.run(
        """
        MATCH (e:CalendarEvent) WHERE e.source_id = $eid
        MATCH (e)-[:ATTENDED_BY]->(a:Person)
        WHERE NOT coalesce(a.is_self, false)
        RETURN a.source_id AS pid
        """,
        eid=event_id,
    ).data()
    out: list[list[str]] = []
    for row in rows:
        pid = row.get("pid")
        if not pid:
            continue
        cluster = _cluster_ids(tx, str(pid))
        if cluster:
            out.append(cluster)
    return out


# ---------------------------------------------------------------------------
# linked_tasks_for_event
# ---------------------------------------------------------------------------


_LINKED_TASKS_CYPHER = """\
MATCH (t:Task)-[:MENTIONS]->(p:Person)
WHERE p.source_id IN $cluster_union
  AND coalesce(t.status, 'active') <> 'completed'
  AND coalesce(t.status, 'active') <> 'dropped'
  AND ($since IS NULL OR coalesce(t.modification_date, t.creation_date, '') >= $since)
OPTIONAL MATCH (t)-[:IN_PROJECT]->(proj:Project)
OPTIONAL MATCH (t)-[:TAGGED]->(tag:Tag)
WITH DISTINCT t, proj, collect(DISTINCT tag.name) AS tags
RETURN t.name AS title,
       t.source_id AS source_id,
       proj.name AS project,
       tags,
       t.due_date AS due,
       t.defer_date AS defer,
       coalesce(t.flagged, false) AS flagged,
       t.source_id AS task_id
ORDER BY flagged DESC,
         CASE WHEN t.due_date IS NULL THEN 1 ELSE 0 END ASC,
         t.due_date ASC,
         t.source_id ASC
LIMIT $k
"""


def linked_tasks_for_event(
    event_id: str,
    *,
    k: int = _DEFAULT_K_TASKS,
    horizon: timedelta = _DEFAULT_HORIZON,
    neo4j_cfg: Neo4jConfig | None = None,
    driver: Driver | None = None,
) -> list[OpenTask]:
    """Return up to *k* open OmniFocus tasks tied to this event's attendees.

    Matches Tasks that ``MENTIONS`` any Person in any attendee's SAME_AS
    cluster.  Filters to status NOT IN ``('completed', 'dropped')`` and
    modification within *horizon*.  Ordered by ``flagged DESC, due ASC
    NULLS LAST``.
    """
    own_driver = driver is None
    drv = driver or _open_driver(neo4j_cfg)
    try:
        with drv.session() as session:
            clusters = session.execute_read(_attendee_cluster_ids, event_id)
            cluster_union: list[str] = sorted({pid for c in clusters for pid in c})
            if not cluster_union:
                return []
            since_iso = _horizon_iso(horizon)
            rows = session.execute_read(
                lambda tx: tx.run(
                    _LINKED_TASKS_CYPHER,
                    cluster_union=cluster_union,
                    since=since_iso,
                    k=int(k),
                ).data()
            )
    finally:
        if own_driver:
            drv.close()
    return [
        OpenTask(
            title=str(r.get("title") or ""),
            project=r.get("project"),
            tags=[str(t) for t in (r.get("tags") or []) if t],
            due=r.get("due"),
            defer=r.get("defer"),
            flagged=bool(r.get("flagged")),
            source_id=r.get("source_id"),
        )
        for r in rows
    ]


def _horizon_iso(horizon: timedelta) -> str | None:
    """Convert a horizon ``timedelta`` to an ISO cutoff (now − horizon, UTC)."""
    if horizon is None or horizon.total_seconds() <= 0:
        return None
    cutoff = datetime.now(timezone.utc) - horizon
    return cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# linked_notes_for_event
# ---------------------------------------------------------------------------


# Source-types persisted in Qdrant payload.source_type whose parent
# Neo4j nodes correspond to "File / Document / SlackMessage" in the spec.
_NOTE_SOURCE_TYPES = ("files", "obsidian", "slack")


def linked_notes_for_event(
    event_id: str,
    *,
    k: int = _DEFAULT_K_NOTES,
    horizon: timedelta = _DEFAULT_HORIZON,
    registry: Any = None,
    qdrant_cfg: QdrantConfig | None = None,
    qdrant_client: Any = None,
    neo4j_cfg: Neo4jConfig | None = None,
    driver: Driver | None = None,
) -> list[NoteHit]:
    """Vector-search File/Obsidian/Slack chunks similar to the event title.

    Embeds the event title with the ``embed`` role from *registry*, runs
    a Qdrant top-N similarity search, and re-ranks so chunks whose
    parent doc has a ``MENTIONS`` edge to any attendee come first.
    Returns at most *k* hits.

    *registry* must implement ``for_role("embed")`` returning an object
    with ``embed(EmbedRequest)``.  Tests inject a stub.  *qdrant_client*
    is provided as an injection point alongside *qdrant_cfg* — callers
    that already hold a client can reuse it.
    """
    if registry is None:
        raise ValueError(
            "linked_notes_for_event requires a model registry "
            "(registry.for_role('embed'))"
        )

    own_driver = driver is None
    drv = driver or _open_driver(neo4j_cfg)
    try:
        with drv.session() as session:
            event_row = session.execute_read(
                lambda tx: tx.run(
                    "MATCH (e:CalendarEvent) WHERE e.source_id = $eid "
                    "RETURN e.summary AS title",
                    eid=event_id,
                ).single()
            )
            if event_row is None:
                return []
            title = str(event_row.get("title") or "").strip()
            if not title:
                return []

            attendee_clusters = session.execute_read(_attendee_cluster_ids, event_id)
            attendee_pid_union: set[str] = {pid for c in attendee_clusters for pid in c}

        # Embed the title using the configured 'embed' role.  Lazy import
        # keeps this module load cleanly when callers don't use vector
        # search (e.g. itinerary CLI without notes).
        from worker.models.base import EmbedRequest

        model = registry.for_role("embed")
        emb = model.embed(EmbedRequest(texts=[title]))
        if not emb.vectors:
            return []
        query_vector = emb.vectors[0]

        prefetch = max(_NOTES_PREFETCH_MIN, int(k) * _NOTES_PREFETCH_MULTIPLIER)
        hits = _qdrant_search(
            query_vector=query_vector,
            top_k=prefetch,
            qdrant_cfg=qdrant_cfg,
            qdrant_client=qdrant_client,
        )

        cutoff_iso = _horizon_iso(horizon)
        candidates: list[tuple[str, str, str, float]] = []
        for h in hits:
            payload = getattr(h, "payload", None) or {}
            stype = str(payload.get("source_type") or "")
            if stype not in _NOTE_SOURCE_TYPES:
                continue
            sid = str(payload.get("source_id") or "")
            if not sid:
                continue
            d = str(payload.get("date") or "")
            if cutoff_iso and d and d < cutoff_iso:
                continue
            text = str(payload.get("text") or "")
            score = float(getattr(h, "score", 0.0) or 0.0)
            candidates.append((sid, d, text, score))

        if not candidates:
            return []

        # Look up parent doc title + mtime + attendee overlap in one pass.
        sids = list({c[0] for c in candidates})
        with drv.session() as session:
            parents = session.execute_read(
                lambda tx: _fetch_parents(tx, sids, attendee_pid_union)
            )
    finally:
        if own_driver:
            drv.close()

    # Dedupe by source_id keeping the best chunk score.
    by_sid: dict[str, NoteHit] = {}
    for sid, d, text, score in candidates:
        meta = parents.get(sid, {})
        snippet = text[:200].strip()
        hit = NoteHit(
            source_id=sid,
            title=str(meta.get("title") or sid),
            snippet=snippet,
            mtime=meta.get("mtime") or (d or None),
            attendee_overlap=bool(meta.get("attendee_overlap")),
            score=score,
        )
        existing = by_sid.get(sid)
        if existing is None or hit.score > existing.score:
            by_sid[sid] = hit

    ranked = sorted(
        by_sid.values(),
        key=lambda h: (0 if h.attendee_overlap else 1, -h.score),
    )
    return ranked[: max(0, int(k))]


def _qdrant_search(
    *,
    query_vector: list[float],
    top_k: int,
    qdrant_cfg: QdrantConfig | None,
    qdrant_client: Any,
) -> list[Any]:
    """Run a Qdrant similarity search.  Lazy-imports the client lib."""
    if qdrant_client is not None:
        client = qdrant_client
        own_client = False
        collection = (qdrant_cfg or QdrantConfig()).collection
    else:
        from qdrant_client import QdrantClient  # noqa: WPS433

        cfg = qdrant_cfg or QdrantConfig()
        client = QdrantClient(host=cfg.host, port=cfg.port)
        own_client = True
        collection = cfg.collection
    try:
        response = client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        return list(response.points)
    finally:
        if own_client:
            try:
                client.close()
            except Exception:  # pragma: no cover - best-effort cleanup
                logger.debug("Qdrant client close failed", exc_info=True)


def _fetch_parents(
    tx: Any, source_ids: list[str], attendee_pid_union: set[int]
) -> dict[str, dict[str, Any]]:
    """Look up parent doc metadata + attendee overlap by source_id."""
    if not source_ids:
        return {}
    rows = tx.run(
        """
        MATCH (n) WHERE n.source_id IN $sids
          AND (n:File OR n:Document OR n:SlackMessage)
        OPTIONAL MATCH (n)-[:MENTIONS]->(p:Person)
        WITH n,
             collect(DISTINCT p.source_id) AS mentioned_ids
        RETURN n.source_id AS source_id,
               coalesce(n.title, n.path, n.channel_name, n.source_id) AS title,
               coalesce(n.modified_at, n.updated, n.created, n.last_ts, n.first_ts) AS mtime,
               mentioned_ids
        """,
        sids=source_ids,
    ).data()
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        mentioned = set(str(x) for x in (r.get("mentioned_ids") or []) if x is not None)
        out[str(r["source_id"])] = {
            "title": r.get("title"),
            "mtime": r.get("mtime"),
            "attendee_overlap": bool(mentioned & attendee_pid_union),
        }
    return out


# ---------------------------------------------------------------------------
# recent_thread_with_attendees
# ---------------------------------------------------------------------------


def recent_thread_with_attendees(
    event_id: str,
    *,
    horizon: timedelta = _DEFAULT_HORIZON,
    neo4j_cfg: Neo4jConfig | None = None,
    driver: Driver | None = None,
) -> ThreadHit | None:
    """Return the most-recent email thread or Slack window with all attendees.

    Email: Threads where every attendee's SAME_AS cluster is touched by
    some Email in the thread (via ``SENT`` from the Person or ``TO``
    from the Email).  Slack: SlackMessages where every attendee is
    present via ``SENT_BY`` or ``MENTIONS``.  Returns whichever is most
    recent within *horizon*, or ``None`` if no candidate covers all
    attendees.
    """
    own_driver = driver is None
    drv = driver or _open_driver(neo4j_cfg)
    try:
        with drv.session() as session:
            attendee_clusters = session.execute_read(_attendee_cluster_ids, event_id)
            if not attendee_clusters:
                return None
            since_iso = _horizon_iso(horizon)
            email_hit = session.execute_read(
                _recent_email_thread_tx, attendee_clusters, since_iso
            )
            slack_hit = session.execute_read(
                _recent_slack_window_tx, attendee_clusters, since_iso
            )
    finally:
        if own_driver:
            drv.close()

    candidates = [c for c in (email_hit, slack_hit) if c is not None]
    if not candidates:
        return None
    candidates.sort(
        key=lambda h: _epoch(h.last_ts) or 0.0,
        reverse=True,
    )
    return candidates[0]


_EMAIL_THREAD_CANDIDATES_CYPHER = """\
MATCH (em:Email)-[:PART_OF]->(thread:Thread)
WHERE ($since IS NULL OR coalesce(em.date, '') >= $since)
OPTIONAL MATCH (em)-[:TO]->(p_to:Person)
OPTIONAL MATCH (p_sent:Person)-[:SENT]->(em)
WITH thread, em,
     [x IN collect(DISTINCT p_to.source_id) + collect(DISTINCT p_sent.source_id) WHERE x IS NOT NULL] AS em_persons
WITH thread,
     max(em.date) AS last_ts,
     reduce(acc = [], lst IN collect(em_persons) | acc + lst) AS person_ids
RETURN thread.source_id AS source_id,
       coalesce(thread.subject, thread.thread_id, thread.source_id) AS title,
       last_ts,
       person_ids
ORDER BY last_ts DESC
LIMIT 50
"""


def _recent_email_thread_tx(
    tx: Any, attendee_clusters: list[list[int]], since_iso: str | None
) -> ThreadHit | None:
    rows = tx.run(
        _EMAIL_THREAD_CANDIDATES_CYPHER,
        since=since_iso,
    ).data()
    cluster_sets = [set(c) for c in attendee_clusters]
    for r in rows:
        last_ts = str(r.get("last_ts") or "")
        if not last_ts:
            continue
        person_ids = set(str(x) for x in (r.get("person_ids") or []) if x is not None)
        if not all(person_ids & cluster for cluster in cluster_sets):
            continue
        return ThreadHit(
            kind="email",
            source_id=str(r.get("source_id") or ""),
            title=str(r.get("title") or ""),
            last_ts=last_ts,
        )
    return None


_SLACK_WINDOW_CANDIDATES_CYPHER = """\
MATCH (s:SlackMessage)
WHERE ($since IS NULL OR coalesce(s.last_ts, s.first_ts, '') >= $since)
OPTIONAL MATCH (s)-[:SENT_BY|MENTIONS]->(p:Person)
WITH s, [x IN collect(DISTINCT p.source_id) WHERE x IS NOT NULL] AS person_ids
RETURN s.source_id AS source_id,
       coalesce(s.channel_name, s.source_id) AS title,
       coalesce(s.last_ts, s.first_ts) AS last_ts,
       person_ids
ORDER BY coalesce(s.last_ts, s.first_ts) DESC
LIMIT 50
"""


def _recent_slack_window_tx(
    tx: Any, attendee_clusters: list[list[int]], since_iso: str | None
) -> ThreadHit | None:
    rows = tx.run(
        _SLACK_WINDOW_CANDIDATES_CYPHER,
        since=_slack_since(since_iso),
    ).data()
    cluster_sets = [set(c) for c in attendee_clusters]
    for r in rows:
        last_ts = r.get("last_ts")
        if last_ts is None:
            continue
        person_ids = set(str(x) for x in (r.get("person_ids") or []) if x is not None)
        if not all(person_ids & cluster for cluster in cluster_sets):
            continue
        return ThreadHit(
            kind="slack",
            source_id=str(r.get("source_id") or ""),
            title=str(r.get("title") or ""),
            last_ts=str(last_ts),
        )
    return None


def _slack_since(iso: str | None) -> str | None:
    """Slack stores ``last_ts`` as Unix seconds; convert ISO cutoff to that."""
    if iso is None:
        return None
    epoch = _epoch(iso)
    if epoch is None:
        return None
    return f"{epoch:.6f}"


# ---------------------------------------------------------------------------
# get_itinerary
# ---------------------------------------------------------------------------


def get_itinerary(
    day: date | str = "today",
    account: str | None = None,
    *,
    k_tasks: int = _DEFAULT_K_TASKS,
    k_notes: int = _DEFAULT_K_NOTES,
    horizon: timedelta = _DEFAULT_HORIZON,
    timezone_: tzinfo | None = None,
    registry: Any = None,
    qdrant_cfg: QdrantConfig | None = None,
    qdrant_client: Any = None,
    neo4j_cfg: Neo4jConfig | None = None,
) -> Itinerary:
    """Aggregate calendar events with their linked tasks/notes/threads.

    Reuses a single Neo4j driver across all sub-queries.  When *registry*
    is ``None`` the notes section is skipped silently — callers that
    don't have an embedding model wired up still get events + tasks +
    threads.
    """
    tz = _local_tz(timezone_)
    target_day = _resolve_day(day, tz)

    drv = _open_driver(neo4j_cfg)
    try:
        events = events_for_day(
            target_day,
            account=account,
            timezone_=tz,
            driver=drv,
        )
        out_events: list[EventWithLinks] = []
        for ev in events:
            tasks = linked_tasks_for_event(
                ev.id, k=k_tasks, horizon=horizon, driver=drv
            )
            notes: list[NoteHit] = []
            if registry is not None:
                try:
                    notes = linked_notes_for_event(
                        ev.id,
                        k=k_notes,
                        horizon=horizon,
                        registry=registry,
                        qdrant_cfg=qdrant_cfg,
                        qdrant_client=qdrant_client,
                        driver=drv,
                    )
                except Exception as exc:
                    logger.warning(
                        "linked_notes_for_event failed for event %s: %s",
                        ev.id,
                        exc,
                    )
                    notes = []
            thread = recent_thread_with_attendees(ev.id, horizon=horizon, driver=drv)
            out_events.append(
                EventWithLinks(
                    event=ev,
                    tasks=tasks,
                    notes=notes,
                    thread=thread,
                )
            )
    finally:
        drv.close()

    return Itinerary(
        day=target_day,
        timezone=getattr(tz, "key", None) or str(tz),
        events=out_events,
    )
