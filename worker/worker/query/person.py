"""Person profile queries: read-only Cypher against Neo4j.

Powers the ``fieldnotes person <id>`` view (CLI/MCP/LLM landing on top
of this module — those layers are separate beads).  All functions are
read-only and traverse only the edges already produced by the
ingestion parsers:

  * Email     — ``(p)-[:SENT]->(e)`` / ``(e)-[:TO|MENTIONS]->(p)``
  * Calendar  — ``(e)-[:ORGANIZED_BY|ATTENDED_BY|CREATED_BY]->(p)``
  * Slack     — ``(s)-[:SENT_BY|MENTIONS]->(p)``
  * Files     — ``(f)-[:MENTIONS]->(p)``
  * Tasks     — ``(t:Task)-[:MENTIONS]->(p)``
  * Topics    — ``(d)-[:TAGGED]->(t:Topic)``
  * Identity  — ``(p)-[:SAME_AS]-(q)``

A Person node may appear multiple times in the graph (one per source
identity that hasn't been reconciled into the canonical email-keyed
node); :func:`find_person` always returns the highest-degree node in
the SAME_AS cluster, and the per-section queries traverse the cluster
so an alias node's edges are attributed to the canonical Person.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from neo4j import Driver
from rapidfuzz import fuzz

from worker.config import Neo4jConfig
from worker.neo4j_driver import build_driver
from worker.parsers.base import canonicalize_email

logger = logging.getLogger(__name__)


_DEFAULT_INTERACTION_LIMIT = 10
_DEFAULT_TOPIC_LIMIT = 5
_DEFAULT_RELATED_LIMIT = 5
_DEFAULT_FILES_LIMIT = 10
_FUZZY_NAME_THRESHOLD = 90  # rapidfuzz token_sort_ratio cutoff for find_person
_SLACK_USER_PREFIX = "slack-user:"


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Person:
    """A canonical Person node returned by :func:`find_person`."""

    id: int
    email: str | None = None
    name: str | None = None
    slack_user_id: str | None = None
    team_id: str | None = None
    source_id: str | None = None
    is_self: bool = False


@dataclass
class Interaction:
    """One interaction row from :func:`recent_interactions`."""

    timestamp: str
    source_type: str
    title: str
    snippet: str
    edge_kind: str


@dataclass
class TopicCount:
    """One row from :func:`top_topics`."""

    topic_name: str
    doc_count: int


@dataclass
class RelatedPerson:
    """One row from :func:`related_people`."""

    name: str | None
    email: str | None
    shared_count: int


@dataclass
class OpenTask:
    """One row from :func:`open_tasks`."""

    title: str
    project: str | None
    tags: list[str] = field(default_factory=list)
    due: str | None = None
    defer: str | None = None
    flagged: bool = False


@dataclass
class FileMention:
    """One row from :func:`files_mentioning`."""

    path: str
    mtime: str | None = None
    source: str | None = None


@dataclass
class IdentityMember:
    """One row from :func:`identity_cluster`."""

    member: str
    match_type: str | None = None
    confidence: float | None = None
    cross_source: bool | None = None


@dataclass
class PersonProfile:
    """Aggregated profile returned by :func:`get_profile`."""

    person: Person
    recent_interactions: list[Interaction] = field(default_factory=list)
    top_topics: list[TopicCount] = field(default_factory=list)
    related_people: list[RelatedPerson] = field(default_factory=list)
    open_tasks: list[OpenTask] = field(default_factory=list)
    files: list[FileMention] = field(default_factory=list)
    identity_cluster: list[IdentityMember] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Driver helpers
# ---------------------------------------------------------------------------


def _open_driver(neo4j_cfg: Neo4jConfig | None) -> Driver:
    cfg = neo4j_cfg or Neo4jConfig()
    return build_driver(cfg.uri, cfg.user, cfg.password)


def _person_from_row(row: dict[str, Any]) -> Person:
    return Person(
        id=int(row["id"]),
        email=row.get("email"),
        name=row.get("name"),
        slack_user_id=row.get("slack_user_id"),
        team_id=row.get("team_id"),
        source_id=row.get("source_id"),
        is_self=bool(row.get("is_self") or False),
    )


def _canonical_for_id(tx: Any, person_id: int) -> Person | None:
    """Return the highest-degree Person in *person_id*'s SAME_AS cluster.

    Degree counts every edge except SAME_AS itself, so a richly-connected
    email-keyed Person beats a stub slack-user-keyed Person that only
    carries identity edges.  Ties break on the lowest internal id for
    deterministic output.
    """
    row = tx.run(
        """
        MATCH (seed:Person) WHERE id(seed) = $pid
        OPTIONAL MATCH (seed)-[:SAME_AS*0..]-(p:Person)
        WITH collect(DISTINCT p) + seed AS members
        UNWIND members AS m
        WITH DISTINCT m AS p
        OPTIONAL MATCH (p)-[r]-()
        WITH p, sum(CASE WHEN type(r) = 'SAME_AS' OR r IS NULL THEN 0 ELSE 1 END) AS deg
        RETURN id(p) AS id,
               p.email AS email,
               p.name AS name,
               p.slack_user_id AS slack_user_id,
               p.team_id AS team_id,
               p.source_id AS source_id,
               COALESCE(p.is_self, false) AS is_self
        ORDER BY deg DESC, id(p) ASC
        LIMIT 1
        """,
        pid=person_id,
    ).single()
    return _person_from_row(row.data()) if row else None


def _cluster_ids(tx: Any, person_id: int) -> list[int]:
    """All Neo4j ids in *person_id*'s SAME_AS cluster (including itself)."""
    rows = tx.run(
        """
        MATCH (seed:Person) WHERE id(seed) = $pid
        OPTIONAL MATCH (seed)-[:SAME_AS*0..]-(p:Person)
        WITH collect(DISTINCT p) + seed AS members
        UNWIND members AS m
        WITH DISTINCT m
        RETURN id(m) AS id
        """,
        pid=person_id,
    ).data()
    return [int(r["id"]) for r in rows]


# ---------------------------------------------------------------------------
# find_person
# ---------------------------------------------------------------------------


def find_person(
    identifier: str,
    *,
    neo4j_cfg: Neo4jConfig | None = None,
    driver: Driver | None = None,
) -> Person | list[Person] | None:
    """Resolve *identifier* to a canonical :class:`Person`.

    Resolution order:

    1. ``alice@example.com`` — canonicalized via
       :func:`canonicalize_email` and matched on ``Person.email``.
    2. ``slack-user:<team>/<user>`` — matched on
       ``(slack_user_id, team_id)``.
    3. Free text — fuzzy-matched against ``Person.name`` using
       ``rapidfuzz.token_sort_ratio`` with a 90/100 cutoff.

    Returns the canonical (highest-degree) Person in each match's
    SAME_AS cluster.  Fuzzy lookups that yield more than one canonical
    Person return a list so the caller can disambiguate.
    """
    own_driver = driver is None
    drv = driver or _open_driver(neo4j_cfg)
    try:
        with drv.session() as session:
            return session.execute_read(_find_person_tx, identifier)
    finally:
        if own_driver:
            drv.close()


def _find_person_tx(tx: Any, identifier: str) -> Person | list[Person] | None:
    text = identifier.strip()
    if not text:
        return None

    # 1. Email
    if "@" in text:
        norm = canonicalize_email(text)
        row = tx.run(
            "MATCH (p:Person {email: $email}) RETURN id(p) AS id LIMIT 1",
            email=norm,
        ).single()
        if not row:
            return None
        return _canonical_for_id(tx, int(row["id"]))

    # 2. Slack user
    if text.startswith(_SLACK_USER_PREFIX):
        rest = text[len(_SLACK_USER_PREFIX) :]
        if "/" not in rest:
            return None
        team_id, user_id = rest.split("/", 1)
        if not team_id or not user_id:
            return None
        row = tx.run(
            "MATCH (p:Person {slack_user_id: $uid, team_id: $tid}) "
            "RETURN id(p) AS id LIMIT 1",
            uid=user_id,
            tid=team_id,
        ).single()
        if not row:
            return None
        return _canonical_for_id(tx, int(row["id"]))

    # 3. Fuzzy name
    return _find_by_fuzzy_name(tx, text)


def _find_by_fuzzy_name(tx: Any, query: str) -> Person | list[Person] | None:
    rows = tx.run(
        """
        MATCH (p:Person)
        WHERE p.name IS NOT NULL AND trim(p.name) <> ''
        RETURN id(p) AS id, p.name AS name
        """
    ).data()

    target = query.lower()
    hits: list[int] = []
    for row in rows:
        score = fuzz.token_sort_ratio(target, str(row["name"]).lower())
        if score >= _FUZZY_NAME_THRESHOLD:
            hits.append(int(row["id"]))

    if not hits:
        return None

    canonicals: dict[int, Person] = {}
    for hit_id in hits:
        person = _canonical_for_id(tx, hit_id)
        if person is not None:
            canonicals[person.id] = person

    if not canonicals:
        return None
    if len(canonicals) == 1:
        return next(iter(canonicals.values()))
    return list(canonicals.values())


# ---------------------------------------------------------------------------
# recent_interactions
# ---------------------------------------------------------------------------


_INTERACTIONS_CYPHER = """\
MATCH (p:Person) WHERE id(p) IN $cluster
MATCH (p)-[:SENT]->(e:Email)
RETURN id(e) AS doc_id,
       COALESCE(e.date, '') AS ts,
       'gmail' AS source_type,
       COALESCE(e.subject, e.source_id, '(no subject)') AS title,
       COALESCE(e.subject, '') AS snippet,
       'SENT' AS edge_kind

UNION

MATCH (p:Person) WHERE id(p) IN $cluster
MATCH (e:Email)-[r:TO|MENTIONS]->(p)
RETURN id(e) AS doc_id,
       COALESCE(e.date, '') AS ts,
       'gmail' AS source_type,
       COALESCE(e.subject, e.source_id, '(no subject)') AS title,
       COALESCE(e.subject, '') AS snippet,
       type(r) AS edge_kind

UNION

MATCH (p:Person) WHERE id(p) IN $cluster
MATCH (c)-[r:ATTENDED_BY|ORGANIZED_BY|CREATED_BY]->(p)
WHERE c:CalendarEvent OR c:CalendarSeries
RETURN id(c) AS doc_id,
       COALESCE(c.start_time, '') AS ts,
       'google_calendar' AS source_type,
       COALESCE(c.summary, c.source_id, '(no summary)') AS title,
       COALESCE(c.location, '') AS snippet,
       type(r) AS edge_kind

UNION

MATCH (p:Person) WHERE id(p) IN $cluster
MATCH (s:SlackMessage)-[r:SENT_BY|MENTIONS]->(p)
RETURN id(s) AS doc_id,
       COALESCE(s.last_ts, s.first_ts, '') AS ts,
       'slack' AS source_type,
       COALESCE(s.channel_name, s.source_id, '(slack)') AS title,
       '' AS snippet,
       type(r) AS edge_kind

UNION

MATCH (p:Person) WHERE id(p) IN $cluster
MATCH (f:File)-[:MENTIONS]->(p)
RETURN id(f) AS doc_id,
       COALESCE(f.modified_at, f.updated, f.created, '') AS ts,
       'file' AS source_type,
       COALESCE(f.title, f.name, f.path, f.source_id, '(file)') AS title,
       '' AS snippet,
       'MENTIONS' AS edge_kind
"""


def recent_interactions(
    person_id: int,
    since: datetime,
    *,
    limit: int = _DEFAULT_INTERACTION_LIMIT,
    neo4j_cfg: Neo4jConfig | None = None,
    driver: Driver | None = None,
) -> list[Interaction]:
    """Return up to *limit* interactions on or after *since*, newest first.

    Spans Email, CalendarEvent (incl. CalendarSeries), SlackMessage and
    File nodes that connect to *any* member of the canonical Person's
    SAME_AS cluster.  Filtering by *since* and ordering happen in Python
    because timestamps live in heterogeneous string/numeric formats
    across sources (ISO date for Email/Calendar, Slack ``ts`` floats,
    File ``modified_at`` ISO).
    """
    own_driver = driver is None
    drv = driver or _open_driver(neo4j_cfg)
    try:
        with drv.session() as session:
            cluster = session.execute_read(_cluster_ids, person_id)
            if not cluster:
                return []
            rows = session.execute_read(
                lambda tx: tx.run(
                    _INTERACTIONS_CYPHER,
                    cluster=cluster,
                ).data()
            )
    finally:
        if own_driver:
            drv.close()

    since_epoch = _epoch_seconds(since)
    filtered: list[tuple[float, Interaction]] = []
    for row in rows:
        ts = str(row.get("ts") or "")
        if not ts:
            continue
        epoch = _ts_to_epoch(ts)
        if epoch is None or epoch < since_epoch:
            continue
        filtered.append(
            (
                epoch,
                Interaction(
                    timestamp=ts,
                    source_type=str(row.get("source_type") or ""),
                    title=str(row.get("title") or ""),
                    snippet=str(row.get("snippet") or ""),
                    edge_kind=str(row.get("edge_kind") or ""),
                ),
            )
        )

    filtered.sort(key=lambda pair: pair[0], reverse=True)
    return [interaction for _, interaction in filtered[: max(0, int(limit))]]


def _epoch_seconds(value: datetime) -> float:
    """UTC epoch seconds.  Naive datetimes are treated as UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc).timestamp()
    return value.timestamp()


def _ts_to_epoch(ts: str) -> float | None:
    """Best-effort epoch-seconds conversion for the timestamp shapes
    written by parsers: ISO-8601 (``2024-08-06T14:55:23Z`` / with offset)
    and Slack-style Unix seconds (``"1722956123.456"``).
    """
    if not ts:
        return None
    # Slack ``ts``: pure float-seconds-with-fraction.  Anything else with
    # a leading digit and no 'T' is also treated as epoch (defensive).
    if ts[0].isdigit() and "T" not in ts and "-" not in ts[1:]:
        try:
            return float(ts)
        except ValueError:
            return None
    # ISO-8601.  Accept trailing 'Z' (Python <3.11 didn't).
    iso = ts.replace("Z", "+00:00") if ts.endswith("Z") else ts
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


# ---------------------------------------------------------------------------
# top_topics
# ---------------------------------------------------------------------------


_PERSON_DOC_TYPES = (
    "MENTIONS",
    "SENT",
    "SENT_BY",
    "AUTHORED_BY",
    "TO",
    "ORGANIZED_BY",
    "ATTENDED_BY",
    "CREATED_BY",
)


_TOP_TOPICS_CYPHER = """\
MATCH (p:Person) WHERE id(p) IN $cluster
MATCH (d)-[r]-(p)
WHERE NOT d:Person AND type(r) IN $edge_types
MATCH (d)-[:TAGGED]->(t:Topic)
WITH t.name AS topic_name, count(DISTINCT d) AS doc_count
RETURN topic_name, doc_count
ORDER BY doc_count DESC, topic_name ASC
LIMIT $k
"""


def top_topics(
    person_id: int,
    *,
    k: int = _DEFAULT_TOPIC_LIMIT,
    neo4j_cfg: Neo4jConfig | None = None,
    driver: Driver | None = None,
) -> list[TopicCount]:
    """Return the *k* topics most often co-occurring with this Person."""
    own_driver = driver is None
    drv = driver or _open_driver(neo4j_cfg)
    try:
        with drv.session() as session:
            cluster = session.execute_read(_cluster_ids, person_id)
            if not cluster:
                return []
            rows = session.execute_read(
                lambda tx: tx.run(
                    _TOP_TOPICS_CYPHER,
                    cluster=cluster,
                    edge_types=list(_PERSON_DOC_TYPES),
                    k=int(k),
                ).data()
            )
    finally:
        if own_driver:
            drv.close()
    return [
        TopicCount(topic_name=str(r["topic_name"]), doc_count=int(r["doc_count"]))
        for r in rows
    ]


# ---------------------------------------------------------------------------
# related_people
# ---------------------------------------------------------------------------


_RELATED_CYPHER = """\
MATCH (p:Person) WHERE id(p) IN $cluster
MATCH (d)-[r1]-(p)
WHERE NOT d:Person AND type(r1) IN $edge_types
MATCH (d)-[r2]-(other:Person)
WHERE NOT id(other) IN $cluster AND type(r2) IN $edge_types
WITH other, count(DISTINCT d) AS shared_count
RETURN id(other) AS other_id,
       other.name AS name,
       other.email AS email,
       shared_count
ORDER BY shared_count DESC, other_id ASC
LIMIT $k
"""


def related_people(
    person_id: int,
    *,
    k: int = _DEFAULT_RELATED_LIMIT,
    neo4j_cfg: Neo4jConfig | None = None,
    driver: Driver | None = None,
) -> list[RelatedPerson]:
    """Return Persons most frequently sharing documents with *person_id*.

    Other members of the SAME_AS cluster are excluded from the result.
    """
    own_driver = driver is None
    drv = driver or _open_driver(neo4j_cfg)
    try:
        with drv.session() as session:
            cluster = session.execute_read(_cluster_ids, person_id)
            if not cluster:
                return []
            rows = session.execute_read(
                lambda tx: tx.run(
                    _RELATED_CYPHER,
                    cluster=cluster,
                    edge_types=list(_PERSON_DOC_TYPES),
                    k=int(k),
                ).data()
            )
    finally:
        if own_driver:
            drv.close()
    return [
        RelatedPerson(
            name=r.get("name"),
            email=r.get("email"),
            shared_count=int(r["shared_count"]),
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# open_tasks
# ---------------------------------------------------------------------------


_OPEN_TASKS_CYPHER = """\
MATCH (p:Person) WHERE id(p) IN $cluster
MATCH (t:Task)-[:MENTIONS]->(p)
WHERE COALESCE(t.status, 'active') <> 'completed'
  AND COALESCE(t.status, 'active') <> 'dropped'
OPTIONAL MATCH (t)-[:IN_PROJECT]->(proj:Project)
OPTIONAL MATCH (t)-[:TAGGED]->(tag:Tag)
WITH t, proj, collect(DISTINCT tag.name) AS tags
RETURN t.name AS title,
       proj.name AS project,
       tags,
       t.due_date AS due,
       t.defer_date AS defer,
       COALESCE(t.flagged, false) AS flagged,
       id(t) AS task_id
ORDER BY flagged DESC, due ASC, task_id ASC
"""


def open_tasks(
    person_id: int,
    *,
    neo4j_cfg: Neo4jConfig | None = None,
    driver: Driver | None = None,
) -> list[OpenTask]:
    """Return open OmniFocus tasks that mention this Person."""
    own_driver = driver is None
    drv = driver or _open_driver(neo4j_cfg)
    try:
        with drv.session() as session:
            cluster = session.execute_read(_cluster_ids, person_id)
            if not cluster:
                return []
            rows = session.execute_read(
                lambda tx: tx.run(
                    _OPEN_TASKS_CYPHER,
                    cluster=cluster,
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
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# files_mentioning
# ---------------------------------------------------------------------------


_FILES_CYPHER = """\
MATCH (p:Person) WHERE id(p) IN $cluster
MATCH (f:File)-[:MENTIONS]->(p)
RETURN id(f) AS file_id,
       COALESCE(f.path, f.source_id) AS path,
       COALESCE(f.modified_at, f.updated, f.created) AS mtime,
       f.source AS source
ORDER BY mtime DESC, file_id ASC
LIMIT $k
"""


def files_mentioning(
    person_id: int,
    *,
    k: int = _DEFAULT_FILES_LIMIT,
    neo4j_cfg: Neo4jConfig | None = None,
    driver: Driver | None = None,
) -> list[FileMention]:
    """Return up to *k* File nodes mentioning this Person, newest first."""
    own_driver = driver is None
    drv = driver or _open_driver(neo4j_cfg)
    try:
        with drv.session() as session:
            cluster = session.execute_read(_cluster_ids, person_id)
            if not cluster:
                return []
            rows = session.execute_read(
                lambda tx: tx.run(
                    _FILES_CYPHER,
                    cluster=cluster,
                    k=int(k),
                ).data()
            )
    finally:
        if own_driver:
            drv.close()
    return [
        FileMention(
            path=str(r.get("path") or ""),
            mtime=r.get("mtime"),
            source=r.get("source"),
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# identity_cluster
# ---------------------------------------------------------------------------


_CLUSTER_CYPHER = """\
MATCH (seed:Person) WHERE id(seed) = $pid
OPTIONAL MATCH (seed)-[r:SAME_AS]-(other:Person)
RETURN id(other) AS other_id,
       COALESCE(other.email, other.source_id, toString(id(other))) AS member,
       r.match_type AS match_type,
       r.confidence AS confidence,
       r.cross_source AS cross_source
"""


def identity_cluster(
    person_id: int,
    *,
    neo4j_cfg: Neo4jConfig | None = None,
    driver: Driver | None = None,
) -> list[IdentityMember]:
    """Return SAME_AS neighbours of *person_id* with edge metadata."""
    own_driver = driver is None
    drv = driver or _open_driver(neo4j_cfg)
    try:
        with drv.session() as session:
            rows = session.execute_read(
                lambda tx: tx.run(_CLUSTER_CYPHER, pid=person_id).data()
            )
    finally:
        if own_driver:
            drv.close()
    members: list[IdentityMember] = []
    for r in rows:
        if r.get("other_id") is None:
            continue
        confidence = r.get("confidence")
        members.append(
            IdentityMember(
                member=str(r.get("member") or ""),
                match_type=r.get("match_type"),
                confidence=float(confidence) if confidence is not None else None,
                cross_source=(
                    bool(r["cross_source"])
                    if r.get("cross_source") is not None
                    else None
                ),
            )
        )
    return members


# ---------------------------------------------------------------------------
# get_profile
# ---------------------------------------------------------------------------


def get_profile(
    identifier: str,
    since: datetime,
    *,
    limit: int = _DEFAULT_INTERACTION_LIMIT,
    neo4j_cfg: Neo4jConfig | None = None,
) -> PersonProfile | list[Person] | None:
    """One-shot helper that resolves *identifier* and assembles a profile.

    Reuses a single Neo4j driver across all sub-queries.  When the
    identifier resolves ambiguously (multiple fuzzy-name hits) the list
    of candidate Persons is returned for the caller to disambiguate; on
    a miss returns ``None``.
    """
    drv = _open_driver(neo4j_cfg)
    try:
        with drv.session() as session:
            resolved = session.execute_read(_find_person_tx, identifier)

        if resolved is None:
            return None
        if isinstance(resolved, list):
            return resolved

        person = resolved
        profile = PersonProfile(person=person)
        profile.recent_interactions = recent_interactions(
            person.id, since, limit=limit, driver=drv
        )
        profile.top_topics = top_topics(person.id, driver=drv)
        profile.related_people = related_people(person.id, driver=drv)
        profile.open_tasks = open_tasks(person.id, driver=drv)
        profile.files = files_mentioning(person.id, driver=drv)
        profile.identity_cluster = identity_cluster(person.id, driver=drv)
        return profile
    finally:
        drv.close()
