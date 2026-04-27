"""Pre-brief assembler for ``fieldnotes person <id> --summary``.

Pulls structured input for the LLM brief from existing graph edges:

  1. Identity line (name, primary email, source count)
  2. Open OmniFocus tasks mentioning this Person
  3. Outstanding email threads — Threads where the last Email was sent
     by the target (so the user owes them a reply) within ``since``.
  4. Recent Slack mentions of self by the target with no later reply
     from a self-Person in the same channel within ``since``.
  5. Top active topics over docs touching this Person modified within
     ``since`` (TAGGED edges).
  6. Optional: Obsidian note at ``<vault>/People/<name>.md`` raw body
     (capped at 2000 chars).
  7. Optional: ``meeting_id`` resolves to a ``CalendarEvent`` and adds
     summary, description, location, attendees, and ATTACHED_TO docs.

No vector search — this assembler is purely structured.  It is the
deterministic input to the LLM call in :mod:`worker.cli.person_brief`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

from neo4j import Driver

from worker.query.person import (
    OpenTask,
    Person,
    TopicCount,
    _cluster_ids,
    open_tasks,
)
from worker.query.person import _epoch_seconds, _ts_to_epoch


_OBSIDIAN_NOTE_CAP = 2000
_DEFAULT_THREADS = 10
_DEFAULT_SLACK_MENTIONS = 10
_DEFAULT_TOPICS = 5


@dataclass
class EmailThreadSnippet:
    subject: str
    last_date: str
    last_subject: str | None = None


@dataclass
class SlackMentionSnippet:
    channel: str
    ts: str
    text: str | None = None


@dataclass
class MeetingContext:
    event_id: str
    summary: str
    start_time: str | None = None
    description: str | None = None
    location: str | None = None
    attendees: list[str] = field(default_factory=list)
    attachments: list[str] = field(default_factory=list)


@dataclass
class PreBrief:
    identity_name: str
    identity_email: str
    source_count: int
    open_tasks: list[OpenTask] = field(default_factory=list)
    email_threads: list[EmailThreadSnippet] = field(default_factory=list)
    slack_mentions: list[SlackMentionSnippet] = field(default_factory=list)
    top_topics: list[TopicCount] = field(default_factory=list)
    obsidian_note: str | None = None
    meeting: MeetingContext | None = None


# ---------------------------------------------------------------------------
# Sub-queries
# ---------------------------------------------------------------------------


_EMAIL_THREADS_CYPHER = """\
MATCH (p:Person) WHERE id(p) IN $cluster
MATCH (p)-[:SENT]->(:Email)-[:PART_OF]->(t:Thread)
WITH DISTINCT t
MATCH (e:Email)-[:PART_OF]->(t)
WITH t, max(e.date) AS last_date
MATCH (last:Email)-[:PART_OF]->(t)
WHERE last.date = last_date
MATCH (last)<-[:SENT]-(sender:Person)
WHERE id(sender) IN $cluster
RETURN COALESCE(t.subject, last.subject, '(no subject)') AS subject,
       last_date AS last_date,
       last.subject AS last_subject
ORDER BY last_date DESC
"""


def _email_threads(
    driver: Driver, cluster: list[int], since: datetime, limit: int
) -> list[EmailThreadSnippet]:
    if not cluster:
        return []
    with driver.session() as session:
        rows = session.execute_read(
            lambda tx: tx.run(_EMAIL_THREADS_CYPHER, cluster=cluster).data()
        )
    since_epoch = _epoch_seconds(since)
    out: list[EmailThreadSnippet] = []
    for row in rows:
        last_date = str(row.get("last_date") or "")
        epoch = _ts_to_epoch(last_date)
        if epoch is None or epoch < since_epoch:
            continue
        out.append(
            EmailThreadSnippet(
                subject=str(row.get("subject") or ""),
                last_date=last_date,
                last_subject=row.get("last_subject"),
            )
        )
        if len(out) >= limit:
            break
    return out


_SLACK_MENTIONS_CYPHER = """\
MATCH (target:Person) WHERE id(target) IN $cluster
MATCH (s:SlackMessage)-[:SENT_BY]->(target)
MATCH (s)-[:MENTIONS]->(self_p:Person)
WHERE self_p.is_self = true OR self_p.email IN $self_emails
WITH s, self_p, target
OPTIONAL MATCH (later:SlackMessage)-[:SENT_BY]->(reply:Person)
  WHERE (reply.is_self = true OR reply.email IN $self_emails)
    AND later.channel_id = s.channel_id
    AND later.first_ts > s.last_ts
WITH s, count(later) AS replies
WHERE replies = 0
RETURN COALESCE(s.channel_name, s.channel_id, '(slack)') AS channel,
       COALESCE(s.last_ts, s.first_ts, '') AS ts,
       COALESCE(s.text_preview, s.summary, '') AS text
ORDER BY ts DESC
"""


def _slack_mentions(
    driver: Driver,
    cluster: list[int],
    since: datetime,
    self_emails: list[str],
    limit: int,
) -> list[SlackMentionSnippet]:
    if not cluster:
        return []
    with driver.session() as session:
        rows = session.execute_read(
            lambda tx: tx.run(
                _SLACK_MENTIONS_CYPHER,
                cluster=cluster,
                self_emails=self_emails,
            ).data()
        )
    since_epoch = _epoch_seconds(since)
    out: list[SlackMentionSnippet] = []
    for row in rows:
        ts = str(row.get("ts") or "")
        epoch = _ts_to_epoch(ts)
        if epoch is None or epoch < since_epoch:
            continue
        out.append(
            SlackMentionSnippet(
                channel=str(row.get("channel") or ""),
                ts=ts,
                text=row.get("text") or None,
            )
        )
        if len(out) >= limit:
            break
    return out


_TOPICS_CYPHER = """\
MATCH (p:Person) WHERE id(p) IN $cluster
MATCH (d)-[r]-(p)
WHERE NOT d:Person AND type(r) IN $edge_types
WITH DISTINCT d
WITH d, COALESCE(
    d.modified_at, d.updated, d.date, d.start_time, d.last_ts, d.created
) AS doc_ts
WHERE doc_ts IS NOT NULL
MATCH (d)-[:TAGGED]->(t:Topic)
RETURN t.name AS topic_name, id(d) AS doc_id, doc_ts AS doc_ts
"""


_PERSON_DOC_EDGES = (
    "MENTIONS",
    "SENT",
    "SENT_BY",
    "AUTHORED_BY",
    "TO",
    "ORGANIZED_BY",
    "ATTENDED_BY",
    "CREATED_BY",
)


def _top_active_topics(
    driver: Driver, cluster: list[int], since: datetime, k: int
) -> list[TopicCount]:
    if not cluster:
        return []
    with driver.session() as session:
        rows = session.execute_read(
            lambda tx: tx.run(
                _TOPICS_CYPHER,
                cluster=cluster,
                edge_types=list(_PERSON_DOC_EDGES),
            ).data()
        )
    since_epoch = _epoch_seconds(since)
    counts: dict[str, set[int]] = {}
    for row in rows:
        ts = str(row.get("doc_ts") or "")
        epoch = _ts_to_epoch(ts)
        if epoch is None or epoch < since_epoch:
            continue
        topic = str(row.get("topic_name") or "")
        if not topic:
            continue
        counts.setdefault(topic, set()).add(int(row["doc_id"]))
    ranked = sorted(
        ((name, len(ids)) for name, ids in counts.items()),
        key=lambda pair: (-pair[1], pair[0]),
    )
    return [TopicCount(topic_name=name, doc_count=count) for name, count in ranked[:k]]


_MEETING_CYPHER = """\
MATCH (c) WHERE c.source_id = $event_id AND (c:CalendarEvent OR c:CalendarSeries)
OPTIONAL MATCH (c)-[:ORGANIZED_BY|ATTENDED_BY|CREATED_BY]->(att:Person)
WITH c, collect(DISTINCT COALESCE(att.email, att.name)) AS attendees
OPTIONAL MATCH (a:Attachment)-[:ATTACHED_TO]->(c)
WITH c, attendees, collect(DISTINCT COALESCE(a.title, a.file_id)) AS attachments
RETURN COALESCE(c.summary, '(no summary)') AS summary,
       c.description AS description,
       c.location AS location,
       c.start_time AS start_time,
       attendees AS attendees,
       attachments AS attachments
LIMIT 1
"""


def _meeting_context(driver: Driver, event_id: str) -> MeetingContext | None:
    with driver.session() as session:
        row = session.execute_read(
            lambda tx: tx.run(_MEETING_CYPHER, event_id=event_id).single()
        )
    if not row:
        return None
    data = row.data()
    return MeetingContext(
        event_id=event_id,
        summary=str(data.get("summary") or "(no summary)"),
        description=data.get("description") or None,
        location=data.get("location") or None,
        start_time=data.get("start_time") or None,
        attendees=[a for a in (data.get("attendees") or []) if a],
        attachments=[a for a in (data.get("attachments") or []) if a],
    )


def _obsidian_people_note(vault_path: Path | None, name: str) -> str | None:
    if not vault_path or not name:
        return None
    candidate = vault_path / "People" / f"{name}.md"
    try:
        if not candidate.is_file():
            return None
        body = candidate.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if len(body) > _OBSIDIAN_NOTE_CAP:
        body = body[:_OBSIDIAN_NOTE_CAP].rstrip() + "\n…"
    return body


# ---------------------------------------------------------------------------
# Source counter
# ---------------------------------------------------------------------------


_SOURCE_TYPES_CYPHER = """\
MATCH (p:Person) WHERE id(p) IN $cluster
OPTIONAL MATCH (p)-[:SENT]->(out:Email)
OPTIONAL MATCH (e_in:Email)-[:TO|MENTIONS]->(p)
OPTIONAL MATCH (c)-[:ATTENDED_BY|ORGANIZED_BY|CREATED_BY]->(p)
  WHERE c:CalendarEvent OR c:CalendarSeries
OPTIONAL MATCH (s:SlackMessage)-[:SENT_BY|MENTIONS]->(p)
OPTIONAL MATCH (f:File)-[:MENTIONS]->(p)
OPTIONAL MATCH (t:Task)-[:MENTIONS]->(p)
RETURN
  (CASE WHEN count(DISTINCT out) + count(DISTINCT e_in) > 0 THEN 1 ELSE 0 END) +
  (CASE WHEN count(DISTINCT c) > 0 THEN 1 ELSE 0 END) +
  (CASE WHEN count(DISTINCT s) > 0 THEN 1 ELSE 0 END) +
  (CASE WHEN count(DISTINCT f) > 0 THEN 1 ELSE 0 END) +
  (CASE WHEN count(DISTINCT t) > 0 THEN 1 ELSE 0 END) AS sources
"""


def _source_count(driver: Driver, cluster: list[int]) -> int:
    if not cluster:
        return 0
    with driver.session() as session:
        row = session.execute_read(
            lambda tx: tx.run(_SOURCE_TYPES_CYPHER, cluster=cluster).single()
        )
    return int(row["sources"]) if row else 0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def assemble_prebrief(
    person: Person,
    *,
    driver: Driver,
    since: datetime,
    self_emails: Iterable[str] = (),
    vault_path: Path | None = None,
    meeting_id: str | None = None,
) -> PreBrief:
    """Build a :class:`PreBrief` for *person*."""
    self_email_list = [e for e in self_emails if e]
    with driver.session() as session:
        cluster = session.execute_read(_cluster_ids, person.id)

    prebrief = PreBrief(
        identity_name=person.name or "(unnamed)",
        identity_email=person.email or "",
        source_count=_source_count(driver, cluster),
    )

    prebrief.open_tasks = open_tasks(person.id, driver=driver)
    prebrief.email_threads = _email_threads(driver, cluster, since, _DEFAULT_THREADS)
    prebrief.slack_mentions = _slack_mentions(
        driver, cluster, since, self_email_list, _DEFAULT_SLACK_MENTIONS
    )
    prebrief.top_topics = _top_active_topics(driver, cluster, since, _DEFAULT_TOPICS)

    prebrief.obsidian_note = _obsidian_people_note(vault_path, person.name or "")

    if meeting_id:
        meeting = _meeting_context(driver, meeting_id)
        if meeting is None:
            raise ValueError(f"Calendar event not found for meeting_id {meeting_id!r}")
        prebrief.meeting = meeting

    return prebrief


# ---------------------------------------------------------------------------
# Render to plain text (the LLM user message)
# ---------------------------------------------------------------------------


def format_prebrief(prebrief: PreBrief, *, since_label: str) -> str:
    """Render *prebrief* as the deterministic user-message context."""
    lines: list[str] = []

    lines.append("[Identity]")
    line = prebrief.identity_name
    if prebrief.identity_email:
        line += f" <{prebrief.identity_email}>"
    line += f" — {prebrief.source_count} sources"
    lines.append(line)
    lines.append("")

    lines.append(f"[Open OmniFocus tasks ({len(prebrief.open_tasks)})]")
    if prebrief.open_tasks:
        for t in prebrief.open_tasks:
            bits = [f"- {t.title}"]
            if t.project:
                bits.append(f"(project: {t.project})")
            if t.due:
                bits.append(f"due {t.due}")
            if t.flagged:
                bits.append("[flagged]")
            lines.append(" ".join(bits))
    else:
        lines.append("(none)")
    lines.append("")

    lines.append(
        f"[Outstanding email threads (last {since_label}, "
        f"{len(prebrief.email_threads)})]"
    )
    if prebrief.email_threads:
        for th in prebrief.email_threads:
            lines.append(f"- {th.subject} (last reply {th.last_date})")
    else:
        lines.append("(none)")
    lines.append("")

    lines.append(
        f"[Unresolved Slack mentions of you (last {since_label}, "
        f"{len(prebrief.slack_mentions)})]"
    )
    if prebrief.slack_mentions:
        for m in prebrief.slack_mentions:
            text = (m.text or "").replace("\n", " ").strip()
            if len(text) > 200:
                text = text[:200].rstrip() + "…"
            suffix = f" — {text}" if text else ""
            lines.append(f"- #{m.channel} @ {m.ts}{suffix}")
    else:
        lines.append("(none)")
    lines.append("")

    lines.append(f"[Top active topics (last {since_label})]")
    if prebrief.top_topics:
        for tp in prebrief.top_topics:
            lines.append(f"- {tp.topic_name} ({tp.doc_count} docs)")
    else:
        lines.append("(none)")
    lines.append("")

    if prebrief.obsidian_note is not None:
        lines.append("[Obsidian People note]")
        lines.append(prebrief.obsidian_note.strip())
        lines.append("")

    if prebrief.meeting is not None:
        m = prebrief.meeting
        lines.append("[Upcoming meeting context]")
        lines.append(f"Event: {m.summary}")
        if m.start_time:
            lines.append(f"When: {m.start_time}")
        if m.location:
            lines.append(f"Where: {m.location}")
        if m.attendees:
            lines.append(f"Attendees: {', '.join(m.attendees)}")
        if m.description:
            desc = m.description.strip()
            if len(desc) > 1000:
                desc = desc[:1000].rstrip() + "…"
            lines.append("Description:")
            lines.append(desc)
        if m.attachments:
            lines.append("Attachments: " + ", ".join(m.attachments))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
