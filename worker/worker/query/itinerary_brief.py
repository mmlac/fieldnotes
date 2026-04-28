"""Pre-brief assembler for ``fieldnotes itinerary`` per-event LLM brief.

Builds a structured plain-text context block per ``EventWithLinks`` —
no fresh retrieval beyond what :mod:`worker.query.itinerary` already
returned.  The six blocks (header, description, tasks, notes, thread,
attachments) are the deterministic input to the LLM call in
:mod:`worker.cli.itinerary_brief_prompt`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from neo4j import Driver

from worker.query.itinerary import (
    EventWithLinks,
    NoteHit,
    OpenTask,
    PersonRef,
    ThreadHit,
)


_DESCRIPTION_CAP = 600
_NOTE_SNIPPET_CAP = 200
_THREAD_SNIPPET_CAP = 200
_MAX_TASKS = 2
_MAX_NOTES = 2
_MAX_ATTACHMENTS = 3
_MAX_THREAD_MESSAGES = 3


@dataclass
class ThreadMessage:
    """One row in a thread's recent-message tail."""

    sender: str | None
    ts: str
    snippet: str


@dataclass
class EventBrief:
    """Structured per-event pre-brief consumed by the LLM prompt builder."""

    title: str
    start_ts: str
    end_ts: str
    location: str | None = None
    organizer: str | None = None
    attendees: list[str] = field(default_factory=list)
    description: str | None = None
    tasks: list[OpenTask] = field(default_factory=list)
    notes: list[NoteHit] = field(default_factory=list)
    thread: ThreadHit | None = None
    thread_messages: list[ThreadMessage] = field(default_factory=list)
    attachments: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Sub-queries
# ---------------------------------------------------------------------------


_EMAIL_THREAD_TAIL_CYPHER = """\
MATCH (em:Email)-[:PART_OF]->(t:Thread {source_id: $sid})
OPTIONAL MATCH (sender:Person)-[:SENT]->(em)
RETURN coalesce(sender.name, sender.email) AS sender,
       em.date AS ts,
       coalesce(em.subject, '') AS snippet
ORDER BY em.date DESC
LIMIT $k
"""


_SLACK_TAIL_CYPHER = """\
MATCH (s:SlackMessage {source_id: $sid})
OPTIONAL MATCH (s)-[:SENT_BY]->(p:Person)
RETURN coalesce(p.name, p.email) AS sender,
       coalesce(s.last_ts, s.first_ts, '') AS ts,
       coalesce(s.text_preview, s.channel_name, '') AS snippet
LIMIT $k
"""


def _fetch_thread_tail(
    driver: Driver, hit: ThreadHit, k: int = _MAX_THREAD_MESSAGES
) -> list[ThreadMessage]:
    """Fetch up to *k* recent messages for an Email thread or Slack window."""
    cypher = _EMAIL_THREAD_TAIL_CYPHER if hit.kind == "email" else _SLACK_TAIL_CYPHER
    try:
        with driver.session() as session:
            rows = session.execute_read(
                lambda tx: tx.run(cypher, sid=hit.source_id, k=int(k)).data()
            )
    except Exception:  # noqa: BLE001 - thread tail is best-effort context
        return []
    out: list[ThreadMessage] = []
    for r in rows:
        snippet = str(r.get("snippet") or "")
        if len(snippet) > _THREAD_SNIPPET_CAP:
            snippet = snippet[:_THREAD_SNIPPET_CAP].rstrip() + "…"
        out.append(
            ThreadMessage(
                sender=r.get("sender"),
                ts=str(r.get("ts") or ""),
                snippet=snippet,
            )
        )
    return out


_ATTACHMENTS_CYPHER = """\
MATCH (a:Attachment)-[:ATTACHED_TO]->(e:CalendarEvent)
WHERE id(e) = $eid
RETURN coalesce(a.title, a.filename, a.file_id, a.source_id) AS name
LIMIT $k
"""


def _fetch_attachments(
    driver: Driver, event_id: int, k: int = _MAX_ATTACHMENTS
) -> list[str]:
    try:
        with driver.session() as session:
            rows = session.execute_read(
                lambda tx: tx.run(_ATTACHMENTS_CYPHER, eid=event_id, k=int(k)).data()
            )
    except Exception:  # noqa: BLE001 - attachments are best-effort context
        return []
    return [str(r.get("name")) for r in rows if r.get("name")]


# ---------------------------------------------------------------------------
# Public assembler + formatter
# ---------------------------------------------------------------------------


def _format_attendee(ref: PersonRef) -> str:
    if ref.name and ref.email:
        return f"{ref.name} <{ref.email}>"
    return ref.name or ref.email or "(unknown)"


def assemble_event_brief(
    ew: EventWithLinks,
    *,
    driver: Driver,
) -> EventBrief:
    """Build the structured pre-brief for one event.

    Reuses the tasks/notes/thread already attached to *ew*; only fetches
    additional context (recent thread messages, event attachments) that
    isn't part of :class:`EventWithLinks`.
    """
    ev = ew.event
    organizer = _format_attendee(ev.organizer) if ev.organizer else None
    attendees = [_format_attendee(a) for a in ev.attendees]

    description = ev.description
    if description and len(description) > _DESCRIPTION_CAP:
        description = description[:_DESCRIPTION_CAP].rstrip() + "…"

    thread_msgs: list[ThreadMessage] = []
    if ew.thread is not None:
        thread_msgs = _fetch_thread_tail(driver, ew.thread)

    attachments = _fetch_attachments(driver, ev.id)

    return EventBrief(
        title=ev.title or "(untitled)",
        start_ts=ev.start_ts,
        end_ts=ev.end_ts,
        location=ev.location,
        organizer=organizer,
        attendees=attendees,
        description=description,
        tasks=list(ew.tasks[:_MAX_TASKS]),
        notes=list(ew.notes[:_MAX_NOTES]),
        thread=ew.thread,
        thread_messages=thread_msgs,
        attachments=attachments[:_MAX_ATTACHMENTS],
    )


def format_event_brief(brief: EventBrief) -> str:
    """Render *brief* as the deterministic plain-text user message.

    Six blocks; an empty block still renders so the LLM sees the schema.
    """
    lines: list[str] = []

    # 1. Event header
    lines.append("[Calendar event]")
    lines.append(f"Title: {brief.title}")
    if brief.start_ts or brief.end_ts:
        when = brief.start_ts or "?"
        if brief.end_ts and brief.end_ts != brief.start_ts:
            when += f" → {brief.end_ts}"
        lines.append(f"When: {when}")
    if brief.location:
        lines.append(f"Where: {brief.location}")
    if brief.organizer:
        lines.append(f"Organizer: {brief.organizer}")
    if brief.attendees:
        lines.append(f"Attendees: {', '.join(brief.attendees)}")
    lines.append("")

    # 2. Event description
    lines.append("[Event description]")
    if brief.description:
        lines.append(brief.description.strip())
    else:
        lines.append("(none)")
    lines.append("")

    # 3. Linked tasks
    lines.append(f"[Linked OmniFocus tasks ({len(brief.tasks)})]")
    if brief.tasks:
        for t in brief.tasks:
            bits = [f"- {t.title}"]
            if t.project:
                bits.append(f"(project: {t.project})")
            if t.tags:
                bits.append(f"[tags: {', '.join(t.tags)}]")
            if t.due:
                bits.append(f"due {t.due}")
            if t.flagged:
                bits.append("[flagged]")
            lines.append(" ".join(bits))
    else:
        lines.append("(none)")
    lines.append("")

    # 4. Linked notes
    lines.append(f"[Linked notes ({len(brief.notes)})]")
    if brief.notes:
        for n in brief.notes:
            snippet = (n.snippet or "").replace("\n", " ").strip()
            if len(snippet) > _NOTE_SNIPPET_CAP:
                snippet = snippet[:_NOTE_SNIPPET_CAP].rstrip() + "…"
            suffix = f" — {snippet}" if snippet else ""
            lines.append(f"- {n.source_id}{suffix}")
    else:
        lines.append("(none)")
    lines.append("")

    # 5. Linked thread (max 1)
    lines.append("[Linked thread]")
    if brief.thread is not None:
        lines.append(
            f"{brief.thread.kind}: {brief.thread.title} "
            f"(last_ts {brief.thread.last_ts})"
        )
        for m in brief.thread_messages:
            sender = m.sender or "(unknown)"
            snippet = (m.snippet or "").replace("\n", " ").strip()
            line = f"- {sender} @ {m.ts}"
            if snippet:
                line += f": {snippet}"
            lines.append(line)
    else:
        lines.append("(none)")
    lines.append("")

    # 6. Attachments on the event itself
    lines.append(f"[Attachments ({len(brief.attachments)})]")
    if brief.attachments:
        for name in brief.attachments:
            lines.append(f"- {name}")
    else:
        lines.append("(none)")

    return "\n".join(lines).rstrip() + "\n"
