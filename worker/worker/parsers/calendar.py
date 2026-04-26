"""Google Calendar event parser.

Strips HTML from event descriptions, extracts structured metadata, and
produces GraphHints for Person/CalendarEvent relationships — enabling
cross-source linking with Gmail (shared Person nodes via email) and
Obsidian (via entity resolution on person names).
"""

from __future__ import annotations

import logging
from typing import Any

from bs4 import BeautifulSoup

from .base import BaseParser, GraphHint, ParsedDocument, canonicalize_email
from .registry import register

logger = logging.getLogger(__name__)

_MAX_HTML_SIZE = 10 * 1024 * 1024  # 10 MiB
_MAX_ATTENDEES = 200  # prevent graph explosion for all-hands meetings


def _strip_html(html: str) -> str:
    """Strip HTML tags and return plain text."""
    soup = BeautifulSoup(html, "html.parser")
    for element in soup(["script", "style"]):
        element.decompose()
    text = soup.get_text(separator="\n")
    lines = (line.strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


@register
class GoogleCalendarParser(BaseParser):
    """Parses Google Calendar IngestEvents into ParsedDocuments with graph hints.

    Creates Person nodes for attendees and the organizer, linked via email
    addresses.  Because Gmail creates Person nodes with the same
    ``email`` merge key, calendar attendees automatically connect to
    email correspondents in the knowledge graph.
    """

    @property
    def source_type(self) -> str:
        return "google_calendar"

    def parse(self, event: dict[str, Any]) -> list[ParsedDocument]:
        source_id: str = event["source_id"]
        operation: str = event.get("operation", "created")
        meta: dict[str, Any] = event.get("meta", {})

        if operation == "deleted":
            return [
                ParsedDocument(
                    source_type=self.source_type,
                    source_id=source_id,
                    operation="deleted",
                    text="",
                )
            ]

        account: str = meta.get("account", "")
        event_id: str = meta.get("event_id", "")
        summary: str = meta.get("summary", "(No title)")
        description: str = meta.get("description", "")
        location: str = meta.get("location", "")
        start_time: str = meta.get("start_time", "")
        end_time: str = meta.get("end_time", "")
        organizer_email: str = meta.get("organizer_email", "")
        organizer_name: str = meta.get("organizer_name", "")
        creator_email: str = meta.get("creator_email", "")
        attendees: list[dict[str, Any]] = meta.get("attendees", [])
        calendar_id: str = meta.get("calendar_id", "")
        html_link: str = meta.get("html_link", "")
        recurring_event_id: str = meta.get("recurring_event_id", "")
        status: str = meta.get("status", "confirmed")

        # All edges leaving this CalendarEvent/Series carry the account so
        # the graph can answer "show me attendees of work-only events".
        edge_props: dict[str, Any] = {"account": account} if account else {}

        # --- Clean description (may contain HTML) ---
        body = event.get("text", "")
        if description and (
            "<" in description and ">" in description
        ):
            if len(description) <= _MAX_HTML_SIZE:
                description = _strip_html(description)

        # --- Node properties ---
        node_props: dict[str, Any] = {
            "summary": summary,
            "start_time": start_time,
            "end_time": end_time,
            "status": status,
        }
        if account:
            node_props["account"] = account
        if location:
            node_props["location"] = location
        if html_link:
            node_props["html_link"] = html_link
        if recurring_event_id:
            node_props["recurring_event_id"] = recurring_event_id
        if calendar_id:
            node_props["calendar_id"] = calendar_id

        # --- GraphHints: Person ↔ CalendarEvent relationships ---
        graph_hints: list[GraphHint] = []

        # For recurring events, create a CalendarSeries node and link
        # person relationships to the series instead of each instance.
        # This prevents N×M edge explosion (N instances × M attendees).
        if recurring_event_id:
            # CalendarSeries node is account-namespaced for the same reason
            # CalendarEvent is: Google reuses series IDs across calendars.
            series_uri = (
                f"google-calendar://{account}/series/{recurring_event_id}"
            )
            series_props: dict[str, Any] = {
                "series_id": recurring_event_id,
                "summary": summary,
            }
            if account:
                series_props["account"] = account
            if calendar_id:
                series_props["calendar_id"] = calendar_id

            # INSTANCE_OF: CalendarEvent → CalendarSeries
            graph_hints.append(
                GraphHint(
                    subject_id=source_id,
                    subject_label="CalendarEvent",
                    predicate="INSTANCE_OF",
                    object_id=series_uri,
                    object_label="CalendarSeries",
                    object_props=series_props,
                    object_merge_key="source_id",
                    edge_props=edge_props,
                    confidence=1.0,
                )
            )

        # Determine the anchor node for person relationships:
        # series node for recurring events, event node otherwise.
        if recurring_event_id:
            person_anchor_id = series_uri
            person_anchor_label = "CalendarSeries"
            person_anchor_merge_key = "source_id"
        else:
            person_anchor_id = source_id
            person_anchor_label = "CalendarEvent"
            person_anchor_merge_key = "source_id"

        # ORGANIZED_BY: anchor → Person (organizer).  Person merges on
        # email so cross-account organizers (alice@... seen by both work
        # and personal calendars) collapse into one node.
        if organizer_email:
            norm_org = canonicalize_email(organizer_email)
            org_props: dict[str, Any] = {"email": norm_org}
            if organizer_name:
                org_props["name"] = organizer_name
            graph_hints.append(
                GraphHint(
                    subject_id=person_anchor_id,
                    subject_label=person_anchor_label,
                    subject_merge_key=person_anchor_merge_key,
                    predicate="ORGANIZED_BY",
                    object_id=f"person:{norm_org}",
                    object_label="Person",
                    object_props=org_props,
                    object_merge_key="email",
                    edge_props=edge_props,
                    confidence=1.0,
                )
            )

        # ATTENDED_BY: anchor → Person (each attendee)
        if len(attendees) > _MAX_ATTENDEES:
            logger.warning(
                "Event %s has %d attendees, truncating to %d",
                source_id,
                len(attendees),
                _MAX_ATTENDEES,
            )
            attendees = attendees[:_MAX_ATTENDEES]

        for idx, att in enumerate(attendees):
            if att.get("self", False):
                continue
            att_email = att.get("email", "")
            att_name = att.get("name", "")

            if att_email:
                # Email known: Person merges on email across accounts.
                norm_att = canonicalize_email(att_email)
                att_props: dict[str, Any] = {"email": norm_att}
                if att_name:
                    att_props["name"] = att_name
                obj_id = f"person:{norm_att}"
                obj_merge_key = "email"
            else:
                # Email unknown (display-name-only attendee).  Fall back to
                # an account+event-scoped Person ID so the same display name
                # in two accounts produces two Person nodes — they merge
                # later via reconcile once the email becomes known.
                if not att_name:
                    continue
                obj_id = (
                    f"google-calendar://{account}/event/{event_id}"
                    f"/attendee/{idx}"
                )
                att_props = {"name": att_name, "account": account}
                obj_merge_key = "source_id"

            graph_hints.append(
                GraphHint(
                    subject_id=person_anchor_id,
                    subject_label=person_anchor_label,
                    subject_merge_key=person_anchor_merge_key,
                    predicate="ATTENDED_BY",
                    object_id=obj_id,
                    object_label="Person",
                    object_props=att_props,
                    object_merge_key=obj_merge_key,
                    edge_props=edge_props,
                    confidence=1.0,
                )
            )

        # CREATED_BY: anchor → Person (creator, if different from organizer)
        if creator_email and canonicalize_email(creator_email) != canonicalize_email(organizer_email):
            norm_creator = canonicalize_email(creator_email)
            graph_hints.append(
                GraphHint(
                    subject_id=person_anchor_id,
                    subject_label=person_anchor_label,
                    subject_merge_key=person_anchor_merge_key,
                    predicate="CREATED_BY",
                    object_id=f"person:{norm_creator}",
                    object_label="Person",
                    object_props={"email": norm_creator},
                    object_merge_key="email",
                    edge_props=edge_props,
                    confidence=1.0,
                )
            )

        return [
            ParsedDocument(
                source_type=self.source_type,
                source_id=source_id,
                operation=operation,
                text=body,
                mime_type="text/plain",
                node_label="CalendarEvent",
                node_props=node_props,
                graph_hints=graph_hints,
                source_metadata={
                    "source_type": "calendar",
                    "calendar_id": calendar_id,
                    "start_time": start_time,
                    "account": account,
                },
            )
        ]
