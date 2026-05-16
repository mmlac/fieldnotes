"""Google Calendar event parser.

Strips HTML from event descriptions, extracts structured metadata, and
produces GraphHints for Person/CalendarEvent relationships — enabling
cross-source linking with Gmail (shared Person nodes via email) and
Obsidian (via entity resolution on person names).
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from bs4 import BeautifulSoup

from .attachments import (
    AttachmentDownloadError,
    AttachmentParseError,
    build_parent_url,
    classify_attachment,
    stream_and_parse,
)
from ._safe_filename import sanitize_for_inline
from .base import BaseParser, GraphHint, ParsedDocument, canonicalize_email, extract_source_link_hints
from .registry import register

logger = logging.getLogger(__name__)

_MAX_HTML_SIZE = 10 * 1024 * 1024  # 10 MiB
_MAX_ATTENDEES = 200  # prevent graph explosion for all-hands meetings


# Module-level seam so tests can substitute a fake Drive fetch closure
# without standing up real OAuth credentials.  In production this builds
# a closure that pulls the bytes via ``drive.files.get_media`` on a Drive
# service constructed from the per-account Calendar token (which has the
# ``drive.readonly`` scope when ``download_attachments=True``).
def _build_drive_fetcher(account: str, file_id: str) -> Callable[[], bytes]:
    """Return a zero-arg closure that downloads *file_id* from Drive.

    Lazily imports the Google client libs so the parser still loads
    cleanly when only the Calendar parser is used (no Drive access).
    """

    def fetch() -> bytes:
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseDownload

        from worker.sources.calendar_auth import (
            get_credentials,
            token_path_for_account,
        )

        # We rely on the persisted token, which must already carry
        # drive.readonly because the source ran the install flow with
        # download_attachments=True.  Passing a non-existent secrets
        # path would only matter if we needed to run a fresh OAuth
        # flow, which by construction we do not.
        token_path = token_path_for_account(account)
        creds = get_credentials(
            token_path,  # unused when token already valid
            account,
            download_attachments=True,
        )
        drive = build("drive", "v3", credentials=creds)

        import io

        request = drive.files().get_media(fileId=file_id)
        buf = io.BytesIO()
        downloader = MediaIoBaseDownload(buf, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        return buf.getvalue()

    return fetch


def _format_size(size_bytes: int) -> str:
    """Render an attachment size as a short human-friendly string."""
    if size_bytes <= 0:
        return "size unknown"
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


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
        if description and ("<" in description and ">" in description):
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
            series_uri = f"google-calendar://{account}/series/{recurring_event_id}"
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
                obj_id = f"google-calendar://{account}/event/{event_id}/attendee/{idx}"
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
        if creator_email and canonicalize_email(creator_email) != canonicalize_email(
            organizer_email
        ):
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

        # --- Attachments (Drive links surfaced by the source) ---
        attachments_meta: list[dict[str, Any]] = meta.get("attachments", []) or []
        download_attachments: bool = bool(meta.get("download_attachments", False))
        indexable: list[str] = list(meta.get("attachment_indexable_mimetypes", []))
        max_size_mb: int = int(meta.get("attachment_max_size_mb", 25))
        pdf_max_pages: int = int(meta.get("attachment_pdf_max_pages", 1000))
        pdf_per_page_chars: int = int(
            meta.get("attachment_pdf_per_page_chars", 1_000_000)
        )
        pdf_timeout_seconds: int = int(meta.get("attachment_pdf_timeout_seconds", 60))

        if attachments_meta:
            body = self._augment_text_with_attachments(body, attachments_meta)

        attachment_docs, attachment_counters = self._build_attachment_documents(
            attachments_meta=attachments_meta,
            event_source_id=source_id,
            event_id=event_id,
            account=account,
            html_link=html_link,
            edge_props=edge_props,
            download_attachments=download_attachments,
            indexable=indexable,
            max_size_mb=max_size_mb,
            pdf_max_pages=pdf_max_pages,
            pdf_per_page_chars=pdf_per_page_chars,
            pdf_timeout_seconds=pdf_timeout_seconds,
        )

        if attachment_counters["intended"]:
            node_props["attachments_count_intended"] = attachment_counters["intended"]
            node_props["attachments_count_indexed"] = attachment_counters["indexed"]
            node_props["attachments_count_metadata_only"] = attachment_counters[
                "metadata_only"
            ]
            # Deprecated alias of attachments_count_intended; retained for one
            # release so existing Cypher queries keep working.
            node_props["has_attachments"] = attachment_counters["intended"]

        link_hints = extract_source_link_hints(description, source_id, "CalendarEvent")
        if link_hints:
            graph_hints.extend(link_hints)

        event_doc = ParsedDocument(
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
        return [event_doc, *attachment_docs]

    @staticmethod
    def _augment_text_with_attachments(
        body: str,
        attachments: list[dict[str, Any]],
    ) -> str:
        """Append a human-readable 'Attachments:' section to *body*.

        The section ships into the chunker so a search for 'budget.pdf'
        will land on the parent event even when the attachment itself is
        metadata-only (no body chunks of its own).
        """
        lines = ["Attachments:"]
        for att in attachments:
            raw_title = att.get("title") or "(untitled)"
            title = sanitize_for_inline(str(raw_title))
            if not title:
                title = "(untitled)"
            mime = att.get("mime_type", "")
            size = _format_size(int(att.get("size_bytes", 0) or 0))
            lines.append(f"- {title} ({mime}, {size})")
        section = "\n".join(lines)
        return f"{body}\n\n{section}" if body else section

    def _build_attachment_documents(
        self,
        *,
        attachments_meta: list[dict[str, Any]],
        event_source_id: str,
        event_id: str,
        account: str,
        html_link: str,
        edge_props: dict[str, Any],
        download_attachments: bool,
        indexable: list[str],
        max_size_mb: int,
        pdf_max_pages: int = 1000,
        pdf_per_page_chars: int = 1_000_000,
        pdf_timeout_seconds: int = 60,
    ) -> tuple[list[ParsedDocument], dict[str, int]]:
        """Emit one ParsedDocument per attachment.

        Each Attachment is linked to the parent CalendarEvent via an
        ATTACHED_TO graph hint.  ORGANIZED_BY / ATTENDED_BY edges that
        the parent event already carries are deliberately NOT propagated
        — the graph keeps them on the event only.

        Returns a tuple of (documents, counters) where counters carries
        ``intended`` (total attachments seen with a valid file_id),
        ``indexed`` (successful download+parse) and ``metadata_only``
        (fallback or non-indexable).
        """
        empty_counters = {"intended": 0, "indexed": 0, "metadata_only": 0}
        if not attachments_meta:
            return [], empty_counters

        parent_url = ""
        if html_link:
            try:
                parent_url = build_parent_url("calendar", html_link=html_link)
            except ValueError:
                parent_url = ""

        out: list[ParsedDocument] = []
        intended_count = 0
        indexed_count = 0
        for att in attachments_meta:
            file_id = att.get("file_id", "")
            if not file_id:
                continue
            intended_count += 1
            mime = att.get("mime_type", "")
            title = att.get("title", "")
            size_bytes = int(att.get("size_bytes", 0) or 0)

            att_source_id = (
                f"google-calendar://{account}/event/{event_id}/attachment/{file_id}"
            )

            # Unknown size + an explicit max bound = treat as too-large so
            # we don't accidentally pull a multi-GB file by guessing wrong.
            decision = (
                classify_attachment(
                    mime=mime,
                    size_bytes=size_bytes
                    if size_bytes > 0
                    else (max_size_mb * 1024 * 1024 + 1),
                    indexable=indexable,
                    max_size_mb=max_size_mb,
                )
                if download_attachments
                else "metadata_only"
            )

            text = ""
            description = ""
            extra_hints: list[GraphHint] = []
            if decision == "download_and_index":
                fetch = _build_drive_fetcher(account, file_id)
                try:
                    parsed = stream_and_parse(
                        fetch=fetch,
                        filename=title or file_id,
                        mime=mime,
                        source_id=att_source_id,
                        pdf_max_pages=pdf_max_pages,
                        pdf_per_page_chars=pdf_per_page_chars,
                        pdf_timeout_seconds=pdf_timeout_seconds,
                    )
                    text = parsed.text
                    description = parsed.description
                    extra_hints.extend(parsed.extracted_entities)
                    indexed_count += 1
                except (AttachmentDownloadError, AttachmentParseError) as exc:
                    # Drive 404s, 403s, vision failures, etc. — log and
                    # downgrade to metadata-only so the event itself
                    # still indexes cleanly.
                    logger.info(
                        "Calendar attachment %s/%s fell back to metadata-only: %s",
                        event_id,
                        file_id,
                        exc,
                    )
                    decision = "metadata_only"

            if decision == "metadata_only" and not description:
                description = (
                    f"Calendar attachment '{title}' ({mime}) on event {event_id}"
                )

            node_props: dict[str, Any] = {
                "title": title,
                "mime_type": mime,
                "file_id": file_id,
                "decision": decision,
                "account": account,
            }
            if size_bytes > 0:
                node_props["size_bytes"] = size_bytes
            if att.get("file_url"):
                node_props["file_url"] = att["file_url"]
            if att.get("icon_link"):
                node_props["icon_link"] = att["icon_link"]
            if parent_url:
                node_props["parent_url"] = parent_url
            if description:
                node_props["description"] = description

            attachment_hints: list[GraphHint] = [
                GraphHint(
                    subject_id=att_source_id,
                    subject_label="Attachment",
                    predicate="ATTACHED_TO",
                    object_id=event_source_id,
                    object_label="CalendarEvent",
                    object_merge_key="source_id",
                    edge_props=edge_props,
                    confidence=1.0,
                ),
                *extra_hints,
            ]

            source_metadata: dict[str, Any] = {
                "source_type": "calendar_attachment",
                "account": account,
                "event_id": event_id,
                "file_id": file_id,
                "mime_type": mime,
                "decision": decision,
            }
            if parent_url:
                source_metadata["parent_url"] = parent_url

            out.append(
                ParsedDocument(
                    source_type=self.source_type,
                    source_id=att_source_id,
                    operation="created",
                    text=text or description,
                    mime_type="text/plain",
                    node_label="Attachment",
                    node_props=node_props,
                    graph_hints=attachment_hints,
                    source_metadata=source_metadata,
                )
            )
        counters = {
            "intended": intended_count,
            "indexed": indexed_count,
            "metadata_only": intended_count - indexed_count,
        }
        return out, counters
