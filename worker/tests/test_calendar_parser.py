"""Tests for the Google Calendar parser."""

from unittest.mock import patch

import worker.parsers.calendar as calendar_parser_module
from worker.parsers.attachments import (
    AttachmentDownloadError,
    AttachmentParseError,
    ParsedAttachment,
)
from worker.parsers.calendar import GoogleCalendarParser, _strip_html


def _make_event(
    text: str = "Team standup\n\nLocation: Conference Room A\n\nDaily sync meeting",
    account: str = "personal",
    **overrides,
) -> dict:
    meta = overrides.pop("meta", {})
    base_meta = {
        "event_id": "evt-123",
        "calendar_id": f"{account}/primary",
        "account": account,
        "summary": "Team standup",
        "description": "Daily sync meeting",
        "location": "Conference Room A",
        "start_time": "2026-03-21T09:00:00-07:00",
        "end_time": "2026-03-21T09:30:00-07:00",
        "organizer_email": "alice@example.com",
        "organizer_name": "Alice Smith",
        "creator_email": "alice@example.com",
        "attendees": [
            {
                "email": "bob@example.com",
                "name": "Bob Jones",
                "response": "accepted",
                "self": False,
            },
            {
                "email": "carol@example.com",
                "name": "Carol Lee",
                "response": "tentative",
                "self": False,
            },
            {
                "email": "me@example.com",
                "name": "Me",
                "response": "accepted",
                "self": True,
            },
        ],
        "html_link": "https://calendar.google.com/calendar/event?eid=evt123",
        "recurring_event_id": "",
        "status": "confirmed",
    }
    base_meta.update(meta)
    base = {
        "source_type": "google_calendar",
        "source_id": (
            f"google-calendar://{base_meta['account']}/event/{base_meta['event_id']}"
        ),
        "operation": "created",
        "text": text,
        "mime_type": "text/plain",
        "meta": base_meta,
    }
    base.update(overrides)
    return base


class TestStripHtml:
    def test_plain_text_passthrough(self):
        assert _strip_html("Hello world") == "Hello world"

    def test_basic_tags(self):
        assert _strip_html("<p>Hello</p><p>World</p>") == "Hello\nWorld"

    def test_script_removed(self):
        html = "<script>alert(1)</script><p>Content</p>"
        assert _strip_html(html) == "Content"

    def test_empty_string(self):
        assert _strip_html("") == ""


class TestGoogleCalendarParser:
    def setup_method(self):
        self.parser = GoogleCalendarParser()

    def test_source_type(self):
        assert self.parser.source_type == "google_calendar"

    def test_basic_parse(self):
        docs = self.parser.parse(_make_event())
        assert len(docs) == 1
        doc = docs[0]
        assert doc.source_type == "google_calendar"
        assert doc.source_id == "google-calendar://personal/event/evt-123"
        assert doc.operation == "created"
        assert doc.node_label == "CalendarEvent"

    def test_node_props(self):
        docs = self.parser.parse(_make_event())
        props = docs[0].node_props
        assert props["summary"] == "Team standup"
        assert props["start_time"] == "2026-03-21T09:00:00-07:00"
        assert props["end_time"] == "2026-03-21T09:30:00-07:00"
        assert props["location"] == "Conference Room A"
        assert props["status"] == "confirmed"
        assert props["account"] == "personal"
        assert props["calendar_id"] == "personal/primary"
        assert "html_link" in props

    def test_source_metadata(self):
        docs = self.parser.parse(_make_event())
        meta = docs[0].source_metadata
        assert meta["source_type"] == "calendar"
        assert meta["calendar_id"] == "personal/primary"
        assert meta["start_time"] == "2026-03-21T09:00:00-07:00"
        assert meta["account"] == "personal"

    def test_organizer_hint_non_recurring(self):
        """Non-recurring events attach person hints to CalendarEvent."""
        docs = self.parser.parse(_make_event())
        org_hints = [h for h in docs[0].graph_hints if h.predicate == "ORGANIZED_BY"]
        assert len(org_hints) == 1
        h = org_hints[0]
        assert h.subject_id == "google-calendar://personal/event/evt-123"
        assert h.subject_label == "CalendarEvent"
        assert h.object_id == "person:alice@example.com"
        assert h.object_label == "Person"
        assert h.object_props["email"] == "alice@example.com"
        assert h.object_props["name"] == "Alice Smith"
        assert h.object_merge_key == "email"
        assert h.edge_props == {"account": "personal"}
        assert h.confidence == 1.0

    def test_recurring_creates_series_node(self):
        """Recurring events create a CalendarSeries node with INSTANCE_OF."""
        event = _make_event(meta={"recurring_event_id": "series-abc"})
        docs = self.parser.parse(event)
        instance_hints = [
            h for h in docs[0].graph_hints if h.predicate == "INSTANCE_OF"
        ]
        assert len(instance_hints) == 1
        h = instance_hints[0]
        assert h.subject_id == "google-calendar://personal/event/evt-123"
        assert h.subject_label == "CalendarEvent"
        assert h.object_id == "google-calendar://personal/series/series-abc"
        assert h.object_label == "CalendarSeries"
        assert h.object_merge_key == "source_id"
        assert h.object_props["series_id"] == "series-abc"
        assert h.object_props["summary"] == "Team standup"
        assert h.object_props["account"] == "personal"
        assert h.edge_props == {"account": "personal"}

    def test_recurring_person_hints_on_series(self):
        """Recurring events attach person hints to CalendarSeries, not instance."""
        event = _make_event(meta={"recurring_event_id": "series-abc"})
        docs = self.parser.parse(event)
        org_hints = [h for h in docs[0].graph_hints if h.predicate == "ORGANIZED_BY"]
        assert len(org_hints) == 1
        h = org_hints[0]
        assert h.subject_id == "google-calendar://personal/series/series-abc"
        assert h.subject_label == "CalendarSeries"
        assert h.subject_merge_key == "source_id"
        assert h.object_props["email"] == "alice@example.com"

        att_hints = [h for h in docs[0].graph_hints if h.predicate == "ATTENDED_BY"]
        for ah in att_hints:
            assert ah.subject_label == "CalendarSeries"
            assert ah.subject_merge_key == "source_id"

    def test_attendee_hints_exclude_self(self):
        """The 'self' attendee (me@example.com) should not generate a hint."""
        docs = self.parser.parse(_make_event())
        att_hints = [h for h in docs[0].graph_hints if h.predicate == "ATTENDED_BY"]
        # 3 attendees total, but self=True is excluded → 2 hints
        assert len(att_hints) == 2
        attendee_ids = {h.object_id for h in att_hints}
        assert "person:bob@example.com" in attendee_ids
        assert "person:carol@example.com" in attendee_ids
        assert "person:me@example.com" not in attendee_ids

    def test_attendee_hint_properties(self):
        docs = self.parser.parse(_make_event())
        att_hints = [h for h in docs[0].graph_hints if h.predicate == "ATTENDED_BY"]
        bob_hint = next(h for h in att_hints if "bob" in h.object_id)
        assert bob_hint.subject_id == "google-calendar://personal/event/evt-123"
        assert bob_hint.subject_label == "CalendarEvent"
        assert bob_hint.object_label == "Person"
        assert bob_hint.object_props["email"] == "bob@example.com"
        assert bob_hint.object_props["name"] == "Bob Jones"
        assert bob_hint.object_merge_key == "email"
        assert bob_hint.edge_props == {"account": "personal"}

    def test_cross_source_person_linking_with_gmail(self):
        """Calendar attendees use the SAME Person merge key (email) as Gmail.

        This means person:bob@example.com from Calendar will MERGE with
        person:bob@example.com from Gmail, creating cross-source links.
        """
        docs = self.parser.parse(_make_event())
        att_hints = [h for h in docs[0].graph_hints if h.predicate == "ATTENDED_BY"]
        for h in att_hints:
            # Same pattern Gmail uses for Person nodes
            assert h.object_merge_key == "email"
            assert h.object_id.startswith("person:")
            assert h.object_label == "Person"

    def test_no_created_by_when_same_as_organizer(self):
        """CREATED_BY hint should be omitted when creator == organizer."""
        docs = self.parser.parse(_make_event())
        created_hints = [h for h in docs[0].graph_hints if h.predicate == "CREATED_BY"]
        assert len(created_hints) == 0

    def test_created_by_when_different_from_organizer(self):
        event = _make_event(
            meta={
                "organizer_email": "alice@example.com",
                "creator_email": "delegate@example.com",
            }
        )
        docs = self.parser.parse(event)
        created_hints = [h for h in docs[0].graph_hints if h.predicate == "CREATED_BY"]
        assert len(created_hints) == 1
        assert created_hints[0].object_id == "person:delegate@example.com"
        assert created_hints[0].edge_props == {"account": "personal"}

    def test_deleted_operation(self):
        docs = self.parser.parse(_make_event(operation="deleted"))
        assert len(docs) == 1
        assert docs[0].operation == "deleted"
        assert docs[0].text == ""
        assert docs[0].graph_hints == []

    def test_no_organizer_no_hint(self):
        event = _make_event(meta={"organizer_email": ""})
        docs = self.parser.parse(event)
        org_hints = [h for h in docs[0].graph_hints if h.predicate == "ORGANIZED_BY"]
        assert len(org_hints) == 0

    def test_no_attendees_no_hints(self):
        event = _make_event(meta={"attendees": []})
        docs = self.parser.parse(event)
        att_hints = [h for h in docs[0].graph_hints if h.predicate == "ATTENDED_BY"]
        assert len(att_hints) == 0

    def test_missing_meta_fields(self):
        """Parser should handle missing meta fields gracefully."""
        event = {
            "source_type": "google_calendar",
            "source_id": "google-calendar:///event/bare",
            "operation": "created",
            "text": "minimal event",
            "mime_type": "text/plain",
            "meta": {},
        }
        docs = self.parser.parse(event)
        assert len(docs) == 1
        assert docs[0].node_props["summary"] == "(No title)"

    def test_no_location_omitted_from_props(self):
        event = _make_event(meta={"location": ""})
        docs = self.parser.parse(event)
        assert "location" not in docs[0].node_props

    def test_recurring_event_id_in_props(self):
        event = _make_event(meta={"recurring_event_id": "recurse-abc"})
        docs = self.parser.parse(event)
        assert docs[0].node_props["recurring_event_id"] == "recurse-abc"

    def test_attendee_truncation(self):
        """More than _MAX_ATTENDEES should be truncated."""
        many_attendees = [
            {
                "email": f"user{i}@example.com",
                "name": f"User {i}",
                "response": "accepted",
                "self": False,
            }
            for i in range(250)
        ]
        event = _make_event(meta={"attendees": many_attendees})
        docs = self.parser.parse(event)
        att_hints = [h for h in docs[0].graph_hints if h.predicate == "ATTENDED_BY"]
        assert len(att_hints) == 200  # _MAX_ATTENDEES

    def test_registry_registration(self):
        from worker.parsers.registry import get

        parser = get("google_calendar")
        assert isinstance(parser, GoogleCalendarParser)


def _attachment(
    file_id: str,
    title: str,
    mime: str,
    size_bytes: int = 0,
) -> dict:
    return {
        "title": title,
        "mime_type": mime,
        "file_id": file_id,
        "file_url": f"https://drive.google.com/file/d/{file_id}",
        "icon_link": f"https://example.com/icons/{mime.replace('/', '-')}.png",
        "size_bytes": size_bytes,
    }


_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


class TestCalendarAttachmentsMetadataOnly:
    """download_attachments=False keeps every attachment metadata-only."""

    def setup_method(self):
        self.parser = GoogleCalendarParser()

    def _three_attachment_event(self) -> dict:
        return _make_event(
            meta={
                "download_attachments": False,
                "attachment_indexable_mimetypes": [
                    "application/pdf",
                    "image/png",
                ],
                "attachment_max_size_mb": 25,
                "attachments": [
                    _attachment("drv-pdf", "report.pdf", "application/pdf"),
                    _attachment("drv-img", "photo.png", "image/png"),
                    _attachment("drv-doc", "spec.docx", _DOCX_MIME),
                ],
            }
        )

    def test_emits_three_metadata_only_documents(self):
        """No Drive calls are made; sizes stay unknown but every attachment
        still surfaces as its own Document so search can find them."""

        # Patch the Drive fetcher seam to a sentinel that explodes when
        # called.  download_attachments=False must never reach this code path.
        def boom(_account, _file_id):
            raise AssertionError("Drive fetcher must not be invoked")

        with patch.object(calendar_parser_module, "_build_drive_fetcher", boom):
            docs = self.parser.parse(self._three_attachment_event())

        # 1 event Document + 3 attachment Documents
        assert len(docs) == 4
        attachment_docs = [d for d in docs if d.node_label == "Attachment"]
        assert len(attachment_docs) == 3
        for d in attachment_docs:
            assert d.node_props["decision"] == "metadata_only"

    def test_event_text_contains_attachments_section(self):
        docs = self.parser.parse(self._three_attachment_event())
        event_doc = docs[0]
        assert "Attachments:" in event_doc.text
        assert "report.pdf" in event_doc.text
        assert "photo.png" in event_doc.text
        assert "spec.docx" in event_doc.text

    def test_attachment_with_newline_title_does_not_inject_lines(self):
        """A title containing '\\n' must not break the Attachments: bullet
        into multiple lines (which would let an attacker plant fake metadata
        in the parent event chunk)."""
        evil = "ok.pdf\nFrom: ceo@example.com"
        event = _make_event(
            meta={
                "download_attachments": False,
                "attachment_indexable_mimetypes": [],
                "attachment_max_size_mb": 25,
                "attachments": [_attachment("drv-evil", evil, "application/pdf")],
            }
        )
        docs = self.parser.parse(event)
        event_doc = docs[0]
        attachments_section = event_doc.text.split("Attachments:\n", 1)[1]
        # Exactly one bullet line, no embedded newline.
        assert attachments_section.count("\n") == 0
        assert "\nFrom: ceo@example.com" not in event_doc.text

        # Attachment Document keeps the original (unsanitized) title.
        att = next(d for d in docs if d.node_label == "Attachment")
        assert att.node_props["title"] == evil

    def test_attachment_source_id_uses_account_event_file(self):
        docs = self.parser.parse(self._three_attachment_event())
        att_docs = [d for d in docs if d.node_label == "Attachment"]
        ids = {d.source_id for d in att_docs}
        assert "google-calendar://personal/event/evt-123/attachment/drv-pdf" in ids
        assert "google-calendar://personal/event/evt-123/attachment/drv-img" in ids

    def test_attached_to_edge_points_at_event(self):
        docs = self.parser.parse(self._three_attachment_event())
        att_docs = [d for d in docs if d.node_label == "Attachment"]
        for doc in att_docs:
            attached = [h for h in doc.graph_hints if h.predicate == "ATTACHED_TO"]
            assert len(attached) == 1
            edge = attached[0]
            assert edge.subject_id == doc.source_id
            assert edge.subject_label == "Attachment"
            assert edge.object_id == "google-calendar://personal/event/evt-123"
            assert edge.object_label == "CalendarEvent"
            assert edge.edge_props == {"account": "personal"}

    def test_attendee_edges_not_propagated_to_attachment(self):
        """ORGANIZED_BY / ATTENDED_BY must not be cloned onto attachments."""
        docs = self.parser.parse(self._three_attachment_event())
        att_docs = [d for d in docs if d.node_label == "Attachment"]
        for doc in att_docs:
            preds = {h.predicate for h in doc.graph_hints}
            assert "ORGANIZED_BY" not in preds
            assert "ATTENDED_BY" not in preds

    def test_parent_url_uses_event_html_link(self):
        docs = self.parser.parse(self._three_attachment_event())
        att_docs = [d for d in docs if d.node_label == "Attachment"]
        for doc in att_docs:
            assert (
                doc.source_metadata["parent_url"]
                == "https://calendar.google.com/calendar/event?eid=evt123"
            )


class TestCalendarAttachmentsDownloadAndIndex:
    """download_attachments=True routes indexable MIMEs through stream_and_parse."""

    def setup_method(self):
        self.parser = GoogleCalendarParser()

    def _three_attachment_event(self) -> dict:
        return _make_event(
            meta={
                "download_attachments": True,
                "attachment_indexable_mimetypes": [
                    "application/pdf",
                    "image/png",
                ],
                "attachment_max_size_mb": 25,
                "attachments": [
                    _attachment(
                        "drv-pdf",
                        "report.pdf",
                        "application/pdf",
                        size_bytes=2048,
                    ),
                    _attachment(
                        "drv-img",
                        "photo.png",
                        "image/png",
                        size_bytes=4096,
                    ),
                    _attachment(
                        "drv-doc",
                        "spec.docx",
                        _DOCX_MIME,
                        size_bytes=0,
                    ),
                ],
            }
        )

    def test_indexable_mimes_invoke_stream_and_parse_others_metadata_only(self):
        fetched: list[tuple[str, str]] = []

        def fake_fetcher(account: str, file_id: str):
            fetched.append((account, file_id))
            return lambda: b"FAKEBYTES"

        # Track which file_ids land in stream_and_parse and supply
        # plausible parsed bytes for each.
        parsed_for: list[str] = []

        def fake_stream_and_parse(*, fetch, filename, mime, source_id=None, **_kwargs):
            parsed_for.append(filename)
            fetch()  # exercise the closure to confirm the fetcher reached
            return ParsedAttachment(
                text=f"parsed {filename}",
                description=f"summary of {filename}",
            )

        with (
            patch.object(
                calendar_parser_module,
                "_build_drive_fetcher",
                fake_fetcher,
            ),
            patch.object(
                calendar_parser_module,
                "stream_and_parse",
                fake_stream_and_parse,
            ),
        ):
            docs = self.parser.parse(self._three_attachment_event())

        # PDF and PNG flow through; .docx stays metadata-only.
        assert sorted(parsed_for) == ["photo.png", "report.pdf"]
        # Only the indexable file_ids reached the fetcher seam.
        assert sorted(f for _, f in fetched) == ["drv-img", "drv-pdf"]

        att_docs_by_id = {
            d.node_props["file_id"]: d for d in docs if d.node_label == "Attachment"
        }
        assert att_docs_by_id["drv-pdf"].node_props["decision"] == "download_and_index"
        assert att_docs_by_id["drv-img"].node_props["decision"] == "download_and_index"
        assert att_docs_by_id["drv-doc"].node_props["decision"] == "metadata_only"

        # Indexed attachments carry their parsed text.
        assert "parsed report.pdf" in att_docs_by_id["drv-pdf"].text
        assert "parsed photo.png" in att_docs_by_id["drv-img"].text

    def test_404_on_get_media_falls_back_to_metadata_only(self):
        def fake_fetcher(_account, _file_id):
            return lambda: b""

        def boom(*, fetch, filename, mime, source_id=None, **_kwargs):
            raise AttachmentDownloadError(
                f"404 not found: {filename}", source_id=source_id
            )

        evt = _make_event(
            meta={
                "download_attachments": True,
                "attachment_indexable_mimetypes": ["application/pdf"],
                "attachment_max_size_mb": 25,
                "attachments": [
                    _attachment(
                        "drv-pdf",
                        "deleted.pdf",
                        "application/pdf",
                        size_bytes=2048,
                    )
                ],
            }
        )

        with (
            patch.object(
                calendar_parser_module,
                "_build_drive_fetcher",
                fake_fetcher,
            ),
            patch.object(
                calendar_parser_module,
                "stream_and_parse",
                boom,
            ),
        ):
            docs = self.parser.parse(evt)

        att_docs = [d for d in docs if d.node_label == "Attachment"]
        assert len(att_docs) == 1
        assert att_docs[0].node_props["decision"] == "metadata_only"

    def test_unsupported_mime_parse_error_falls_back_to_metadata_only(self):
        """Unexpected parser errors must downgrade to metadata-only, not crash."""

        def fake_fetcher(_account, _file_id):
            return lambda: b""

        def boom(*, fetch, filename, mime, source_id=None, **_kwargs):
            raise AttachmentParseError(f"refused {mime}", source_id=source_id)

        evt = _make_event(
            meta={
                "download_attachments": True,
                "attachment_indexable_mimetypes": ["application/pdf"],
                "attachment_max_size_mb": 25,
                "attachments": [
                    _attachment(
                        "drv-x",
                        "weird.pdf",
                        "application/pdf",
                        size_bytes=1024,
                    )
                ],
            }
        )

        with (
            patch.object(
                calendar_parser_module,
                "_build_drive_fetcher",
                fake_fetcher,
            ),
            patch.object(
                calendar_parser_module,
                "stream_and_parse",
                boom,
            ),
        ):
            docs = self.parser.parse(evt)

        att = next(d for d in docs if d.node_label == "Attachment")
        assert att.node_props["decision"] == "metadata_only"

    def test_unknown_size_treated_as_too_large(self):
        """size_bytes=0 must not slip past the max_size_mb gate when the
        caller has set a conservative bound — an attachment of unknown
        size is treated as 'too large' and stays metadata-only."""
        evt = _make_event(
            meta={
                "download_attachments": True,
                "attachment_indexable_mimetypes": ["application/pdf"],
                "attachment_max_size_mb": 1,
                "attachments": [
                    _attachment(
                        "drv-mystery",
                        "huge.pdf",
                        "application/pdf",
                        size_bytes=0,
                    )
                ],
            }
        )

        # If we erroneously hit the fetcher this test fails loudly.
        def boom(*_args, **_kwargs):
            raise AssertionError("unknown-size attachment must stay metadata-only")

        with patch.object(
            calendar_parser_module,
            "_build_drive_fetcher",
            boom,
        ):
            docs = self.parser.parse(evt)

        att = next(d for d in docs if d.node_label == "Attachment")
        assert att.node_props["decision"] == "metadata_only"


class TestCalendarAttachmentCounters:
    """Parent CalendarEvent carries intended/indexed/metadata_only counters."""

    def setup_method(self):
        self.parser = GoogleCalendarParser()

    def _three_attachment_event(self) -> dict:
        return _make_event(
            meta={
                "download_attachments": True,
                "attachment_indexable_mimetypes": ["application/pdf"],
                "attachment_max_size_mb": 25,
                "attachments": [
                    _attachment("drv-1", "f1.pdf", "application/pdf", size_bytes=2048),
                    _attachment("drv-2", "f2.pdf", "application/pdf", size_bytes=2048),
                    _attachment("drv-3", "f3.pdf", "application/pdf", size_bytes=2048),
                ],
            }
        )

    def test_no_attachments_omits_counters(self):
        docs = self.parser.parse(_make_event())
        event_doc = next(d for d in docs if d.node_label == "CalendarEvent")
        assert "has_attachments" not in event_doc.node_props
        assert "attachments_count_intended" not in event_doc.node_props
        assert "attachments_count_indexed" not in event_doc.node_props
        assert "attachments_count_metadata_only" not in event_doc.node_props

    def test_intended_matches_total_when_all_fail(self):
        """3 attachments, all fail to fetch ⇒ intended=3, indexed=0, metadata_only=3."""

        def fetcher(_account, _file_id):
            return lambda: b""

        def boom(*, fetch, filename, mime, source_id=None, **_kwargs):
            raise AttachmentDownloadError(
                f"network down: {filename}", source_id=source_id
            )

        with (
            patch.object(calendar_parser_module, "_build_drive_fetcher", fetcher),
            patch.object(calendar_parser_module, "stream_and_parse", boom),
        ):
            docs = self.parser.parse(self._three_attachment_event())

        event_doc = next(d for d in docs if d.node_label == "CalendarEvent")
        assert event_doc.node_props["attachments_count_intended"] == 3
        assert event_doc.node_props["attachments_count_indexed"] == 0
        assert event_doc.node_props["attachments_count_metadata_only"] == 3
        assert event_doc.node_props["has_attachments"] == 3  # deprecated alias

    def test_indexed_only_successes(self):
        """3 attachments, 1 fails to fetch ⇒ intended=3, indexed=2, metadata_only=1."""

        def fetcher(_account, _file_id):
            return lambda: b"FAKEBYTES"

        def maybe_parse(*, fetch, filename, mime, source_id=None, **_kwargs):
            fetch()
            if filename == "f1.pdf":
                raise AttachmentParseError("malformed", source_id=source_id)
            return ParsedAttachment(text=f"parsed {filename}")

        with (
            patch.object(calendar_parser_module, "_build_drive_fetcher", fetcher),
            patch.object(calendar_parser_module, "stream_and_parse", maybe_parse),
        ):
            docs = self.parser.parse(self._three_attachment_event())

        event_doc = next(d for d in docs if d.node_label == "CalendarEvent")
        assert event_doc.node_props["attachments_count_intended"] == 3
        assert event_doc.node_props["attachments_count_indexed"] == 2
        assert event_doc.node_props["attachments_count_metadata_only"] == 1


class TestCrossAccount:
    """Cross-account invariants required by the multi-account schema."""

    def setup_method(self):
        self.parser = GoogleCalendarParser()

    def test_same_event_id_two_accounts_yields_distinct_event_nodes(self):
        """Same event_id on 'primary' in two accounts MUST produce two
        distinct CalendarEvent nodes — Google reuses event IDs."""
        event_personal = _make_event(
            account="personal",
            meta={"event_id": "shared-eid"},
        )
        event_work = _make_event(
            account="work",
            meta={"event_id": "shared-eid"},
        )
        doc_personal = self.parser.parse(event_personal)[0]
        doc_work = self.parser.parse(event_work)[0]

        assert doc_personal.source_id == "google-calendar://personal/event/shared-eid"
        assert doc_work.source_id == "google-calendar://work/event/shared-eid"
        assert doc_personal.source_id != doc_work.source_id

    def test_same_attendee_email_two_accounts_yields_one_person_node(self):
        """bob@example.com attending in both accounts MUST produce one
        Person node (email-keyed merge bridges accounts)."""
        event_personal = _make_event(
            account="personal",
            meta={
                "attendees": [
                    {
                        "email": "Bob@Example.COM",
                        "name": "Bob",
                        "self": False,
                    }
                ]
            },
        )
        event_work = _make_event(
            account="work",
            meta={
                "attendees": [
                    {
                        "email": "bob@example.com",
                        "name": "Bob",
                        "self": False,
                    }
                ]
            },
        )
        att_personal = next(
            h
            for h in self.parser.parse(event_personal)[0].graph_hints
            if h.predicate == "ATTENDED_BY"
        )
        att_work = next(
            h
            for h in self.parser.parse(event_work)[0].graph_hints
            if h.predicate == "ATTENDED_BY"
        )

        # Canonicalized through canonicalize_email() and merging on email.
        assert att_personal.object_id == "person:bob@example.com"
        assert att_work.object_id == "person:bob@example.com"
        assert att_personal.object_merge_key == "email"
        assert att_work.object_merge_key == "email"
        # Edges are scoped to their respective accounts.
        assert att_personal.edge_props["account"] == "personal"
        assert att_work.edge_props["account"] == "work"

    def test_attendee_without_email_uses_account_scoped_fallback_id(self):
        """Display-name-only attendees fall back to an account+event scoped
        Person ID so the same display name in two accounts does NOT
        accidentally collapse before the email is known."""
        event_personal = _make_event(
            account="personal",
            meta={
                "event_id": "evt-A",
                "attendees": [
                    {"name": "John Doe", "self": False},
                ],
            },
        )
        event_work = _make_event(
            account="work",
            meta={
                "event_id": "evt-B",
                "attendees": [
                    {"name": "John Doe", "self": False},
                ],
            },
        )
        att_personal = next(
            h
            for h in self.parser.parse(event_personal)[0].graph_hints
            if h.predicate == "ATTENDED_BY"
        )
        att_work = next(
            h
            for h in self.parser.parse(event_work)[0].graph_hints
            if h.predicate == "ATTENDED_BY"
        )

        assert att_personal.object_id == (
            "google-calendar://personal/event/evt-A/attendee/0"
        )
        assert att_work.object_id == ("google-calendar://work/event/evt-B/attendee/0")
        assert att_personal.object_id != att_work.object_id
        assert att_personal.object_merge_key == "source_id"
        assert att_work.object_merge_key == "source_id"
        # Account stamped on the Person node so reconcile can walk it.
        assert att_personal.object_props["account"] == "personal"
        assert att_work.object_props["account"] == "work"


class TestCalendarParser_EmitsReferencesEdge:
    def setup_method(self):
        self.parser = GoogleCalendarParser()

    def test_obsidian_url_in_description_produces_references_hint(self):
        event = _make_event(
            meta={
                "description": "See obsidian://open?vault=Personal&file=Notes.md for context.",
            },
        )
        doc = self.parser.parse(event)[0]
        refs = [h for h in doc.graph_hints if h.predicate == "REFERENCES"]
        assert len(refs) == 1
        h = refs[0]
        assert h.object_id == "obsidian://open?vault=Personal&file=Notes.md"
        assert h.object_label == "File"
        assert h.subject_label == "CalendarEvent"

    def test_google_calendar_url_in_description_produces_references_hint(self):
        event = _make_event(
            meta={
                "description": "Linked event: google-calendar://acct/event/abc.",
            },
        )
        doc = self.parser.parse(event)[0]
        refs = [h for h in doc.graph_hints if h.predicate == "REFERENCES"]
        assert len(refs) == 1
        assert refs[0].object_id == "google-calendar://acct/event/abc"
        assert refs[0].object_label == "CalendarEvent"

    def test_no_source_url_in_description_produces_no_references_hints(self):
        event = _make_event(meta={"description": "Plain description, no links."})
        doc = self.parser.parse(event)[0]
        refs = [h for h in doc.graph_hints if h.predicate == "REFERENCES"]
        assert refs == []
