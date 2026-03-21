"""Tests for the Google Calendar parser."""

from worker.parsers.calendar import GoogleCalendarParser, _strip_html


def _make_event(
    text: str = "Team standup\n\nLocation: Conference Room A\n\nDaily sync meeting",
    **overrides,
) -> dict:
    meta = overrides.pop("meta", {})
    base_meta = {
        "event_id": "evt-123",
        "calendar_id": "primary",
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
        "source_id": "gcal:primary:evt-123",
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
        assert doc.source_id == "gcal:primary:evt-123"
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
        assert "html_link" in props

    def test_source_metadata(self):
        docs = self.parser.parse(_make_event())
        meta = docs[0].source_metadata
        assert meta["source_type"] == "calendar"
        assert meta["calendar_id"] == "primary"
        assert meta["start_time"] == "2026-03-21T09:00:00-07:00"

    def test_organizer_hint(self):
        docs = self.parser.parse(_make_event())
        org_hints = [
            h for h in docs[0].graph_hints if h.predicate == "ORGANIZED_BY"
        ]
        assert len(org_hints) == 1
        h = org_hints[0]
        assert h.subject_id == "gcal:primary:evt-123"
        assert h.subject_label == "CalendarEvent"
        assert h.object_id == "person:alice@example.com"
        assert h.object_label == "Person"
        assert h.object_props["email"] == "alice@example.com"
        assert h.object_props["name"] == "Alice Smith"
        assert h.object_merge_key == "email"
        assert h.confidence == 1.0

    def test_attendee_hints_exclude_self(self):
        """The 'self' attendee (me@example.com) should not generate a hint."""
        docs = self.parser.parse(_make_event())
        att_hints = [
            h for h in docs[0].graph_hints if h.predicate == "ATTENDED_BY"
        ]
        # 3 attendees total, but self=True is excluded → 2 hints
        assert len(att_hints) == 2
        attendee_ids = {h.object_id for h in att_hints}
        assert "person:bob@example.com" in attendee_ids
        assert "person:carol@example.com" in attendee_ids
        assert "person:me@example.com" not in attendee_ids

    def test_attendee_hint_properties(self):
        docs = self.parser.parse(_make_event())
        att_hints = [
            h for h in docs[0].graph_hints if h.predicate == "ATTENDED_BY"
        ]
        bob_hint = next(h for h in att_hints if "bob" in h.object_id)
        assert bob_hint.subject_id == "gcal:primary:evt-123"
        assert bob_hint.subject_label == "CalendarEvent"
        assert bob_hint.object_label == "Person"
        assert bob_hint.object_props["email"] == "bob@example.com"
        assert bob_hint.object_props["name"] == "Bob Jones"
        assert bob_hint.object_merge_key == "email"

    def test_cross_source_person_linking_with_gmail(self):
        """Calendar attendees use the SAME Person merge key (email) as Gmail.

        This means person:bob@example.com from Calendar will MERGE with
        person:bob@example.com from Gmail, creating cross-source links.
        """
        docs = self.parser.parse(_make_event())
        att_hints = [
            h for h in docs[0].graph_hints if h.predicate == "ATTENDED_BY"
        ]
        for h in att_hints:
            # Same pattern Gmail uses for Person nodes
            assert h.object_merge_key == "email"
            assert h.object_id.startswith("person:")
            assert h.object_label == "Person"

    def test_no_created_by_when_same_as_organizer(self):
        """CREATED_BY hint should be omitted when creator == organizer."""
        docs = self.parser.parse(_make_event())
        created_hints = [
            h for h in docs[0].graph_hints if h.predicate == "CREATED_BY"
        ]
        assert len(created_hints) == 0

    def test_created_by_when_different_from_organizer(self):
        event = _make_event(
            meta={
                "organizer_email": "alice@example.com",
                "creator_email": "delegate@example.com",
            }
        )
        docs = self.parser.parse(event)
        created_hints = [
            h for h in docs[0].graph_hints if h.predicate == "CREATED_BY"
        ]
        assert len(created_hints) == 1
        assert created_hints[0].object_id == "person:delegate@example.com"

    def test_deleted_operation(self):
        docs = self.parser.parse(_make_event(operation="deleted"))
        assert len(docs) == 1
        assert docs[0].operation == "deleted"
        assert docs[0].text == ""
        assert docs[0].graph_hints == []

    def test_no_organizer_no_hint(self):
        event = _make_event(meta={"organizer_email": ""})
        docs = self.parser.parse(event)
        org_hints = [
            h for h in docs[0].graph_hints if h.predicate == "ORGANIZED_BY"
        ]
        assert len(org_hints) == 0

    def test_no_attendees_no_hints(self):
        event = _make_event(meta={"attendees": []})
        docs = self.parser.parse(event)
        att_hints = [
            h for h in docs[0].graph_hints if h.predicate == "ATTENDED_BY"
        ]
        assert len(att_hints) == 0

    def test_missing_meta_fields(self):
        """Parser should handle missing meta fields gracefully."""
        event = {
            "source_type": "google_calendar",
            "source_id": "gcal:primary:bare",
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
            {"email": f"user{i}@example.com", "name": f"User {i}",
             "response": "accepted", "self": False}
            for i in range(250)
        ]
        event = _make_event(meta={"attendees": many_attendees})
        docs = self.parser.parse(event)
        att_hints = [
            h for h in docs[0].graph_hints if h.predicate == "ATTENDED_BY"
        ]
        assert len(att_hints) == 200  # _MAX_ATTENDEES

    def test_registry_registration(self):
        from worker.parsers.registry import get

        parser = get("google_calendar")
        assert isinstance(parser, GoogleCalendarParser)
