"""Tests for the Gmail parser."""

from worker.parsers.base import GraphHint, ParsedDocument
from worker.parsers.gmail import GmailParser, _parse_email_address, _strip_html


def _make_event(
    text: str = "",
    mime_type: str = "message/rfc822",
    **overrides,
) -> dict:
    meta = overrides.pop("meta", {})
    base_meta = {
        "message_id": "msg-123",
        "thread_id": "thread-456",
        "subject": "Test Subject",
        "date": "Tue, 11 Mar 2026 12:00:00 +0000",
        "sender_email": "Alice <alice@example.com>",
        "recipients": ["Bob <bob@example.com>", "carol@example.com"],
    }
    base_meta.update(meta)
    base = {
        "source_type": "gmail",
        "source_id": "gmail:msg-123",
        "operation": "created",
        "text": text,
        "mime_type": mime_type,
        "meta": base_meta,
    }
    base.update(overrides)
    return base


class TestStripHtml:
    def test_plain_text_passthrough(self):
        assert _strip_html("Hello world") == "Hello world"

    def test_basic_tags(self):
        assert _strip_html("<p>Hello</p><p>World</p>") == "Hello\nWorld"

    def test_script_and_style_removed(self):
        html = "<style>body{color:red}</style><script>alert(1)</script><p>Content</p>"
        assert _strip_html(html) == "Content"

    def test_nested_html(self):
        html = "<div><table><tr><td>Cell</td></tr></table></div>"
        assert "Cell" in _strip_html(html)

    def test_empty_string(self):
        assert _strip_html("") == ""

    def test_whitespace_collapse(self):
        html = "<p>A</p>\n\n\n<p>B</p>"
        result = _strip_html(html)
        assert "A" in result
        assert "B" in result


class TestParseEmailAddress:
    def test_display_name_format(self):
        assert _parse_email_address("Alice <alice@example.com>") == "alice@example.com"

    def test_bare_address(self):
        assert _parse_email_address("alice@example.com") == "alice@example.com"

    def test_whitespace(self):
        assert _parse_email_address("  alice@example.com  ") == "alice@example.com"

    def test_quoted_display_name(self):
        assert (
            _parse_email_address('"Alice B" <alice@example.com>')
            == "alice@example.com"
        )


class TestGmailParser:
    def setup_method(self):
        self.parser = GmailParser()

    def test_source_type(self):
        assert self.parser.source_type == "gmail"

    def test_basic_parse(self):
        docs = self.parser.parse(_make_event("Hello from the email body"))
        assert len(docs) == 1
        doc = docs[0]
        assert doc.source_type == "gmail"
        assert doc.source_id == "gmail:msg-123"
        assert doc.operation == "created"
        assert doc.text == "Hello from the email body"
        assert doc.node_label == "Email"

    def test_node_props(self):
        docs = self.parser.parse(_make_event("body"))
        doc = docs[0]
        assert doc.node_props["message_id"] == "msg-123"
        assert doc.node_props["subject"] == "Test Subject"
        assert doc.node_props["date"] == "Tue, 11 Mar 2026 12:00:00 +0000"

    def test_source_metadata(self):
        docs = self.parser.parse(_make_event("body"))
        doc = docs[0]
        assert doc.source_metadata["source_type"] == "email"
        assert doc.source_metadata["thread_id"] == "thread-456"

    def test_sender_sent_hint(self):
        docs = self.parser.parse(_make_event("body"))
        sent_hints = [h for h in docs[0].graph_hints if h.predicate == "SENT"]
        assert len(sent_hints) == 1
        h = sent_hints[0]
        assert h.subject_id == "person:alice@example.com"
        assert h.subject_label == "Person"
        assert h.object_id == "gmail:msg-123"
        assert h.object_label == "Email"
        assert h.subject_props == {"email": "alice@example.com"}
        assert h.subject_merge_key == "email"
        assert h.confidence == 1.0

    def test_recipient_to_hints(self):
        docs = self.parser.parse(_make_event("body"))
        to_hints = [h for h in docs[0].graph_hints if h.predicate == "TO"]
        assert len(to_hints) == 2
        recipient_ids = {h.object_id for h in to_hints}
        assert "person:bob@example.com" in recipient_ids
        assert "person:carol@example.com" in recipient_ids
        for h in to_hints:
            assert h.subject_id == "gmail:msg-123"
            assert h.subject_label == "Email"
            assert h.object_label == "Person"
            assert h.object_merge_key == "email"
            assert h.confidence == 1.0

    def test_thread_part_of_hint(self):
        docs = self.parser.parse(_make_event("body"))
        part_of = [h for h in docs[0].graph_hints if h.predicate == "PART_OF"]
        assert len(part_of) == 1
        h = part_of[0]
        assert h.subject_id == "gmail:msg-123"
        assert h.subject_label == "Email"
        assert h.object_id == "gmail-thread:thread-456"
        assert h.object_label == "Thread"
        assert h.object_merge_key == "thread_id"
        assert h.object_props == {"thread_id": "thread-456", "subject": "Test Subject"}
        assert h.confidence == 1.0

    def test_no_thread_hint_when_thread_id_empty(self):
        event = _make_event("body", meta={"thread_id": ""})
        docs = self.parser.parse(event)
        part_of = [h for h in docs[0].graph_hints if h.predicate == "PART_OF"]
        assert len(part_of) == 0

    def test_html_body_stripped(self):
        html_body = "<html><body><p>Hello</p><p>World</p></body></html>"
        docs = self.parser.parse(_make_event(html_body, mime_type="text/html"))
        assert "<p>" not in docs[0].text
        assert "Hello" in docs[0].text
        assert "World" in docs[0].text

    def test_html_body_detected_by_content(self):
        """HTML body should be stripped even with non-HTML mime_type if content starts with <."""
        html_body = "<div>Email content</div>"
        docs = self.parser.parse(_make_event(html_body, mime_type="message/rfc822"))
        assert "<div>" not in docs[0].text
        assert "Email content" in docs[0].text

    def test_plain_text_not_stripped(self):
        plain = "Just plain text, no HTML here"
        docs = self.parser.parse(_make_event(plain, mime_type="text/plain"))
        assert docs[0].text == plain

    def test_deleted_operation(self):
        docs = self.parser.parse(_make_event("", operation="deleted"))
        assert len(docs) == 1
        assert docs[0].operation == "deleted"
        assert docs[0].text == ""
        assert docs[0].graph_hints == []

    def test_no_sender_no_sent_hint(self):
        event = _make_event("body", meta={"sender_email": ""})
        docs = self.parser.parse(event)
        sent_hints = [h for h in docs[0].graph_hints if h.predicate == "SENT"]
        assert len(sent_hints) == 0

    def test_empty_recipients_no_to_hints(self):
        event = _make_event("body", meta={"recipients": []})
        docs = self.parser.parse(event)
        to_hints = [h for h in docs[0].graph_hints if h.predicate == "TO"]
        assert len(to_hints) == 0

    def test_missing_meta_fields(self):
        """Parser should handle missing meta fields gracefully."""
        event = {
            "source_type": "gmail",
            "source_id": "gmail:bare",
            "operation": "created",
            "text": "body",
            "mime_type": "text/plain",
            "meta": {},
        }
        docs = self.parser.parse(event)
        assert len(docs) == 1
        assert docs[0].node_props["message_id"] == ""
        assert docs[0].node_props["subject"] == ""

    def test_registry_registration(self):
        from worker.parsers.registry import get

        parser = get("gmail")
        assert isinstance(parser, GmailParser)
