"""Tests for the Gmail parser."""

from __future__ import annotations

from typing import Any

import pytest

from worker.parsers import gmail as gmail_parser_mod
from worker.parsers.attachments import (
    AttachmentDownloadError,
    ParsedAttachment,
)
from worker.parsers.base import GraphHint
from worker.parsers.gmail import GmailParser, _parse_email_address, _strip_html


def _make_event(
    text: str = "",
    mime_type: str = "message/rfc822",
    account: str = "personal",
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
        "account": account,
    }
    base_meta.update(meta)
    base = {
        "source_type": "gmail",
        "source_id": f"gmail://{base_meta['account']}/message/{base_meta['message_id']}",
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
            _parse_email_address('"Alice B" <alice@example.com>') == "alice@example.com"
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
        assert doc.source_id == "gmail://personal/message/msg-123"
        assert doc.operation == "created"
        assert doc.text == "Hello from the email body"
        assert doc.node_label == "Email"

    def test_node_props(self):
        docs = self.parser.parse(_make_event("body"))
        doc = docs[0]
        assert doc.node_props["message_id"] == "msg-123"
        assert doc.node_props["subject"] == "Test Subject"
        assert doc.node_props["date"] == "Tue, 11 Mar 2026 12:00:00 +0000"
        assert doc.node_props["account"] == "personal"

    def test_source_metadata(self):
        docs = self.parser.parse(_make_event("body"))
        doc = docs[0]
        assert doc.source_metadata["source_type"] == "email"
        assert doc.source_metadata["thread_id"] == "thread-456"
        assert doc.source_metadata["account"] == "personal"

    def test_sender_sent_hint(self):
        docs = self.parser.parse(_make_event("body"))
        sent_hints = [h for h in docs[0].graph_hints if h.predicate == "SENT"]
        assert len(sent_hints) == 1
        h = sent_hints[0]
        assert h.subject_id == "person:alice@example.com"
        assert h.subject_label == "Person"
        assert h.object_id == "gmail://personal/message/msg-123"
        assert h.object_label == "Email"
        assert h.subject_props == {"email": "alice@example.com"}
        assert h.subject_merge_key == "email"
        assert h.edge_props == {"account": "personal"}
        assert h.confidence == 1.0

    def test_recipient_to_hints(self):
        docs = self.parser.parse(_make_event("body"))
        to_hints = [h for h in docs[0].graph_hints if h.predicate == "TO"]
        assert len(to_hints) == 2
        recipient_ids = {h.object_id for h in to_hints}
        assert "person:bob@example.com" in recipient_ids
        assert "person:carol@example.com" in recipient_ids
        for h in to_hints:
            assert h.subject_id == "gmail://personal/message/msg-123"
            assert h.subject_label == "Email"
            assert h.object_label == "Person"
            assert h.object_merge_key == "email"
            assert h.edge_props == {"account": "personal"}
            assert h.confidence == 1.0

    def test_thread_part_of_hint(self):
        docs = self.parser.parse(_make_event("body"))
        part_of = [h for h in docs[0].graph_hints if h.predicate == "PART_OF"]
        assert len(part_of) == 1
        h = part_of[0]
        assert h.subject_id == "gmail://personal/message/msg-123"
        assert h.subject_label == "Email"
        assert h.object_id == "gmail://personal/thread/thread-456"
        assert h.object_label == "Thread"
        assert h.object_merge_key == "source_id"
        assert h.object_props == {
            "thread_id": "thread-456",
            "subject": "Test Subject",
            "account": "personal",
        }
        assert h.edge_props == {"account": "personal"}
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

    def test_gmail_plus_addressing_dedupes_to_single_person(self):
        """Two Gmail messages — one to me@gmail.com, one to me+work@gmail.com —
        produce hints that target a single canonical Person merge key.
        """
        event_plain = _make_event(
            "body",
            meta={
                "message_id": "msg-plain",
                "sender_email": "me@gmail.com",
                "recipients": ["other@example.com"],
            },
        )
        event_tagged = _make_event(
            "body",
            meta={
                "message_id": "msg-tagged",
                "sender_email": "me+work@gmail.com",
                "recipients": ["other@example.com"],
            },
        )
        sent_plain = [
            h
            for h in self.parser.parse(event_plain)[0].graph_hints
            if h.predicate == "SENT"
        ]
        sent_tagged = [
            h
            for h in self.parser.parse(event_tagged)[0].graph_hints
            if h.predicate == "SENT"
        ]
        assert sent_plain[0].subject_id == "person:me@gmail.com"
        assert sent_tagged[0].subject_id == "person:me@gmail.com"
        assert sent_plain[0].subject_props == {"email": "me@gmail.com"}
        assert sent_tagged[0].subject_props == {"email": "me@gmail.com"}

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
            "source_id": "gmail:///message/bare",
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


def _three_attachments() -> list[dict[str, Any]]:
    return [
        {
            "filename": "report.pdf",
            "mime_type": "application/pdf",
            "size_bytes": 1_200_000,  # ~1.1 MB
            "attachment_id": "att-pdf",
        },
        {
            "filename": "screenshot.png",
            "mime_type": "image/png",
            "size_bytes": 220 * 1024,
            "attachment_id": "att-png",
        },
        {
            "filename": "logs.zip",
            "mime_type": "application/zip",
            "size_bytes": 100 * 1024,
            "attachment_id": "att-zip",
        },
    ]


def _event_with_attachments(
    *,
    download_attachments: bool,
    indexable: list[str],
    attachments: list[dict[str, Any]] | None = None,
    max_size_mb: int = 25,
) -> dict:
    base = _make_event("Body of the email", mime_type="text/plain")
    base["meta"]["attachments"] = (
        attachments if attachments is not None else _three_attachments()
    )
    base["meta"]["download_attachments"] = download_attachments
    base["meta"]["attachment_indexable_mimetypes"] = indexable
    base["meta"]["attachment_max_size_mb"] = max_size_mb
    base["meta"]["client_secrets_path"] = "/tmp/x.json"
    return base


class TestParserAttachmentsMetadataOnly:
    def setup_method(self):
        self.parser = GmailParser()

    def test_download_disabled_emits_three_metadata_only_documents(self, monkeypatch):
        """download_attachments=False ⇒ no fetch, three metadata-only docs."""
        called: list[str] = []

        def boom(**kwargs):
            called.append(kwargs.get("filename", "?"))
            raise AssertionError("stream_and_parse must NOT be called")

        monkeypatch.setattr(gmail_parser_mod, "_stream_and_parse", boom)

        event = _event_with_attachments(
            download_attachments=False,
            indexable=["application/pdf", "image/png"],
        )
        docs = self.parser.parse(event)

        assert called == []  # no fetch closure ever invoked
        # 1 parent + 3 attachments
        assert len(docs) == 4
        att_docs = [d for d in docs if d.node_label == "Attachment"]
        assert len(att_docs) == 3
        for d in att_docs:
            assert d.node_props["indexed"] is False
            assert d.node_props["parent_url"] == (
                "https://mail.google.com/mail/?ui=2&view=cv&th=thread-456"
            )
            assert "content not indexed" in d.text

    def test_parent_text_gains_attachments_section(self, monkeypatch):
        monkeypatch.setattr(
            gmail_parser_mod,
            "_stream_and_parse",
            lambda **kwargs: pytest.fail("must not fetch"),
        )
        event = _event_with_attachments(
            download_attachments=False,
            indexable=["application/pdf"],
        )
        docs = self.parser.parse(event)
        parent = next(d for d in docs if d.node_label == "Email")
        assert "Attachments:" in parent.text
        assert "- report.pdf (PDF, 1.1 MB)" in parent.text
        assert "- screenshot.png (PNG, 220.0 KB)" in parent.text
        assert "- logs.zip (ZIP, 100.0 KB)" in parent.text
        # Deprecated alias still mirrors the intended count.
        assert parent.node_props["has_attachments"] == 3
        # download_attachments=False ⇒ none indexed, all metadata-only.
        assert parent.node_props["attachments_count_intended"] == 3
        assert parent.node_props["attachments_count_indexed"] == 0
        assert parent.node_props["attachments_count_metadata_only"] == 3

    def test_zero_attachments_leaves_parent_text_unchanged(self):
        event = _make_event("plain body", mime_type="text/plain")
        event["meta"]["attachments"] = []
        event["meta"]["download_attachments"] = False
        docs = self.parser.parse(event)
        assert len(docs) == 1
        parent = docs[0]
        assert parent.text == "plain body"
        assert "Attachments:" not in parent.text
        assert "has_attachments" not in parent.node_props
        assert "attachments_count_intended" not in parent.node_props
        assert "attachments_count_indexed" not in parent.node_props
        assert "attachments_count_metadata_only" not in parent.node_props

    def test_attachments_count_intended_matches_total_when_all_fail(self, monkeypatch):
        """3 attachments, all fail to fetch ⇒ intended=3, indexed=0, metadata_only=3."""

        def fake_stream(**kwargs):
            raise AttachmentDownloadError(
                "network down", source_id=kwargs.get("source_id")
            )

        monkeypatch.setattr(gmail_parser_mod, "_stream_and_parse", fake_stream)

        event = _event_with_attachments(
            download_attachments=True,
            indexable=["application/pdf", "image/png", "application/zip"],
        )
        docs = self.parser.parse(event)
        parent = next(d for d in docs if d.node_label == "Email")
        assert parent.node_props["attachments_count_intended"] == 3
        assert parent.node_props["attachments_count_indexed"] == 0
        assert parent.node_props["attachments_count_metadata_only"] == 3

    def test_attachments_count_indexed_only_successes(self, monkeypatch):
        """3 attachments, 1 fails to fetch ⇒ intended=3, indexed=2, metadata_only=1."""

        def fake_stream(**kwargs):
            if kwargs["filename"] == "report.pdf":
                raise AttachmentDownloadError("boom", source_id=kwargs.get("source_id"))
            return ParsedAttachment(text=f"OK:{kwargs['filename']}")

        monkeypatch.setattr(gmail_parser_mod, "_stream_and_parse", fake_stream)

        event = _event_with_attachments(
            download_attachments=True,
            indexable=["application/pdf", "image/png", "application/zip"],
        )
        docs = self.parser.parse(event)
        parent = next(d for d in docs if d.node_label == "Email")
        assert parent.node_props["attachments_count_intended"] == 3
        assert parent.node_props["attachments_count_indexed"] == 2
        assert parent.node_props["attachments_count_metadata_only"] == 1

    def test_attachment_with_newline_filename_does_not_inject_lines(self, monkeypatch):
        """A filename containing '\\n' must not break the Attachments: bullet
        into multiple lines (which would let an attacker plant fake metadata
        in the parent thread chunk)."""
        monkeypatch.setattr(
            gmail_parser_mod,
            "_stream_and_parse",
            lambda **kwargs: pytest.fail("must not fetch"),
        )
        evil = "evil.pdf\nFrom: ceo@example.com\n\n[file] secret.pdf"
        event = _event_with_attachments(
            download_attachments=False,
            indexable=[],
            attachments=[
                {
                    "filename": evil,
                    "mime_type": "application/pdf",
                    "size_bytes": 1024,
                    "attachment_id": "att-evil",
                }
            ],
        )
        docs = self.parser.parse(event)
        parent = next(d for d in docs if d.node_label == "Email")

        # The Attachments: section must contain exactly one bullet.
        attachments_section = parent.text.split("Attachments:\n", 1)[1]
        assert attachments_section.count("\n- ") == 0
        assert attachments_section.count("\n") == 0
        # Injected fake-header text must not appear on its own line.
        assert "\nFrom: ceo@example.com" not in parent.text
        assert "\n[file] secret.pdf" not in parent.text

        # The Attachment Document keeps the original (unsanitized) filename.
        att = next(d for d in docs if d.node_label == "Attachment")
        assert att.node_props["filename"] == evil

    def test_attachment_filename_with_unicode_line_separator_is_sanitized(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            gmail_parser_mod,
            "_stream_and_parse",
            lambda **kwargs: pytest.fail("must not fetch"),
        )
        evil = "ok.pdf injected"
        event = _event_with_attachments(
            download_attachments=False,
            indexable=[],
            attachments=[
                {
                    "filename": evil,
                    "mime_type": "application/pdf",
                    "size_bytes": 1024,
                    "attachment_id": "att-uls",
                }
            ],
        )
        docs = self.parser.parse(event)
        parent = next(d for d in docs if d.node_label == "Email")
        assert " " not in parent.text

        att = next(d for d in docs if d.node_label == "Attachment")
        assert att.node_props["filename"] == evil


class TestParserAttachmentsDownload:
    def setup_method(self):
        self.parser = GmailParser()

    def test_download_path_calls_stream_and_parse_for_indexable(self, monkeypatch):
        """PDF + PNG indexable → fetched; ZIP → metadata-only (not in allowlist)."""
        calls: list[str] = []

        def fake_stream(**kwargs):
            calls.append(kwargs["filename"])
            return ParsedAttachment(text=f"PARSED:{kwargs['filename']}")

        monkeypatch.setattr(gmail_parser_mod, "_stream_and_parse", fake_stream)

        event = _event_with_attachments(
            download_attachments=True,
            indexable=["application/pdf", "image/png"],
        )
        docs = self.parser.parse(event)

        # PDF + PNG fetched, ZIP not.
        assert sorted(calls) == ["report.pdf", "screenshot.png"]

        att_docs = {
            d.node_props["filename"]: d for d in docs if d.node_label == "Attachment"
        }
        assert att_docs["report.pdf"].node_props["indexed"] is True
        assert att_docs["report.pdf"].text == "PARSED:report.pdf"
        assert att_docs["screenshot.png"].node_props["indexed"] is True
        assert att_docs["logs.zip"].node_props["indexed"] is False
        assert "content not indexed" in att_docs["logs.zip"].text

    def test_attached_to_edge_to_thread_for_each_attachment(self, monkeypatch):
        monkeypatch.setattr(
            gmail_parser_mod,
            "_stream_and_parse",
            lambda **kwargs: ParsedAttachment(text=""),
        )
        event = _event_with_attachments(
            download_attachments=False,
            indexable=[],
        )
        docs = self.parser.parse(event)
        att_docs = [d for d in docs if d.node_label == "Attachment"]
        for d in att_docs:
            attached = [h for h in d.graph_hints if h.predicate == "ATTACHED_TO"]
            assert len(attached) == 1
            assert attached[0].object_id == "gmail://personal/thread/thread-456"
            assert attached[0].object_label == "Thread"
            assert attached[0].object_merge_key == "source_id"

    def test_sent_by_edge_follows_parent_author(self, monkeypatch):
        monkeypatch.setattr(
            gmail_parser_mod,
            "_stream_and_parse",
            lambda **kwargs: ParsedAttachment(text=""),
        )
        event = _event_with_attachments(
            download_attachments=False,
            indexable=[],
        )
        docs = self.parser.parse(event)
        att_docs = [d for d in docs if d.node_label == "Attachment"]
        for d in att_docs:
            sent_by = [h for h in d.graph_hints if h.predicate == "SENT_BY"]
            assert len(sent_by) == 1
            assert sent_by[0].object_id == "person:alice@example.com"
            assert sent_by[0].object_merge_key == "email"
            assert sent_by[0].edge_props == {"account": "personal"}

    def test_fetch_failure_falls_through_to_metadata_only(self, monkeypatch):
        """A download error on one attachment does not affect the others."""
        seen: list[str] = []

        def fake_stream(**kwargs):
            seen.append(kwargs["filename"])
            if kwargs["filename"] == "report.pdf":
                raise AttachmentDownloadError(
                    "network down", source_id=kwargs.get("source_id")
                )
            return ParsedAttachment(text=f"OK:{kwargs['filename']}")

        monkeypatch.setattr(gmail_parser_mod, "_stream_and_parse", fake_stream)

        event = _event_with_attachments(
            download_attachments=True,
            indexable=["application/pdf", "image/png"],
        )
        docs = self.parser.parse(event)

        att_docs = {
            d.node_props["filename"]: d for d in docs if d.node_label == "Attachment"
        }
        # PDF degraded gracefully.
        assert att_docs["report.pdf"].node_props["indexed"] is False
        assert "content not indexed" in att_docs["report.pdf"].text
        # PNG still indexed.
        assert att_docs["screenshot.png"].node_props["indexed"] is True
        assert att_docs["screenshot.png"].text == "OK:screenshot.png"
        # ZIP never even attempted.
        assert att_docs["logs.zip"].node_props["indexed"] is False
        assert sorted(seen) == ["report.pdf", "screenshot.png"]

    def test_attachment_source_id_shape(self, monkeypatch):
        monkeypatch.setattr(
            gmail_parser_mod,
            "_stream_and_parse",
            lambda **kwargs: ParsedAttachment(text=""),
        )
        event = _event_with_attachments(
            download_attachments=False,
            indexable=[],
        )
        docs = self.parser.parse(event)
        ids = sorted(d.source_id for d in docs if d.node_label == "Attachment")
        assert ids == [
            "gmail://personal/thread/thread-456/attachment/att-pdf",
            "gmail://personal/thread/thread-456/attachment/att-png",
            "gmail://personal/thread/thread-456/attachment/att-zip",
        ]

    def test_extracted_entities_become_attachment_graph_hints(self, monkeypatch):
        """Entities returned by stream_and_parse propagate as GraphHints."""
        hint = GraphHint(
            subject_id="placeholder",
            subject_label="Attachment",
            predicate="MENTIONS",
            object_id="Whiskers",
            object_label="Person",
            object_props={"name": "Whiskers"},
            object_merge_key="name",
            confidence=0.8,
        )

        monkeypatch.setattr(
            gmail_parser_mod,
            "_stream_and_parse",
            lambda **kwargs: ParsedAttachment(text="parsed", extracted_entities=[hint]),
        )
        event = _event_with_attachments(
            download_attachments=True,
            indexable=["application/pdf"],
            attachments=[
                {
                    "filename": "r.pdf",
                    "mime_type": "application/pdf",
                    "size_bytes": 1024,
                    "attachment_id": "a-pdf",
                }
            ],
        )
        docs = self.parser.parse(event)
        att = next(d for d in docs if d.node_label == "Attachment")
        mentions = [h for h in att.graph_hints if h.predicate == "MENTIONS"]
        assert len(mentions) == 1
        assert mentions[0].object_id == "Whiskers"

    def test_dedupe_by_attachment_id_within_event(self):
        """Two records with the same attachment_id collapse to one Document."""
        atts = [
            {
                "filename": "dup.pdf",
                "mime_type": "application/pdf",
                "size_bytes": 100,
                "attachment_id": "same",
            },
            {
                "filename": "dup.pdf",
                "mime_type": "application/pdf",
                "size_bytes": 100,
                "attachment_id": "same",
            },
        ]
        event = _event_with_attachments(
            download_attachments=False,
            indexable=[],
            attachments=atts,
        )
        docs = GmailParser().parse(event)
        att_docs = [d for d in docs if d.node_label == "Attachment"]
        assert len(att_docs) == 1


class TestCrossAccount:
    """Cross-account invariants required by the multi-account schema."""

    def setup_method(self):
        self.parser = GmailParser()

    def test_same_thread_id_two_accounts_yields_distinct_thread_nodes(self):
        """Same thread_id seen on two accounts MUST produce two distinct
        Thread nodes — Gmail thread IDs are scoped to a mailbox."""
        event_personal = _make_event(
            "body",
            account="personal",
            meta={"thread_id": "shared-tid", "message_id": "m1"},
        )
        event_work = _make_event(
            "body",
            account="work",
            meta={"thread_id": "shared-tid", "message_id": "m2"},
        )
        thread_personal = next(
            h
            for h in self.parser.parse(event_personal)[0].graph_hints
            if h.predicate == "PART_OF"
        )
        thread_work = next(
            h
            for h in self.parser.parse(event_work)[0].graph_hints
            if h.predicate == "PART_OF"
        )

        assert thread_personal.object_id == "gmail://personal/thread/shared-tid"
        assert thread_work.object_id == "gmail://work/thread/shared-tid"
        assert thread_personal.object_id != thread_work.object_id
        # Both must merge on source_id (the URI), NOT thread_id, so the
        # writer doesn't collapse them.
        assert thread_personal.object_merge_key == "source_id"
        assert thread_work.object_merge_key == "source_id"

    def test_same_email_address_two_accounts_yields_one_person_node(self):
        """alice@example.com seen by both accounts MUST produce one Person
        node (email-keyed merge bridges accounts)."""
        event_personal = _make_event(
            "body",
            account="personal",
            meta={"sender_email": "Alice <alice@example.com>"},
        )
        event_work = _make_event(
            "body",
            account="work",
            meta={"sender_email": "Alice <alice@example.com>"},
        )
        sent_personal = next(
            h
            for h in self.parser.parse(event_personal)[0].graph_hints
            if h.predicate == "SENT"
        )
        sent_work = next(
            h
            for h in self.parser.parse(event_work)[0].graph_hints
            if h.predicate == "SENT"
        )

        # Same Person: same email, same merge key.
        assert sent_personal.subject_id == sent_work.subject_id
        assert sent_personal.subject_merge_key == "email"
        assert sent_work.subject_merge_key == "email"
        # But the edges are stamped with different accounts.
        assert sent_personal.edge_props["account"] == "personal"
        assert sent_work.edge_props["account"] == "work"

    def test_account_appears_as_metadata_not_in_person_merge_key(self):
        """Sender display name + account must not bleed into the Person
        merge key — that is reserved for email."""
        event = _make_event(
            "body",
            account="work",
            meta={"sender_email": "Markus <markus@example.com>"},
        )
        sent = next(
            h for h in self.parser.parse(event)[0].graph_hints if h.predicate == "SENT"
        )
        assert sent.subject_merge_key == "email"
        assert sent.subject_props == {"email": "markus@example.com"}
        # account lives on the edge, not the Person identity.
        assert "account" not in sent.subject_props
        assert sent.edge_props == {"account": "work"}


class TestGmailParser_EmitsReferencesEdge:
    def setup_method(self):
        self.parser = GmailParser()

    def test_calendar_url_in_body_produces_references_hint(self):
        event = _make_event(
            text="See google-calendar://acct/event/abc for the meeting.",
        )
        doc = self.parser.parse(event)[0]
        refs = [h for h in doc.graph_hints if h.predicate == "REFERENCES"]
        assert len(refs) == 1
        h = refs[0]
        assert h.object_id == "google-calendar://acct/event/abc"
        assert h.object_label == "CalendarEvent"
        assert h.subject_label == "Email"

    def test_multiple_source_urls_in_body(self):
        event = _make_event(
            text="gmail://work/message/001 and omnifocus://task/task-42",
        )
        doc = self.parser.parse(event)[0]
        refs = [h for h in doc.graph_hints if h.predicate == "REFERENCES"]
        assert len(refs) == 2
        ids = {h.object_id for h in refs}
        assert "gmail://work/message/001" in ids
        assert "omnifocus://task-42" in ids

    def test_no_source_url_in_body_produces_no_references_hints(self):
        event = _make_event(text="Plain email body, no source links.")
        doc = self.parser.parse(event)[0]
        refs = [h for h in doc.graph_hints if h.predicate == "REFERENCES"]
        assert refs == []
