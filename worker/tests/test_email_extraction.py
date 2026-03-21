"""Tests for extract_email_person_hints() in parsers/base.py."""

from __future__ import annotations

from worker.parsers.base import extract_email_person_hints


class TestExtractEmailPersonHints:
    def test_single_email(self):
        hints = extract_email_person_hints(
            "Contact alice@example.com for details.", "doc:1"
        )
        assert len(hints) == 1
        h = hints[0]
        assert h.object_label == "Person"
        assert h.object_props["email"] == "alice@example.com"
        assert h.object_id == "person:alice@example.com"
        assert h.object_merge_key == "email"
        assert h.predicate == "MENTIONS"
        assert h.subject_id == "doc:1"
        assert h.confidence == 1.0

    def test_multiple_emails(self):
        text = "CC: bob@corp.io and carol@company.org please."
        hints = extract_email_person_hints(text, "doc:2")
        emails = {h.object_props["email"] for h in hints}
        assert emails == {"bob@corp.io", "carol@company.org"}

    def test_duplicate_email_deduplicated(self):
        text = "alice@example.com and again alice@example.com"
        hints = extract_email_person_hints(text, "doc:3")
        assert len(hints) == 1

    def test_googlemail_canonical(self):
        text = "Email me at user@googlemail.com"
        hints = extract_email_person_hints(text, "doc:4")
        assert hints[0].object_props["email"] == "user@gmail.com"
        assert hints[0].object_id == "person:user@gmail.com"

    def test_no_emails_returns_empty(self):
        hints = extract_email_person_hints("No emails here.", "doc:5")
        assert hints == []

    def test_empty_text(self):
        hints = extract_email_person_hints("", "doc:6")
        assert hints == []

    def test_subject_label_default(self):
        hints = extract_email_person_hints("a@b.com", "doc:7")
        assert hints[0].subject_label == "File"

    def test_subject_label_custom(self):
        hints = extract_email_person_hints("a@b.com", "doc:8", subject_label="Task")
        assert hints[0].subject_label == "Task"

    def test_email_in_markdown_link(self):
        text = "See [Alice](mailto:alice@example.com) for details"
        hints = extract_email_person_hints(text, "doc:9")
        assert len(hints) == 1
        assert hints[0].object_props["email"] == "alice@example.com"

    def test_uppercase_email_lowered(self):
        text = "Contact ALICE@EXAMPLE.COM now"
        hints = extract_email_person_hints(text, "doc:10")
        assert hints[0].object_props["email"] == "alice@example.com"

    def test_email_with_plus_tag(self):
        text = "Send to user+tag@example.com"
        hints = extract_email_person_hints(text, "doc:11")
        assert hints[0].object_props["email"] == "user+tag@example.com"

    def test_multiline_text(self):
        text = "Line 1\nalice@a.com\nLine 3\nbob@b.com\n"
        hints = extract_email_person_hints(text, "doc:12")
        assert len(hints) == 2
