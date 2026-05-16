"""Tests for extract_source_link_hints() in parsers/base.py."""

from __future__ import annotations

from worker.parsers.base import extract_source_link_hints


class TestExtractSourceLinks_Empty:
    def test_empty_text(self):
        assert extract_source_link_hints("", "doc:1") == []

    def test_plain_text_no_urls(self):
        assert extract_source_link_hints("No links here at all.", "doc:2") == []


class TestExtractSourceLinks_Gmail:
    def test_single_gmail_url(self):
        hints = extract_source_link_hints(
            "See gmail://account@gmail.com/message/abc123 for details.", "doc:3"
        )
        assert len(hints) == 1
        h = hints[0]
        assert h.object_id == "gmail://account@gmail.com/message/abc123"
        assert h.object_label == "EmailMessage"
        assert h.predicate == "REFERENCES"
        assert h.confidence == 1.0
        assert h.subject_id == "doc:3"
        assert h.subject_label == "File"


class TestExtractSourceLinks_Calendar:
    def test_single_calendar_url(self):
        hints = extract_source_link_hints(
            "Meeting at google-calendar://user@gmail.com/event/xyz789.", "doc:4"
        )
        assert len(hints) == 1
        h = hints[0]
        assert h.object_id == "google-calendar://user@gmail.com/event/xyz789"
        assert h.object_label == "CalendarEvent"
        assert h.predicate == "REFERENCES"
        assert h.confidence == 1.0


class TestExtractSourceLinks_Omnifocus:
    def test_single_omnifocus_url(self):
        hints = extract_source_link_hints(
            "Task: omnifocus://task/task-id-42 needs review.", "doc:5"
        )
        assert len(hints) == 1
        h = hints[0]
        assert h.object_id == "omnifocus://task/task-id-42"
        assert h.object_label == "OmniFocusTask"
        assert h.predicate == "REFERENCES"
        assert h.confidence == 1.0


class TestExtractSourceLinks_SlackBare:
    def test_single_slack_url(self):
        hints = extract_source_link_hints(
            "Slack thread at slack://T01234567/C01234567/1234567890.123456 here.",
            "doc:6",
        )
        assert len(hints) == 1
        h = hints[0]
        assert h.object_id == "slack://T01234567/C01234567/1234567890.123456"
        assert h.object_label == "SlackMessage"
        assert h.predicate == "REFERENCES"
        assert h.confidence == 1.0


class TestExtractSourceLinks_ObsidianWithVaultMap:
    def test_obsidian_with_vault_map(self):
        vault_map = {"Personal": "/Users/mmlac/Obsidian Vaults/Personal"}
        hints = extract_source_link_hints(
            "See obsidian://open?vault=Personal&file=Meetings%2FKris.md for notes.",
            "doc:7",
            obsidian_vaults=vault_map,
        )
        assert len(hints) == 1
        h = hints[0]
        assert h.object_id == "/Users/mmlac/Obsidian Vaults/Personal/Meetings/Kris.md"
        assert h.object_label == "ObsidianNote"
        assert h.predicate == "REFERENCES"
        assert h.confidence == 1.0


class TestExtractSourceLinks_ObsidianWithoutVaultMap:
    def test_obsidian_without_vault_map(self):
        hints = extract_source_link_hints(
            "See obsidian://open?vault=Personal&file=Meetings%2FKris.md for notes.",
            "doc:8",
        )
        assert len(hints) == 1
        h = hints[0]
        assert h.object_id == "obsidian://open?vault=Personal&file=Meetings%2FKris.md"
        assert h.object_label == "ObsidianNote"
        assert h.predicate == "REFERENCES"

    def test_obsidian_unknown_vault_in_map(self):
        vault_map = {"Work": "/Users/mmlac/Work"}
        hints = extract_source_link_hints(
            "See obsidian://open?vault=Personal&file=note.md here.",
            "doc:9",
            obsidian_vaults=vault_map,
        )
        assert len(hints) == 1
        assert hints[0].object_id == "obsidian://open?vault=Personal&file=note.md"


class TestExtractSourceLinks_Mixed:
    def test_all_five_schemes(self):
        text = (
            "gmail://acct/message/001 "
            "google-calendar://acct/event/002 "
            "omnifocus://task/003 "
            "slack://T1/C1/004 "
            "obsidian://open?vault=V&file=note.md"
        )
        hints = extract_source_link_hints(text, "doc:10")
        assert len(hints) == 5
        labels = {h.object_label for h in hints}
        assert labels == {
            "EmailMessage",
            "CalendarEvent",
            "OmniFocusTask",
            "SlackMessage",
            "ObsidianNote",
        }


class TestExtractSourceLinks_Dedup:
    def test_same_url_twice(self):
        text = "gmail://acct/message/abc and gmail://acct/message/abc again"
        hints = extract_source_link_hints(text, "doc:11")
        assert len(hints) == 1

    def test_dedup_obsidian_with_vault_map(self):
        vault_map = {"V": "/vault"}
        text = (
            "obsidian://open?vault=V&file=note.md and "
            "obsidian://open?vault=V&file=note.md repeated"
        )
        hints = extract_source_link_hints(text, "doc:12", obsidian_vaults=vault_map)
        assert len(hints) == 1


class TestExtractSourceLinks_TrailingPunctuation:
    def test_trailing_period_stripped(self):
        hints = extract_source_link_hints("see gmail://acct/message/abc.", "doc:13")
        assert len(hints) == 1
        assert hints[0].object_id == "gmail://acct/message/abc"

    def test_trailing_comma_stripped(self):
        hints = extract_source_link_hints(
            "links: gmail://acct/message/abc, more text", "doc:14"
        )
        assert len(hints) == 1
        assert hints[0].object_id == "gmail://acct/message/abc"

    def test_trailing_paren_stripped(self):
        hints = extract_source_link_hints(
            "Note (gmail://acct/message/abc) here", "doc:15"
        )
        assert len(hints) == 1
        assert hints[0].object_id == "gmail://acct/message/abc"


class TestExtractSourceLinks_NotMatchedInsideAnotherURL:
    def test_gmail_inside_https_query_string(self):
        """
        gmail:// preceded by '=' should not be extracted.
        The lookbehind (?<![=&]) skips matches where the scheme is preceded
        by a query-string separator. Most parsers don't context-track the outer
        URL, but the lookbehind handles the common case.
        """
        hints = extract_source_link_hints(
            "https://example.com/q?ref=gmail://acct/message/abc", "doc:16"
        )
        assert hints == []
