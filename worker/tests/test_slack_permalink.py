"""Tests for the Slack permalink resolver (fn-86y.3).

Covers URL parsing, ts formatting, workspace map I/O, fallback behaviour,
the extract_source_link_hints integration, and node identity alignment with
Slack ingestion (CONTAINS_MESSAGE hints from the Slack parser must use the
same source_id form as REFERENCES hints from the permalink resolver).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from worker.parsers._slack_permalink import (
    load_workspace_team_map,
    save_workspace_team_map,
    slack_permalink_to_source_id,
    ts_packed_to_ts,
)
from worker.parsers.base import extract_source_link_hints
from worker.parsers.slack import SlackParser


_WORKSPACE_MAP = {"terra2": "T012XYZ"}
_BASE_URL = "https://terra2.slack.com/archives/C09ABCDEF/p1715800000123456"


class TestSlackPermalink_ParsesURL:
    def test_full_url_resolves_to_source_id(self) -> None:
        result = slack_permalink_to_source_id(_BASE_URL, _WORKSPACE_MAP)
        assert result == "slack://T012XYZ/C09ABCDEF/1715800000.123456"

    def test_channel_and_ts_preserved(self) -> None:
        url = "https://terra2.slack.com/archives/C99ZZZZZ/p1600000000654321"
        result = slack_permalink_to_source_id(url, {"terra2": "T999"})
        assert result == "slack://T999/C99ZZZZZ/1600000000.654321"


class TestSlackPermalink_NoWorkspaceMap:
    def test_empty_map_returns_fallback(self) -> None:
        # Strategy B fallback: empty team_id in source_id (triple slash).
        result = slack_permalink_to_source_id(_BASE_URL, {})
        assert result == "slack:///C09ABCDEF/1715800000.123456"

    def test_unknown_workspace_returns_fallback(self) -> None:
        result = slack_permalink_to_source_id(_BASE_URL, {"otherws": "T999"})
        assert result == "slack:///C09ABCDEF/1715800000.123456"


class TestSlackPermalink_TsFormatting:
    def test_leading_zeros_preserved_in_micros(self) -> None:
        # p1715800000000001 → 1715800000.000001
        assert ts_packed_to_ts("1715800000000001") == "1715800000.000001"

    def test_normal_ts(self) -> None:
        assert ts_packed_to_ts("1715800000123456") == "1715800000.123456"

    def test_all_zero_micros(self) -> None:
        assert ts_packed_to_ts("1715800000000000") == "1715800000.000000"


class TestSlackPermalink_NonSlackURL:
    def test_non_slack_url_returns_none(self) -> None:
        assert slack_permalink_to_source_id("https://example.com/anything", _WORKSPACE_MAP) is None

    def test_empty_string_returns_none(self) -> None:
        assert slack_permalink_to_source_id("", _WORKSPACE_MAP) is None

    def test_plain_text_returns_none(self) -> None:
        assert slack_permalink_to_source_id("hello world", _WORKSPACE_MAP) is None

    def test_bare_slack_scheme_returns_none(self) -> None:
        # Bare slack:// URLs are handled by fn-86y.1, not this resolver.
        assert slack_permalink_to_source_id("slack://T012XYZ/C09ABCDEF/1715800000.123456", _WORKSPACE_MAP) is None


class TestSlackPermalink_TrailingThreadParam:
    def test_thread_ts_param_ignored(self) -> None:
        # thread_ts param captured but not used; main message ts returned.
        url = (
            "https://terra2.slack.com/archives/C09ABCDEF/p1715800000123456"
            "?thread_ts=1715799000.000100&cid=C09ABCDEF"
        )
        result = slack_permalink_to_source_id(url, _WORKSPACE_MAP)
        assert result == "slack://T012XYZ/C09ABCDEF/1715800000.123456"

    def test_cid_only_param_ignored(self) -> None:
        url = _BASE_URL + "?cid=C09ABCDEF"
        result = slack_permalink_to_source_id(url, _WORKSPACE_MAP)
        assert result == "slack://T012XYZ/C09ABCDEF/1715800000.123456"


class TestExtractSourceLinks_SlackPermalinkIntegration:
    """Wires the permalink resolver into extract_source_link_hints (fn-86y.1 helper)."""

    def test_produces_references_hint(self) -> None:
        text = f"Check this out: {_BASE_URL} and discuss."
        hints = extract_source_link_hints(
            text,
            source_id="google-calendar://acct/event/abc",
            subject_label="CalendarEvent",
            workspace_map=_WORKSPACE_MAP,
        )
        assert len(hints) == 1
        h = hints[0]
        assert h.predicate == "REFERENCES"
        assert h.object_label == "SlackMessage"
        assert h.object_id == "slack://T012XYZ/C09ABCDEF/1715800000.123456"
        assert h.subject_id == "google-calendar://acct/event/abc"
        assert h.subject_label == "CalendarEvent"
        assert h.confidence == 1.0

    def test_deduplication(self) -> None:
        text = f"{_BASE_URL} and again {_BASE_URL}"
        hints = extract_source_link_hints(
            text, source_id="test://subject", workspace_map=_WORKSPACE_MAP
        )
        assert len(hints) == 1

    def test_no_match_returns_empty(self) -> None:
        hints = extract_source_link_hints(
            "no slack links here",
            source_id="test://subject",
            workspace_map=_WORKSPACE_MAP,
        )
        assert hints == []

    def test_multiple_distinct_urls(self) -> None:
        url2 = "https://terra2.slack.com/archives/C00000001/p1700000000000001"
        text = f"{_BASE_URL} and also {url2}"
        hints = extract_source_link_hints(
            text, source_id="test://subject", workspace_map=_WORKSPACE_MAP
        )
        assert len(hints) == 2
        obj_ids = {h.object_id for h in hints}
        assert "slack://T012XYZ/C09ABCDEF/1715800000.123456" in obj_ids
        assert "slack://T012XYZ/C00000001/1700000000.000001" in obj_ids

    def test_default_subject_label(self) -> None:
        hints = extract_source_link_hints(
            _BASE_URL,
            source_id="obsidian:///vault/file.md",
            workspace_map=_WORKSPACE_MAP,
        )
        assert hints[0].subject_label == "File"

    def test_fallback_source_id_when_workspace_unknown(self) -> None:
        hints = extract_source_link_hints(
            _BASE_URL, source_id="test://subject", workspace_map={}
        )
        assert len(hints) == 1
        assert hints[0].object_id == "slack:///C09ABCDEF/1715800000.123456"


def _thread_event(
    team_id: str,
    channel_id: str,
    parent_ts: str,
    *,
    reply_ts: str | None = None,
) -> dict[str, Any]:
    """Minimal thread IngestEvent for parser tests."""
    msgs: list[dict[str, Any]] = [{"ts": parent_ts, "user": "U1", "text": "parent"}]
    if reply_ts:
        msgs.append({"ts": reply_ts, "user": "U2", "text": "reply"})
    return {
        "id": "ev-test",
        "source_type": "slack",
        "source_id": f"slack://{team_id}/{channel_id}/thread/{parent_ts}",
        "operation": "created",
        "text": "",
        "mime_type": "text/plain",
        "meta": {
            "team_id": team_id,
            "channel_id": channel_id,
            "channel_name": "general",
            "is_im": False,
            "is_mpim": False,
            "is_private": False,
            "is_archived": False,
            "kind": "thread",
            "parent_ts": parent_ts,
            "last_ts": (reply_ts or parent_ts),
            "message_ts": [m["ts"] for m in msgs],
            "messages": msgs,
            "users_info": {},
        },
    }


class TestPermalinkNodeIdentityAlignment:
    """Prove permalink REFERENCES and Slack parser CONTAINS_MESSAGE share the same source_id.

    The reviewer required that a permalink REFERENCES edge lands on the same
    SlackMessage node produced by Slack ingestion, not a disconnected placeholder.
    We verify this by comparing the object_id from each hint type.
    """

    def test_contains_message_object_id_matches_permalink_source_id(self) -> None:
        """CONTAINS_MESSAGE from parser == REFERENCES target from permalink resolver."""
        team_id = "T012XYZ"
        channel_id = "C09ABCDEF"
        ts_packed = "1715800000123456"
        ts = "1715800000.123456"

        # What the permalink resolver produces
        url = f"https://terra2.slack.com/archives/{channel_id}/p{ts_packed}"
        permalink_target = slack_permalink_to_source_id(url, {"terra2": team_id})
        assert permalink_target == f"slack://{team_id}/{channel_id}/{ts}"

        # What the Slack parser emits during thread ingestion
        event = _thread_event(team_id, channel_id, ts)
        parser = SlackParser()
        docs = parser.parse(event)
        parent_doc = docs[0]
        contains_hints = [
            h for h in parent_doc.graph_hints if h.predicate == "CONTAINS_MESSAGE"
        ]
        contains_ids = {h.object_id for h in contains_hints}
        assert permalink_target in contains_ids, (
            f"permalink target {permalink_target!r} not in CONTAINS_MESSAGE ids "
            f"{contains_ids!r}"
        )

    def test_thread_reply_ts_also_emits_contains_message(self) -> None:
        """Each message in a thread (parent and replies) gets its own CONTAINS_MESSAGE."""
        team_id = "T1"
        channel_id = "C99"
        parent_ts = "1715800000.000001"
        reply_ts = "1715800001.000002"
        event = _thread_event(team_id, channel_id, parent_ts, reply_ts=reply_ts)
        parser = SlackParser()
        docs = parser.parse(event)
        parent_doc = docs[0]
        contains_ids = {
            h.object_id
            for h in parent_doc.graph_hints
            if h.predicate == "CONTAINS_MESSAGE"
        }
        assert f"slack://{team_id}/{channel_id}/{parent_ts}" in contains_ids
        assert f"slack://{team_id}/{channel_id}/{reply_ts}" in contains_ids

    def test_e2e_calendar_event_with_slack_permalink(self) -> None:
        """A calendar/email/obsidian document with a Slack permalink produces
        a REFERENCES GraphHint whose object_id matches what Slack ingestion
        would emit as a CONTAINS_MESSAGE target for the same message.
        """
        team_id = "T012XYZ"
        channel_id = "C09ABCDEF"
        ts_packed = "1715800000123456"
        ts = "1715800000.123456"
        workspace_map = {"terra2": team_id}

        # Simulate a CalendarEvent document that references a Slack message
        cal_text = (
            f"Meeting notes: see discussion at "
            f"https://terra2.slack.com/archives/{channel_id}/p{ts_packed}"
        )
        hints = extract_source_link_hints(
            cal_text,
            source_id="google-calendar://acct/event/xyz",
            subject_label="CalendarEvent",
            workspace_map=workspace_map,
        )
        assert len(hints) == 1
        ref_hint = hints[0]
        assert ref_hint.predicate == "REFERENCES"
        assert ref_hint.object_label == "SlackMessage"
        expected_id = f"slack://{team_id}/{channel_id}/{ts}"
        assert ref_hint.object_id == expected_id

        # Confirm Slack ingestion of that message produces CONTAINS_MESSAGE to the same id
        event = _thread_event(team_id, channel_id, ts)
        parser = SlackParser()
        docs = parser.parse(event)
        contains_ids = {
            h.object_id
            for h in docs[0].graph_hints
            if h.predicate == "CONTAINS_MESSAGE"
        }
        assert expected_id in contains_ids, (
            "REFERENCES target from calendar doc does not match any "
            "CONTAINS_MESSAGE from Slack ingestion"
        )


class TestWorkspaceMapIO:
    def test_load_nonexistent_returns_empty(self, tmp_path: Path) -> None:
        result = load_workspace_team_map(tmp_path / "nope.json")
        assert result == {}

    def test_load_invalid_json_returns_empty(self, tmp_path: Path) -> None:
        p = tmp_path / "map.json"
        p.write_text("not json")
        assert load_workspace_team_map(p) == {}

    def test_load_non_dict_returns_empty(self, tmp_path: Path) -> None:
        p = tmp_path / "map.json"
        p.write_text('["terra2", "T012XYZ"]')
        assert load_workspace_team_map(p) == {}

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        p = tmp_path / "map.json"
        save_workspace_team_map("myworkspace", "T999", path=p)
        assert load_workspace_team_map(p) == {"myworkspace": "T999"}

    def test_save_preserves_existing_entries(self, tmp_path: Path) -> None:
        p = tmp_path / "map.json"
        save_workspace_team_map("ws1", "T001", path=p)
        save_workspace_team_map("ws2", "T002", path=p)
        assert load_workspace_team_map(p) == {"ws1": "T001", "ws2": "T002"}

    def test_save_noop_when_already_current(self, tmp_path: Path) -> None:
        p = tmp_path / "map.json"
        save_workspace_team_map("ws1", "T001", path=p)
        mtime_before = p.stat().st_mtime
        save_workspace_team_map("ws1", "T001", path=p)
        assert p.stat().st_mtime == mtime_before

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        p = tmp_path / "nested" / "deep" / "map.json"
        save_workspace_team_map("ws", "T1", path=p)
        assert p.exists()

    def test_save_ignores_empty_workspace(self, tmp_path: Path) -> None:
        p = tmp_path / "map.json"
        save_workspace_team_map("", "T001", path=p)
        assert not p.exists()

    def test_save_ignores_empty_team_id(self, tmp_path: Path) -> None:
        p = tmp_path / "map.json"
        save_workspace_team_map("ws1", "", path=p)
        assert not p.exists()
