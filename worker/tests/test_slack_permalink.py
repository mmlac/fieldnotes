"""Tests for the Slack permalink → slack:// source_id resolver (fn-86y.3)."""

from __future__ import annotations

import json
from pathlib import Path

from worker.parsers._slack_permalink import (
    find_slack_permalink_source_ids,
    load_workspace_map,
    persist_workspace_map,
    resolve_slack_permalink,
)
from worker.parsers.base import extract_source_link_hints


_WORKSPACE_MAP = {"terra2": "T012XYZ"}


# ---------------------------------------------------------------------------
# _ts_from_packed
# ---------------------------------------------------------------------------


class TestTsFormatting:
    def test_standard_ts(self) -> None:
        from worker.parsers._slack_permalink import _ts_from_packed

        assert _ts_from_packed("1715800000123456") == "1715800000.123456"

    def test_leading_zeros_preserved(self) -> None:
        from worker.parsers._slack_permalink import _ts_from_packed

        assert _ts_from_packed("1715800000000001") == "1715800000.000001"

    def test_zeros_in_micros(self) -> None:
        from worker.parsers._slack_permalink import _ts_from_packed

        assert _ts_from_packed("1715800000000000") == "1715800000.000000"


# ---------------------------------------------------------------------------
# resolve_slack_permalink
# ---------------------------------------------------------------------------


class TestSlackPermalinkParsesURL:
    def test_basic_permalink(self) -> None:
        url = "https://terra2.slack.com/archives/C09ABCDEF/p1715800000123456"
        result = resolve_slack_permalink(url, _WORKSPACE_MAP)
        assert result == "slack://T012XYZ/C09ABCDEF/1715800000.123456"

    def test_workspace_with_hyphen(self) -> None:
        url = "https://my-workspace.slack.com/archives/C09ABCDEF/p1715800000000001"
        wmap = {"my-workspace": "TABC123"}
        result = resolve_slack_permalink(url, wmap)
        assert result == "slack://TABC123/C09ABCDEF/1715800000.000001"


class TestSlackPermalinkNoWorkspaceMap:
    def test_empty_map_returns_none(self) -> None:
        url = "https://terra2.slack.com/archives/C09ABCDEF/p1715800000123456"
        result = resolve_slack_permalink(url, {})
        assert result is None

    def test_allow_partial_returns_channel_ts(self) -> None:
        url = "https://terra2.slack.com/archives/C09ABCDEF/p1715800000123456"
        result = resolve_slack_permalink(url, {}, allow_partial=True)
        assert result == "slack:///C09ABCDEF/1715800000.123456"


class TestSlackPermalinkNonSlackURL:
    def test_non_slack_url_returns_none(self) -> None:
        assert resolve_slack_permalink("https://example.com/anything", _WORKSPACE_MAP) is None

    def test_bare_slack_scheme_not_matched(self) -> None:
        assert resolve_slack_permalink("slack://T012XYZ/C09ABCDEF/1715800000.123456", _WORKSPACE_MAP) is None

    def test_empty_string(self) -> None:
        assert resolve_slack_permalink("", _WORKSPACE_MAP) is None

    def test_plain_text(self) -> None:
        assert resolve_slack_permalink("no urls here", _WORKSPACE_MAP) is None


class TestSlackPermalinkTrailingThreadParam:
    def test_thread_ts_param_ignored(self) -> None:
        url = (
            "https://terra2.slack.com/archives/C09ABCDEF/p1715800000123456"
            "?thread_ts=1715799000.000100&cid=C09ABCDEF"
        )
        result = resolve_slack_permalink(url, _WORKSPACE_MAP)
        assert result == "slack://T012XYZ/C09ABCDEF/1715800000.123456"


# ---------------------------------------------------------------------------
# workspace map persistence
# ---------------------------------------------------------------------------


class TestWorkspaceMapPersistence:
    def test_persist_and_load(self, tmp_path: Path) -> None:
        map_file = tmp_path / "workspace_map.json"
        persist_workspace_map("myworkspace", "TMYTEAM", path=map_file)
        loaded = load_workspace_map(map_file)
        assert loaded == {"myworkspace": "TMYTEAM"}

    def test_persist_merges_existing(self, tmp_path: Path) -> None:
        map_file = tmp_path / "workspace_map.json"
        persist_workspace_map("ws1", "T001", path=map_file)
        persist_workspace_map("ws2", "T002", path=map_file)
        loaded = load_workspace_map(map_file)
        assert loaded == {"ws1": "T001", "ws2": "T002"}

    def test_persist_noop_on_empty(self, tmp_path: Path) -> None:
        map_file = tmp_path / "workspace_map.json"
        persist_workspace_map("", "T001", path=map_file)
        persist_workspace_map("ws1", "", path=map_file)
        assert not map_file.exists()

    def test_load_missing_file(self, tmp_path: Path) -> None:
        assert load_workspace_map(tmp_path / "nonexistent.json") == {}

    def test_load_malformed_json(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("not json")
        assert load_workspace_map(bad) == {}


# ---------------------------------------------------------------------------
# find_slack_permalink_source_ids
# ---------------------------------------------------------------------------


class TestFindSlackPermalinkSourceIds:
    def test_single_permalink(self) -> None:
        text = "see https://terra2.slack.com/archives/C09ABCDEF/p1715800000123456 for details"
        ids = find_slack_permalink_source_ids(text, _WORKSPACE_MAP)
        assert ids == ["slack://T012XYZ/C09ABCDEF/1715800000.123456"]

    def test_dedup(self) -> None:
        url = "https://terra2.slack.com/archives/C09ABCDEF/p1715800000123456"
        text = f"see {url} and also {url}"
        ids = find_slack_permalink_source_ids(text, _WORKSPACE_MAP)
        assert len(ids) == 1

    def test_multiple_permalinks(self) -> None:
        text = (
            "ref1: https://terra2.slack.com/archives/C09ABCDEF/p1715800000000001 "
            "ref2: https://terra2.slack.com/archives/C09ABCDEF/p1715800000000002"
        )
        ids = find_slack_permalink_source_ids(text, _WORKSPACE_MAP)
        assert len(ids) == 2

    def test_unknown_workspace_skipped(self) -> None:
        text = "see https://unknown.slack.com/archives/C09ABCDEF/p1715800000000001"
        ids = find_slack_permalink_source_ids(text, _WORKSPACE_MAP)
        assert ids == []

    def test_no_permalinks(self) -> None:
        assert find_slack_permalink_source_ids("plain text", _WORKSPACE_MAP) == []


# ---------------------------------------------------------------------------
# extract_source_link_hints (integration with base.py)
# ---------------------------------------------------------------------------


class TestExtractSourceLinksSlackPermalinkIntegration:
    def test_permalink_in_obsidian_note(self) -> None:
        text = (
            "Meeting notes. See also the discussion at "
            "https://terra2.slack.com/archives/C09ABCDEF/p1715800000123456 "
            "for follow-up."
        )
        hints = extract_source_link_hints(
            text,
            source_id="obsidian:///Notes/meeting.md",
            subject_label="ObsidianNote",
            slack_workspace_map=_WORKSPACE_MAP,
        )
        assert len(hints) == 1
        h = hints[0]
        assert h.predicate == "REFERENCES"
        assert h.object_id == "slack://T012XYZ/C09ABCDEF/1715800000.123456"
        assert h.object_label == "SlackMessage"
        assert h.subject_id == "obsidian:///Notes/meeting.md"
        assert h.subject_label == "ObsidianNote"
        assert h.confidence == 1.0

    def test_permalink_in_email(self) -> None:
        text = "FYI https://terra2.slack.com/archives/CABC123/p1715900000000001"
        hints = extract_source_link_hints(
            text,
            source_id="gmail://user@example.com/message/abc123",
            subject_label="EmailMessage",
            slack_workspace_map=_WORKSPACE_MAP,
        )
        assert len(hints) == 1
        assert hints[0].object_id == "slack://T012XYZ/CABC123/1715900000.000001"
        assert hints[0].subject_label == "EmailMessage"

    def test_no_permalink_returns_empty(self) -> None:
        hints = extract_source_link_hints(
            "just some text with no links",
            source_id="obsidian:///Notes/empty.md",
            slack_workspace_map=_WORKSPACE_MAP,
        )
        assert hints == []

    def test_unknown_workspace_returns_empty(self) -> None:
        text = "see https://unknown.slack.com/archives/C09ABCDEF/p1715800000123456"
        hints = extract_source_link_hints(
            text,
            source_id="obsidian:///Notes/meeting.md",
            slack_workspace_map={},
        )
        assert hints == []

    def test_dedup(self) -> None:
        url = "https://terra2.slack.com/archives/C09ABCDEF/p1715800000123456"
        text = f"see {url} and {url} twice"
        hints = extract_source_link_hints(
            text,
            source_id="obsidian:///Notes/note.md",
            slack_workspace_map=_WORKSPACE_MAP,
        )
        assert len(hints) == 1

    def test_loads_workspace_map_from_file(self, tmp_path: Path) -> None:
        map_file = tmp_path / "ws_map.json"
        map_file.write_text(json.dumps({"terra2": "T012XYZ"}))
        text = "see https://terra2.slack.com/archives/C09ABCDEF/p1715800000123456"
        hints = extract_source_link_hints(
            text,
            source_id="obsidian:///Notes/note.md",
            slack_workspace_map_path=map_file,
        )
        assert len(hints) == 1
        assert hints[0].object_id == "slack://T012XYZ/C09ABCDEF/1715800000.123456"
