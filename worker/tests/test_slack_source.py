"""Tests for the Slack ingestion source.

Covers backfill, polling cursor, threads, burst-window splitting (gap
and token), overlap, edits/deletes, and channel filters.  The Slack
WebClient is mocked at the module level via ``unittest.mock.MagicMock``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from worker.sources.slack import (
    DEFAULT_WINDOW_GAP_SECONDS,
    SlackSource,
    _build_thread_event,
    _build_window_event,
    _channel_passes_filter,
    _doc_id_thread,
    _doc_id_window,
    _estimate_tokens,
    _is_thread_parent,
    _is_thread_reply,
    _load_cursor,
    split_into_windows,
)


# ---------------------------------------------------------------------------
# Test queue (no dedup — we want to see every enqueue)
# ---------------------------------------------------------------------------


class _ListQueue:
    """Records every enqueue without dedup so tests can inspect history."""

    def __init__(self) -> None:
        self.enqueued: list[dict[str, Any]] = []
        self.cursors: dict[str, str] = {}

    def enqueue(
        self,
        event: dict[str, Any],
        cursor_key: str | None = None,
        cursor_value: str | None = None,
    ) -> str:
        self.enqueued.append(event)
        if cursor_key is not None and cursor_value is not None:
            self.cursors[cursor_key] = cursor_value
        return event.get("id", "")

    def is_enqueued(self, source_id: str) -> bool:
        return any(e.get("source_id") == source_id for e in self.enqueued)

    def load_cursor(self, key: str) -> str | None:
        return self.cursors.get(key)

    def save_cursor(self, key: str, value: str) -> None:
        self.cursors[key] = value


# ---------------------------------------------------------------------------
# Builders for fake Slack messages
# ---------------------------------------------------------------------------


def _msg(
    ts: float,
    text: str = "hi",
    user: str = "U1",
    *,
    thread_ts: float | None = None,
    subtype: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    m: dict[str, Any] = {
        "ts": f"{ts:.6f}",
        "user": user,
        "text": text,
    }
    if thread_ts is not None:
        m["thread_ts"] = f"{thread_ts:.6f}"
    if subtype is not None:
        m["subtype"] = subtype
    if extra:
        m.update(extra)
    return m


def _channel(
    cid: str = "C1",
    name: str = "general",
    *,
    is_archived: bool = False,
    is_im: bool = False,
    is_mpim: bool = False,
    is_private: bool = False,
) -> dict[str, Any]:
    return {
        "id": cid,
        "name": name,
        "is_archived": is_archived,
        "is_im": is_im,
        "is_mpim": is_mpim,
        "is_private": is_private,
    }


# ---------------------------------------------------------------------------
# Pure-function helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_doc_id_thread(self) -> None:
        assert _doc_id_thread("T", "C", "1.000") == "slack://T/C/thread/1.000"

    def test_doc_id_window(self) -> None:
        assert _doc_id_window("T", "C", "1.0", "5.0") == "slack://T/C/window/1.0-5.0"

    def test_estimate_tokens(self) -> None:
        assert _estimate_tokens("") == 0
        # 8 chars / 4 = 2 tokens
        assert _estimate_tokens("12345678") == 2
        # short strings still produce ≥1 token
        assert _estimate_tokens("hi") == 1

    def test_is_thread_parent_and_reply(self) -> None:
        parent = {"ts": "1.0", "thread_ts": "1.0"}
        reply = {"ts": "1.5", "thread_ts": "1.0"}
        plain = {"ts": "2.0"}
        assert _is_thread_parent(parent)
        assert not _is_thread_parent(reply)
        assert _is_thread_reply(reply)
        assert not _is_thread_reply(parent)
        assert not _is_thread_reply(plain)


# ---------------------------------------------------------------------------
# Channel filter
# ---------------------------------------------------------------------------


class TestChannelFilter:
    def test_drops_archived_when_disabled(self) -> None:
        ch = _channel(is_archived=True)
        assert not _channel_passes_filter(
            ch,
            include_channels=[],
            exclude_channels=[],
            include_dms=True,
            include_archived=False,
        )

    def test_keeps_archived_when_enabled(self) -> None:
        ch = _channel(is_archived=True)
        assert _channel_passes_filter(
            ch,
            include_channels=[],
            exclude_channels=[],
            include_dms=True,
            include_archived=True,
        )

    def test_drops_dms_when_disabled(self) -> None:
        ch = _channel(cid="D1", is_im=True)
        assert not _channel_passes_filter(
            ch,
            include_channels=[],
            exclude_channels=[],
            include_dms=False,
            include_archived=False,
        )

    def test_exclude_by_id(self) -> None:
        ch = _channel(cid="C42", name="random")
        assert not _channel_passes_filter(
            ch,
            include_channels=[],
            exclude_channels=["C42"],
            include_dms=True,
            include_archived=False,
        )

    def test_exclude_by_name(self) -> None:
        ch = _channel(cid="C1", name="random")
        assert not _channel_passes_filter(
            ch,
            include_channels=[],
            exclude_channels=["random"],
            include_dms=True,
            include_archived=False,
        )

    def test_include_only(self) -> None:
        ok = _channel(cid="C1", name="general")
        nope = _channel(cid="C2", name="random")
        kwargs = dict(
            include_channels=["general"],
            exclude_channels=[],
            include_dms=True,
            include_archived=False,
        )
        assert _channel_passes_filter(ok, **kwargs)
        assert not _channel_passes_filter(nope, **kwargs)


# ---------------------------------------------------------------------------
# Burst-window splitting
# ---------------------------------------------------------------------------


class TestSplitIntoWindows:
    def test_empty_returns_empty(self) -> None:
        assert split_into_windows([]) == []

    def test_single_message(self) -> None:
        msgs = [_msg(100.0, "hi")]
        assert split_into_windows(msgs) == [msgs]

    def test_no_split_within_gap(self) -> None:
        # 3 messages within 30 min — single window.
        msgs = [
            _msg(0.0, "a"),
            _msg(600.0, "b"),  # +10 min
            _msg(1200.0, "c"),  # +10 min
        ]
        windows = split_into_windows(msgs)
        assert len(windows) == 1
        assert [m["ts"] for m in windows[0]] == [m["ts"] for m in msgs]

    def test_gap_split_no_overlap(self) -> None:
        """Test 4: 3 messages, 45-min gap, 2 messages → 2 windows; no overlap on gap."""
        gap_seconds = 45 * 60  # 45 min — exceeds 30 min default
        msgs = [
            _msg(0.0, "a"),
            _msg(600.0, "b"),  # within burst 1
            _msg(1200.0, "c"),  # within burst 1
            _msg(1200.0 + gap_seconds, "d"),  # new burst (gap > default)
            _msg(1200.0 + gap_seconds + 300, "e"),  # within burst 2
        ]
        windows = split_into_windows(
            msgs,
            window_gap_seconds=DEFAULT_WINDOW_GAP_SECONDS,
            window_max_tokens=4096,
            window_overlap_messages=3,
        )
        assert len(windows) == 2
        # No overlap on gap-based split: window 2 is exactly d, e.
        assert [m["text"] for m in windows[0]] == ["a", "b", "c"]
        assert [m["text"] for m in windows[1]] == ["d", "e"]

    def test_token_split_with_overlap(self) -> None:
        """Test 5: 6 short messages exceeding 512 tokens split mid-burst with overlap."""
        # ~200 chars/msg → ~50 tokens.  Build a 12-msg burst so the first
        # token-driven close happens with at least 3 messages in the
        # closed window (so 3-message overlap is meaningful).
        text = "x" * 200
        msgs = [_msg(float(i * 30), text) for i in range(12)]
        windows = split_into_windows(
            msgs,
            window_gap_seconds=3600,  # large enough to never trigger
            window_max_tokens=512,
            window_overlap_messages=3,
        )
        # We expect at least one mid-burst split, with each follow-up
        # window opening on a 3-message overlap from the previous window.
        assert len(windows) >= 2, f"expected ≥2 windows, got {len(windows)}"
        first = windows[0]
        second = windows[1]
        assert len(first) >= 3
        overlap_ts = [m["ts"] for m in first[-3:]]
        assert [m["ts"] for m in second[:3]] == overlap_ts

    def test_token_split_zero_overlap(self) -> None:
        text = "y" * 200
        msgs = [_msg(float(i * 30), text) for i in range(12)]
        windows = split_into_windows(
            msgs,
            window_gap_seconds=3600,
            window_max_tokens=512,
            window_overlap_messages=0,
        )
        # No overlap → second window starts where first ended.
        assert len(windows) >= 2
        # Disjoint message sets.
        first_ts = {m["ts"] for m in windows[0]}
        second_ts = {m["ts"] for m in windows[1]}
        assert first_ts.isdisjoint(second_ts)


# ---------------------------------------------------------------------------
# Cursor file
# ---------------------------------------------------------------------------


class TestLoadCursor:
    def test_missing_file(self, tmp_path: Path) -> None:
        assert _load_cursor(tmp_path / "no.json") == {}

    def test_valid(self, tmp_path: Path) -> None:
        p = tmp_path / "cur.json"
        data = {"T1": {"C1": {"latest_ts": "100.0", "last_synced": "x"}}}
        p.write_text(json.dumps(data))
        assert _load_cursor(p) == data

    def test_corrupt_returns_empty(self, tmp_path: Path) -> None:
        p = tmp_path / "cur.json"
        p.write_text("not json")
        assert _load_cursor(p) == {}


# ---------------------------------------------------------------------------
# Event builders
# ---------------------------------------------------------------------------


class TestBuildEvents:
    def test_thread_event(self) -> None:
        ch = _channel("C1", "general")
        parent = _msg(100.0, "parent", thread_ts=100.0, user="U1")
        replies = [
            _msg(110.0, "reply1", thread_ts=100.0, user="U2"),
            _msg(120.0, "reply2", thread_ts=100.0, user="U3"),
        ]
        ev = _build_thread_event(
            team_id="T1", channel=ch, parent=parent, replies=replies
        )
        assert ev["source_type"] == "slack"
        assert ev["source_id"] == _doc_id_thread("T1", "C1", "100.000000")
        assert ev["operation"] == "created"
        assert ev["meta"]["kind"] == "thread"
        assert ev["meta"]["reply_count"] == 2
        assert ev["meta"]["users"] == ["U1", "U2", "U3"]
        assert "parent" in ev["text"] and "reply1" in ev["text"]

    def test_window_event(self) -> None:
        ch = _channel("C1", "general")
        msgs = [_msg(100.0, "a"), _msg(110.0, "b"), _msg(120.0, "c")]
        ev = _build_window_event(team_id="T1", channel=ch, messages=msgs)
        assert ev["meta"]["kind"] == "window"
        assert ev["meta"]["first_ts"] == "100.000000"
        assert ev["meta"]["last_ts"] == "120.000000"
        assert ev["meta"]["message_count"] == 3
        assert ev["source_id"] == _doc_id_window("T1", "C1", "100.000000", "120.000000")


# ---------------------------------------------------------------------------
# SlackSource — async behaviour
# ---------------------------------------------------------------------------


def _seed_source(source: SlackSource, *, team_id: str = "T1") -> None:
    """Wire a mock WebClient onto *source* to bypass auth.test."""
    source._client = MagicMock()
    source._team_id = team_id


class TestSlackSourceConfigure:
    def test_defaults(self) -> None:
        source = SlackSource()
        source.configure({})
        assert source._poll_interval == 300
        assert source._max_initial_days == 90
        assert source._window_max_tokens == 512
        assert source._window_gap_seconds == 1800
        assert source._window_overlap_messages == 3
        assert source._include_dms is True

    def test_overrides(self) -> None:
        source = SlackSource()
        source.configure(
            {
                "poll_interval_seconds": 60,
                "max_initial_days": 7,
                "window_max_tokens": 1024,
                "window_gap_seconds": 600,
                "window_overlap_messages": 1,
                "include_channels": ["general"],
                "include_dms": False,
            }
        )
        assert source._poll_interval == 60
        assert source._max_initial_days == 7
        assert source._window_max_tokens == 1024
        assert source._window_gap_seconds == 600
        assert source._window_overlap_messages == 1
        assert source._include_channels == ["general"]
        assert source._include_dms is False


@pytest.mark.asyncio
class TestSlackSourcePolling:
    async def test_backfill_emits_thread_and_window(self) -> None:
        """Backfill: 1 thread + 3 plain messages (single window) → 2 events."""
        source = SlackSource()
        source.configure({"poll_interval_seconds": 0})
        _seed_source(source)

        ch = _channel("C1", "general")
        thread_parent = _msg(100.0, "thread parent", thread_ts=100.0)
        plains = [_msg(200.0, "p1"), _msg(260.0, "p2"), _msg(320.0, "p3")]

        source._client.conversations_history.return_value = {
            "messages": [thread_parent, *plains],
            "has_more": False,
            "response_metadata": {"next_cursor": ""},
        }
        source._client.conversations_replies.return_value = {
            "messages": [thread_parent, _msg(105.0, "reply", thread_ts=100.0)],
            "has_more": False,
            "response_metadata": {"next_cursor": ""},
        }

        queue = _ListQueue()
        team_cursor: dict[str, dict[str, Any]] = {}
        await source._poll_conversation(ch, team_cursor, queue)

        kinds = [ev["meta"]["kind"] for ev in queue.enqueued]
        assert sorted(kinds) == ["thread", "window"]
        # Cursor advanced to latest plain message ts.
        assert team_cursor["C1"]["latest_ts"] == "320.000000"
        # The cursor was persisted with the last enqueue.
        assert "slack" in queue.cursors
        persisted = json.loads(queue.cursors["slack"])
        assert persisted["T1"]["C1"]["latest_ts"] == "320.000000"

    async def test_thread_with_five_replies_emits_one_event(self) -> None:
        """Test 6: parent + 5 replies = 1 IngestEvent with all 6 messages."""
        source = SlackSource()
        source.configure({})
        _seed_source(source)

        ch = _channel("C1", "general")
        parent = _msg(100.0, "parent", thread_ts=100.0, user="U1")
        replies = [
            _msg(100.0 + i, f"r{i}", thread_ts=100.0, user=f"U{i + 2}")
            for i in range(1, 6)
        ]
        source._client.conversations_history.return_value = {
            "messages": [parent],
            "has_more": False,
            "response_metadata": {"next_cursor": ""},
        }
        source._client.conversations_replies.return_value = {
            "messages": [parent, *replies],
            "has_more": False,
            "response_metadata": {"next_cursor": ""},
        }

        queue = _ListQueue()
        team_cursor: dict[str, dict[str, Any]] = {}
        await source._poll_conversation(ch, team_cursor, queue)

        thread_events = [e for e in queue.enqueued if e["meta"].get("kind") == "thread"]
        assert len(thread_events) == 1
        ev = thread_events[0]
        assert ev["meta"]["reply_count"] == 5
        assert len(ev["meta"]["message_ts"]) == 6

    async def test_burst_gap_split_emits_two_windows(self) -> None:
        """Test 4: 3 + gap + 2 messages → 2 window events."""
        source = SlackSource()
        source.configure({})
        _seed_source(source)
        ch = _channel("C1", "general")
        gap = 45 * 60
        plains = [
            _msg(0.0, "a"),
            _msg(600.0, "b"),
            _msg(1200.0, "c"),
            _msg(1200.0 + gap, "d"),
            _msg(1200.0 + gap + 60, "e"),
        ]
        source._client.conversations_history.return_value = {
            "messages": plains,
            "has_more": False,
            "response_metadata": {"next_cursor": ""},
        }

        queue = _ListQueue()
        team_cursor: dict[str, dict[str, Any]] = {}
        await source._poll_conversation(ch, team_cursor, queue)

        windows = [e for e in queue.enqueued if e["meta"]["kind"] == "window"]
        assert len(windows) == 2
        assert windows[0]["meta"]["message_count"] == 3
        assert windows[1]["meta"]["message_count"] == 2

    async def test_burst_token_split_emits_overlapping_windows(self) -> None:
        """Test 5: long burst split mid-burst with 3-message overlap."""
        source = SlackSource()
        source.configure({"window_max_tokens": 512, "window_overlap_messages": 3})
        _seed_source(source)
        ch = _channel("C1", "general")
        text = "z" * 200  # ~50 estimated tokens each
        plains = [_msg(float(i * 30), text) for i in range(12)]
        source._client.conversations_history.return_value = {
            "messages": plains,
            "has_more": False,
            "response_metadata": {"next_cursor": ""},
        }
        queue = _ListQueue()
        team_cursor: dict[str, dict[str, Any]] = {}
        await source._poll_conversation(ch, team_cursor, queue)

        windows = [e for e in queue.enqueued if e["meta"]["kind"] == "window"]
        assert len(windows) >= 2
        first_tail = windows[0]["meta"]["message_ts"][-3:]
        second_head = windows[1]["meta"]["message_ts"][:3]
        assert first_tail == second_head

    async def test_polling_honours_cursor(self) -> None:
        """Polling: cursor.latest_ts is passed to history as `oldest`."""
        source = SlackSource()
        source.configure({})
        _seed_source(source)
        ch = _channel("C1")

        new_msg = _msg(500.0, "fresh")
        source._client.conversations_history.return_value = {
            "messages": [new_msg],
            "has_more": False,
            "response_metadata": {"next_cursor": ""},
        }

        queue = _ListQueue()
        team_cursor: dict[str, dict[str, Any]] = {
            "C1": {"latest_ts": "300.000000", "last_synced": "x"}
        }
        await source._poll_conversation(ch, team_cursor, queue)

        # The history call should have used the existing latest_ts as oldest.
        call_kwargs = source._client.conversations_history.call_args.kwargs
        assert call_kwargs["oldest"] == "300.000000"
        # Cursor advances to the new max ts.
        assert team_cursor["C1"]["latest_ts"] == "500.000000"

    async def test_edit_emits_delete_then_modified(self) -> None:
        """Test 3: a late edit produces a delete + re-emit for the same doc id."""
        source = SlackSource()
        source.configure({})
        _seed_source(source)
        ch = _channel("C1")
        # Pre-populate the in-memory ts→doc map as if an earlier poll
        # had emitted a window containing ts=100.0.
        original_window_id = _doc_id_window("T1", "C1", "100.000000", "100.000000")
        source._ts_to_doc[("C1", "100.000000")] = original_window_id

        edited_msg = {
            "subtype": "message_changed",
            "ts": "200.000000",  # event ts (the change event)
            "message": {
                "ts": "100.000000",
                "user": "U1",
                "text": "edited body",
            },
            "previous_message": {
                "ts": "100.000000",
                "user": "U1",
                "text": "original body",
            },
        }
        source._client.conversations_history.side_effect = [
            # First call: the polling fetch returns the message_changed event.
            {
                "messages": [edited_msg],
                "has_more": False,
                "response_metadata": {"next_cursor": ""},
            },
            # Second call: the refetch around the edited ts returns the
            # rebuilt single message.
            {
                "messages": [_msg(100.0, "edited body")],
                "has_more": False,
                "response_metadata": {"next_cursor": ""},
            },
        ]

        queue = _ListQueue()
        team_cursor: dict[str, dict[str, Any]] = {
            "C1": {"latest_ts": "50.000000", "last_synced": "x"}
        }
        await source._poll_conversation(ch, team_cursor, queue)

        ops = [e["operation"] for e in queue.enqueued]
        ids = [e["source_id"] for e in queue.enqueued]
        # First a delete for the original document, then a modified rewrite
        # bearing the same source_id (since first/last ts of the rebuild
        # match the original).
        assert "deleted" in ops
        assert "modified" in ops
        delete_idx = ops.index("deleted")
        modify_idx = ops.index("modified")
        assert delete_idx < modify_idx
        assert ids[delete_idx] == original_window_id
        assert ids[modify_idx] == original_window_id

    async def test_archived_channel_excluded_at_discovery(self) -> None:
        """Discovery: archived channels are skipped when include_archived=False."""
        source = SlackSource()
        source.configure({"include_archived": False})
        _seed_source(source)
        source._client.conversations_list.return_value = {
            "channels": [
                _channel("C1", "general"),
                _channel("C2", "old", is_archived=True),
            ],
            "response_metadata": {"next_cursor": ""},
        }
        chans = source._discover_conversations()
        ids = sorted(c["id"] for c in chans)
        assert ids == ["C1"]

    async def test_exclude_channels_filter(self) -> None:
        """Discovery: exclude_channels by name drops the matching channel."""
        source = SlackSource()
        source.configure({"exclude_channels": ["random"]})
        _seed_source(source)
        source._client.conversations_list.return_value = {
            "channels": [
                _channel("C1", "general"),
                _channel("C2", "random"),
            ],
            "response_metadata": {"next_cursor": ""},
        }
        chans = source._discover_conversations()
        assert sorted(c["id"] for c in chans) == ["C1"]


# ---------------------------------------------------------------------------
# System / empty-bot subtype filter (fn-dge)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFilteredSubtypes:
    """The source must drop system and empty bot_messages BEFORE window
    splitting so the token budget driving chunk boundaries matches the
    text the parser ultimately renders."""

    async def test_source_filters_empty_bot_message_from_window(self) -> None:
        """A window with one real msg + one empty bot_message + one
        channel_join must emit a single window containing only the real
        message — both filtered events drop out at the source layer."""
        source = SlackSource()
        source.configure({})
        _seed_source(source)
        ch = _channel("C1", "general")

        history = [
            _msg(100.0, "real text", user="U1"),
            _msg(110.0, "", user="U1", subtype="channel_join"),
            _msg(120.0, "", user="", subtype="bot_message", extra={"bot_id": "B1"}),
        ]
        source._client.conversations_history.return_value = {
            "messages": history,
            "has_more": False,
            "response_metadata": {"next_cursor": ""},
        }

        queue = _ListQueue()
        team_cursor: dict[str, dict[str, Any]] = {}
        await source._poll_conversation(ch, team_cursor, queue)

        windows = [e for e in queue.enqueued if e["meta"]["kind"] == "window"]
        assert len(windows) == 1
        meta = windows[0]["meta"]
        assert meta["message_count"] == 1
        assert meta["message_ts"] == ["100.000000"]
        # Rendered body matches what the parser sees: only the real msg.
        assert "real text" in windows[0]["text"]
        assert "channel_join" not in windows[0]["text"]

    async def test_window_boundaries_after_filter(self) -> None:
        """Adding empty bot_messages between real messages must not
        shift burst-window boundaries — the source filter strips them
        before split_into_windows runs, so chunk count is stable."""
        cfg = {"window_max_tokens": 512, "window_overlap_messages": 0}
        ch = _channel("C1", "general")

        # Baseline: only real messages.
        text_chunk = "x" * 200  # ~50 estimated tokens each
        plain_msgs = [_msg(float(i * 30), text_chunk, user="U1") for i in range(12)]

        baseline = SlackSource()
        baseline.configure(cfg)
        _seed_source(baseline)
        baseline._client.conversations_history.return_value = {
            "messages": plain_msgs,
            "has_more": False,
            "response_metadata": {"next_cursor": ""},
        }
        baseline_queue = _ListQueue()
        await baseline._poll_conversation(ch, {}, baseline_queue)
        baseline_windows = [
            e for e in baseline_queue.enqueued if e["meta"]["kind"] == "window"
        ]

        # Same 12 real messages, but with an empty bot_message inserted
        # between every consecutive pair.  The filter strips them
        # before window splitting, so the resulting chunks are identical.
        polluted: list[dict[str, Any]] = []
        for i, m in enumerate(plain_msgs):
            polluted.append(m)
            polluted.append(
                _msg(
                    float(i * 30) + 0.5,
                    "",
                    user="",
                    subtype="bot_message",
                    extra={"bot_id": f"B{i}"},
                )
            )

        polluted_source = SlackSource()
        polluted_source.configure(cfg)
        _seed_source(polluted_source)
        polluted_source._client.conversations_history.return_value = {
            "messages": polluted,
            "has_more": False,
            "response_metadata": {"next_cursor": ""},
        }
        polluted_queue = _ListQueue()
        await polluted_source._poll_conversation(ch, {}, polluted_queue)
        polluted_windows = [
            e for e in polluted_queue.enqueued if e["meta"]["kind"] == "window"
        ]

        assert len(polluted_windows) == len(baseline_windows)
        for base_win, polluted_win in zip(baseline_windows, polluted_windows):
            assert polluted_win["meta"]["message_ts"] == base_win["meta"]["message_ts"]
            assert polluted_win["source_id"] == base_win["source_id"]


# ---------------------------------------------------------------------------
# Backfill resume / cursor monotonicity (fn-fhr)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBackfillResume:
    """Cursor-monotonicity invariants from fn-fhr.

    Before this fix, an interrupted backfill would persist its partial
    ``latest_ts`` and on resume the source treated that cursor as
    'polling forward' state, silently skipping every message older than
    the partial high-water mark.  The cursor schema now carries an
    explicit ``backfill_complete`` flag plus an ``oldest_ts_seen``
    resume point so a crash leaves a recoverable footprint.
    """

    async def test_backfill_resumes_from_oldest_ts_seen(self) -> None:
        source = SlackSource()
        source.configure({})
        _seed_source(source)
        ch = _channel("C1")

        # Mid-backfill cursor: latest_ts is the newest emitted, oldest_ts_seen
        # is how far back we've walked, backfill_complete is False.
        team_cursor: dict[str, dict[str, Any]] = {
            "C1": {
                "latest_ts": "500.000000",
                "oldest_ts_seen": "200.000000",
                "backfill_complete": False,
                "last_synced": "x",
            }
        }
        # Walker fetches one older page and then exhausts.
        source._client.conversations_history.return_value = {
            "messages": [_msg(150.0, "older")],
            "has_more": False,
            "response_metadata": {"next_cursor": ""},
        }

        queue = _ListQueue()
        await source._poll_conversation(ch, team_cursor, queue)

        call_kwargs = source._client.conversations_history.call_args.kwargs
        # Backfill walker passes `latest=oldest_ts_seen`; polling would have
        # passed `oldest=latest_ts` ("500.000000") instead.
        assert call_kwargs["latest"] == "200.000000"
        assert call_kwargs.get("inclusive") is False
        # Resume point advanced to the older ts emitted in this page.
        assert team_cursor["C1"]["oldest_ts_seen"] == "150.000000"
        # latest_ts must NOT regress to an older value.
        assert team_cursor["C1"]["latest_ts"] == "500.000000"

    async def test_backfill_complete_flips_to_polling(self) -> None:
        source = SlackSource()
        source.configure({})
        _seed_source(source)
        ch = _channel("C1")

        # Fresh state — no cursor entry.  First poll triggers backfill walk.
        team_cursor: dict[str, dict[str, Any]] = {}
        source._client.conversations_history.return_value = {
            "messages": [_msg(100.0, "a"), _msg(200.0, "b")],
            "has_more": False,
            "response_metadata": {"next_cursor": ""},
        }
        queue = _ListQueue()
        await source._poll_conversation(ch, team_cursor, queue)

        # Walk exhausted on the first page → flag flips.
        assert team_cursor["C1"]["backfill_complete"] is True
        assert team_cursor["C1"]["latest_ts"] == "200.000000"
        # Persisted cursor (via the explicit save_cursor at walk end)
        # carries the flipped flag.
        persisted = json.loads(queue.cursors["slack"])
        assert persisted["T1"]["C1"]["backfill_complete"] is True

        # Second poll: now we're in polling mode — `oldest=latest_ts`,
        # no `latest=` (which would be the backfill-walker shape).
        source._client.conversations_history.reset_mock()
        source._client.conversations_history.return_value = {
            "messages": [],
            "has_more": False,
            "response_metadata": {"next_cursor": ""},
        }
        await source._poll_conversation(ch, team_cursor, queue)

        call_kwargs = source._client.conversations_history.call_args.kwargs
        assert call_kwargs["oldest"] == "200.000000"
        assert "latest" not in call_kwargs

    async def test_polling_does_not_decrement_cursor(self) -> None:
        """Slack-Connect out-of-order timestamps must not regress latest_ts."""
        source = SlackSource()
        source.configure({})
        _seed_source(source)
        ch = _channel("C1")

        team_cursor: dict[str, dict[str, Any]] = {
            "C1": {
                "latest_ts": "500.000000",
                "backfill_complete": True,
                "last_synced": "x",
            }
        }
        # Page returns ts strictly older than latest_ts (workspace clock skew).
        source._client.conversations_history.return_value = {
            "messages": [_msg(450.0, "a"), _msg(480.0, "b")],
            "has_more": False,
            "response_metadata": {"next_cursor": ""},
        }
        queue = _ListQueue()
        await source._poll_conversation(ch, team_cursor, queue)

        # Cursor must hold at the prior high-water mark.
        assert team_cursor["C1"]["latest_ts"] == "500.000000"

    async def test_polling_blocked_until_backfill_complete(self) -> None:
        """When ``backfill_complete=False`` the source must not enter the
        forward-poll path, even if ``latest_ts`` is set."""
        source = SlackSource()
        source.configure({})
        _seed_source(source)
        ch = _channel("C1")

        team_cursor: dict[str, dict[str, Any]] = {
            "C1": {
                "latest_ts": "500.000000",
                "oldest_ts_seen": "100.000000",
                "backfill_complete": False,
                "last_synced": "x",
            }
        }
        source._client.conversations_history.return_value = {
            "messages": [],  # backfill exhausted
            "has_more": False,
            "response_metadata": {"next_cursor": ""},
        }
        queue = _ListQueue()
        await source._poll_conversation(ch, team_cursor, queue)

        call_kwargs = source._client.conversations_history.call_args.kwargs
        # Backfill walk uses `latest=oldest_ts_seen`; polling-style call
        # would have set `oldest=latest_ts`.  Assert the walker shape.
        assert call_kwargs["latest"] == "100.000000"
        assert call_kwargs.get("oldest") != "500.000000"
        # Empty page → clean exhaustion → flag flips.
        assert team_cursor["C1"]["backfill_complete"] is True

    async def test_legacy_cursor_treated_as_complete(self) -> None:
        """Cursors written by old code (no ``backfill_complete`` field) must
        not retroactively trigger a fresh backfill — that would double-emit
        90 days of history on every existing deployment."""
        source = SlackSource()
        source.configure({})
        _seed_source(source)
        ch = _channel("C1")

        team_cursor: dict[str, dict[str, Any]] = {
            "C1": {"latest_ts": "300.000000", "last_synced": "x"}
        }
        source._client.conversations_history.return_value = {
            "messages": [_msg(400.0, "new")],
            "has_more": False,
            "response_metadata": {"next_cursor": ""},
        }
        queue = _ListQueue()
        await source._poll_conversation(ch, team_cursor, queue)

        call_kwargs = source._client.conversations_history.call_args.kwargs
        # Legacy entry → polling forward, not backfill walk.
        assert call_kwargs["oldest"] == "300.000000"
        assert "latest" not in call_kwargs


# ---------------------------------------------------------------------------
# Attachments (fn-bu3)
# ---------------------------------------------------------------------------


class TestAttachmentExtraction:
    def test_window_event_includes_attachments_from_message_files(self) -> None:
        ch = _channel("C1", "general")
        msg_with_files = _msg(
            100.0,
            "see the docs",
            user="U1",
            extra={
                "files": [
                    {
                        "id": "F1",
                        "name": "report.pdf",
                        "mimetype": "application/pdf",
                        "filetype": "pdf",
                        "size": 4096,
                        "url_private_download": (
                            "https://files.slack.com/files-pri/T-F1/download/report.pdf"
                        ),
                        "user": "U1",
                    },
                    {
                        "id": "F2",
                        "name": "notes.docx",
                        "mimetype": (
                            "application/vnd.openxmlformats-officedocument."
                            "wordprocessingml.document"
                        ),
                        "filetype": "docx",
                        "size": 8192,
                        "url_private_download": (
                            "https://files.slack.com/files-pri/T-F2/download/notes.docx"
                        ),
                        "user": "U1",
                    },
                ]
            },
        )
        plain = _msg(110.0, "no files here", user="U2")
        ev = _build_window_event(
            team_id="T1",
            channel=ch,
            messages=[msg_with_files, plain],
            team_domain="acme",
        )
        atts = ev["meta"]["attachments"]
        assert len(atts) == 2
        ids = {a["id"] for a in atts}
        assert ids == {"F1", "F2"}
        # Each attachment carries its parent ts so the parser can place
        # markers next to the right message.
        assert all(a["ts"] == "100.000000" for a in atts)
        # Workspace identifiers travel on the parent meta, not per-file.
        assert ev["meta"]["team_domain"] == "acme"

    def test_thread_event_includes_attachments_from_each_reply(self) -> None:
        ch = _channel("C1", "general")
        parent = _msg(100.0, "kickoff", user="U1", thread_ts=100.0)
        reply_with_file = _msg(
            110.0,
            "here it is",
            user="U2",
            thread_ts=100.0,
            extra={
                "files": [
                    {
                        "id": "Fdiagram",
                        "name": "diagram.png",
                        "mimetype": "image/png",
                        "filetype": "png",
                        "size": 2048,
                        "url_private_download": "https://files.slack.com/png",
                        "user": "U2",
                    }
                ]
            },
        )
        ev = _build_thread_event(
            team_id="T1",
            channel=ch,
            parent=parent,
            replies=[reply_with_file],
            team_domain="acme",
        )
        atts = ev["meta"]["attachments"]
        assert [a["id"] for a in atts] == ["Fdiagram"]
        # Reply ts is preserved so the parser can indent the marker.
        assert atts[0]["ts"] == "110.000000"

    def test_bot_uploader_is_extracted_from_file_user(self) -> None:
        ch = _channel("C1", "general")
        # Message author is U1; file uploader is bot user B1.
        msg = _msg(
            100.0,
            "deploy notes",
            user="U1",
            extra={
                "files": [
                    {
                        "id": "Fbot",
                        "name": "deploy.pdf",
                        "mimetype": "application/pdf",
                        "size": 1024,
                        "url_private_download": "https://files.slack.com/pdf",
                        "user": "B1",  # bot uploader
                    }
                ]
            },
        )
        ev = _build_window_event(
            team_id="T1", channel=ch, messages=[msg], team_domain="acme"
        )
        att = ev["meta"]["attachments"][0]
        assert att["user"] == "B1"


class TestSlackSourceDownloadAttachments:
    """The 'download_files' field is gone; the alias survives at parse time."""

    def test_configure_reads_download_attachments_from_settings(self) -> None:
        source = SlackSource()
        source.configure({"download_attachments": True})
        assert source._download_attachments is True

    def test_configure_legacy_alias_promotes_to_download_attachments(self) -> None:
        # In production the alias is resolved by worker.config; this guards
        # the source against a hand-rolled settings dict that still uses
        # the legacy key (per fn-0yl).
        source = SlackSource()
        source.configure({"download_files": True})
        assert source._download_attachments is True

    def test_configure_attachment_knobs_propagate(self) -> None:
        source = SlackSource()
        source.configure(
            {
                "download_attachments": True,
                "attachment_indexable_mimetypes": ["application/pdf"],
                "attachment_max_size_mb": 5,
            }
        )
        assert source._attachment_indexable_mimetypes == ["application/pdf"]
        assert source._attachment_max_size_mb == 5

    def test_no_legacy_download_files_attribute_remains(self) -> None:
        # Defensive check: the renamed-out attribute must NOT come back.
        source = SlackSource()
        source.configure({"download_attachments": True})
        assert not hasattr(source, "_download_files")


# ---------------------------------------------------------------------------
# Users cache (workspace user directory) — fn-dob
# ---------------------------------------------------------------------------


def _users_list_response(
    members: list[dict[str, Any]], *, next_cursor: str = ""
) -> dict[str, Any]:
    return {
        "ok": True,
        "members": members,
        "response_metadata": {"next_cursor": next_cursor},
    }


class TestUsersCache:
    """Slack source must populate IngestEvent.meta['users_info'] from a
    cached users.list call (fn-dob).  Without this cache the parser
    falls back to '@Uxxx' rendering and Person nodes can't merge with
    Gmail/Calendar by canonical email.
    """

    def test_populates_on_refresh(self) -> None:
        """Single page of 5 users → cache holds 5 entries keyed by id."""
        source = SlackSource()
        source.configure({})
        _seed_source(source)
        members = [
            {
                "id": f"U{i}",
                "name": f"u{i}",
                "real_name": f"User {i}",
                "profile": {
                    "real_name": f"User {i}",
                    "email": f"user{i}@example.com",
                },
            }
            for i in range(5)
        ]
        source._client.users_list.return_value = _users_list_response(members)

        source._refresh_users_cache()

        assert len(source._users_cache) == 5
        assert source._users_cache["U0"]["profile"]["email"] == "user0@example.com"

    def test_paginates(self) -> None:
        """Three pages of users.list → cache merges all entries."""
        source = SlackSource()
        source.configure({})
        _seed_source(source)
        page1 = [{"id": "U1", "profile": {"email": "u1@e.com"}}]
        page2 = [{"id": "U2", "profile": {"email": "u2@e.com"}}]
        page3 = [{"id": "U3", "profile": {"email": "u3@e.com"}}]
        source._client.users_list.side_effect = [
            _users_list_response(page1, next_cursor="cur2"),
            _users_list_response(page2, next_cursor="cur3"),
            _users_list_response(page3, next_cursor=""),
        ]

        source._refresh_users_cache()

        assert sorted(source._users_cache) == ["U1", "U2", "U3"]
        # Pagination chained the cursors correctly.
        cursors_seen = [
            call.kwargs.get("cursor")
            for call in source._client.users_list.call_args_list
        ]
        # First call has no cursor; subsequent calls forward next_cursor.
        assert cursors_seen == [None, "cur2", "cur3"]

    def test_handles_missing_email_scope(self) -> None:
        """``users:read.email`` missing → entries lack profile.email; the
        parser already degrades to name-only (no exception)."""
        source = SlackSource()
        source.configure({})
        _seed_source(source)
        # When the token lacks users:read.email, Slack returns members
        # without an email field on the profile (or with it empty).  The
        # cache must still hold the entry — the parser handles the gap.
        members = [
            {
                "id": "U-NOEMAIL",
                "name": "anon",
                "real_name": "Anon User",
                "profile": {"real_name": "Anon User", "display_name": "anon"},
            }
        ]
        source._client.users_list.return_value = _users_list_response(members)

        source._refresh_users_cache()

        cached = source._users_cache["U-NOEMAIL"]
        assert cached["profile"].get("email") in (None, "")
        assert cached["real_name"] == "Anon User"

    def test_refresh_interval(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Within the interval window → no refetch.  After it elapses → refetch."""
        source = SlackSource()
        source.configure({"users_refresh_interval_seconds": 60})
        _seed_source(source)
        source._client.users_list.return_value = _users_list_response(
            [{"id": "U1", "profile": {"email": "u1@e.com"}}]
        )

        clock = {"now": 1000.0}
        monkeypatch.setattr("worker.sources.slack.time.monotonic", lambda: clock["now"])

        source._maybe_refresh_users_cache()
        assert source._client.users_list.call_count == 1

        # 30s elapsed — under the 60s interval, no extra call.
        clock["now"] = 1030.0
        source._maybe_refresh_users_cache()
        assert source._client.users_list.call_count == 1

        # 70s elapsed since last refresh — interval elapsed, refetch.
        clock["now"] = 1100.0
        source._maybe_refresh_users_cache()
        assert source._client.users_list.call_count == 2

    def test_api_error_preserves_previous_cache(self) -> None:
        """A transient SlackApiError must not blank the cache."""
        from slack_sdk.errors import SlackApiError

        source = SlackSource()
        source.configure({})
        _seed_source(source)
        source._users_cache = {"U-OLD": {"id": "U-OLD"}}
        source._client.users_list.side_effect = SlackApiError("rate_limited", {})

        source._refresh_users_cache()

        assert source._users_cache == {"U-OLD": {"id": "U-OLD"}}


@pytest.mark.asyncio
class TestUsersCacheInEvents:
    async def test_e2e_uses_real_users_cache(self) -> None:
        """End-to-end: the source's users.list call IS the source of the
        ``users_info`` dict on every emitted IngestEvent.  No test
        injection required.
        """
        source = SlackSource()
        source.configure({})
        _seed_source(source)
        source._client.users_list.return_value = _users_list_response(
            [
                {
                    "id": "U-ALICE",
                    "name": "alice",
                    "real_name": "Alice Example",
                    "profile": {
                        "real_name": "Alice Example",
                        "email": "alice@example.com",
                    },
                }
            ]
        )
        ch = _channel("C1", "general")
        source._client.conversations_history.return_value = {
            "messages": [_msg(100.0, "hi everyone", user="U-ALICE")],
            "has_more": False,
            "response_metadata": {"next_cursor": ""},
        }

        queue = _ListQueue()
        await source._poll_conversation(ch, {}, queue)

        assert queue.enqueued, "no events emitted"
        ev = queue.enqueued[0]
        users_info = ev["meta"].get("users_info") or {}
        assert "U-ALICE" in users_info
        assert users_info["U-ALICE"]["profile"]["email"] == "alice@example.com"

    async def test_poll_triggers_first_refresh(self) -> None:
        """``_poll_conversation`` must lazily populate the cache when empty."""
        source = SlackSource()
        source.configure({})
        _seed_source(source)
        source._client.users_list.return_value = _users_list_response(
            [{"id": "U1", "profile": {"email": "u1@e.com"}}]
        )
        source._client.conversations_history.return_value = {
            "messages": [],
            "has_more": False,
            "response_metadata": {"next_cursor": ""},
        }
        ch = _channel("C1")

        await source._poll_conversation(ch, {}, _ListQueue())

        assert source._client.users_list.called
        assert "U1" in source._users_cache


@pytest.mark.asyncio
async def test_cursor_is_persisted_atomically(tmp_path: Path) -> None:
    """Acceptance criterion 5: cursor is per-conversation, atomic, and survives restart."""
    from worker.sources.slack import _save_cursor

    cursor_path = tmp_path / "slack_cursor.json"
    cursor = {"T1": {"C1": {"latest_ts": "999.0", "last_synced": "x"}}}
    _save_cursor(cursor_path, cursor)
    assert cursor_path.exists()
    # Mode 0600 — owner read/write only (per save_json_atomic).
    assert cursor_path.stat().st_mode & 0o777 == 0o600
    assert _load_cursor(cursor_path) == cursor
