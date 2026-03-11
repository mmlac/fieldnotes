"""Tests for Gmail source async methods and helpers (sources/gmail.py).

Covers: _load_cursor, _save_cursor, _extract_recipients, _backfill,
_poll_incremental, and rate-limiting/backoff in backfill.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from worker.sources.gmail import (
    BACKFILL_PAGE_DELAY,
    GmailSource,
    _extract_recipients,
    _load_cursor,
    _save_cursor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gmail_message(
    msg_id: str = "msg-1",
    thread_id: str = "thread-1",
    history_id: str = "100",
    subject: str = "Hello",
    sender: str = "a@b.com",
    to: str = "c@d.com",
    internal_date: str = "1700000000000",
) -> dict[str, Any]:
    return {
        "id": msg_id,
        "threadId": thread_id,
        "historyId": history_id,
        "internalDate": internal_date,
        "payload": {
            "headers": [
                {"name": "Subject", "value": subject},
                {"name": "From", "value": sender},
                {"name": "To", "value": to},
                {"name": "Date", "value": "Mon, 1 Jan 2024 00:00:00 +0000"},
            ],
        },
    }


def _mock_messages_api(messages: list[dict[str, Any]], page_token: str | None = None):
    """Build a mock Gmail messages API that returns given messages."""
    api = MagicMock()

    stubs = [{"id": m["id"]} for m in messages]
    list_result = {"messages": stubs}
    if page_token:
        list_result["nextPageToken"] = page_token

    list_req = MagicMock()
    list_req.execute.return_value = list_result
    api.list.return_value = list_req

    msg_map = {m["id"]: m for m in messages}

    def get_side_effect(**kwargs):
        req = MagicMock()
        req.execute.return_value = msg_map[kwargs["id"]]
        return req

    api.get.side_effect = get_side_effect

    return api


# ---------------------------------------------------------------------------
# _load_cursor / _save_cursor
# ---------------------------------------------------------------------------

class TestLoadCursor:
    def test_returns_none_when_file_missing(self, tmp_path: Path) -> None:
        assert _load_cursor(tmp_path / "nope.json") is None

    def test_reads_history_id(self, tmp_path: Path) -> None:
        f = tmp_path / "cursor.json"
        f.write_text(json.dumps({"history_id": "42"}))
        assert _load_cursor(f) == "42"

    def test_returns_none_on_invalid_json(self, tmp_path: Path) -> None:
        f = tmp_path / "cursor.json"
        f.write_text("not-json")
        assert _load_cursor(f) is None

    def test_returns_none_on_missing_key(self, tmp_path: Path) -> None:
        f = tmp_path / "cursor.json"
        f.write_text(json.dumps({"other": "data"}))
        assert _load_cursor(f) is None


class TestSaveCursor:
    def test_writes_history_id(self, tmp_path: Path) -> None:
        f = tmp_path / "cursor.json"
        _save_cursor(f, "99")
        data = json.loads(f.read_text())
        assert data["history_id"] == "99"

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        f = tmp_path / "a" / "b" / "cursor.json"
        _save_cursor(f, "1")
        assert f.exists()

    def test_sets_restrictive_permissions(self, tmp_path: Path) -> None:
        f = tmp_path / "cursor.json"
        _save_cursor(f, "42")
        assert f.stat().st_mode & 0o777 == 0o600


# ---------------------------------------------------------------------------
# _extract_recipients
# ---------------------------------------------------------------------------

class TestExtractRecipients:
    def test_extracts_to_and_cc(self) -> None:
        headers = [
            {"name": "To", "value": "a@b.com, c@d.com"},
            {"name": "Cc", "value": "e@f.com"},
            {"name": "Subject", "value": "ignored"},
        ]
        result = _extract_recipients(headers)
        assert result == ["a@b.com", "c@d.com", "e@f.com"]

    def test_empty_headers(self) -> None:
        assert _extract_recipients([]) == []

    def test_empty_values_skipped(self) -> None:
        headers = [{"name": "To", "value": ""}]
        assert _extract_recipients(headers) == []

    def test_case_insensitive(self) -> None:
        headers = [{"name": "to", "value": "x@y.com"}]
        assert _extract_recipients(headers) == ["x@y.com"]

    def test_whitespace_stripped(self) -> None:
        headers = [{"name": "To", "value": "  a@b.com , c@d.com  "}]
        result = _extract_recipients(headers)
        assert result == ["a@b.com", "c@d.com"]


# ---------------------------------------------------------------------------
# GmailSource._backfill
# ---------------------------------------------------------------------------

class TestBackfill:
    @pytest.mark.asyncio
    async def test_backfill_fetches_messages_and_returns_history_id(self) -> None:
        msgs = [
            _gmail_message("m1", history_id="10"),
            _gmail_message("m2", history_id="20"),
        ]
        api = _mock_messages_api(msgs)

        source = GmailSource()
        source._max_initial_threads = 100
        source._label_filter = "INBOX"

        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        history_id = await source._backfill(api, queue)

        assert history_id == "20"  # highest
        assert queue.qsize() == 2

    @pytest.mark.asyncio
    async def test_backfill_respects_max_limit(self) -> None:
        msgs = [_gmail_message(f"m{i}", history_id=str(i)) for i in range(10)]
        api = _mock_messages_api(msgs)

        source = GmailSource()
        source._max_initial_threads = 3
        source._label_filter = "INBOX"

        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        await source._backfill(api, queue)

        assert queue.qsize() == 3

    @pytest.mark.asyncio
    async def test_backfill_handles_empty_result(self) -> None:
        api = MagicMock()
        list_req = MagicMock()
        list_req.execute.return_value = {"messages": []}
        api.list.return_value = list_req

        source = GmailSource()
        source._max_initial_threads = 100
        source._label_filter = "INBOX"

        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        result = await source._backfill(api, queue)

        assert result is None
        assert queue.qsize() == 0

    @pytest.mark.asyncio
    async def test_backfill_handles_no_messages_key(self) -> None:
        api = MagicMock()
        list_req = MagicMock()
        list_req.execute.return_value = {}
        api.list.return_value = list_req

        source = GmailSource()
        source._max_initial_threads = 100
        source._label_filter = "INBOX"

        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        result = await source._backfill(api, queue)

        assert result is None
        assert queue.qsize() == 0


# ---------------------------------------------------------------------------
# GmailSource._poll_incremental
# ---------------------------------------------------------------------------

class TestPollIncremental:
    @pytest.mark.asyncio
    async def test_fetches_new_messages(self, tmp_path: Path) -> None:
        new_msg = _gmail_message("m-new", history_id="200")
        messages_api = MagicMock()
        get_req = MagicMock()
        get_req.execute.return_value = new_msg
        messages_api.get.return_value = get_req

        service = MagicMock()
        history_api = MagicMock()
        history_list_req = MagicMock()
        history_list_req.execute.return_value = {
            "historyId": "200",
            "history": [
                {"messagesAdded": [{"message": {"id": "m-new"}}]},
            ],
        }
        history_api.list.return_value = history_list_req
        service.users.return_value.history.return_value = history_api

        source = GmailSource()
        source._cursor_path = tmp_path / "cursor.json"
        source._label_filter = "INBOX"

        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        new_cursor = await source._poll_incremental(
            service, messages_api, queue, "100"
        )

        assert new_cursor == "200"
        assert queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_returns_old_cursor_on_no_cursor(self, tmp_path: Path) -> None:
        source = GmailSource()
        source._cursor_path = tmp_path / "cursor.json"

        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        result = await source._poll_incremental(
            MagicMock(), MagicMock(), queue, None
        )

        assert result is None
        assert queue.qsize() == 0

    @pytest.mark.asyncio
    async def test_deduplicates_messages(self, tmp_path: Path) -> None:
        """Same message ID appearing twice in history should only be fetched once."""
        msg = _gmail_message("m-dup", history_id="300")
        messages_api = MagicMock()
        get_req = MagicMock()
        get_req.execute.return_value = msg
        messages_api.get.return_value = get_req

        service = MagicMock()
        history_api = MagicMock()
        history_list_req = MagicMock()
        history_list_req.execute.return_value = {
            "historyId": "300",
            "history": [
                {"messagesAdded": [{"message": {"id": "m-dup"}}]},
                {"messagesAdded": [{"message": {"id": "m-dup"}}]},
            ],
        }
        history_api.list.return_value = history_list_req
        service.users.return_value.history.return_value = history_api

        source = GmailSource()
        source._cursor_path = tmp_path / "cursor.json"
        source._label_filter = "INBOX"

        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        await source._poll_incremental(service, messages_api, queue, "100")

        assert queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_handles_api_error_gracefully(self, tmp_path: Path) -> None:
        """API error during history fetch returns old cursor."""
        service = MagicMock()
        history_api = MagicMock()
        history_list_req = MagicMock()
        history_list_req.execute.side_effect = Exception("API error")
        history_api.list.return_value = history_list_req
        service.users.return_value.history.return_value = history_api

        source = GmailSource()
        source._cursor_path = tmp_path / "cursor.json"
        source._label_filter = "INBOX"

        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        result = await source._poll_incremental(
            service, MagicMock(), queue, "100"
        )

        assert result == "100"
        assert queue.qsize() == 0

    @pytest.mark.asyncio
    async def test_saves_new_cursor_to_disk(self, tmp_path: Path) -> None:
        service = MagicMock()
        history_api = MagicMock()
        history_list_req = MagicMock()
        history_list_req.execute.return_value = {
            "historyId": "500",
            "history": [],
        }
        history_api.list.return_value = history_list_req
        service.users.return_value.history.return_value = history_api

        cursor_file = tmp_path / "cursor.json"
        source = GmailSource()
        source._cursor_path = cursor_file
        source._label_filter = "INBOX"

        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        await source._poll_incremental(service, MagicMock(), queue, "400")

        data = json.loads(cursor_file.read_text())
        assert data["history_id"] == "500"


# ---------------------------------------------------------------------------
# Rate limiting and retry
# ---------------------------------------------------------------------------

class TestApiCallWithRetry:
    @pytest.mark.asyncio
    async def test_succeeds_on_first_try(self) -> None:
        result = await GmailSource._api_call_with_retry(
            asyncio.get_running_loop(),
            lambda: "ok",
        )
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_retries_on_transient_failure(self) -> None:
        call_count = 0

        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("transient")
            return "recovered"

        with patch("worker.sources.gmail.asyncio.sleep", new_callable=AsyncMock):
            result = await GmailSource._api_call_with_retry(
                asyncio.get_running_loop(),
                flaky,
                max_retries=3,
                initial_backoff=0.01,
            )

        assert result == "recovered"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self) -> None:
        def always_fail():
            raise Exception("permanent")

        with (
            patch("worker.sources.gmail.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(Exception, match="permanent"),
        ):
            await GmailSource._api_call_with_retry(
                asyncio.get_running_loop(),
                always_fail,
                max_retries=2,
                initial_backoff=0.01,
            )

    @pytest.mark.asyncio
    async def test_exponential_backoff_delays(self) -> None:
        """Backoff doubles on each retry."""
        call_count = 0

        def fail_twice():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("fail")
            return "ok"

        sleep_calls: list[float] = []

        async def mock_sleep(seconds):
            sleep_calls.append(seconds)

        with patch("worker.sources.gmail.asyncio.sleep", side_effect=mock_sleep):
            await GmailSource._api_call_with_retry(
                asyncio.get_running_loop(),
                fail_twice,
                max_retries=3,
                initial_backoff=1.0,
            )

        assert len(sleep_calls) == 2
        assert sleep_calls[0] == 1.0
        assert sleep_calls[1] == 2.0


class TestBackfillRateLimiting:
    @pytest.mark.asyncio
    async def test_backfill_pauses_between_pages(self) -> None:
        """Backfill should sleep between page fetches to avoid rate limits."""
        page1_msgs = [_gmail_message(f"m{i}", history_id=str(i)) for i in range(3)]
        page2_msgs = [_gmail_message(f"n{i}", history_id=str(100 + i)) for i in range(2)]

        call_count = [0]

        def list_side_effect(**kwargs):
            call_count[0] += 1
            req = MagicMock()
            if call_count[0] == 1:
                req.execute.return_value = {
                    "messages": [{"id": m["id"]} for m in page1_msgs],
                    "nextPageToken": "page2",
                }
            else:
                req.execute.return_value = {
                    "messages": [{"id": m["id"]} for m in page2_msgs],
                }
            return req

        all_msgs = {m["id"]: m for m in page1_msgs + page2_msgs}

        api = MagicMock()
        api.list.side_effect = list_side_effect

        def get_side_effect(**kwargs):
            req = MagicMock()
            req.execute.return_value = all_msgs[kwargs["id"]]
            return req

        api.get.side_effect = get_side_effect

        source = GmailSource()
        source._max_initial_threads = 100
        source._label_filter = "INBOX"

        sleep_calls: list[float] = []
        original_sleep = asyncio.sleep

        async def mock_sleep(seconds):
            sleep_calls.append(seconds)

        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        with patch("worker.sources.gmail.asyncio.sleep", side_effect=mock_sleep):
            await source._backfill(api, queue)

        assert queue.qsize() == 5
        # Should have paused between pages
        assert BACKFILL_PAGE_DELAY in sleep_calls
