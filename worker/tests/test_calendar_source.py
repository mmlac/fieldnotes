"""Tests for Google Calendar source helpers and async methods."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from worker.sources.calendar import (
    GoogleCalendarSource,
    _build_ingest_event,
    _event_end_iso,
    _event_start_iso,
    _load_cursor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _TestQueue:
    """Thin wrapper around asyncio.Queue that exposes PersistentQueue API."""

    def __init__(self) -> None:
        self._q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._cursors: dict[str, str] = {}
        self._enqueued_ids: set[str] = set()

    def enqueue(
        self,
        event: dict[str, Any],
        cursor_key: str | None = None,
        cursor_value: str | None = None,
    ) -> str:
        self._q.put_nowait(event)
        sid = event.get("source_id", "")
        if sid:
            self._enqueued_ids.add(sid)
        if cursor_key is not None and cursor_value is not None:
            self._cursors[cursor_key] = cursor_value
        return event.get("id", "")

    def is_enqueued(self, source_id: str) -> bool:
        return source_id in self._enqueued_ids

    def load_cursor(self, key: str) -> str | None:
        return self._cursors.get(key)

    def save_cursor(self, key: str, value: str) -> None:
        self._cursors[key] = value

    async def get(self) -> dict[str, Any]:
        return await self._q.get()

    def get_nowait(self) -> dict[str, Any]:
        return self._q.get_nowait()

    def qsize(self) -> int:
        return self._q.qsize()


def _calendar_event(
    event_id: str = "evt-1",
    summary: str = "Team Meeting",
    description: str = "Weekly sync",
    location: str = "Room A",
    start_dt: str = "2026-03-21T09:00:00-07:00",
    end_dt: str = "2026-03-21T09:30:00-07:00",
    status: str = "confirmed",
    organizer_email: str = "alice@example.com",
    attendees: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "id": event_id,
        "status": status,
        "summary": summary,
        "description": description,
        "location": location,
        "start": {"dateTime": start_dt},
        "end": {"dateTime": end_dt},
        "organizer": {"email": organizer_email, "displayName": "Alice"},
        "creator": {"email": organizer_email},
        "htmlLink": f"https://calendar.google.com/event?eid={event_id}",
        "updated": "2026-03-21T08:00:00Z",
        "attendees": attendees or [],
    }
    return event


def _all_day_event(
    event_id: str = "evt-ad",
    summary: str = "All Day Event",
) -> dict[str, Any]:
    return {
        "id": event_id,
        "status": "confirmed",
        "summary": summary,
        "start": {"date": "2026-03-21"},
        "end": {"date": "2026-03-22"},
        "organizer": {"email": "alice@example.com"},
        "creator": {"email": "alice@example.com"},
        "updated": "2026-03-20T12:00:00Z",
    }


# ---------------------------------------------------------------------------
# _load_cursor
# ---------------------------------------------------------------------------


class TestLoadCursor:
    def test_returns_empty_when_file_missing(self, tmp_path: Path) -> None:
        assert _load_cursor(tmp_path / "nope.json") == {}

    def test_returns_dict_from_valid_json(self, tmp_path: Path) -> None:
        data = {"primary": "sync-token-123"}
        cursor = tmp_path / "cursor.json"
        cursor.write_text(json.dumps(data))
        assert _load_cursor(cursor) == data

    def test_returns_empty_on_invalid_json(self, tmp_path: Path) -> None:
        cursor = tmp_path / "cursor.json"
        cursor.write_text("not json")
        assert _load_cursor(cursor) == {}

    def test_returns_empty_on_non_dict(self, tmp_path: Path) -> None:
        cursor = tmp_path / "cursor.json"
        cursor.write_text('"just a string"')
        assert _load_cursor(cursor) == {}


# ---------------------------------------------------------------------------
# _event_start_iso / _event_end_iso
# ---------------------------------------------------------------------------


class TestEventTimeHelpers:
    def test_datetime_event(self) -> None:
        event = {"start": {"dateTime": "2026-03-21T09:00:00-07:00"}}
        assert _event_start_iso(event) == "2026-03-21T09:00:00-07:00"

    def test_all_day_event(self) -> None:
        event = {"start": {"date": "2026-03-21"}}
        assert _event_start_iso(event) == "2026-03-21"

    def test_end_datetime(self) -> None:
        event = {"end": {"dateTime": "2026-03-21T10:00:00-07:00"}}
        assert _event_end_iso(event) == "2026-03-21T10:00:00-07:00"

    def test_missing_start(self) -> None:
        assert _event_start_iso({}) == ""


# ---------------------------------------------------------------------------
# _build_ingest_event
# ---------------------------------------------------------------------------


class TestBuildIngestEvent:
    def test_basic_event(self) -> None:
        event = _calendar_event()
        result = _build_ingest_event(event, "primary")
        assert result is not None
        assert result["source_type"] == "google_calendar"
        assert result["source_id"] == "gcal:primary:evt-1"
        assert result["operation"] == "created"
        assert "Team Meeting" in result["text"]

    def test_cancelled_event_is_delete(self) -> None:
        event = _calendar_event(status="cancelled")
        result = _build_ingest_event(event, "primary")
        assert result is not None
        assert result["operation"] == "deleted"

    def test_meta_fields(self) -> None:
        event = _calendar_event()
        result = _build_ingest_event(event, "primary")
        meta = result["meta"]
        assert meta["event_id"] == "evt-1"
        assert meta["calendar_id"] == "primary"
        assert meta["summary"] == "Team Meeting"
        assert meta["location"] == "Room A"
        assert meta["organizer_email"] == "alice@example.com"
        assert meta["start_time"] == "2026-03-21T09:00:00-07:00"

    def test_attendees_normalized(self) -> None:
        attendees = [
            {"email": "Bob@Example.com", "displayName": "Bob", "responseStatus": "accepted"},
            {"email": "carol@example.com"},
        ]
        event = _calendar_event(attendees=attendees)
        result = _build_ingest_event(event, "primary")
        meta_attendees = result["meta"]["attendees"]
        assert len(meta_attendees) == 2
        assert meta_attendees[0]["email"] == "bob@example.com"
        assert meta_attendees[0]["name"] == "Bob"

    def test_all_day_event(self) -> None:
        event = _all_day_event()
        result = _build_ingest_event(event, "work")
        assert result["source_id"] == "gcal:work:evt-ad"
        assert result["meta"]["start_time"] == "2026-03-21"

    def test_body_includes_location_and_attendees(self) -> None:
        attendees = [
            {"email": "bob@example.com", "displayName": "Bob Jones"},
        ]
        event = _calendar_event(attendees=attendees)
        result = _build_ingest_event(event, "primary")
        assert "Room A" in result["text"]
        assert "Bob Jones" in result["text"]

    def test_no_title_fallback(self) -> None:
        event = _calendar_event()
        del event["summary"]
        result = _build_ingest_event(event, "primary")
        assert result["meta"]["summary"] == "(No title)"


# ---------------------------------------------------------------------------
# GoogleCalendarSource
# ---------------------------------------------------------------------------


class TestGoogleCalendarSource:
    def test_name(self) -> None:
        source = GoogleCalendarSource()
        assert source.name() == "google_calendar"

    def test_configure_defaults(self) -> None:
        source = GoogleCalendarSource()
        source.configure({})
        assert source._poll_interval == 300
        assert source._max_initial_days == 90
        assert source._calendar_ids == ["primary"]

    def test_configure_custom(self) -> None:
        source = GoogleCalendarSource()
        source.configure({
            "poll_interval_seconds": 600,
            "max_initial_days": 30,
            "calendar_ids": ["primary", "work@group.calendar.google.com"],
            "client_secrets_path": "/tmp/secrets.json",
        })
        assert source._poll_interval == 600
        assert source._max_initial_days == 30
        assert len(source._calendar_ids) == 2
        assert source._client_secrets_path == "/tmp/secrets.json"

    @pytest.mark.asyncio
    async def test_missing_secrets_stays_idle(self, tmp_path: Path) -> None:
        """Source should log error and stay alive when secrets file is missing."""
        source = GoogleCalendarSource()
        source.configure({
            "client_secrets_path": str(tmp_path / "nonexistent.json"),
        })
        queue = _TestQueue()

        task = asyncio.create_task(source.start(queue))
        # Give it a moment to enter the idle loop
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_poll_calendar_initial_backfill(self) -> None:
        """_poll_calendar should fetch events and return a sync token."""
        source = GoogleCalendarSource()
        source.configure({})
        queue = _TestQueue()

        events_list_result = {
            "items": [_calendar_event()],
            "nextSyncToken": "sync-abc",
        }
        mock_events = MagicMock()
        mock_list = MagicMock()
        mock_list.execute.return_value = events_list_result
        mock_events.list.return_value = mock_list
        mock_service = MagicMock()
        mock_service.events.return_value = mock_events

        token = await source._poll_calendar(
            mock_service, queue, "primary", None, {}
        )

        assert token == "sync-abc"
        assert queue.qsize() > 0
        event = queue.get_nowait()
        assert event["source_type"] == "google_calendar"
        assert event["source_id"] == "gcal:primary:evt-1"
        # orderBy must NOT be set during backfill — it causes the
        # Google Calendar API to omit nextSyncToken from the response.
        call_kwargs = mock_events.list.call_args[1]
        assert "orderBy" not in call_kwargs

    @pytest.mark.asyncio
    async def test_poll_calendar_incremental_sync(self) -> None:
        """With a sync token, should pass syncToken and showDeleted."""
        source = GoogleCalendarSource()
        source.configure({})
        queue = _TestQueue()

        events_list_result = {
            "items": [],
            "nextSyncToken": "sync-def",
        }
        mock_events = MagicMock()
        mock_list = MagicMock()
        mock_list.execute.return_value = events_list_result
        mock_events.list.return_value = mock_list
        mock_service = MagicMock()
        mock_service.events.return_value = mock_events

        token = await source._poll_calendar(
            mock_service, queue, "primary", "sync-old", {}
        )

        assert token == "sync-def"
        # Verify syncToken was passed
        call_kwargs = mock_events.list.call_args
        assert call_kwargs[1]["syncToken"] == "sync-old"
        assert call_kwargs[1]["showDeleted"] is True

    @pytest.mark.asyncio
    async def test_poll_calendar_pagination(self) -> None:
        """Should handle paginated results."""
        source = GoogleCalendarSource()
        source.configure({})
        queue = _TestQueue()

        page1 = {
            "items": [_calendar_event(event_id="evt-1")],
            "nextPageToken": "page2",
        }
        page2 = {
            "items": [_calendar_event(event_id="evt-2")],
            "nextSyncToken": "sync-final",
        }

        mock_events = MagicMock()
        mock_list = MagicMock()
        mock_list.execute.side_effect = [page1, page2]
        mock_events.list.return_value = mock_list
        mock_service = MagicMock()
        mock_service.events.return_value = mock_events

        token = await source._poll_calendar(
            mock_service, queue, "primary", None, {}
        )

        assert token == "sync-final"
        assert queue.qsize() == 2

    @pytest.mark.asyncio
    async def test_recurring_events_deduped_during_backfill(self) -> None:
        """Recurring instances should have text stripped after the first."""
        source = GoogleCalendarSource()
        source.configure({})
        queue = _TestQueue()

        # Three instances of the same recurring series
        items = []
        for i in range(3):
            evt = _calendar_event(
                event_id=f"base_20260{i+1}01T090000Z",
                summary="Daily Standup",
            )
            evt["recurringEventId"] = "recurring-base-id"
            items.append(evt)
        # One non-recurring event
        items.append(_calendar_event(event_id="one-off", summary="Lunch"))

        result_page = {"items": items, "nextSyncToken": "sync-x"}
        mock_events = MagicMock()
        mock_list = MagicMock()
        mock_list.execute.return_value = result_page
        mock_events.list.return_value = mock_list
        mock_service = MagicMock()
        mock_service.events.return_value = mock_events

        await source._poll_calendar(mock_service, queue, "primary", None, {})

        assert queue.qsize() == 4
        events = [queue.get_nowait() for _ in range(4)]

        # First recurring instance keeps its text
        assert events[0]["text"] != ""
        assert "Daily Standup" in events[0]["text"]

        # Subsequent recurring instances have text stripped
        assert events[1]["text"] == ""
        assert events[2]["text"] == ""

        # Non-recurring event keeps its text
        assert events[3]["text"] != ""
        assert "Lunch" in events[3]["text"]

    @pytest.mark.asyncio
    async def test_recurring_dedup_applied_during_incremental(self) -> None:
        """Incremental sync should also dedup recurring events."""
        source = GoogleCalendarSource()
        source.configure({})
        queue = _TestQueue()

        items = []
        for i in range(2):
            evt = _calendar_event(
                event_id=f"base_20260{i+1}01T090000Z",
                summary="Daily Standup",
            )
            evt["recurringEventId"] = "recurring-base-id"
            items.append(evt)

        result_page = {"items": items, "nextSyncToken": "sync-inc"}
        mock_events = MagicMock()
        mock_list = MagicMock()
        mock_list.execute.return_value = result_page
        mock_events.list.return_value = mock_list
        mock_service = MagicMock()
        mock_service.events.return_value = mock_events

        # Pass a sync token → incremental mode
        await source._poll_calendar(mock_service, queue, "primary", "old-token", {})

        assert queue.qsize() == 2
        first = queue.get_nowait()
        second = queue.get_nowait()
        # First instance keeps text, second is stripped
        assert first["text"] != ""
        assert second["text"] == ""

    @pytest.mark.asyncio
    async def test_seen_series_persisted_across_polls(self) -> None:
        """Persistent seen_series prevents re-extraction on subsequent polls."""
        source = GoogleCalendarSource()
        source.configure({})
        queue = _TestQueue()

        # Simulate a series already seen in a previous poll
        seen_series = {"recurring-base-id"}

        items = [_calendar_event(event_id="new-instance", summary="Daily Standup")]
        items[0]["recurringEventId"] = "recurring-base-id"

        result_page = {"items": items, "nextSyncToken": "sync-2"}
        mock_events = MagicMock()
        mock_list = MagicMock()
        mock_list.execute.return_value = result_page
        mock_events.list.return_value = mock_list
        mock_service = MagicMock()
        mock_service.events.return_value = mock_events

        await source._poll_calendar(
            mock_service, queue, "primary", "old-token", {}, seen_series,
        )

        assert queue.qsize() == 1
        event = queue.get_nowait()
        # Series was already seen — text stripped even for first instance in this poll
        assert event["text"] == ""
