"""Google Calendar polling source with syncToken-based incremental sync.

Uses the Google Calendar API ``events.list`` endpoint with ``syncToken``
for efficient incremental polling.  On first run, performs a backfill of
events from the configured lookback period.  Emits one IngestEvent per
calendar event with attendees, times, and metadata.

Config section ``[sources.google_calendar]``::

    poll_interval_seconds = 300
    max_initial_days = 90
    calendar_ids = ["primary"]
    client_secrets_path = "~/.fieldnotes/credentials.json"
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from worker.log_sanitizer import redact_home_path
from worker.metrics import (
    SOURCE_WATCHER_EVENTS,
    WATCHER_ACTIVE,
    WATCHER_LAST_EVENT_TIMESTAMP,
    initial_sync_add_items,
)

from .base import PythonSource
from .calendar_auth import get_credentials
from .cursor import save_json_atomic

logger = logging.getLogger(__name__)

DEFAULT_CURSOR_PATH = (
    Path.home() / ".fieldnotes" / "data" / "calendar_cursor.json"
)
DEFAULT_POLL_INTERVAL = 300  # 5 minutes
DEFAULT_MAX_INITIAL_DAYS = 90  # backfill ~3 months
DEFAULT_CALENDAR_IDS = ["primary"]
API_CALL_TIMEOUT = 60


def _load_cursor(path: Path) -> dict[str, str]:
    """Load per-calendar sync tokens from disk.

    Returns a dict mapping calendar_id → syncToken.
    """
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load calendar cursor: %s", exc)
    return {}


def _event_start_iso(event: dict[str, Any]) -> str:
    """Return the RFC3339 start time of a calendar event."""
    start = event.get("start", {})
    return start.get("dateTime") or start.get("date", "")


def _event_end_iso(event: dict[str, Any]) -> str:
    """Return the RFC3339 end time of a calendar event."""
    end = event.get("end", {})
    return end.get("dateTime") or end.get("date", "")


def _build_ingest_event(
    event: dict[str, Any], calendar_id: str
) -> dict[str, Any] | None:
    """Build an IngestEvent dict from a Calendar API event resource.

    Returns None for cancelled events (they are handled as deletes).
    """
    event_id = event.get("id", "")
    status = event.get("status", "confirmed")
    summary = event.get("summary", "(No title)")
    description = event.get("description", "")
    location = event.get("location", "")
    start_time = _event_start_iso(event)
    end_time = _event_end_iso(event)
    html_link = event.get("htmlLink", "")

    # Organiser
    organizer = event.get("organizer", {})
    organizer_email = organizer.get("email", "")
    organizer_name = organizer.get("displayName", "")

    # Creator (may differ from organizer for forwarded events)
    creator = event.get("creator", {})
    creator_email = creator.get("email", "")

    # Attendees
    attendees_raw = event.get("attendees", [])
    attendees = []
    for att in attendees_raw:
        email = att.get("email", "")
        if email:
            attendees.append(
                {
                    "email": email.strip().lower(),
                    "name": att.get("displayName", ""),
                    "response": att.get("responseStatus", "needsAction"),
                    "self": att.get("self", False),
                }
            )

    # Recurrence info
    recurring_event_id = event.get("recurringEventId", "")

    # Build body text from summary + description + location
    text_parts = [summary]
    if location:
        text_parts.append(f"Location: {location}")
    if description:
        text_parts.append(description)
    if attendees:
        names = [
            a["name"] or a["email"] for a in attendees if not a.get("self")
        ]
        if names:
            text_parts.append(f"Attendees: {', '.join(names)}")
    body_text = "\n\n".join(text_parts)

    operation = "deleted" if status == "cancelled" else "created"

    # Determine source_modified_at from the event's updated timestamp
    updated = event.get("updated", "")
    source_modified = updated or datetime.now(timezone.utc).isoformat()

    meta: dict[str, Any] = {
        "event_id": event_id,
        "calendar_id": calendar_id,
        "summary": summary,
        "description": description,
        "location": location,
        "start_time": start_time,
        "end_time": end_time,
        "organizer_email": organizer_email,
        "organizer_name": organizer_name,
        "creator_email": creator_email,
        "attendees": attendees,
        "html_link": html_link,
        "recurring_event_id": recurring_event_id,
        "status": status,
    }

    return {
        "id": str(uuid.uuid4()),
        "source_type": "google_calendar",
        "source_id": f"gcal:{calendar_id}:{event_id}",
        "operation": operation,
        "text": body_text,
        "mime_type": "text/plain",
        "meta": meta,
        "enqueued_at": datetime.now(timezone.utc).isoformat(),
        "source_modified_at": source_modified,
    }


class GoogleCalendarSource(PythonSource):
    """Polls Google Calendar via the API and emits IngestEvent dicts.

    Uses syncToken-based incremental sync — the first poll fetches events
    from the last ``max_initial_days``, subsequent polls use the sync
    token returned by the API to fetch only changed events.
    """

    def __init__(self) -> None:
        self._poll_interval = DEFAULT_POLL_INTERVAL
        self._max_initial_days = DEFAULT_MAX_INITIAL_DAYS
        self._calendar_ids: list[str] = list(DEFAULT_CALENDAR_IDS)
        self._client_secrets_path = "~/.fieldnotes/credentials.json"
        self._cursor_path = DEFAULT_CURSOR_PATH
        self._token_path = (
            Path.home() / ".fieldnotes" / "data" / "calendar_token.json"
        )

    def name(self) -> str:
        return "google_calendar"

    def configure(self, cfg: dict[str, Any]) -> None:
        self._poll_interval = int(
            cfg.get("poll_interval_seconds", DEFAULT_POLL_INTERVAL)
        )
        self._max_initial_days = int(
            cfg.get("max_initial_days", DEFAULT_MAX_INITIAL_DAYS)
        )
        if "calendar_ids" in cfg:
            self._calendar_ids = list(cfg["calendar_ids"])
        if "client_secrets_path" in cfg:
            self._client_secrets_path = cfg["client_secrets_path"]
        if "cursor_path" in cfg:
            self._cursor_path = Path(cfg["cursor_path"]).expanduser().resolve()
        if "token_path" in cfg:
            self._token_path = Path(cfg["token_path"]).expanduser().resolve()

    async def start(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        secrets_path = Path(self._client_secrets_path).expanduser().resolve()
        if not secrets_path.exists():
            logger.error(
                "Calendar client secrets not found: %s",
                redact_home_path(str(secrets_path)),
            )
            # Stay alive but idle so the daemon doesn't restart us
            while True:
                await asyncio.sleep(3600)

        creds = await asyncio.to_thread(
            get_credentials, secrets_path, self._token_path
        )
        service = build("calendar", "v3", credentials=creds)

        # Load persisted sync tokens
        sync_tokens = _load_cursor(self._cursor_path)

        WATCHER_ACTIVE.labels(source_type="google_calendar").set(1)
        logger.info(
            "Calendar source started — polling %d calendar(s) every %ds",
            len(self._calendar_ids),
            self._poll_interval,
        )

        try:
            while True:
                for cal_id in self._calendar_ids:
                    try:
                        sync_token = sync_tokens.get(cal_id)
                        new_token = await self._poll_calendar(
                            service, queue, cal_id, sync_token
                        )
                        if new_token:
                            sync_tokens[cal_id] = new_token
                            # Persist immediately so a restart doesn't
                            # redo the full backfill for this calendar.
                            save_json_atomic(self._cursor_path, sync_tokens)
                    except HttpError as exc:
                        if exc.resp.status == 410:
                            # Sync token expired — clear and do full resync
                            logger.warning(
                                "Calendar sync token expired for %s, "
                                "performing full resync",
                                cal_id,
                            )
                            sync_tokens.pop(cal_id, None)
                        else:
                            logger.error(
                                "Calendar API error for %s: %s", cal_id, exc
                            )
                    except Exception:
                        logger.exception(
                            "Unexpected error polling calendar %s", cal_id
                        )

                # Persist sync tokens after each poll cycle
                save_json_atomic(self._cursor_path, sync_tokens)

                await asyncio.sleep(self._poll_interval)
        except asyncio.CancelledError:
            WATCHER_ACTIVE.labels(source_type="google_calendar").set(0)
            save_json_atomic(self._cursor_path, sync_tokens)
            raise

    async def _poll_calendar(
        self,
        service: Any,
        queue: asyncio.Queue[dict[str, Any]],
        calendar_id: str,
        sync_token: str | None,
    ) -> str | None:
        """Fetch events for a single calendar.  Returns new sync token."""
        is_backfill = sync_token is None
        kwargs: dict[str, Any] = {
            "calendarId": calendar_id,
            "singleEvents": True,
            "maxResults": 250,
        }

        if sync_token:
            # Incremental sync — only changes since last sync
            kwargs["syncToken"] = sync_token
            kwargs["showDeleted"] = True
        else:
            # Initial backfill — last N days
            time_min = (
                datetime.now(timezone.utc)
                - timedelta(days=self._max_initial_days)
            ).isoformat()
            kwargs["timeMin"] = time_min
            kwargs["orderBy"] = "startTime"
            logger.info(
                "Calendar %s: initial backfill from %s",
                calendar_id,
                time_min[:10],
            )

        new_sync_token: str | None = None
        total_events = 0
        page_token: str | None = None

        try:
            while True:
                if page_token:
                    kwargs["pageToken"] = page_token

                result = await asyncio.to_thread(
                    lambda: service.events().list(**kwargs).execute()
                )

                events = result.get("items", [])
                for event in events:
                    ingest = _build_ingest_event(event, calendar_id)
                    if ingest is not None:
                        if is_backfill:
                            ingest["initial_scan"] = True
                        await queue.put(ingest)
                        total_events += 1
                        SOURCE_WATCHER_EVENTS.labels(
                            source_type="google_calendar",
                            event_type=ingest["operation"],
                        ).inc()
                        WATCHER_LAST_EVENT_TIMESTAMP.labels(
                            source_type="google_calendar"
                        ).set(datetime.now(timezone.utc).timestamp())

                page_token = result.get("nextPageToken")
                new_sync_token = result.get("nextSyncToken")

                if not page_token:
                    break

                # Brief pause between pages
                await asyncio.sleep(0.2)
        except asyncio.CancelledError:
            # Pagination was interrupted — propagate but let the caller
            # capture whatever sync token we obtained so far.
            raise

        if total_events > 0:
            if is_backfill:
                initial_sync_add_items(total_events)
            logger.info(
                "Calendar %s: processed %d event(s)", calendar_id, total_events
            )

        return new_sync_token
