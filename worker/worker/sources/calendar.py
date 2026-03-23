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
import contextlib
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
    initial_sync_source_done,
)

from .base import PythonSource
from .calendar_auth import get_credentials
from .cursor import save_json_atomic, load_processed_ids, _ProgressTracker

logger = logging.getLogger(__name__)

DEFAULT_CURSOR_PATH = (
    Path.home() / ".fieldnotes" / "data" / "calendar_cursor.json"
)
_PROCESSED_SIDECAR = (
    Path.home() / ".fieldnotes" / "data" / "calendar_processed.json"
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

        first_cycle = True
        try:
            while True:
                for cal_id in self._calendar_ids:
                    try:
                        sync_token = sync_tokens.get(cal_id)
                        new_token = await self._poll_calendar(
                            service, queue, cal_id, sync_token,
                            sync_tokens,
                        )
                        if new_token:
                            sync_tokens[cal_id] = new_token
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

                if first_cycle:
                    initial_sync_source_done()
                    first_cycle = False

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
        sync_tokens: dict[str, str],
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
            # NOTE: do NOT set orderBy here — Google Calendar API omits
            # nextSyncToken from the response when orderBy is specified,
            # which would cause every poll cycle to re-backfill.
            time_min = (
                datetime.now(timezone.utc)
                - timedelta(days=self._max_initial_days)
            ).isoformat()
            kwargs["timeMin"] = time_min
            logger.info(
                "Calendar %s: initial backfill from %s",
                calendar_id,
                time_min[:10],
            )

        new_sync_token: str | None = None
        total_events = 0
        total_recurring_skipped = 0
        total_skipped_processed = 0
        page_token: str | None = None
        # During backfill, track recurring series we've already emitted a
        # full (text-bearing) event for.  Subsequent instances of the same
        # series get their CalendarEvent node + GraphHints written but
        # skip the expensive chunk → embed → extract pipeline.
        seen_recurring: set[str] = set()
        # Load any previously-processed IDs from an interrupted backfill
        # so we can skip them.
        already_processed = load_processed_ids(_PROCESSED_SIDECAR)
        # Collect events in memory first so we know the total count
        # before building the _ProgressTracker.
        pending_events: list[dict[str, Any]] = []

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
                        # Skip events already processed in a previous
                        # interrupted backfill.
                        if is_backfill and ingest["source_id"] in already_processed:
                            total_skipped_processed += 1
                            continue
                        if is_backfill:
                            ingest["initial_scan"] = True
                            # Dedup recurring instances: keep full text
                            # only for the first occurrence of each series.
                            rid = event.get("recurringEventId", "")
                            if rid:
                                if rid in seen_recurring:
                                    # Strip text so pipeline writes the
                                    # CalendarEvent node (with its own
                                    # start/end times) and GraphHints
                                    # but skips embed + extract.
                                    ingest["text"] = ""
                                    total_recurring_skipped += 1
                                else:
                                    seen_recurring.add(rid)
                        pending_events.append(ingest)
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

        # Enqueue the collected events with _on_indexed callbacks.
        if is_backfill and pending_events:
            # Capture the sync token to save once ALL events are indexed.
            def _save_sync_token() -> None:
                if new_sync_token:
                    sync_tokens[calendar_id] = new_sync_token
                    save_json_atomic(self._cursor_path, sync_tokens)
                    logger.info(
                        "Calendar %s: sync token saved after backfill",
                        calendar_id,
                    )

            tracker = _ProgressTracker(
                total=len(pending_events),
                sidecar_path=_PROCESSED_SIDECAR,
                on_all_done=_save_sync_token,
            )
            for ingest in pending_events:
                sid = ingest["source_id"]
                ingest["_on_indexed"] = lambda s=sid: tracker.ack(s)
                await queue.put(ingest)
        elif is_backfill and not pending_events and already_processed:
            # Crash recovery: all events were already processed (sidecar
            # survived) but the cursor was never saved.  Save it now and
            # clean up the sidecar to break out of the re-backfill loop.
            if new_sync_token:
                sync_tokens[calendar_id] = new_sync_token
                save_json_atomic(self._cursor_path, sync_tokens)
                logger.info(
                    "Calendar %s: sync token saved (crash-recovery, "
                    "all %d events already processed)",
                    calendar_id,
                    len(already_processed),
                )
            with contextlib.suppress(OSError):
                _PROCESSED_SIDECAR.unlink()
        else:
            # Incremental poll — attach _on_indexed to defer cursor save.
            if pending_events:
                def _save_inc_token() -> None:
                    if new_sync_token:
                        sync_tokens[calendar_id] = new_sync_token
                        save_json_atomic(self._cursor_path, sync_tokens)

                tracker = _ProgressTracker(
                    total=len(pending_events),
                    sidecar_path=_PROCESSED_SIDECAR,
                    on_all_done=_save_inc_token,
                )
                for ingest in pending_events:
                    sid = ingest["source_id"]
                    ingest["_on_indexed"] = lambda s=sid: tracker.ack(s)
                    await queue.put(ingest)
            elif new_sync_token:
                # No events but token refreshed — save immediately.
                sync_tokens[calendar_id] = new_sync_token
                save_json_atomic(self._cursor_path, sync_tokens)

        if total_events > 0:
            if is_backfill:
                initial_sync_add_items(total_events)
            skipped_msg = ""
            if total_skipped_processed:
                skipped_msg = (
                    f", {total_skipped_processed} already processed"
                )
            if total_recurring_skipped:
                logger.info(
                    "Calendar %s: queued %d event(s) "
                    "(%d recurring instances will skip embed/extract%s)",
                    calendar_id,
                    total_events,
                    total_recurring_skipped,
                    skipped_msg,
                )
            else:
                logger.info(
                    "Calendar %s: queued %d event(s)%s",
                    calendar_id,
                    total_events,
                    skipped_msg,
                )

        return new_sync_token
