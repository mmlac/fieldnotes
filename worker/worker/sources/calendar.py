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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from worker.queue import PersistentQueue

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from worker.log_sanitizer import redact_home_path
from worker.metrics import (
    INDEXED_PREFILTER_SKIPPED,
    SOURCE_WATCHER_EVENTS,
    WATCHER_ACTIVE,
    WATCHER_LAST_EVENT_TIMESTAMP,
    initial_sync_add_items,
    initial_sync_source_done,
)

from .base import IndexedCheck, PythonSource
from .calendar_auth import get_credentials
from .cursor import save_json_atomic

logger = logging.getLogger(__name__)

def _default_cursor_path(account: str) -> Path:
    """Per-account cursor file: ~/.fieldnotes/data/calendar_cursor-{account}.json."""
    return Path.home() / ".fieldnotes" / "data" / f"calendar_cursor-{account}.json"


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
    event: dict[str, Any], calendar_id: str, account: str = ""
) -> dict[str, Any] | None:
    """Build an IngestEvent dict from a Calendar API event resource.

    *calendar_id* is the synthetic ``{account}/{cal_id}`` identifier so the
    raw cal-id is preserved in event metadata for downstream queries.  The
    document URI is account-scoped (``google-calendar://{account}/event/{id}``)
    so two accounts each polling ``primary`` produce distinct
    CalendarEvent nodes even when Google reuses event IDs across calendars.

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
        "account": account,
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
        "source_id": f"google-calendar://{account}/event/{event_id}",
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
        self._account: str = ""
        self._poll_interval = DEFAULT_POLL_INTERVAL
        self._max_initial_days = DEFAULT_MAX_INITIAL_DAYS
        self._calendar_ids: list[str] = list(DEFAULT_CALENDAR_IDS)
        self._client_secrets_path = "~/.fieldnotes/credentials.json"
        self._cursor_path: Path | None = None

    def name(self) -> str:
        return "google_calendar"

    @property
    def source_id(self) -> str:
        """Stable identifier surfacing the account: ``google_calendar:{account}``."""
        return f"google_calendar:{self._account}"

    @property
    def _cursor_key(self) -> str:
        """Per-account PersistentQueue cursor key (no cross-account stomping)."""
        return f"calendar:{self._account}"

    def _synthetic_calendar_id(self, calendar_id: str) -> str:
        """Account-namespaced calendar identifier so two ``primary`` calendars
        from different accounts do not collide in the graph."""
        return f"{self._account}/{calendar_id}"

    def configure(self, cfg: dict[str, Any]) -> None:
        account = cfg.get("account")
        if not account or not isinstance(account, str):
            raise ValueError(
                "GoogleCalendarSource requires 'account' in config (the "
                "account label from [sources.google_calendar.<account>])"
            )
        self._account = account

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
        else:
            self._cursor_path = _default_cursor_path(self._account)

    async def start(
        self,
        queue: PersistentQueue,
        *,
        indexed_check: IndexedCheck | None = None,
    ) -> None:
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
            get_credentials, secrets_path, self._account
        )
        service = build("calendar", "v3", credentials=creds)

        # Load persisted cursor state from queue DB (migrated from JSON file).
        raw_cursor = queue.load_cursor(self._cursor_key)
        sync_tokens: dict[str, str] = {}
        seen_series: set[str] = set()
        if raw_cursor:
            try:
                data = json.loads(raw_cursor)
                if isinstance(data, dict):
                    # New format: {"sync_tokens": {...}, "seen_series": [...]}
                    if "sync_tokens" in data:
                        sync_tokens = data["sync_tokens"]
                        seen_series = set(data.get("seen_series", []))
                    else:
                        # Legacy format: {calendar_id: sync_token}
                        sync_tokens = data
            except (json.JSONDecodeError, AttributeError):
                pass
        # Fall back to legacy JSON file if not yet migrated.
        if not sync_tokens:
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
                            sync_tokens, seen_series,
                            indexed_check=indexed_check,
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
        seen_series: set[str] | None = None,
        *,
        indexed_check: IndexedCheck | None = None,
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
        # Track recurring series we've already emitted a full (text-bearing)
        # event for.  Subsequent instances of the same series get their
        # CalendarEvent node + GraphHints written but skip the expensive
        # chunk → embed → extract pipeline.  The persistent seen_series set
        # is shared across poll cycles to prevent re-extraction on re-index.
        if seen_series is None:
            seen_series = set()
        pending_events: list[dict[str, Any]] = []

        try:
            while True:
                if page_token:
                    kwargs["pageToken"] = page_token

                result = await asyncio.to_thread(
                    lambda: service.events().list(**kwargs).execute()
                )

                events = result.get("items", [])

                # Phase 1 backstop: on backfill, batch-check Neo4j for the
                # entire page so we can skip events whose source_id is
                # already chunked (e.g. on a cold start with no cursor but
                # an existing graph).
                synthetic_id = self._synthetic_calendar_id(calendar_id)
                already_indexed_sids: set[str] = set()
                if is_backfill and indexed_check is not None and events:
                    candidate_sids: list[str] = []
                    for ev in events:
                        ev_id = ev.get("id")
                        if ev_id:
                            candidate_sids.append(
                                f"google-calendar://{self._account}/event/{ev_id}"
                            )
                    if candidate_sids:
                        try:
                            already_indexed_sids = await asyncio.to_thread(
                                indexed_check, candidate_sids
                            )
                        except Exception:
                            logger.warning(
                                "Calendar indexed_check failed for %s; "
                                "processing all events in this page",
                                calendar_id,
                                exc_info=True,
                            )
                            already_indexed_sids = set()

                for event in events:
                    ingest = _build_ingest_event(
                        event, synthetic_id, self._account
                    )
                    if ingest is not None:
                        # Phase 1: skip events already chunked in Neo4j.
                        if (
                            is_backfill
                            and ingest["source_id"] in already_indexed_sids
                        ):
                            INDEXED_PREFILTER_SKIPPED.labels(
                                source_type="google_calendar"
                            ).inc()
                            total_skipped_processed += 1
                            continue
                        # Skip events already in the queue.
                        if is_backfill and queue.is_enqueued(ingest["source_id"]):
                            total_skipped_processed += 1
                            continue
                        if is_backfill:
                            ingest["initial_scan"] = True
                        # Dedup recurring instances: keep full text only
                        # for the first occurrence of each series.  Applied
                        # in both backfill and incremental modes — the
                        # persistent seen_series set prevents re-extraction
                        # across poll cycles.
                        rid = event.get("recurringEventId", "")
                        if rid:
                            if rid in seen_series:
                                # Strip text so pipeline writes the
                                # CalendarEvent node (with its own
                                # start/end times) and GraphHints
                                # but skips embed + extract.
                                ingest["text"] = ""
                                total_recurring_skipped += 1
                            else:
                                seen_series.add(rid)
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

        def _cursor_json() -> str:
            """Build cursor JSON with sync tokens and seen series."""
            return json.dumps({
                "sync_tokens": sync_tokens,
                "seen_series": sorted(seen_series),
            })

        # Enqueue the collected events with cursor persistence.
        if is_backfill and pending_events:
            if new_sync_token:
                sync_tokens[calendar_id] = new_sync_token

            cursor_val = _cursor_json() if new_sync_token else None
            for i, ingest in enumerate(pending_events):
                is_last = i == len(pending_events) - 1
                queue.enqueue(
                    ingest,
                    cursor_key=self._cursor_key if is_last and cursor_val else None,
                    cursor_value=cursor_val if is_last else None,
                )
            # Also persist to legacy file.
            if new_sync_token:
                save_json_atomic(self._cursor_path, sync_tokens)
                logger.info(
                    "Calendar %s: sync token saved after backfill",
                    calendar_id,
                )
        else:
            # Incremental poll.
            if pending_events:
                if new_sync_token:
                    sync_tokens[calendar_id] = new_sync_token
                cursor_val = _cursor_json() if new_sync_token else None
                for i, ingest in enumerate(pending_events):
                    is_last = i == len(pending_events) - 1
                    queue.enqueue(
                        ingest,
                        cursor_key=self._cursor_key if is_last and cursor_val else None,
                        cursor_value=cursor_val if is_last else None,
                    )
            # Save sync token (whether or not there were events).
            if new_sync_token:
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
