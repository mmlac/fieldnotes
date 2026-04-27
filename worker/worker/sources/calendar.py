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

from worker.parsers.attachments import DEFAULT_INDEXABLE_MIMETYPES

from .base import IndexedCheck, PythonSource
from .calendar_auth import check_calendar_auth, get_credentials

logger = logging.getLogger(__name__)


DEFAULT_POLL_INTERVAL = 300  # 5 minutes
DEFAULT_MAX_INITIAL_DAYS = 90  # backfill ~3 months
DEFAULT_CALENDAR_IDS = ["primary"]
DEFAULT_ATTACHMENT_MAX_SIZE_MB = 25
API_CALL_TIMEOUT = 60


def _extract_attachment_metadata(
    event: dict[str, Any],
) -> list[dict[str, Any]]:
    """Pull the ``attachments`` array off a Calendar API event resource.

    Surfaces ``(title, mime_type, file_id, file_url, icon_link)`` per
    attachment.  The ``size_bytes`` field is left at ``0`` here — the
    Drive ``files.get`` size lookup happens in the source layer when the
    account opts in to attachment indexing.
    """
    out: list[dict[str, Any]] = []
    for att in event.get("attachments", []) or []:
        if not isinstance(att, dict):
            continue
        out.append(
            {
                "title": att.get("title", ""),
                "mime_type": att.get("mimeType", ""),
                "file_id": att.get("fileId", ""),
                "file_url": att.get("fileUrl", ""),
                "icon_link": att.get("iconLink", ""),
                "size_bytes": 0,
            }
        )
    return out


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

    attachments = _extract_attachment_metadata(event)

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
        "attachments": attachments,
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
        self._download_attachments = False
        self._attachment_indexable_mimetypes: list[str] = list(
            DEFAULT_INDEXABLE_MIMETYPES
        )
        self._attachment_max_size_mb = DEFAULT_ATTACHMENT_MAX_SIZE_MB

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
        if "download_attachments" in cfg:
            self._download_attachments = bool(cfg["download_attachments"])
        if "attachment_indexable_mimetypes" in cfg:
            self._attachment_indexable_mimetypes = list(
                cfg["attachment_indexable_mimetypes"]
            )
        if "attachment_max_size_mb" in cfg:
            self._attachment_max_size_mb = int(cfg["attachment_max_size_mb"])
        if "cursor_path" in cfg:
            raise ValueError(
                "GoogleCalendarSource: 'cursor_path' is no longer supported. "
                "Cursors live in queue.db (cursors table, key "
                "'calendar:<account>'); remove this key from your config."
            )

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

        # Fail fast if attachment indexing was just turned on but the
        # persisted token still has the narrower calendar-only scope.
        try:
            check_calendar_auth(
                self._account,
                download_attachments=self._download_attachments,
            )
        except Exception as exc:
            logger.error(
                "Calendar account=%s scope check failed: %s — staying idle",
                self._account,
                exc,
            )
            while True:
                await asyncio.sleep(3600)

        creds = await asyncio.to_thread(
            get_credentials,
            secrets_path,
            self._account,
            download_attachments=self._download_attachments,
        )
        service = build("calendar", "v3", credentials=creds)
        drive_service = (
            build("drive", "v3", credentials=creds)
            if self._download_attachments
            else None
        )

        # Load persisted cursor state from queue DB (single source of truth).
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
                            drive_service=drive_service,
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
            raise

    async def _enrich_attachment_sizes(
        self,
        drive_service: Any,
        attachments: list[dict[str, Any]],
    ) -> None:
        """Populate ``size_bytes`` on attachments whose MIME may be indexed.

        Drive ``files.get(fields='size')`` is one HTTP round-trip per
        file.  We skip attachments whose MIME is already disqualified by
        the indexable allowlist — those are guaranteed to land in
        ``metadata_only`` regardless of size, so the size call is wasted
        quota.  On 404 / 403 / other errors the size is left at 0; the
        parser treats unknown size as 'too large' under a conservative
        ``max_size_mb`` and falls back to metadata-only.
        """
        for att in attachments:
            mime = att.get("mime_type", "")
            file_id = att.get("file_id", "")
            if not file_id:
                continue
            if mime not in self._attachment_indexable_mimetypes:
                continue
            try:
                meta = await asyncio.to_thread(
                    lambda: drive_service.files()
                    .get(fileId=file_id, fields="size")
                    .execute()
                )
            except HttpError as exc:
                logger.info(
                    "Drive files.get(size) failed for %s: %s — leaving size=0",
                    file_id,
                    exc,
                )
                continue
            except Exception:
                logger.warning(
                    "Unexpected Drive size lookup error for %s",
                    file_id,
                    exc_info=True,
                )
                continue
            try:
                att["size_bytes"] = int(meta.get("size", 0) or 0)
            except (TypeError, ValueError):
                att["size_bytes"] = 0

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
        drive_service: Any | None = None,
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
                        # Stamp attachment policy onto every event so the
                        # parser can apply classify_attachment uniformly,
                        # and (when enabled) batch a Drive size lookup for
                        # any attachment whose MIME could be indexed.
                        meta = ingest["meta"]
                        meta["download_attachments"] = self._download_attachments
                        meta["attachment_indexable_mimetypes"] = list(
                            self._attachment_indexable_mimetypes
                        )
                        meta["attachment_max_size_mb"] = (
                            self._attachment_max_size_mb
                        )
                        if (
                            self._download_attachments
                            and drive_service is not None
                            and meta.get("attachments")
                        ):
                            await self._enrich_attachment_sizes(
                                drive_service, meta["attachments"]
                            )
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
            if new_sync_token:
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
                queue.save_cursor(self._cursor_key, _cursor_json())

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
