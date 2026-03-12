"""Gmail polling source shim with cursor-based incremental sync.

Mirrors the Go GmailSource design: ticker-based polling at a configurable
interval, cursor persistence for incremental polling, initial backfill of
recent threads, and label filtering.  Emits one IngestEvent per email
message with structured metadata.

Config section ``[sources.gmail]``::

    poll_interval_seconds = 300
    max_initial_threads = 500
    label_filter = "INBOX"
    client_secrets_path = "~/.fieldnotes/credentials.json"
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from worker.metrics import (
    GMAIL_POLL_DURATION,
    SOURCE_WATCHER_EVENTS,
    WATCHER_ACTIVE,
    WATCHER_LAST_EVENT_TIMESTAMP,
    observe_duration,
)

from .base import PythonSource
from .gmail_auth import get_credentials

logger = logging.getLogger(__name__)

DEFAULT_CURSOR_PATH = Path.home() / ".fieldnotes" / "data" / "gmail_cursor.json"
DEFAULT_POLL_INTERVAL = 300  # seconds
DEFAULT_MAX_INITIAL_THREADS = 500
BACKFILL_PAGE_DELAY = 0.5  # seconds between backfill page fetches
BACKFILL_MAX_RETRIES = 3
BACKFILL_INITIAL_BACKOFF = 1.0  # seconds
API_CALL_TIMEOUT = 60  # seconds — max wait for a single Gmail API call


def _load_cursor(path: Path) -> str | None:
    """Load the persisted history ID from disk."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return data.get("history_id")
    except (json.JSONDecodeError, OSError):
        logger.warning("Failed to read cursor file %s, starting fresh", path)
        return None


def _save_cursor(path: Path, history_id: str) -> None:
    """Persist the latest history ID to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"history_id": history_id}))
    path.chmod(0o600)


def _extract_recipients(headers: list[dict[str, str]]) -> list[str]:
    """Pull To/Cc addresses from message headers."""
    recipients: list[str] = []
    for hdr in headers:
        name = hdr.get("name", "").lower()
        if name in ("to", "cc"):
            value = hdr.get("value", "")
            if value:
                recipients.extend(
                    addr.strip() for addr in value.split(",") if addr.strip()
                )
    return recipients


def _header_value(headers: list[dict[str, str]], name: str) -> str:
    """Get the value of a specific header (case-insensitive)."""
    target = name.lower()
    for hdr in headers:
        if hdr.get("name", "").lower() == target:
            return hdr.get("value", "")
    return ""


def _build_ingest_event(msg: dict[str, Any]) -> dict[str, Any]:
    """Build an IngestEvent dict from a Gmail API message resource."""
    headers = msg.get("payload", {}).get("headers", [])
    subject = _header_value(headers, "Subject")
    sender = _header_value(headers, "From")
    date_str = _header_value(headers, "Date")
    recipients = _extract_recipients(headers)

    internal_ts = int(msg.get("internalDate", "0"))
    source_modified = datetime.fromtimestamp(
        internal_ts / 1000, tz=timezone.utc
    ).isoformat()

    meta: dict[str, Any] = {
        "message_id": msg["id"],
        "thread_id": msg.get("threadId", ""),
        "subject": subject,
        "date": date_str,
        "sender_email": sender,
        "recipients": recipients,
    }

    return {
        "id": str(uuid.uuid4()),
        "source_type": "gmail",
        "source_id": f"gmail:{msg['id']}",
        "operation": "created",
        "mime_type": "message/rfc822",
        "meta": meta,
        "enqueued_at": datetime.now(timezone.utc).isoformat(),
        "source_modified_at": source_modified,
    }


class GmailSource(PythonSource):
    """Polls Gmail via the API and emits IngestEvent dicts.

    Config keys (from ``[sources.gmail]``):
        poll_interval_seconds: int  — polling interval (default: 300)
        max_initial_threads: int    — backfill limit (default: 500)
        label_filter: str           — Gmail label to poll (default: "INBOX")
        client_secrets_path: str    — OAuth2 client secrets file (required)
        cursor_path: str            — cursor persistence file (optional)
    """

    def __init__(self) -> None:
        self._poll_interval: int = DEFAULT_POLL_INTERVAL
        self._max_initial_threads: int = DEFAULT_MAX_INITIAL_THREADS
        self._label_filter: str = "INBOX"
        self._client_secrets_path: Path | None = None
        self._cursor_path: Path = DEFAULT_CURSOR_PATH

    def name(self) -> str:
        return "gmail"

    def configure(self, cfg: dict[str, Any]) -> None:
        secrets = cfg.get("client_secrets_path")
        if not secrets:
            raise ValueError(
                "GmailSource requires 'client_secrets_path' in config"
            )
        self._client_secrets_path = Path(secrets).expanduser().resolve()

        self._poll_interval = int(
            cfg.get("poll_interval_seconds", DEFAULT_POLL_INTERVAL)
        )
        self._max_initial_threads = int(
            cfg.get("max_initial_threads", DEFAULT_MAX_INITIAL_THREADS)
        )
        self._label_filter = cfg.get("label_filter", "INBOX")

        cursor = cfg.get("cursor_path")
        if cursor:
            self._cursor_path = Path(cursor).expanduser().resolve()

    async def start(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        assert self._client_secrets_path is not None

        creds = get_credentials(self._client_secrets_path)
        service = build("gmail", "v1", credentials=creds)
        messages_api = service.users().messages()

        cursor = _load_cursor(self._cursor_path)

        WATCHER_ACTIVE.labels(source_type="gmail").set(1)

        if cursor is None:
            # Initial backfill: fetch recent messages up to max_initial_threads
            logger.info(
                "No cursor found — backfilling up to %d messages (label=%s)",
                self._max_initial_threads,
                self._label_filter,
            )
            cursor = await self._backfill(messages_api, queue)
            if cursor:
                _save_cursor(self._cursor_path, cursor)

        # Polling loop
        try:
            while True:
                await asyncio.sleep(self._poll_interval)
                with observe_duration(GMAIL_POLL_DURATION):
                    cursor = await self._poll_incremental(
                        service, messages_api, queue, cursor
                    )
        except asyncio.CancelledError:
            WATCHER_ACTIVE.labels(source_type="gmail").set(0)
            raise

    async def _backfill(
        self,
        messages_api: Any,
        queue: asyncio.Queue[dict[str, Any]],
    ) -> str | None:
        """Fetch up to max_initial_threads recent messages and return the
        latest historyId for subsequent incremental polling.

        Includes rate limiting (delay between pages) and retry with
        exponential backoff to avoid hitting Gmail API rate limits.
        """
        loop = asyncio.get_running_loop()
        fetched = 0
        page_token: str | None = None
        latest_history_id: str | None = None

        while fetched < self._max_initial_threads:
            batch_size = min(100, self._max_initial_threads - fetched)
            kwargs: dict[str, Any] = {
                "userId": "me",
                "labelIds": [self._label_filter],
                "maxResults": batch_size,
            }
            if page_token:
                kwargs["pageToken"] = page_token

            result = await self._api_call_with_retry(
                loop, lambda kw=kwargs: messages_api.list(**kw).execute()
            )
            msg_stubs = result.get("messages", [])
            if not msg_stubs:
                break

            for stub in msg_stubs:
                if fetched >= self._max_initial_threads:
                    break
                msg = await self._api_call_with_retry(
                    loop,
                    lambda mid=stub["id"]: messages_api.get(
                        userId="me", id=mid, format="metadata",
                        metadataHeaders=["From", "To", "Cc", "Subject", "Date"],
                    ).execute(),
                )
                event = _build_ingest_event(msg)
                await queue.put(event)
                SOURCE_WATCHER_EVENTS.labels(
                    source_type="gmail", event_type="created",
                ).inc()
                WATCHER_LAST_EVENT_TIMESTAMP.labels(
                    source_type="gmail",
                ).set_to_current_time()
                fetched += 1

                # Track the highest historyId seen
                hid = msg.get("historyId")
                if hid and (
                    latest_history_id is None
                    or int(hid) > int(latest_history_id)
                ):
                    latest_history_id = hid

            page_token = result.get("nextPageToken")
            if not page_token:
                break

            # Rate-limit: pause between page fetches
            await asyncio.sleep(BACKFILL_PAGE_DELAY)

        logger.info("Backfill complete: %d messages fetched", fetched)
        return latest_history_id

    # HTTP status codes that are safe to retry (transient errors).
    _RETRYABLE_STATUS_CODES = frozenset({429, 500, 503})

    @staticmethod
    async def _api_call_with_retry(
        loop: asyncio.AbstractEventLoop,
        call: Any,
        max_retries: int = BACKFILL_MAX_RETRIES,
        initial_backoff: float = BACKFILL_INITIAL_BACKOFF,
        timeout: float = API_CALL_TIMEOUT,
    ) -> Any:
        """Execute a Gmail API call in an executor with retry, backoff, and timeout."""
        backoff = initial_backoff
        for attempt in range(max_retries + 1):
            try:
                return await asyncio.wait_for(
                    loop.run_in_executor(None, call),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                if attempt == max_retries:
                    raise TimeoutError(
                        f"Gmail API call timed out after {timeout}s "
                        f"(attempt {attempt + 1}/{max_retries + 1})"
                    )
                logger.warning(
                    "Gmail API call timed out (attempt %d/%d), retrying in %.1fs",
                    attempt + 1,
                    max_retries + 1,
                    backoff,
                )
                await asyncio.sleep(backoff)
                backoff *= 2
                continue
            except HttpError as exc:
                if exc.resp.status not in GmailSource._RETRYABLE_STATUS_CODES:
                    raise
                if attempt == max_retries:
                    raise
                logger.warning(
                    "Gmail API call failed (attempt %d/%d, status %d), retrying in %.1fs",
                    attempt + 1,
                    max_retries + 1,
                    exc.resp.status,
                    backoff,
                )
                await asyncio.sleep(backoff)
                backoff *= 2

    async def _poll_incremental(
        self,
        service: Any,
        messages_api: Any,
        queue: asyncio.Queue[dict[str, Any]],
        cursor: str | None,
    ) -> str | None:
        """Use Gmail history API to fetch messages added since *cursor*."""
        if not cursor:
            logger.warning("No cursor available for incremental poll, skipping")
            return cursor

        loop = asyncio.get_running_loop()
        history_api = service.users().history()

        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: history_api.list(
                        userId="me",
                        startHistoryId=cursor,
                        labelId=self._label_filter,
                        historyTypes=["messageAdded"],
                    ).execute(),
                ),
                timeout=API_CALL_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.error(
                "Gmail history API call timed out after %ds (cursor=%s)",
                API_CALL_TIMEOUT,
                cursor,
            )
            return cursor
        except Exception:
            logger.exception("Failed to fetch Gmail history (cursor=%s)", cursor)
            return cursor

        new_history_id = result.get("historyId", cursor)
        records = result.get("history", [])

        seen: set[str] = set()
        count = 0
        for record in records:
            for added in record.get("messagesAdded", []):
                mid = added["message"]["id"]
                if mid in seen:
                    continue
                seen.add(mid)
                try:
                    msg = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            lambda m=mid: messages_api.get(
                                userId="me", id=m, format="metadata",
                                metadataHeaders=[
                                    "From", "To", "Cc", "Subject", "Date",
                                ],
                            ).execute(),
                        ),
                        timeout=API_CALL_TIMEOUT,
                    )
                    event = _build_ingest_event(msg)
                    await queue.put(event)
                    SOURCE_WATCHER_EVENTS.labels(
                        source_type="gmail", event_type="created",
                    ).inc()
                    WATCHER_LAST_EVENT_TIMESTAMP.labels(
                        source_type="gmail",
                    ).set_to_current_time()
                    count += 1
                except Exception:
                    logger.exception("Failed to fetch message %s", mid)

        if count:
            logger.info("Incremental poll: %d new message(s)", count)

        _save_cursor(self._cursor_path, new_history_id)
        return new_history_id
