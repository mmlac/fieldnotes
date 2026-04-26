"""Gmail polling source shim with cursor-based incremental sync.

Mirrors the Go GmailSource design: ticker-based polling at a configurable
interval, cursor persistence for incremental polling, initial backfill of
recent threads, and label filtering.  Emits one IngestEvent per email
message with structured metadata.

Config section ``[sources.gmail]``::

    poll_interval_seconds = 300
    max_initial_threads = 500
    # label_filter = "INBOX"   # optional — omit to fetch all messages
    client_secrets_path = "~/.fieldnotes/credentials.json"
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from worker.queue import PersistentQueue

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from worker.metrics import (
    GMAIL_POLL_DURATION,
    INDEXED_PREFILTER_SKIPPED,
    SOURCE_WATCHER_EVENTS,
    WATCHER_ACTIVE,
    WATCHER_LAST_EVENT_TIMESTAMP,
    initial_sync_add_items,
    initial_sync_source_done,
    observe_duration,
)

from worker.log_sanitizer import redact_home_path

from .base import IndexedCheck, PythonSource
from .cursor import save_json_atomic
from .gmail_auth import get_credentials

logger = logging.getLogger(__name__)

def _default_cursor_path(account: str) -> Path:
    """Per-account cursor file: ~/.fieldnotes/data/gmail_cursor-{account}.json."""
    return Path.home() / ".fieldnotes" / "data" / f"gmail_cursor-{account}.json"


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
        hid = data.get("history_id")
        if hid is None:
            return None
        # Validate cursor is a valid integer string before returning
        int(hid)
        return hid
    except (json.JSONDecodeError, OSError):
        logger.warning(
            "Failed to read cursor file %s, starting fresh", redact_home_path(str(path))
        )
        return None
    except (ValueError, TypeError):
        logger.warning(
            "Corrupted cursor in %s (non-integer value %r), resetting",
            redact_home_path(str(path)),
            hid,
        )
        return None


def _save_cursor(path: Path, history_id: str) -> None:
    """Persist the latest history ID to disk."""
    save_json_atomic(path, {"history_id": history_id})


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


def _extract_body(payload: dict[str, Any]) -> tuple[str, str]:
    """Extract body text from a Gmail full-format message payload.

    Returns ``(text, mime_type)``.  Prefers *text/plain* over *text/html*;
    recurses into multipart payloads.  Returns ``("", "")`` when no body
    is found.
    """
    mime = payload.get("mimeType", "")
    if mime in ("text/plain", "text/html"):
        data = payload.get("body", {}).get("data", "")
        if data:
            decoded = base64.urlsafe_b64decode(data + "==").decode(
                "utf-8", errors="replace"
            )
            return decoded, mime
        return "", mime

    # Recurse into multipart payloads, preferring text/plain over text/html
    parts = payload.get("parts", [])
    for part in parts:
        text, found_mime = _extract_body(part)
        if found_mime == "text/plain" and text:
            return text, "text/plain"
    for part in parts:
        text, found_mime = _extract_body(part)
        if found_mime == "text/html" and text:
            return text, "text/html"
    return "", mime


def _build_ingest_event(msg: dict[str, Any], account: str = "") -> dict[str, Any]:
    """Build an IngestEvent dict from a Gmail API message resource."""
    payload = msg.get("payload", {})
    headers = payload.get("headers", [])
    subject = _header_value(headers, "Subject")
    sender = _header_value(headers, "From")
    date_str = _header_value(headers, "Date")
    recipients = _extract_recipients(headers)

    internal_ts = int(msg.get("internalDate", "0"))
    source_modified = datetime.fromtimestamp(
        internal_ts / 1000, tz=timezone.utc
    ).isoformat()

    body_text, body_mime = _extract_body(payload)

    meta: dict[str, Any] = {
        "message_id": msg["id"],
        "thread_id": msg.get("threadId", ""),
        "subject": subject,
        "date": date_str,
        "sender_email": sender,
        "recipients": recipients,
        "account": account,
    }

    return {
        "id": str(uuid.uuid4()),
        "source_type": "gmail",
        "source_id": f"gmail:{msg['id']}",
        "operation": "created",
        "text": body_text,
        "mime_type": body_mime or "message/rfc822",
        "meta": meta,
        "enqueued_at": datetime.now(timezone.utc).isoformat(),
        "source_modified_at": source_modified,
    }


class GmailSource(PythonSource):
    """Polls Gmail via the API and emits IngestEvent dicts.

    Config keys (from ``[sources.gmail]``):
        poll_interval_seconds: int  — polling interval (default: 300)
        max_initial_threads: int    — backfill limit (default: 500)
        label_filter: str           — Gmail label to poll (optional, omit for all)
        client_secrets_path: str    — OAuth2 client secrets file (required)
        cursor_path: str            — cursor persistence file (optional)
    """

    def __init__(self) -> None:
        self._account: str = ""
        self._poll_interval: int = DEFAULT_POLL_INTERVAL
        self._max_initial_threads: int = DEFAULT_MAX_INITIAL_THREADS
        self._label_filter: str | None = None
        self._client_secrets_path: Path | None = None
        self._cursor_path: Path | None = None

    def name(self) -> str:
        return "gmail"

    @property
    def source_id(self) -> str:
        """Stable identifier surfacing the account: ``gmail:{account}``."""
        return f"gmail:{self._account}"

    @property
    def _cursor_key(self) -> str:
        """Per-account PersistentQueue cursor key (no cross-account stomping)."""
        return f"gmail:{self._account}"

    def configure(self, cfg: dict[str, Any]) -> None:
        account = cfg.get("account")
        if not account or not isinstance(account, str):
            raise ValueError(
                "GmailSource requires 'account' in config (the account label "
                "from [sources.gmail.<account>])"
            )
        self._account = account

        secrets = cfg.get("client_secrets_path")
        if not secrets:
            raise ValueError("GmailSource requires 'client_secrets_path' in config")
        self._client_secrets_path = Path(secrets).expanduser().resolve()

        self._poll_interval = int(
            cfg.get("poll_interval_seconds", DEFAULT_POLL_INTERVAL)
        )
        self._max_initial_threads = int(
            cfg.get("max_initial_threads", DEFAULT_MAX_INITIAL_THREADS)
        )
        lf = cfg.get("label_filter")
        if isinstance(lf, str) and lf:
            self._label_filter = lf
        else:
            self._label_filter = None

        cursor = cfg.get("cursor_path")
        if cursor:
            self._cursor_path = Path(cursor).expanduser().resolve()
        else:
            self._cursor_path = _default_cursor_path(self._account)

    async def start(
        self,
        queue: PersistentQueue,
        *,
        indexed_check: IndexedCheck | None = None,
    ) -> None:
        if self._client_secrets_path is None or not self._account:
            raise ValueError(
                "GmailSource.start() called before configure() — "
                "client_secrets_path or account is not set"
            )

        creds = get_credentials(self._client_secrets_path, account=self._account)
        service = build("gmail", "v1", credentials=creds)
        messages_api = service.users().messages()

        # Validate that the configured label exists
        if self._label_filter:
            await self._validate_label(service)

        # Load cursor from persistent queue (migrated from JSON file).
        raw_cursor = queue.load_cursor(self._cursor_key)
        cursor: str | None = None
        if raw_cursor:
            try:
                cursor = json.loads(raw_cursor).get("history_id")
            except (json.JSONDecodeError, AttributeError):
                cursor = None
        # Fall back to legacy JSON file if not yet migrated.
        if cursor is None:
            cursor = _load_cursor(self._cursor_path)

        WATCHER_ACTIVE.labels(source_type="gmail").set(1)

        if cursor is None:
            # Initial backfill: fetch recent messages up to max_initial_threads
            logger.info(
                "No cursor found — backfilling up to %d messages (label=%s)",
                self._max_initial_threads,
                self._label_filter,
            )
            cursor = await self._backfill(
                messages_api, queue, is_initial=True, indexed_check=indexed_check
            )

        initial_sync_source_done()

        # Polling loop
        try:
            while True:
                await asyncio.sleep(self._poll_interval)
                with observe_duration(GMAIL_POLL_DURATION):
                    if cursor is None:
                        # No valid cursor (e.g. history ID expired) — re-backfill.
                        logger.info("No cursor available — re-running backfill")
                        cursor = await self._backfill(
                            messages_api,
                            queue,
                            is_initial=False,
                            indexed_check=indexed_check,
                        )
                        # history_id saved via _ProgressTracker callbacks
                    else:
                        cursor = await self._poll_incremental(
                            service, messages_api, queue, cursor
                        )
        except asyncio.CancelledError:
            WATCHER_ACTIVE.labels(source_type="gmail").set(0)
            if cursor:
                _save_cursor(self._cursor_path, cursor)
            raise

    async def _validate_label(self, service: Any) -> None:
        """Verify the configured label_filter exists in the user's Gmail account."""
        loop = asyncio.get_running_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: service.users().labels().list(userId="me").execute(),
                ),
                timeout=API_CALL_TIMEOUT,
            )
            label_names = {lb.get("name", "") for lb in result.get("labels", [])}
            label_ids = {lb.get("id", "") for lb in result.get("labels", [])}
            if (
                self._label_filter not in label_names
                and self._label_filter not in label_ids
            ):
                logger.warning(
                    "Configured label_filter %r not found in Gmail account. "
                    "Available labels: %s. Queries using this label may return "
                    "empty results.",
                    self._label_filter,
                    ", ".join(sorted(label_names)),
                )
        except Exception:
            logger.warning(
                "Failed to validate label %r — continuing anyway",
                self._label_filter,
                exc_info=True,
            )

    async def _backfill(
        self,
        messages_api: Any,
        queue: asyncio.Queue[dict[str, Any]],
        *,
        is_initial: bool = True,
        indexed_check: IndexedCheck | None = None,
    ) -> str | None:
        """Fetch up to max_initial_threads recent messages and return the
        latest historyId for subsequent incremental polling.

        Includes rate limiting (delay between pages) and retry with
        exponential backoff to avoid hitting Gmail API rate limits.

        When *indexed_check* is provided, each page of message stubs is
        first batched against Neo4j to skip messages whose ``gmail:{id}``
        source_id is already chunked.  This avoids the per-message
        ``messages.get(format="full")`` call entirely for items that have
        already been processed — Gmail messages are content-immutable
        once sent, so re-fetching them would only re-do the same work.

        Tradeoff: label-only changes (archive, star, folder move) on
        already-indexed messages are not picked up by this skip path.
        That's intentional; if labels become important the right fix is
        a separate label-sync pass, not re-running the full pipeline.
        """
        loop = asyncio.get_running_loop()
        fetched = 0
        page_token: str | None = None
        latest_history_id: str | None = None
        seen_ids: set[str] = set()  # Dedup across interruption/restart
        pending_events: list[dict[str, Any]] = []

        while fetched < self._max_initial_threads:
            batch_size = min(100, self._max_initial_threads - fetched)
            kwargs: dict[str, Any] = {
                "userId": "me",
                "maxResults": batch_size,
            }
            if self._label_filter:
                kwargs["labelIds"] = [self._label_filter]
            if page_token:
                kwargs["pageToken"] = page_token

            result = await self._api_call_with_retry(
                loop, lambda kw=kwargs: messages_api.list(**kw).execute()
            )
            msg_stubs = result.get("messages", [])
            if not msg_stubs:
                break

            # Phase 1 pre-filter: batch-check Neo4j for the entire page so
            # we can skip the per-message GET call (the expensive one).
            already_indexed_sids: set[str] = set()
            if indexed_check is not None:
                candidate_sids = [f"gmail:{stub['id']}" for stub in msg_stubs]
                try:
                    already_indexed_sids = await loop.run_in_executor(
                        None, indexed_check, candidate_sids
                    )
                except Exception:
                    logger.warning(
                        "Gmail indexed_check failed; processing all messages "
                        "in this page",
                        exc_info=True,
                    )
                    already_indexed_sids = set()

            for stub in msg_stubs:
                if fetched >= self._max_initial_threads:
                    break
                if stub["id"] in seen_ids:
                    continue
                seen_ids.add(stub["id"])
                if f"gmail:{stub['id']}" in already_indexed_sids:
                    INDEXED_PREFILTER_SKIPPED.labels(source_type="gmail").inc()
                    fetched += 1
                    continue
                msg = await self._api_call_with_retry(
                    loop,
                    lambda mid=stub["id"]: messages_api.get(
                        userId="me",
                        id=mid,
                        format="full",
                    ).execute(),
                )
                event = _build_ingest_event(msg, self._account)
                if is_initial:
                    event["initial_scan"] = True
                # Skip events already in the queue (pending/processing).
                if queue.is_enqueued(event["source_id"]):
                    fetched += 1
                    # Still track historyId for cursor
                    hid = msg.get("historyId")
                    if hid:
                        try:
                            if latest_history_id is None or int(hid) > int(
                                latest_history_id
                            ):
                                latest_history_id = hid
                        except (ValueError, TypeError):
                            pass
                    continue
                pending_events.append(event)
                SOURCE_WATCHER_EVENTS.labels(
                    source_type="gmail",
                    event_type="created",
                ).inc()
                WATCHER_LAST_EVENT_TIMESTAMP.labels(
                    source_type="gmail",
                ).set_to_current_time()
                fetched += 1

                # Track the highest historyId seen
                hid = msg.get("historyId")
                if hid:
                    try:
                        if latest_history_id is None or int(hid) > int(
                            latest_history_id
                        ):
                            latest_history_id = hid
                    except (ValueError, TypeError):
                        logger.warning(
                            "Malformed historyId %r in message %s, skipping",
                            hid,
                            stub["id"],
                        )

            page_token = result.get("nextPageToken")
            if not page_token:
                break

            # Rate-limit: pause between page fetches
            await asyncio.sleep(BACKFILL_PAGE_DELAY)

        # Enqueue collected events; save cursor atomically with the last
        # event so a crash mid-enqueue doesn't leave cursor stale.
        if pending_events:
            cursor_val = (
                json.dumps({"history_id": latest_history_id})
                if latest_history_id
                else None
            )
            for i, ev in enumerate(pending_events):
                is_last = i == len(pending_events) - 1
                queue.enqueue(
                    ev,
                    cursor_key=self._cursor_key if is_last and cursor_val else None,
                    cursor_value=cursor_val if is_last else None,
                )
            if latest_history_id:
                logger.info(
                    "Gmail backfill cursor saved (account=%s, history_id=%s)",
                    self._account,
                    latest_history_id,
                )

        if fetched > 0 and is_initial:
            initial_sync_add_items(fetched)
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
                        **({
                            "labelId": self._label_filter,
                        } if self._label_filter else {}),
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
        except HttpError as exc:
            if exc.resp.status == 404:
                # History ID has expired (Google purges history after ~30 days).
                # Reset the cursor so the next poll triggers a full backfill.
                logger.warning(
                    "Gmail history ID %s no longer valid (404) — resetting cursor for full backfill",
                    cursor,
                )
                try:
                    self._cursor_path.unlink(missing_ok=True)
                except OSError:
                    logger.warning(
                        "Failed to delete stale cursor file %s",
                        redact_home_path(str(self._cursor_path)),
                    )
                return None
            logger.exception("Failed to fetch Gmail history (cursor=%s)", cursor)
            return cursor

        new_history_id = result.get("historyId", cursor)
        records = result.get("history", [])

        seen: set[str] = set()
        pending_events: list[dict[str, Any]] = []
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
                                userId="me",
                                id=m,
                                format="full",
                            ).execute(),
                        ),
                        timeout=API_CALL_TIMEOUT,
                    )
                    event = _build_ingest_event(msg, self._account)
                    pending_events.append(event)
                    SOURCE_WATCHER_EVENTS.labels(
                        source_type="gmail",
                        event_type="created",
                    ).inc()
                    WATCHER_LAST_EVENT_TIMESTAMP.labels(
                        source_type="gmail",
                    ).set_to_current_time()
                except asyncio.TimeoutError:
                    logger.error(
                        "Timed out fetching message %s after %ds",
                        mid,
                        API_CALL_TIMEOUT,
                    )
                except HttpError:
                    logger.error("Failed to fetch message %s", mid, exc_info=True)

        if pending_events:
            logger.info("Incremental poll: %d new message(s)", len(pending_events))
            cursor_val = json.dumps({"history_id": new_history_id})
            for i, ev in enumerate(pending_events):
                is_last = i == len(pending_events) - 1
                queue.enqueue(
                    ev,
                    cursor_key=self._cursor_key if is_last else None,
                    cursor_value=cursor_val if is_last else None,
                )
        else:
            # No new messages but historyId may have advanced — safe to
            # save immediately since there's nothing in the pipeline.
            queue.save_cursor(
                self._cursor_key, json.dumps({"history_id": new_history_id})
            )

        return new_history_id
