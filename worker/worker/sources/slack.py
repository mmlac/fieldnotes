"""Slack ingestion source: backfill, polling, burst-window splitting.

Discovers Slack conversations (public/private channels, DMs, MPIMs),
backfills their history, and polls for new messages.  Emits one
IngestEvent per thread (parent + replies) and one per "burst window" of
un-threaded messages — windows close on a configurable time gap or
token budget so chunking sees coherent slices of conversation rather
than an unbounded firehose.

Document IDs are stable so the pipeline's delete-before-rewrite path
handles edits without duplicating points/nodes:

* Thread       — ``slack://{team_id}/{channel_id}/thread/{parent_ts}``
* Burst window — ``slack://{team_id}/{channel_id}/window/{first_ts}-{last_ts}``

Out of scope (future beads): file uploads, image OCR, reactions.

Config section ``[sources.slack]`` is parsed by ``worker.config`` into a
:class:`SlackSourceConfig`; pass ``dataclasses.asdict(cfg.slack)`` to
:meth:`SlackSource.configure`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from worker.queue import PersistentQueue

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from worker.metrics import (
    SOURCE_WATCHER_EVENTS,
    WATCHER_ACTIVE,
    WATCHER_LAST_EVENT_TIMESTAMP,
    initial_sync_add_items,
    initial_sync_source_done,
)

from .base import IndexedCheck, PythonSource
from .cursor import save_json_atomic
from .slack_auth import DEFAULT_TOKEN_PATH, SlackToken, get_slack_client

logger = logging.getLogger(__name__)

DEFAULT_CURSOR_PATH = Path.home() / ".fieldnotes" / "data" / "slack_cursor.json"
DEFAULT_POLL_INTERVAL = 300
DEFAULT_MAX_INITIAL_DAYS = 90
DEFAULT_WINDOW_MAX_TOKENS = 512
DEFAULT_WINDOW_GAP_SECONDS = 1800
DEFAULT_WINDOW_OVERLAP = 3
HISTORY_PAGE_SIZE = 200
TOKEN_CHARS_PER_TOKEN = 4  # rough estimator: ~4 chars per token


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _ts_seconds(ts: str | None) -> float:
    """Parse a Slack ``ts`` ('1234567890.123456') into float seconds."""
    if not ts:
        return 0.0
    try:
        return float(ts)
    except (TypeError, ValueError):
        return 0.0


def _ts_to_iso(ts: str | None) -> str:
    """Convert a Slack ts to an ISO-8601 UTC string (or '' on bad input)."""
    sec = _ts_seconds(ts)
    if sec <= 0:
        return ""
    return datetime.fromtimestamp(sec, tz=timezone.utc).isoformat()


def _estimate_tokens(text: str) -> int:
    """Rough token estimate; one token per ~4 characters."""
    if not text:
        return 0
    return max(1, len(text) // TOKEN_CHARS_PER_TOKEN)


def _doc_id_thread(team_id: str, channel_id: str, parent_ts: str) -> str:
    return f"slack://{team_id}/{channel_id}/thread/{parent_ts}"


def _doc_id_window(team_id: str, channel_id: str, first_ts: str, last_ts: str) -> str:
    return f"slack://{team_id}/{channel_id}/window/{first_ts}-{last_ts}"


def _format_messages(messages: list[dict[str, Any]]) -> str:
    """Render a list of Slack messages to a single text body for indexing."""
    lines: list[str] = []
    for m in messages:
        user = m.get("user") or m.get("bot_id") or "unknown"
        iso = _ts_to_iso(m.get("ts"))
        text = (m.get("text") or "").rstrip()
        lines.append(f"[{iso}] {user}: {text}")
    return "\n".join(lines)


def _channel_meta(channel: dict[str, Any]) -> dict[str, Any]:
    return {
        "channel_id": channel.get("id", ""),
        "channel_name": channel.get("name", ""),
        "is_im": bool(channel.get("is_im")),
        "is_mpim": bool(channel.get("is_mpim")),
        "is_private": bool(channel.get("is_private")) or bool(channel.get("is_group")),
        "is_archived": bool(channel.get("is_archived")),
    }


def _is_thread_parent(msg: dict[str, Any]) -> bool:
    """True iff *msg* is a thread parent (has replies)."""
    thread_ts = msg.get("thread_ts")
    return bool(thread_ts) and thread_ts == msg.get("ts")


def _is_thread_reply(msg: dict[str, Any]) -> bool:
    """True iff *msg* is a reply within someone else's thread."""
    thread_ts = msg.get("thread_ts")
    return bool(thread_ts) and thread_ts != msg.get("ts")


# ---------------------------------------------------------------------------
# IngestEvent builders
# ---------------------------------------------------------------------------


def _build_thread_event(
    *,
    team_id: str,
    channel: dict[str, Any],
    parent: dict[str, Any],
    replies: list[dict[str, Any]],
    operation: str = "created",
) -> dict[str, Any]:
    """Build an IngestEvent for a thread (parent + ordered replies)."""
    parent_ts = parent.get("ts", "")
    cmeta = _channel_meta(channel)
    # Replies API returns the parent first; drop dupes if any.
    seen: set[str] = set()
    ordered: list[dict[str, Any]] = []
    for m in [parent, *replies]:
        ts = m.get("ts", "")
        if ts and ts in seen:
            continue
        seen.add(ts)
        ordered.append(m)
    ordered.sort(key=lambda m: _ts_seconds(m.get("ts")))

    last_ts = ordered[-1].get("ts", parent_ts) if ordered else parent_ts
    text = _format_messages(ordered)
    source_id = _doc_id_thread(team_id, cmeta["channel_id"], parent_ts)

    meta: dict[str, Any] = {
        "team_id": team_id,
        **cmeta,
        "kind": "thread",
        "parent_ts": parent_ts,
        "last_ts": last_ts,
        "message_ts": [m.get("ts", "") for m in ordered],
        "users": sorted({m.get("user", "") for m in ordered if m.get("user")}),
        "reply_count": max(0, len(ordered) - 1),
    }

    return {
        "id": str(uuid.uuid4()),
        "source_type": "slack",
        "source_id": source_id,
        "operation": operation,
        "text": text,
        "mime_type": "text/plain",
        "meta": meta,
        "enqueued_at": datetime.now(timezone.utc).isoformat(),
        "source_modified_at": _ts_to_iso(last_ts)
        or datetime.now(timezone.utc).isoformat(),
    }


def _build_window_event(
    *,
    team_id: str,
    channel: dict[str, Any],
    messages: list[dict[str, Any]],
    operation: str = "created",
) -> dict[str, Any]:
    """Build an IngestEvent for a burst window of un-threaded messages."""
    if not messages:
        raise ValueError("_build_window_event requires at least one message")
    cmeta = _channel_meta(channel)
    first_ts = messages[0].get("ts", "")
    last_ts = messages[-1].get("ts", "")
    source_id = _doc_id_window(team_id, cmeta["channel_id"], first_ts, last_ts)
    text = _format_messages(messages)

    meta: dict[str, Any] = {
        "team_id": team_id,
        **cmeta,
        "kind": "window",
        "first_ts": first_ts,
        "last_ts": last_ts,
        "message_ts": [m.get("ts", "") for m in messages],
        "users": sorted({m.get("user", "") for m in messages if m.get("user")}),
        "message_count": len(messages),
    }

    return {
        "id": str(uuid.uuid4()),
        "source_type": "slack",
        "source_id": source_id,
        "operation": operation,
        "text": text,
        "mime_type": "text/plain",
        "meta": meta,
        "enqueued_at": datetime.now(timezone.utc).isoformat(),
        "source_modified_at": _ts_to_iso(last_ts)
        or datetime.now(timezone.utc).isoformat(),
    }


def _build_delete_event(*, source_id: str, team_id: str = "") -> dict[str, Any]:
    """Build an IngestEvent that instructs the pipeline to delete *source_id*."""
    return {
        "id": str(uuid.uuid4()),
        "source_type": "slack",
        "source_id": source_id,
        "operation": "deleted",
        "text": "",
        "mime_type": "text/plain",
        "meta": {"team_id": team_id, "source_id": source_id},
        "enqueued_at": datetime.now(timezone.utc).isoformat(),
        "source_modified_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Burst-window splitting
# ---------------------------------------------------------------------------


def split_into_windows(
    messages: list[dict[str, Any]],
    *,
    window_gap_seconds: int = DEFAULT_WINDOW_GAP_SECONDS,
    window_max_tokens: int = DEFAULT_WINDOW_MAX_TOKENS,
    window_overlap_messages: int = DEFAULT_WINDOW_OVERLAP,
) -> list[list[dict[str, Any]]]:
    """Split a chronologically-ordered list of messages into burst windows.

    A window closes when EITHER the gap between consecutive ``ts`` values
    exceeds *window_gap_seconds* OR the running token estimate would
    exceed *window_max_tokens*.

    Token-based splits within a continuous burst overlap by
    *window_overlap_messages* whole messages so chunk boundaries don't
    sever a single thought.  Gap-based splits do NOT overlap — a quiet
    period is a natural conversational break.
    """
    if not messages:
        return []

    sorted_msgs = sorted(messages, key=lambda m: _ts_seconds(m.get("ts")))
    windows: list[list[dict[str, Any]]] = []
    buf: list[dict[str, Any]] = []
    buf_tokens = 0

    for msg in sorted_msgs:
        msg_tokens = _estimate_tokens(msg.get("text", ""))

        if not buf:
            buf.append(msg)
            buf_tokens = msg_tokens
            continue

        prev_ts = _ts_seconds(buf[-1].get("ts"))
        cur_ts = _ts_seconds(msg.get("ts"))
        gap = cur_ts - prev_ts

        # (a) Gap-based close: natural break, no overlap.
        if gap > window_gap_seconds:
            windows.append(buf)
            buf = [msg]
            buf_tokens = msg_tokens
            continue

        # (b) Token-based close: mid-burst split, carry overlap.
        if buf_tokens + msg_tokens >= window_max_tokens:
            windows.append(buf)
            overlap = (
                buf[-window_overlap_messages:] if window_overlap_messages > 0 else []
            )
            buf = [*overlap, msg]
            buf_tokens = sum(_estimate_tokens(m.get("text", "")) for m in buf)
            continue

        buf.append(msg)
        buf_tokens += msg_tokens

    if buf:
        windows.append(buf)
    return windows


# ---------------------------------------------------------------------------
# Cursor file
# ---------------------------------------------------------------------------


def _load_cursor(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Load slack cursor: ``{team_id: {channel_id: {latest_ts, last_synced}}}``."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load slack cursor at %s: %s", path, exc)
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for team, channels in data.items():
        if not isinstance(channels, dict):
            continue
        out[team] = {}
        for ch, entry in channels.items():
            if isinstance(entry, dict):
                out[team][ch] = dict(entry)
    return out


def _save_cursor(path: Path, cursor: dict[str, dict[str, dict[str, Any]]]) -> None:
    save_json_atomic(path, cursor)


# ---------------------------------------------------------------------------
# Channel filter
# ---------------------------------------------------------------------------


def _channel_passes_filter(
    channel: dict[str, Any],
    *,
    include_channels: list[str],
    exclude_channels: list[str],
    include_dms: bool,
    include_archived: bool,
) -> bool:
    """Return True if *channel* should be polled, given config filters."""
    if not include_archived and channel.get("is_archived"):
        return False
    is_im = channel.get("is_im") or channel.get("is_mpim")
    if is_im and not include_dms:
        return False

    cid = channel.get("id", "")
    cname = channel.get("name", "")

    if include_channels:
        # When include is set, only items matching by name OR id pass.
        return cid in include_channels or cname in include_channels

    if exclude_channels:
        if cid in exclude_channels or cname in exclude_channels:
            return False
    return True


# ---------------------------------------------------------------------------
# SlackSource
# ---------------------------------------------------------------------------


class SlackSource(PythonSource):
    """Polls Slack via the Web API and emits IngestEvent dicts.

    Per-conversation cursors track the latest ``ts`` we've ingested.  On
    first poll for a conversation, backfills history from
    ``now - max_initial_days`` and emits one IngestEvent per thread plus
    one per burst window.  On subsequent polls fetches messages with
    ``oldest = cursor.latest_ts`` and applies the same windowing logic
    against the buffered new arrivals.

    Edits and deletes (subtype ``message_changed`` / ``message_deleted``)
    are translated into a delete event for the affected document plus a
    re-emit of the rebuilt window or thread.
    """

    def __init__(self) -> None:
        self._poll_interval = DEFAULT_POLL_INTERVAL
        self._max_initial_days = DEFAULT_MAX_INITIAL_DAYS
        self._include_channels: list[str] = []
        self._exclude_channels: list[str] = []
        self._include_dms = True
        self._include_archived = False
        self._window_max_tokens = DEFAULT_WINDOW_MAX_TOKENS
        self._window_gap_seconds = DEFAULT_WINDOW_GAP_SECONDS
        self._window_overlap_messages = DEFAULT_WINDOW_OVERLAP
        self._download_files = False
        self._client_secrets_path = "~/.fieldnotes/slack_credentials.json"
        self._token_path: Path = DEFAULT_TOKEN_PATH
        self._cursor_path: Path = DEFAULT_CURSOR_PATH

        # Test seam: when set, used in place of running auth.test/OAuth.
        self._client: WebClient | None = None
        self._team_id: str = ""
        # In-memory map of (channel_id, ts) → emitted document source_id.
        # Used to translate edits/deletes into delete-before-rewrite.
        self._ts_to_doc: dict[tuple[str, str], str] = {}
        # Most recent open window per channel: (channel_id, first_ts) →
        # last emitted source_id.  Lets a polling cycle close a previously
        # open window cleanly when a new burst arrives.
        self._open_window: dict[str, str] = {}

    def name(self) -> str:
        return "slack"

    # ------------------------------------------------------------------
    # configure
    # ------------------------------------------------------------------

    def configure(self, cfg: dict[str, Any]) -> None:
        self._poll_interval = int(
            cfg.get("poll_interval_seconds", DEFAULT_POLL_INTERVAL)
        )
        self._max_initial_days = int(
            cfg.get("max_initial_days", DEFAULT_MAX_INITIAL_DAYS)
        )
        self._include_channels = list(cfg.get("include_channels", []) or [])
        self._exclude_channels = list(cfg.get("exclude_channels", []) or [])
        self._include_dms = bool(cfg.get("include_dms", True))
        self._include_archived = bool(cfg.get("include_archived", False))
        self._window_max_tokens = int(
            cfg.get("window_max_tokens", DEFAULT_WINDOW_MAX_TOKENS)
        )
        self._window_gap_seconds = int(
            cfg.get("window_gap_seconds", DEFAULT_WINDOW_GAP_SECONDS)
        )
        self._window_overlap_messages = int(
            cfg.get("window_overlap_messages", DEFAULT_WINDOW_OVERLAP)
        )
        self._download_files = bool(cfg.get("download_files", False))
        if "client_secrets_path" in cfg:
            self._client_secrets_path = cfg["client_secrets_path"]
        if "token_path" in cfg:
            self._token_path = Path(cfg["token_path"]).expanduser().resolve()
        if "cursor_path" in cfg:
            self._cursor_path = Path(cfg["cursor_path"]).expanduser().resolve()

    # ------------------------------------------------------------------
    # start
    # ------------------------------------------------------------------

    async def start(
        self,
        queue: PersistentQueue,
        *,
        indexed_check: IndexedCheck | None = None,
    ) -> None:
        client, team_id = await asyncio.to_thread(self._resolve_client)
        self._client = client
        self._team_id = team_id

        # Load cursor (queue-backed first, falling back to legacy file).
        raw = queue.load_cursor("slack")
        cursor: dict[str, dict[str, dict[str, Any]]] = {}
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    cursor = {
                        t: dict(c) for t, c in parsed.items() if isinstance(c, dict)
                    }
            except json.JSONDecodeError:
                cursor = {}
        if not cursor:
            cursor = _load_cursor(self._cursor_path)

        WATCHER_ACTIVE.labels(source_type="slack").set(1)
        first_cycle = True
        try:
            while True:
                channels = await asyncio.to_thread(self._discover_conversations)
                team_cur = cursor.setdefault(team_id, {})
                for ch in channels:
                    cid = ch.get("id", "")
                    if not cid:
                        continue
                    try:
                        await self._poll_conversation(
                            ch,
                            team_cur,
                            queue,
                            indexed_check=indexed_check,
                            is_initial_cycle=first_cycle,
                        )
                    except SlackApiError as exc:
                        logger.error("Slack API error polling channel %s: %s", cid, exc)
                    except Exception:
                        logger.exception("Unexpected error polling channel %s", cid)
                    # Persist after each channel — atomic, cheap.
                    _save_cursor(self._cursor_path, cursor)
                    queue.save_cursor("slack", json.dumps(cursor))

                if first_cycle:
                    initial_sync_source_done()
                    first_cycle = False

                await asyncio.sleep(self._poll_interval)
        except asyncio.CancelledError:
            WATCHER_ACTIVE.labels(source_type="slack").set(0)
            _save_cursor(self._cursor_path, cursor)
            raise

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def _resolve_client(self) -> tuple[WebClient, str]:
        """Return an authenticated WebClient and the workspace team_id."""
        if self._client is not None:
            team_id = self._team_id
            if not team_id:
                # Fall back to auth.test for the workspace id.
                resp = self._client.auth_test()
                team_id = resp.get("team_id", "")
            return self._client, team_id

        config: dict[str, Any] = {}
        secrets_path = Path(self._client_secrets_path).expanduser()
        if secrets_path.is_file():
            try:
                config.update(json.loads(secrets_path.read_text()))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "Failed to read slack client secrets at %s: %s",
                    secrets_path,
                    exc,
                )

        client = get_slack_client(config, token_path=self._token_path)

        # Recover team_id from the persisted token bundle.
        team_id = ""
        if self._token_path.exists():
            try:
                tok = SlackToken.from_dict(json.loads(self._token_path.read_text()))
                team_id = tok.team_id
            except (json.JSONDecodeError, KeyError, OSError):
                team_id = ""
        if not team_id:
            try:
                resp = client.auth_test()
                team_id = resp.get("team_id", "")
            except SlackApiError:
                logger.exception("auth.test failed; team_id will be empty")
        return client, team_id

    # ------------------------------------------------------------------
    # Conversation discovery
    # ------------------------------------------------------------------

    def _discover_conversations(self) -> list[dict[str, Any]]:
        """Page conversations.list and apply config filters."""
        assert self._client is not None
        types = "public_channel,private_channel,mpim"
        if self._include_dms:
            types = types + ",im"
        out: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            kwargs: dict[str, Any] = {
                "types": types,
                "limit": HISTORY_PAGE_SIZE,
                "exclude_archived": not self._include_archived,
            }
            if cursor:
                kwargs["cursor"] = cursor
            resp = self._client.conversations_list(**kwargs)
            for ch in resp.get("channels", []) or []:
                if _channel_passes_filter(
                    ch,
                    include_channels=self._include_channels,
                    exclude_channels=self._exclude_channels,
                    include_dms=self._include_dms,
                    include_archived=self._include_archived,
                ):
                    out.append(ch)
            cursor = (resp.get("response_metadata") or {}).get("next_cursor") or ""
            if not cursor:
                break
        return out

    # ------------------------------------------------------------------
    # Polling / backfill core
    # ------------------------------------------------------------------

    async def _poll_conversation(
        self,
        channel: dict[str, Any],
        team_cursor: dict[str, dict[str, Any]],
        queue: Any,
        *,
        indexed_check: IndexedCheck | None = None,
        is_initial_cycle: bool = False,
    ) -> None:
        """Backfill or incremental poll for a single conversation."""
        cid = channel.get("id", "")
        entry = team_cursor.get(cid)
        is_backfill = entry is None or not entry.get("latest_ts")

        if is_backfill:
            oldest = (
                datetime.now(timezone.utc).timestamp() - self._max_initial_days * 86400
            )
            oldest_str = f"{oldest:.6f}"
        else:
            oldest_str = entry["latest_ts"]

        messages = await asyncio.to_thread(self._fetch_history, cid, oldest_str)
        if not messages:
            # Even with no messages, advance cursor's last_synced timestamp.
            team_cursor[cid] = {
                "latest_ts": entry.get("latest_ts", "") if entry else "",
                "last_synced": datetime.now(timezone.utc).isoformat(),
            }
            return

        # Sort ascending by ts so windowing/threads see chronological order.
        messages.sort(key=lambda m: _ts_seconds(m.get("ts")))

        latest_ts_seen = max(
            (m.get("ts", "") for m in messages),
            key=lambda t: _ts_seconds(t),
            default=entry.get("latest_ts", "") if entry else "",
        )

        # 1) Edits / deletes — emit delete + re-emit per affected doc.
        await self._handle_edits_and_deletes(channel, messages, queue)

        # 2) Threads — fetch replies and emit one event per thread parent.
        thread_events = await asyncio.to_thread(
            self._build_thread_events, channel, messages
        )

        # 3) Burst windows — non-thread, non-reply messages only.
        non_thread = [
            m
            for m in messages
            if not _is_thread_parent(m)
            and not _is_thread_reply(m)
            and m.get("subtype") not in {"message_changed", "message_deleted"}
        ]
        windows = split_into_windows(
            non_thread,
            window_gap_seconds=self._window_gap_seconds,
            window_max_tokens=self._window_max_tokens,
            window_overlap_messages=self._window_overlap_messages,
        )
        window_events: list[dict[str, Any]] = []
        for win in windows:
            ev = _build_window_event(
                team_id=self._team_id, channel=channel, messages=win
            )
            window_events.append(ev)
            for m in win:
                ts = m.get("ts", "")
                if ts:
                    self._ts_to_doc[(cid, ts)] = ev["source_id"]

        # 4) Enqueue all events.  Mark as initial_scan during backfill.
        all_events = [*thread_events, *window_events]
        for ev in all_events:
            if is_backfill:
                ev["initial_scan"] = True
            SOURCE_WATCHER_EVENTS.labels(
                source_type="slack",
                event_type=ev["operation"],
            ).inc()
            WATCHER_LAST_EVENT_TIMESTAMP.labels(source_type="slack").set(
                datetime.now(timezone.utc).timestamp()
            )

        # Update cursor entry alongside the last event for atomicity.
        team_cursor[cid] = {
            "latest_ts": latest_ts_seen
            or (entry.get("latest_ts", "") if entry else ""),
            "last_synced": datetime.now(timezone.utc).isoformat(),
        }
        cursor_value = json.dumps({self._team_id: team_cursor}) if all_events else None

        for i, ev in enumerate(all_events):
            is_last = i == len(all_events) - 1
            queue.enqueue(
                ev,
                cursor_key="slack" if is_last and cursor_value else None,
                cursor_value=cursor_value if is_last else None,
            )

        if is_backfill and all_events:
            initial_sync_add_items(len(all_events))
        if all_events:
            logger.info(
                "Slack channel %s: emitted %d event(s) (%d threads, %d windows)",
                cid,
                len(all_events),
                len(thread_events),
                len(window_events),
            )

    def _fetch_history(self, channel_id: str, oldest: str) -> list[dict[str, Any]]:
        """Page conversations.history starting at *oldest* (exclusive of older)."""
        assert self._client is not None
        out: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            kwargs: dict[str, Any] = {
                "channel": channel_id,
                "limit": HISTORY_PAGE_SIZE,
                "oldest": oldest,
                "inclusive": False,
            }
            if cursor:
                kwargs["cursor"] = cursor
            resp = self._client.conversations_history(**kwargs)
            out.extend(resp.get("messages", []) or [])
            cursor = (resp.get("response_metadata") or {}).get("next_cursor") or ""
            if not cursor or not resp.get("has_more"):
                break
        return out

    def _fetch_replies(self, channel_id: str, parent_ts: str) -> list[dict[str, Any]]:
        """Page conversations.replies for *parent_ts*."""
        assert self._client is not None
        out: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            kwargs: dict[str, Any] = {
                "channel": channel_id,
                "ts": parent_ts,
                "limit": HISTORY_PAGE_SIZE,
            }
            if cursor:
                kwargs["cursor"] = cursor
            resp = self._client.conversations_replies(**kwargs)
            out.extend(resp.get("messages", []) or [])
            cursor = (resp.get("response_metadata") or {}).get("next_cursor") or ""
            if not cursor or not resp.get("has_more"):
                break
        return out

    def _build_thread_events(
        self,
        channel: dict[str, Any],
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """For each thread parent in *messages*, fetch replies and build event."""
        cid = channel.get("id", "")
        events: list[dict[str, Any]] = []
        for m in messages:
            if not _is_thread_parent(m):
                continue
            parent_ts = m.get("ts", "")
            try:
                replies_full = self._fetch_replies(cid, parent_ts)
            except SlackApiError as exc:
                logger.warning(
                    "Failed to fetch replies for %s/%s: %s", cid, parent_ts, exc
                )
                continue
            # _fetch_replies returns parent + replies; use parent from API for freshness.
            parent = replies_full[0] if replies_full else m
            replies = replies_full[1:] if replies_full else []
            ev = _build_thread_event(
                team_id=self._team_id,
                channel=channel,
                parent=parent,
                replies=replies,
            )
            events.append(ev)
            # Index every ts in the thread → its document id for edit handling.
            for sub in [parent, *replies]:
                ts = sub.get("ts", "")
                if ts:
                    self._ts_to_doc[(cid, ts)] = ev["source_id"]
        return events

    async def _handle_edits_and_deletes(
        self,
        channel: dict[str, Any],
        messages: list[dict[str, Any]],
        queue: Any,
    ) -> None:
        """Translate message_changed / message_deleted into delete events.

        For each affected message-ts we emit an IngestEvent with
        ``operation="deleted"`` for the document that previously contained
        it, and re-fetch the surrounding context so the next pass rebuilds
        the window or thread cleanly.
        """
        cid = channel.get("id", "")
        for m in messages:
            subtype = m.get("subtype")
            if subtype not in {"message_changed", "message_deleted"}:
                continue
            target_ts = ""
            if subtype == "message_changed":
                inner = m.get("message") or {}
                target_ts = inner.get("ts", "")
            else:
                target_ts = (
                    (m.get("previous_message") or {}).get("ts")
                    or m.get("deleted_ts", "")
                    or ""
                )
            if not target_ts:
                continue
            doc_id = self._ts_to_doc.get((cid, target_ts))
            if not doc_id:
                # We never emitted this ts in the current process — most
                # likely the original was indexed in a previous run.  Build
                # a best-effort doc id from ts so the pipeline can still
                # delete it before the rebuild.
                doc_id = _doc_id_window(self._team_id, cid, target_ts, target_ts)
            del_ev = _build_delete_event(source_id=doc_id, team_id=self._team_id)
            SOURCE_WATCHER_EVENTS.labels(
                source_type="slack", event_type="deleted"
            ).inc()
            queue.enqueue(del_ev)
            # Drop from the index so a subsequent re-emit for the same ts
            # isn't silently considered a duplicate.
            self._ts_to_doc.pop((cid, target_ts), None)

            # If the changed message belongs to a thread, refetch the
            # thread and re-emit it as a "modified" event.  Otherwise the
            # outer loop will pick up the rebuilt window when it processes
            # the affected ts in the same poll cycle.
            inner = (
                m.get("message")
                if subtype == "message_changed"
                else m.get("previous_message")
            )
            thread_ts = (inner or {}).get("thread_ts") or m.get("thread_ts")
            if thread_ts:
                try:
                    replies_full = await asyncio.to_thread(
                        self._fetch_replies, cid, thread_ts
                    )
                except SlackApiError as exc:
                    logger.warning(
                        "Replies refetch failed for %s/%s: %s", cid, thread_ts, exc
                    )
                    continue
                if replies_full:
                    parent = replies_full[0]
                    replies = replies_full[1:]
                    rebuilt = _build_thread_event(
                        team_id=self._team_id,
                        channel=channel,
                        parent=parent,
                        replies=replies,
                        operation="modified",
                    )
                    queue.enqueue(rebuilt)
                    for sub in [parent, *replies]:
                        ts = sub.get("ts", "")
                        if ts:
                            self._ts_to_doc[(cid, ts)] = rebuilt["source_id"]
            else:
                # Non-thread edit: re-fetch a small history window centered
                # on the affected ts and rebuild a single window event.
                rebuilt = await asyncio.to_thread(
                    self._refetch_window_around, channel, target_ts
                )
                if rebuilt:
                    rebuilt["operation"] = "modified"
                    queue.enqueue(rebuilt)
                    for ts_iter in rebuilt["meta"].get("message_ts", []):
                        self._ts_to_doc[(cid, ts_iter)] = rebuilt["source_id"]

    def _refetch_window_around(
        self, channel: dict[str, Any], around_ts: str
    ) -> dict[str, Any] | None:
        """Refetch a single message and emit it as a 1-message window."""
        assert self._client is not None
        cid = channel.get("id", "")
        try:
            resp = self._client.conversations_history(
                channel=cid,
                latest=around_ts,
                oldest=around_ts,
                inclusive=True,
                limit=1,
            )
        except SlackApiError as exc:
            logger.warning("history refetch failed for %s/%s: %s", cid, around_ts, exc)
            return None
        msgs = resp.get("messages", []) or []
        if not msgs:
            return None
        return _build_window_event(
            team_id=self._team_id, channel=channel, messages=msgs
        )
