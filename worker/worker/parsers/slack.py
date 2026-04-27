"""Slack thread/window parser.

Converts a Slack IngestEvent (thread or burst window) into a
:class:`ParsedDocument` with rendered text and high-confidence GraphHints
that link the document to its Channel and to Person nodes shared with
Gmail/Calendar via canonical email addresses.

The Slack source emits two kinds of events: ``kind='thread'`` (parent +
ordered replies) and ``kind='window'`` (a burst of un-threaded messages
in a single channel).  Both are rendered as plain text for the chunker
in the form::

    [HH:MM UTC] <Display Name> (@handle): message text

Threads keep the parent on the first line and indent each reply by two
spaces so the chunker preserves visible structure.

Cross-source linking: an author whose Slack profile carries an email
canonicalised to the same address as a Gmail correspondent yields a
single Person node — Gmail's ``person:{email}`` and Slack's
``person:{email}`` MERGE on the ``email`` key.  Authors without an
email known to Slack fall back to a slack-user-keyed Person so the
graph is never lossy.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Callable

from .attachments import (
    AttachmentDownloadError,
    AttachmentParseError,
    VisionExtractor,
    build_parent_url,
    classify_attachment,
    stream_and_parse,
)
from ._safe_filename import sanitize_for_inline
from .base import (
    _EMAIL_RE,
    BaseParser,
    GraphHint,
    ParsedDocument,
    canonicalize_email,
)
from .registry import register

logger = logging.getLogger(__name__)

NODE_LABEL = "SlackMessage"
ATTACHMENT_NODE_LABEL = "Attachment"

# Closure type for the Slack attachment downloader.  Takes a Slack
# ``url_private_download`` URL and returns the file bytes (or raises).
# ``configure_slack_parser`` wires the production version (httpx + bot
# token); tests inject fakes.
SlackAttachmentFetcher = Callable[[str], bytes]


# Module-level defaults used by every ``SlackParser`` instance unless
# the caller overrides them per-instance.  We keep these out of class
# scope so plain functions don't trigger Python's descriptor binding
# when read via an instance.
_default_fetcher: SlackAttachmentFetcher | None = None
_default_vision_extractor: VisionExtractor | None = None

# Slack mention syntax inside message text.
_USER_MENTION_RE = re.compile(r"<@([UW][A-Z0-9]+)>")
_CHANNEL_MENTION_RE = re.compile(r"<#(C[A-Z0-9]+)(?:\|[^>]*)?>")

# Subtypes that carry no meaningful body text.  ``bot_message`` is
# retained when it has body text; the empty-detection below catches the
# silent variants automatically.
_SYSTEM_SUBTYPES = frozenset({"channel_join", "channel_leave"})


def _conversation_type(meta: dict[str, Any]) -> str:
    """Map the Slack channel-shape booleans to a single string label."""
    if meta.get("is_im"):
        return "im"
    if meta.get("is_mpim"):
        return "mpim"
    if meta.get("is_private"):
        return "private"
    return "public"


def _format_clock(ts: str) -> str:
    """Format a Slack ``ts`` ('1234567890.123') as ``HH:MM UTC``."""
    if not ts:
        return ""
    try:
        sec = float(ts)
    except (TypeError, ValueError):
        return ""
    if sec <= 0:
        return ""
    return datetime.fromtimestamp(sec, tz=timezone.utc).strftime("%H:%M UTC")


def _user_profile(users_info: dict[str, Any], user_id: str) -> dict[str, Any]:
    """Return the profile dict for *user_id* (empty if unknown)."""
    info = users_info.get(user_id) or {}
    if not isinstance(info, dict):
        return {}
    profile = info.get("profile") or {}
    return profile if isinstance(profile, dict) else {}


def _user_email(users_info: dict[str, Any], user_id: str) -> str:
    profile = _user_profile(users_info, user_id)
    info = users_info.get(user_id) or {}
    raw = profile.get("email") or (info.get("email") if isinstance(info, dict) else "")
    return raw or ""


def _user_real_name(users_info: dict[str, Any], user_id: str) -> str:
    profile = _user_profile(users_info, user_id)
    info = users_info.get(user_id) or {}
    return (
        profile.get("real_name")
        or (info.get("real_name") if isinstance(info, dict) else "")
        or profile.get("display_name")
        or ""
    )


def _user_handle(users_info: dict[str, Any], user_id: str) -> str:
    profile = _user_profile(users_info, user_id)
    info = users_info.get(user_id) or {}
    return (
        (info.get("name") if isinstance(info, dict) else "")
        or profile.get("display_name")
        or ""
    )


def _format_message(
    msg: dict[str, Any], users_info: dict[str, Any], *, indent: bool
) -> str:
    """Render a single message line in the parser's canonical format."""
    ts = msg.get("ts", "")
    user = msg.get("user") or msg.get("bot_id") or ""
    text = (msg.get("text") or "").rstrip()
    clock = _format_clock(ts)

    if user and (user.startswith("U") or user.startswith("W")):
        display = _user_real_name(users_info, user) or f"@{user}"
        handle = _user_handle(users_info, user) or user
        author = f"{display} (@{handle})"
    elif user:
        # Bot id or other opaque identifier — render verbatim.
        author = user
    else:
        author = "unknown"

    line = f"[{clock}] {author}: {text}" if clock else f"{author}: {text}"
    return f"  {line}" if indent else line


def _is_system_message(msg: dict[str, Any]) -> bool:
    """Return True if the message has no body and a system subtype."""
    if (msg.get("text") or "").strip():
        return False
    subtype = msg.get("subtype")
    if subtype in _SYSTEM_SUBTYPES:
        return True
    if subtype == "bot_message":
        return True
    return False


@register
class SlackParser(BaseParser):
    """Parses Slack thread/window IngestEvents into ParsedDocuments.

    Attachment-fetch wiring is read from module-level defaults at
    construction time so the registry's fresh-instance-per-call pattern
    still picks up the runtime fetcher set by ``configure_slack_parser``
    at startup.  Tests can shadow ``_fetcher`` / ``_vision_extractor``
    on the instance.
    """

    def __init__(self) -> None:
        super().__init__()
        self._fetcher: SlackAttachmentFetcher | None = _default_fetcher
        self._vision_extractor: VisionExtractor | None = _default_vision_extractor

    @property
    def source_type(self) -> str:
        return "slack"

    def parse(self, event: dict[str, Any]) -> list[ParsedDocument]:
        source_id: str = event["source_id"]
        operation: str = event.get("operation", "created")
        meta: dict[str, Any] = event.get("meta", {})

        if operation == "deleted":
            return [
                ParsedDocument(
                    source_type=self.source_type,
                    source_id=source_id,
                    operation="deleted",
                    text="",
                    node_label=NODE_LABEL,
                )
            ]

        team_id: str = meta.get("team_id", "")
        team_domain: str = meta.get("team_domain", "")
        channel_id: str = meta.get("channel_id", "")
        channel_name: str = meta.get("channel_name", "") or channel_id
        kind: str = meta.get("kind", "window")
        users_info: dict[str, Any] = meta.get("users_info") or {}
        messages: list[dict[str, Any]] = list(meta.get("messages") or [])
        attachments: list[dict[str, Any]] = list(meta.get("attachments") or [])

        conversation_type = _conversation_type(meta)
        first_ts = (
            meta.get("first_ts")
            or meta.get("parent_ts")
            or (messages[0].get("ts") if messages else "")
        )
        last_ts = meta.get("last_ts") or (messages[-1].get("ts") if messages else "")
        message_count = (
            len(messages) if messages else len(meta.get("message_ts", []) or [])
        )
        has_thread = kind == "thread"

        text = _render_text(messages, users_info, kind=kind, attachments=attachments)

        # Empty/system path: produce a metadata-only chunk so the document
        # still exists in the graph.  We bypass the rendered (empty) text
        # in favour of a one-line synthetic body so the chunker has
        # something to embed.
        meaningful = [m for m in messages if not _is_system_message(m)]
        if messages and not meaningful:
            text = (
                f"[Slack {conversation_type} #{channel_name}: "
                f"{len(messages)} system message(s)]"
            )

        node_props: dict[str, Any] = {
            "team_id": team_id,
            "channel_id": channel_id,
            "channel_name": channel_name,
            "conversation_type": conversation_type,
            "first_ts": first_ts,
            "last_ts": last_ts,
            "message_count": message_count,
            "has_thread": has_thread,
        }
        if attachments:
            node_props["has_attachments"] = len(attachments)

        graph_hints = _build_graph_hints(
            source_id=source_id,
            team_id=team_id,
            channel_id=channel_id,
            channel_name=channel_name,
            conversation_type=conversation_type,
            messages=messages,
            users_info=users_info,
        )

        parent_doc = ParsedDocument(
            source_type=self.source_type,
            source_id=source_id,
            operation=operation,
            text=text,
            mime_type="text/plain",
            node_label=NODE_LABEL,
            node_props=node_props,
            graph_hints=graph_hints,
            source_metadata={
                "source_type": "slack",
                "team_id": team_id,
                "channel_id": channel_id,
                "conversation_type": conversation_type,
                "first_ts": first_ts,
                "last_ts": last_ts,
                "message_count": message_count,
                "has_thread": has_thread,
            },
        )

        # Configurable attachment policy: source forwards its config so
        # the parser doesn't need to reach back into Config / settings.
        download_attachments = bool(meta.get("download_attachments", False))
        indexable: list[str] = list(meta.get("attachment_indexable_mimetypes") or [])
        max_size_mb = int(meta.get("attachment_max_size_mb", 25))

        attachment_docs = _build_attachment_documents(
            parent_source_id=source_id,
            team_id=team_id,
            team_domain=team_domain,
            channel_id=channel_id,
            channel_name=channel_name,
            conversation_type=conversation_type,
            attachments=attachments,
            users_info=users_info,
            download_attachments=download_attachments,
            indexable=indexable,
            max_size_mb=max_size_mb,
            fetcher=self._fetcher,
            vision_extractor=self._vision_extractor,
        )

        return [parent_doc, *attachment_docs]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _render_text(
    messages: list[dict[str, Any]],
    users_info: dict[str, Any],
    *,
    kind: str,
    attachments: list[dict[str, Any]] | None = None,
) -> str:
    if not messages:
        return ""

    # Bucket attachments by their parent message ts so each '[file] …'
    # marker appears immediately after the line that referenced it.
    by_ts: dict[str, list[dict[str, Any]]] = {}
    for a in attachments or []:
        ts = a.get("ts", "")
        if not ts:
            continue
        by_ts.setdefault(ts, []).append(a)

    def _file_marker(att: dict[str, Any], *, indent: bool) -> str:
        safe_name = sanitize_for_inline(str(att.get("name", "")))
        line = (
            f"[file] {safe_name} "
            f"({att.get('mimetype', '')}, {_human_size(int(att.get('size') or 0))})"
        )
        return f"  {line}" if indent else line

    lines: list[str] = []
    if kind == "thread":
        parent = messages[0]
        replies = messages[1:]
        lines.append(_format_message(parent, users_info, indent=False))
        for att in by_ts.get(parent.get("ts", ""), []):
            lines.append(_file_marker(att, indent=False))
        for r in replies:
            lines.append(_format_message(r, users_info, indent=True))
            for att in by_ts.get(r.get("ts", ""), []):
                lines.append(_file_marker(att, indent=True))
    else:
        for m in messages:
            lines.append(_format_message(m, users_info, indent=False))
            for att in by_ts.get(m.get("ts", ""), []):
                lines.append(_file_marker(att, indent=False))
    return "\n".join(lines)


def _human_size(n: int) -> str:
    """Format *n* bytes as a short human-readable string (e.g. ``1.2 MB``)."""
    if n < 1024:
        return f"{n} B"
    units = ["KB", "MB", "GB", "TB"]
    val = float(n)
    for unit in units:
        val /= 1024
        if val < 1024:
            return f"{val:.1f} {unit}"
    return f"{val:.1f} PB"


def configure_slack_parser(
    *,
    fetcher: SlackAttachmentFetcher | None = None,
    vision_extractor: VisionExtractor | None = None,
) -> None:
    """Wire production-side dependencies for new :class:`SlackParser`s.

    Called by the worker startup once the Slack bot token is loaded —
    binds a fetcher closure that authenticates ``url_private_download``
    requests and (optionally) the vision extractor for image OCR.  New
    parser instances pick the defaults up at construction time.

    Tests may shadow ``_fetcher`` / ``_vision_extractor`` on a parser
    instance directly without affecting these defaults.
    """
    global _default_fetcher, _default_vision_extractor
    _default_fetcher = fetcher
    _default_vision_extractor = vision_extractor


def _attachment_doc_id(team_id: str, channel_id: str, ts: str, file_id: str) -> str:
    """``slack://{team_id}/{channel_id}/{ts}/file/{file_id}``."""
    return f"slack://{team_id}/{channel_id}/{ts}/file/{file_id}"


def _person_hint_for_user(
    *,
    source_id: str,
    predicate: str,
    user_id: str,
    team_id: str,
    users_info: dict[str, Any],
) -> GraphHint:
    """Build a GraphHint for a Slack-resolved Person.

    If the user has a known email we MERGE on ``email`` so the Person
    aligns with Gmail/Calendar.  Otherwise we MERGE on ``source_id``
    keyed by ``slack-user:{team}/{user_id}``.
    """
    email_raw = _user_email(users_info, user_id)
    real_name = _user_real_name(users_info, user_id)
    handle = _user_handle(users_info, user_id)

    if email_raw:
        norm = canonicalize_email(email_raw)
        # Carry slack identity on the email-keyed Person too, so a later
        # reconcile pass can SAME_AS-link this node with any earlier
        # slack-user-keyed Person emitted before the email was known.
        props: dict[str, Any] = {
            "email": norm,
            "slack_user_id": user_id,
            "team_id": team_id,
        }
        if real_name:
            props["name"] = real_name
        if handle:
            props["slack_handle"] = handle
        return GraphHint(
            subject_id=source_id,
            subject_label=NODE_LABEL,
            predicate=predicate,
            object_id=f"person:{norm}",
            object_label="Person",
            object_props=props,
            object_merge_key="email",
            confidence=1.0,
        )

    props = {"slack_user_id": user_id, "team_id": team_id}
    if real_name:
        props["name"] = real_name
    if handle:
        props["slack_handle"] = handle
    return GraphHint(
        subject_id=source_id,
        subject_label=NODE_LABEL,
        predicate=predicate,
        object_id=f"slack-user:{team_id}/{user_id}",
        object_label="Person",
        object_props=props,
        object_merge_key="source_id",
        confidence=1.0,
    )


def _build_graph_hints(
    *,
    source_id: str,
    team_id: str,
    channel_id: str,
    channel_name: str,
    conversation_type: str,
    messages: list[dict[str, Any]],
    users_info: dict[str, Any],
) -> list[GraphHint]:
    hints: list[GraphHint] = []

    # ---- Channel node + IN_CHANNEL edge -----------------------------------
    channel_props: dict[str, Any] = {
        "name": channel_name,
        "team_id": team_id,
        "channel_id": channel_id,
        "type": conversation_type,
    }
    hints.append(
        GraphHint(
            subject_id=source_id,
            subject_label=NODE_LABEL,
            predicate="IN_CHANNEL",
            object_id=f"slack-channel:{team_id}/{channel_id}",
            object_label="Channel",
            object_props=channel_props,
            confidence=1.0,
        )
    )

    # ---- Authors: Person + SENT_BY ---------------------------------------
    author_ids: list[str] = []
    seen_authors: set[str] = set()
    for m in messages:
        uid = m.get("user") or ""
        if not uid or uid in seen_authors:
            continue
        seen_authors.add(uid)
        author_ids.append(uid)

    author_emails: set[str] = set()
    for uid in author_ids:
        hint = _person_hint_for_user(
            source_id=source_id,
            predicate="SENT_BY",
            user_id=uid,
            team_id=team_id,
            users_info=users_info,
        )
        hints.append(hint)
        if hint.object_merge_key == "email":
            author_emails.add(hint.object_props["email"])

    # ---- Mentions: <@U…>, <#C…>, and inline emails ------------------------
    mentioned_users: list[str] = []
    seen_mu: set[str] = set()
    mentioned_channels: list[str] = []
    seen_mc: set[str] = set()
    mentioned_emails: list[str] = []
    seen_me: set[str] = set()

    for m in messages:
        body = m.get("text") or ""
        for um in _USER_MENTION_RE.finditer(body):
            uid = um.group(1)
            if uid in seen_authors:
                # Don't double-link an author as a mention of themselves.
                continue
            if uid not in seen_mu:
                seen_mu.add(uid)
                mentioned_users.append(uid)
        for cm in _CHANNEL_MENTION_RE.finditer(body):
            cid = cm.group(1)
            if cid == channel_id or cid in seen_mc:
                continue
            seen_mc.add(cid)
            mentioned_channels.append(cid)
        for em in _EMAIL_RE.finditer(body):
            norm = canonicalize_email(em.group(0))
            if not norm or norm in seen_me:
                continue
            seen_me.add(norm)
            mentioned_emails.append(norm)

    for uid in mentioned_users:
        hints.append(
            _person_hint_for_user(
                source_id=source_id,
                predicate="MENTIONS",
                user_id=uid,
                team_id=team_id,
                users_info=users_info,
            )
        )

    for cid in mentioned_channels:
        hints.append(
            GraphHint(
                subject_id=source_id,
                subject_label=NODE_LABEL,
                predicate="MENTIONS",
                object_id=f"slack-channel:{team_id}/{cid}",
                object_label="Channel",
                object_props={"channel_id": cid, "team_id": team_id},
                confidence=1.0,
            )
        )

    for email in mentioned_emails:
        if email in author_emails:
            continue
        hints.append(
            GraphHint(
                subject_id=source_id,
                subject_label=NODE_LABEL,
                predicate="MENTIONS",
                object_id=f"person:{email}",
                object_label="Person",
                object_props={"email": email},
                object_merge_key="email",
                confidence=1.0,
            )
        )

    return hints


# ---------------------------------------------------------------------------
# Attachment Document construction
# ---------------------------------------------------------------------------


def _build_attachment_documents(
    *,
    parent_source_id: str,
    team_id: str,
    team_domain: str,
    channel_id: str,
    channel_name: str,
    conversation_type: str,
    attachments: list[dict[str, Any]],
    users_info: dict[str, Any],
    download_attachments: bool,
    indexable: list[str],
    max_size_mb: int,
    fetcher: SlackAttachmentFetcher | None,
    vision_extractor: VisionExtractor | None,
) -> list[ParsedDocument]:
    """Build one ParsedDocument per Slack file attachment.

    The result is downstream of the parent message Document and shares
    its source_id via the ATTACHED_TO edge.  Each attachment is either:

    * **download_and_index** — bytes fetched via *fetcher*, parsed by
      :func:`stream_and_parse`, full text stored on the Document.
    * **metadata_only** — empty body, file metadata on node_props only.

    A failed fetch falls back to metadata_only and logs a warning so a
    transient error does not crash the parent ingest event.
    """
    docs: list[ParsedDocument] = []
    for att in attachments:
        file_id = att.get("id") or ""
        if not file_id:
            continue
        ts = att.get("ts", "")
        att_source_id = _attachment_doc_id(team_id, channel_id, ts, file_id)
        mime = att.get("mimetype") or "application/octet-stream"
        size = int(att.get("size") or 0)

        decision = (
            classify_attachment(
                mime=mime,
                size_bytes=size,
                indexable=indexable,
                max_size_mb=max_size_mb,
            )
            if download_attachments
            else "metadata_only"
        )

        text = ""
        ocr_text: str | None = None
        description = ""
        extra_hints: list[GraphHint] = []

        if decision == "download_and_index":
            url = att.get("url_private_download") or ""
            if not url or fetcher is None:
                logger.warning(
                    "Slack attachment %s lacks url_private_download or fetcher; "
                    "downgrading to metadata_only",
                    att_source_id,
                )
                decision = "metadata_only"
            else:
                try:
                    parsed = stream_and_parse(
                        fetch=lambda u=url: fetcher(u),
                        filename=att.get("name") or file_id,
                        mime=mime,
                        vision_extractor=vision_extractor,
                        source_id=att_source_id,
                    )
                except (AttachmentDownloadError, AttachmentParseError) as exc:
                    logger.warning(
                        "Attachment %s fetch/parse failed (%s); falling back to "
                        "metadata_only",
                        att_source_id,
                        exc,
                    )
                    decision = "metadata_only"
                else:
                    text = parsed.text
                    ocr_text = parsed.ocr_text
                    description = parsed.description
                    extra_hints = list(parsed.extracted_entities)

        node_props: dict[str, Any] = {
            "team_id": team_id,
            "channel_id": channel_id,
            "channel_name": channel_name,
            "conversation_type": conversation_type,
            "parent_ts": ts,
            "file_id": file_id,
            "name": att.get("name") or "",
            "mimetype": mime,
            "filetype": att.get("filetype") or "",
            "size": size,
            "indexed": decision == "download_and_index",
        }
        try:
            parent_url = build_parent_url(
                "slack",
                team_domain=team_domain or "slack",
                channel_id=channel_id,
                ts=ts,
            )
            node_props["parent_url"] = parent_url
        except ValueError:
            # Missing team_domain/ts/channel_id is a soft error — the
            # node still exists, just without a clickable back-link.
            pass
        if description:
            node_props["description"] = description
        if ocr_text:
            node_props["ocr_text"] = ocr_text

        graph_hints: list[GraphHint] = [
            # ATTACHED_TO: links the attachment to its parent message
            # Document.  ``object_merge_key='source_id'`` is the default
            # but we set it explicitly so the writer doesn't have to
            # guess against an external label.
            GraphHint(
                subject_id=att_source_id,
                subject_label=ATTACHMENT_NODE_LABEL,
                predicate="ATTACHED_TO",
                object_id=parent_source_id,
                object_label=NODE_LABEL,
                confidence=1.0,
            ),
            # IN_CHANNEL: same Channel node Gmail/Calendar consumers see
            # via the parent message; redundant but lets queries that
            # walk Attachment → Channel skip the Message hop.
            GraphHint(
                subject_id=att_source_id,
                subject_label=ATTACHMENT_NODE_LABEL,
                predicate="IN_CHANNEL",
                object_id=f"slack-channel:{team_id}/{channel_id}",
                object_label="Channel",
                object_props={
                    "name": channel_name,
                    "team_id": team_id,
                    "channel_id": channel_id,
                    "type": conversation_type,
                },
                confidence=1.0,
            ),
        ]

        # SENT_BY: the file uploader (file.user, falling back to
        # message.user).  Bot uploaders without an email round-trip via
        # the slack-user fallback Person.
        uploader = att.get("user") or ""
        if uploader:
            uploader_hint = _person_hint_for_user(
                source_id=att_source_id,
                predicate="SENT_BY",
                user_id=uploader,
                team_id=team_id,
                users_info=users_info,
            )
            uploader_hint.subject_label = ATTACHMENT_NODE_LABEL
            graph_hints.append(uploader_hint)

        graph_hints.extend(extra_hints)

        docs.append(
            ParsedDocument(
                source_type="slack",
                source_id=att_source_id,
                operation="created",
                text=text,
                mime_type=mime,
                node_label=ATTACHMENT_NODE_LABEL,
                node_props=node_props,
                graph_hints=graph_hints,
                source_metadata={
                    "source_type": "slack",
                    "team_id": team_id,
                    "channel_id": channel_id,
                    "parent_ts": ts,
                    "file_id": file_id,
                    "mimetype": mime,
                    "size": size,
                    "indexed": decision == "download_and_index",
                },
            )
        )
    return docs
