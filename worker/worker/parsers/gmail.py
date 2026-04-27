"""Gmail email parser.

Strips HTML from email bodies, extracts structured metadata from Gmail
IngestEvents, and produces GraphHints for Person/Email/Thread relationships.

Also surfaces attachments: each attachment becomes its own ParsedDocument
linked to the parent thread via ATTACHED_TO, with the policy decision
(``download_and_index`` vs ``metadata_only``) applied per-attachment from
the source's ``download_attachments`` knob.  Filenames are appended to the
parent text in an ``Attachments:`` section so they're picked up by the
chunker / embedder.
"""

from __future__ import annotations

import base64
import binascii
import logging
from email.utils import parseaddr
from pathlib import Path
from typing import Any, Callable

from bs4 import BeautifulSoup

from .attachments import (
    AttachmentDownloadError,
    AttachmentParseError,
    ParsedAttachment,
    build_parent_url,
    classify_attachment,
    stream_and_parse,
)
from ._safe_filename import sanitize_for_inline
from .base import BaseParser, GraphHint, ParsedDocument, canonicalize_email
from .registry import register

logger = logging.getLogger(__name__)

_MAX_HTML_BODY_SIZE = 10 * 1024 * 1024  # 10 MiB
_MAX_RECIPIENTS = 100  # Cap recipients to prevent 100k graph hints per email


# Test seam: ``stream_and_parse`` is monkeypatched in unit tests; the
# default runtime hook calls the real helper.  Module-level so tests can
# patch ``worker.parsers.gmail._stream_and_parse``.
_stream_and_parse = stream_and_parse


def _gmail_attachment_fetcher(
    *,
    account: str,
    message_id: str,
    attachment_id: str,
    client_secrets_path: str | None,
) -> Callable[[], bytes]:
    """Return a zero-arg closure that fetches an attachment's bytes.

    The closure builds a Gmail API client lazily (so tests that never
    classify attachments as indexable don't pay the auth cost) and calls
    ``users.messages.attachments.get`` followed by URL-safe base64
    decode.  The ``client_secrets_path`` flows from the source's
    ``configure()`` via the IngestEvent meta so the parser can rebuild
    credentials per message.
    """

    def fetch() -> bytes:
        if not client_secrets_path:
            raise RuntimeError(
                "Gmail attachment fetch requested but client_secrets_path "
                "missing from event meta — source must populate it"
            )
        # Imports are local: callers that never run download paths should
        # not pay googleapiclient/gmail_auth import cost at parser import.
        from googleapiclient.discovery import build

        from worker.sources.gmail_auth import get_credentials

        creds = get_credentials(Path(client_secrets_path), account=account)
        service = build("gmail", "v1", credentials=creds)
        result = (
            service.users()
            .messages()
            .attachments()
            .get(userId="me", messageId=message_id, id=attachment_id)
            .execute()
        )
        data = result.get("data", "")
        if not data:
            raise RuntimeError(
                f"Gmail attachments.get returned no data for "
                f"message {message_id} attachment {attachment_id}"
            )
        try:
            return base64.urlsafe_b64decode(data + "==")
        except (binascii.Error, ValueError) as exc:
            raise RuntimeError(
                f"malformed base64 in Gmail attachment {attachment_id}: {exc}"
            ) from exc

    return fetch


_MIME_LABELS: dict[str, str] = {
    "application/pdf": "PDF",
    "application/zip": "ZIP",
    "application/x-zip-compressed": "ZIP",
    "application/msword": "DOC",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "DOCX",
    "application/vnd.ms-excel": "XLS",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "XLSX",
    "application/vnd.ms-powerpoint": "PPT",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "PPTX",
    "image/png": "PNG",
    "image/jpeg": "JPEG",
    "image/gif": "GIF",
    "image/webp": "WEBP",
    "image/heic": "HEIC",
    "image/heif": "HEIF",
    "image/tiff": "TIFF",
    "image/bmp": "BMP",
    "text/plain": "TXT",
    "text/markdown": "MD",
    "text/csv": "CSV",
    "application/json": "JSON",
    "application/yaml": "YAML",
    "application/x-yaml": "YAML",
}


def _mime_label(mime: str) -> str:
    """Render a short uppercase label for an attachment MIME type."""
    if mime in _MIME_LABELS:
        return _MIME_LABELS[mime]
    if "/" in mime:
        return mime.split("/", 1)[1].upper()
    return mime.upper() or "FILE"


def _human_size(size_bytes: int) -> str:
    """Render bytes as a short human-readable string (KB / MB / GB)."""
    if size_bytes <= 0:
        return "0 B"
    for unit, divisor in (("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)):
        if size_bytes >= divisor:
            return f"{size_bytes / divisor:.1f} {unit}"
    return f"{size_bytes} B"


def _render_attachment_bullet(att: dict[str, Any]) -> str:
    safe_name = sanitize_for_inline(str(att.get("filename", "")))
    return (
        f"- {safe_name} "
        f"({_mime_label(att.get('mime_type', ''))}, "
        f"{_human_size(int(att.get('size_bytes') or 0))})"
    )


def _strip_html(html: str) -> str:
    """Strip HTML tags and return plain text.

    Uses BeautifulSoup to handle real-world email HTML (unclosed tags,
    nested tables, style blocks, etc.).
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove script and style elements entirely
    for element in soup(["script", "style"]):
        element.decompose()

    text = soup.get_text(separator="\n")

    # Collapse excessive whitespace while preserving paragraph breaks
    lines = (line.strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def _parse_email_address(raw: str) -> str:
    """Extract bare email address from 'Display Name <addr>' format."""
    _, addr = parseaddr(raw)
    return addr.strip()


@register
class GmailParser(BaseParser):
    """Parses Gmail IngestEvents into ParsedDocuments with graph hints."""

    @property
    def source_type(self) -> str:
        return "gmail"

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
                )
            ]

        account: str = meta.get("account", "")
        message_id: str = meta.get("message_id", "")
        thread_id: str = meta.get("thread_id", "")
        subject: str = meta.get("subject", "")
        date: str = meta.get("date", "")
        sender_raw: str = meta.get("sender_email", "")
        recipients_raw: list[str] = meta.get("recipients", [])

        # All edges leaving this Email carry the account so the graph can
        # answer "show me only personal-account SENT_BY edges for Bob".
        edge_props: dict[str, Any] = {"account": account} if account else {}

        # --- Strip HTML from body ---------------------------------------------
        body: str = event.get("text", "")
        mime = event.get("mime_type", "")
        if body and ("html" in mime or body.lstrip().startswith("<")):
            if len(body) > _MAX_HTML_BODY_SIZE:
                logger.warning(
                    "Email %s HTML body too large (%d bytes > %d), skipping HTML parse",
                    source_id,
                    len(body),
                    _MAX_HTML_BODY_SIZE,
                )
                body = body[:_MAX_HTML_BODY_SIZE]
            body = _strip_html(body)

        # --- Node properties ---------------------------------------------------
        node_props: dict[str, Any] = {
            "message_id": message_id,
            "subject": subject,
            "date": date,
            "account": account,
        }

        # --- GraphHints: Person ↔ Email ↔ Thread relationships -----------------
        graph_hints: list[GraphHint] = []
        sender_addr = _parse_email_address(sender_raw)

        # SENT: Person → Email (sender sent this email).  Person merges
        # on email so cross-account Persons (alice@... seen by both work
        # and personal Gmail accounts) collapse into a single node.
        if sender_addr:
            norm_sender = canonicalize_email(sender_addr)
            graph_hints.append(
                GraphHint(
                    subject_id=f"person:{norm_sender}",
                    subject_label="Person",
                    predicate="SENT",
                    object_id=source_id,
                    object_label="Email",
                    subject_props={"email": norm_sender},
                    subject_merge_key="email",
                    edge_props=edge_props,
                    confidence=1.0,
                )
            )

        # TO: Email → Person (email was sent to each recipient)
        if len(recipients_raw) > _MAX_RECIPIENTS:
            logger.warning(
                "Email %s has %d recipients, truncating to %d",
                source_id,
                len(recipients_raw),
                _MAX_RECIPIENTS,
            )
            recipients_raw = recipients_raw[:_MAX_RECIPIENTS]
        for recip_raw in recipients_raw:
            recip_addr = _parse_email_address(recip_raw)
            if not recip_addr:
                continue
            norm_recip = canonicalize_email(recip_addr)
            graph_hints.append(
                GraphHint(
                    subject_id=source_id,
                    subject_label="Email",
                    predicate="TO",
                    object_id=f"person:{norm_recip}",
                    object_label="Person",
                    object_props={"email": norm_recip},
                    object_merge_key="email",
                    edge_props=edge_props,
                    confidence=1.0,
                )
            )

        # PART_OF: Email → Thread.  Thread node is account-namespaced
        # because Gmail thread IDs are scoped to a mailbox — the same
        # thread_id in two accounts is two unrelated conversations.
        thread_uri = f"gmail://{account}/thread/{thread_id}" if thread_id else ""
        if thread_id:
            graph_hints.append(
                GraphHint(
                    subject_id=source_id,
                    subject_label="Email",
                    predicate="PART_OF",
                    object_id=thread_uri,
                    object_label="Thread",
                    object_props={
                        "thread_id": thread_id,
                        "subject": subject,
                        "account": account,
                    },
                    object_merge_key="source_id",
                    edge_props=edge_props,
                    confidence=1.0,
                )
            )

        # --- Attachments -------------------------------------------------------
        attachments_meta: list[dict[str, Any]] = list(meta.get("attachments") or [])
        # Dedupe within a single message (same attachment_id appearing twice
        # in nested parts).  Cross-message dedup happens via source_id MERGE
        # in the writer — Documents with the same source_id collapse to one
        # Neo4j node regardless of how many parse() calls emit them.
        seen_att_ids: set[str] = set()
        deduped: list[dict[str, Any]] = []
        for att in attachments_meta:
            aid = att.get("attachment_id")
            if not aid or aid in seen_att_ids:
                continue
            seen_att_ids.add(aid)
            deduped.append(att)

        attachment_docs: list[ParsedDocument] = []
        if deduped:
            # Augment parent body with an Attachments: section BEFORE chunking
            # so filenames are retrievable via the parent thread.
            bullets = [_render_attachment_bullet(att) for att in deduped]
            body = (
                (body + "\n\n" if body else "") + "Attachments:\n" + "\n".join(bullets)
            )

            attachment_docs, counters = self._build_attachment_documents(
                attachments=deduped,
                account=account,
                message_id=message_id,
                thread_id=thread_id,
                thread_uri=thread_uri,
                sender_addr=sender_addr,
                edge_props=edge_props,
                meta=meta,
            )
            # Three counters distinguish intent (pre-fetch) from outcome
            # (post-fetch).  Queries that need "documents with parsed
            # attachment text" must use ``attachments_count_indexed``; the
            # other two are diagnostic.
            node_props["attachments_count_intended"] = counters["intended"]
            node_props["attachments_count_indexed"] = counters["indexed"]
            node_props["attachments_count_metadata_only"] = counters["metadata_only"]
            # Deprecated alias of attachments_count_intended; retained for one
            # release so existing Cypher queries keep working.  Remove after
            # downstream consumers migrate to the explicit counter.
            node_props["has_attachments"] = counters["intended"]

        parent_doc = ParsedDocument(
            source_type=self.source_type,
            source_id=source_id,
            operation=operation,
            text=body,
            node_label="Email",
            node_props=node_props,
            graph_hints=graph_hints,
            source_metadata={
                "source_type": "email",
                "thread_id": thread_id,
                "account": account,
            },
        )
        return [parent_doc, *attachment_docs]

    # ------------------------------------------------------------------
    # Attachments
    # ------------------------------------------------------------------

    def _build_attachment_documents(
        self,
        *,
        attachments: list[dict[str, Any]],
        account: str,
        message_id: str,
        thread_id: str,
        thread_uri: str,
        sender_addr: str,
        edge_props: dict[str, Any],
        meta: dict[str, Any],
    ) -> tuple[list[ParsedDocument], dict[str, int]]:
        """Emit one ParsedDocument per attachment (indexed or metadata-only).

        Each Document carries an ATTACHED_TO edge to the parent thread and
        a SENT_BY edge to the sender Person (so attachment provenance is
        queryable without traversing the email node).  Indexable
        attachments fetch + parse on the fly via :func:`stream_and_parse`;
        any download or parse error is logged and degrades cleanly to a
        metadata-only Document — the parent message is still ingested.

        Returns a tuple of (documents, counters) where counters carries
        ``intended`` (total attachments seen), ``indexed`` (successful
        fetch+parse), and ``metadata_only`` (fallback or non-indexable).
        """
        download_attachments = bool(meta.get("download_attachments", False))
        indexable = list(meta.get("attachment_indexable_mimetypes") or [])
        max_size_mb = int(meta.get("attachment_max_size_mb", 25))
        pdf_max_pages = int(meta.get("attachment_pdf_max_pages", 1000))
        pdf_per_page_chars = int(meta.get("attachment_pdf_per_page_chars", 1_000_000))
        pdf_timeout_seconds = int(meta.get("attachment_pdf_timeout_seconds", 60))
        client_secrets_path = meta.get("client_secrets_path")

        parent_url = build_parent_url("gmail", thread_id=thread_id) if thread_id else ""

        norm_sender = canonicalize_email(sender_addr) if sender_addr else ""
        docs: list[ParsedDocument] = []
        indexed_count = 0

        for att in attachments:
            filename = att.get("filename", "")
            mime = att.get("mime_type", "application/octet-stream")
            size_bytes = int(att.get("size_bytes") or 0)
            attachment_id = att.get("attachment_id", "")

            decision = (
                classify_attachment(
                    mime=mime,
                    size_bytes=size_bytes,
                    indexable=indexable,
                    max_size_mb=max_size_mb,
                )
                if download_attachments
                else "metadata_only"
            )

            att_source_id = (
                f"gmail://{account}/thread/{thread_id}/attachment/{attachment_id}"
            )

            parsed: ParsedAttachment | None = None
            if decision == "download_and_index":
                try:
                    parsed = _stream_and_parse(
                        fetch=_gmail_attachment_fetcher(
                            account=account,
                            message_id=message_id,
                            attachment_id=attachment_id,
                            client_secrets_path=client_secrets_path,
                        ),
                        filename=filename,
                        mime=mime,
                        source_id=att_source_id,
                        pdf_max_pages=pdf_max_pages,
                        pdf_per_page_chars=pdf_per_page_chars,
                        pdf_timeout_seconds=pdf_timeout_seconds,
                    )
                except (AttachmentDownloadError, AttachmentParseError) as exc:
                    logger.warning(
                        "Attachment %s on message %s fell back to metadata-only: %s",
                        filename,
                        message_id,
                        exc,
                    )
                    parsed = None  # fall through to metadata-only render

            if parsed is not None:
                indexed_count += 1
                text = parsed.text or _metadata_only_description(
                    filename, mime, size_bytes
                )
            else:
                text = _metadata_only_description(filename, mime, size_bytes)

            att_props: dict[str, Any] = {
                "filename": filename,
                "mime_type": mime,
                "size_bytes": size_bytes,
                "attachment_id": attachment_id,
                "account": account,
                "thread_id": thread_id,
                "message_id": message_id,
                "indexed": parsed is not None,
            }
            if parent_url:
                att_props["parent_url"] = parent_url

            att_hints: list[GraphHint] = []
            if thread_uri:
                att_hints.append(
                    GraphHint(
                        subject_id=att_source_id,
                        subject_label="Attachment",
                        predicate="ATTACHED_TO",
                        object_id=thread_uri,
                        object_label="Thread",
                        object_props={
                            "thread_id": thread_id,
                            "account": account,
                        },
                        object_merge_key="source_id",
                        edge_props=edge_props,
                        confidence=1.0,
                    )
                )
            if norm_sender:
                # SENT_BY follows the parent's author so attachment
                # provenance is queryable without traversing the Email.
                att_hints.append(
                    GraphHint(
                        subject_id=att_source_id,
                        subject_label="Attachment",
                        predicate="SENT_BY",
                        object_id=f"person:{norm_sender}",
                        object_label="Person",
                        object_props={"email": norm_sender},
                        object_merge_key="email",
                        edge_props=edge_props,
                        confidence=1.0,
                    )
                )
            if parsed is not None:
                att_hints.extend(parsed.extracted_entities)

            docs.append(
                ParsedDocument(
                    source_type=self.source_type,
                    source_id=att_source_id,
                    operation="created",
                    text=text,
                    mime_type=mime,
                    node_label="Attachment",
                    node_props=att_props,
                    graph_hints=att_hints,
                    source_metadata={
                        "source_type": "email_attachment",
                        "thread_id": thread_id,
                        "account": account,
                        "filename": filename,
                        "indexed": parsed is not None,
                    },
                )
            )

        counters = {
            "intended": len(attachments),
            "indexed": indexed_count,
            "metadata_only": len(attachments) - indexed_count,
        }
        return docs, counters


def _metadata_only_description(filename: str, mime: str, size_bytes: int) -> str:
    """Render the human-readable fallback text for non-indexable attachments."""
    label = _mime_label(mime)
    size = _human_size(size_bytes)
    return f"{filename} ({label}, {size}) — content not indexed"
