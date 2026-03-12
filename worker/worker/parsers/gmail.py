"""Gmail email parser.

Strips HTML from email bodies, extracts structured metadata from Gmail
IngestEvents, and produces GraphHints for Person/Email/Thread relationships.
"""

from __future__ import annotations

import logging
from email.utils import parseaddr
from typing import Any

from bs4 import BeautifulSoup

from .base import BaseParser, GraphHint, ParsedDocument
from .registry import register

logger = logging.getLogger(__name__)

_MAX_HTML_BODY_SIZE = 10 * 1024 * 1024  # 10 MiB


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

        message_id: str = meta.get("message_id", "")
        thread_id: str = meta.get("thread_id", "")
        subject: str = meta.get("subject", "")
        date: str = meta.get("date", "")
        sender_raw: str = meta.get("sender_email", "")
        recipients_raw: list[str] = meta.get("recipients", [])

        # --- Strip HTML from body ---------------------------------------------
        body: str = event.get("text", "")
        mime = event.get("mime_type", "")
        if body and ("html" in mime or body.lstrip().startswith("<")):
            if len(body) > _MAX_HTML_BODY_SIZE:
                logger.warning(
                    "Email %s HTML body too large (%d bytes > %d), skipping HTML parse",
                    source_id, len(body), _MAX_HTML_BODY_SIZE,
                )
                body = body[:_MAX_HTML_BODY_SIZE]
            body = _strip_html(body)

        # --- Node properties ---------------------------------------------------
        node_props: dict[str, Any] = {
            "message_id": message_id,
            "subject": subject,
            "date": date,
        }

        # --- GraphHints: Person ↔ Email ↔ Thread relationships -----------------
        graph_hints: list[GraphHint] = []
        sender_addr = _parse_email_address(sender_raw)

        # SENT: Person → Email (sender sent this email)
        if sender_addr:
            norm_sender = sender_addr.strip().lower()
            graph_hints.append(
                GraphHint(
                    subject_id=f"person:{norm_sender}",
                    subject_label="Person",
                    predicate="SENT",
                    object_id=source_id,
                    object_label="Email",
                    subject_props={"email": norm_sender},
                    subject_merge_key="email",
                    confidence=1.0,
                )
            )

        # TO: Email → Person (email was sent to each recipient)
        for recip_raw in recipients_raw:
            recip_addr = _parse_email_address(recip_raw)
            if not recip_addr:
                continue
            norm_recip = recip_addr.strip().lower()
            graph_hints.append(
                GraphHint(
                    subject_id=source_id,
                    subject_label="Email",
                    predicate="TO",
                    object_id=f"person:{norm_recip}",
                    object_label="Person",
                    object_props={"email": norm_recip},
                    object_merge_key="email",
                    confidence=1.0,
                )
            )

        # PART_OF: Email → Thread
        if thread_id:
            graph_hints.append(
                GraphHint(
                    subject_id=source_id,
                    subject_label="Email",
                    predicate="PART_OF",
                    object_id=f"gmail-thread:{thread_id}",
                    object_label="Thread",
                    object_props={"thread_id": thread_id, "subject": subject},
                    object_merge_key="thread_id",
                    confidence=1.0,
                )
            )

        return [
            ParsedDocument(
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
                },
            )
        ]
