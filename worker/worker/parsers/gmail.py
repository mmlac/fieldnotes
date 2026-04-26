"""Gmail email parser.

Strips HTML from email bodies, extracts structured metadata from Gmail
IngestEvents, and produces GraphHints for Person/Email/Thread relationships.
"""

from __future__ import annotations

import logging
from email.utils import parseaddr
from typing import Any

from bs4 import BeautifulSoup

from .base import BaseParser, GraphHint, ParsedDocument, canonicalize_email
from .registry import register

logger = logging.getLogger(__name__)

_MAX_HTML_BODY_SIZE = 10 * 1024 * 1024  # 10 MiB
_MAX_RECIPIENTS = 100  # Cap recipients to prevent 100k graph hints per email


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
        if thread_id:
            thread_uri = f"gmail://{account}/thread/{thread_id}"
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
                    "account": account,
                },
            )
        ]
