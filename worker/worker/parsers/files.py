"""FileParser: text and PDF content extraction.

Registered for source_type='files'. Handles:
- text/* MIME types: reads text directly from the event
- application/pdf: extracts text via pymupdf
- Unsupported formats: returns empty list
"""

from __future__ import annotations

import base64
import hashlib
import os
from typing import Any

import pymupdf

from .base import BaseParser, ParsedDocument
from .registry import register


@register
class FileParser(BaseParser):
    """Parses file-system events into ParsedDocuments."""

    @property
    def source_type(self) -> str:
        return "files"

    def parse(self, event: dict[str, Any]) -> list[ParsedDocument]:
        mime = event.get("mime_type", "")
        source_id = event.get("source_id", "")
        operation = event.get("operation", "modified")

        if mime.startswith("text/"):
            return self._parse_text(event, mime, source_id, operation)
        elif mime == "application/pdf":
            return self._parse_pdf(event, source_id, operation)
        else:
            return []

    def _node_props(self, source_id: str, event: dict[str, Any]) -> dict[str, Any]:
        meta = event.get("meta", {})
        props: dict[str, Any] = {
            "path": source_id,
            "name": os.path.basename(source_id),
            "ext": os.path.splitext(source_id)[1],
        }
        if "source_modified_at" in event:
            props["modified_at"] = event["source_modified_at"]
        if modified_at := meta.get("modified_at"):
            props["modified_at"] = modified_at
        return props

    def _sha256(self, data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def _parse_text(
        self,
        event: dict[str, Any],
        mime: str,
        source_id: str,
        operation: str,
    ) -> list[ParsedDocument]:
        text = event.get("text", "")
        props = self._node_props(source_id, event)
        props["sha256"] = self._sha256(text.encode("utf-8"))
        return [
            ParsedDocument(
                source_type="files",
                source_id=source_id,
                operation=operation,
                text=text,
                mime_type=mime,
                node_label="File",
                node_props=props,
            )
        ]

    def _parse_pdf(
        self,
        event: dict[str, Any],
        source_id: str,
        operation: str,
    ) -> list[ParsedDocument]:
        raw = event.get("raw_bytes")
        if isinstance(raw, str):
            raw = base64.b64decode(raw)
        if not raw:
            return []

        props = self._node_props(source_id, event)
        props["sha256"] = self._sha256(raw)

        doc = pymupdf.open(stream=raw, filetype="pdf")
        pages = []
        for page in doc:
            pages.append(page.get_text())
        doc.close()

        text = "\n".join(pages)

        return [
            ParsedDocument(
                source_type="files",
                source_id=source_id,
                operation=operation,
                text=text,
                mime_type="application/pdf",
                node_label="File",
                node_props=props,
            )
        ]
