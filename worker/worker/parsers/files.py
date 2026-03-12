"""FileParser: text and PDF content extraction.

Registered for source_type='files'. Handles:
- text/* MIME types: reads text directly from the event
- application/pdf: extracts text via pymupdf
- Unsupported formats: returns empty list
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
from typing import Any

import pymupdf

from .base import BaseParser, ParsedDocument
from .iwork import IWORK_MIME_TYPES, IWorkParser
from .registry import register

log = logging.getLogger(__name__)

_DEFAULT_MAX_PDF_BYTES = 100 * 1024 * 1024  # 100 MiB
_DEFAULT_MAX_PDF_PAGES = 2000
_DEFAULT_MAX_TEXT_BYTES = 10 * 1024 * 1024  # 10 MiB
_DEFAULT_MAX_IMAGE_BYTES = 50 * 1024 * 1024  # 50 MiB


@register
class FileParser(BaseParser):
    """Parses file-system events into ParsedDocuments."""

    def __init__(self) -> None:
        self._max_pdf_bytes: int = _DEFAULT_MAX_PDF_BYTES
        self._max_pdf_pages: int = _DEFAULT_MAX_PDF_PAGES
        self._max_text_bytes: int = _DEFAULT_MAX_TEXT_BYTES
        self._max_image_bytes: int = _DEFAULT_MAX_IMAGE_BYTES
        self._iwork_parser = IWorkParser()

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
        elif mime.startswith("image/"):
            return self._parse_image(event, mime, source_id, operation)
        elif mime in IWORK_MIME_TYPES:
            return self._iwork_parser.parse(event)
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
        text_bytes = text.encode("utf-8")
        if len(text_bytes) > self._max_text_bytes:
            log.warning(
                "Text file %s exceeds max size (%d bytes > %d), skipping",
                source_id, len(text_bytes), self._max_text_bytes,
            )
            return []
        props = self._node_props(source_id, event)
        props["sha256"] = self._sha256(text_bytes)
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

    def _parse_image(
        self,
        event: dict[str, Any],
        mime: str,
        source_id: str,
        operation: str,
    ) -> list[ParsedDocument]:
        raw = event.get("raw_bytes")
        if isinstance(raw, str):
            if len(raw) > self._max_image_bytes:
                log.warning(
                    "Image %s exceeds max size (%d > %d bytes encoded), skipping",
                    source_id, len(raw), self._max_image_bytes,
                )
                return []
            raw = base64.b64decode(raw)
        if not raw:
            return []

        if len(raw) > self._max_image_bytes:
            log.warning(
                "Image %s exceeds max size (%d > %d bytes), skipping",
                source_id, len(raw), self._max_image_bytes,
            )
            return []

        props = self._node_props(source_id, event)
        props["sha256"] = self._sha256(raw)

        return [
            ParsedDocument(
                source_type="files",
                source_id=source_id,
                operation=operation,
                text="",
                mime_type=mime,
                node_label="File",
                node_props=props,
                image_bytes=raw,
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
            # Check encoded size before decoding — base64 inflates ~33%,
            # so encoded len < limit guarantees decoded len < limit.
            if len(raw) > self._max_pdf_bytes:
                log.warning(
                    "PDF %s exceeds max size (%d > %d bytes encoded), skipping",
                    source_id, len(raw), self._max_pdf_bytes,
                )
                return []
            raw = base64.b64decode(raw)
        if not raw:
            return []

        if len(raw) > self._max_pdf_bytes:
            log.warning(
                "PDF %s exceeds max size (%d > %d bytes), skipping",
                source_id, len(raw), self._max_pdf_bytes,
            )
            return []

        props = self._node_props(source_id, event)
        props["sha256"] = self._sha256(raw)

        try:
            doc = pymupdf.open(stream=raw, filetype="pdf")
        except (pymupdf.FileDataError, pymupdf.EmptyFileError, ValueError, RuntimeError):
            log.error("Failed to open PDF %s", source_id, exc_info=True)
            return []

        try:
            pages: list[str] = []
            for i, page in enumerate(doc):
                if i >= self._max_pdf_pages:
                    log.warning(
                        "PDF %s truncated at %d pages (limit %d)",
                        source_id, i, self._max_pdf_pages,
                    )
                    break
                pages.append(page.get_text())
        finally:
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
