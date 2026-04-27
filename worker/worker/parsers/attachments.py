"""Stream-and-forget attachment parsing + parent-URL builders.

``stream_and_parse`` fetches an attachment's bytes via a caller-supplied
closure, hands them to the appropriate inline parser (PDF / image-vision /
text), returns a :class:`ParsedAttachment`, and drops the bytes. There is
no on-disk cache: bytes live only in the function's stack frame, and are
released as soon as the parse completes.

``build_parent_url`` produces the documented per-source back-link so the
attachment Document carries a clickable pointer to where the file lives.

Temp-file boundary
------------------
- ``application/pdf`` is parsed via ``pymupdf.open(stream=...)`` which reads
  from a memory buffer and never writes to disk.
- ``image/*`` is parsed via :func:`worker.pipeline.vision.extract_image_from_registry`,
  which sends base64-encoded bytes to a multimodal model and never writes
  to disk.
- ``text/*`` is decoded directly with ``bytes.decode``.

If a future image parser needs to wrap in a ``NamedTemporaryFile``, that
is acceptable as long as ``delete=True`` and the file is gone before this
function returns. A persistent cache is NOT acceptable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pymupdf

from .base import GraphHint

if TYPE_CHECKING:
    from worker.models.resolver import ModelRegistry


# ---------------------------------------------------------------------------
# Result + error types
# ---------------------------------------------------------------------------


@dataclass
class ParsedAttachment:
    """Output of :func:`stream_and_parse` — text + extracted metadata, no bytes."""

    text: str = ""
    description: str = ""
    extracted_entities: list[GraphHint] = field(default_factory=list)
    ocr_text: str | None = None


class AttachmentDownloadError(Exception):
    """Raised when the fetch closure fails (network, auth, missing).

    Carries the parent ``source_id`` so logs are searchable. Callers should
    fall through to a metadata-only Document; the ingest pipeline must NOT
    crash on this error.
    """

    def __init__(self, message: str, *, source_id: str = "") -> None:
        super().__init__(message)
        self.source_id = source_id


class AttachmentParseError(Exception):
    """Raised when bytes arrived but the parser could not produce text.

    Carries the parent ``source_id`` so logs are searchable. Callers should
    fall through to a metadata-only Document.
    """

    def __init__(self, message: str, *, source_id: str = "") -> None:
        super().__init__(message)
        self.source_id = source_id


# ---------------------------------------------------------------------------
# Parent-URL builders
# ---------------------------------------------------------------------------


def build_parent_url(source_type: str, **kwargs: str) -> str:
    """Build the canonical back-link URL for an attachment's parent record.

    ``gmail``    requires ``thread_id``.
    ``slack``    requires ``team_domain``, ``channel_id``, ``ts``.
    ``calendar`` requires ``html_link`` and returns it verbatim.

    Raises ``ValueError`` for unknown ``source_type`` or missing kwargs.
    """
    if source_type == "gmail":
        thread_id = kwargs.get("thread_id")
        if not thread_id:
            raise ValueError("gmail parent_url requires 'thread_id'")
        return f"https://mail.google.com/mail/?ui=2&view=cv&th={thread_id}"

    if source_type == "slack":
        team_domain = kwargs.get("team_domain")
        channel_id = kwargs.get("channel_id")
        ts = kwargs.get("ts")
        missing = [
            n
            for n, v in (
                ("team_domain", team_domain),
                ("channel_id", channel_id),
                ("ts", ts),
            )
            if not v
        ]
        if missing:
            raise ValueError(f"slack parent_url requires {missing!r}")
        return (
            f"https://{team_domain}.slack.com/archives/"
            f"{channel_id}/p{ts.replace('.', '')}"
        )

    if source_type == "calendar":
        html_link = kwargs.get("html_link")
        if not html_link:
            raise ValueError("calendar parent_url requires 'html_link'")
        return html_link

    raise ValueError(f"unknown source_type for parent_url: {source_type!r}")


# ---------------------------------------------------------------------------
# Stream-and-forget downloader
# ---------------------------------------------------------------------------


def stream_and_parse(
    fetch: Callable[[], bytes],
    filename: str,
    mime: str,
    *,
    model_registry: ModelRegistry | None = None,
    source_id: str = "",
) -> ParsedAttachment:
    """Fetch bytes, parse them, drop them.

    Parameters
    ----------
    fetch:
        Source-specific closure that returns the attachment bytes (or raises).
    filename:
        Original attachment filename — used in error messages only.
    mime:
        MIME type used for parser routing.
    model_registry:
        Required for ``image/*`` MIMEs (resolves the vision model). For PDF
        and text MIMEs, may be ``None``.
    source_id:
        Parent record ``source_id`` — stamped into errors and onto graph
        hints so downstream logs / queries can find the parent record.

    Returns
    -------
    ParsedAttachment
        ``text``, ``description``, ``extracted_entities`` (graph hints), and
        optional ``ocr_text``. Never carries the raw bytes.

    Raises
    ------
    AttachmentDownloadError
        ``fetch()`` raised before bytes arrived.
    AttachmentParseError
        Bytes arrived but the parser could not produce text, or the MIME is
        not in the supported routing table.
    """
    try:
        raw = fetch()
    except Exception as exc:  # fetch closures raise anything they want
        raise AttachmentDownloadError(
            f"failed to fetch attachment {filename!r}: {exc}",
            source_id=source_id,
        ) from exc

    try:
        if mime == "application/pdf":
            return _parse_pdf_bytes(raw, source_id=source_id)
        if mime.startswith("image/"):
            return _parse_image_bytes(
                raw,
                mime=mime,
                model_registry=model_registry,
                source_id=source_id,
            )
        if mime.startswith("text/"):
            return ParsedAttachment(text=raw.decode("utf-8", errors="replace"))
        raise AttachmentParseError(
            f"unsupported MIME {mime!r} for attachment {filename!r}",
            source_id=source_id,
        )
    finally:
        # Drop the local reference so CPython refcount-collects immediately.
        del raw


def _parse_pdf_bytes(raw: bytes, *, source_id: str) -> ParsedAttachment:
    """Extract text from PDF bytes using pymupdf in-memory (no temp file)."""
    try:
        doc = pymupdf.open(stream=raw, filetype="pdf")
    except (
        pymupdf.FileDataError,
        pymupdf.EmptyFileError,
        ValueError,
        RuntimeError,
    ) as exc:
        raise AttachmentParseError(
            f"PDF parse failed: {exc}",
            source_id=source_id,
        ) from exc

    try:
        pages = [page.get_text() for page in doc]
    finally:
        doc.close()

    return ParsedAttachment(text="\n".join(pages))


_VISION_ENTITY_CONFIDENCE = 0.80


def _parse_image_bytes(
    raw: bytes,
    *,
    mime: str,
    model_registry: ModelRegistry | None,
    source_id: str,
) -> ParsedAttachment:
    """Route image bytes to the vision pipeline (no temp file)."""
    if model_registry is None:
        raise AttachmentParseError(
            f"model_registry is required for image MIME {mime!r}",
            source_id=source_id,
        )

    # Imported lazily so non-image callers don't pay the import cost.
    from worker.pipeline.vision import extract_image_from_registry

    try:
        result = extract_image_from_registry(raw, model_registry, mime_type=mime)
    except Exception as exc:
        raise AttachmentParseError(
            f"vision pipeline failed: {exc}",
            source_id=source_id,
        ) from exc

    parts: list[str] = []
    if result.description:
        parts.append(result.description)
    if result.visible_text:
        parts.append(result.visible_text)

    hints: list[GraphHint] = []
    for ent in result.entities:
        name = ent.get("name", "")
        if not name:
            continue
        ent_type = ent.get("type", "Concept") or "Concept"
        hints.append(
            GraphHint(
                subject_id=source_id,
                subject_label="File",
                predicate="MENTIONS",
                object_id=f"{ent_type.lower()}:{name}",
                object_label=ent_type,
                object_props={"name": name},
                confidence=_VISION_ENTITY_CONFIDENCE,
            )
        )

    return ParsedAttachment(
        text="\n\n".join(parts),
        description=result.description,
        ocr_text=result.visible_text or None,
        extracted_entities=hints,
    )
