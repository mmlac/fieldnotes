"""Shared attachment helpers.

Two responsibilities:

1. **Policy**: :func:`classify_attachment` decides whether a given attachment
   (by MIME and size) should be downloaded and indexed or kept metadata-only.
   Pure decision logic, no I/O.

2. **Stream-and-forget download + parse**: :func:`stream_and_parse` takes a
   caller-supplied closure that returns the attachment's bytes, hands them to
   the appropriate parser (PDF / image / text), and returns the parsed result.
   The bytes are not written to disk and are released as soon as the parser
   returns.

A small :func:`build_parent_url` helper is also provided for source adapters
that need a clickable link back to where the attachment lives (Gmail thread,
Slack message permalink, Calendar event).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import pymupdf

from .base import GraphHint

log = logging.getLogger(__name__)

AttachmentDecision = Literal["download_and_index", "metadata_only"]


# Defaults for PDF-bomb defenses applied by ``_parse_pdf_bytes``.
# A 25 MB input PDF can expand to gigabytes during text extraction
# (zip-bomb-style nested streams, millions of empty pages, single pages
# with megabytes of text), so we cap pages, per-page text length, and
# wallclock parse time. Per-source config (``attachment_pdf_max_pages``,
# ``attachment_pdf_per_page_chars``, ``attachment_pdf_timeout_seconds``)
# overrides these.
DEFAULT_PDF_MAX_PAGES: int = 1000
DEFAULT_PDF_PER_PAGE_CHARS: int = 1_000_000
DEFAULT_PDF_TIMEOUT_SECONDS: int = 60


# Default allowlist of MIME types we know how to index.  Extending this
# without also teaching a parser to handle the new type just wastes
# bandwidth, so keep it narrow.  Per-source config may override.
DEFAULT_INDEXABLE_MIMETYPES: list[str] = [
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
    "image/heic",
    "image/heif",
    "image/tiff",
    "image/bmp",
    "text/plain",
    "text/markdown",
    "text/csv",
    "application/json",
    "application/yaml",
    "application/x-yaml",
]


# --- Errors -----------------------------------------------------------------


class AttachmentDownloadError(Exception):
    """Raised when the caller-supplied fetch closure fails.

    The ``source_id`` of the parent (when known) is included in ``args`` so
    log scraping can correlate failures back to a specific Gmail message,
    Slack message, or Calendar event.
    """

    def __init__(self, message: str, *, source_id: str | None = None) -> None:
        super().__init__(message)
        self.source_id = source_id


class AttachmentParseError(Exception):
    """Raised when bytes were fetched but the parser refused them.

    Carries ``source_id`` for the same logging reasons as
    :class:`AttachmentDownloadError`.  Callers should catch this and emit a
    metadata-only Document rather than crashing the ingest event.
    """

    def __init__(self, message: str, *, source_id: str | None = None) -> None:
        super().__init__(message)
        self.source_id = source_id


# --- Result -----------------------------------------------------------------


@dataclass
class ParsedAttachment:
    """The output of :func:`stream_and_parse`.

    Deliberately omits any reference to the original bytes — once
    :func:`stream_and_parse` returns, the bytes must be eligible for GC.
    """

    text: str = ""
    description: str = ""
    extracted_entities: list[GraphHint] = field(default_factory=list)
    # OCR text from the vision pipeline for images; ``None`` for non-images.
    ocr_text: str | None = None


# --- Policy -----------------------------------------------------------------


def classify_attachment(
    mime: str,
    size_bytes: int,
    indexable: list[str],
    max_size_mb: int,
) -> AttachmentDecision:
    """Decide whether to download-and-index an attachment or keep metadata only.

    Args:
        mime: MIME type as reported by the source (e.g. ``application/pdf``).
        size_bytes: Attachment size in bytes.
        indexable: Allowlist of MIME types the caller is willing to index
            (typically ``cfg.attachment_indexable_mimetypes``).
        max_size_mb: Inclusive upper bound on attachment size in megabytes
            (typically ``cfg.attachment_max_size_mb``).  Sizes equal to the
            bound count as in-range.

    Returns:
        ``"download_and_index"`` when the MIME is allowlisted *and* the size
        is within bound; ``"metadata_only"`` otherwise.
    """
    if mime in indexable and size_bytes <= max_size_mb * 1024 * 1024:
        return "download_and_index"
    return "metadata_only"


# --- Stream-and-forget download + parse -------------------------------------


# Type alias for the optional vision extractor.  Takes (bytes, mime) and
# returns an object with ``description``, ``visible_text``, and ``entities``
# attributes.  Wired up to ``worker.pipeline.vision.extract_image`` (or a
# bound variant of ``extract_image_from_registry``) by callers; tests inject
# fakes.  Kept structural rather than referencing ``VisionResult`` directly
# so this module avoids importing the vision pipeline at module load.
VisionExtractor = Callable[[bytes, str], Any]


def stream_and_parse(
    fetch: Callable[[], bytes],
    filename: str,
    mime: str,
    *,
    vision_extractor: VisionExtractor | None = None,
    source_id: str | None = None,
    pdf_max_pages: int = DEFAULT_PDF_MAX_PAGES,
    pdf_per_page_chars: int = DEFAULT_PDF_PER_PAGE_CHARS,
    pdf_timeout_seconds: int = DEFAULT_PDF_TIMEOUT_SECONDS,
) -> ParsedAttachment:
    """Fetch attachment bytes, parse them, and discard.

    The caller supplies ``fetch`` as a zero-arg closure that returns the raw
    bytes (typically wrapping a Gmail / Slack / Drive API call).  The bytes
    are routed to the appropriate in-process parser and released as soon as
    parsing returns; nothing is cached on disk.

    Args:
        fetch: Zero-arg closure returning the attachment bytes, or raising on
            network / auth / not-found failure.
        filename: Original filename (only used in error messages and logs).
        mime: MIME type the source reported for the attachment.
        vision_extractor: Optional callable ``(bytes, mime) -> VisionResult``
            used for ``image/*`` attachments.  Required to actually parse
            images; if ``None``, image attachments raise
            :class:`AttachmentParseError`.
        source_id: Parent source identifier (Gmail message id, Slack ts,
            Calendar event id) — propagated onto raised errors for log
            correlation.
        pdf_max_pages: PDFs with more pages raise
            :class:`AttachmentParseError` before any text extraction.
        pdf_per_page_chars: Per-page text is truncated to this many
            characters; the rest of the page is silently dropped.
        pdf_timeout_seconds: Wallclock cap on the entire PDF parse;
            exceeded parses raise :class:`AttachmentParseError`.

    Returns:
        A :class:`ParsedAttachment` carrying the parsed text plus optional
        description / entities / OCR text.

    Raises:
        AttachmentDownloadError: ``fetch`` raised.
        AttachmentParseError: bytes returned but parser refused them, or
            ``mime`` is not one we know how to index.
    """
    try:
        data = fetch()
    except Exception as exc:
        log.warning(
            "Attachment download failed: filename=%s mime=%s source_id=%s: %s",
            filename,
            mime,
            source_id,
            exc,
        )
        raise AttachmentDownloadError(
            f"fetch failed for {filename!r} ({mime}): {exc}",
            source_id=source_id,
        ) from exc

    if mime == "application/pdf":
        return _parse_pdf_bytes(
            data,
            filename=filename,
            source_id=source_id,
            max_pages=pdf_max_pages,
            per_page_chars=pdf_per_page_chars,
            timeout_seconds=pdf_timeout_seconds,
        )
    if mime.startswith("image/"):
        return _parse_image_bytes(
            data,
            mime=mime,
            filename=filename,
            source_id=source_id,
            vision_extractor=vision_extractor,
        )
    if mime.startswith("text/"):
        return _parse_text_bytes(data)

    raise AttachmentParseError(
        f"unsupported MIME for {filename!r}: {mime}",
        source_id=source_id,
    )


def _parse_pdf_bytes(
    data: bytes,
    *,
    filename: str,
    source_id: str | None,
    max_pages: int = DEFAULT_PDF_MAX_PAGES,
    per_page_chars: int = DEFAULT_PDF_PER_PAGE_CHARS,
    timeout_seconds: int = DEFAULT_PDF_TIMEOUT_SECONDS,
) -> ParsedAttachment:
    """Extract text from PDF bytes via pymupdf, in-memory, with bomb guards.

    pymupdf opens the PDF from a bytes stream — no temp file is written.

    Three defenses against malicious / pathological inputs:

    * ``max_pages`` — refuse documents with more pages than this. ``page_count``
      is cheap to read and lets us bail before enumerating the pages.
    * ``per_page_chars`` — truncate each page's extracted text to this length.
      Caps memory regardless of how many text streams the page nests.
    * ``timeout_seconds`` — wallclock cap on the entire parse. The parse runs
      on a daemon worker thread; if it doesn't finish in time the caller gets
      :class:`AttachmentParseError` while the orphan thread eventually drops
      its references and releases memory.
    """
    holder: dict[str, Any] = {}

    def _worker() -> None:
        try:
            try:
                doc = pymupdf.open(stream=data, filetype="pdf")
            except (
                pymupdf.FileDataError,
                pymupdf.EmptyFileError,
                ValueError,
                RuntimeError,
            ) as exc:
                raise AttachmentParseError(
                    f"PDF parse failed for {filename!r}: {exc}",
                    source_id=source_id,
                ) from exc

            try:
                page_count = doc.page_count
                if page_count > max_pages:
                    raise AttachmentParseError(
                        f"PDF {filename!r} has too many pages: "
                        f"{page_count} > {max_pages}",
                        source_id=source_id,
                    )
                pages: list[str] = []
                for page in doc:
                    text = page.get_text()
                    if len(text) > per_page_chars:
                        text = text[:per_page_chars]
                    pages.append(text)
                holder["value"] = ParsedAttachment(text="\n".join(pages))
            finally:
                doc.close()
        except BaseException as exc:
            holder["error"] = exc

    t = threading.Thread(
        target=_worker,
        name=f"pdf-parse-{filename}",
        daemon=True,
    )
    t.start()
    t.join(timeout=timeout_seconds)
    if t.is_alive():
        raise AttachmentParseError(
            f"PDF parse for {filename!r} exceeded timeout {timeout_seconds}s",
            source_id=source_id,
        )
    if "error" in holder:
        raise holder["error"]
    return holder["value"]


def _parse_image_bytes(
    data: bytes,
    *,
    mime: str,
    filename: str,
    source_id: str | None,
    vision_extractor: VisionExtractor | None,
) -> ParsedAttachment:
    """Run image bytes through the vision pipeline in memory.

    The vision pipeline (``worker.pipeline.vision.extract_image``) consumes
    bytes directly and never touches disk, so no temp-file boundary exists.
    If the caller did not provide a ``vision_extractor`` we cannot do
    anything useful with the bytes, so we surface :class:`AttachmentParseError`
    and let the caller fall through to metadata-only.
    """
    if vision_extractor is None:
        raise AttachmentParseError(
            f"no vision extractor available for image {filename!r} ({mime})",
            source_id=source_id,
        )

    try:
        result = vision_extractor(data, mime)
    except Exception as exc:
        raise AttachmentParseError(
            f"vision extract failed for {filename!r} ({mime}): {exc}",
            source_id=source_id,
        ) from exc

    description = getattr(result, "description", "") or ""
    visible_text = getattr(result, "visible_text", "") or ""
    raw_entities = getattr(result, "entities", []) or []

    # Convert vision entity dicts into GraphHints with the attachment as
    # subject so downstream graph-write picks them up.  ``source_id`` may be
    # ``None`` for synthetic test fetches; in that case skip hint emission.
    hints: list[GraphHint] = []
    if source_id is not None:
        for ent in raw_entities:
            if not isinstance(ent, dict):
                continue
            name = ent.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            hints.append(
                GraphHint(
                    subject_id=source_id,
                    subject_label="Attachment",
                    predicate="MENTIONS",
                    object_id=name,
                    object_label=ent.get("type", "Concept")
                    if isinstance(ent.get("type"), str)
                    else "Concept",
                    object_props={"name": name},
                    object_merge_key="name",
                    confidence=0.8,
                )
            )

    parts = [p for p in (description, visible_text) if p]
    text = "\n\n".join(parts)

    return ParsedAttachment(
        text=text,
        description=description,
        extracted_entities=hints,
        ocr_text=visible_text or None,
    )


def _parse_text_bytes(data: bytes) -> ParsedAttachment:
    return ParsedAttachment(text=data.decode("utf-8", errors="replace"))


# --- Parent-URL builder -----------------------------------------------------


def build_parent_url(source_type: str, **kwargs: Any) -> str:
    """Build a clickable back-link to the attachment's parent record.

    Per source:

    * ``gmail``    — requires ``thread_id``; returns the Gmail thread URL.
    * ``slack``    — requires ``team_domain``, ``channel_id``, ``ts``;
      returns the message permalink (with the ``.`` stripped from ``ts``).
    * ``calendar`` — requires ``html_link``; returns it verbatim (Calendar
      already provides a canonical URL for events).

    Raises:
        ValueError: ``source_type`` is unknown or required kwargs are missing.
    """
    if source_type == "gmail":
        thread_id = kwargs.get("thread_id")
        if not thread_id:
            raise ValueError("gmail parent_url requires thread_id")
        return f"https://mail.google.com/mail/?ui=2&view=cv&th={thread_id}"
    if source_type == "slack":
        team_domain = kwargs.get("team_domain")
        channel_id = kwargs.get("channel_id")
        ts = kwargs.get("ts")
        if not (team_domain and channel_id and ts):
            raise ValueError("slack parent_url requires team_domain, channel_id, ts")
        ts_compact = str(ts).replace(".", "")
        return f"https://{team_domain}.slack.com/archives/{channel_id}/p{ts_compact}"
    if source_type == "calendar":
        html_link = kwargs.get("html_link")
        if not html_link:
            raise ValueError("calendar parent_url requires html_link")
        return str(html_link)
    raise ValueError(f"unknown source_type: {source_type!r}")
