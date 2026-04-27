"""Shared attachment-policy helper.

Pure decision logic shared by the Gmail, Calendar and Slack source
adapters: given an attachment's MIME type, byte size, and the
per-source attachment knobs from config, decide whether we should
download-and-index the bytes or stop at metadata-only.

This module is intentionally I/O-free — no network, no disk.  Source
adapters call :func:`classify_attachment` once per attachment after
they've discovered its MIME and size; the source is then responsible
for actually downloading (or not) and feeding the bytes into the
parser pipeline.
"""

from __future__ import annotations

from typing import Literal

AttachmentDecision = Literal["download_and_index", "metadata_only"]


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
