"""IWorkParser: stub parser for Apple iWork files (.pages, .key).

Registered for source_type='files' MIME types:
- application/x-iwork-pages
- application/x-iwork-keynote

Extraction logic (via osascript on macOS) is implemented in subsequent beads.
For now, returns an empty list on unsupported platforms with a debug log.
"""

from __future__ import annotations

import logging
import platform
from typing import Any

from .base import BaseParser, ParsedDocument

log = logging.getLogger(__name__)

# MIME types handled by this parser.
IWORK_MIME_TYPES: frozenset[str] = frozenset({
    "application/x-iwork-pages",
    "application/x-iwork-keynote",
})


class IWorkParser:
    """Parses iWork file events into ParsedDocuments.

    This is not a standalone registered parser — it is invoked by FileParser
    when it encounters an iWork MIME type.
    """

    def parse(self, event: dict[str, Any]) -> list[ParsedDocument]:
        mime = event.get("mime_type", "")
        source_id = event.get("source_id", "")

        if platform.system() != "Darwin":
            log.debug(
                "iWork extraction not supported on %s, skipping %s",
                platform.system(),
                source_id,
            )
            return []

        # macOS extraction via osascript — implemented in subsequent beads
        return []
