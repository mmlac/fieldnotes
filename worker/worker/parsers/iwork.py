"""IWorkParser: text extraction from Apple iWork files (.pages, .key).

Registered for source_type='files' MIME types:
- application/x-iwork-pages
- application/x-iwork-keynote

On macOS with Pages/Keynote installed, extracts text via osascript.
On other platforms, returns an empty list with a debug log.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import tempfile
from typing import Any

from worker.metrics import IWORK_EXTRACTION_DURATION_SECONDS, observe_duration

from .base import BaseParser, ParsedDocument

log = logging.getLogger(__name__)

# MIME types handled by this parser.
IWORK_MIME_TYPES: frozenset[str] = frozenset({
    "application/x-iwork-pages",
    "application/x-iwork-keynote",
})

# Map MIME type to the iWork application name used in osascript.
_MIME_TO_APP: dict[str, str] = {
    "application/x-iwork-pages": "Pages",
    "application/x-iwork-keynote": "Keynote",
}

_DEFAULT_TIMEOUT_SECONDS = 60


def _pages_installed() -> bool:
    """Check whether Pages.app is available on this Mac."""
    return os.path.isdir("/Applications/Pages.app")


class IWorkParser:
    """Parses iWork file events into ParsedDocuments.

    This is not a standalone registered parser — it is invoked by FileParser
    when it encounters an iWork MIME type.
    """

    def __init__(self, timeout: int = _DEFAULT_TIMEOUT_SECONDS) -> None:
        self._timeout = timeout

    def parse(self, event: dict[str, Any]) -> list[ParsedDocument]:
        mime = event.get("mime_type", "")
        source_id = event.get("source_id", "")
        operation = event.get("operation", "modified")

        if platform.system() != "Darwin":
            log.debug(
                "iWork extraction not supported on %s, skipping %s",
                platform.system(),
                source_id,
            )
            return []

        if not _pages_installed():
            log.warning("Pages.app not installed — skipping %s", source_id)
            return []

        app_name = _MIME_TO_APP.get(mime, "Pages")

        try:
            with observe_duration(IWORK_EXTRACTION_DURATION_SECONDS, app=app_name):
                text = self._extract_text(source_id, app_name)
        except FileNotFoundError:
            log.error("File not found: %s", source_id)
            return []
        except subprocess.TimeoutExpired:
            log.error("osascript timed out after %ds for %s", self._timeout, source_id)
            return []
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr or ""
            if "password" in stderr.lower():
                log.warning("Password-protected file, skipping %s", source_id)
            else:
                log.error(
                    "osascript failed for %s: %s", source_id, stderr,
                )
            return []
        except OSError as exc:
            log.error("OS error extracting %s: %s", source_id, exc)
            return []

        if not text.strip():
            return []

        title = os.path.splitext(os.path.basename(source_id))[0]

        return [
            ParsedDocument(
                source_type="files",
                source_id=source_id,
                operation=operation,
                text=text,
                mime_type=mime,
                node_label="File",
                node_props={
                    "path": source_id,
                    "name": os.path.basename(source_id),
                    "ext": os.path.splitext(source_id)[1],
                    "title": title,
                },
            )
        ]

    def _extract_text(self, path: str, app_name: str) -> str:
        """Run osascript to export the iWork document as plain text."""
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

        fd, tmp_path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
        try:
            script = (
                f'tell application "{app_name}"\n'
                f'  open POSIX file "{path}"\n'
                f'  export front document to POSIX file "{tmp_path}" as unformatted text\n'
                f"  close front document\n"
                f"end tell"
            )
            subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=True,
            )
            with open(tmp_path, encoding="utf-8", errors="replace") as f:
                return f.read()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
