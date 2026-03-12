"""IWorkParser: text extraction from Apple iWork files (.pages, .key).

Registered for source_type='files' MIME types:
- application/x-iwork-pages
- application/x-iwork-keynote

On macOS with Pages/Keynote installed, extracts text via osascript.
Pages uses export-to-file; Keynote iterates slide text items directly.
On other platforms, returns an empty list with a debug log.
"""

from __future__ import annotations

import logging
import os
import platform
import re
import subprocess
import tempfile
import time
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
_DEFAULT_KEYNOTE_TIMEOUT = 120  # presentations can be large

_KEYNOTE_SCRIPT_TEMPLATE = """\
tell application "Keynote"
  open POSIX file "{path}"
  set theText to ""
  repeat with aSlide in slides of front document
    repeat with anItem in text items of aSlide
      set theText to theText & object text of anItem & return
    end repeat
    set theText to theText & return
  end repeat
  close front document
  return theText
end tell"""


def _pages_installed() -> bool:
    """Check whether Pages.app is available on this Mac."""
    return os.path.isdir("/Applications/Pages.app")


def _keynote_installed() -> bool:
    """Check whether Keynote.app is installed on macOS."""
    if os.path.isdir("/Applications/Keynote.app"):
        return True
    try:
        result = subprocess.run(
            [
                "mdfind",
                "kMDItemCFBundleIdentifier == com.apple.iWork.Keynote",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return bool(result.stdout.strip())
    except (subprocess.TimeoutExpired, OSError):
        return False


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

        if mime == "application/x-iwork-keynote":
            return self._parse_keynote(source_id, operation, event)

        # Pages extraction
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

    def _parse_keynote(
        self,
        source_id: str,
        operation: str,
        event: dict[str, Any],
    ) -> list[ParsedDocument]:
        """Extract text from Keynote by iterating slide text items."""
        if not _keynote_installed():
            log.warning("Keynote.app not installed — skipping %s", source_id)
            return []

        if not os.path.isfile(source_id):
            log.error("Keynote file not found: %s", source_id)
            return []

        script = _KEYNOTE_SCRIPT_TEMPLATE.format(path=source_id)
        timeout = int(
            event.get("meta", {}).get("keynote_timeout", _DEFAULT_KEYNOTE_TIMEOUT)
        )

        start = time.monotonic()
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            log.error(
                "Keynote extraction timed out after %ds for %s",
                timeout,
                source_id,
            )
            return []
        finally:
            elapsed = time.monotonic() - start
            IWORK_EXTRACTION_DURATION_SECONDS.labels(app="Keynote").observe(elapsed)

        if result.returncode != 0:
            stderr = result.stderr.strip()
            log.error(
                "Keynote extraction failed for %s: %s",
                source_id,
                stderr,
            )
            return []

        text = result.stdout.strip()
        if not text:
            log.debug("Empty presentation (no text extracted): %s", source_id)
            return []

        # Collapse runs of 3+ newlines into double-newline for chunker-friendly output
        text = re.sub(r"\n{3,}", "\n\n", text)

        props: dict[str, Any] = {
            "path": source_id,
            "name": os.path.basename(source_id),
            "ext": os.path.splitext(source_id)[1],
        }
        if "source_modified_at" in event:
            props["modified_at"] = event["source_modified_at"]
        if modified_at := event.get("meta", {}).get("modified_at"):
            props["modified_at"] = modified_at

        return [
            ParsedDocument(
                source_type="files",
                source_id=source_id,
                operation=operation,
                text=text,
                mime_type="application/x-iwork-keynote",
                node_label="File",
                node_props=props,
            )
        ]
