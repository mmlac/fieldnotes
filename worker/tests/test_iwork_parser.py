"""Tests for iWork MIME type registration and parser routing."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from worker.parsers.files import FileParser
from worker.parsers.iwork import IWorkParser
from worker.sources._handler import guess_mime


class TestIWorkMimeTypes:
    def test_pages_mime_type(self) -> None:
        assert guess_mime("document.pages") == "application/x-iwork-pages"

    def test_key_mime_type(self) -> None:
        assert guess_mime("presentation.key") == "application/x-iwork-keynote"

    def test_pages_uppercase_extension(self) -> None:
        assert guess_mime("document.PAGES") == "application/x-iwork-pages"

    def test_key_uppercase_extension(self) -> None:
        assert guess_mime("presentation.KEY") == "application/x-iwork-keynote"


class TestIWorkParserRouting:
    @pytest.fixture()
    def parser(self) -> FileParser:
        return FileParser()

    def test_pages_routes_to_iwork_parser(self, parser: FileParser) -> None:
        event = {
            "mime_type": "application/x-iwork-pages",
            "source_id": "/docs/report.pages",
            "operation": "created",
        }
        # On non-Darwin, returns empty list
        docs = parser.parse(event)
        assert docs == []

    def test_key_routes_to_iwork_parser(self, parser: FileParser) -> None:
        event = {
            "mime_type": "application/x-iwork-keynote",
            "source_id": "/slides/deck.key",
            "operation": "created",
        }
        docs = parser.parse(event)
        assert docs == []


class TestIWorkParserLinux:
    """Verify that unsupported platforms return empty list with debug log."""

    def test_returns_empty_on_linux(self) -> None:
        parser = IWorkParser()
        event = {
            "mime_type": "application/x-iwork-pages",
            "source_id": "/docs/report.pages",
            "operation": "created",
        }
        with patch("worker.parsers.iwork.platform.system", return_value="Linux"):
            docs = parser.parse(event)
        assert docs == []

    def test_debug_log_on_unsupported_platform(self) -> None:
        parser = IWorkParser()
        event = {
            "mime_type": "application/x-iwork-keynote",
            "source_id": "/slides/deck.key",
            "operation": "modified",
        }
        with (
            patch("worker.parsers.iwork.platform.system", return_value="Linux"),
            patch("worker.parsers.iwork.log") as mock_log,
        ):
            parser.parse(event)
        mock_log.debug.assert_called_once()
        assert "Linux" in mock_log.debug.call_args[0][1]
        assert "/slides/deck.key" in mock_log.debug.call_args[0][2]

    def test_returns_empty_on_darwin_stub(self) -> None:
        """Even on Darwin, the stub returns empty (extraction in later beads)."""
        parser = IWorkParser()
        event = {
            "mime_type": "application/x-iwork-pages",
            "source_id": "/docs/report.pages",
            "operation": "created",
        }
        with patch("worker.parsers.iwork.platform.system", return_value="Darwin"):
            docs = parser.parse(event)
        assert docs == []
