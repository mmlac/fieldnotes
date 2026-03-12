"""Tests for iWork MIME type registration, parser routing, and text extraction."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, mock_open, patch

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


class TestIWorkParserDarwin:
    """Tests for macOS text extraction via osascript."""

    @pytest.fixture()
    def parser(self) -> IWorkParser:
        return IWorkParser(timeout=30)

    @pytest.fixture()
    def pages_event(self) -> dict:
        return {
            "mime_type": "application/x-iwork-pages",
            "source_id": "/docs/report.pages",
            "operation": "created",
        }

    def _darwin_and_installed(self):
        """Context manager patches: Darwin platform + Pages installed."""
        return (
            patch("worker.parsers.iwork.platform.system", return_value="Darwin"),
            patch("worker.parsers.iwork._pages_installed", return_value=True),
        )

    def test_happy_path_pages(self, parser: IWorkParser, pages_event: dict) -> None:
        p_sys, p_inst = self._darwin_and_installed()
        with (
            p_sys,
            p_inst,
            patch("worker.parsers.iwork.os.path.isfile", return_value=True),
            patch("worker.parsers.iwork.tempfile.mkstemp", return_value=(99, "/tmp/out.txt")),
            patch("worker.parsers.iwork.os.close"),
            patch("worker.parsers.iwork.subprocess.run"),
            patch("builtins.open", mock_open(read_data="Hello from Pages")),
            patch("worker.parsers.iwork.os.unlink"),
        ):
            docs = parser.parse(pages_event)

        assert len(docs) == 1
        doc = docs[0]
        assert doc.source_type == "files"
        assert doc.source_id == "/docs/report.pages"
        assert doc.text == "Hello from Pages"
        assert doc.operation == "created"
        assert doc.mime_type == "application/x-iwork-pages"
        assert doc.node_props["title"] == "report"
        assert doc.node_props["name"] == "report.pages"
        assert doc.node_props["ext"] == ".pages"

    def test_happy_path_keynote(self, parser: IWorkParser) -> None:
        event = {
            "mime_type": "application/x-iwork-keynote",
            "source_id": "/slides/deck.key",
            "operation": "modified",
        }
        p_sys, p_inst = self._darwin_and_installed()
        with (
            p_sys,
            p_inst,
            patch("worker.parsers.iwork.os.path.isfile", return_value=True),
            patch("worker.parsers.iwork.tempfile.mkstemp", return_value=(99, "/tmp/out.txt")),
            patch("worker.parsers.iwork.os.close"),
            patch("worker.parsers.iwork.subprocess.run") as mock_run,
            patch("builtins.open", mock_open(read_data="Slide content")),
            patch("worker.parsers.iwork.os.unlink"),
        ):
            docs = parser.parse(event)

        assert len(docs) == 1
        # Verify osascript was called with Keynote app
        script_arg = mock_run.call_args[0][0][2]  # osascript -e <script>
        assert 'tell application "Keynote"' in script_arg

    def test_app_not_installed(self, parser: IWorkParser, pages_event: dict) -> None:
        with (
            patch("worker.parsers.iwork.platform.system", return_value="Darwin"),
            patch("worker.parsers.iwork._pages_installed", return_value=False),
            patch("worker.parsers.iwork.log") as mock_log,
        ):
            docs = parser.parse(pages_event)

        assert docs == []
        mock_log.warning.assert_called_once()
        assert "Pages.app not installed" in mock_log.warning.call_args[0][0]

    def test_file_not_found(self, parser: IWorkParser, pages_event: dict) -> None:
        p_sys, p_inst = self._darwin_and_installed()
        with (
            p_sys,
            p_inst,
            patch("worker.parsers.iwork.os.path.isfile", return_value=False),
            patch("worker.parsers.iwork.tempfile.mkstemp", return_value=(99, "/tmp/out.txt")),
            patch("worker.parsers.iwork.os.close"),
            patch("worker.parsers.iwork.os.unlink"),
            patch("worker.parsers.iwork.log") as mock_log,
        ):
            docs = parser.parse(pages_event)

        assert docs == []
        mock_log.error.assert_called_once()
        assert "File not found" in mock_log.error.call_args[0][0]

    def test_timeout(self, parser: IWorkParser, pages_event: dict) -> None:
        p_sys, p_inst = self._darwin_and_installed()
        with (
            p_sys,
            p_inst,
            patch("worker.parsers.iwork.os.path.isfile", return_value=True),
            patch("worker.parsers.iwork.tempfile.mkstemp", return_value=(99, "/tmp/out.txt")),
            patch("worker.parsers.iwork.os.close"),
            patch(
                "worker.parsers.iwork.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="osascript", timeout=30),
            ),
            patch("worker.parsers.iwork.os.unlink"),
            patch("worker.parsers.iwork.log") as mock_log,
        ):
            docs = parser.parse(pages_event)

        assert docs == []
        mock_log.error.assert_called_once()
        assert "timed out" in mock_log.error.call_args[0][0]

    def test_password_protected(self, parser: IWorkParser, pages_event: dict) -> None:
        p_sys, p_inst = self._darwin_and_installed()
        exc = subprocess.CalledProcessError(1, "osascript", stderr="The document is password protected")
        with (
            p_sys,
            p_inst,
            patch("worker.parsers.iwork.os.path.isfile", return_value=True),
            patch("worker.parsers.iwork.tempfile.mkstemp", return_value=(99, "/tmp/out.txt")),
            patch("worker.parsers.iwork.os.close"),
            patch("worker.parsers.iwork.subprocess.run", side_effect=exc),
            patch("worker.parsers.iwork.os.unlink"),
            patch("worker.parsers.iwork.log") as mock_log,
        ):
            docs = parser.parse(pages_event)

        assert docs == []
        mock_log.warning.assert_called_once()
        assert "Password-protected" in mock_log.warning.call_args[0][0]

    def test_osascript_generic_error(self, parser: IWorkParser, pages_event: dict) -> None:
        p_sys, p_inst = self._darwin_and_installed()
        exc = subprocess.CalledProcessError(1, "osascript", stderr="Some unknown error")
        with (
            p_sys,
            p_inst,
            patch("worker.parsers.iwork.os.path.isfile", return_value=True),
            patch("worker.parsers.iwork.tempfile.mkstemp", return_value=(99, "/tmp/out.txt")),
            patch("worker.parsers.iwork.os.close"),
            patch("worker.parsers.iwork.subprocess.run", side_effect=exc),
            patch("worker.parsers.iwork.os.unlink"),
            patch("worker.parsers.iwork.log") as mock_log,
        ):
            docs = parser.parse(pages_event)

        assert docs == []
        mock_log.error.assert_called_once()

    def test_empty_text_returns_empty(self, parser: IWorkParser, pages_event: dict) -> None:
        p_sys, p_inst = self._darwin_and_installed()
        with (
            p_sys,
            p_inst,
            patch("worker.parsers.iwork.os.path.isfile", return_value=True),
            patch("worker.parsers.iwork.tempfile.mkstemp", return_value=(99, "/tmp/out.txt")),
            patch("worker.parsers.iwork.os.close"),
            patch("worker.parsers.iwork.subprocess.run"),
            patch("builtins.open", mock_open(read_data="   \n  ")),
            patch("worker.parsers.iwork.os.unlink"),
        ):
            docs = parser.parse(pages_event)

        assert docs == []

    def test_temp_file_cleanup_on_success(self, parser: IWorkParser, pages_event: dict) -> None:
        p_sys, p_inst = self._darwin_and_installed()
        with (
            p_sys,
            p_inst,
            patch("worker.parsers.iwork.os.path.isfile", return_value=True),
            patch("worker.parsers.iwork.tempfile.mkstemp", return_value=(99, "/tmp/out.txt")),
            patch("worker.parsers.iwork.os.close") as mock_close,
            patch("worker.parsers.iwork.subprocess.run"),
            patch("builtins.open", mock_open(read_data="content")),
            patch("worker.parsers.iwork.os.unlink") as mock_unlink,
        ):
            parser.parse(pages_event)

        mock_close.assert_called_once_with(99)
        mock_unlink.assert_called_once_with("/tmp/out.txt")

    def test_temp_file_cleanup_on_error(self, parser: IWorkParser, pages_event: dict) -> None:
        p_sys, p_inst = self._darwin_and_installed()
        exc = subprocess.CalledProcessError(1, "osascript", stderr="error")
        with (
            p_sys,
            p_inst,
            patch("worker.parsers.iwork.os.path.isfile", return_value=True),
            patch("worker.parsers.iwork.tempfile.mkstemp", return_value=(99, "/tmp/out.txt")),
            patch("worker.parsers.iwork.os.close"),
            patch("worker.parsers.iwork.subprocess.run", side_effect=exc),
            patch("worker.parsers.iwork.os.unlink") as mock_unlink,
        ):
            parser.parse(pages_event)

        mock_unlink.assert_called_once_with("/tmp/out.txt")
