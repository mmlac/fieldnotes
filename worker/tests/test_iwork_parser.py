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
    """Tests for macOS Pages text extraction via osascript."""

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


class TestKeynoteExtraction:
    """Tests for Keynote text extraction via osascript."""

    @pytest.fixture()
    def parser(self) -> IWorkParser:
        return IWorkParser()

    @pytest.fixture()
    def keynote_event(self) -> dict:
        return {
            "mime_type": "application/x-iwork-keynote",
            "source_id": "/slides/deck.key",
            "operation": "created",
        }

    def _darwin_patches(
        self,
        *,
        installed: bool = True,
        exists: bool = True,
        stdout: str = "",
        stderr: str = "",
        returncode: int = 0,
        side_effect: Exception | None = None,
    ):
        """Return a combined context manager with all standard macOS mocks."""
        mock_result = MagicMock()
        mock_result.stdout = stdout
        mock_result.stderr = stderr
        mock_result.returncode = returncode

        class _Ctx:
            def __enter__(self_inner):
                self_inner._p1 = patch("worker.parsers.iwork.platform.system", return_value="Darwin")
                self_inner._p2 = patch("worker.parsers.iwork._keynote_installed", return_value=installed)
                self_inner._p3 = patch("worker.parsers.iwork.os.path.isfile", return_value=exists)
                if side_effect:
                    self_inner._p4 = patch("worker.parsers.iwork.subprocess.run", side_effect=side_effect)
                else:
                    self_inner._p4 = patch("worker.parsers.iwork.subprocess.run", return_value=mock_result)
                self_inner._p5 = patch("worker.parsers.iwork.IWORK_EXTRACTION_DURATION_SECONDS")
                self_inner._p1.start()
                self_inner._p2.start()
                self_inner._p3.start()
                self_inner._p4.start()
                self_inner._p5.start()
                return self_inner

            def __exit__(self_inner, *args):
                self_inner._p5.stop()
                self_inner._p4.stop()
                self_inner._p3.stop()
                self_inner._p2.stop()
                self_inner._p1.stop()

        return _Ctx()

    def test_happy_path_extracts_text(self, parser: IWorkParser, keynote_event: dict) -> None:
        slide_text = "Slide 1 Title\nBullet point\n\nSlide 2 Title\nMore content"
        with self._darwin_patches(stdout=slide_text):
            docs = parser.parse(keynote_event)

        assert len(docs) == 1
        doc = docs[0]
        assert doc.source_type == "files"
        assert doc.source_id == "/slides/deck.key"
        assert doc.operation == "created"
        assert doc.mime_type == "application/x-iwork-keynote"
        assert doc.text == slide_text
        assert doc.node_props["name"] == "deck.key"
        assert doc.node_props["ext"] == ".key"
        assert doc.node_props["path"] == "/slides/deck.key"

    def test_app_not_installed_returns_empty(self, parser: IWorkParser, keynote_event: dict) -> None:
        with (
            patch("worker.parsers.iwork.platform.system", return_value="Darwin"),
            patch("worker.parsers.iwork._keynote_installed", return_value=False),
            patch("worker.parsers.iwork.log") as mock_log,
        ):
            docs = parser.parse(keynote_event)

        assert docs == []
        mock_log.warning.assert_called_once()
        assert "Keynote.app not installed" in mock_log.warning.call_args[0][0]

    def test_file_not_found_returns_empty(self, parser: IWorkParser, keynote_event: dict) -> None:
        with (
            patch("worker.parsers.iwork.platform.system", return_value="Darwin"),
            patch("worker.parsers.iwork._keynote_installed", return_value=True),
            patch("worker.parsers.iwork.os.path.isfile", return_value=False),
            patch("worker.parsers.iwork.log") as mock_log,
        ):
            docs = parser.parse(keynote_event)

        assert docs == []
        mock_log.error.assert_called_once()
        assert "not found" in mock_log.error.call_args[0][0]

    def test_timeout_returns_empty(self, parser: IWorkParser, keynote_event: dict) -> None:
        with (
            self._darwin_patches(
                side_effect=subprocess.TimeoutExpired(cmd="osascript", timeout=120),
            ),
            patch("worker.parsers.iwork.log") as mock_log,
        ):
            docs = parser.parse(keynote_event)

        assert docs == []
        mock_log.error.assert_called_once()
        assert "timed out" in mock_log.error.call_args[0][0]

    def test_osascript_error_returns_empty(self, parser: IWorkParser, keynote_event: dict) -> None:
        with (
            self._darwin_patches(returncode=1, stderr="execution error: password protected"),
            patch("worker.parsers.iwork.log") as mock_log,
        ):
            docs = parser.parse(keynote_event)

        assert docs == []
        mock_log.error.assert_called_once()
        assert "failed" in mock_log.error.call_args[0][0].lower()

    def test_empty_presentation_returns_empty(self, parser: IWorkParser, keynote_event: dict) -> None:
        with (
            self._darwin_patches(stdout=""),
            patch("worker.parsers.iwork.log") as mock_log,
        ):
            docs = parser.parse(keynote_event)

        assert docs == []
        mock_log.debug.assert_called()

    def test_collapses_excessive_newlines(self, parser: IWorkParser, keynote_event: dict) -> None:
        raw_text = "Slide 1\n\n\n\nSlide 2\n\n\n\n\nSlide 3"
        with self._darwin_patches(stdout=raw_text):
            docs = parser.parse(keynote_event)

        assert len(docs) == 1
        assert docs[0].text == "Slide 1\n\nSlide 2\n\nSlide 3"

    def test_metadata_propagation(self, parser: IWorkParser) -> None:
        event = {
            "mime_type": "application/x-iwork-keynote",
            "source_id": "/slides/deck.key",
            "operation": "modified",
            "source_modified_at": "2026-01-15T10:00:00Z",
        }
        with self._darwin_patches(stdout="Some content"):
            docs = parser.parse(event)

        assert len(docs) == 1
        assert docs[0].node_props["modified_at"] == "2026-01-15T10:00:00Z"
        assert docs[0].operation == "modified"
