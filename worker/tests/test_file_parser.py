"""Tests for parsers/files.py — FileParser text and PDF handling."""

import base64
import hashlib

import pytest

from worker.parsers.files import FileParser


@pytest.fixture()
def parser() -> FileParser:
    return FileParser()


class TestFileParserSourceType:
    def test_source_type_is_files(self, parser: FileParser) -> None:
        assert parser.source_type == "files"


class TestFileParserDefaults:
    def test_default_limits(self, parser: FileParser) -> None:
        assert parser._max_pdf_bytes == 100 * 1024 * 1024
        assert parser._max_pdf_pages == 2000


class TestFileParserText:
    def test_parses_text_event(self, parser: FileParser) -> None:
        event = {
            "mime_type": "text/markdown",
            "source_id": "notes/hello.md",
            "operation": "created",
            "text": "Hello world",
        }
        docs = parser.parse(event)
        assert len(docs) == 1
        doc = docs[0]
        assert doc.source_type == "files"
        assert doc.source_id == "notes/hello.md"
        assert doc.operation == "created"
        assert doc.text == "Hello world"
        assert doc.mime_type == "text/markdown"
        assert doc.node_label == "File"

    def test_text_node_props(self, parser: FileParser) -> None:
        event = {
            "mime_type": "text/plain",
            "source_id": "/home/user/notes/test.txt",
            "text": "content",
        }
        docs = parser.parse(event)
        props = docs[0].node_props
        assert props["path"] == "/home/user/notes/test.txt"
        assert props["name"] == "test.txt"
        assert props["ext"] == ".txt"
        assert props["sha256"] == hashlib.sha256(b"content").hexdigest()

    def test_text_default_operation_is_modified(self, parser: FileParser) -> None:
        event = {"mime_type": "text/plain", "source_id": "x.txt", "text": "hi"}
        docs = parser.parse(event)
        assert docs[0].operation == "modified"

    def test_meta_modified_at_in_props(self, parser: FileParser) -> None:
        event = {
            "mime_type": "text/plain",
            "source_id": "x.txt",
            "text": "hi",
            "meta": {"modified_at": "2025-01-01"},
        }
        docs = parser.parse(event)
        assert docs[0].node_props["modified_at"] == "2025-01-01"

    def test_source_modified_at_in_props(self, parser: FileParser) -> None:
        event = {
            "mime_type": "text/plain",
            "source_id": "x.txt",
            "text": "hi",
            "source_modified_at": "2025-02-01",
        }
        docs = parser.parse(event)
        assert docs[0].node_props["modified_at"] == "2025-02-01"

    def test_empty_text(self, parser: FileParser) -> None:
        event = {"mime_type": "text/plain", "source_id": "x.txt", "text": ""}
        docs = parser.parse(event)
        assert len(docs) == 1
        assert docs[0].text == ""


class TestFileParserUnsupported:
    def test_returns_empty_for_unsupported_mime(self, parser: FileParser) -> None:
        event = {"mime_type": "image/png", "source_id": "photo.png"}
        docs = parser.parse(event)
        assert docs == []

    def test_returns_empty_for_empty_mime(self, parser: FileParser) -> None:
        event = {"source_id": "unknown"}
        docs = parser.parse(event)
        assert docs == []


class TestFileParserPdf:
    def test_parses_pdf_bytes(self, parser: FileParser) -> None:
        # Create a minimal valid PDF
        import pymupdf

        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Test PDF content")
        pdf_bytes = doc.tobytes()
        doc.close()

        event = {
            "mime_type": "application/pdf",
            "source_id": "docs/test.pdf",
            "operation": "created",
            "raw_bytes": pdf_bytes,
        }
        docs = parser.parse(event)
        assert len(docs) == 1
        assert "Test PDF content" in docs[0].text
        assert docs[0].mime_type == "application/pdf"
        assert docs[0].node_label == "File"
        assert docs[0].node_props["sha256"] == hashlib.sha256(pdf_bytes).hexdigest()

    def test_parses_base64_encoded_pdf(self, parser: FileParser) -> None:
        import pymupdf

        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Base64 content")
        pdf_bytes = doc.tobytes()
        doc.close()

        encoded = base64.b64encode(pdf_bytes).decode()
        event = {
            "mime_type": "application/pdf",
            "source_id": "test.pdf",
            "raw_bytes": encoded,
        }
        docs = parser.parse(event)
        assert len(docs) == 1
        assert "Base64 content" in docs[0].text

    def test_returns_empty_for_no_raw_bytes(self, parser: FileParser) -> None:
        event = {"mime_type": "application/pdf", "source_id": "test.pdf"}
        docs = parser.parse(event)
        assert docs == []

    def test_returns_empty_for_empty_raw_bytes(self, parser: FileParser) -> None:
        event = {"mime_type": "application/pdf", "source_id": "test.pdf", "raw_bytes": b""}
        docs = parser.parse(event)
        assert docs == []

    def test_oversized_pdf_bytes_skipped(self, parser: FileParser) -> None:
        parser._max_pdf_bytes = 10
        event = {
            "mime_type": "application/pdf",
            "source_id": "big.pdf",
            "raw_bytes": b"x" * 100,
        }
        docs = parser.parse(event)
        assert docs == []

    def test_oversized_base64_pdf_skipped(self, parser: FileParser) -> None:
        parser._max_pdf_bytes = 10
        encoded = base64.b64encode(b"x" * 100).decode()
        event = {
            "mime_type": "application/pdf",
            "source_id": "big.pdf",
            "raw_bytes": encoded,
        }
        docs = parser.parse(event)
        assert docs == []

    def test_invalid_pdf_returns_empty(self, parser: FileParser) -> None:
        event = {
            "mime_type": "application/pdf",
            "source_id": "bad.pdf",
            "raw_bytes": b"not a pdf",
        }
        docs = parser.parse(event)
        assert docs == []

    def test_page_limit_truncates(self, parser: FileParser) -> None:
        import pymupdf

        doc = pymupdf.open()
        for i in range(5):
            page = doc.new_page()
            page.insert_text((72, 72), f"Page {i}")
        pdf_bytes = doc.tobytes()
        doc.close()

        parser._max_pdf_pages = 2
        event = {
            "mime_type": "application/pdf",
            "source_id": "multi.pdf",
            "raw_bytes": pdf_bytes,
        }
        docs = parser.parse(event)
        assert len(docs) == 1
        # Only pages 0 and 1 should be included
        assert "Page 0" in docs[0].text
        assert "Page 1" in docs[0].text
        assert "Page 2" not in docs[0].text


class TestFileParserRegistration:
    def test_registered_in_parser_registry(self) -> None:
        from worker.parsers.registry import get

        parser = get("files")
        assert isinstance(parser, FileParser)
