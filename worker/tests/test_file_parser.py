"""Tests for parsers/files.py — FileParser text and PDF handling."""

import base64
import hashlib

import pytest

from worker.parsers.files import FileParser, _EXTENSION_DESCRIPTIONS


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
    def test_returns_metadata_for_unsupported_mime(self, parser: FileParser) -> None:
        event = {"mime_type": "application/octet-stream", "source_id": "data.bin"}
        docs = parser.parse(event)
        assert len(docs) == 1
        doc = docs[0]
        assert doc.source_type == "files"
        assert doc.source_id == "data.bin"
        assert doc.mime_type == "application/octet-stream"
        assert doc.node_label == "File"
        assert "File: data.bin" in doc.text
        assert doc.image_bytes is None

    def test_returns_metadata_for_empty_mime(self, parser: FileParser) -> None:
        event = {"source_id": "/tmp/unknown"}
        docs = parser.parse(event)
        assert len(docs) == 1
        assert docs[0].mime_type == "application/octet-stream"
        assert "File: unknown" in docs[0].text


class TestFileParserImage:
    """Tests for standalone image file parsing."""

    _SAMPLE_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64

    def test_standalone_png_emits_parsed_document_with_image_bytes(
        self, parser: FileParser
    ) -> None:
        event = {
            "mime_type": "image/png",
            "source_id": "/vault/photos/screenshot.png",
            "operation": "created",
            "raw_bytes": self._SAMPLE_PNG,
        }
        docs = parser.parse(event)
        assert len(docs) == 1
        doc = docs[0]
        assert doc.source_type == "files"
        assert doc.source_id == "/vault/photos/screenshot.png"
        assert doc.operation == "created"
        assert doc.text == ""
        assert doc.image_bytes == self._SAMPLE_PNG
        assert doc.mime_type == "image/png"
        assert doc.node_label == "File"
        assert doc.node_props["sha256"] == hashlib.sha256(self._SAMPLE_PNG).hexdigest()

    def test_image_jpeg_parsed(self, parser: FileParser) -> None:
        jpeg_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 64
        event = {
            "mime_type": "image/jpeg",
            "source_id": "photo.jpg",
            "raw_bytes": jpeg_bytes,
        }
        docs = parser.parse(event)
        assert len(docs) == 1
        assert docs[0].image_bytes == jpeg_bytes
        assert docs[0].mime_type == "image/jpeg"

    def test_image_no_raw_bytes_returns_empty(self, parser: FileParser) -> None:
        event = {"mime_type": "image/png", "source_id": "photo.png"}
        docs = parser.parse(event)
        assert docs == []

    def test_image_empty_raw_bytes_returns_empty(self, parser: FileParser) -> None:
        event = {"mime_type": "image/png", "source_id": "photo.png", "raw_bytes": b""}
        docs = parser.parse(event)
        assert docs == []

    def test_oversized_image_skipped(self, parser: FileParser) -> None:
        parser._max_image_bytes = 10
        event = {
            "mime_type": "image/png",
            "source_id": "huge.png",
            "raw_bytes": b"x" * 100,
        }
        docs = parser.parse(event)
        assert docs == []

    def test_base64_encoded_image(self, parser: FileParser) -> None:
        encoded = base64.b64encode(self._SAMPLE_PNG).decode()
        event = {
            "mime_type": "image/png",
            "source_id": "photo.png",
            "raw_bytes": encoded,
        }
        docs = parser.parse(event)
        assert len(docs) == 1
        assert docs[0].image_bytes == self._SAMPLE_PNG

    def test_image_node_props(self, parser: FileParser) -> None:
        event = {
            "mime_type": "image/png",
            "source_id": "/vault/img/test.png",
            "raw_bytes": self._SAMPLE_PNG,
        }
        docs = parser.parse(event)
        props = docs[0].node_props
        assert props["path"] == "/vault/img/test.png"
        assert props["name"] == "test.png"
        assert props["ext"] == ".png"

    def test_webp_mime_type(self, parser: FileParser) -> None:
        event = {
            "mime_type": "image/webp",
            "source_id": "photo.webp",
            "raw_bytes": b"RIFF" + b"\x00" * 64,
        }
        docs = parser.parse(event)
        assert len(docs) == 1
        assert docs[0].mime_type == "image/webp"


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


class TestFileParserMetadataOnly:
    """Tests for metadata-only indexing of unparseable files."""

    def test_unknown_binary_gets_metadata_doc(self, parser: FileParser) -> None:
        event = {
            "mime_type": "application/octet-stream",
            "source_id": "/projects/furniture/drawer.3mf",
            "operation": "created",
            "meta": {"size_bytes": 1024},
        }
        docs = parser.parse(event)
        assert len(docs) == 1
        doc = docs[0]
        assert doc.source_type == "files"
        assert doc.source_id == "/projects/furniture/drawer.3mf"
        assert doc.operation == "created"
        assert doc.text == "File: drawer.3mf (3D model) in /projects/furniture/"
        assert doc.mime_type == "application/octet-stream"
        assert doc.node_label == "File"
        assert doc.image_bytes is None
        props = doc.node_props
        assert props["name"] == "drawer.3mf"
        assert props["ext"] == ".3mf"
        assert props["path"] == "/projects/furniture/drawer.3mf"
        assert props["size_bytes"] == 1024

    def test_known_extension_descriptions(self, parser: FileParser) -> None:
        """Spot-check the extension description map."""
        assert _EXTENSION_DESCRIPTIONS[".mp4"] == "video"
        assert _EXTENSION_DESCRIPTIONS[".zip"] == "archive"
        assert _EXTENSION_DESCRIPTIONS[".psd"] == "design file"
        assert _EXTENSION_DESCRIPTIONS[".xlsx"] == "spreadsheet"
        assert _EXTENSION_DESCRIPTIONS[".docx"] == "document"
        assert _EXTENSION_DESCRIPTIONS[".db"] == "database"
        assert _EXTENSION_DESCRIPTIONS[".exe"] == "executable"
        assert _EXTENSION_DESCRIPTIONS[".dmg"] == "disk image"

    def test_unknown_extension_falls_back_to_extension(self, parser: FileParser) -> None:
        event = {
            "mime_type": "application/octet-stream",
            "source_id": "/data/file.xyz",
            "operation": "modified",
        }
        docs = parser.parse(event)
        assert len(docs) == 1
        assert docs[0].text == "File: file.xyz in /data/"

    def test_delete_event_for_unparseable_file(self, parser: FileParser) -> None:
        event = {
            "mime_type": "application/octet-stream",
            "source_id": "/projects/model.stl",
            "operation": "deleted",
        }
        docs = parser.parse(event)
        assert len(docs) == 1
        assert docs[0].operation == "deleted"
        assert docs[0].source_id == "/projects/model.stl"

    def test_metadata_node_props_include_directory(self, parser: FileParser) -> None:
        event = {
            "mime_type": "application/octet-stream",
            "source_id": "/home/user/docs/archive.7z",
            "operation": "created",
            "source_modified_at": "2026-01-15T10:00:00Z",
            "meta": {"size_bytes": 5000},
        }
        docs = parser.parse(event)
        props = docs[0].node_props
        assert props["path"] == "/home/user/docs/archive.7z"
        assert props["name"] == "archive.7z"
        assert props["ext"] == ".7z"
        assert props["size_bytes"] == 5000
        assert props["source_modified_at"] == "2026-01-15T10:00:00Z"

    def test_file_without_directory(self, parser: FileParser) -> None:
        event = {
            "mime_type": "application/octet-stream",
            "source_id": "loose.bin",
            "operation": "created",
        }
        docs = parser.parse(event)
        # No trailing " in /" when there's no directory
        assert docs[0].text == "File: loose.bin"


class TestFileParserRegistration:
    def test_registered_in_parser_registry(self) -> None:
        from worker.parsers.registry import get

        parser = get("files")
        assert isinstance(parser, FileParser)
