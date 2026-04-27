"""Tests for worker.parsers.attachments.

Covers:

* The pure ``classify_attachment`` policy helper.
* The ``stream_and_parse`` stream-and-forget downloader (PDF / image / text /
  error paths, temp-file cleanup, weakref-based memory release check).
* The ``build_parent_url`` per-source URL builder.
"""

from __future__ import annotations

import gc
import os
import tempfile
import weakref

import pymupdf
import pytest

from worker.parsers.attachments import (
    DEFAULT_INDEXABLE_MIMETYPES,
    AttachmentDownloadError,
    AttachmentParseError,
    ParsedAttachment,
    build_parent_url,
    classify_attachment,
    stream_and_parse,
)


# --- classify_attachment ----------------------------------------------------


class TestDefaultIndexableMimetypes:
    """The bead pins the default allowlist; flag accidental drift."""

    def test_is_a_non_empty_list_of_strings(self) -> None:
        assert isinstance(DEFAULT_INDEXABLE_MIMETYPES, list)
        assert len(DEFAULT_INDEXABLE_MIMETYPES) > 0
        for m in DEFAULT_INDEXABLE_MIMETYPES:
            assert isinstance(m, str) and "/" in m

    def test_contains_required_types(self) -> None:
        # Spec: PDF, common image types, plaintext / markdown / CSV,
        # JSON / YAML.  Anything missing here is a regression.
        required = {
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
        }
        assert required.issubset(set(DEFAULT_INDEXABLE_MIMETYPES))


class TestClassifyAttachment:
    """classify_attachment(mime, size_bytes, indexable, max_size_mb)."""

    def test_small_pdf_is_downloaded(self) -> None:
        # 1 KB PDF, default indexable list, 25 MB cap.
        decision = classify_attachment(
            mime="application/pdf",
            size_bytes=1024,
            indexable=DEFAULT_INDEXABLE_MIMETYPES,
            max_size_mb=25,
        )
        assert decision == "download_and_index"

    def test_large_pdf_is_metadata_only(self) -> None:
        # 30 MB PDF exceeds the 25 MB cap.
        decision = classify_attachment(
            mime="application/pdf",
            size_bytes=30 * 1024 * 1024,
            indexable=DEFAULT_INDEXABLE_MIMETYPES,
            max_size_mb=25,
        )
        assert decision == "metadata_only"

    def test_zip_is_metadata_only(self) -> None:
        # Even a tiny zip is metadata-only — not in the allowlist.
        decision = classify_attachment(
            mime="application/zip",
            size_bytes=1024,
            indexable=DEFAULT_INDEXABLE_MIMETYPES,
            max_size_mb=25,
        )
        assert decision == "metadata_only"

    def test_size_at_boundary_is_inclusive(self) -> None:
        # Exactly max_size_mb * 1MiB → still downloaded (<= bound).
        decision = classify_attachment(
            mime="application/pdf",
            size_bytes=25 * 1024 * 1024,
            indexable=DEFAULT_INDEXABLE_MIMETYPES,
            max_size_mb=25,
        )
        assert decision == "download_and_index"

    def test_one_byte_over_boundary_is_metadata_only(self) -> None:
        decision = classify_attachment(
            mime="application/pdf",
            size_bytes=25 * 1024 * 1024 + 1,
            indexable=DEFAULT_INDEXABLE_MIMETYPES,
            max_size_mb=25,
        )
        assert decision == "metadata_only"

    def test_zero_byte_indexable_is_downloaded(self) -> None:
        # 0 bytes is in-range (0 <= 25*1MiB) and the MIME is allowlisted.
        decision = classify_attachment(
            mime="text/plain",
            size_bytes=0,
            indexable=DEFAULT_INDEXABLE_MIMETYPES,
            max_size_mb=25,
        )
        assert decision == "download_and_index"

    def test_custom_indexable_list_excludes_pdf(self) -> None:
        decision = classify_attachment(
            mime="application/pdf",
            size_bytes=1024,
            indexable=["text/plain"],
            max_size_mb=25,
        )
        assert decision == "metadata_only"

    def test_custom_indexable_list_includes_zip(self) -> None:
        decision = classify_attachment(
            mime="application/zip",
            size_bytes=1024,
            indexable=["application/zip"],
            max_size_mb=25,
        )
        assert decision == "download_and_index"

    def test_custom_max_size_mb_one(self) -> None:
        # 1 MB cap: 512 KB JPEG passes, 2 MB JPEG does not.
        small = classify_attachment(
            mime="image/jpeg",
            size_bytes=512 * 1024,
            indexable=DEFAULT_INDEXABLE_MIMETYPES,
            max_size_mb=1,
        )
        big = classify_attachment(
            mime="image/jpeg",
            size_bytes=2 * 1024 * 1024,
            indexable=DEFAULT_INDEXABLE_MIMETYPES,
            max_size_mb=1,
        )
        assert small == "download_and_index"
        assert big == "metadata_only"

    @pytest.mark.parametrize(
        "mime,size_bytes,expected",
        [
            ("image/png", 100, "download_and_index"),
            ("image/heic", 24 * 1024 * 1024, "download_and_index"),
            ("application/octet-stream", 100, "metadata_only"),
            ("text/markdown", 25 * 1024 * 1024 + 1, "metadata_only"),
        ],
    )
    def test_parametrised_cases(
        self, mime: str, size_bytes: int, expected: str
    ) -> None:
        assert (
            classify_attachment(
                mime=mime,
                size_bytes=size_bytes,
                indexable=DEFAULT_INDEXABLE_MIMETYPES,
                max_size_mb=25,
            )
            == expected
        )


# --- helpers shared across stream_and_parse tests ---------------------------


def _tiny_pdf_bytes(text: str = "hello world from a test PDF") -> bytes:
    """Build an in-memory PDF using pymupdf."""
    doc = pymupdf.open()  # blank document
    try:
        page = doc.new_page()
        page.insert_text((72, 72), text)
        return doc.tobytes()
    finally:
        doc.close()


class _FakeVisionResult:
    def __init__(
        self,
        description: str = "",
        visible_text: str = "",
        entities: list[dict[str, str]] | None = None,
    ) -> None:
        self.description = description
        self.visible_text = visible_text
        self.entities = entities or []


def _tempdir_snapshot() -> set[str]:
    return set(os.listdir(tempfile.gettempdir()))


# --- stream_and_parse -------------------------------------------------------


class TestStreamAndParsePDF:
    def test_small_pdf_is_parsed(self) -> None:
        pdf = _tiny_pdf_bytes("attachment body text")
        before = _tempdir_snapshot()

        result = stream_and_parse(
            fetch=lambda: pdf,
            filename="report.pdf",
            mime="application/pdf",
        )

        after = _tempdir_snapshot()
        assert isinstance(result, ParsedAttachment)
        assert "attachment body text" in result.text
        # Stream-and-forget invariant: no temp files left behind by pymupdf.
        assert after == before

    def test_corrupted_pdf_raises_parse_error(self) -> None:
        with pytest.raises(AttachmentParseError) as exc:
            stream_and_parse(
                fetch=lambda: b"%PDF-1.4 not actually a pdf",
                filename="bad.pdf",
                mime="application/pdf",
                source_id="msg-123",
            )
        assert exc.value.source_id == "msg-123"


class TestStreamAndParseImage:
    def test_image_with_vision_extractor(self) -> None:
        captured: dict[str, object] = {}

        def vision(data: bytes, mime: str) -> _FakeVisionResult:
            captured["bytes_len"] = len(data)
            captured["mime"] = mime
            return _FakeVisionResult(
                description="a cat on a sofa",
                visible_text="WELCOME",
                entities=[
                    {"name": "Whiskers", "type": "Person"},
                    {"name": "Sofa Co", "type": "Organization"},
                ],
            )

        before = _tempdir_snapshot()

        result = stream_and_parse(
            fetch=lambda: b"\x89PNG\r\n\x1a\n" + b"\x00" * 64,
            filename="cat.png",
            mime="image/png",
            vision_extractor=vision,
            source_id="msg-7",
        )

        after = _tempdir_snapshot()
        assert captured["mime"] == "image/png"
        assert "a cat on a sofa" in result.text
        assert "WELCOME" in result.text
        assert result.description == "a cat on a sofa"
        assert result.ocr_text == "WELCOME"
        assert len(result.extracted_entities) == 2
        # Vision extractor today consumes bytes directly; no temp files.
        assert after == before

    def test_image_without_vision_extractor_raises_parse_error(self) -> None:
        with pytest.raises(AttachmentParseError) as exc:
            stream_and_parse(
                fetch=lambda: b"\x89PNG",
                filename="x.png",
                mime="image/png",
                source_id="msg-9",
            )
        assert exc.value.source_id == "msg-9"

    def test_vision_extractor_failure_raises_parse_error(self) -> None:
        def boom(_data: bytes, _mime: str) -> _FakeVisionResult:
            raise RuntimeError("model timeout")

        with pytest.raises(AttachmentParseError):
            stream_and_parse(
                fetch=lambda: b"\x89PNG",
                filename="x.png",
                mime="image/png",
                vision_extractor=boom,
            )


class TestStreamAndParseText:
    def test_utf8_text_decodes(self) -> None:
        result = stream_and_parse(
            fetch=lambda: "résumé\nline 2".encode("utf-8"),
            filename="notes.txt",
            mime="text/plain",
        )
        assert result.text == "résumé\nline 2"

    def test_invalid_utf8_uses_replacement(self) -> None:
        result = stream_and_parse(
            fetch=lambda: b"\xff\xfeokay",
            filename="notes.txt",
            mime="text/plain",
        )
        # 'errors=replace' means we never crash on bad bytes.
        assert "okay" in result.text


class TestStreamAndParseErrors:
    def test_fetch_failure_raises_download_error(self) -> None:
        def boom() -> bytes:
            raise ConnectionError("network down")

        with pytest.raises(AttachmentDownloadError) as exc:
            stream_and_parse(
                fetch=boom,
                filename="r.pdf",
                mime="application/pdf",
                source_id="msg-42",
            )
        assert exc.value.source_id == "msg-42"
        assert isinstance(exc.value.__cause__, ConnectionError)

    def test_unsupported_mime_raises_parse_error(self) -> None:
        # Even when a caller has classified the attachment as indexable
        # (e.g. application/zip in a custom allowlist) the parser router
        # still refuses anything it does not know how to handle.
        with pytest.raises(AttachmentParseError) as exc:
            stream_and_parse(
                fetch=lambda: b"PK\x03\x04",
                filename="archive.zip",
                mime="application/zip",
                source_id="msg-x",
            )
        assert exc.value.source_id == "msg-x"


class _Holder:
    """Tiny weakref-able container so we can track byte-buffer lifetime."""

    __slots__ = ("data", "__weakref__")

    def __init__(self, data: bytes) -> None:
        self.data = data


class TestStreamAndParseMemoryRelease:
    """The 'stream-and-forget' invariant: no reference to the bytes
    survives the call. We allocate a 5 MiB buffer, give the helper its
    own copy via the fetch closure, and verify the closure-internal
    holder dies after ``stream_and_parse`` returns."""

    def test_bytes_are_released_after_return(self) -> None:
        holder_ref: list[weakref.ref[_Holder]] = []

        def fetch() -> bytes:
            holder = _Holder(b"x" * (5 * 1024 * 1024))
            holder_ref.append(weakref.ref(holder))
            return holder.data

        stream_and_parse(
            fetch=fetch,
            filename="big.txt",
            mime="text/plain",
        )

        gc.collect()
        # The helper kept neither the bytes nor anything that transitively
        # references the closure-local holder.
        assert holder_ref[0]() is None


# --- build_parent_url -------------------------------------------------------


class TestBuildParentURL:
    def test_gmail_shape(self) -> None:
        url = build_parent_url("gmail", thread_id="18a2b3c4d5")
        assert url == "https://mail.google.com/mail/?ui=2&view=cv&th=18a2b3c4d5"

    def test_gmail_missing_thread_id_raises(self) -> None:
        with pytest.raises(ValueError):
            build_parent_url("gmail")

    def test_slack_shape(self) -> None:
        # Slack permalinks strip the "." from the timestamp.
        url = build_parent_url(
            "slack",
            team_domain="acme",
            channel_id="C12345",
            ts="1700000000.123456",
        )
        assert url == "https://acme.slack.com/archives/C12345/p1700000000123456"

    def test_slack_missing_kwargs_raises(self) -> None:
        with pytest.raises(ValueError):
            build_parent_url("slack", team_domain="acme", channel_id="C1")

    def test_calendar_returns_html_link_verbatim(self) -> None:
        link = "https://www.google.com/calendar/event?eid=abc123"
        assert build_parent_url("calendar", html_link=link) == link

    def test_calendar_missing_html_link_raises(self) -> None:
        with pytest.raises(ValueError):
            build_parent_url("calendar")

    def test_unknown_source_type_raises(self) -> None:
        with pytest.raises(ValueError):
            build_parent_url("teams", thread_id="x")
