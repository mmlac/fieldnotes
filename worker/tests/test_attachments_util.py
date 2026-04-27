"""Tests for worker.parsers.attachments.

Covers:

* The pure ``classify_attachment`` policy helper.
* The ``stream_and_parse`` stream-and-forget downloader (PDF / image / text /
  error paths, temp-file cleanup, refcount-based memory release check).
* The ``build_parent_url`` per-source URL builder.
"""

from __future__ import annotations

import gc
import os
import sys
import tempfile

import io
from unittest.mock import patch

import pymupdf
import pytest
from PIL import Image

from worker.parsers import attachments as attachments_mod
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

    def test_classify_case_insensitive(self) -> None:
        # RFC 6838: MIME type/subtype are case-insensitive. An upstream
        # emitting 'image/JPEG' must still match the lowercase allowlist.
        assert (
            classify_attachment(
                mime="image/JPEG",
                size_bytes=1024,
                indexable=DEFAULT_INDEXABLE_MIMETYPES,
                max_size_mb=25,
            )
            == "download_and_index"
        )
        assert (
            classify_attachment(
                mime="APPLICATION/PDF",
                size_bytes=1024,
                indexable=DEFAULT_INDEXABLE_MIMETYPES,
                max_size_mb=25,
            )
            == "download_and_index"
        )

    def test_classify_strips_parameters(self) -> None:
        # 'text/plain; charset=utf-8' is the same media type as 'text/plain';
        # the optional parameter must not block allowlist match.
        assert (
            classify_attachment(
                mime="text/plain; charset=utf-8",
                size_bytes=1024,
                indexable=DEFAULT_INDEXABLE_MIMETYPES,
                max_size_mb=25,
            )
            == "download_and_index"
        )
        # Parameter with no leading space is also valid syntax.
        assert (
            classify_attachment(
                mime="application/json;charset=UTF-8",
                size_bytes=1024,
                indexable=DEFAULT_INDEXABLE_MIMETYPES,
                max_size_mb=25,
            )
            == "download_and_index"
        )

    def test_classify_strips_whitespace(self) -> None:
        # Stray surrounding whitespace from upstream label cleanup.
        assert (
            classify_attachment(
                mime=" image/png ",
                size_bytes=1024,
                indexable=DEFAULT_INDEXABLE_MIMETYPES,
                max_size_mb=25,
            )
            == "download_and_index"
        )

    def test_indexable_list_normalized_at_load(self) -> None:
        # A misconfigured allowlist with mixed-case / parameterized /
        # whitespace-padded entries still matches a clean incoming MIME.
        custom = ["IMAGE/PNG", " text/Plain; charset=utf-8 ", "Application/PDF"]
        assert (
            classify_attachment(
                mime="image/png",
                size_bytes=1024,
                indexable=custom,
                max_size_mb=25,
            )
            == "download_and_index"
        )
        assert (
            classify_attachment(
                mime="text/plain",
                size_bytes=1024,
                indexable=custom,
                max_size_mb=25,
            )
            == "download_and_index"
        )
        assert (
            classify_attachment(
                mime="application/pdf",
                size_bytes=1024,
                indexable=custom,
                max_size_mb=25,
            )
            == "download_and_index"
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


def _tiny_png_bytes(width: int = 16, height: int = 16) -> bytes:
    """Build a real in-memory PNG of the given dimensions."""
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(123, 45, 67)).save(buf, format="PNG")
    return buf.getvalue()


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

    def test_normal_doc_unaffected_by_default_caps(self) -> None:
        # Sanity: a normal small PDF passes through with all three default
        # caps in place, no truncation or timeout. (Fixture from fn-abr.)
        pdf = _tiny_pdf_bytes("normal document, well under all caps")
        result = stream_and_parse(
            fetch=lambda: pdf,
            filename="report.pdf",
            mime="application/pdf",
        )
        assert "normal document" in result.text

    def test_pdf_too_many_pages_raises_parse_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Stub pymupdf.open to return a fake doc whose page_count exceeds
        # the cap.  We don't synthesize 2000 real pages — that would slow
        # the test for no extra coverage.
        class _FakeDoc:
            page_count = 5_000

            def __iter__(self):  # pragma: no cover - guard fires first
                raise AssertionError("page enumeration should not run")

            def close(self) -> None:
                pass

        monkeypatch.setattr(
            attachments_mod.pymupdf,
            "open",
            lambda **_kwargs: _FakeDoc(),
        )

        with pytest.raises(AttachmentParseError) as exc:
            stream_and_parse(
                fetch=lambda: b"%PDF-1.4 stub",
                filename="bomb.pdf",
                mime="application/pdf",
                source_id="msg-bomb",
                pdf_max_pages=1000,
            )
        assert "too many pages" in str(exc.value)
        assert exc.value.source_id == "msg-bomb"

    def test_pdf_per_page_text_is_truncated(self) -> None:
        # Build a real one-page PDF whose extracted text is comfortably
        # longer than the per-page cap, then assert the parser truncated it.
        long_body = "abcdefghij " * 200  # > 1 KB of extractable text
        pdf = _tiny_pdf_bytes(long_body)

        cap = 50
        result = stream_and_parse(
            fetch=lambda: pdf,
            filename="long.pdf",
            mime="application/pdf",
            pdf_per_page_chars=cap,
        )
        # The parser concatenates pages with a newline; for a single-page
        # PDF the result text must not exceed the cap.
        assert len(result.text) <= cap

    def test_pdf_timeout_raises_parse_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Stub pymupdf.open to sleep longer than the configured timeout;
        # parse must abandon and raise AttachmentParseError.
        import time

        def _slow_open(**_kwargs):
            time.sleep(2.0)
            raise AssertionError("should have been abandoned by timeout")

        monkeypatch.setattr(attachments_mod.pymupdf, "open", _slow_open)

        # threading.Thread.join() accepts a float timeout; the parser's
        # int annotation is operator-facing — passing a fractional value
        # here keeps the test fast.
        with pytest.raises(AttachmentParseError) as exc:
            stream_and_parse(
                fetch=lambda: b"%PDF-1.4 stub",
                filename="slow.pdf",
                mime="application/pdf",
                source_id="msg-slow",
                pdf_timeout_seconds=0.2,  # type: ignore[arg-type]
            )
        assert "exceeded timeout" in str(exc.value)
        assert exc.value.source_id == "msg-slow"


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
            fetch=lambda: _tiny_png_bytes(),
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
                fetch=lambda: _tiny_png_bytes(),
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
                fetch=lambda: _tiny_png_bytes(),
                filename="x.png",
                mime="image/png",
                vision_extractor=boom,
            )


class TestStreamAndParseImageBomb:
    """Image-bomb defenses applied before the vision extractor runs."""

    def _ok_extractor(self) -> "_FakeVisionResult":
        return _FakeVisionResult(description="ok")

    def test_image_normal_unaffected(self) -> None:
        # 16x16 PNG well under both caps — vision_extractor must run.
        called: dict[str, bool] = {}

        def vision(_data: bytes, _mime: str) -> _FakeVisionResult:
            called["yes"] = True
            return _FakeVisionResult(description="cute")

        result = stream_and_parse(
            fetch=lambda: _tiny_png_bytes(16, 16),
            filename="ok.png",
            mime="image/png",
            vision_extractor=vision,
        )
        assert called.get("yes") is True
        assert result.description == "cute"

    def test_image_too_many_pixels(self) -> None:
        # Spoof Image.open(...).size to declare a 6000x5000 image: stays
        # under the per-side cap (8000) but blows past the 25M pixel cap
        # (= 30M).  Fixture would otherwise allocate ~120 MB on full decode.
        png = _tiny_png_bytes(16, 16)

        class _FakeImg:
            size = (6_000, 5_000)

            def __enter__(self) -> "_FakeImg":
                return self

            def __exit__(self, *_a: object) -> None:
                return None

        with patch.object(attachments_mod.Image, "open", lambda _stream: _FakeImg()):
            with pytest.raises(AttachmentParseError) as exc:
                stream_and_parse(
                    fetch=lambda: png,
                    filename="bomb.png",
                    mime="image/png",
                    vision_extractor=lambda *_: _FakeVisionResult(),
                    source_id="msg-bomb",
                )
        assert "pixel count" in str(exc.value)
        assert exc.value.source_id == "msg-bomb"

    def test_image_too_long_dimension(self) -> None:
        # 100x20000 stays under DEFAULT_IMAGE_MAX_PIXELS (25M) but exceeds
        # DEFAULT_IMAGE_MAX_DIMENSION (8000) on the height axis.
        png = _tiny_png_bytes(16, 16)

        class _FakeImg:
            size = (100, 20_000)

            def __enter__(self) -> "_FakeImg":
                return self

            def __exit__(self, *_a: object) -> None:
                return None

        with patch.object(attachments_mod.Image, "open", lambda _stream: _FakeImg()):
            with pytest.raises(AttachmentParseError) as exc:
                stream_and_parse(
                    fetch=lambda: png,
                    filename="thin.png",
                    mime="image/png",
                    vision_extractor=lambda *_: _FakeVisionResult(),
                    source_id="msg-thin",
                )
        assert "per-side cap" in str(exc.value)
        assert exc.value.source_id == "msg-thin"

    def test_image_decompression_bomb(self) -> None:
        # Pillow itself raises DecompressionBombError when MAX_IMAGE_PIXELS
        # is exceeded by a wide margin.  The guard must surface this as
        # AttachmentParseError rather than letting it propagate raw.
        def _bomb(_stream: object) -> object:
            raise attachments_mod.Image.DecompressionBombError("100M pixels")

        png = _tiny_png_bytes(16, 16)
        with patch.object(attachments_mod.Image, "open", _bomb):
            with pytest.raises(AttachmentParseError) as exc:
                stream_and_parse(
                    fetch=lambda: png,
                    filename="evil.png",
                    mime="image/png",
                    vision_extractor=lambda *_: _FakeVisionResult(),
                    source_id="msg-evil",
                )
        assert "decompression bomb" in str(exc.value).lower()
        assert exc.value.source_id == "msg-evil"

    def test_image_max_pixels_kwarg_overrides_default(self) -> None:
        # Operator-supplied tighter cap rejects the 16x16 fixture.
        with pytest.raises(AttachmentParseError) as exc:
            stream_and_parse(
                fetch=lambda: _tiny_png_bytes(16, 16),
                filename="tiny.png",
                mime="image/png",
                vision_extractor=lambda *_: _FakeVisionResult(),
                source_id="msg-tight",
                image_max_pixels=10,
            )
        assert "pixel count" in str(exc.value)
        assert exc.value.source_id == "msg-tight"

    def test_image_unreadable_header_raises_parse_error(self) -> None:
        # Bytes that are not a recognizable image at all — header parse
        # fails inside the guard rather than the extractor.
        with pytest.raises(AttachmentParseError) as exc:
            stream_and_parse(
                fetch=lambda: b"not an image",
                filename="garbage.png",
                mime="image/png",
                vision_extractor=lambda *_: _FakeVisionResult(),
                source_id="msg-junk",
            )
        assert "unreadable" in str(exc.value)
        assert exc.value.source_id == "msg-junk"


class TestStreamAndParseText:
    def test_utf8_text_decodes(self) -> None:
        result = stream_and_parse(
            fetch=lambda: "résumé\nline 2".encode("utf-8"),
            filename="notes.txt",
            mime="text/plain",
        )
        assert result.text == "résumé\nline 2"

    def test_low_ratio_invalid_utf8_uses_replacement(self) -> None:
        # One stray invalid byte amid otherwise-valid UTF-8 stays under the
        # 5% replacement-ratio limit and decodes with U+FFFD substitution
        # rather than raising — text/* attachments often contain truncated
        # multibyte sequences or accidental encoding mixups.
        body = ("hello world " * 20).encode("utf-8") + b"\xff"
        result = stream_and_parse(
            fetch=lambda: body,
            filename="notes.txt",
            mime="text/plain",
        )
        assert "hello world" in result.text


class TestStreamAndParseTextSpoofing:
    """fn-2hw: refuse text/* attachments whose payload is binary."""

    def test_text_with_pdf_magic_rejected(self) -> None:
        with pytest.raises(AttachmentParseError) as exc:
            stream_and_parse(
                fetch=lambda: b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\nrest of binary",
                filename="spoof.txt",
                mime="text/plain",
                source_id="msg-pdf",
            )
        assert "binary magic" in str(exc.value)
        assert exc.value.source_id == "msg-pdf"

    def test_text_with_zip_magic_rejected(self) -> None:
        with pytest.raises(AttachmentParseError) as exc:
            stream_and_parse(
                fetch=lambda: b"PK\x03\x04\x14\x00\x00\x00stuff",
                filename="spoof.md",
                mime="text/markdown",
                source_id="msg-zip",
            )
        assert "binary magic" in str(exc.value)
        assert exc.value.source_id == "msg-zip"

    def test_text_high_replacement_ratio_rejected(self) -> None:
        # Random non-UTF-8 bytes that don't match any magic prefix in our
        # list — still mostly U+FFFD after decode, must be refused.
        garbage = bytes(range(0x80, 0xC0)) * 4  # 256 invalid lead bytes
        with pytest.raises(AttachmentParseError) as exc:
            stream_and_parse(
                fetch=lambda: garbage,
                filename="spoof.csv",
                mime="text/csv",
                source_id="msg-garbage",
            )
        assert "replacement characters" in str(exc.value)
        assert exc.value.source_id == "msg-garbage"

    def test_text_legitimate_utf8_unaffected(self) -> None:
        result = stream_and_parse(
            fetch=lambda: "Hello, world! 你好".encode("utf-8"),
            filename="hello.txt",
            mime="text/plain",
        )
        assert result.text == "Hello, world! 你好"

    def test_text_empty_unaffected(self) -> None:
        result = stream_and_parse(
            fetch=lambda: b"",
            filename="empty.txt",
            mime="text/plain",
        )
        assert result.text == ""

    def test_e2e_spoofed_text_falls_back_to_metadata_only(self) -> None:
        # End-to-end: a Gmail/Slack attachment surfaced as text/plain whose
        # payload is actually a PDF must raise AttachmentParseError so the
        # source adapter's existing fallback emits a metadata-only Document
        # instead of a wall of garbage chunks.  We verify by simulating the
        # caller's try/except: the raise is the contract.
        spoofed_pdf = _tiny_pdf_bytes("contents that should never be indexed")
        assert spoofed_pdf.startswith(b"%PDF-")

        emitted_metadata_only = False
        try:
            stream_and_parse(
                fetch=lambda: spoofed_pdf,
                filename="invoice.txt",
                mime="text/plain",
                source_id="gmail-msg-7",
            )
        except AttachmentParseError as exc:
            assert exc.source_id == "gmail-msg-7"
            emitted_metadata_only = True
        assert emitted_metadata_only


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


class TestStreamAndParseMemoryRelease:
    """The 'stream-and-forget' invariant: no reference to the bytes
    survives the call. We track the bytes object's refcount across the
    call — if it returns to baseline after ``gc.collect()``, no parser
    is holding the buffer. ``weakref`` would be cleaner but ``bytes`` is
    not weakref-able and a wrapper container's lifetime is independent
    of the bytes (the wrapper dies as soon as the fetch closure returns,
    regardless of whether the parser cached the underlying buffer).

    Covers PDF, image, and text/* paths so a future parser that retains
    a reference (module-level cache keyed by ``id()``, persistent-state
    object pinning the buffer) is caught on whichever branch introduces
    it."""

    def _assert_released(
        self,
        payload: bytes,
        *,
        filename: str,
        mime: str,
        **kwargs: object,
    ) -> None:
        gc.collect()
        baseline = sys.getrefcount(payload)
        stream_and_parse(
            fetch=lambda: payload,
            filename=filename,
            mime=mime,
            **kwargs,  # type: ignore[arg-type]
        )
        gc.collect()
        leaked = sys.getrefcount(payload) - baseline
        assert leaked == 0, (
            f"stream_and_parse retained {leaked} reference(s) to "
            f"{filename!r} bytes after returning"
        )

    def test_text_bytes_are_released_after_return(self) -> None:
        # 5 MiB so a leak would be obvious under memory pressure too.
        self._assert_released(
            b"x" * (5 * 1024 * 1024),
            filename="big.txt",
            mime="text/plain",
        )

    def test_pdf_bytes_are_released_after_return(self) -> None:
        # Real parseable PDF — exercises the pymupdf branch end-to-end.
        self._assert_released(
            _tiny_pdf_bytes("release me"),
            filename="release.pdf",
            mime="application/pdf",
        )

    def test_image_bytes_are_released_after_return(self) -> None:
        def vision(_data: bytes, _mime: str) -> _FakeVisionResult:
            return _FakeVisionResult(description="ok")

        self._assert_released(
            _tiny_png_bytes(16, 16),
            filename="release.png",
            mime="image/png",
            vision_extractor=vision,
        )


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
