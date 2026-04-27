"""Tests for worker.parsers.attachments — stream_and_parse + build_parent_url."""

from __future__ import annotations

import gc
import os
import sys
import tempfile
import weakref
from unittest.mock import MagicMock

import pymupdf
import pytest

from worker.parsers.attachments import (
    AttachmentDownloadError,
    AttachmentParseError,
    ParsedAttachment,
    build_parent_url,
    stream_and_parse,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pdf_bytes(text: str = "hello attachment") -> bytes:
    """Build a tiny valid PDF in memory (no disk)."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    raw = doc.tobytes()
    doc.close()
    return raw


# ---------------------------------------------------------------------------
# stream_and_parse — happy paths
# ---------------------------------------------------------------------------


def test_stream_and_parse_pdf_returns_text_no_temp_files():
    pdf = _make_pdf_bytes("the quick brown fox")
    tmp_before = set(os.listdir(tempfile.gettempdir()))

    result = stream_and_parse(
        lambda: pdf, "doc.pdf", "application/pdf", source_id="parent-123"
    )

    assert isinstance(result, ParsedAttachment)
    assert "quick brown fox" in result.text
    assert result.description == ""
    assert result.ocr_text is None
    assert result.extracted_entities == []

    tmp_after = set(os.listdir(tempfile.gettempdir()))
    leaked = tmp_after - tmp_before
    assert not leaked, f"PDF parse leaked temp files: {leaked}"


def test_stream_and_parse_text_decodes_utf8():
    payload = "café résumé naïve".encode("utf-8")
    result = stream_and_parse(lambda: payload, "notes.txt", "text/plain")
    assert result.text == "café résumé naïve"


def test_stream_and_parse_text_replaces_invalid_utf8():
    payload = b"hello \xff\xfe world"
    result = stream_and_parse(lambda: payload, "x.txt", "text/plain")
    assert "hello" in result.text and "world" in result.text


def test_stream_and_parse_image_calls_vision_pipeline_and_no_temp_files(monkeypatch):
    """Vision routing produces text + description + ocr + GraphHints, no temp files."""
    from worker.pipeline import vision as vision_mod

    fake_result = vision_mod.VisionResult(
        description="A diagram of a steam engine",
        visible_text="GAS TOWN",
        entities=[
            {"name": "Steam Engine", "type": "Technology"},
            {"name": "GasTown", "type": "Project"},
            {"name": "", "type": "Concept"},  # ignored — empty name
        ],
    )

    captured: dict[str, object] = {}

    def fake_extract(image_bytes, registry, mime_type="image/png"):
        captured["bytes_len"] = len(image_bytes)
        captured["mime"] = mime_type
        captured["registry"] = registry
        return fake_result

    monkeypatch.setattr(
        "worker.pipeline.vision.extract_image_from_registry", fake_extract
    )

    registry = MagicMock(name="ModelRegistry")
    image_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 256

    tmp_before = set(os.listdir(tempfile.gettempdir()))
    result = stream_and_parse(
        lambda: image_bytes,
        "diagram.png",
        "image/png",
        model_registry=registry,
        source_id="parent-img-1",
    )
    tmp_after = set(os.listdir(tempfile.gettempdir()))
    assert not (tmp_after - tmp_before), "image parse leaked temp files"

    assert captured["bytes_len"] == len(image_bytes)
    assert captured["mime"] == "image/png"
    assert captured["registry"] is registry

    assert result.description == "A diagram of a steam engine"
    assert result.ocr_text == "GAS TOWN"
    assert "A diagram of a steam engine" in result.text
    assert "GAS TOWN" in result.text

    # Two valid hints (the empty-named one was filtered).
    assert len(result.extracted_entities) == 2
    h1, h2 = result.extracted_entities
    assert h1.subject_id == "parent-img-1"
    assert h1.subject_label == "File"
    assert h1.predicate == "MENTIONS"
    assert h1.object_label == "Technology"
    assert h1.object_id == "technology:Steam Engine"
    assert h1.object_props == {"name": "Steam Engine"}
    assert h1.confidence == 0.80
    assert h2.object_label == "Project"


# ---------------------------------------------------------------------------
# stream_and_parse — error semantics
# ---------------------------------------------------------------------------


def test_stream_and_parse_fetch_failure_raises_download_error():
    def fetch():
        raise ConnectionError("boom")

    with pytest.raises(AttachmentDownloadError) as exc_info:
        stream_and_parse(fetch, "x.pdf", "application/pdf", source_id="parent-1")

    assert exc_info.value.source_id == "parent-1"
    assert "x.pdf" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, ConnectionError)


def test_stream_and_parse_unsupported_mime_raises_parse_error():
    """Even MIMEs that might appear on an indexable list must fail loudly here."""
    with pytest.raises(AttachmentParseError) as exc_info:
        stream_and_parse(
            lambda: b"\x50\x4b\x03\x04",  # ZIP magic
            "weirdo.zip",
            "application/zip",
            source_id="parent-z",
        )
    assert exc_info.value.source_id == "parent-z"
    assert "weirdo.zip" in str(exc_info.value)


def test_stream_and_parse_pdf_bad_bytes_raises_parse_error():
    with pytest.raises(AttachmentParseError) as exc_info:
        stream_and_parse(
            lambda: b"not a pdf",
            "broken.pdf",
            "application/pdf",
            source_id="parent-pdf",
        )
    assert exc_info.value.source_id == "parent-pdf"


def test_stream_and_parse_image_without_registry_raises_parse_error():
    with pytest.raises(AttachmentParseError) as exc_info:
        stream_and_parse(
            lambda: b"\x89PNG\r\n\x1a\n",
            "x.png",
            "image/png",
            model_registry=None,
            source_id="parent-img",
        )
    assert exc_info.value.source_id == "parent-img"


def test_stream_and_parse_image_vision_failure_wraps_as_parse_error(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("model down")

    monkeypatch.setattr("worker.pipeline.vision.extract_image_from_registry", boom)

    with pytest.raises(AttachmentParseError) as exc_info:
        stream_and_parse(
            lambda: b"\x89PNG",
            "x.png",
            "image/png",
            model_registry=MagicMock(),
            source_id="parent-img-2",
        )
    assert exc_info.value.source_id == "parent-img-2"
    assert isinstance(exc_info.value.__cause__, RuntimeError)


# ---------------------------------------------------------------------------
# Memory release — bytes are not retained after return
# ---------------------------------------------------------------------------


class _BytesHolder:
    """Weakref-supporting holder. The bytes object itself isn't weakreffable
    (CPython built-in), so we wrap it and verify lifetime via the holder."""

    __slots__ = ("payload", "__weakref__")

    def __init__(self, payload: bytes) -> None:
        self.payload = payload


def test_stream_and_parse_releases_bytes_after_return():
    """5 MB byte payload is not retained by stream_and_parse after return."""
    payload = b"x" * (5 * 1024 * 1024)
    initial_refcount = sys.getrefcount(payload)

    holder_box: list[_BytesHolder | None] = [_BytesHolder(payload)]
    holder_ref = weakref.ref(holder_box[0])

    def fetch() -> bytes:
        # Hand bytes to stream_and_parse and break the holder's grip so the
        # holder is collectible as soon as the test drops its reference.
        h = holder_box[0]
        assert h is not None
        out = h.payload
        h.payload = None  # type: ignore[assignment]
        return out

    result = stream_and_parse(fetch, "big.txt", "text/plain")
    assert isinstance(result, ParsedAttachment)
    assert len(result.text) == len(payload)

    # Drop our holder reference and force collection.
    holder_box[0] = None
    gc.collect()
    assert holder_ref() is None, "holder lingered — something retained it"

    # And refcount on the bytes object itself returns to baseline.
    gc.collect()
    assert sys.getrefcount(payload) == initial_refcount, (
        "stream_and_parse retained a reference to the bytes payload"
    )


# ---------------------------------------------------------------------------
# build_parent_url — exact shape per source
# ---------------------------------------------------------------------------


def test_build_parent_url_gmail_shape():
    url = build_parent_url("gmail", thread_id="18a1b2c3d4e5f6")
    assert url == "https://mail.google.com/mail/?ui=2&view=cv&th=18a1b2c3d4e5f6"


def test_build_parent_url_slack_shape_strips_dot_in_ts():
    url = build_parent_url(
        "slack",
        team_domain="acme",
        channel_id="C012ABC",
        ts="1714180812.001234",
    )
    assert url == "https://acme.slack.com/archives/C012ABC/p1714180812001234"


def test_build_parent_url_calendar_passes_html_link_verbatim():
    link = "https://www.google.com/calendar/event?eid=abc123"
    assert build_parent_url("calendar", html_link=link) == link


def test_build_parent_url_unknown_source_raises():
    with pytest.raises(ValueError, match="unknown source_type"):
        build_parent_url("twitter", id="42")


def test_build_parent_url_gmail_missing_kwargs_raises():
    with pytest.raises(ValueError, match="thread_id"):
        build_parent_url("gmail")


def test_build_parent_url_slack_missing_kwargs_raises():
    with pytest.raises(ValueError, match="channel_id"):
        build_parent_url("slack", team_domain="acme", ts="1.0")


def test_build_parent_url_calendar_missing_kwargs_raises():
    with pytest.raises(ValueError, match="html_link"):
        build_parent_url("calendar")
