"""Tests for worker.parsers.attachments — pure classify_attachment helper."""

from __future__ import annotations

import pytest

from worker.parsers.attachments import (
    DEFAULT_INDEXABLE_MIMETYPES,
    classify_attachment,
)


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
