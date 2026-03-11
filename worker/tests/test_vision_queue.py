"""Tests for the vision async worker queue with SHA256 dedup."""

from __future__ import annotations

import asyncio
import hashlib
from unittest.mock import MagicMock

import pytest

from worker.config import VisionConfig
from worker.parsers.base import ParsedDocument
from worker.pipeline.vision_queue import (
    VisionQueue,
    VisionResult,
    VisionQueueStats,
    _compute_sha256,
    check_vision_processed_neo4j,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _image_doc(
    source_id: str = "vault/image.png",
    image_bytes: bytes = b"\x89PNG\r\n\x1a\n" + b"x" * 2048,
    **overrides,
) -> ParsedDocument:
    defaults = dict(
        source_type="obsidian",
        source_id=source_id,
        operation="created",
        text="",
        image_bytes=image_bytes,
        node_label="File",
        node_props={"name": "image.png"},
    )
    defaults.update(overrides)
    return ParsedDocument(**defaults)


def _make_queue(
    config: VisionConfig | None = None,
    dedup_checker: object | None = None,
    process_fn: object | None = None,
    result_callback: object | None = None,
) -> VisionQueue:
    config = config or VisionConfig(concurrency=1, queue_size=16)
    dedup_checker = dedup_checker or (lambda sha: False)
    process_fn = process_fn or (lambda doc: VisionResult(
        source_id=doc.source_id,
        sha256=hashlib.sha256(doc.image_bytes).hexdigest(),
        text="extracted text",
    ))
    result_callback = result_callback or (lambda r: None)
    return VisionQueue(
        config=config,
        dedup_checker=dedup_checker,
        process_fn=process_fn,
        result_callback=result_callback,
    )


# ------------------------------------------------------------------
# SHA256
# ------------------------------------------------------------------


class TestComputeSha256:
    def test_deterministic(self) -> None:
        data = b"hello world"
        assert _compute_sha256(data) == hashlib.sha256(data).hexdigest()

    def test_different_data(self) -> None:
        assert _compute_sha256(b"a") != _compute_sha256(b"b")


# ------------------------------------------------------------------
# Size filtering
# ------------------------------------------------------------------


class TestSizeFiltering:
    @pytest.mark.asyncio
    async def test_skip_too_small(self) -> None:
        cfg = VisionConfig(min_file_size_kb=10)  # 10 KB minimum
        q = _make_queue(config=cfg)
        doc = _image_doc(image_bytes=b"x" * 100)  # 100 bytes < 10 KB
        accepted = await q.submit(doc)
        assert not accepted
        assert q.stats.skipped_size == 1

    @pytest.mark.asyncio
    async def test_skip_too_large(self) -> None:
        cfg = VisionConfig(max_file_size_mb=1)  # 1 MB max
        q = _make_queue(config=cfg)
        doc = _image_doc(image_bytes=b"x" * (2 * 1024 * 1024))  # 2 MB
        accepted = await q.submit(doc)
        assert not accepted
        assert q.stats.skipped_size == 1

    @pytest.mark.asyncio
    async def test_accept_within_range(self) -> None:
        cfg = VisionConfig(min_file_size_kb=1, max_file_size_mb=10)
        q = _make_queue(config=cfg)
        doc = _image_doc(image_bytes=b"x" * 5000)  # 5 KB
        accepted = await q.submit(doc)
        assert accepted
        assert q.stats.submitted == 1


# ------------------------------------------------------------------
# Skip patterns
# ------------------------------------------------------------------


class TestSkipPatterns:
    @pytest.mark.asyncio
    async def test_skip_icon(self) -> None:
        q = _make_queue()
        doc = _image_doc(source_id="assets/icon-home.png")
        accepted = await q.submit(doc)
        assert not accepted
        assert q.stats.skipped_pattern == 1

    @pytest.mark.asyncio
    async def test_skip_avatar(self) -> None:
        q = _make_queue()
        doc = _image_doc(source_id="users/avatar_john.jpg")
        accepted = await q.submit(doc)
        assert not accepted
        assert q.stats.skipped_pattern == 1

    @pytest.mark.asyncio
    async def test_skip_favicon(self) -> None:
        q = _make_queue()
        doc = _image_doc(source_id="static/favicon.ico")
        accepted = await q.submit(doc)
        assert not accepted
        assert q.stats.skipped_pattern == 1

    @pytest.mark.asyncio
    async def test_case_insensitive(self) -> None:
        q = _make_queue()
        doc = _image_doc(source_id="assets/ICON_big.png")
        accepted = await q.submit(doc)
        assert not accepted

    @pytest.mark.asyncio
    async def test_no_match_passes(self) -> None:
        q = _make_queue()
        doc = _image_doc(source_id="notes/diagram.png")
        accepted = await q.submit(doc)
        assert accepted


# ------------------------------------------------------------------
# SHA256 deduplication
# ------------------------------------------------------------------


class TestSha256Dedup:
    @pytest.mark.asyncio
    async def test_in_memory_dedup(self) -> None:
        q = _make_queue()
        data = b"x" * 2048
        doc1 = _image_doc(source_id="a.png", image_bytes=data)
        doc2 = _image_doc(source_id="b.png", image_bytes=data)

        accepted1 = await q.submit(doc1)
        accepted2 = await q.submit(doc2)

        assert accepted1
        assert not accepted2
        assert q.stats.skipped_dedup == 1

    @pytest.mark.asyncio
    async def test_external_dedup_checker(self) -> None:
        data = b"already processed image" * 100  # >1 KB to pass size filter
        sha = hashlib.sha256(data).hexdigest()

        checker = MagicMock(return_value=True)
        q = _make_queue(dedup_checker=checker)
        doc = _image_doc(image_bytes=data)

        accepted = await q.submit(doc)
        assert not accepted
        assert q.stats.skipped_dedup == 1
        checker.assert_called_once_with(sha)

    @pytest.mark.asyncio
    async def test_different_images_both_accepted(self) -> None:
        q = _make_queue()
        doc1 = _image_doc(source_id="a.png", image_bytes=b"image-a" * 200)
        doc2 = _image_doc(source_id="b.png", image_bytes=b"image-b" * 200)

        assert await q.submit(doc1)
        assert await q.submit(doc2)


# ------------------------------------------------------------------
# No image_bytes
# ------------------------------------------------------------------


class TestNoImageBytes:
    @pytest.mark.asyncio
    async def test_reject_no_image_bytes(self) -> None:
        q = _make_queue()
        doc = _image_doc(image_bytes=None)
        accepted = await q.submit(doc)
        assert not accepted


# ------------------------------------------------------------------
# Worker processing (integration-style)
# ------------------------------------------------------------------


class TestWorkerProcessing:
    @pytest.mark.asyncio
    async def test_process_and_callback(self) -> None:
        results: list[VisionResult] = []

        def process_fn(doc: ParsedDocument) -> VisionResult:
            return VisionResult(
                source_id=doc.source_id,
                sha256=hashlib.sha256(doc.image_bytes).hexdigest(),
                text="description of image",
            )

        q = _make_queue(
            process_fn=process_fn,
            result_callback=lambda r: results.append(r),
        )
        await q.start()

        doc = _image_doc(image_bytes=b"x" * 2048)
        await q.submit(doc)

        # Give workers time to process
        await asyncio.sleep(0.5)
        await q.stop()

        assert len(results) == 1
        assert results[0].text == "description of image"
        assert q.stats.processed == 1

    @pytest.mark.asyncio
    async def test_failed_processing_increments_counter(self) -> None:
        def bad_process_fn(doc: ParsedDocument) -> VisionResult:
            raise RuntimeError("vision model unavailable")

        q = _make_queue(process_fn=bad_process_fn)
        await q.start()

        doc = _image_doc(image_bytes=b"x" * 2048)
        await q.submit(doc)

        await asyncio.sleep(0.5)
        await q.stop()

        assert q.stats.failed == 1
        assert q.stats.processed == 0

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        results: list[VisionResult] = []

        def process_fn(doc: ParsedDocument) -> VisionResult:
            return VisionResult(
                source_id=doc.source_id,
                sha256=hashlib.sha256(doc.image_bytes).hexdigest(),
                text="ok",
            )

        q = _make_queue(
            process_fn=process_fn,
            result_callback=lambda r: results.append(r),
        )

        async with q:
            await q.submit(_image_doc(image_bytes=b"x" * 2048))
            await asyncio.sleep(0.5)

        assert len(results) == 1


# ------------------------------------------------------------------
# Stats
# ------------------------------------------------------------------


class TestStats:
    @pytest.mark.asyncio
    async def test_stats_accumulate(self) -> None:
        checker = MagicMock(side_effect=[False, True])
        q = _make_queue(
            config=VisionConfig(skip_patterns=["icon"]),
            dedup_checker=checker,
        )

        # 1. Accepted
        await q.submit(_image_doc(source_id="a.png", image_bytes=b"x" * 2048))
        # 2. Skipped by pattern
        await q.submit(_image_doc(source_id="icon.png", image_bytes=b"y" * 2048))
        # 3. Skipped by dedup (checker returns True)
        await q.submit(_image_doc(source_id="b.png", image_bytes=b"z" * 2048))

        assert q.stats.submitted == 3
        assert q.stats.skipped_pattern == 1
        assert q.stats.skipped_dedup == 1


# ------------------------------------------------------------------
# Neo4j dedup checker factory
# ------------------------------------------------------------------


class TestCheckVisionProcessedNeo4j:
    def test_returns_true_when_found(self) -> None:
        mock_record = MagicMock()
        mock_record.__getitem__ = MagicMock(return_value=True)

        mock_result = MagicMock()
        mock_result.single.return_value = mock_record

        mock_session = MagicMock()
        mock_session.run.return_value = mock_result
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        factory = MagicMock(return_value=mock_session)
        checker = check_vision_processed_neo4j(factory)

        assert checker("abc123") is True

    def test_returns_false_when_not_found(self) -> None:
        mock_record = MagicMock()
        mock_record.__getitem__ = MagicMock(return_value=False)

        mock_result = MagicMock()
        mock_result.single.return_value = mock_record

        mock_session = MagicMock()
        mock_session.run.return_value = mock_result
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        factory = MagicMock(return_value=mock_session)
        checker = check_vision_processed_neo4j(factory)

        assert checker("abc123") is False
