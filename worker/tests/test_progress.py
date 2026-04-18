"""Tests for the pipeline progress reporters."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from worker.parsers.base import ParsedDocument
from worker.pipeline import Pipeline
from worker.pipeline.chunker import Chunk
from worker.pipeline.extractor import ExtractionResult, extract_chunks
from worker.pipeline.progress import (
    NullProgressReporter,
    ProgressReporter,
    RichProgressReporter,
)
from worker.pipeline.resolver import ResolutionResult
from worker.pipeline.writer import Writer


# ------------------------------------------------------------------
# Fakes
# ------------------------------------------------------------------


@dataclass
class _Event:
    kind: str
    args: tuple = ()


class RecordingReporter(ProgressReporter):
    """Records every method call so tests can assert ordering."""

    def __init__(self) -> None:
        self.events: list[_Event] = []

    def start_file(self, source_id: str, label: str, total_chunks: int) -> None:
        self.events.append(_Event("start", (source_id, label, total_chunks)))

    def set_stage(self, source_id: str, stage: str) -> None:
        self.events.append(_Event("stage", (source_id, stage)))

    def advance(self, source_id: str, n: int = 1) -> None:
        self.events.append(_Event("advance", (source_id, n)))

    def finish_file(self, source_id: str) -> None:
        self.events.append(_Event("finish", (source_id,)))

    def queue_depth(self, depth: int) -> None:
        self.events.append(_Event("queue", (depth,)))

    def stop(self) -> None:
        self.events.append(_Event("stop", ()))


# ------------------------------------------------------------------
# Null reporter
# ------------------------------------------------------------------


class TestNullProgressReporter:
    def test_all_methods_are_no_ops(self):
        r = NullProgressReporter()
        # Just exercise every method — none should raise.
        r.start_file("id", "label", 5)
        r.set_stage("id", "embed")
        r.advance("id")
        r.advance("id", 3)
        r.finish_file("id")
        r.queue_depth(7)
        r.stop()


# ------------------------------------------------------------------
# Extractor → on_chunk callback
# ------------------------------------------------------------------


class TestExtractorOnChunk:
    def test_on_chunk_fires_once_per_chunk(self):
        """The callback should fire exactly len(chunks) times."""
        registry = MagicMock()
        # First role lookup returns the model, second (fallback) raises.
        model = MagicMock()
        model.complete.return_value = MagicMock(
            tool_calls=[
                {
                    "function": {
                        "name": "extract_entities_and_triples",
                        "arguments": '{"entities": [], "triples": []}',
                    }
                }
            ],
            text="",
        )
        registry.for_role.side_effect = [model, KeyError("no fallback")]

        chunks = [Chunk(text=f"chunk {i}", index=i) for i in range(4)]
        called: list[int] = []
        results = extract_chunks(chunks, registry, on_chunk=lambda: called.append(1))
        assert len(results) == 4
        assert len(called) == 4

    def test_on_chunk_fires_even_on_failure(self):
        """A chunk that raises during LLM call should still tick the bar."""
        registry = MagicMock()
        model = MagicMock()
        # An empty-response failure → goes through fallback, which is absent.
        model.complete.return_value = MagicMock(tool_calls=[], text="")
        registry.for_role.side_effect = [model, KeyError("no fallback")]

        chunks = [Chunk(text="x", index=0), Chunk(text="y", index=1)]
        called: list[int] = []
        results = extract_chunks(chunks, registry, on_chunk=lambda: called.append(1))
        assert len(results) == 2
        assert all(r.failed for r in results)
        assert len(called) == 2  # callback fires regardless of success

    def test_on_chunk_omitted_is_safe(self):
        """No callback supplied → extractor still works."""
        registry = MagicMock()
        model = MagicMock()
        model.complete.return_value = MagicMock(
            tool_calls=[
                {
                    "function": {
                        "name": "extract_entities_and_triples",
                        "arguments": '{"entities": [], "triples": []}',
                    }
                }
            ],
            text="",
        )
        registry.for_role.side_effect = [model, KeyError("no fallback")]
        results = extract_chunks([Chunk(text="x", index=0)], registry)
        assert len(results) == 1


# ------------------------------------------------------------------
# Pipeline ↔ reporter wiring
# ------------------------------------------------------------------


def _doc(**overrides) -> ParsedDocument:
    defaults = dict(
        source_type="file",
        source_id="notes/test.md",
        operation="created",
        text="Hello world. This is a test document.",
        node_label="File",
        node_props={"name": "test.md", "path": "notes/test.md"},
    )
    defaults.update(overrides)
    return ParsedDocument(**defaults)


class TestPipelineProgressEvents:
    @patch("worker.pipeline.resolve_entities_from_registry")
    @patch("worker.pipeline.extract_chunks")
    @patch("worker.pipeline.embed_chunks")
    @patch("worker.pipeline.chunk_text")
    def test_full_text_pipeline_emits_expected_events(
        self, mock_chunk, mock_embed, mock_extract, mock_resolve
    ):
        reporter = RecordingReporter()
        registry = MagicMock()
        writer = MagicMock(spec=Writer)
        writer.fetch_existing_entities.return_value = []
        writer.fetch_candidate_entities.return_value = []
        pipeline = Pipeline(registry=registry, writer=writer, progress=reporter)

        chunks = [Chunk(text=f"c{i}", index=i) for i in range(3)]
        mock_chunk.return_value = chunks
        mock_embed.return_value = [(c.text, [0.1]) for c in chunks]

        # Simulate the extractor's on_chunk callbacks: fire once per chunk.
        def fake_extract(received_chunks, _registry, *, on_chunk=None):
            for _ in received_chunks:
                if on_chunk is not None:
                    on_chunk()
            return [ExtractionResult() for _ in received_chunks]

        mock_extract.side_effect = fake_extract
        mock_resolve.return_value = ResolutionResult(entities=[])

        pipeline.process(_doc())

        kinds = [e.kind for e in reporter.events]
        assert kinds[0] == "start"
        assert reporter.events[0].args == ("notes/test.md", "notes/test.md", 3)

        # Every stage should be announced in order.
        stages = [e.args[1] for e in reporter.events if e.kind == "stage"]
        assert stages == ["embed", "extract", "resolve", "write"]

        # Three advance events from the (faked) extractor.
        advances = [e for e in reporter.events if e.kind == "advance"]
        assert len(advances) == 3
        assert all(a.args == ("notes/test.md", 1) for a in advances)

        # Finish always fires last.
        assert reporter.events[-1].kind == "finish"
        assert reporter.events[-1].args == ("notes/test.md",)

    @patch("worker.pipeline.embed_chunks")
    @patch("worker.pipeline.chunk_text")
    def test_finish_called_even_when_embed_raises(self, mock_chunk, mock_embed):
        """Reporter.finish_file must run even if a stage explodes."""
        reporter = RecordingReporter()
        registry = MagicMock()
        writer = MagicMock(spec=Writer)
        writer.fetch_existing_entities.return_value = []
        pipeline = Pipeline(registry=registry, writer=writer, progress=reporter)

        mock_chunk.return_value = [Chunk(text="c0", index=0)]
        mock_embed.side_effect = RuntimeError("embed exploded")

        with pytest.raises(RuntimeError):
            pipeline.process(_doc())

        kinds = [e.kind for e in reporter.events]
        assert kinds[0] == "start"
        assert kinds[-1] == "finish"

    @patch("worker.pipeline.chunk_text")
    def test_no_chunks_skips_progress(self, mock_chunk):
        """Empty-chunk docs short-circuit before start_file is called."""
        reporter = RecordingReporter()
        registry = MagicMock()
        writer = MagicMock(spec=Writer)
        pipeline = Pipeline(registry=registry, writer=writer, progress=reporter)

        mock_chunk.return_value = []
        pipeline.process(_doc())

        # No file-scoped events should have been emitted.
        assert reporter.events == []


# ------------------------------------------------------------------
# RichProgressReporter — smoke test (no real TTY needed)
# ------------------------------------------------------------------


class TestRichProgressReporter:
    def test_lifecycle_smoke(self):
        """Rich reporter constructs, drives a file end-to-end, and stops."""
        # We don't need a real terminal — Rich's Console accepts any file
        # object.  The reporter writes to sys.stderr by default; pytest
        # captures it.
        r = RichProgressReporter()
        try:
            r.queue_depth(5)
            r.start_file("doc:1", "notes/long.md", total_chunks=4)
            r.set_stage("doc:1", "embed")
            r.set_stage("doc:1", "extract")
            for _ in range(4):
                r.advance("doc:1")
            r.finish_file("doc:1")
            r.queue_depth(0)
        finally:
            r.stop()

    def test_advance_unknown_file_is_safe(self):
        """Advancing or finishing an unknown source_id must not raise."""
        r = RichProgressReporter()
        try:
            r.advance("never-started")
            r.set_stage("never-started", "embed")
            r.finish_file("never-started")
        finally:
            r.stop()

    def test_stop_restores_log_handlers(self):
        """The reporter swaps stderr handlers on start and restores on stop."""
        root = logging.getLogger()
        original = list(root.handlers)
        # Ensure at least one stderr handler exists for the reporter to swap.
        sentinel = logging.StreamHandler(sys.stderr)
        root.addHandler(sentinel)
        try:
            r = RichProgressReporter()
            # Sentinel should have been removed during __init__.
            assert sentinel not in root.handlers
            r.stop()
            # Sentinel should now be back.
            assert sentinel in root.handlers
        finally:
            root.removeHandler(sentinel)
            root.handlers = original
