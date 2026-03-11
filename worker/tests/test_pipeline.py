"""Tests for the pipeline orchestrator.

Uses unittest.mock to stub all pipeline stages so tests run without
external services or models.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, call

import pytest

from worker.parsers.base import GraphHint, ParsedDocument
from worker.pipeline import Pipeline, _resolved_to_entity_dicts
from worker.pipeline.chunker import Chunk
from worker.pipeline.extractor import ExtractionResult
from worker.pipeline.resolver import ResolvedEntity, ResolutionResult
from worker.pipeline.writer import WriteUnit, Writer


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


def _doc(**overrides) -> ParsedDocument:
    """Create a minimal ParsedDocument for testing."""
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


def _make_pipeline() -> tuple[Pipeline, MagicMock, MagicMock]:
    """Create a Pipeline with mocked registry and writer."""
    registry = MagicMock()
    writer = MagicMock(spec=Writer)
    pipeline = Pipeline(registry=registry, writer=writer)
    return pipeline, registry, writer


# ------------------------------------------------------------------
# Deletion
# ------------------------------------------------------------------


class TestDeletion:
    def test_deleted_doc_forwarded_to_writer(self):
        pipeline, _, writer = _make_pipeline()
        doc = _doc(operation="deleted")

        pipeline.process(doc)

        writer.write.assert_called_once()
        unit = writer.write.call_args[0][0]
        assert isinstance(unit, WriteUnit)
        assert unit.doc is doc
        assert unit.chunks == []
        assert unit.vectors == []

    def test_deleted_doc_skips_text_pipeline(self):
        pipeline, registry, _ = _make_pipeline()
        doc = _doc(operation="deleted", text="some text")

        with patch("worker.pipeline.chunk_text") as mock_chunk:
            pipeline.process(doc)
            mock_chunk.assert_not_called()


# ------------------------------------------------------------------
# Image-only (Phase 2 skip)
# ------------------------------------------------------------------


class TestImageOnly:
    def test_image_only_doc_skipped(self):
        pipeline, _, writer = _make_pipeline()
        doc = _doc(text="", image_bytes=b"\x89PNG")

        pipeline.process(doc)

        writer.write.assert_not_called()


# ------------------------------------------------------------------
# No-text, no-image documents
# ------------------------------------------------------------------


class TestNoContent:
    def test_empty_text_no_image_writes_source_only(self):
        pipeline, _, writer = _make_pipeline()
        doc = _doc(text="")

        pipeline.process(doc)

        writer.write.assert_called_once()
        unit = writer.write.call_args[0][0]
        assert unit.chunks == []
        assert unit.vectors == []


# ------------------------------------------------------------------
# Full text pipeline
# ------------------------------------------------------------------


class TestTextPipeline:
    @patch("worker.pipeline.resolve_entities_from_registry")
    @patch("worker.pipeline.extract_chunks")
    @patch("worker.pipeline.embed_chunks")
    @patch("worker.pipeline.chunk_text")
    def test_text_pipeline_wires_stages(
        self, mock_chunk, mock_embed, mock_extract, mock_resolve
    ):
        pipeline, registry, writer = _make_pipeline()
        doc = _doc()

        # Set up stage mocks
        chunks = [Chunk(text="Hello world.", index=0)]
        mock_chunk.return_value = chunks

        mock_embed.return_value = [("Hello world.", [0.1, 0.2, 0.3])]

        mock_extract.return_value = [
            ExtractionResult(
                entities=[{"name": "World", "type": "Concept", "confidence": 0.9}],
                triples=[{"subject": "Hello", "predicate": "greets", "object": "World"}],
            )
        ]

        mock_resolve.return_value = ResolutionResult(
            entities=[
                ResolvedEntity(name="World", type="Concept", confidence=0.9)
            ]
        )

        pipeline.process(doc)

        # Verify stage calls
        mock_chunk.assert_called_once_with(doc.text)
        mock_embed.assert_called_once_with(["Hello world."], registry)
        mock_extract.assert_called_once_with(chunks, registry)
        mock_resolve.assert_called_once()

        # Verify writer called with correct WriteUnit
        writer.write.assert_called_once()
        unit = writer.write.call_args[0][0]
        assert unit.doc is doc
        assert unit.chunks == chunks
        assert unit.vectors == [[0.1, 0.2, 0.3]]
        assert len(unit.entities) == 1
        assert unit.entities[0]["name"] == "World"
        assert len(unit.triples) == 1

    @patch("worker.pipeline.resolve_entities_from_registry")
    @patch("worker.pipeline.extract_chunks")
    @patch("worker.pipeline.embed_chunks")
    @patch("worker.pipeline.chunk_text")
    def test_empty_chunks_writes_source_only(
        self, mock_chunk, mock_embed, mock_extract, mock_resolve
    ):
        pipeline, _, writer = _make_pipeline()
        doc = _doc()

        mock_chunk.return_value = []

        pipeline.process(doc)

        # Should write source node + hints only, not call embed/extract
        mock_embed.assert_not_called()
        mock_extract.assert_not_called()
        writer.write.assert_called_once()
        unit = writer.write.call_args[0][0]
        assert unit.chunks == []

    @patch("worker.pipeline.resolve_entities_from_registry")
    @patch("worker.pipeline.extract_chunks")
    @patch("worker.pipeline.embed_chunks")
    @patch("worker.pipeline.chunk_text")
    def test_multiple_chunks_flatten_entities(
        self, mock_chunk, mock_embed, mock_extract, mock_resolve
    ):
        pipeline, registry, writer = _make_pipeline()
        doc = _doc()

        chunks = [
            Chunk(text="Chunk one.", index=0),
            Chunk(text="Chunk two.", index=1),
        ]
        mock_chunk.return_value = chunks
        mock_embed.return_value = [
            ("Chunk one.", [0.1, 0.2]),
            ("Chunk two.", [0.3, 0.4]),
        ]
        mock_extract.return_value = [
            ExtractionResult(
                entities=[{"name": "A", "type": "Person", "confidence": 0.8}],
                triples=[],
            ),
            ExtractionResult(
                entities=[{"name": "B", "type": "Concept", "confidence": 0.7}],
                triples=[{"subject": "A", "predicate": "knows", "object": "B"}],
            ),
        ]
        mock_resolve.return_value = ResolutionResult(
            entities=[
                ResolvedEntity(name="A", type="Person", confidence=0.8),
                ResolvedEntity(name="B", type="Concept", confidence=0.7),
            ]
        )

        pipeline.process(doc)

        # resolve_entities_from_registry should receive flattened entities
        call_args = mock_resolve.call_args
        extracted = call_args[0][0]
        assert len(extracted) == 2
        assert extracted[0]["name"] == "A"
        assert extracted[1]["name"] == "B"

        # Writer should get both entities and the triple
        unit = writer.write.call_args[0][0]
        assert len(unit.entities) == 2
        assert len(unit.triples) == 1


# ------------------------------------------------------------------
# Batch processing
# ------------------------------------------------------------------


class TestBatchProcess:
    def test_process_batch_calls_process_for_each(self):
        pipeline, _, writer = _make_pipeline()

        with patch.object(pipeline, "process") as mock_process:
            docs = [_doc(source_id=f"doc/{i}") for i in range(3)]
            pipeline.process_batch(docs)

            assert mock_process.call_count == 3


# ------------------------------------------------------------------
# _resolved_to_entity_dicts
# ------------------------------------------------------------------


class TestResolvedToEntityDicts:
    def test_new_entity(self):
        result = ResolutionResult(
            entities=[ResolvedEntity(name="Foo", type="Concept", confidence=0.8)]
        )
        dicts = _resolved_to_entity_dicts(result)
        assert dicts == [{"name": "Foo", "type": "Concept", "confidence": 0.8}]

    def test_merged_entity_uses_canonical_name(self):
        result = ResolutionResult(
            entities=[
                ResolvedEntity(
                    name="foo", type="Concept", confidence=0.9,
                    merged_into="Foo",
                )
            ]
        )
        dicts = _resolved_to_entity_dicts(result)
        assert dicts[0]["name"] == "Foo"

    def test_same_as_entity_keeps_original_name(self):
        result = ResolutionResult(
            entities=[
                ResolvedEntity(
                    name="Bar", type="Person", confidence=0.7,
                    same_as="BarAlias",
                )
            ]
        )
        dicts = _resolved_to_entity_dicts(result)
        assert dicts[0]["name"] == "Bar"


# ------------------------------------------------------------------
# Close
# ------------------------------------------------------------------


class TestClose:
    def test_close_delegates_to_writer(self):
        pipeline, _, writer = _make_pipeline()
        pipeline.close()
        writer.close.assert_called_once()
