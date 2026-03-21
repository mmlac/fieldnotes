"""Tests for the pipeline orchestrator.

Uses unittest.mock to stub all pipeline stages so tests run without
external services or models.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


from worker.parsers.base import ParsedDocument
from worker.pipeline import Pipeline, _resolved_to_entity_dicts
from worker.pipeline.chunker import Chunk
from worker.pipeline.extractor import ExtractionResult
from worker.pipeline.resolver import ResolvedEntity, ResolutionResult
from worker.pipeline.vision import VisionResult as VisionModuleResult
from worker.pipeline.vision_queue import VisionResult as VisionQueueResult
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
    writer.fetch_existing_entities.return_value = []
    writer.fetch_candidate_entities.return_value = []
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
# Image documents: route to vision queue
# ------------------------------------------------------------------


class TestImageRouting:
    def test_image_only_doc_not_written_directly(self):
        """Image-only docs go to vision queue, not directly to writer."""
        pipeline, _, writer = _make_pipeline()
        doc = _doc(text="", image_bytes=b"\x89PNG")

        pipeline.process(doc)

        writer.write.assert_not_called()

    def test_image_doc_submitted_to_vision_queue(self):
        """Image docs are submitted to the vision queue when available."""

        async def _fake_submit(doc):
            return True

        registry = MagicMock()
        writer = MagicMock(spec=Writer)
        vision_queue = MagicMock()
        vision_queue.submit = _fake_submit

        pipeline = Pipeline(registry=registry, writer=writer, vision_queue=vision_queue)
        doc = _doc(text="", image_bytes=b"\x89PNG")

        with patch("asyncio.get_running_loop", side_effect=RuntimeError):
            with patch("asyncio.run") as mock_run:
                pipeline.process(doc)
                mock_run.assert_called_once()

    def test_image_with_text_uses_text_pipeline(self):
        """Docs with both text and image_bytes use the text pipeline."""
        pipeline, _, writer = _make_pipeline()
        doc = _doc(text="Some text", image_bytes=b"\x89PNG")

        with patch("worker.pipeline.chunk_text", return_value=[]):
            pipeline.process(doc)

        # Goes through text pipeline (chunk_text returns [], so source-only write)
        writer.write.assert_called_once()


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
                triples=[
                    {"subject": "Hello", "predicate": "greets", "object": "World"}
                ],
            )
        ]

        mock_resolve.return_value = ResolutionResult(
            entities=[ResolvedEntity(name="World", type="Concept", confidence=0.9)]
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
    def test_existing_entities_passed_to_resolver(
        self, mock_chunk, mock_embed, mock_extract, mock_resolve
    ):
        """Existing entities from Neo4j should be forwarded to the resolver."""
        pipeline, registry, writer = _make_pipeline()
        doc = _doc()

        existing = [{"name": "Alice", "type": "Person", "confidence": 0.95}]
        writer.fetch_candidate_entities.return_value = existing

        chunks = [Chunk(text="Hello Alice.", index=0)]
        mock_chunk.return_value = chunks
        mock_embed.return_value = [("Hello Alice.", [0.1, 0.2, 0.3])]
        mock_extract.return_value = [
            ExtractionResult(
                entities=[{"name": "alice", "type": "Person", "confidence": 0.8}],
                triples=[],
            )
        ]
        mock_resolve.return_value = ResolutionResult(
            entities=[
                ResolvedEntity(
                    name="Alice",
                    type="Person",
                    confidence=0.95,
                    merged_into="Alice",
                )
            ]
        )

        pipeline.process(doc)

        # The resolver should receive candidate entities from Neo4j
        call_args = mock_resolve.call_args[0]
        assert call_args[1] is existing
        # Verify fetch_candidate_entities was called with extracted entity names
        writer.fetch_candidate_entities.assert_called_once_with(["alice"])

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

    def test_process_batch_isolates_errors(self):
        """One document failure must not prevent processing of remaining docs."""
        pipeline, _, writer = _make_pipeline()

        docs = [_doc(source_id=f"doc/{i}") for i in range(3)]

        call_count = 0

        def side_effect(doc):
            nonlocal call_count
            call_count += 1
            if doc.source_id == "doc/1":
                raise RuntimeError("simulated failure")

        with patch.object(pipeline, "process", side_effect=side_effect):
            failed = pipeline.process_batch(docs)

        # All three documents were attempted
        assert call_count == 3
        # Only the failing document is returned
        assert len(failed) == 1
        assert failed[0].source_id == "doc/1"

    def test_process_batch_returns_empty_on_success(self):
        pipeline, _, writer = _make_pipeline()

        with patch.object(pipeline, "process"):
            docs = [_doc(source_id=f"doc/{i}") for i in range(2)]
            failed = pipeline.process_batch(docs)

        assert failed == []


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
                    name="foo",
                    type="Concept",
                    confidence=0.9,
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
                    name="Bar",
                    type="Person",
                    confidence=0.7,
                    same_as="BarAlias",
                )
            ]
        )
        dicts = _resolved_to_entity_dicts(result)
        assert dicts[0]["name"] == "Bar"


# ------------------------------------------------------------------
# Vision pipeline callbacks
# ------------------------------------------------------------------


class TestVisionProcessFn:
    @patch("worker.pipeline.extract_image_from_registry")
    def test_returns_vision_queue_result(self, mock_extract):
        pipeline, registry, _ = _make_pipeline()

        mock_extract.return_value = VisionModuleResult(
            description="A photo of a cat.",
            visible_text="MEOW",
            entities=[{"name": "Cat", "type": "Concept"}],
        )

        doc = _doc(
            text="",
            image_bytes=b"\x89PNG\x00" * 16,
            mime_type="image/png",
        )

        result = pipeline.vision_process_fn(doc)

        assert isinstance(result, VisionQueueResult)
        assert result.source_id == doc.source_id
        assert result.text == "A photo of a cat.\n\nMEOW"
        assert len(result.entities) == 1
        assert result.entities[0]["name"] == "Cat"
        assert result.entities[0]["confidence"] == 0.80
        assert len(result.sha256) == 64  # SHA256 hex digest

    @patch("worker.pipeline.extract_image_from_registry")
    def test_empty_vision_result(self, mock_extract):
        pipeline, _, _ = _make_pipeline()
        mock_extract.return_value = VisionModuleResult()

        doc = _doc(text="", image_bytes=b"\x89PNG\x00" * 16)
        result = pipeline.vision_process_fn(doc)

        assert result.text == ""
        assert result.entities == []


class TestOnVisionResult:
    @patch("worker.pipeline.resolve_entities_from_registry")
    @patch("worker.pipeline.embed_chunks")
    def test_processes_vision_result_through_embed_resolve_write(
        self, mock_embed, mock_resolve
    ):
        pipeline, registry, writer = _make_pipeline()

        mock_embed.return_value = [("A photo of a cat.\n\nMEOW", [0.1, 0.2, 0.3])]
        mock_resolve.return_value = ResolutionResult(
            entities=[ResolvedEntity(name="Cat", type="Concept", confidence=0.80)]
        )

        result = VisionQueueResult(
            source_id="images/cat.png",
            sha256="abc123" * 10 + "abcd",
            text="A photo of a cat.\n\nMEOW",
            entities=[{"name": "Cat", "type": "Concept", "confidence": 0.80}],
        )

        pipeline.on_vision_result(result)

        # Verify embed was called with synthetic chunk text
        mock_embed.assert_called_once_with(["A photo of a cat.\n\nMEOW"], registry)

        # Verify resolve was called with vision entities
        mock_resolve.assert_called_once()

        # Verify writer was called with DEPICTS entities
        writer.write.assert_called_once()
        unit = writer.write.call_args[0][0]
        assert unit.doc.node_label == "Image"
        assert unit.doc.source_id == "images/cat.png"
        assert unit.doc.node_props["vision_processed"] is True
        assert len(unit.chunks) == 1
        assert unit.vectors == [[0.1, 0.2, 0.3]]
        assert len(unit.depicts_entities) == 1
        assert unit.depicts_entities[0]["name"] == "Cat"
        # DEPICTS entities, not regular entities
        assert unit.entities == []

    def test_empty_result_skipped(self):
        pipeline, _, writer = _make_pipeline()

        result = VisionQueueResult(
            source_id="images/empty.png",
            sha256="abc123" * 10 + "abcd",
            text="",
            entities=[],
        )

        pipeline.on_vision_result(result)
        writer.write.assert_not_called()

    @patch("worker.pipeline.embed_chunks")
    def test_text_only_no_entities(self, mock_embed):
        pipeline, _, writer = _make_pipeline()
        mock_embed.return_value = [("desc", [0.5, 0.6])]

        result = VisionQueueResult(
            source_id="images/simple.png",
            sha256="def456" * 10 + "defg",
            text="desc",
            entities=[],
        )

        pipeline.on_vision_result(result)

        writer.write.assert_called_once()
        unit = writer.write.call_args[0][0]
        assert len(unit.chunks) == 1
        assert unit.depicts_entities == []

    @patch("worker.pipeline.resolve_entities_from_registry")
    def test_entities_only_no_text(self, mock_resolve):
        pipeline, _, writer = _make_pipeline()
        mock_resolve.return_value = ResolutionResult(
            entities=[ResolvedEntity(name="Alice", type="Person", confidence=0.80)]
        )

        result = VisionQueueResult(
            source_id="images/person.png",
            sha256="ghi789" * 10 + "ghij",
            text="",
            entities=[{"name": "Alice", "type": "Person", "confidence": 0.80}],
        )

        pipeline.on_vision_result(result)

        writer.write.assert_called_once()
        unit = writer.write.call_args[0][0]
        assert unit.chunks == []
        assert unit.vectors == []
        assert len(unit.depicts_entities) == 1


# ------------------------------------------------------------------
# Close
# ------------------------------------------------------------------


class TestClose:
    def test_close_delegates_to_writer(self):
        pipeline, _, writer = _make_pipeline()
        pipeline.close()
        writer.close.assert_called_once()
