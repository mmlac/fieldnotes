"""Integration test: full pipeline from parsing through writing.

Stubs external services (Ollama, Neo4j, Qdrant) but exercises the real
parsing, chunking, extraction-result handling, and resolution logic.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from worker.config import Config, ModelConfig, ProviderConfig, RolesConfig, VisionConfig
from worker.models.resolver import ModelRegistry
from worker.parsers.base import ParsedDocument, GraphHint
from worker.parsers.files import FileParser
from worker.parsers.obsidian import ObsidianParser
from worker.pipeline import Pipeline
from worker.pipeline.chunker import Chunk, chunk_text
from worker.pipeline.extractor import ExtractionResult
from worker.pipeline.resolver import ResolvedEntity, ResolutionResult
from worker.pipeline.vision import VisionResult
from worker.pipeline.vision_queue import VisionResult as VisionQueueResult
from worker.pipeline.writer import WriteUnit, Writer

# Ensure provider decorators run
import worker.models.providers  # noqa: F401


def _make_config() -> Config:
    """Build a minimal Config for the pipeline."""
    cfg = Config()
    cfg.providers["local"] = ProviderConfig(
        name="local", type="ollama", settings={},
    )
    cfg.models["llm"] = ModelConfig(alias="llm", provider="local", model="qwen3.5:27b")
    cfg.models["embedder"] = ModelConfig(alias="embedder", provider="local", model="nomic-embed-text")
    cfg.roles = RolesConfig(mapping={
        "extraction": "llm",
        "embedding": "embedder",
    })
    return cfg


class TestEndToEndTextPipeline:
    """Parse a text file → chunk → embed → extract → resolve → write."""

    @patch("worker.pipeline.resolve_entities_from_registry")
    @patch("worker.pipeline.extract_chunks")
    @patch("worker.pipeline.embed_chunks")
    def test_text_file_full_pipeline(
        self, mock_embed, mock_extract, mock_resolve,
    ) -> None:
        # --- Setup ---
        registry = ModelRegistry(_make_config())
        writer = MagicMock(spec=Writer)
        writer.fetch_existing_entities.return_value = []

        pipeline = Pipeline(registry=registry, writer=writer)

        # 1. Parse: FileParser produces a ParsedDocument
        parser = FileParser()
        event = {
            "mime_type": "text/markdown",
            "source_id": "notes/test.md",
            "operation": "created",
            "text": "The quick brown fox jumped over the lazy dog. " * 20,
        }
        docs = parser.parse(event)
        assert len(docs) == 1
        doc = docs[0]

        # 2. Mock embeddings — return one vector per chunk
        def fake_embed(texts, reg):
            return [(t, [0.1] * 768) for t in texts]
        mock_embed.side_effect = fake_embed

        # 3. Mock extraction — return one entity per chunk
        def fake_extract(chunks, reg):
            return [
                ExtractionResult(
                    entities=[{"name": "Fox", "type": "Animal", "confidence": 0.9}],
                    triples=[{"subject": "Fox", "predicate": "JUMPED_OVER", "object": "Dog"}],
                )
                for _ in chunks
            ]
        mock_extract.side_effect = fake_extract

        # 4. Mock resolution — pass through entities
        mock_resolve.return_value = ResolutionResult(
            entities=[
                ResolvedEntity(name="Fox", type="Animal", confidence=0.9, merged_into=None),
                ResolvedEntity(name="Dog", type="Animal", confidence=0.9, merged_into=None),
            ],
        )

        # --- Execute ---
        pipeline.process(doc)

        # --- Verify ---
        # Writer should have been called once with a WriteUnit
        writer.write.assert_called_once()
        unit: WriteUnit = writer.write.call_args[0][0]

        assert unit.doc.source_id == "notes/test.md"
        assert unit.chunks is not None
        assert len(unit.chunks) >= 1
        assert unit.vectors is not None
        assert len(unit.vectors) == len(unit.chunks)
        assert all(len(v) == 768 for v in unit.vectors)
        assert unit.entities is not None
        assert any(e["name"] == "Fox" for e in unit.entities)
        assert unit.triples is not None
        assert any(t["predicate"] == "JUMPED_OVER" for t in unit.triples)


class TestEndToEndDeletion:
    """Delete operation goes straight to writer without chunking/extraction."""

    def test_delete_skips_pipeline_stages(self) -> None:
        registry = ModelRegistry(_make_config())
        writer = MagicMock(spec=Writer)
        pipeline = Pipeline(registry=registry, writer=writer)

        doc = ParsedDocument(
            source_type="files",
            source_id="notes/deleted.md",
            operation="deleted",
            text="",
        )
        pipeline.process(doc)

        writer.write.assert_called_once()
        unit: WriteUnit = writer.write.call_args[0][0]
        assert unit.doc.operation == "deleted"
        # No chunks/vectors/entities for deletions (defaults to empty lists)
        assert unit.chunks == []
        assert unit.vectors == []


class TestEndToEndBatch:
    """process_batch isolates failures between documents."""

    @patch("worker.pipeline.resolve_entities_from_registry")
    @patch("worker.pipeline.extract_chunks")
    @patch("worker.pipeline.embed_chunks")
    def test_batch_isolates_failure(
        self, mock_embed, mock_extract, mock_resolve,
    ) -> None:
        registry = ModelRegistry(_make_config())
        writer = MagicMock(spec=Writer)
        writer.fetch_existing_entities.return_value = []
        pipeline = Pipeline(registry=registry, writer=writer)

        # First doc will fail (embed raises), second should succeed as deletion
        mock_embed.side_effect = RuntimeError("embed failed")

        doc_fail = ParsedDocument(
            source_type="files",
            source_id="fail.md",
            operation="created",
            text="Some text here to chunk.",
        )
        doc_ok = ParsedDocument(
            source_type="files",
            source_id="ok.md",
            operation="deleted",
            text="",
        )

        failed = pipeline.process_batch([doc_fail, doc_ok])

        assert len(failed) == 1
        assert failed[0].source_id == "fail.md"
        # The deletion should have succeeded
        assert any(
            call[0][0].doc.source_id == "ok.md"
            for call in writer.write.call_args_list
        )


class TestEndToEndGraphHints:
    """Documents with graph_hints get them passed through to the writer."""

    @patch("worker.pipeline.resolve_entities_from_registry")
    @patch("worker.pipeline.extract_chunks")
    @patch("worker.pipeline.embed_chunks")
    def test_graph_hints_preserved(
        self, mock_embed, mock_extract, mock_resolve,
    ) -> None:
        registry = ModelRegistry(_make_config())
        writer = MagicMock(spec=Writer)
        writer.fetch_existing_entities.return_value = []
        pipeline = Pipeline(registry=registry, writer=writer)

        mock_embed.side_effect = lambda texts, reg: [(t, [0.1] * 768) for t in texts]
        mock_extract.side_effect = lambda chunks, reg: [
            ExtractionResult(entities=[], triples=[]) for _ in chunks
        ]
        mock_resolve.return_value = ResolutionResult(entities=[])

        hint = GraphHint(
            subject_id="notes/a.md",
            subject_label="File",
            predicate="LINKS_TO",
            object_id="notes/b.md",
            object_label="File",
        )
        doc = ParsedDocument(
            source_type="files",
            source_id="notes/a.md",
            operation="created",
            text="Link to b. " * 50,
            graph_hints=[hint],
        )

        pipeline.process(doc)

        unit: WriteUnit = writer.write.call_args[0][0]
        assert len(unit.doc.graph_hints) == 1
        assert unit.doc.graph_hints[0].predicate == "LINKS_TO"


# --- Fake image bytes: minimal valid PNG header ---
_FAKE_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 2048  # >1KB to pass size filter


def _make_vision_config() -> Config:
    """Build a Config with vision role configured."""
    cfg = Config()
    cfg.providers["local"] = ProviderConfig(
        name="local", type="ollama", settings={},
    )
    cfg.models["llm"] = ModelConfig(alias="llm", provider="local", model="qwen3.5:27b")
    cfg.models["embedder"] = ModelConfig(alias="embedder", provider="local", model="nomic-embed-text")
    cfg.models["vision_model"] = ModelConfig(alias="vision_model", provider="local", model="llava:13b")
    cfg.roles = RolesConfig(mapping={
        "extraction": "llm",
        "embedding": "embedder",
        "vision": "vision_model",
    })
    return cfg


class TestEndToEndVision:
    """Full vision pipeline: Obsidian note with ![[image.png]] → parse → vision → embed → write.

    Exercises the complete flow:
      1. ObsidianParser extracts image ParsedDocument with image_bytes
      2. Pipeline routes image to vision_process_fn
      3. Vision extraction returns description + visible_text + entities
      4. on_vision_result embeds synthetic chunk, resolves entities, writes
      5. Writer receives WriteUnit with Image node, DEPICTS edges, ATTACHED_TO edge
    """

    @patch("worker.pipeline.resolve_entities_from_registry")
    @patch("worker.pipeline.extract_chunks")
    @patch("worker.pipeline.embed_chunks")
    @patch("worker.pipeline.extract_image_from_registry")
    def test_vision_full_pipeline(
        self, mock_vision, mock_embed, mock_extract, mock_resolve,
    ) -> None:
        # --- Setup ---
        registry = ModelRegistry(_make_vision_config())
        writer = MagicMock(spec=Writer)
        writer.fetch_existing_entities.return_value = []

        pipeline = Pipeline(registry=registry, writer=writer)

        # 1. Parse: ObsidianParser with an embedded image
        parser = ObsidianParser()
        note_text = "---\ntitle: Test Note\n---\nHere is a diagram:\n\n![[architecture.png]]\n\n" + "More text here. " * 30
        event = {
            "source_id": "notes/test-vision.md",
            "operation": "created",
            "text": note_text,
            "meta": {},
        }
        docs = parser.parse(event)

        # Should produce 2 docs: one text File, one image
        assert len(docs) == 2
        text_doc = docs[0]
        image_doc = docs[1]

        assert text_doc.node_label == "File"
        assert text_doc.text  # has body text
        assert image_doc.source_id == "architecture.png"
        assert image_doc.mime_type == "image/png"
        # image_bytes is None because no vault_path was given — inject manually
        image_doc = ParsedDocument(
            source_type=image_doc.source_type,
            source_id=image_doc.source_id,
            operation=image_doc.operation,
            text="",
            mime_type=image_doc.mime_type,
            node_label="Image",
            node_props={
                "embedded_in": text_doc.source_id,
                "parent_source_id": text_doc.source_id,
            },
            image_bytes=_FAKE_PNG,
            source_metadata=image_doc.source_metadata,
        )

        # 2. Mock vision extraction — return structured result
        mock_vision.return_value = VisionResult(
            description="Architecture diagram showing microservices connected via message queue",
            visible_text="API Gateway → Auth Service → Queue → Worker",
            entities=[
                {"name": "API Gateway", "type": "Technology"},
                {"name": "Auth Service", "type": "Technology"},
                {"name": "Message Queue", "type": "Technology"},
            ],
        )

        # 3. Mock embeddings
        def fake_embed(texts, reg):
            return [(t, [0.2] * 768) for t in texts]
        mock_embed.side_effect = fake_embed

        # 4. Mock extraction (for text doc)
        mock_extract.side_effect = lambda chunks, reg: [
            ExtractionResult(entities=[], triples=[]) for _ in chunks
        ]

        # 5. Mock resolution — pass through entities
        mock_resolve.return_value = ResolutionResult(
            entities=[
                ResolvedEntity(name="API Gateway", type="Technology", confidence=0.8, merged_into=None),
                ResolvedEntity(name="Auth Service", type="Technology", confidence=0.8, merged_into=None),
                ResolvedEntity(name="Message Queue", type="Technology", confidence=0.8, merged_into=None),
            ],
        )

        # --- Execute: process text doc normally ---
        pipeline.process(text_doc)

        # --- Execute: simulate vision pipeline for image doc ---
        # Step A: vision_process_fn extracts structured data from image
        vision_result = pipeline.vision_process_fn(image_doc)

        assert vision_result.source_id == image_doc.source_id
        assert vision_result.sha256 == hashlib.sha256(_FAKE_PNG).hexdigest()
        assert "Architecture diagram" in vision_result.text
        assert len(vision_result.entities) == 3

        # Step B: on_vision_result embeds, resolves, and writes
        pipeline.on_vision_result(vision_result)

        # --- Verify ---
        # Writer should have been called twice: once for text doc, once for image
        assert writer.write.call_count == 2

        # Find the vision write call (node_label == "Image")
        vision_unit: WriteUnit | None = None
        text_unit: WriteUnit | None = None
        for call in writer.write.call_args_list:
            unit: WriteUnit = call[0][0]
            if unit.doc.node_label == "Image":
                vision_unit = unit
            else:
                text_unit = unit

        assert vision_unit is not None, "Writer was not called with Image WriteUnit"
        assert text_unit is not None, "Writer was not called with text WriteUnit"

        # Image node properties
        assert vision_unit.doc.source_type == "image"
        assert vision_unit.doc.source_id == image_doc.source_id
        assert vision_unit.doc.node_props["sha256"] == hashlib.sha256(_FAKE_PNG).hexdigest()
        assert vision_unit.doc.node_props["vision_processed"] is True

        # Synthetic text chunk was embedded
        assert len(vision_unit.chunks) == 1
        assert "Architecture diagram" in vision_unit.chunks[0].text
        assert "API Gateway" in vision_unit.chunks[0].text
        assert len(vision_unit.vectors) == 1
        assert len(vision_unit.vectors[0]) == 768

        # DEPICTS entities (not MENTIONS) for vision-extracted entities
        assert len(vision_unit.depicts_entities) == 3
        entity_names = {e["name"] for e in vision_unit.depicts_entities}
        assert "API Gateway" in entity_names
        assert "Auth Service" in entity_names
        assert "Message Queue" in entity_names

        # Regular entities/triples should be empty for vision path
        assert vision_unit.entities == []
        assert vision_unit.triples == []

    @patch("worker.pipeline.resolve_entities_from_registry")
    @patch("worker.pipeline.extract_chunks")
    @patch("worker.pipeline.embed_chunks")
    @patch("worker.pipeline.extract_image_from_registry")
    def test_vision_empty_result_skips_write(
        self, mock_vision, mock_embed, mock_extract, mock_resolve,
    ) -> None:
        """Empty vision result (no description, no entities) should not write."""
        registry = ModelRegistry(_make_vision_config())
        writer = MagicMock(spec=Writer)
        pipeline = Pipeline(registry=registry, writer=writer)

        mock_vision.return_value = VisionResult()  # empty

        image_doc = ParsedDocument(
            source_type="obsidian",
            source_id="empty-image.png",
            operation="created",
            text="",
            mime_type="image/png",
            node_label="Image",
            image_bytes=_FAKE_PNG,
        )

        result = pipeline.vision_process_fn(image_doc)
        pipeline.on_vision_result(result)

        # Writer should NOT be called — empty vision result is skipped
        writer.write.assert_not_called()

    @patch("worker.pipeline.resolve_entities_from_registry")
    @patch("worker.pipeline.extract_chunks")
    @patch("worker.pipeline.embed_chunks")
    @patch("worker.pipeline.extract_image_from_registry")
    def test_vision_with_obsidian_parser_image_bytes(
        self, mock_vision, mock_embed, mock_extract, mock_resolve,
        tmp_path: Path,
    ) -> None:
        """ObsidianParser loads image_bytes from disk when vault_path is set."""
        # Create a fake image file in a temp vault
        (tmp_path / "photo.png").write_bytes(_FAKE_PNG)

        parser = ObsidianParser()
        event = {
            "source_id": "notes/photo-note.md",
            "operation": "created",
            "text": "---\n---\nCheck this photo:\n\n![[photo.png]]",
            "meta": {"vault_path": str(tmp_path)},
        }
        docs = parser.parse(event)

        # Text doc + image doc
        assert len(docs) == 2
        image_doc = docs[1]
        assert image_doc.image_bytes == _FAKE_PNG
        assert image_doc.mime_type == "image/png"

        # Now process through vision pipeline
        registry = ModelRegistry(_make_vision_config())
        writer = MagicMock(spec=Writer)
        writer.fetch_existing_entities.return_value = []
        pipeline = Pipeline(registry=registry, writer=writer)

        mock_vision.return_value = VisionResult(
            description="A photograph of a sunset over mountains",
            visible_text="",
            entities=[{"name": "Mountains", "type": "Concept"}],
        )
        mock_embed.side_effect = lambda texts, reg: [(t, [0.3] * 768) for t in texts]
        mock_resolve.return_value = ResolutionResult(
            entities=[
                ResolvedEntity(name="Mountains", type="Concept", confidence=0.8, merged_into=None),
            ],
        )

        # Simulate vision pipeline
        result = pipeline.vision_process_fn(image_doc)
        assert result.sha256 == hashlib.sha256(_FAKE_PNG).hexdigest()

        pipeline.on_vision_result(result)

        writer.write.assert_called_once()
        unit: WriteUnit = writer.write.call_args[0][0]
        assert unit.doc.node_label == "Image"
        assert len(unit.depicts_entities) == 1
        assert unit.depicts_entities[0]["name"] == "Mountains"
        assert len(unit.chunks) == 1
        assert "sunset over mountains" in unit.chunks[0].text

    def test_image_doc_without_vision_queue_is_skipped(self) -> None:
        """Image doc with no vision queue configured gets skipped (not crashed)."""
        registry = ModelRegistry(_make_vision_config())
        writer = MagicMock(spec=Writer)
        pipeline = Pipeline(registry=registry, writer=writer, vision_queue=None)

        image_doc = ParsedDocument(
            source_type="obsidian",
            source_id="skip-me.png",
            operation="created",
            text="",
            mime_type="image/png",
            node_label="Image",
            image_bytes=_FAKE_PNG,
        )

        # Should not crash, just log and skip
        pipeline.process(image_doc)

        # Writer should NOT be called — image was skipped
        writer.write.assert_not_called()

    @patch("worker.pipeline.resolve_entities_from_registry")
    @patch("worker.pipeline.extract_chunks")
    @patch("worker.pipeline.embed_chunks")
    @patch("worker.pipeline.extract_image_from_registry")
    def test_vision_entities_use_depicts_not_mentions(
        self, mock_vision, mock_embed, mock_extract, mock_resolve,
    ) -> None:
        """Vision entities must go through depicts_entities (DEPICTS edge), not entities (MENTIONS)."""
        registry = ModelRegistry(_make_vision_config())
        writer = MagicMock(spec=Writer)
        writer.fetch_existing_entities.return_value = []
        pipeline = Pipeline(registry=registry, writer=writer)

        mock_vision.return_value = VisionResult(
            description="Screenshot of Claude chat",
            visible_text="Hello Claude",
            entities=[{"name": "Claude", "type": "Technology"}],
        )
        mock_embed.side_effect = lambda texts, reg: [(t, [0.1] * 768) for t in texts]
        mock_resolve.return_value = ResolutionResult(
            entities=[
                ResolvedEntity(name="Claude", type="Technology", confidence=0.8, merged_into=None),
            ],
        )

        image_doc = ParsedDocument(
            source_type="obsidian",
            source_id="screenshot.png",
            operation="created",
            text="",
            mime_type="image/png",
            node_label="Image",
            image_bytes=_FAKE_PNG,
        )

        result = pipeline.vision_process_fn(image_doc)
        pipeline.on_vision_result(result)

        unit: WriteUnit = writer.write.call_args[0][0]

        # DEPICTS entities present
        assert len(unit.depicts_entities) == 1
        assert unit.depicts_entities[0]["name"] == "Claude"

        # Regular MENTIONS entities empty
        assert unit.entities == []
