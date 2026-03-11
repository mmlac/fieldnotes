"""Integration test: full pipeline from parsing through writing.

Stubs external services (Ollama, Neo4j, Qdrant) but exercises the real
parsing, chunking, extraction-result handling, and resolution logic.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from worker.config import Config, ModelConfig, ProviderConfig, RolesConfig
from worker.models.resolver import ModelRegistry
from worker.parsers.base import ParsedDocument, GraphHint
from worker.parsers.files import FileParser
from worker.pipeline import Pipeline
from worker.pipeline.chunker import Chunk, chunk_text
from worker.pipeline.extractor import ExtractionResult
from worker.pipeline.resolver import ResolvedEntity, ResolutionResult
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
