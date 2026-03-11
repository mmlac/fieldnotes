"""Pipeline orchestrator: wire chunker → embedder → extractor → resolver → writer.

The Pipeline class is the central coordinator that ties all pipeline stages
together. It takes a ModelRegistry, Neo4j driver, and Qdrant client, and
orchestrates document processing through: chunking, embedding, entity/triple
extraction, entity resolution, and writing to Neo4j + Qdrant.
"""

from __future__ import annotations

import logging
from typing import Any

from neo4j import Driver

from worker.models.resolver import ModelRegistry
from worker.parsers.base import ParsedDocument
from worker.pipeline.chunker import Chunk, chunk_text
from worker.pipeline.embedder import embed_chunks
from worker.pipeline.extractor import ExtractionResult, extract_chunks
from worker.pipeline.resolver import (
    ResolutionResult,
    resolve_entities_from_registry,
)
from worker.pipeline.writer import WriteUnit, Writer

logger = logging.getLogger(__name__)


class Pipeline:
    """Orchestrates document processing through all pipeline stages.

    Stages for text documents:
      1. Write GraphHints directly to Neo4j (via Writer)
      2. Chunk text → embed chunks → extract entities/triples → resolve → write

    Deletions are forwarded directly to the Writer which removes data from
    both Neo4j and Qdrant.

    Image documents (image_bytes) are Phase 2 — skipped for now.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        writer: Writer,
    ) -> None:
        self._registry = registry
        self._writer = writer

    def process(self, parsed_doc: ParsedDocument) -> None:
        """Process a single parsed document through the full pipeline.

        Parameters
        ----------
        parsed_doc:
            A normalised document from any adapter parser.
        """
        # Deletions: remove from both stores and return early
        if parsed_doc.operation == "deleted":
            self._writer.write(WriteUnit(doc=parsed_doc))
            logger.info(
                "Deleted %s %s", parsed_doc.source_type, parsed_doc.source_id
            )
            return

        # Image-only documents: Phase 2, skip for now
        if parsed_doc.image_bytes and not parsed_doc.text:
            logger.debug(
                "Skipping image-only document %s (Phase 2)",
                parsed_doc.source_id,
            )
            return

        # Text pipeline: chunk → embed → extract → resolve → write
        if parsed_doc.text:
            self._process_text(parsed_doc)
        else:
            # No text and no image — just write graph hints and source node
            self._writer.write(WriteUnit(doc=parsed_doc))

    def process_batch(self, docs: list[ParsedDocument]) -> list[ParsedDocument]:
        """Process multiple documents sequentially with error isolation.

        Each document is processed independently — a failure in one document
        does not prevent processing of remaining documents.

        Returns
        -------
        list[ParsedDocument]
            Documents that failed processing (empty list if all succeeded).
        """
        failed: list[ParsedDocument] = []
        for doc in docs:
            try:
                self.process(doc)
            except Exception:
                logger.exception(
                    "Failed to process %s %s — skipping",
                    doc.source_type,
                    doc.source_id,
                )
                failed.append(doc)
        if failed:
            logger.warning(
                "Batch complete: %d/%d documents failed",
                len(failed),
                len(docs),
            )
        return failed

    def close(self) -> None:
        """Release resources held by the writer."""
        self._writer.close()

    def _process_text(self, doc: ParsedDocument) -> None:
        """Run the full text pipeline for a document with text content."""
        # 1. Chunk
        chunks = chunk_text(doc.text)
        if not chunks:
            # No meaningful text to process — write source node + hints only
            self._writer.write(WriteUnit(doc=doc))
            return

        chunk_texts = [c.text for c in chunks]

        # 2. Embed
        embedded = embed_chunks(chunk_texts, self._registry)
        vectors = [vec for _, vec in embedded]

        # 3. Extract entities and triples
        extraction_results = extract_chunks(chunks, self._registry)

        # Flatten all entities and triples across chunks
        all_entities: list[dict[str, Any]] = []
        all_triples: list[dict[str, str]] = []
        for result in extraction_results:
            all_entities.extend(result.entities)
            all_triples.extend(result.triples)

        # 4. Resolve entities against existing entities in Neo4j
        existing = self._writer.fetch_existing_entities()
        resolved = resolve_entities_from_registry(
            all_entities, existing, self._registry
        )

        # Build final entity list from resolved results
        write_entities = _resolved_to_entity_dicts(resolved)

        # 5. Write everything to Neo4j + Qdrant
        unit = WriteUnit(
            doc=doc,
            chunks=chunks,
            vectors=vectors,
            entities=write_entities,
            triples=all_triples,
        )
        self._writer.write(unit)

        logger.info(
            "Processed %s %s: %d chunks, %d entities, %d triples",
            doc.source_type,
            doc.source_id,
            len(chunks),
            len(write_entities),
            len(all_triples),
        )


def _resolved_to_entity_dicts(resolved: ResolutionResult) -> list[dict[str, Any]]:
    """Convert resolved entities back to dicts for the writer.

    Merged entities use the canonical name. SAME_AS entities are kept with
    their original name (the writer will create SAME_AS edges separately).
    """
    entities: list[dict[str, Any]] = []
    for ent in resolved.entities:
        name = ent.merged_into if ent.merged_into else ent.name
        entities.append({
            "name": name,
            "type": ent.type,
            "confidence": ent.confidence,
        })
    return entities
