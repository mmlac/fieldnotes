"""Pipeline orchestrator: wire chunker → embedder → extractor → resolver → writer.

The Pipeline class is the central coordinator that ties all pipeline stages
together. It takes a ModelRegistry, Neo4j driver, and Qdrant client, and
orchestrates document processing through: chunking, embedding, entity/triple
extraction, entity resolution, and writing to Neo4j + Qdrant.

Image documents are routed through the async vision queue for non-blocking
processing. Vision results are fed back through the embed → resolve → write
path with DEPICTS edges linking Image nodes to extracted entities.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from neo4j import Driver

from worker.config import VisionConfig
from worker.models.resolver import ModelRegistry
from worker.parsers.base import ParsedDocument
from worker.pipeline.chunker import Chunk, chunk_text
from worker.pipeline.embedder import embed_chunks
from worker.pipeline.extractor import ExtractionResult, extract_chunks
from worker.pipeline.resolver import (
    ResolutionResult,
    resolve_entities_from_registry,
)
from worker.pipeline.vision import (
    extract_image_from_registry,
    vision_result_to_chunk,
    vision_result_to_entities,
)
from worker.pipeline.vision_queue import VisionQueue, VisionResult as VisionQueueResult
from worker.pipeline.writer import WriteUnit, Writer

logger = logging.getLogger(__name__)


class Pipeline:
    """Orchestrates document processing through all pipeline stages.

    Stages for text documents:
      1. Write GraphHints directly to Neo4j (via Writer)
      2. Chunk text → embed chunks → extract entities/triples → resolve → write

    Stages for image documents:
      1. Submit to async vision queue (non-blocking)
      2. Vision extraction → description + visible_text + entities
      3. Synthetic text chunk → embed → resolve → write with DEPICTS edges

    Deletions are forwarded directly to the Writer which removes data from
    both Neo4j and Qdrant.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        writer: Writer,
        vision_queue: VisionQueue | None = None,
    ) -> None:
        self._registry = registry
        self._writer = writer
        self._vision_queue = vision_queue

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

        # Image documents: route to vision queue if available
        if parsed_doc.image_bytes and not parsed_doc.text:
            if self._vision_queue is not None:
                import asyncio

                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self._vision_queue.submit(parsed_doc))
                except RuntimeError:
                    # No running event loop — run synchronously
                    asyncio.run(self._vision_queue.submit(parsed_doc))
            else:
                logger.debug(
                    "Skipping image-only document %s (vision queue not configured)",
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

    def __enter__(self) -> Pipeline:
        return self

    def __exit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: object) -> None:
        self.close()

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
        # Use full-text index pre-filtering to avoid loading all entities
        entity_names = [e["name"] for e in all_entities]
        existing = self._writer.fetch_candidate_entities(entity_names)
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

    # ------------------------------------------------------------------
    # Vision pipeline
    # ------------------------------------------------------------------

    def vision_process_fn(self, doc: ParsedDocument) -> VisionQueueResult:
        """Process function for VisionQueue: extract structured data from image.

        Called by VisionQueue workers in a thread pool. Takes a ParsedDocument
        with image_bytes, runs vision extraction, and returns a VisionQueueResult
        with synthetic text and entities ready for the result callback.
        """
        result = extract_image_from_registry(
            doc.image_bytes, self._registry, doc.mime_type
        )
        chunk = vision_result_to_chunk(result)
        entities = vision_result_to_entities(result)

        sha256 = hashlib.sha256(doc.image_bytes).hexdigest()

        return VisionQueueResult(
            source_id=doc.source_id,
            sha256=sha256,
            text=chunk.text if chunk else "",
            entities=entities,
        )

    def on_vision_result(self, result: VisionQueueResult) -> None:
        """Callback for VisionQueue: embed, resolve, and write vision results.

        Receives processed vision output and feeds it through the normal
        embed → resolve → write path. Creates DEPICTS edges from the Image
        node to each extracted entity.
        """
        if not result.text and not result.entities:
            logger.debug(
                "Empty vision result for %s — skipping", result.source_id
            )
            return

        # Build synthetic chunk from vision text and embed it
        chunks: list[Chunk] = []
        vectors: list[list[float]] = []
        if result.text:
            chunk = Chunk(text=result.text, index=0)
            chunks = [chunk]
            embedded = embed_chunks([chunk.text], self._registry)
            vectors = [vec for _, vec in embedded]

        # Resolve vision entities against existing graph
        depicts_entities: list[dict[str, Any]] = []
        if result.entities:
            vision_entity_names = [e["name"] for e in result.entities]
            existing = self._writer.fetch_candidate_entities(vision_entity_names)
            resolved = resolve_entities_from_registry(
                result.entities, existing, self._registry
            )
            depicts_entities = _resolved_to_entity_dicts(resolved)

        # Build a ParsedDocument for the Image source node
        doc = ParsedDocument(
            source_type="image",
            source_id=result.source_id,
            operation="modified",
            text=result.text,
            node_label="Image",
            node_props={
                "sha256": result.sha256,
                "vision_processed": True,
            },
        )

        # Write with DEPICTS edges (not MENTIONS) for vision-extracted entities
        unit = WriteUnit(
            doc=doc,
            chunks=chunks,
            vectors=vectors,
            depicts_entities=depicts_entities,
        )
        self._writer.write(unit)

        logger.info(
            "Vision processed %s: %d chunks, %d entities",
            result.source_id,
            len(chunks),
            len(depicts_entities),
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
