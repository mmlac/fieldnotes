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

from worker.circuit_breaker import CircuitOpenError
from worker.metrics import (
    CHUNKS_EMBEDDED,
    DOCUMENTS_FAILED,
    DOCUMENTS_PROCESSED,
    ENTITIES_EXTRACTED,
    ENTITIES_RESOLVED,
    PIPELINE_DURATION,
    PIPELINE_TOTAL_DURATION,
    observe_duration,
)
from worker.models.resolver import ModelRegistry
from worker.parsers.base import ParsedDocument
from worker.pipeline.chunker import Chunk, chunk_text
from worker.pipeline.embedder import embed_chunks
from worker.pipeline.extractor import (
    ExtractionResult as ExtractionResult,
    extract_chunks,
)
from worker.pipeline.resolver import (
    ResolutionResult,
    resolve_entities_from_registry,
)
from worker.pipeline.vision import (
    extract_image_from_registry,
    vision_result_to_chunk,
    vision_result_to_entities,
)
from worker.pipeline.app_describer import (
    AppDescriptionCache,
    AppInfo,
    describe_apps,
)
from worker.pipeline.vision_queue import VisionQueue, VisionResult as VisionQueueResult
from worker.pipeline.writer import WriteUnit, Writer

logger = logging.getLogger(__name__)

MAX_CHUNKS_PER_DOC = 10_000


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
        self._app_desc_cache = AppDescriptionCache()

    def process(self, parsed_doc: ParsedDocument) -> None:
        """Process a single parsed document through the full pipeline.

        Parameters
        ----------
        parsed_doc:
            A normalised document from any adapter parser.
        """
        with observe_duration(PIPELINE_TOTAL_DURATION):
            self._process_inner(parsed_doc)

    def _process_inner(self, parsed_doc: ParsedDocument) -> None:
        """Inner process logic, wrapped by total duration timing."""
        # Enrich single app doc with LLM description if needed
        if (
            parsed_doc.source_type == "macos_apps"
            and parsed_doc.operation != "deleted"
            and not parsed_doc.node_props.get("description")
        ):
            try:
                self._enrich_app_descriptions([parsed_doc])
            except Exception:
                logger.debug(
                    "App description enrichment failed for %s",
                    parsed_doc.source_id,
                    exc_info=True,
                )

        # Deletions: remove from both stores and return early
        if parsed_doc.operation == "deleted":
            with observe_duration(PIPELINE_DURATION, stage="write"):
                self._writer.write(WriteUnit(doc=parsed_doc))
            DOCUMENTS_PROCESSED.labels(
                source_type=parsed_doc.source_type, operation="deleted"
            ).inc()
            logger.info("Deleted %s %s", parsed_doc.source_type, parsed_doc.source_id)
            return

        # Image documents: route to vision queue if available
        if parsed_doc.image_bytes and not parsed_doc.text:
            if self._vision_queue is not None:
                _run_coro_sync(self._vision_queue.submit(parsed_doc))
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
            with observe_duration(PIPELINE_DURATION, stage="write"):
                self._writer.write(WriteUnit(doc=parsed_doc))

    def process_batch(self, docs: list[ParsedDocument]) -> list[ParsedDocument]:
        """Process multiple documents sequentially with error isolation.

        Each document is processed independently — a failure in one document
        does not prevent processing of remaining documents.

        Documents that fail due to a circuit breaker being open are collected
        separately and returned for retry on the next pipeline run (they are
        not counted as permanent failures).

        Returns
        -------
        list[ParsedDocument]
            Documents that failed processing (empty list if all succeeded).
            Includes both permanent failures and circuit-breaker-deferred docs.
        """
        # Enrich app descriptions before processing
        try:
            self._enrich_app_descriptions(docs)
        except Exception:
            logger.warning(
                "App description enrichment failed — continuing without descriptions",
                exc_info=True,
            )

        failed: list[ParsedDocument] = []
        circuit_deferred: list[ParsedDocument] = []
        for doc in docs:
            try:
                self.process(doc)
            except CircuitOpenError as exc:
                logger.warning(
                    "Circuit breaker open (%s) — deferring %s %s for retry",
                    exc.breaker_name,
                    doc.source_type,
                    doc.source_id,
                )
                DOCUMENTS_FAILED.labels(
                    source_type=doc.source_type, stage="circuit_breaker"
                ).inc()
                circuit_deferred.append(doc)
            except Exception:
                logger.exception(
                    "Failed to process %s %s — skipping",
                    doc.source_type,
                    doc.source_id,
                )
                DOCUMENTS_FAILED.labels(
                    source_type=doc.source_type, stage="process"
                ).inc()
                failed.append(doc)

        if circuit_deferred:
            logger.warning(
                "Batch: %d/%d documents deferred (circuit breaker open) — queued for retry",
                len(circuit_deferred),
                len(docs),
            )
        if failed:
            logger.warning(
                "Batch complete: %d/%d documents failed",
                len(failed),
                len(docs),
            )

        # After batch processing, reconcile Person nodes across sources
        try:
            self._writer.reconcile_persons()
        except Exception:
            logger.warning(
                "Person reconciliation failed — will retry on next batch",
                exc_info=True,
            )

        # Cross-source entity resolution: link entities mentioned in
        # different source types (email, git, obsidian) via SAME_AS edges
        try:
            self._writer.resolve_cross_source_entities()
        except Exception:
            logger.warning(
                "Cross-source entity resolution failed — will retry on next batch",
                exc_info=True,
            )

        # Return all docs that need re-processing: permanent failures +
        # circuit-breaker-deferred docs.  Callers should re-submit deferred
        # docs on the next pipeline run when services recover.
        return failed + circuit_deferred

    def __enter__(self) -> Pipeline:
        return self

    def __exit__(
        self, exc_type: type | None, exc_val: BaseException | None, exc_tb: object
    ) -> None:
        self.close()

    def close(self) -> None:
        """Release resources held by the writer."""
        self._writer.close()

    def _process_text(self, doc: ParsedDocument) -> None:
        """Run the full text pipeline for a document with text content."""
        # 1. Chunk
        with observe_duration(PIPELINE_DURATION, stage="chunk"):
            chunks = chunk_text(doc.text)
        if len(chunks) > MAX_CHUNKS_PER_DOC:
            logger.warning(
                "Document %s produced %d chunks (limit %d) — truncating",
                doc.source_id,
                len(chunks),
                MAX_CHUNKS_PER_DOC,
            )
            chunks = chunks[:MAX_CHUNKS_PER_DOC]
        if not chunks:
            # No meaningful text to process — write source node + hints only
            with observe_duration(PIPELINE_DURATION, stage="write"):
                self._writer.write(WriteUnit(doc=doc))
            return

        chunk_texts = [c.text for c in chunks]

        # 2. Embed
        with observe_duration(PIPELINE_DURATION, stage="embed"):
            embedded = embed_chunks(chunk_texts, self._registry)
        vectors = [vec for _, vec in embedded]

        if len(vectors) != len(chunks):
            DOCUMENTS_FAILED.labels(source_type=doc.source_type, stage="embed").inc()
            raise RuntimeError(
                f"Embedding returned {len(vectors)} vectors for {len(chunks)} chunks "
                f"(doc {doc.source_id}) — refusing to proceed with mismatched data"
            )

        CHUNKS_EMBEDDED.inc(len(chunks))

        # 3. Extract entities and triples
        with observe_duration(PIPELINE_DURATION, stage="extract"):
            extraction_results = extract_chunks(chunks, self._registry)

        # Check for extraction failures
        failed_count = sum(1 for r in extraction_results if r.failed)
        if failed_count:
            logger.warning(
                "Extraction failed for %d/%d chunks in %s %s",
                failed_count,
                len(extraction_results),
                doc.source_type,
                doc.source_id,
            )
            if failed_count == len(extraction_results):
                DOCUMENTS_FAILED.labels(
                    source_type=doc.source_type, stage="extract"
                ).inc()
                raise RuntimeError(
                    f"All {failed_count} extraction chunks failed for "
                    f"{doc.source_type} {doc.source_id}"
                )

        # Flatten all entities and triples across chunks (skip failed)
        all_entities: list[dict[str, Any]] = []
        all_triples: list[dict[str, str]] = []
        for result in extraction_results:
            if not result.failed:
                all_entities.extend(result.entities)
                all_triples.extend(result.triples)

        ENTITIES_EXTRACTED.inc(len(all_entities))

        # 4. Resolve entities against existing entities in Neo4j
        # Use full-text index pre-filtering to avoid loading all entities
        with observe_duration(PIPELINE_DURATION, stage="resolve"):
            entity_names = [e["name"] for e in all_entities]
            existing = self._writer.fetch_candidate_entities(entity_names)
            resolved = resolve_entities_from_registry(
                all_entities, existing, self._registry
            )

        # Build final entity list from resolved results
        write_entities = _resolved_to_entity_dicts(resolved)
        resolved_count = sum(1 for e in resolved.entities if e.merged_into)
        ENTITIES_RESOLVED.inc(resolved_count)

        # 5. Write everything to Neo4j + Qdrant
        unit = WriteUnit(
            doc=doc,
            chunks=chunks,
            vectors=vectors,
            entities=write_entities,
            triples=all_triples,
        )
        with observe_duration(PIPELINE_DURATION, stage="write"):
            self._writer.write(unit)

        DOCUMENTS_PROCESSED.labels(
            source_type=doc.source_type, operation=doc.operation
        ).inc()

        logger.info(
            "Processed %s %s: %d chunks, %d entities, %d triples",
            doc.source_type,
            doc.source_id,
            len(chunks),
            len(write_entities),
            len(all_triples),
        )

    # ------------------------------------------------------------------
    # App description enrichment
    # ------------------------------------------------------------------

    def _enrich_app_descriptions(self, docs: list[ParsedDocument]) -> None:
        """Enrich macOS app documents with LLM-generated descriptions.

        Collects apps that have no description (not from brew, no cached
        description), generates descriptions via LLM in batches, and
        updates the documents' node_props and text in-place.

        Brew descriptions are never overwritten.
        """
        apps_needing_desc: list[tuple[ParsedDocument, AppInfo]] = []

        for doc in docs:
            if doc.source_type != "macos_apps":
                continue
            if doc.operation == "deleted":
                continue
            # Skip if already has a description (e.g. from brew or prior enrichment)
            if doc.node_props.get("description"):
                continue

            bundle_id = doc.node_props.get("bundle_id", "")
            if not bundle_id:
                continue

            apps_needing_desc.append(
                (
                    doc,
                    AppInfo(
                        bundle_id=bundle_id,
                        display_name=doc.node_props.get("name", ""),
                        category="",  # category not in node_props, check source_metadata
                        version=doc.node_props.get("version", ""),
                    ),
                )
            )

        if not apps_needing_desc:
            return

        # Batch describe via LLM (with cache)
        app_infos = [info for _, info in apps_needing_desc]
        descriptions = describe_apps(app_infos, self._registry, self._app_desc_cache)

        # Update documents in-place
        for doc, info in apps_needing_desc:
            desc = descriptions.get(info.bundle_id)
            if desc and desc != "Unknown application":
                doc.node_props["description"] = desc
                # Rebuild text to include description
                name = doc.node_props.get("name", "")
                version = doc.node_props.get("version", "")
                parts = [name]
                if version:
                    parts[0] = f"{name} (v{version})"
                parts.append(desc)
                doc.text = " — ".join(p for p in parts if p)
            elif desc == "Unknown application":
                doc.node_props["description"] = desc

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
            logger.debug("Empty vision result for %s — skipping", result.source_id)
            return

        # Build synthetic chunk from vision text and embed it
        chunks: list[Chunk] = []
        vectors: list[list[float]] = []
        if result.text:
            chunk = Chunk(text=result.text, index=0)
            chunks = [chunk]
            embedded = embed_chunks([chunk.text], self._registry)
            vectors = [vec for _, vec in embedded]

            if len(vectors) != len(chunks):
                raise RuntimeError(
                    f"Embedding returned {len(vectors)} vectors for {len(chunks)} "
                    f"chunks (vision doc {result.source_id})"
                )

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


def _run_coro_sync(coro: Any) -> None:
    """Run a coroutine from synchronous code, blocking until complete.

    If no event loop is running, uses ``asyncio.run()``.  If called from a
    thread that already has a running loop (e.g. inside an async web server),
    dispatches to a temporary worker thread to avoid deadlocking the loop.
    Exceptions always propagate to the caller.
    """
    import asyncio
    import concurrent.futures

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running event loop — safe to use asyncio.run()
        asyncio.run(coro)
    else:
        # Running event loop — submit from a worker thread to avoid deadlock
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(asyncio.run, coro).result()


def _resolved_to_entity_dicts(resolved: ResolutionResult) -> list[dict[str, Any]]:
    """Convert resolved entities back to dicts for the writer.

    Merged entities use the canonical name. SAME_AS entities are kept with
    their original name (the writer will create SAME_AS edges separately).
    """
    entities: list[dict[str, Any]] = []
    for ent in resolved.entities:
        name = ent.merged_into if ent.merged_into else ent.name
        entities.append(
            {
                "name": name,
                "type": ent.type,
                "confidence": ent.confidence,
            }
        )
    return entities
