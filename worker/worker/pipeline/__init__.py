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

from collections import defaultdict

from worker.circuit_breaker import CircuitOpenError
from worker.metrics import (
    CHUNKS_EMBEDDED,
    DOCUMENTS_FAILED,
    DOCUMENTS_PROCESSED,
    ENTITIES_EXTRACTED,
    ENTITIES_RESOLVED,
    PIPELINE_DURATION,
    PIPELINE_SKIPPED,
    PIPELINE_TOTAL_DURATION,
    observe_duration,
)
from worker.models.resolver import ModelRegistry
from worker.parsers.base import ParsedDocument, extract_email_person_hints
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
from worker.pipeline.exif import apply_exif_to_doc
from worker.pipeline.geocode import (
    link_image_to_location,
    reverse_geocode_cached,
)
from worker.pipeline.app_describer import (
    AppDescriptionCache,
    AppInfo,
    describe_apps,
)
from worker.pipeline.progress import NullProgressReporter, ProgressReporter
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
        progress: ProgressReporter | None = None,
    ) -> None:
        self._registry = registry
        self._writer = writer
        self._vision_queue = vision_queue
        self._app_desc_cache = AppDescriptionCache()
        self._progress: ProgressReporter = progress or NullProgressReporter()

    def process(
        self,
        parsed_doc: ParsedDocument,
        *,
        already_indexed: bool | None = None,
        existing_hash: str | None = None,
    ) -> None:
        """Process a single parsed document through the full pipeline.

        Parameters
        ----------
        parsed_doc:
            A normalised document from any adapter parser.
        already_indexed:
            If True and ``operation == "created"``, skip the entire pipeline
            because Neo4j already has chunks for this ``source_id``.  When
            left at the default ``None``, the writer is queried directly
            (one indexed lookup) so single-item callers from the consumer
            loop also benefit from Phase 1 dedup — important for draining
            an existing queue backlog without re-doing work.
        existing_hash:
            Previously stored ``content_hash`` for this source node, if any.
            When set, ``_process_text`` will compare it against the SHA-256
            of ``doc.text`` and short-circuit on a match.  When omitted,
            ``process()`` performs its own per-doc lookup.

        For batch processing, prefer :meth:`process_batch` which pre-fetches
        both values for the entire batch in bounded round-trips.
        """
        # Phase-1 skip: created docs whose source_id is already chunked.
        # Modified/deleted ops always fall through (modified is handled by
        # the content-hash check; deletions must reach the writer).
        if already_indexed is None and parsed_doc.operation == "created":
            try:
                hit = self._writer.indexed_source_ids([parsed_doc.source_id])
                already_indexed = parsed_doc.source_id in hit
            except Exception:
                logger.debug(
                    "indexed_source_ids lookup failed for %s — processing anyway",
                    parsed_doc.source_id,
                    exc_info=True,
                )
                already_indexed = False

        if already_indexed and parsed_doc.operation == "created":
            PIPELINE_SKIPPED.labels(
                source_type=parsed_doc.source_type, reason="already_indexed"
            ).inc()
            logger.debug(
                "Skipping %s %s (already indexed)",
                parsed_doc.source_type,
                parsed_doc.source_id,
            )
            return

        # Phase 2 single-doc fallback: look up content_hash if the caller
        # didn't pre-fetch it.  Only for documents that have a label and
        # text (otherwise there's no hash to compare against).
        if (
            existing_hash is None
            and parsed_doc.text
            and parsed_doc.node_label
            and parsed_doc.operation != "deleted"
        ):
            try:
                hashes = self._writer.get_content_hashes(
                    {parsed_doc.node_label: [parsed_doc.source_id]}
                )
                existing_hash = hashes.get(parsed_doc.source_id)
            except Exception:
                logger.debug(
                    "get_content_hashes lookup failed for %s — processing anyway",
                    parsed_doc.source_id,
                    exc_info=True,
                )

        with observe_duration(PIPELINE_TOTAL_DURATION):
            self._process_inner(parsed_doc, existing_hash=existing_hash)

    def _process_inner(
        self,
        parsed_doc: ParsedDocument,
        *,
        existing_hash: str | None = None,
    ) -> None:
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

        # EXIF extraction: pull GPS coordinates and date from image metadata
        if parsed_doc.image_bytes:
            try:
                apply_exif_to_doc(parsed_doc.image_bytes, parsed_doc.node_props)
            except Exception:
                logger.debug(
                    "EXIF extraction failed for %s",
                    parsed_doc.source_id,
                    exc_info=True,
                )

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
            self._process_text(parsed_doc, existing_hash=existing_hash)
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

        # Pre-fetch dedup state for the entire batch in bounded round-trips:
        #   - indexed_source_ids → which created docs are already chunked
        #   - get_content_hashes → previously stored hashes for modified docs
        # See ``process()`` for the per-doc skip semantics.
        already_indexed, existing_hashes = self._prefetch_dedup_state(docs)

        failed: list[ParsedDocument] = []
        circuit_deferred: list[ParsedDocument] = []
        for doc in docs:
            try:
                # Pass explicit bool (not None) so process() doesn't redo
                # the per-doc lookup we already batched above.
                self.process(
                    doc,
                    already_indexed=(doc.source_id in already_indexed),
                    existing_hash=existing_hashes.get(doc.source_id),
                )
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

        # Fuzzy name matching for Person nodes (catches name variants)
        try:
            self._writer.reconcile_persons_by_name()
        except Exception:
            logger.warning(
                "Person name reconciliation failed — will retry on next batch",
                exc_info=True,
            )

        # Bridge LLM-extracted Entity(Person) nodes to structured Person nodes
        try:
            self._writer.bridge_entity_persons()
        except Exception:
            logger.warning(
                "Entity→Person bridging failed — will retry on next batch",
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

        # Transitive SAME_AS closure: if A↔B and B↔C, infer A↔C
        try:
            self._writer.close_same_as_transitive()
        except Exception:
            logger.warning(
                "Transitive SAME_AS closure failed — will retry on next batch",
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

    # ------------------------------------------------------------------
    # Dedup pre-fetch
    # ------------------------------------------------------------------

    def _prefetch_dedup_state(
        self, docs: list[ParsedDocument]
    ) -> tuple[set[str], dict[str, str]]:
        """Return ``(already_indexed_source_ids, existing_content_hashes)``.

        - ``already_indexed_source_ids`` is the subset of *created* docs
          whose source_id already has chunks in Neo4j.  Their pipeline
          run will short-circuit before any chunk/embed/extract work.
        - ``existing_content_hashes`` is ``source_id → stored content_hash``
          for non-skipped docs that have a node label.  Used by
          ``_process_text`` to detect unchanged content and skip the
          chunk → embed → extract path.

        Failures in either lookup are non-fatal: the pipeline falls back
        to processing every document, which is correct (just slower).
        """
        if not docs:
            return set(), {}

        already_indexed: set[str] = set()
        existing_hashes: dict[str, str] = {}

        try:
            created_sids = [
                d.source_id for d in docs if d.operation == "created"
            ]
            if created_sids:
                already_indexed = self._writer.indexed_source_ids(created_sids)
        except Exception:
            logger.warning(
                "indexed_source_ids pre-fetch failed; processing all docs",
                exc_info=True,
            )
            already_indexed = set()

        try:
            grouped: dict[str, list[str]] = defaultdict(list)
            for d in docs:
                # Skip docs we'll already short-circuit on (no need to
                # look up their hash).  Also skip docs without a label —
                # they have no source node to query.
                if d.operation == "created" and d.source_id in already_indexed:
                    continue
                if not d.node_label:
                    continue
                grouped[d.node_label].append(d.source_id)
            if grouped:
                existing_hashes = self._writer.get_content_hashes(grouped)
        except Exception:
            logger.warning(
                "get_content_hashes pre-fetch failed; processing all docs",
                exc_info=True,
            )
            existing_hashes = {}

        return already_indexed, existing_hashes

    def _process_text(
        self, doc: ParsedDocument, *, existing_hash: str | None = None
    ) -> None:
        """Run the full text pipeline for a document with text content."""
        # 0. Extract email addresses from text → Person MENTIONS hints
        email_hints = extract_email_person_hints(
            doc.text, doc.source_id, doc.node_label
        )
        if email_hints:
            doc.graph_hints.extend(email_hints)

        # 0.5. Content-hash skip: if the canonical text matches what we
        # previously indexed for this source_id, the chunks/embeddings/
        # extracted entities are still valid.  Re-upsert the source node
        # so any metadata-only changes (mtime, labels, etc.) land, but
        # bypass the expensive chunk → embed → extract → resolve path.
        content_hash = hashlib.sha256(doc.text.encode("utf-8")).hexdigest()
        if existing_hash is not None and existing_hash == content_hash:
            with observe_duration(PIPELINE_DURATION, stage="write"):
                self._writer.write(WriteUnit(doc=doc, content_hash=content_hash))
            PIPELINE_SKIPPED.labels(
                source_type=doc.source_type, reason="content_hash"
            ).inc()
            DOCUMENTS_PROCESSED.labels(
                source_type=doc.source_type, operation=doc.operation
            ).inc()
            logger.debug(
                "Skipping %s %s (content_hash unchanged)",
                doc.source_type,
                doc.source_id,
            )
            return

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

        progress_label = doc.node_props.get("path") or doc.source_id
        self._progress.start_file(
            doc.source_id, str(progress_label), total_chunks=len(chunks)
        )
        try:
            # 2. Embed
            self._progress.set_stage(doc.source_id, "embed")
            with observe_duration(PIPELINE_DURATION, stage="embed"):
                embedded = embed_chunks(chunk_texts, self._registry)
            vectors = [vec for _, vec in embedded]

            if len(vectors) != len(chunks):
                DOCUMENTS_FAILED.labels(
                    source_type=doc.source_type, stage="embed"
                ).inc()
                raise RuntimeError(
                    f"Embedding returned {len(vectors)} vectors for {len(chunks)} chunks "
                    f"(doc {doc.source_id}) — refusing to proceed with mismatched data"
                )

            CHUNKS_EMBEDDED.inc(len(chunks))

            # 3. Extract entities and triples
            self._progress.set_stage(doc.source_id, "extract")
            with observe_duration(PIPELINE_DURATION, stage="extract"):
                extraction_results = extract_chunks(
                    chunks,
                    self._registry,
                    on_chunk=lambda sid=doc.source_id: self._progress.advance(sid),
                )

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
            self._progress.set_stage(doc.source_id, "resolve")
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
                content_hash=content_hash,
            )
            self._progress.set_stage(doc.source_id, "write")
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
        finally:
            self._progress.finish_file(doc.source_id)

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
            latitude=doc.node_props.get("latitude"),
            longitude=doc.node_props.get("longitude"),
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

        # Reverse geocode if the image has GPS coordinates
        if result.latitude is not None and result.longitude is not None:
            try:
                geo = reverse_geocode_cached(
                    result.latitude,
                    result.longitude,
                    self._writer.neo4j_session,
                )
                if geo.display_name:
                    link_image_to_location(
                        result.source_id,
                        result.latitude,
                        result.longitude,
                        self._writer.neo4j_session,
                    )
                    logger.info(
                        "Geocoded %s → %s",
                        result.source_id,
                        geo.display_name,
                    )
            except Exception:
                logger.debug(
                    "Reverse geocoding failed for %s",
                    result.source_id,
                    exc_info=True,
                )

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
