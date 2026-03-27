"""LLM-based entity and triple extraction from text chunks.

For each text chunk, prompts the 'extract' role model to extract named entities
(people, technologies, projects, organizations, concepts) and relationship
triples (subject-predicate-object). Uses structured JSON output via tool_use.

Falls back to 'extract_fallback' role model on malformed JSON (JSONDecodeError,
ValidationError). Returns ExtractionResult with entities list and triples list
per chunk.
"""

from __future__ import annotations

import json
import os
import logging
from dataclasses import dataclass, field
from typing import Any

from worker.models.base import CompletionRequest
from worker.models.resolver import ModelRegistry, ResolvedModel
from worker.pipeline.chunker import Chunk

logger = logging.getLogger(__name__)

EXTRACT_ROLE = "extract"
FALLBACK_ROLE = "extract_fallback"
LLM_TIMEOUT = float(os.environ.get("FIELDNOTES_LLM_TIMEOUT", 600.0))

ALLOWED_ENTITY_TYPES = frozenset(
    {"Person", "Technology", "Project", "Organization", "Concept", "Location"}
)
DEFAULT_CONFIDENCE = 0.75
MAX_ENTITY_NAME_LEN = 512

SYSTEM_PROMPT = """\
You are an entity and relationship extraction system. Given a text chunk, \
extract all named entities and relationship triples.

Entities: people, technologies, projects, organizations, concepts, and locations. \
For each entity provide a name, type, and confidence score (0.0-1.0).

Triples: subject-predicate-object relationships between entities. \
The subject and object must be entity names from your extraction. \
Use ONLY these predicate types: WORKS_AT, WORKS_FOR, WORKS_ON, KNOWS, \
COLLABORATES_WITH, USES, USED_BY, CREATED_BY, CREATED, PART_OF, \
BELONGS_TO, CONTAINS, DEPENDS_ON, IS_A, HAS_A, LOCATED_IN, MANAGES, \
REPORTS_TO, CONTRIBUTED_TO, BASED_ON, ASSOCIATED_WITH, FOUNDED, \
FUNDED_BY, PUBLISHED, AUTHORED, EMPLOYED_BY, AFFILIATED_WITH, MEMBER_OF, \
MENTIONS, REFERENCES, IMPLEMENTS, EXTENDS, INTEGRATES_WITH, SAME_AS, \
INFLUENCED_BY, SUPPORTS, ACQUIRED_BY, INVESTED_IN, DEVELOPED_BY, \
MAINTAINED_BY, OWNED_BY, LED_BY, TAUGHT_BY, ATTENDED, PARTICIPATED_IN, \
SPOKE_AT, INSTALLED_VIA, CATEGORIZED_AS, PROVIDES, HAS_ROLE, MEETS_WITH, \
LOCATED_AT, SENT_BY, SENT_TO, HOSTED_BY, SCHEDULED_AT, RELATED_TO. \
If none of these fit, use RELATED_TO.

Call the extract_entities_and_triples tool with your results."""

EXTRACTION_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_entities_and_triples",
        "description": "Submit extracted entities and relationship triples from text.",
        "parameters": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "type": {
                                "type": "string",
                                "enum": [
                                    "Person",
                                    "Technology",
                                    "Project",
                                    "Organization",
                                    "Concept",
                                ],
                            },
                            "confidence": {"type": "number"},
                        },
                        "required": ["name", "type", "confidence"],
                    },
                },
                "triples": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "subject": {"type": "string"},
                            "predicate": {
                                "type": "string",
                                "enum": [
                                    "RELATED_TO",
                                    "WORKS_AT",
                                    "WORKS_FOR",
                                    "WORKS_ON",
                                    "KNOWS",
                                    "COLLABORATES_WITH",
                                    "USES",
                                    "USED_BY",
                                    "CREATED_BY",
                                    "CREATED",
                                    "PART_OF",
                                    "BELONGS_TO",
                                    "CONTAINS",
                                    "DEPENDS_ON",
                                    "IS_A",
                                    "HAS_A",
                                    "LOCATED_IN",
                                    "MANAGES",
                                    "REPORTS_TO",
                                    "CONTRIBUTED_TO",
                                    "BASED_ON",
                                    "ASSOCIATED_WITH",
                                    "FOUNDED",
                                    "FUNDED_BY",
                                    "PUBLISHED",
                                    "AUTHORED",
                                    "EMPLOYED_BY",
                                    "AFFILIATED_WITH",
                                    "MEMBER_OF",
                                    "MENTIONS",
                                    "REFERENCES",
                                    "IMPLEMENTS",
                                    "EXTENDS",
                                    "INTEGRATES_WITH",
                                    "SAME_AS",
                                    "INFLUENCED_BY",
                                    "SUPPORTS",
                                    "ACQUIRED_BY",
                                    "INVESTED_IN",
                                    "DEVELOPED_BY",
                                    "MAINTAINED_BY",
                                    "OWNED_BY",
                                    "LED_BY",
                                    "TAUGHT_BY",
                                    "ATTENDED",
                                    "PARTICIPATED_IN",
                                    "SPOKE_AT",
                                    "INSTALLED_VIA",
                                    "CATEGORIZED_AS",
                                    "PROVIDES",
                                    "HAS_ROLE",
                                    "MEETS_WITH",
                                    "LOCATED_AT",
                                    "SENT_BY",
                                    "SENT_TO",
                                    "HOSTED_BY",
                                    "SCHEDULED_AT",
                                ],
                            },
                            "object": {"type": "string"},
                        },
                        "required": ["subject", "predicate", "object"],
                    },
                },
            },
            "required": ["entities", "triples"],
        },
    },
}


@dataclass
class ExtractionResult:
    """Extraction output for a single chunk.

    When ``failed`` is True the result is a sentinel indicating that extraction
    could not be performed (both primary and fallback models failed).  The
    pipeline should treat this chunk as an extraction failure rather than as a
    document that simply contained no entities.
    """

    entities: list[dict[str, Any]] = field(default_factory=list)
    triples: list[dict[str, str]] = field(default_factory=list)
    failed: bool = False


def extract_chunk(
    chunk: Chunk,
    model: ResolvedModel,
    fallback_model: ResolvedModel | None = None,
) -> ExtractionResult:
    """Extract entities and triples from a single chunk.

    Parameters
    ----------
    chunk:
        The text chunk to process.
    model:
        The primary extraction model (resolved from 'extract' role).
    fallback_model:
        Optional fallback model (resolved from 'extract_fallback' role).
        Used when the primary model returns malformed JSON.

    Returns
    -------
    ExtractionResult
        Extracted entities and triples. Empty on unrecoverable failure.
    """
    req = CompletionRequest(
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": chunk.text}],
        tools=[EXTRACTION_TOOL],
        temperature=0.0,
        timeout=LLM_TIMEOUT,
    )

    try:
        return _call_and_parse(model, req)
    except (json.JSONDecodeError, ValidationError, ExtractionError) as exc:
        logger.warning("Primary extraction failed for chunk %d: %s", chunk.index, exc)
        if fallback_model is not None:
            try:
                return _call_and_parse(fallback_model, req)
            except (json.JSONDecodeError, ValidationError, ExtractionError) as exc2:
                logger.error(
                    "Fallback extraction also failed for chunk %d: %s",
                    chunk.index,
                    exc2,
                )
        return ExtractionResult(failed=True)


def extract_chunks(
    chunks: list[Chunk],
    registry: ModelRegistry,
) -> list[ExtractionResult]:
    """Extract entities and triples from all chunks.

    Parameters
    ----------
    chunks:
        Text chunks to process.
    registry:
        Model registry for resolving role models.

    Returns
    -------
    list[ExtractionResult]
        One result per chunk, in the same order.
    """
    if not chunks:
        return []

    model = registry.for_role(EXTRACT_ROLE)

    fallback_model: ResolvedModel | None = None
    try:
        fallback_model = registry.for_role(FALLBACK_ROLE)
    except KeyError:
        pass

    return [extract_chunk(c, model, fallback_model) for c in chunks]


def _call_and_parse(model: ResolvedModel, req: CompletionRequest) -> ExtractionResult:
    """Call the model and parse tool_call arguments into an ExtractionResult.

    Raises
    ------
    json.JSONDecodeError
        If the tool_call arguments cannot be parsed as JSON.
    ValidationError
        If the parsed JSON is missing required fields.
    """
    resp = model.complete(req, task="extract_entities")

    # Try to extract from tool_calls first
    if resp.tool_calls:
        for tc in resp.tool_calls:
            fn = tc.get("function", tc)
            name = fn.get("name", "")
            if name == "extract_entities_and_triples":
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                return _validate_and_build(args)

    # Some models return JSON in the text body instead of tool_calls
    if resp.text and resp.text.strip():
        data = json.loads(resp.text)
        return _validate_and_build(data)

    raise ExtractionError("LLM returned empty response (no tool_calls and no text)")


def _validate_and_build(data: dict[str, Any]) -> ExtractionResult:
    """Validate parsed JSON and build an ExtractionResult.

    Raises
    ------
    ValidationError
        If required fields are missing or have wrong types.
    """
    if not isinstance(data, dict):
        raise ValidationError(f"Expected dict, got {type(data).__name__}")

    entities_raw = data.get("entities", [])
    triples_raw = data.get("triples", [])

    if not isinstance(entities_raw, list):
        raise ValidationError(
            f"'entities' must be a list, got {type(entities_raw).__name__}"
        )
    if not isinstance(triples_raw, list):
        raise ValidationError(
            f"'triples' must be a list, got {type(triples_raw).__name__}"
        )

    entities: list[dict[str, Any]] = []
    for ent in entities_raw:
        if not isinstance(ent, dict) or "name" not in ent:
            continue
        name = ent["name"]
        if not isinstance(name, str) or len(name) > MAX_ENTITY_NAME_LEN:
            logger.warning(
                "Rejecting entity with oversized name (%d chars, limit %d): %.80s...",
                len(name) if isinstance(name, str) else 0,
                MAX_ENTITY_NAME_LEN,
                str(name)[:80],
            )
            continue
        entity_type = ent.get("type", "Concept")
        if entity_type not in ALLOWED_ENTITY_TYPES:
            entity_type = "Concept"
        try:
            confidence = float(ent.get("confidence", DEFAULT_CONFIDENCE))
        except (ValueError, TypeError):
            confidence = DEFAULT_CONFIDENCE
        confidence = max(0.0, min(1.0, confidence))
        entities.append(
            {
                "name": name,
                "type": entity_type,
                "confidence": confidence,
            }
        )

    triples: list[dict[str, str]] = []
    for tri in triples_raw:
        if not isinstance(tri, dict):
            continue
        if not all(k in tri for k in ("subject", "predicate", "object")):
            continue
        subj = tri["subject"]
        pred = tri["predicate"]
        obj = tri["object"]
        # Reject non-string values instead of coercing (dicts/lists become garbage repr)
        if (
            not isinstance(subj, str)
            or not isinstance(pred, str)
            or not isinstance(obj, str)
        ):
            logger.warning(
                "Rejecting triple with non-string field(s): subject=%s, predicate=%s, object=%s",
                type(subj).__name__,
                type(pred).__name__,
                type(obj).__name__,
            )
            continue
        if len(subj) > MAX_ENTITY_NAME_LEN or len(obj) > MAX_ENTITY_NAME_LEN:
            logger.warning(
                "Rejecting triple with oversized subject/object name "
                "(subject=%d chars, object=%d chars, limit %d)",
                len(subj),
                len(obj),
                MAX_ENTITY_NAME_LEN,
            )
            continue
        if len(pred) > MAX_ENTITY_NAME_LEN:
            logger.warning(
                "Rejecting triple with oversized predicate (%d chars, limit %d)",
                len(pred),
                MAX_ENTITY_NAME_LEN,
            )
            continue
        triples.append(
            {
                "subject": subj,
                "predicate": pred,
                "object": obj,
            }
        )

    return ExtractionResult(entities=entities, triples=triples)


class ValidationError(Exception):
    """Raised when extraction output fails structural validation."""


class ExtractionError(Exception):
    """Raised when extraction fails entirely (empty LLM response, all retries exhausted)."""
