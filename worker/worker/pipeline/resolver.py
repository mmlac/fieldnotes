"""Entity resolver: fuzzy dedup with rapidfuzz + embedding similarity.

Before writing to Neo4j, resolves extracted entity names against existing
entities using three strategies:
  1. Exact match on lowercased name
  2. Fuzzy match via rapidfuzz with threshold 88 similarity
  3. Embedding similarity fallback — cosine > 0.92 creates SAME_AS edge
     instead of merge, preserving both forms

Wikilink-sourced entities (confidence >= 0.95) act as anchors — ambiguous
LLM mentions preferentially merge into them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rapidfuzz import fuzz, process

from worker.models.base import EmbedRequest
from worker.models.resolver import ModelRegistry, ResolvedModel

logger = logging.getLogger(__name__)

FUZZY_THRESHOLD = 88
FUZZY_THRESHOLD_SHORT = 100  # exact match for names < 4 chars
FUZZY_THRESHOLD_MEDIUM = 95  # stricter threshold for names 4-5 chars
SHORT_NAME_LIMIT = 4
MEDIUM_NAME_LIMIT = 6
COSINE_THRESHOLD = 0.92
ANCHOR_CONFIDENCE = 0.95
EMBED_ROLE = "embed"


@dataclass
class ResolvedEntity:
    """An entity after resolution with merge/link decisions."""

    name: str
    type: str
    confidence: float
    merged_into: str | None = None
    same_as: str | None = None


@dataclass
class ResolutionResult:
    """Output of the entity resolution pass."""

    entities: list[ResolvedEntity] = field(default_factory=list)
    same_as_edges: list[tuple[str, str]] = field(default_factory=list)


def resolve_entities(
    extracted: list[dict[str, Any]],
    existing: list[dict[str, Any]],
    embed_model: ResolvedModel | None = None,
) -> ResolutionResult:
    """Resolve extracted entities against existing ones.

    Parameters
    ----------
    extracted:
        Entities from LLM extraction. Each dict has 'name', 'type',
        'confidence' keys.
    existing:
        Entities already in Neo4j. Same schema as extracted.
    embed_model:
        Optional embedding model for cosine similarity fallback.
        If None, strategy 3 is skipped.

    Returns
    -------
    ResolutionResult
        Deduplicated entities with merge/link decisions.
    """
    if not extracted:
        return ResolutionResult()

    # Build lookup of existing entities by lowercased name
    existing_by_lower: dict[str, dict[str, Any]] = {}
    for ent in existing:
        existing_by_lower[ent["name"].lower()] = ent

    # Separate anchors (high-confidence wikilink entities) from ambiguous
    anchors: list[dict[str, Any]] = [
        e for e in existing if e.get("confidence", 0) >= ANCHOR_CONFIDENCE
    ]
    anchor_names = [a["name"] for a in anchors]

    result = ResolutionResult()
    unresolved: list[dict[str, Any]] = []

    for entity in extracted:
        name_lower = entity["name"].lower()

        # Strategy 1: exact match on lowercased name
        if name_lower in existing_by_lower:
            target = existing_by_lower[name_lower]
            result.entities.append(ResolvedEntity(
                name=target["name"],
                type=entity.get("type", target.get("type", "Concept")),
                confidence=max(
                    entity.get("confidence", 0.75),
                    target.get("confidence", 0.75),
                ),
                merged_into=target["name"],
            ))
            continue

        # Strategy 2: fuzzy match via rapidfuzz
        match = _fuzzy_match(entity, existing, anchors)
        if match is not None:
            result.entities.append(match)
            if match.merged_into:
                existing_by_lower[name_lower] = {
                    "name": match.merged_into,
                    "type": match.type,
                    "confidence": match.confidence,
                }
            continue

        # Collect for embedding fallback
        unresolved.append(entity)

    # Strategy 3: embedding similarity fallback
    if unresolved and embed_model is not None and existing:
        _resolve_by_embedding(unresolved, existing, embed_model, result)
    else:
        # No embedding model or no existing entities — keep as new
        for entity in unresolved:
            result.entities.append(ResolvedEntity(
                name=entity["name"],
                type=entity.get("type", "Concept"),
                confidence=entity.get("confidence", 0.75),
            ))

    return result


def resolve_entities_from_registry(
    extracted: list[dict[str, Any]],
    existing: list[dict[str, Any]],
    registry: ModelRegistry,
) -> ResolutionResult:
    """Convenience wrapper that resolves the embed model from a registry.

    Parameters
    ----------
    extracted:
        Entities from LLM extraction.
    existing:
        Entities already in Neo4j.
    registry:
        Model registry for resolving the embedding role.

    Returns
    -------
    ResolutionResult
    """
    embed_model: ResolvedModel | None = None
    try:
        embed_model = registry.for_role(EMBED_ROLE)
    except KeyError:
        pass

    return resolve_entities(extracted, existing, embed_model)


def _fuzzy_threshold_for_length(name: str) -> int:
    """Return the appropriate fuzzy threshold based on entity name length.

    Short names are highly susceptible to false positive merges because a
    single character difference represents a large fraction of the name.
    E.g., "AWS" vs "AMS" scores ~89 with token_sort_ratio but are distinct.

    - < 4 chars: require exact match (threshold 100)
    - 4-5 chars: stricter threshold (95)
    - >= 6 chars: standard threshold (88)
    """
    length = len(name.strip())
    if length < SHORT_NAME_LIMIT:
        return FUZZY_THRESHOLD_SHORT
    if length < MEDIUM_NAME_LIMIT:
        return FUZZY_THRESHOLD_MEDIUM
    return FUZZY_THRESHOLD


def _fuzzy_match(
    entity: dict[str, Any],
    existing: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
) -> ResolvedEntity | None:
    """Try to fuzzy-match entity against existing entities.

    Anchors (wikilink-sourced, confidence >= 0.95) are preferred targets.
    Returns None if no match meets the threshold.

    Uses rapidfuzz.process.extractOne for C-optimized matching with
    score_cutoff early termination instead of manual O(N) iteration.

    The score_cutoff is adjusted based on entity name length to prevent
    false positive merges on short names (< 5 chars).
    """
    name_lower = entity["name"].lower()
    threshold = _fuzzy_threshold_for_length(entity["name"])

    # Check anchors first — prefer merging into them
    if anchors:
        anchor_names_lower = [a["name"].lower() for a in anchors]
        result = process.extractOne(
            name_lower,
            anchor_names_lower,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=threshold,
        )
        if result is not None:
            _, _, idx = result
            best_target = anchors[idx]
            return ResolvedEntity(
                name=best_target["name"],
                type=entity.get("type", best_target.get("type", "Concept")),
                confidence=max(
                    entity.get("confidence", 0.75),
                    best_target.get("confidence", 0.75),
                ),
                merged_into=best_target["name"],
            )

    # Check all existing entities
    if existing:
        existing_names_lower = [e["name"].lower() for e in existing]
        result = process.extractOne(
            name_lower,
            existing_names_lower,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=threshold,
        )
        if result is not None:
            _, _, idx = result
            best_target = existing[idx]
            return ResolvedEntity(
                name=best_target["name"],
                type=entity.get("type", best_target.get("type", "Concept")),
                confidence=max(
                    entity.get("confidence", 0.75),
                    best_target.get("confidence", 0.75),
                ),
                merged_into=best_target["name"],
            )

    return None


def _resolve_by_embedding(
    unresolved: list[dict[str, Any]],
    existing: list[dict[str, Any]],
    embed_model: ResolvedModel,
    result: ResolutionResult,
) -> None:
    """Use embedding cosine similarity to create SAME_AS edges or keep as new.

    Entities with cosine > COSINE_THRESHOLD get a SAME_AS edge to the closest
    existing entity (preserving both forms). Others are kept as new entities.
    """
    # Embed unresolved names
    unresolved_names = [e["name"] for e in unresolved]
    existing_names = [e["name"] for e in existing]

    try:
        unresolved_resp = embed_model.embed(EmbedRequest(texts=unresolved_names))
        existing_resp = embed_model.embed(EmbedRequest(texts=existing_names))
    except Exception:
        logger.warning("Embedding fallback failed, keeping entities as new", exc_info=True)
        for entity in unresolved:
            result.entities.append(ResolvedEntity(
                name=entity["name"],
                type=entity.get("type", "Concept"),
                confidence=entity.get("confidence", 0.75),
            ))
        return

    unresolved_vecs = unresolved_resp.vectors
    existing_vecs = existing_resp.vectors

    sim_matrix = _batch_cosine_similarity(unresolved_vecs, existing_vecs)

    for i, entity in enumerate(unresolved):
        best_idx = int(np.argmax(sim_matrix[i]))
        best_sim = float(sim_matrix[i, best_idx])

        if best_sim > COSINE_THRESHOLD and best_idx >= 0:
            target_name = existing_names[best_idx]
            result.entities.append(ResolvedEntity(
                name=entity["name"],
                type=entity.get("type", "Concept"),
                confidence=entity.get("confidence", 0.75),
                same_as=target_name,
            ))
            result.same_as_edges.append((entity["name"], target_name))
            logger.debug(
                "SAME_AS edge: %r → %r (cosine=%.3f)",
                entity["name"], target_name, best_sim,
            )
        else:
            result.entities.append(ResolvedEntity(
                name=entity["name"],
                type=entity.get("type", "Concept"),
                confidence=entity.get("confidence", 0.75),
            ))


def _batch_cosine_similarity(
    a: list[list[float]], b: list[list[float]]
) -> np.ndarray:
    """Compute cosine similarity matrix between two sets of vectors.

    Returns an (n, m) matrix where entry [i, j] is the cosine similarity
    between a[i] and b[j].  Uses numpy for vectorized computation.
    """
    mat_a = np.asarray(a, dtype=np.float64)
    mat_b = np.asarray(b, dtype=np.float64)
    norms_a = np.linalg.norm(mat_a, axis=1, keepdims=True)
    norms_b = np.linalg.norm(mat_b, axis=1, keepdims=True)
    # Replace zero norms with 1 to avoid division by zero (result will be 0).
    norms_a = np.where(norms_a == 0, 1.0, norms_a)
    norms_b = np.where(norms_b == 0, 1.0, norms_b)
    return (mat_a / norms_a) @ (mat_b / norms_b).T
