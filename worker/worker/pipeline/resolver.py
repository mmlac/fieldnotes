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
import re
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

# Cross-source resolution defaults
CROSS_SOURCE_CONFIDENCE_THRESHOLD = 0.8
CROSS_SOURCE_FUZZY_THRESHOLD = 85  # slightly relaxed for known cross-source pairs
EMAIL_MAX_LENGTH = 254  # RFC 5321 max email length
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _clamp_confidence(value: Any) -> float:
    """Clamp a confidence score to [0.0, 1.0]."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.75  # default
    if f != f:  # NaN check
        return 0.75
    return max(0.0, min(1.0, f))


def _is_valid_email(email: str) -> bool:
    """Validate email format and length per RFC 5321."""
    if not email or len(email) > EMAIL_MAX_LENGTH:
        return False
    return _EMAIL_RE.match(email) is not None


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

    # Clamp confidence scores to [0.0, 1.0] on ingestion
    for ent in extracted:
        ent["confidence"] = _clamp_confidence(ent.get("confidence", 0.75))
    for ent in existing:
        ent["confidence"] = _clamp_confidence(ent.get("confidence", 0.75))

    # Build lookup of existing entities by lowercased name
    existing_by_lower: dict[str, dict[str, Any]] = {}
    for ent in existing:
        existing_by_lower[ent["name"].lower()] = ent

    # Separate anchors (high-confidence wikilink entities) from ambiguous
    anchors: list[dict[str, Any]] = [
        e for e in existing if e.get("confidence", 0) >= ANCHOR_CONFIDENCE
    ]
    result = ResolutionResult()
    unresolved: list[dict[str, Any]] = []

    for entity in extracted:
        name_lower = entity["name"].lower()

        # Strategy 1: exact match on lowercased name
        if name_lower in existing_by_lower:
            target = existing_by_lower[name_lower]
            result.entities.append(
                ResolvedEntity(
                    name=target["name"],
                    type=entity.get("type", target.get("type", "Concept")),
                    confidence=max(
                        entity.get("confidence", 0.75),
                        target.get("confidence", 0.75),
                    ),
                    merged_into=target["name"],
                )
            )
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
            result.entities.append(
                ResolvedEntity(
                    name=entity["name"],
                    type=entity.get("type", "Concept"),
                    confidence=entity.get("confidence", 0.75),
                )
            )

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
        unresolved_resp = embed_model.embed(
            EmbedRequest(texts=unresolved_names), task="resolve_entities"
        )
        existing_resp = embed_model.embed(
            EmbedRequest(texts=existing_names), task="resolve_entities"
        )
    except Exception:
        logger.warning(
            "Embedding fallback failed, keeping entities as new", exc_info=True
        )
        for entity in unresolved:
            result.entities.append(
                ResolvedEntity(
                    name=entity["name"],
                    type=entity.get("type", "Concept"),
                    confidence=entity.get("confidence", 0.75),
                )
            )
        return

    unresolved_vecs = unresolved_resp.vectors
    existing_vecs = existing_resp.vectors

    sim_matrix = _batch_cosine_similarity(unresolved_vecs, existing_vecs)

    for i, entity in enumerate(unresolved):
        row = sim_matrix[i]
        # Handle NaN vectors: if entire row is NaN, treat as no match
        if np.all(np.isnan(row)):
            result.entities.append(
                ResolvedEntity(
                    name=entity["name"],
                    type=entity.get("type", "Concept"),
                    confidence=entity.get("confidence", 0.75),
                )
            )
            continue
        # Replace NaN entries with -1 so argmax ignores them
        safe_row = np.where(np.isnan(row), -1.0, row)
        best_idx = int(np.argmax(safe_row))
        best_sim = float(safe_row[best_idx])

        if best_sim > COSINE_THRESHOLD and best_idx < len(existing_names):
            target_name = existing_names[best_idx]
            result.entities.append(
                ResolvedEntity(
                    name=entity["name"],
                    type=entity.get("type", "Concept"),
                    confidence=entity.get("confidence", 0.75),
                    same_as=target_name,
                )
            )
            result.same_as_edges.append((entity["name"], target_name))
            logger.debug(
                "SAME_AS edge: %r → %r (cosine=%.3f)",
                entity["name"],
                target_name,
                best_sim,
            )
        else:
            result.entities.append(
                ResolvedEntity(
                    name=entity["name"],
                    type=entity.get("type", "Concept"),
                    confidence=entity.get("confidence", 0.75),
                )
            )


@dataclass
class CrossSourceMatch:
    """A match between entities found across different source types."""

    entity_a: str
    entity_b: str
    confidence: float
    match_type: str  # "exact", "email", "fuzzy"


def resolve_cross_source(
    entities_by_source: dict[str, list[dict[str, Any]]],
    *,
    confidence_threshold: float = CROSS_SOURCE_CONFIDENCE_THRESHOLD,
) -> list[CrossSourceMatch]:
    """Find matching entities across different source types.

    Parameters
    ----------
    entities_by_source:
        Mapping of source_type → list of entity dicts. Each entity dict has
        'name', 'type', and optionally 'email' keys.
    confidence_threshold:
        Minimum confidence to include a match. Default 0.8.

    Returns
    -------
    list[CrossSourceMatch]
        High-confidence matches between entities from different source types.
    """
    if len(entities_by_source) < 2:
        return []

    matches: list[CrossSourceMatch] = []
    source_types = list(entities_by_source.keys())

    for i, src_a in enumerate(source_types):
        for src_b in source_types[i + 1 :]:
            entities_a = entities_by_source[src_a]
            entities_b = entities_by_source[src_b]
            matches.extend(
                _match_entity_lists(entities_a, entities_b, confidence_threshold)
            )

    return matches


def _match_entity_lists(
    entities_a: list[dict[str, Any]],
    entities_b: list[dict[str, Any]],
    confidence_threshold: float,
) -> list[CrossSourceMatch]:
    """Match two lists of entities using exact name, email, and fuzzy strategies."""
    matches: list[CrossSourceMatch] = []
    matched_b: set[int] = set()

    # Build email index for list B (validated emails only)
    email_index_b: dict[str, int] = {}
    for idx, ent in enumerate(entities_b):
        email = ent.get("email", "")
        if email and _is_valid_email(email.strip()):
            email_index_b[email.lower().strip()] = idx

    # Build lowercased name index for list B
    name_index_b: dict[str, int] = {}
    for idx, ent in enumerate(entities_b):
        name_index_b[ent["name"].lower()] = idx

    for ent_a in entities_a:
        name_a = ent_a["name"]
        name_a_lower = name_a.lower()

        # Strategy 1: Exact name match (case-insensitive)
        if name_a_lower in name_index_b:
            idx_b = name_index_b[name_a_lower]
            if idx_b not in matched_b:
                matches.append(
                    CrossSourceMatch(
                        entity_a=name_a,
                        entity_b=entities_b[idx_b]["name"],
                        confidence=1.0,
                        match_type="exact",
                    )
                )
                matched_b.add(idx_b)
                continue

        # Strategy 2: Email-based matching (Person entities)
        email_a = ent_a.get("email", "")
        if email_a and _is_valid_email(email_a.strip()):
            norm_email = email_a.lower().strip()
            if norm_email in email_index_b:
                idx_b = email_index_b[norm_email]
                if idx_b not in matched_b:
                    matches.append(
                        CrossSourceMatch(
                            entity_a=name_a,
                            entity_b=entities_b[idx_b]["name"],
                            confidence=0.95,
                            match_type="email",
                        )
                    )
                    matched_b.add(idx_b)
                    continue

        # Strategy 3: Fuzzy name match (length-aware)
        threshold = _fuzzy_threshold_for_length(name_a)
        # Use a slightly relaxed threshold for cross-source (known different contexts)
        threshold = min(threshold, CROSS_SOURCE_FUZZY_THRESHOLD)
        # But never go below the length-based floor
        threshold = max(threshold, _fuzzy_threshold_for_length(name_a))

        remaining_names = [
            (idx, entities_b[idx]["name"].lower())
            for idx in range(len(entities_b))
            if idx not in matched_b
        ]
        if not remaining_names:
            continue

        indices, names_lower = zip(*remaining_names)
        result = process.extractOne(
            name_a_lower,
            list(names_lower),
            scorer=fuzz.token_sort_ratio,
            score_cutoff=threshold,
        )
        if result is not None:
            _, score, match_idx = result
            actual_idx = indices[match_idx]
            confidence = score / 100.0
            if confidence >= confidence_threshold:
                matches.append(
                    CrossSourceMatch(
                        entity_a=name_a,
                        entity_b=entities_b[actual_idx]["name"],
                        confidence=confidence,
                        match_type="fuzzy",
                    )
                )
                matched_b.add(actual_idx)

    return matches


def _batch_cosine_similarity(a: list[list[float]], b: list[list[float]]) -> np.ndarray:
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
