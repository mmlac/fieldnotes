"""Tests for the entity resolver: fuzzy dedup with rapidfuzz + embedding similarity.

Uses unittest.mock to stub out embedding calls so tests run without
a live model backend.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from worker.models.base import EmbedResponse
from worker.models.resolver import ResolvedModel
from worker.pipeline.resolver import (
    ANCHOR_CONFIDENCE,
    COSINE_THRESHOLD,
    FUZZY_THRESHOLD,
    ResolutionResult,
    ResolvedEntity,
    _cosine_similarity,
    _fuzzy_match,
    resolve_entities,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


def _entity(
    name: str,
    type: str = "Concept",
    confidence: float = 0.75,
) -> dict:
    return {"name": name, "type": type, "confidence": confidence}


def _anchor(name: str, type: str = "Concept") -> dict:
    return {"name": name, "type": type, "confidence": 0.97}


def _mock_embed_model(vectors_map: dict[str, list[float]]) -> ResolvedModel:
    """Create a mock embed model that returns vectors based on text lookup."""
    model = MagicMock(spec=ResolvedModel)

    def embed_side_effect(req):
        vecs = [vectors_map.get(t, [0.0] * 3) for t in req.texts]
        return EmbedResponse(vectors=vecs, model="mock", input_tokens=0)

    model.embed.side_effect = embed_side_effect
    return model


# ------------------------------------------------------------------
# _cosine_similarity
# ------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        assert _cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        assert _cosine_similarity([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        assert _cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_zero_vector(self) -> None:
        assert _cosine_similarity([0, 0], [1, 1]) == 0.0


# ------------------------------------------------------------------
# Strategy 1: exact match
# ------------------------------------------------------------------


class TestExactMatch:
    def test_exact_case_insensitive(self) -> None:
        extracted = [_entity("Neo4j", "Technology", 0.8)]
        existing = [_entity("neo4j", "Technology", 0.9)]

        result = resolve_entities(extracted, existing)
        assert len(result.entities) == 1
        assert result.entities[0].merged_into == "neo4j"
        assert result.entities[0].confidence == 0.9

    def test_exact_match_preserves_existing_name(self) -> None:
        extracted = [_entity("PYTHON", "Technology", 0.7)]
        existing = [_entity("Python", "Technology", 0.85)]

        result = resolve_entities(extracted, existing)
        assert result.entities[0].name == "Python"
        assert result.entities[0].merged_into == "Python"

    def test_no_match_keeps_entity_as_new(self) -> None:
        extracted = [_entity("Rust")]
        existing = [_entity("Python")]

        result = resolve_entities(extracted, existing)
        assert len(result.entities) == 1
        assert result.entities[0].name == "Rust"
        assert result.entities[0].merged_into is None
        assert result.entities[0].same_as is None


# ------------------------------------------------------------------
# Strategy 2: fuzzy match
# ------------------------------------------------------------------


class TestFuzzyMatch:
    def test_fuzzy_similar_names_merge(self) -> None:
        # "JavaScript" vs "Javascript" — very high similarity
        extracted = [_entity("Javascript", "Technology")]
        existing = [_entity("JavaScript", "Technology", 0.9)]

        result = resolve_entities(extracted, existing)
        assert len(result.entities) == 1
        assert result.entities[0].merged_into == "JavaScript"

    def test_fuzzy_below_threshold_no_merge(self) -> None:
        # Very different names should not merge
        extracted = [_entity("Kubernetes")]
        existing = [_entity("Docker")]

        result = resolve_entities(extracted, existing)
        assert result.entities[0].merged_into is None

    def test_anchor_preferred_over_non_anchor(self) -> None:
        extracted = [_entity("React.js", "Technology", 0.7)]
        existing = [
            _entity("ReactJS", "Technology", 0.6),  # non-anchor
            _anchor("React.js", "Technology"),  # anchor
        ]

        result = resolve_entities(extracted, existing)
        assert len(result.entities) == 1
        # Should merge — exact or fuzzy match into the anchor
        assert result.entities[0].merged_into is not None

    def test_fuzzy_match_function_directly(self) -> None:
        entity = _entity("Golang", "Technology")
        existing = [_entity("Go", "Technology")]
        anchors: list[dict] = []

        # "Golang" vs "Go" — low similarity, should not match
        match = _fuzzy_match(entity, existing, anchors)
        assert match is None

    def test_fuzzy_match_close_names(self) -> None:
        entity = _entity("TensorFlow", "Technology")
        existing = [_entity("Tensorflow", "Technology", 0.85)]
        anchors: list[dict] = []

        match = _fuzzy_match(entity, existing, anchors)
        assert match is not None
        assert match.merged_into == "Tensorflow"


# ------------------------------------------------------------------
# Strategy 3: embedding similarity
# ------------------------------------------------------------------


class TestEmbeddingSimilarity:
    def test_high_cosine_creates_same_as(self) -> None:
        # Vectors with cosine > 0.92
        vectors = {
            "ML": [0.9, 0.1, 0.0],
            "Machine Learning": [0.88, 0.12, 0.01],
        }
        embed_model = _mock_embed_model(vectors)

        # "ML" and "Machine Learning" are too different for fuzzy match
        extracted = [_entity("ML", "Concept")]
        existing = [_entity("Machine Learning", "Concept")]

        result = resolve_entities(extracted, existing, embed_model=embed_model)
        assert len(result.entities) == 1
        assert result.entities[0].same_as == "Machine Learning"
        assert len(result.same_as_edges) == 1
        assert result.same_as_edges[0] == ("ML", "Machine Learning")

    def test_low_cosine_keeps_as_new(self) -> None:
        # Orthogonal vectors — no similarity
        vectors = {
            "Quantum": [1.0, 0.0, 0.0],
            "Cooking": [0.0, 1.0, 0.0],
        }
        embed_model = _mock_embed_model(vectors)

        extracted = [_entity("Quantum")]
        existing = [_entity("Cooking")]

        result = resolve_entities(extracted, existing, embed_model=embed_model)
        assert result.entities[0].same_as is None
        assert len(result.same_as_edges) == 0

    def test_embed_failure_keeps_as_new(self) -> None:
        embed_model = MagicMock(spec=ResolvedModel)
        embed_model.embed.side_effect = RuntimeError("connection refused")

        extracted = [_entity("NewThing")]
        existing = [_entity("OldThing")]

        result = resolve_entities(extracted, existing, embed_model=embed_model)
        assert len(result.entities) == 1
        assert result.entities[0].name == "NewThing"
        assert result.entities[0].merged_into is None
        assert result.entities[0].same_as is None


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_extracted(self) -> None:
        result = resolve_entities([], [_entity("X")])
        assert result.entities == []
        assert result.same_as_edges == []

    def test_empty_existing(self) -> None:
        result = resolve_entities([_entity("X")], [])
        assert len(result.entities) == 1
        assert result.entities[0].name == "X"
        assert result.entities[0].merged_into is None

    def test_both_empty(self) -> None:
        result = resolve_entities([], [])
        assert result.entities == []

    def test_multiple_entities_mixed_strategies(self) -> None:
        existing = [
            _anchor("Python", "Technology"),
            _entity("Neo4j", "Technology", 0.9),
        ]
        extracted = [
            _entity("python", "Technology", 0.8),  # exact match
            _entity("Rust", "Technology", 0.85),  # no match
        ]

        result = resolve_entities(extracted, existing)
        assert len(result.entities) == 2

        # First should be merged into Python anchor
        python_ent = next(e for e in result.entities if e.name == "Python")
        assert python_ent.merged_into == "Python"

        # Second should be kept as new
        rust_ent = next(e for e in result.entities if e.name == "Rust")
        assert rust_ent.merged_into is None

    def test_confidence_takes_max(self) -> None:
        extracted = [_entity("X", "Concept", 0.9)]
        existing = [_entity("X", "Concept", 0.5)]

        result = resolve_entities(extracted, existing)
        assert result.entities[0].confidence == 0.9

    def test_no_embed_model_skips_strategy_3(self) -> None:
        # Names too different for fuzzy, no embed model
        extracted = [_entity("AI")]
        existing = [_entity("Artificial Intelligence")]

        result = resolve_entities(extracted, existing, embed_model=None)
        assert len(result.entities) == 1
        assert result.entities[0].name == "AI"
        assert result.entities[0].merged_into is None
        assert result.entities[0].same_as is None


# ------------------------------------------------------------------
# ResolutionResult / ResolvedEntity dataclasses
# ------------------------------------------------------------------


class TestDataclasses:
    def test_resolved_entity_defaults(self) -> None:
        e = ResolvedEntity(name="X", type="Concept", confidence=0.5)
        assert e.merged_into is None
        assert e.same_as is None

    def test_resolution_result_defaults(self) -> None:
        r = ResolutionResult()
        assert r.entities == []
        assert r.same_as_edges == []
