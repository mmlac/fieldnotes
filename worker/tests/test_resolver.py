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
    CROSS_SOURCE_CONFIDENCE_THRESHOLD,
    CrossSourceMatch,
    FUZZY_THRESHOLD,
    FUZZY_THRESHOLD_MEDIUM,
    FUZZY_THRESHOLD_SHORT,
    ResolutionResult,
    ResolvedEntity,
    _clamp_confidence,
    _fuzzy_match,
    _fuzzy_threshold_for_length,
    _is_valid_email,
    resolve_cross_source,
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

    def embed_side_effect(req, **kwargs):
        vecs = [vectors_map.get(t, [0.0] * 3) for t in req.texts]
        return EmbedResponse(vectors=vecs, model="mock", input_tokens=0)

    model.embed.side_effect = embed_side_effect
    return model


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

    def test_short_name_no_false_positive(self) -> None:
        """Short names like 'AWS' vs 'AMS' must not merge despite high fuzzy score."""
        entity = _entity("AWS", "Organization")
        existing = [_entity("AMS", "Organization")]
        anchors: list[dict] = []

        match = _fuzzy_match(entity, existing, anchors)
        assert match is None

    def test_short_name_exact_match_still_works(self) -> None:
        """Short names that are identical (case-insensitive) should still fuzzy-match."""
        entity = _entity("AWS", "Organization")
        existing = [_entity("aws", "Organization")]
        anchors: list[dict] = []

        # token_sort_ratio("aws", "aws") == 100, passes threshold 100
        match = _fuzzy_match(entity, existing, anchors)
        assert match is not None
        assert match.merged_into == "aws"

    def test_medium_name_stricter_threshold(self) -> None:
        """Names 4-5 chars use stricter threshold (95), preventing near-misses."""
        entity = _entity("NATS", "Technology")
        existing = [_entity("NATO", "Organization")]
        anchors: list[dict] = []

        match = _fuzzy_match(entity, existing, anchors)
        assert match is None

    def test_medium_name_high_similarity_merges(self) -> None:
        """Names 4-5 chars with very high similarity (>= 95) should still merge."""
        entity = _entity("Node", "Technology")
        existing = [_entity("node", "Technology")]
        anchors: list[dict] = []

        match = _fuzzy_match(entity, existing, anchors)
        assert match is not None

    def test_long_name_uses_standard_threshold(self) -> None:
        """Names >= 6 chars use the standard 88 threshold."""
        entity = _entity("Javascript", "Technology")
        existing = [_entity("JavaScript", "Technology")]
        anchors: list[dict] = []

        match = _fuzzy_match(entity, existing, anchors)
        assert match is not None
        assert match.merged_into == "JavaScript"


# ------------------------------------------------------------------
# _fuzzy_threshold_for_length
# ------------------------------------------------------------------


class TestFuzzyThresholdForLength:
    def test_very_short_name(self) -> None:
        assert _fuzzy_threshold_for_length("AI") == FUZZY_THRESHOLD_SHORT

    def test_three_char_name(self) -> None:
        assert _fuzzy_threshold_for_length("AWS") == FUZZY_THRESHOLD_SHORT

    def test_four_char_name(self) -> None:
        assert _fuzzy_threshold_for_length("NATS") == FUZZY_THRESHOLD_MEDIUM

    def test_five_char_name(self) -> None:
        assert _fuzzy_threshold_for_length("React") == FUZZY_THRESHOLD_MEDIUM

    def test_six_char_name(self) -> None:
        assert _fuzzy_threshold_for_length("Python") == FUZZY_THRESHOLD

    def test_long_name(self) -> None:
        assert _fuzzy_threshold_for_length("JavaScript") == FUZZY_THRESHOLD


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


# ------------------------------------------------------------------
# Cross-source entity resolution
# ------------------------------------------------------------------


class TestCrossSourceResolution:
    def test_exact_name_match_across_sources(self) -> None:
        """Same entity name in email and git sources should match."""
        entities_by_source = {
            "gmail": [_entity("Python", "Technology")],
            "repositories": [_entity("Python", "Technology")],
        }
        matches = resolve_cross_source(entities_by_source)
        assert len(matches) == 1
        assert matches[0].match_type == "exact"
        assert matches[0].confidence == 1.0

    def test_case_insensitive_cross_source(self) -> None:
        """Case differences across sources should still match."""
        entities_by_source = {
            "gmail": [_entity("PYTHON", "Technology")],
            "obsidian": [_entity("Python", "Technology")],
        }
        matches = resolve_cross_source(entities_by_source)
        assert len(matches) == 1
        assert matches[0].match_type == "exact"

    def test_email_match_across_sources(self) -> None:
        """Person entities with same email across sources should match."""
        entities_by_source = {
            "gmail": [{"name": "Alex Smith", "type": "Person", "confidence": 0.9, "email": "alex@example.com"}],
            "repositories": [{"name": "A. Smith", "type": "Person", "confidence": 0.9, "email": "alex@example.com"}],
        }
        matches = resolve_cross_source(entities_by_source)
        assert len(matches) == 1
        assert matches[0].match_type == "email"
        assert matches[0].confidence == 0.95

    def test_fuzzy_match_across_sources(self) -> None:
        """Fuzzy-similar names across sources should match."""
        entities_by_source = {
            "gmail": [_entity("TensorFlow", "Technology")],
            "obsidian": [_entity("Tensorflow", "Technology")],
        }
        matches = resolve_cross_source(entities_by_source)
        assert len(matches) == 1
        assert matches[0].match_type in ("exact", "fuzzy")
        assert matches[0].confidence >= CROSS_SOURCE_CONFIDENCE_THRESHOLD

    def test_no_false_positive_short_names(self) -> None:
        """Short names that are different should not match across sources."""
        entities_by_source = {
            "gmail": [_entity("AWS", "Organization")],
            "repositories": [_entity("AMS", "Organization")],
        }
        matches = resolve_cross_source(entities_by_source)
        assert len(matches) == 0

    def test_single_source_no_matches(self) -> None:
        """Only one source type produces no matches."""
        entities_by_source = {
            "gmail": [_entity("Python", "Technology")],
        }
        matches = resolve_cross_source(entities_by_source)
        assert len(matches) == 0

    def test_three_source_types(self) -> None:
        """Entity present in all three source types should produce pairwise matches."""
        entities_by_source = {
            "gmail": [_entity("Python", "Technology")],
            "repositories": [_entity("python", "Technology")],
            "obsidian": [_entity("Python", "Technology")],
        }
        matches = resolve_cross_source(entities_by_source)
        # gmail-repos, gmail-obsidian, repos-obsidian = 3 pairs
        assert len(matches) == 3

    def test_no_match_different_entities(self) -> None:
        """Completely different entities across sources should not match."""
        entities_by_source = {
            "gmail": [_entity("Kubernetes", "Technology")],
            "repositories": [_entity("Django", "Technology")],
        }
        matches = resolve_cross_source(entities_by_source)
        assert len(matches) == 0

    def test_multiple_entities_mixed_matches(self) -> None:
        """Some entities match across sources, others don't."""
        entities_by_source = {
            "gmail": [
                _entity("Python", "Technology"),
                _entity("Docker", "Technology"),
            ],
            "repositories": [
                _entity("Python", "Technology"),
                _entity("Rust", "Technology"),
            ],
        }
        matches = resolve_cross_source(entities_by_source)
        assert len(matches) == 1
        assert matches[0].entity_a == "Python"

    def test_empty_sources(self) -> None:
        """Empty entity lists produce no matches."""
        entities_by_source = {
            "gmail": [],
            "repositories": [],
        }
        matches = resolve_cross_source(entities_by_source)
        assert len(matches) == 0

    def test_email_match_case_insensitive(self) -> None:
        """Email matching should be case-insensitive."""
        entities_by_source = {
            "gmail": [{"name": "Bob", "type": "Person", "confidence": 0.9, "email": "Bob@Example.COM"}],
            "repositories": [{"name": "bob", "type": "Person", "confidence": 0.9, "email": "bob@example.com"}],
        }
        matches = resolve_cross_source(entities_by_source)
        # Should match on either exact name or email
        assert len(matches) == 1

    def test_invalid_email_not_used_for_matching(self) -> None:
        """Invalid emails should be ignored in cross-source matching."""
        entities_by_source = {
            "gmail": [{"name": "Alice", "type": "Person", "confidence": 0.9, "email": "not-an-email"}],
            "repositories": [{"name": "Bob", "type": "Person", "confidence": 0.9, "email": "not-an-email"}],
        }
        matches = resolve_cross_source(entities_by_source)
        assert len(matches) == 0

    def test_overlong_email_rejected(self) -> None:
        """Emails exceeding RFC 5321 max length (254) should be rejected."""
        long_email = "a" * 245 + "@test.com"  # 254 chars
        entities_by_source = {
            "gmail": [{"name": "A", "type": "Person", "confidence": 0.9, "email": long_email}],
            "repositories": [{"name": "B", "type": "Person", "confidence": 0.9, "email": long_email}],
        }
        matches = resolve_cross_source(entities_by_source)
        # 254 chars is exactly the limit, should work
        assert any(m.match_type == "email" for m in matches)

        too_long_email = "a" * 246 + "@test.com"  # 255 chars
        entities_by_source2 = {
            "gmail": [{"name": "C", "type": "Person", "confidence": 0.9, "email": too_long_email}],
            "repositories": [{"name": "D", "type": "Person", "confidence": 0.9, "email": too_long_email}],
        }
        matches2 = resolve_cross_source(entities_by_source2)
        assert not any(m.match_type == "email" for m in matches2)


# ------------------------------------------------------------------
# Confidence clamping
# ------------------------------------------------------------------


class TestConfidenceClamping:
    def test_clamp_high_value(self) -> None:
        assert _clamp_confidence(999) == 1.0

    def test_clamp_negative_value(self) -> None:
        assert _clamp_confidence(-0.5) == 0.0

    def test_clamp_normal_value(self) -> None:
        assert _clamp_confidence(0.8) == 0.8

    def test_clamp_nan(self) -> None:
        assert _clamp_confidence(float("nan")) == 0.75

    def test_clamp_non_numeric(self) -> None:
        assert _clamp_confidence("invalid") == 0.75

    def test_clamp_none(self) -> None:
        assert _clamp_confidence(None) == 0.75

    def test_high_confidence_clamped_in_resolve(self) -> None:
        """Confidence of 999 should be clamped to 1.0 during resolution."""
        extracted = [_entity("X", "Concept", 999)]
        existing = [_entity("X", "Concept", 0.5)]
        result = resolve_entities(extracted, existing)
        assert result.entities[0].confidence <= 1.0


# ------------------------------------------------------------------
# Email validation
# ------------------------------------------------------------------


class TestEmailValidation:
    def test_valid_email(self) -> None:
        assert _is_valid_email("user@example.com") is True

    def test_invalid_no_at(self) -> None:
        assert _is_valid_email("userexample.com") is False

    def test_invalid_no_domain(self) -> None:
        assert _is_valid_email("user@") is False

    def test_invalid_empty(self) -> None:
        assert _is_valid_email("") is False

    def test_invalid_spaces(self) -> None:
        assert _is_valid_email("user @example.com") is False


# ------------------------------------------------------------------
# NaN vector handling
# ------------------------------------------------------------------


class TestNaNVectorHandling:
    def test_nan_vector_treated_as_new(self) -> None:
        """All-NaN embedding vectors should not crash or produce false matches."""
        import numpy as np

        vectors = {
            "Unknown": [float("nan")] * 3,
            "Known": [0.9, 0.1, 0.0],
        }
        embed_model = _mock_embed_model(vectors)

        extracted = [_entity("Unknown")]
        existing = [_entity("Known")]

        result = resolve_entities(extracted, existing, embed_model=embed_model)
        assert len(result.entities) == 1
        assert result.entities[0].same_as is None

    def test_zero_vector_no_false_match(self) -> None:
        """All-zero embedding vectors should not produce a false SAME_AS match."""
        vectors = {
            "ZeroVec": [0.0, 0.0, 0.0],
            "Known": [0.9, 0.1, 0.0],
        }
        embed_model = _mock_embed_model(vectors)

        extracted = [_entity("ZeroVec")]
        existing = [_entity("Known")]

        result = resolve_entities(extracted, existing, embed_model=embed_model)
        assert len(result.entities) == 1
        # Zero vector normalized to itself (0/1=0), dot product is 0, below threshold
        assert result.entities[0].same_as is None
