"""Tests for the second-stage reranker."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from worker.config import RerankerConfig
from worker.query.hybrid import merge
from worker.query.graph import GraphQueryResult
from worker.query.reranker import (
    CrossEncoderReranker,
    NullReranker,
    RerankCandidate,
    build_reranker,
    candidates_from_vector_results,
)
from worker.query.vector import VectorQueryResult, VectorResult


def _vr(source_id: str, text: str, score: float, idx: int = 0) -> VectorResult:
    return VectorResult(
        source_type="file",
        source_id=source_id,
        text=text,
        date="2026-04-18",
        score=score,
        chunk_index=idx,
    )


def _candidate(source_id: str, text: str, original: float = 0.0) -> RerankCandidate:
    return RerankCandidate(
        text=text,
        payload=_vr(source_id, text, original),
        original_score=original,
    )


# ------------------------------------------------------------------
# NullReranker
# ------------------------------------------------------------------


class TestNullReranker:
    def test_passes_through_in_order(self):
        cands = [
            _candidate("a", "alpha", 0.9),
            _candidate("b", "beta", 0.8),
            _candidate("c", "gamma", 0.7),
        ]
        result = NullReranker().rerank("q", cands, top_k=10)
        assert [c.payload.source_id for c in result] == ["a", "b", "c"]

    def test_truncates_to_top_k(self):
        cands = [_candidate(f"id{i}", f"t{i}") for i in range(5)]
        result = NullReranker().rerank("q", cands, top_k=2)
        assert len(result) == 2
        assert [c.payload.source_id for c in result] == ["id0", "id1"]

    def test_empty_input(self):
        assert NullReranker().rerank("q", [], top_k=10) == []


# ------------------------------------------------------------------
# CrossEncoderReranker
# ------------------------------------------------------------------


class TestCrossEncoderReranker:
    def test_reorders_by_score_desc(self):
        model = MagicMock()
        model.rerank.return_value = [0.1, 0.9, 0.5]  # b should win, then c, then a
        rr = CrossEncoderReranker(model)
        cands = [
            _candidate("a", "alpha", 0.9),
            _candidate("b", "beta", 0.8),
            _candidate("c", "gamma", 0.7),
        ]
        result = rr.rerank("q", cands, top_k=10)
        assert [c.payload.source_id for c in result] == ["b", "c", "a"]
        # rerank_score is propagated
        assert result[0].rerank_score == pytest.approx(0.9)

    def test_truncates_to_top_k_after_reorder(self):
        model = MagicMock()
        model.rerank.return_value = [0.1, 0.9, 0.5, 0.95]
        rr = CrossEncoderReranker(model)
        cands = [
            _candidate("a", "alpha"),
            _candidate("b", "beta"),
            _candidate("c", "gamma"),
            _candidate("d", "delta"),
        ]
        result = rr.rerank("q", cands, top_k=2)
        assert [c.payload.source_id for c in result] == ["d", "b"]

    def test_score_threshold_drops_candidates(self):
        model = MagicMock()
        model.rerank.return_value = [0.05, 0.6, 0.9]
        rr = CrossEncoderReranker(model, score_threshold=0.5)
        cands = [
            _candidate("a", "alpha"),
            _candidate("b", "beta"),
            _candidate("c", "gamma"),
        ]
        result = rr.rerank("q", cands, top_k=10)
        # 'a' (0.05) is below threshold and gets dropped.
        assert [c.payload.source_id for c in result] == ["c", "b"]

    def test_provider_failure_falls_back_to_original_order(self):
        """A wedged reranker must never break search."""
        model = MagicMock()
        model.rerank.side_effect = RuntimeError("backend exploded")
        rr = CrossEncoderReranker(model)
        cands = [
            _candidate("a", "alpha"),
            _candidate("b", "beta"),
            _candidate("c", "gamma"),
        ]
        result = rr.rerank("q", cands, top_k=2)
        # Original order preserved, truncated to top_k.
        assert [c.payload.source_id for c in result] == ["a", "b"]

    def test_score_count_mismatch_falls_back(self):
        """If the provider returns the wrong number of scores, don't trust it."""
        model = MagicMock()
        model.rerank.return_value = [0.5]  # 1 score for 3 candidates
        rr = CrossEncoderReranker(model)
        cands = [
            _candidate("a", "alpha"),
            _candidate("b", "beta"),
            _candidate("c", "gamma"),
        ]
        result = rr.rerank("q", cands, top_k=10)
        assert [c.payload.source_id for c in result] == ["a", "b", "c"]

    def test_batches_calls(self):
        """With batch_size < N, the provider should be called multiple times."""
        model = MagicMock()
        # 5 candidates, batch_size=2 → 3 calls (2+2+1).
        model.rerank.side_effect = [[0.5, 0.6], [0.7, 0.8], [0.9]]
        rr = CrossEncoderReranker(model, batch_size=2)
        cands = [_candidate(f"id{i}", f"t{i}") for i in range(5)]
        result = rr.rerank("q", cands, top_k=3)
        assert model.rerank.call_count == 3
        # Top 3 should be id4 (0.9), id3 (0.8), id2 (0.7)
        assert [c.payload.source_id for c in result] == ["id4", "id3", "id2"]

    def test_empty_input(self):
        model = MagicMock()
        rr = CrossEncoderReranker(model)
        assert rr.rerank("q", [], top_k=10) == []
        model.rerank.assert_not_called()

    def test_top_k_zero_returns_empty(self):
        model = MagicMock()
        rr = CrossEncoderReranker(model)
        cands = [_candidate("a", "alpha")]
        assert rr.rerank("q", cands, top_k=0) == []


# ------------------------------------------------------------------
# build_reranker
# ------------------------------------------------------------------


class TestBuildReranker:
    def test_returns_null_when_disabled(self):
        cfg = RerankerConfig(enabled=False)
        registry = MagicMock()
        rr = build_reranker(cfg, registry)
        assert isinstance(rr, NullReranker)
        registry.for_role.assert_not_called()

    def test_returns_null_when_no_rerank_role_bound(self):
        cfg = RerankerConfig(enabled=True)
        registry = MagicMock()
        registry.for_role.side_effect = KeyError("rerank")
        rr = build_reranker(cfg, registry)
        assert isinstance(rr, NullReranker)

    def test_returns_cross_encoder_when_role_bound(self):
        cfg = RerankerConfig(
            enabled=True, score_threshold=0.5, batch_size=16, top_k_post=5
        )
        registry = MagicMock()
        model = MagicMock()
        registry.for_role.return_value = model
        rr = build_reranker(cfg, registry)
        assert isinstance(rr, CrossEncoderReranker)
        # Threshold and batch_size are passed through.
        assert rr._score_threshold == 0.5
        assert rr._batch_size == 16


# ------------------------------------------------------------------
# Hybrid merge integration
# ------------------------------------------------------------------


class TestMergeWithReranker:
    def test_no_reranker_preserves_original_order(self):
        """Regression: existing call sites without reranker behave as before."""
        graph = GraphQueryResult(question="q", cypher="", raw_results=[])
        vector = VectorQueryResult(
            question="q",
            results=[
                _vr("a", "alpha", 0.9),
                _vr("b", "beta", 0.8),
            ],
        )
        result = merge("q", graph, vector)
        assert [v.source_id for v in result.vector_results] == ["a", "b"]

    def test_null_reranker_does_not_invoke_rerank(self):
        """NullReranker is bypassed (the merge() optimisation)."""
        graph = GraphQueryResult(question="q", cypher="", raw_results=[])
        vector = VectorQueryResult(
            question="q",
            results=[_vr("a", "alpha", 0.9), _vr("b", "beta", 0.8)],
        )
        result = merge("q", graph, vector, reranker=NullReranker(), top_k_post=1)
        # NullReranker is short-circuited, so the truncation does NOT happen.
        # This documents the intentional behaviour.
        assert [v.source_id for v in result.vector_results] == ["a", "b"]

    def test_reranker_reorders_vector_results(self):
        graph = GraphQueryResult(question="q", cypher="", raw_results=[])
        vector = VectorQueryResult(
            question="q",
            results=[
                _vr("a", "alpha", 0.9),
                _vr("b", "beta", 0.8),
                _vr("c", "gamma", 0.7),
            ],
        )

        class FixedReranker:
            def rerank(self, query, candidates, top_k):
                # Reverse + truncate.
                return list(reversed(candidates))[:top_k]

        result = merge(
            "q", graph, vector, reranker=FixedReranker(), top_k_post=2
        )
        assert [v.source_id for v in result.vector_results] == ["c", "b"]

    def test_reranker_does_not_touch_graph_results(self):
        """Graph results are the precision lane and stay as-is."""
        graph = GraphQueryResult(
            question="q",
            cypher="",
            raw_results=[{"name": "Alice", "source_id": "person:alice"}],
        )
        vector = VectorQueryResult(
            question="q",
            results=[_vr("a", "alpha", 0.9), _vr("b", "beta", 0.8)],
        )

        class CountingReranker:
            def __init__(self):
                self.passages_seen: list[str] = []

            def rerank(self, query, candidates, top_k):
                self.passages_seen = [c.text for c in candidates]
                return candidates[:top_k]

        rr = CountingReranker()
        merge("q", graph, vector, reranker=rr, top_k_post=10)
        # Reranker only saw vector passages, never graph rows.
        assert rr.passages_seen == ["alpha", "beta"]


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------


class TestCandidatesFromVectorResults:
    def test_preserves_text_and_score(self):
        vrs = [_vr("a", "alpha", 0.9), _vr("b", "beta", 0.8)]
        cands = candidates_from_vector_results(vrs)
        assert len(cands) == 2
        assert cands[0].text == "alpha"
        assert cands[0].original_score == 0.9
        assert cands[0].payload is vrs[0]
