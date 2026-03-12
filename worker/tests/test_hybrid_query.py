"""Tests for the hybrid query merge layer."""

from __future__ import annotations


from worker.query.graph import GraphQueryResult
from worker.query.vector import VectorQueryResult, VectorResult
from worker.query.hybrid import HybridResult, merge


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _vr(
    source_id: str = "doc.md",
    text: str = "chunk",
    score: float = 0.9,
    source_type: str = "file",
    date: str = "2026-03-01",
) -> VectorResult:
    return VectorResult(
        source_type=source_type,
        source_id=source_id,
        text=text,
        date=date,
        score=score,
    )


# ------------------------------------------------------------------
# merge()
# ------------------------------------------------------------------


class TestMerge:
    def test_basic_merge(self) -> None:
        graph = GraphQueryResult(
            question="q",
            cypher="MATCH (n) RETURN n",
            raw_results=[{"n": {"source_id": "a.md", "title": "A"}}],
            answer="Found A.",
        )
        vector = VectorQueryResult(
            question="q",
            results=[
                _vr(source_id="a.md", text="duplicate"),
                _vr(source_id="b.md", text="unique"),
            ],
        )

        result = merge("q", graph, vector)

        assert result.question == "q"
        assert len(result.graph_results) == 1
        # a.md should be deduped from vector results
        assert len(result.vector_results) == 1
        assert result.vector_results[0].source_id == "b.md"
        assert result.errors == []

    def test_graph_ranked_first_in_context(self) -> None:
        graph = GraphQueryResult(
            question="q",
            cypher="MATCH ...",
            raw_results=[{"n": {"source_id": "g.md", "name": "G"}}],
            answer="Graph answer.",
        )
        vector = VectorQueryResult(
            question="q",
            results=[_vr(source_id="v.md", text="vector text")],
        )

        result = merge("q", graph, vector)

        assert result.context.index("[Graph context]") < result.context.index(
            "[Semantic context]"
        )

    def test_dedup_by_source_id(self) -> None:
        graph = GraphQueryResult(
            question="q",
            cypher="...",
            raw_results=[
                {"n": {"source_id": "x.md"}},
                {"n": {"source_id": "y.md"}},
            ],
        )
        vector = VectorQueryResult(
            question="q",
            results=[
                _vr(source_id="x.md"),
                _vr(source_id="y.md"),
                _vr(source_id="z.md"),
            ],
        )

        result = merge("q", graph, vector)

        assert len(result.vector_results) == 1
        assert result.vector_results[0].source_id == "z.md"

    def test_top_level_source_id_dedup(self) -> None:
        """Graph results with top-level source_id should also dedup."""
        graph = GraphQueryResult(
            question="q",
            cypher="...",
            raw_results=[{"source_id": "a.md", "title": "A"}],
        )
        vector = VectorQueryResult(
            question="q",
            results=[_vr(source_id="a.md"), _vr(source_id="b.md")],
        )

        result = merge("q", graph, vector)

        assert len(result.vector_results) == 1
        assert result.vector_results[0].source_id == "b.md"

    def test_empty_graph_results(self) -> None:
        graph = GraphQueryResult(question="q", cypher="...")
        vector = VectorQueryResult(
            question="q",
            results=[_vr(source_id="a.md"), _vr(source_id="b.md")],
        )

        result = merge("q", graph, vector)

        assert len(result.vector_results) == 2
        assert "[Graph context]" not in result.context
        assert "[Semantic context]" in result.context

    def test_empty_vector_results(self) -> None:
        graph = GraphQueryResult(
            question="q",
            cypher="...",
            raw_results=[{"n": {"source_id": "a.md"}}],
            answer="Found A.",
        )
        vector = VectorQueryResult(question="q")

        result = merge("q", graph, vector)

        assert result.vector_results == []
        assert "[Graph context]" in result.context
        assert "[Semantic context]" not in result.context

    def test_both_empty(self) -> None:
        graph = GraphQueryResult(question="q", cypher="")
        vector = VectorQueryResult(question="q")

        result = merge("q", graph, vector)

        assert result.context == ""
        assert result.errors == []

    def test_errors_collected(self) -> None:
        graph = GraphQueryResult(question="q", cypher="", error="neo4j down")
        vector = VectorQueryResult(question="q", error="qdrant down")

        result = merge("q", graph, vector)

        assert len(result.errors) == 2
        assert "graph: neo4j down" in result.errors
        assert "vector: qdrant down" in result.errors

    def test_graph_error_only(self) -> None:
        graph = GraphQueryResult(question="q", cypher="", error="fail")
        vector = VectorQueryResult(
            question="q",
            results=[_vr(source_id="a.md", text="still works")],
        )

        result = merge("q", graph, vector)

        assert len(result.errors) == 1
        assert len(result.vector_results) == 1
        assert "[Semantic context]" in result.context

    def test_vector_section_includes_metadata(self) -> None:
        graph = GraphQueryResult(question="q", cypher="")
        vector = VectorQueryResult(
            question="q",
            results=[
                _vr(
                    source_id="notes/march.md",
                    source_type="file",
                    date="2026-03-11",
                    text="important notes",
                )
            ],
        )

        result = merge("q", graph, vector)

        assert "[file] notes/march.md" in result.context
        assert "(2026-03-11)" in result.context
        assert "important notes" in result.context


class TestHybridResult:
    def test_defaults(self) -> None:
        r = HybridResult(question="q")
        assert r.question == "q"
        assert r.graph_results == []
        assert r.vector_results == []
        assert r.context == ""
        assert r.errors == []
