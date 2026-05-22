"""Tests for vector post-filtering in hybrid.merge() — date window and journal folder."""

from __future__ import annotations

from datetime import date

from worker.query.graph import GraphQueryResult
from worker.query.hybrid import merge
from worker.query.vector import VectorQueryResult, VectorResult


def _vr(source_id: str, source_type: str = "files", date_str: str = "") -> VectorResult:
    return VectorResult(
        source_id=source_id,
        source_type=source_type,
        text=f"text for {source_id}",
        date=date_str,
        score=1.0,
    )


def _empty_graph(question: str = "q") -> GraphQueryResult:
    return GraphQueryResult(question=question, cypher="", raw_results=[])


def test_no_filter_when_window_and_folders_are_none() -> None:
    """Pre-existing behaviour must be preserved when no filter args are passed."""
    vec = VectorQueryResult(
        question="q",
        results=[
            _vr("a"),
            _vr("b"),
            _vr("c"),
        ],
    )
    out = merge("q", _empty_graph(), vec)
    assert [r.source_id for r in out.vector_results] == ["a", "b", "c"]


def test_date_window_drops_out_of_range_results() -> None:
    vec = VectorQueryResult(
        question="q",
        results=[
            _vr("in1", date_str="2026-05-19"),       # inside
            _vr("in2", date_str="2026-05-21"),       # inside (boundary)
            _vr("out_old", date_str="2026-01-20"),   # outside (way old)
            _vr("out_new", date_str="2027-01-01"),   # outside (future)
        ],
    )
    window = (date(2026, 5, 16), date(2026, 5, 22))
    out = merge("q", _empty_graph(), vec, date_window=window)
    assert sorted(r.source_id for r in out.vector_results) == ["in1", "in2"]


def test_date_window_keeps_results_with_unparseable_date() -> None:
    """Best-effort strict: don't drop a result because we couldn't parse its date."""
    vec = VectorQueryResult(
        question="q",
        results=[
            _vr("dated_in", date_str="2026-05-20"),
            _vr("undated", date_str=""),
            _vr("garbage", date_str="not-a-date"),
        ],
    )
    window = (date(2026, 5, 16), date(2026, 5, 22))
    out = merge("q", _empty_graph(), vec, date_window=window)
    kept = {r.source_id for r in out.vector_results}
    assert kept == {"dated_in", "undated", "garbage"}


def test_date_window_accepts_iso_datetime_form() -> None:
    vec = VectorQueryResult(
        question="q",
        results=[
            _vr("dt", date_str="2026-05-20T14:00:00Z"),
            _vr("dt_out", date_str="2024-05-20T14:00:00+00:00"),
        ],
    )
    window = (date(2026, 5, 16), date(2026, 5, 22))
    out = merge("q", _empty_graph(), vec, date_window=window)
    assert [r.source_id for r in out.vector_results] == ["dt"]


def test_journal_folder_filter_drops_non_matching_paths() -> None:
    """Reproduces the user's report: random PDF + random gmail get dropped."""
    vec = VectorQueryResult(
        question="summarize my journal last 7 days",
        results=[
            _vr(
                "/Users/me/Vault/Journal/2026-05-19.md",
                date_str="2026-05-19",
            ),
            _vr(
                "/Users/me/Other/Vault/01 Daily/2026-05-21.md",
                date_str="2026-05-21",
            ),
            _vr(
                "/Users/me/Dropbox/Books/Planning The Perfect Date.pdf",
                date_str="2026-05-20",
            ),
            _vr(
                "gmail://personal/message/19d9c5c78695eee4",
                source_type="gmail",
                date_str="2026-05-20",
            ),
        ],
    )
    out = merge(
        "q",
        _empty_graph(),
        vec,
        require_journal_folder=["/Journal/", "/01 Daily/"],
    )
    kept = {r.source_id for r in out.vector_results}
    assert kept == {
        "/Users/me/Vault/Journal/2026-05-19.md",
        "/Users/me/Other/Vault/01 Daily/2026-05-21.md",
    }


def test_journal_folder_filter_is_case_insensitive() -> None:
    vec = VectorQueryResult(
        question="q",
        results=[
            _vr("/users/me/JOURNAL/2026-05-20.md"),
            _vr("/users/me/journal/2026-05-21.md"),
            _vr("/users/me/Other/2026-05-21.md"),
        ],
    )
    out = merge(
        "q",
        _empty_graph(),
        vec,
        require_journal_folder=["/Journal/"],
    )
    assert len(out.vector_results) == 2


def test_date_and_folder_filters_compose() -> None:
    vec = VectorQueryResult(
        question="q",
        results=[
            _vr("/v/Journal/2026-05-19.md", date_str="2026-05-19"),  # kept
            _vr("/v/Journal/2026-01-20.md", date_str="2026-01-20"),  # dropped (date)
            _vr("/v/Other/2026-05-19.md", date_str="2026-05-19"),    # dropped (folder)
        ],
    )
    out = merge(
        "q",
        _empty_graph(),
        vec,
        date_window=(date(2026, 5, 16), date(2026, 5, 22)),
        require_journal_folder=["/Journal/"],
    )
    assert [r.source_id for r in out.vector_results] == [
        "/v/Journal/2026-05-19.md",
    ]


def test_graph_results_are_never_filtered_by_post_filter() -> None:
    """Graph rows are the precision lane; the post-filter only touches vector."""
    graph = GraphQueryResult(
        question="q",
        cypher="...",
        raw_results=[
            {"source_id": "/v/Other/2026-05-19.md", "modified_at": "2026-05-19"},
        ],
    )
    vec = VectorQueryResult(
        question="q",
        results=[
            _vr("/v/Other/2026-05-19.md", date_str="2026-05-19"),  # would be dropped …
        ],
    )
    out = merge(
        "q",
        graph,
        vec,
        require_journal_folder=["/Journal/"],
    )
    # Graph row preserved (in raw_results); vector row dropped (post-filter and/or dedup).
    assert len(out.graph_results) == 1
    assert out.vector_results == []
