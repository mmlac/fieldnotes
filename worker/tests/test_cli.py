"""Tests for the CLI search command."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from worker.cli import main, _build_parser, _run_search
from worker.query.graph import GraphQueryResult
from worker.query.vector import VectorQueryResult, VectorResult


# ------------------------------------------------------------------
# Parser tests
# ------------------------------------------------------------------


class TestParser:
    def test_search_basic(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["search", "hello", "world"])
        assert args.command == "search"
        assert args.query == ["hello", "world"]
        assert args.top_k == 10

    def test_search_custom_top_k(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["search", "-k", "5", "some query"])
        assert args.top_k == 5

    def test_config_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["-c", "/tmp/test.toml", "search", "q"])
        assert str(args.config) == "/tmp/test.toml"

    def test_no_command_returns_1(self) -> None:
        assert main([]) == 1


# ------------------------------------------------------------------
# Search integration (mocked backends)
# ------------------------------------------------------------------


def _mock_graph_querier(result: GraphQueryResult) -> MagicMock:
    q = MagicMock()
    q.query.return_value = result
    q.close.return_value = None
    return q


def _mock_vector_querier(result: VectorQueryResult) -> MagicMock:
    q = MagicMock()
    q.query.return_value = result
    q.close.return_value = None
    return q


class TestRunSearch:
    @patch("worker.cli.VectorQuerier")
    @patch("worker.cli.GraphQuerier")
    @patch("worker.cli.ModelRegistry")
    @patch("worker.cli.load_config")
    def test_prints_context(
        self,
        mock_load: MagicMock,
        mock_reg: MagicMock,
        mock_gq_cls: MagicMock,
        mock_vq_cls: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_load.return_value = MagicMock()
        mock_gq_cls.return_value = _mock_graph_querier(
            GraphQueryResult(
                question="test",
                cypher="MATCH ...",
                raw_results=[{"n": {"source_id": "a.md", "name": "A"}}],
                answer="Found A.",
            )
        )
        mock_vq_cls.return_value = _mock_vector_querier(
            VectorQueryResult(
                question="test",
                results=[
                    VectorResult(
                        source_type="file",
                        source_id="b.md",
                        text="some text",
                        date="2026-03-11",
                        score=0.9,
                    )
                ],
            )
        )

        rc = _run_search("test", config_path=None, top_k=10)

        assert rc == 0
        out = capsys.readouterr().out
        assert "[Graph context]" in out
        assert "[Semantic context]" in out

    @patch("worker.cli.VectorQuerier")
    @patch("worker.cli.GraphQuerier")
    @patch("worker.cli.ModelRegistry")
    @patch("worker.cli.load_config")
    def test_no_results(
        self,
        mock_load: MagicMock,
        mock_reg: MagicMock,
        mock_gq_cls: MagicMock,
        mock_vq_cls: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_load.return_value = MagicMock()
        mock_gq_cls.return_value = _mock_graph_querier(
            GraphQueryResult(question="q", cypher="")
        )
        mock_vq_cls.return_value = _mock_vector_querier(
            VectorQueryResult(question="q")
        )

        rc = _run_search("q", config_path=None, top_k=10)

        assert rc == 0
        assert "No results found." in capsys.readouterr().out

    @patch("worker.cli.VectorQuerier")
    @patch("worker.cli.GraphQuerier")
    @patch("worker.cli.ModelRegistry")
    @patch("worker.cli.load_config")
    def test_errors_on_stderr(
        self,
        mock_load: MagicMock,
        mock_reg: MagicMock,
        mock_gq_cls: MagicMock,
        mock_vq_cls: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_load.return_value = MagicMock()
        mock_gq_cls.return_value = _mock_graph_querier(
            GraphQueryResult(question="q", cypher="", error="neo4j down")
        )
        mock_vq_cls.return_value = _mock_vector_querier(
            VectorQueryResult(
                question="q",
                results=[
                    VectorResult(
                        source_type="file",
                        source_id="b.md",
                        text="text",
                        date="",
                        score=0.8,
                    )
                ],
            )
        )

        rc = _run_search("q", config_path=None, top_k=10)

        assert rc == 0
        captured = capsys.readouterr()
        assert "warning: graph: neo4j down" in captured.err

    @patch("worker.cli.VectorQuerier")
    @patch("worker.cli.GraphQuerier")
    @patch("worker.cli.ModelRegistry")
    @patch("worker.cli.load_config")
    def test_close_called_on_success(
        self,
        mock_load: MagicMock,
        mock_reg: MagicMock,
        mock_gq_cls: MagicMock,
        mock_vq_cls: MagicMock,
    ) -> None:
        mock_load.return_value = MagicMock()
        gq = _mock_graph_querier(GraphQueryResult(question="q", cypher=""))
        vq = _mock_vector_querier(VectorQueryResult(question="q"))
        mock_gq_cls.return_value = gq
        mock_vq_cls.return_value = vq

        _run_search("q", config_path=None, top_k=10)

        gq.close.assert_called_once()
        vq.close.assert_called_once()


class TestMainSearch:
    @patch("worker.cli.VectorQuerier")
    @patch("worker.cli.GraphQuerier")
    @patch("worker.cli.ModelRegistry")
    @patch("worker.cli.load_config")
    def test_main_search_joins_query(
        self,
        mock_load: MagicMock,
        mock_reg: MagicMock,
        mock_gq_cls: MagicMock,
        mock_vq_cls: MagicMock,
    ) -> None:
        mock_load.return_value = MagicMock()
        gq = _mock_graph_querier(GraphQueryResult(question="q", cypher=""))
        vq = _mock_vector_querier(VectorQueryResult(question="q"))
        mock_gq_cls.return_value = gq
        mock_vq_cls.return_value = vq

        rc = main(["search", "hello", "world"])

        assert rc == 0
        gq.query.assert_called_once_with("hello world")
