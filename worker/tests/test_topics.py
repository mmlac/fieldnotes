"""Tests for the topics CLI command and query module."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from worker.cli import main, _build_parser
from worker.query.topics import (
    TopicDetail,
    TopicGap,
    TopicQuerier,
    TopicSummary,
    format_topic_detail,
    format_topic_gaps,
    format_topics_list,
)


# ------------------------------------------------------------------
# Parser tests
# ------------------------------------------------------------------


class TestTopicsParser:
    def test_topics_list(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["topics", "list"])
        assert args.command == "topics"
        assert args.topics_command == "list"
        assert args.json_output is False

    def test_topics_list_json(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["topics", "--json", "list"])
        assert args.json_output is True
        assert args.topics_command == "list"

    def test_topics_show(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["topics", "show", "machine-learning"])
        assert args.topics_command == "show"
        assert args.name == "machine-learning"

    def test_topics_gaps(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["topics", "gaps"])
        assert args.topics_command == "gaps"

    def test_topics_no_subcommand(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["topics"])
        assert args.command == "topics"
        assert args.topics_command is None


# ------------------------------------------------------------------
# Formatter tests
# ------------------------------------------------------------------


class TestFormatTopicsList:
    def test_empty(self) -> None:
        assert format_topics_list([]) == "No topics found."

    def test_grouped_by_source(self) -> None:
        topics = [
            TopicSummary("ML", "cluster", "Machine learning stuff", 5),
            TopicSummary("Python", "user", "Python language", 3),
        ]
        out = format_topics_list(topics)
        assert "Cluster-discovered" in out
        assert "User-defined" in out
        assert "ML" in out
        assert "Python" in out

    def test_json_output(self) -> None:
        topics = [TopicSummary("ML", "cluster", "desc", 2)]
        out = format_topics_list(topics, use_json=True)
        data = json.loads(out)
        assert len(data) == 1
        assert data[0]["name"] == "ML"
        assert data[0]["doc_count"] == 2

    def test_singular_doc(self) -> None:
        topics = [TopicSummary("Solo", "user", "", 1)]
        out = format_topics_list(topics)
        assert "1" in out
        assert "doc" in out


class TestFormatTopicDetail:
    def test_not_found(self) -> None:
        assert format_topic_detail(None) == "Topic not found."

    def test_with_documents(self) -> None:
        detail = TopicDetail(
            name="ML",
            source="cluster",
            description="Machine learning",
            documents=[
                {"source_id": "a.md", "name": "a.md", "type": "File"},
            ],
        )
        out = format_topic_detail(detail)
        assert "ML" in out
        assert "cluster-discovered" in out
        assert "a.md" in out

    def test_json_output(self) -> None:
        detail = TopicDetail(
            name="ML", source="user", description="desc", documents=[]
        )
        out = format_topic_detail(detail, use_json=True)
        data = json.loads(out)
        assert data["name"] == "ML"
        assert data["source"] == "user"

    def test_no_documents(self) -> None:
        detail = TopicDetail(name="Empty", source="user", description="", documents=[])
        out = format_topic_detail(detail)
        assert "No linked documents" in out


class TestFormatTopicGaps:
    def test_no_gaps(self) -> None:
        out = format_topic_gaps([])
        assert "No gaps" in out

    def test_with_gaps(self) -> None:
        gaps = [TopicGap("NewTopic", "Something new", 3)]
        out = format_topic_gaps(gaps)
        assert "Gaps in your thinking" in out
        assert "NewTopic" in out

    def test_json_output(self) -> None:
        gaps = [TopicGap("X", "desc", 1)]
        out = format_topic_gaps(gaps, use_json=True)
        data = json.loads(out)
        assert data[0]["name"] == "X"


# ------------------------------------------------------------------
# CLI integration (mocked TopicQuerier)
# ------------------------------------------------------------------


def _mock_querier(
    topics: list[TopicSummary] | None = None,
    detail: TopicDetail | None = None,
    gaps: list[TopicGap] | None = None,
) -> MagicMock:
    q = MagicMock(spec=TopicQuerier)
    q.list_topics.return_value = topics or []
    q.show_topic.return_value = detail
    q.topic_gaps.return_value = gaps or []
    q.close.return_value = None
    q.__enter__ = MagicMock(return_value=q)
    q.__exit__ = MagicMock(return_value=False)
    return q


class TestTopicsCLI:
    @patch("worker.cli.load_config")
    @patch("worker.query.topics.TopicQuerier")
    def test_list(
        self,
        mock_tq_cls: MagicMock,
        mock_load: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_load.return_value = MagicMock()
        mock_tq_cls.return_value = _mock_querier(
            topics=[TopicSummary("ML", "cluster", "Machine learning", 5)]
        )

        rc = main(["topics", "list"])

        assert rc == 0
        out = capsys.readouterr().out
        assert "ML" in out

    @patch("worker.cli.load_config")
    @patch("worker.query.topics.TopicQuerier")
    def test_show(
        self,
        mock_tq_cls: MagicMock,
        mock_load: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_load.return_value = MagicMock()
        detail = TopicDetail("ML", "cluster", "desc", [])
        mock_tq_cls.return_value = _mock_querier(detail=detail)

        rc = main(["topics", "show", "ML"])

        assert rc == 0
        out = capsys.readouterr().out
        assert "ML" in out

    @patch("worker.cli.load_config")
    @patch("worker.query.topics.TopicQuerier")
    def test_gaps(
        self,
        mock_tq_cls: MagicMock,
        mock_load: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_load.return_value = MagicMock()
        mock_tq_cls.return_value = _mock_querier(
            gaps=[TopicGap("NewTopic", "A new topic", 2)]
        )

        rc = main(["topics", "gaps"])

        assert rc == 0
        out = capsys.readouterr().out
        assert "NewTopic" in out

    @patch("worker.cli.load_config")
    @patch("worker.query.topics.TopicQuerier")
    def test_json_flag(
        self,
        mock_tq_cls: MagicMock,
        mock_load: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_load.return_value = MagicMock()
        mock_tq_cls.return_value = _mock_querier(
            topics=[TopicSummary("ML", "cluster", "desc", 1)]
        )

        rc = main(["topics", "--json", "list"])

        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data[0]["name"] == "ML"

    def test_no_subcommand_returns_1(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # topics with no subcommand should show usage and return 1
        # We need to mock load_config since it's called in _run_topics
        with patch("worker.cli.load_config") as mock_load, \
             patch("worker.query.topics.TopicQuerier"):
            mock_load.return_value = MagicMock()
            rc = main(["topics"])
            assert rc == 1
            err = capsys.readouterr().err
            assert "Usage" in err
