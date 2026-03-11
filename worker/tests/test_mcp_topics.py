"""Tests for MCP topic tools (list_topics, show_topic, topic_gaps)."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from worker.config import Config, Neo4jConfig, QdrantConfig
from worker.mcp_server import FieldnotesServer
from worker.query.topics import TopicDetail, TopicGap, TopicSummary


def _make_server() -> FieldnotesServer:
    cfg = Config(
        neo4j=Neo4jConfig(uri="bolt://localhost:7687", user="neo4j", password="test"),
        qdrant=QdrantConfig(host="localhost", port=6333, collection="fieldnotes"),
    )
    return FieldnotesServer(cfg)


def _run(coro):
    return asyncio.run(coro)


def _mock_querier(
    topics: list[TopicSummary] | None = None,
    detail: TopicDetail | None = None,
    gaps: list[TopicGap] | None = None,
) -> MagicMock:
    q = MagicMock()
    q.list_topics.return_value = topics or []
    q.show_topic.return_value = detail
    q.topic_gaps.return_value = gaps or []
    q.__enter__ = MagicMock(return_value=q)
    q.__exit__ = MagicMock(return_value=False)
    return q


# ------------------------------------------------------------------
# list_topics
# ------------------------------------------------------------------


class TestListTopicsTool:
    @patch("worker.mcp_server.TopicQuerier")
    def test_list_all_topics(self, mock_tq_cls: MagicMock) -> None:
        mock_tq_cls.return_value = _mock_querier(
            topics=[
                TopicSummary("ML", "cluster", "Machine learning", 5),
                TopicSummary("Python", "user", "Python lang", 3),
            ]
        )

        server = _make_server()
        result = _run(server._call_tool("list_topics", {}))

        assert len(result) == 1
        data = json.loads(result[0].text)
        assert len(data) == 2
        names = {t["name"] for t in data}
        assert names == {"ML", "Python"}

    @patch("worker.mcp_server.TopicQuerier")
    def test_list_topics_with_source_filter(self, mock_tq_cls: MagicMock) -> None:
        mock_tq_cls.return_value = _mock_querier(
            topics=[
                TopicSummary("ML", "cluster", "Machine learning", 5),
                TopicSummary("Python", "user", "Python lang", 3),
            ]
        )

        server = _make_server()
        result = _run(server._call_tool("list_topics", {"source": "cluster"}))

        data = json.loads(result[0].text)
        assert len(data) == 1
        assert data[0]["name"] == "ML"

    @patch("worker.mcp_server.TopicQuerier")
    def test_list_topics_user_filter(self, mock_tq_cls: MagicMock) -> None:
        mock_tq_cls.return_value = _mock_querier(
            topics=[
                TopicSummary("ML", "cluster", "Machine learning", 5),
                TopicSummary("Python", "user", "Python lang", 3),
            ]
        )

        server = _make_server()
        result = _run(server._call_tool("list_topics", {"source": "user"}))

        data = json.loads(result[0].text)
        assert len(data) == 1
        assert data[0]["name"] == "Python"

    @patch("worker.mcp_server.TopicQuerier")
    def test_list_topics_empty(self, mock_tq_cls: MagicMock) -> None:
        mock_tq_cls.return_value = _mock_querier(topics=[])

        server = _make_server()
        result = _run(server._call_tool("list_topics", {}))

        data = json.loads(result[0].text)
        assert data == []


# ------------------------------------------------------------------
# show_topic
# ------------------------------------------------------------------


class TestShowTopicTool:
    @patch("worker.mcp_server.TopicQuerier")
    def test_show_existing_topic(self, mock_tq_cls: MagicMock) -> None:
        detail = TopicDetail(
            name="ML",
            source="cluster",
            description="Machine learning research",
            documents=[
                {"source_id": "paper1.md", "name": "paper1.md", "type": "File"},
            ],
        )
        mock_tq_cls.return_value = _mock_querier(detail=detail)

        server = _make_server()
        result = _run(server._call_tool("show_topic", {"name": "ML"}))

        data = json.loads(result[0].text)
        assert data["name"] == "ML"
        assert data["source"] == "cluster"
        assert len(data["documents"]) == 1

    @patch("worker.mcp_server.TopicQuerier")
    def test_show_nonexistent_topic(self, mock_tq_cls: MagicMock) -> None:
        mock_tq_cls.return_value = _mock_querier(detail=None)

        server = _make_server()
        result = _run(server._call_tool("show_topic", {"name": "nonexistent"}))

        assert "not found" in result[0].text.lower() or result[0].text == "Topic not found."


# ------------------------------------------------------------------
# topic_gaps
# ------------------------------------------------------------------


class TestTopicGapsTool:
    @patch("worker.mcp_server.TopicQuerier")
    def test_gaps_with_results(self, mock_tq_cls: MagicMock) -> None:
        mock_tq_cls.return_value = _mock_querier(
            gaps=[TopicGap("NewTopic", "A new discovery", 4)]
        )

        server = _make_server()
        result = _run(server._call_tool("topic_gaps", {}))

        data = json.loads(result[0].text)
        assert len(data) == 1
        assert data[0]["name"] == "NewTopic"
        assert data[0]["doc_count"] == 4

    @patch("worker.mcp_server.TopicQuerier")
    def test_gaps_empty(self, mock_tq_cls: MagicMock) -> None:
        mock_tq_cls.return_value = _mock_querier(gaps=[])

        server = _make_server()
        result = _run(server._call_tool("topic_gaps", {}))

        data = json.loads(result[0].text)
        assert data == []
