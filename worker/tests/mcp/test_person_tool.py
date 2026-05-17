"""Tests for the MCP ``person`` tool handler."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from worker.config import Config, Neo4jConfig, QdrantConfig
from worker.mcp_server import TOOLS, FieldnotesServer
from worker.query.person import (
    FileMention,
    IdentityMember,
    Interaction,
    OpenTask,
    Person,
    PersonProfile,
    RelatedPerson,
    TopicCount,
)


def _make_server() -> FieldnotesServer:
    cfg = Config(
        neo4j=Neo4jConfig(uri="bolt://localhost:7687", user="neo4j", password="x"),
        qdrant=QdrantConfig(host="localhost", port=6333, collection="fieldnotes"),
    )
    return FieldnotesServer(cfg)


def _build_profile() -> PersonProfile:
    person = Person(id=1, email="alice@example.com", name="Alice Example")
    return PersonProfile(
        person=person,
        recent_interactions=[
            Interaction(
                timestamp="2026-04-26T12:00:00Z",
                source_type="gmail",
                title="Re: Q3 plan",
                snippet="Re: Q3 plan",
                edge_kind="SENT",
            )
        ],
        top_topics=[TopicCount(topic_name="planning", doc_count=3)],
        related_people=[
            RelatedPerson(name="Bob", email="bob@example.com", shared_count=4)
        ],
        open_tasks=[
            OpenTask(
                title="Follow up with Alice",
                project="Q3",
                tags=["@waiting"],
                flagged=True,
            )
        ],
        files=[
            FileMention(
                path="notes/alice.md",
                mtime="2026-04-20T08:00:00Z",
                source="obsidian",
            )
        ],
        identity_cluster=[
            IdentityMember(
                member="alice.alt@example.com",
                match_type="email",
                confidence=1.0,
                cross_source=False,
            )
        ],
    )


def test_person_tool_registered() -> None:
    names = {tool.name for tool in TOOLS}
    assert "person" in names

    person_tool = next(t for t in TOOLS if t.name == "person")
    desc = person_tool.description.lower()
    # Resolution order documented in the description.
    assert "email" in desc
    assert "slack" in desc
    assert "name" in desc  # fuzzy name


@pytest.mark.asyncio
@patch("worker.mcp_server.get_profile")
async def test_person_tool_returns_documented_schema(
    mock_get_profile: MagicMock,
) -> None:
    mock_get_profile.return_value = _build_profile()

    server = _make_server()
    result = await server._call_tool("person", {"identifier": "alice@example.com"})

    assert len(result) == 1
    payload = json.loads(result[0].text)

    expected_keys = {
        "identifier",
        "resolved",
        "sources_present",
        "last_seen",
        "total_interactions",
        "recent_interactions",
        "top_topics",
        "related_people",
        "open_tasks",
        "files_mentioning",
        "identity_cluster",
    }
    assert expected_keys <= set(payload.keys())

    assert payload["identifier"] == "alice@example.com"
    assert payload["resolved"]["email"] == "alice@example.com"
    assert payload["resolved"]["name"] == "Alice Example"
    assert payload["resolved"]["is_self"] is False

    assert isinstance(payload["sources_present"], list)
    assert "gmail" in payload["sources_present"]
    assert "file" in payload["sources_present"]
    assert "omnifocus" in payload["sources_present"]

    assert payload["last_seen"] == "2026-04-26T12:00:00Z"
    assert payload["total_interactions"] == 1

    assert len(payload["recent_interactions"]) == 1
    interaction = payload["recent_interactions"][0]
    assert interaction["edge_kind"] == "SENT"
    assert interaction["source_type"] == "gmail"

    assert payload["top_topics"][0]["topic_name"] == "planning"
    assert payload["related_people"][0]["email"] == "bob@example.com"
    assert payload["open_tasks"][0]["flagged"] is True
    assert payload["files_mentioning"][0]["path"] == "notes/alice.md"
    assert payload["identity_cluster"][0]["member"] == "alice.alt@example.com"


@pytest.mark.asyncio
@patch("worker.mcp_server.get_profile")
async def test_person_tool_unknown_identifier_returns_error(
    mock_get_profile: MagicMock,
) -> None:
    mock_get_profile.return_value = None

    server = _make_server()
    result = await server._call_tool("person", {"identifier": "nobody@example.com"})

    payload = json.loads(result[0].text)
    assert payload.get("error") is True
    assert "message" in payload
    assert "nobody@example.com" in payload["message"]


@pytest.mark.asyncio
@patch("worker.mcp_server.get_profile")
async def test_person_tool_ambiguous_name_returns_disambiguation_error(
    mock_get_profile: MagicMock,
) -> None:
    mock_get_profile.return_value = [
        Person(id=1, email="alice.a@example.com", name="Alice A"),
        Person(id=2, email="alice.b@example.com", name="Alice B"),
    ]

    server = _make_server()
    result = await server._call_tool("person", {"identifier": "alice"})

    payload = json.loads(result[0].text)
    assert payload.get("error") is True
    assert "message" in payload
    assert "candidates" in payload
    emails = {c["email"] for c in payload["candidates"]}
    assert emails == {"alice.a@example.com", "alice.b@example.com"}


@pytest.mark.asyncio
@patch("worker.mcp_server.generate_brief")
@patch("worker.mcp_server.build_driver")
@patch("worker.mcp_server.get_profile")
async def test_person_tool_summary_true_returns_next_brief(
    mock_get_profile: MagicMock,
    mock_graph_db: MagicMock,
    mock_generate_brief: MagicMock,
) -> None:
    mock_get_profile.return_value = _build_profile()
    mock_graph_db.return_value = MagicMock()
    mock_generate_brief.return_value = (
        "- [Open OmniFocus tasks] follow up with Alice",
        MagicMock(),
    )

    server = _make_server()
    result = await server._call_tool(
        "person",
        {"identifier": "alice@example.com", "summary": True},
    )

    payload = json.loads(result[0].text)
    assert payload.get("error") is not True
    assert "next_brief" in payload
    assert "follow up with Alice" in payload["next_brief"]


@pytest.mark.asyncio
@patch("worker.mcp_server.get_profile")
async def test_person_tool_summary_false_omits_next_brief(
    mock_get_profile: MagicMock,
) -> None:
    mock_get_profile.return_value = _build_profile()

    server = _make_server()
    result = await server._call_tool(
        "person",
        {"identifier": "alice@example.com", "summary": False},
    )

    payload = json.loads(result[0].text)
    assert "next_brief" not in payload
