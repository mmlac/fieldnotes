"""Tests for the MCP ``itinerary`` tool handler."""

from __future__ import annotations

import json
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from worker.config import (
    CalendarAccountConfig,
    Config,
    Neo4jConfig,
    QdrantConfig,
)
from worker.mcp_server import TOOLS, FieldnotesServer
from worker.models.base import CompletionResponse
from worker.query.itinerary import (
    Event,
    EventWithLinks,
    Itinerary,
    NoteHit,
    OpenTask,
    PersonRef,
    ThreadHit,
)


def _make_server(
    *,
    accounts: list[str] | None = None,
    brief_text: str | None = None,
) -> FieldnotesServer:
    cfg = Config(
        neo4j=Neo4jConfig(uri="bolt://localhost:7687", user="neo4j", password="x"),
        qdrant=QdrantConfig(host="localhost", port=6333, collection="fieldnotes"),
    )
    if accounts:
        cfg.google_calendar = {
            name: CalendarAccountConfig(name=name) for name in accounts
        }
    server = FieldnotesServer(cfg)
    if brief_text is not None:
        registry = MagicMock()
        resolved = MagicMock()
        resolved.complete.return_value = CompletionResponse(text=brief_text)
        registry.for_role.return_value = resolved
        server._registry = registry
    return server


def _build_itinerary() -> Itinerary:
    organizer = PersonRef(id=1, email="me@example.com", name="Me", is_self=True)
    attendee = PersonRef(id=2, email="alice@example.com", name="Alice")
    event = Event(
        id=42,
        source_id="google_calendar.work:abc",
        title="Roadmap sync",
        description="Quarterly roadmap review",
        start_ts="2026-04-28T15:00:00Z",
        end_ts="2026-04-28T16:00:00Z",
        location="Zoom",
        account="work",
        calendar_id="primary",
        html_link="https://cal.example.com/abc",
        organizer=organizer,
        attendees=[attendee],
        is_self_only=False,
    )
    return Itinerary(
        day=date(2026, 4, 28),
        timezone="UTC",
        events=[
            EventWithLinks(
                event=event,
                tasks=[
                    OpenTask(
                        title="Prep roadmap deck",
                        project="Q3",
                        tags=["@waiting"],
                        due="2026-04-28",
                        flagged=True,
                        source_id="omnifocus:task-1",
                    )
                ],
                notes=[
                    NoteHit(
                        source_id="obsidian:notes/roadmap.md",
                        title="Q3 roadmap notes",
                        snippet="Talk about ingestion latency",
                        mtime="2026-04-25T12:00:00Z",
                        attendee_overlap=True,
                        score=0.91,
                    )
                ],
                thread=ThreadHit(
                    kind="email",
                    source_id="gmail:thread-9",
                    title="Re: Q3 roadmap",
                    last_ts="2026-04-26T08:00:00Z",
                ),
            )
        ],
    )


def test_itinerary_tool_registered() -> None:
    names = {tool.name for tool in TOOLS}
    assert "itinerary" in names

    tool = next(t for t in TOOLS if t.name == "itinerary")
    desc = tool.description.lower()
    # Resolution order documented in the description.
    assert "today" in desc
    assert "tomorrow" in desc
    assert "yyyy-mm-dd" in desc
    # Brief flag documented.
    assert "brief" in desc and "skip" in desc


@pytest.mark.asyncio
@patch("worker.query.itinerary.get_itinerary")
async def test_itinerary_tool_returns_documented_schema(
    mock_get_itinerary: MagicMock,
) -> None:
    mock_get_itinerary.return_value = _build_itinerary()

    # ``brief=True`` skips the LLM path so the schema test doesn't need a
    # wired-up completion role.
    server = _make_server()
    result = await server._call_tool("itinerary", {"day": "today", "brief": True})

    assert len(result) == 1
    payload = json.loads(result[0].text)

    # Top-level shape.
    assert payload["day"] == "2026-04-28"
    assert payload["timezone"] == "UTC"
    assert isinstance(payload["events"], list) and len(payload["events"]) == 1

    ev = payload["events"][0]
    expected_keys = {
        "event_id",
        "source_id",
        "title",
        "start",
        "end",
        "account",
        "calendar_id",
        "organizer",
        "attendees",
        "location",
        "html_link",
        "linked",
        "next_brief",
    }
    assert expected_keys <= set(ev.keys())

    assert ev["event_id"] == "42"
    assert ev["source_id"] == "google_calendar.work:abc"
    assert ev["start"] == "2026-04-28T15:00:00Z"
    assert ev["end"] == "2026-04-28T16:00:00Z"
    assert ev["account"] == "work"
    assert ev["organizer"] == {"name": "Me", "email": "me@example.com"}
    assert ev["attendees"] == [{"name": "Alice", "email": "alice@example.com"}]
    assert ev["location"] == "Zoom"
    assert ev["html_link"] == "https://cal.example.com/abc"

    # Linked sub-shape.
    linked = ev["linked"]
    assert {"tasks", "notes", "thread"} <= set(linked.keys())
    assert linked["tasks"][0]["title"] == "Prep roadmap deck"
    assert linked["tasks"][0]["flagged"] is True
    assert linked["notes"][0]["source_id"] == "obsidian:notes/roadmap.md"
    assert linked["notes"][0]["attendee_overlap"] is True
    assert linked["thread"]["kind"] == "email"
    assert linked["thread"]["source_id"] == "gmail:thread-9"

    # ``--brief`` keeps next_brief null on every event.
    assert ev["next_brief"] is None


@pytest.mark.asyncio
async def test_itinerary_tool_invalid_day_returns_error() -> None:
    server = _make_server()
    # No mock — the real query layer raises ValueError on bad day.
    # ``brief=True`` skips completion-role resolution so the test surfaces
    # the day-validation error rather than a missing-role error.
    result = await server._call_tool("itinerary", {"day": "garbage-day", "brief": True})

    payload = json.loads(result[0].text)
    assert payload.get("error") is True
    assert "message" in payload
    # Message references the offending value or the accepted formats.
    msg = payload["message"].lower()
    assert "garbage-day" in msg or "yyyy-mm-dd" in msg


@pytest.mark.asyncio
async def test_itinerary_tool_unknown_account_returns_error() -> None:
    server = _make_server(accounts=["work", "personal"])
    result = await server._call_tool(
        "itinerary",
        {"day": "today", "account": "nope"},
    )

    payload = json.loads(result[0].text)
    assert payload.get("error") is True
    assert "nope" in payload["message"]
    assert payload.get("configured_accounts") == ["personal", "work"]


@pytest.mark.asyncio
@patch("worker.query.itinerary.get_itinerary")
async def test_itinerary_tool_brief_true_keeps_next_brief_null(
    mock_get_itinerary: MagicMock,
) -> None:
    mock_get_itinerary.return_value = _build_itinerary()

    server = _make_server()
    result = await server._call_tool(
        "itinerary",
        {"day": "today", "brief": True},
    )

    payload = json.loads(result[0].text)
    assert payload.get("error") is not True
    for ev in payload["events"]:
        assert ev["next_brief"] is None


@pytest.mark.asyncio
@patch("worker.cli.itinerary.assemble_event_brief")
@patch("worker.query.itinerary.get_itinerary")
async def test_itinerary_tool_brief_false_populates_next_brief(
    mock_get_itinerary: MagicMock,
    mock_assemble: MagicMock,
) -> None:
    """fn-wbc.4: brief=False (default) populates next_brief from the LLM."""
    mock_get_itinerary.return_value = _build_itinerary()
    # Driver-free assembler stub — we don't want to touch Neo4j here.
    from worker.query.itinerary_brief import EventBrief

    mock_assemble.side_effect = lambda ew, *, driver: EventBrief(
        title=ew.event.title,
        start_ts=ew.event.start_ts,
        end_ts=ew.event.end_ts,
    )

    server = _make_server(brief_text="Discuss roadmap. [Calendar]")
    with patch("worker.mcp_server.build_driver") as mock_drv:
        mock_drv.return_value = MagicMock()
        result = await server._call_tool(
            "itinerary",
            {"day": "today", "brief": False},
        )

    payload = json.loads(result[0].text)
    assert payload.get("error") is not True
    for ev in payload["events"]:
        assert ev["next_brief"] == "Discuss roadmap. [Calendar]"
