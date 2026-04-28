"""Tests for the ``fieldnotes itinerary`` per-meeting LLM brief (fn-wbc.4).

Exercises the prompt builder, the per-event pre-brief assembler/formatter,
the CLI gating + sequencing, and the MCP tool integration.  Pure-unit
tests use mocked itineraries; Neo4j-gated tests exercise the live
assembler against a seeded graph.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from datetime import date, datetime, timedelta, timezone
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import pytest
from neo4j import Driver, GraphDatabase

from worker.cli.itinerary import (
    generate_event_briefs,
    run_itinerary,
)
from worker.cli.itinerary_brief_prompt import (
    SYSTEM_PROMPT,
    build_event_brief_request,
)
from worker.config import (
    CalendarAccountConfig,
    Config,
    McpConfig,
    Neo4jConfig,
    QdrantConfig,
)
from worker.mcp_server import FieldnotesServer
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
from worker.query.itinerary_brief import (
    EventBrief,
    ThreadMessage,
    assemble_event_brief,
    format_event_brief,
)


_NEO4J_URI = "bolt://localhost:7687"
_NEO4J_USER = "neo4j"
_NEO4J_PASSWORD = "testpassword"


def _neo4j_available() -> bool:
    try:
        with GraphDatabase.driver(_NEO4J_URI, auth=(_NEO4J_USER, _NEO4J_PASSWORD)) as d:
            d.verify_connectivity()
        return True
    except Exception:
        return False


_NEEDS_NEO4J = pytest.mark.skipif(
    not _neo4j_available(),
    reason="Neo4j not available at bolt://localhost:7687",
)


_NOW = datetime(2026, 4, 27, 12, 0, 0, tzinfo=timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config() -> Config:
    return Config(
        neo4j=Neo4jConfig(uri=_NEO4J_URI, user=_NEO4J_USER, password=_NEO4J_PASSWORD)
    )


def _mock_registry(text: str = "Brief about meeting. [Email]") -> MagicMock:
    registry = MagicMock()
    resolved = MagicMock()
    resolved.complete.return_value = CompletionResponse(text=text)
    registry.for_role.return_value = resolved
    return registry


def _make_event(
    eid: int,
    *,
    title: str = "Sync",
    description: str | None = None,
    attendees: list[PersonRef] | None = None,
    location: str | None = None,
) -> Event:
    return Event(
        id=eid,
        source_id=f"cal://e{eid}",
        title=title,
        description=description,
        start_ts="2026-04-27T09:00:00Z",
        end_ts="2026-04-27T10:00:00Z",
        location=location,
        account="work",
        calendar_id="work/primary",
        html_link=None,
        organizer=PersonRef(id=99, email="me@example.com", name="Me", is_self=True),
        attendees=attendees or [],
    )


def _make_itinerary(events: list[EventWithLinks]) -> Itinerary:
    return Itinerary(day=date(2026, 4, 27), timezone="UTC", events=events)


def _stub_assemble(ew: EventWithLinks, *, driver: Any) -> EventBrief:
    """Driver-free EventBrief that mirrors what the real assembler would build.

    Lets the CLI/MCP gating tests run without a live graph.
    """
    ev = ew.event
    return EventBrief(
        title=ev.title or "(untitled)",
        start_ts=ev.start_ts,
        end_ts=ev.end_ts,
        location=ev.location,
        description=ev.description,
        tasks=list(ew.tasks),
        notes=list(ew.notes),
        thread=ew.thread,
    )


def _patched_run(
    *,
    cfg: Config,
    itinerary: Itinerary,
    registry: MagicMock | None = None,
    **kwargs: Any,
) -> tuple[int, str, str]:
    out = io.StringIO()
    err = io.StringIO()
    with (
        patch("worker.cli.itinerary.load_config", return_value=cfg),
        patch("worker.cli.itinerary.get_itinerary", return_value=itinerary),
        patch("worker.cli.itinerary._open_driver") as mock_drv,
        patch("worker.cli.itinerary.assemble_event_brief", side_effect=_stub_assemble),
    ):
        mock_drv.return_value = MagicMock()
        with redirect_stdout(out), redirect_stderr(err):
            code = run_itinerary(registry=registry, **kwargs)
    return code, out.getvalue(), err.getvalue()


# ---------------------------------------------------------------------------
# Pure-unit: prompt + format
# ---------------------------------------------------------------------------


class TestPrompt:
    def test_summary_prompt_contains_no_invent_instruction(self) -> None:
        """Acceptance: the system prompt forbids invention."""
        assert "Do not invent items not present" in SYSTEM_PROMPT

    def test_request_low_temperature_and_bounded_tokens(self) -> None:
        brief = EventBrief(
            title="Q2 sync",
            start_ts="2026-04-27T09:00:00Z",
            end_ts="2026-04-27T10:00:00Z",
        )
        req = build_event_brief_request(brief)
        assert req.temperature == 0.2
        assert req.max_tokens <= 200
        assert req.system == SYSTEM_PROMPT
        assert req.messages[0]["role"] == "user"
        assert "Q2 sync" in req.messages[0]["content"]


class TestFormatEventBrief:
    def test_summary_assembles_six_input_blocks(self) -> None:
        """Acceptance: the formatted context exposes all six block headers."""
        brief = EventBrief(
            title="Coffee w/ Dan",
            start_ts="2026-04-27T09:00:00Z",
            end_ts="2026-04-27T10:00:00Z",
        )
        text = format_event_brief(brief)
        assert "[Calendar event]" in text
        assert "[Event description]" in text
        assert "[Linked OmniFocus tasks" in text
        assert "[Linked notes" in text
        assert "[Linked thread]" in text
        assert "[Attachments" in text

    def test_format_renders_populated_blocks(self) -> None:
        brief = EventBrief(
            title="Q2 sync",
            start_ts="2026-04-27T09:00:00Z",
            end_ts="2026-04-27T10:00:00Z",
            location="Conf A",
            organizer="Me <me@example.com>",
            attendees=["Alice <alice@example.com>"],
            description="Plan Q2",
            tasks=[
                OpenTask(title="Email Alice about Q2", project="Work", flagged=True)
            ],
            notes=[
                NoteHit(
                    source_id="obs://q2",
                    title="Q2 plan",
                    snippet="plan details",
                    mtime=None,
                    attendee_overlap=True,
                    score=0.8,
                )
            ],
            thread=ThreadHit(
                kind="email",
                source_id="t://1",
                title="Q2 thread",
                last_ts="2026-04-26T10:00:00Z",
            ),
            thread_messages=[
                ThreadMessage(
                    sender="Alice",
                    ts="2026-04-26T10:00:00Z",
                    snippet="latest",
                )
            ],
            attachments=["agenda.pdf"],
        )
        text = format_event_brief(brief)
        assert "Q2 sync" in text
        assert "Conf A" in text
        assert "alice@example.com" in text
        assert "Email Alice about Q2" in text
        assert "obs://q2" in text
        assert "plan details" in text
        assert "Q2 thread" in text
        assert "agenda.pdf" in text


# ---------------------------------------------------------------------------
# generate_event_briefs: sequencing
# ---------------------------------------------------------------------------


class TestGenerateEventBriefs:
    def test_summary_calls_one_per_event_sequentially(self) -> None:
        """Acceptance: 3 events → 3 LLM calls, one per event, in order."""
        events = [
            EventWithLinks(event=_make_event(1, title="A")),
            EventWithLinks(event=_make_event(2, title="B")),
            EventWithLinks(event=_make_event(3, title="C")),
        ]
        itin = _make_itinerary(events)
        completion_model = MagicMock()
        completion_model.complete.side_effect = [
            CompletionResponse(text="Brief A. [Calendar]"),
            CompletionResponse(text="Brief B. [Calendar]"),
            CompletionResponse(text="Brief C. [Calendar]"),
        ]
        driver = MagicMock()
        with patch(
            "worker.cli.itinerary.assemble_event_brief", side_effect=_stub_assemble
        ) as mock_asm:
            briefs = generate_event_briefs(
                itin, driver=driver, completion_model=completion_model
            )
        assert briefs == {
            1: "Brief A. [Calendar]",
            2: "Brief B. [Calendar]",
            3: "Brief C. [Calendar]",
        }
        assert completion_model.complete.call_count == 3
        # Sequential order: assembler called once per event in the same order.
        called_titles = [c.args[0].event.title for c in mock_asm.call_args_list]
        assert called_titles == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# CLI gating with mocked itinerary
# ---------------------------------------------------------------------------


class TestSummaryGating:
    def test_summary_omitted_with_brief_flag_makes_no_llm_call(self) -> None:
        """Acceptance: ``--brief`` makes 0 LLM calls and all next_brief null."""
        cfg = _make_config()
        registry = _mock_registry()
        events = [
            EventWithLinks(event=_make_event(1)),
            EventWithLinks(event=_make_event(2)),
        ]
        code, out, _ = _patched_run(
            cfg=cfg,
            itinerary=_make_itinerary(events),
            registry=registry,
            day="2026-04-27",
            brief=True,
            json_output=True,
        )
        assert code == 0
        registry.for_role.assert_not_called()
        registry.for_role.return_value.complete.assert_not_called()
        payload = json.loads(out)
        assert payload["events"]
        for ev in payload["events"]:
            assert ev["next_brief"] is None

    def test_summary_uses_completion_role(self) -> None:
        """Acceptance: brief generation honours the 'completion' role."""
        cfg = _make_config()
        registry = _mock_registry(text="Discuss Q2. [Calendar]")
        events = [EventWithLinks(event=_make_event(1))]
        code, _out, err = _patched_run(
            cfg=cfg,
            itinerary=_make_itinerary(events),
            registry=registry,
            day="2026-04-27",
            json_output=True,
        )
        assert code == 0, err
        called_roles = [c.args[0] for c in registry.for_role.call_args_list]
        assert "completion" in called_roles

    def test_missing_completion_role_errors_before_first_call(self) -> None:
        """Acceptance: missing completion role surfaces a doctor-style error
        before the first LLM call."""
        cfg = _make_config()
        registry = MagicMock()
        registry.for_role.side_effect = KeyError("completion")
        events = [EventWithLinks(event=_make_event(1))]
        code, _out, err = _patched_run(
            cfg=cfg,
            itinerary=_make_itinerary(events),
            registry=registry,
            day="2026-04-27",
            json_output=True,
        )
        assert code == 2
        assert "completion" in err
        assert "doctor" in err
        # No completion call attempted (for_role failed first).
        registry.for_role.return_value.complete.assert_not_called()

    def test_summary_emits_next_brief_in_json(self) -> None:
        """Acceptance: default JSON includes ``next_brief`` text per event."""
        cfg = _make_config()
        registry = _mock_registry(text="Discuss Q2. [Calendar]")
        events = [EventWithLinks(event=_make_event(1, title="Q2 sync"))]
        code, out, _ = _patched_run(
            cfg=cfg,
            itinerary=_make_itinerary(events),
            registry=registry,
            day="2026-04-27",
            json_output=True,
        )
        assert code == 0
        payload = json.loads(out)
        assert payload["events"][0]["next_brief"] == "Discuss Q2. [Calendar]"

    def test_no_summary_omits_next_brief_in_json(self) -> None:
        """Acceptance: ``--brief`` keeps ``next_brief`` null in JSON."""
        cfg = _make_config()
        registry = _mock_registry()
        events = [EventWithLinks(event=_make_event(1))]
        code, out, _ = _patched_run(
            cfg=cfg,
            itinerary=_make_itinerary(events),
            registry=registry,
            day="2026-04-27",
            brief=True,
            json_output=True,
        )
        assert code == 0
        payload = json.loads(out)
        assert payload["events"][0]["next_brief"] is None

    def test_summary_renders_in_itinerary_view(self) -> None:
        """Acceptance: brief surfaces between attendees and tasks with '▸ '."""
        cfg = _make_config()
        registry = _mock_registry(text="Discuss Q2. [Calendar]")
        events = [EventWithLinks(event=_make_event(1, title="Q2 sync"))]
        code, out, _ = _patched_run(
            cfg=cfg,
            itinerary=_make_itinerary(events),
            registry=registry,
            day="2026-04-27",
        )
        assert code == 0
        assert "▸ Discuss Q2." in out
        idx_brief = out.index("▸ Discuss Q2.")
        idx_tasks = out.index("Tasks")
        assert idx_brief < idx_tasks

    def test_empty_context_event_still_gets_brief(self) -> None:
        """Acceptance: zero-context events still produce a one-sentence brief."""
        cfg = _make_config()
        registry = _mock_registry(text="No linked context. [Calendar]")
        events = [EventWithLinks(event=_make_event(1, title="Coffee w/ Dan"))]
        code, out, _ = _patched_run(
            cfg=cfg,
            itinerary=_make_itinerary(events),
            registry=registry,
            day="2026-04-27",
            json_output=True,
        )
        assert code == 0
        payload = json.loads(out)
        assert payload["events"][0]["next_brief"] == "No linked context. [Calendar]"


# ---------------------------------------------------------------------------
# MCP integration
# ---------------------------------------------------------------------------


def _make_mcp_cfg() -> Config:
    cfg = Config(
        neo4j=Neo4jConfig(uri=_NEO4J_URI, user=_NEO4J_USER, password=_NEO4J_PASSWORD),
        qdrant=QdrantConfig(host="localhost", port=6333, collection="fieldnotes"),
        mcp=McpConfig(),
    )
    cfg.google_calendar = {"work": CalendarAccountConfig(name="work")}
    return cfg


class TestMcpItinerary:
    @pytest.mark.asyncio
    async def test_mcp_itinerary_summary_default_returns_next_brief(self) -> None:
        cfg = _make_mcp_cfg()
        server = FieldnotesServer(cfg)
        registry = _mock_registry(text="Discuss Q2. [Calendar]")
        server._registry = registry
        import concurrent.futures

        server._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            events = [EventWithLinks(event=_make_event(1, title="Q2 sync"))]
            with (
                patch(
                    "worker.query.itinerary.get_itinerary",
                    return_value=_make_itinerary(events),
                ),
                patch(
                    "worker.cli.itinerary.assemble_event_brief",
                    side_effect=_stub_assemble,
                ),
                patch("worker.mcp_server.GraphDatabase.driver") as mock_drv,
            ):
                mock_drv.return_value = MagicMock()
                result = await server._handle_itinerary({"day": "2026-04-27"})
            assert len(result) == 1
            payload = json.loads(result[0].text)
            assert payload["events"][0]["next_brief"] == "Discuss Q2. [Calendar]"
        finally:
            if server._executor is not None:
                server._executor.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_mcp_itinerary_brief_true_keeps_next_brief_null(self) -> None:
        cfg = _make_mcp_cfg()
        server = FieldnotesServer(cfg)
        registry = _mock_registry()
        server._registry = registry
        import concurrent.futures

        server._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            events = [EventWithLinks(event=_make_event(1, title="Q2 sync"))]
            with patch(
                "worker.query.itinerary.get_itinerary",
                return_value=_make_itinerary(events),
            ):
                result = await server._handle_itinerary(
                    {"day": "2026-04-27", "brief": True}
                )
            assert len(result) == 1
            payload = json.loads(result[0].text)
            assert payload["events"][0]["next_brief"] is None
            registry.for_role.assert_not_called()
        finally:
            if server._executor is not None:
                server._executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Neo4j-gated: live assembler
# ---------------------------------------------------------------------------


@pytest.fixture
def driver() -> Generator[Driver, None, None]:
    drv = GraphDatabase.driver(_NEO4J_URI, auth=(_NEO4J_USER, _NEO4J_PASSWORD))
    with drv.session() as s:
        s.run("MATCH (n) DETACH DELETE n")
    yield drv
    with drv.session() as s:
        s.run("MATCH (n) DETACH DELETE n")
    drv.close()


@pytest.fixture
def seeded_event(driver: Driver) -> dict[str, int]:
    """Seed one CalendarEvent with a thread tail and an attachment."""
    cypher = """
    MERGE (me:Person {email: 'me@example.com'})
      SET me.name = 'Me Self', me.is_self = true
    MERGE (alice:Person {email: 'alice@example.com'})
      SET alice.name = 'Alice Example'
    MERGE (cal:CalendarEvent {source_id: 'cal://meet1'})
      SET cal.summary = 'Q2 sync',
          cal.start_time = $start,
          cal.end_time = $end,
          cal.account = 'work',
          cal.description = 'Plan the quarter'
    MERGE (cal)-[:ORGANIZED_BY]->(me)
    MERGE (cal)-[:ATTENDED_BY]->(alice)
    MERGE (att:Attachment {source_id: 'att://1'})
      SET att.title = 'agenda.pdf'
    MERGE (att)-[:ATTACHED_TO]->(cal)
    MERGE (th:Thread {source_id: 'gmail://acct/thread/T1'})
      SET th.subject = 'Q2 planning'
    MERGE (em:Email {source_id: 'gmail://1'})
      SET em.subject = 'Q2 planning details', em.date = $em_date
    MERGE (em)-[:PART_OF]->(th)
    MERGE (alice)-[:SENT]->(em)
    RETURN id(cal) AS cal_id
    """
    params = {
        "start": _iso(_NOW - timedelta(hours=3)),
        "end": _iso(_NOW - timedelta(hours=2)),
        "em_date": _iso(_NOW - timedelta(days=1)),
    }
    with driver.session() as s:
        rec = s.run(cypher, **params).single()
        assert rec is not None
        return {"cal_id": int(rec["cal_id"])}


@_NEEDS_NEO4J
def test_assemble_event_brief_pulls_thread_tail_and_attachments(
    driver: Driver, seeded_event: dict[str, int]
) -> None:
    """Live assembler: thread_messages and attachments come from Cypher."""
    ev = Event(
        id=seeded_event["cal_id"],
        source_id="cal://meet1",
        title="Q2 sync",
        description="Plan the quarter",
        start_ts=_iso(_NOW - timedelta(hours=3)),
        end_ts=_iso(_NOW - timedelta(hours=2)),
        location=None,
        account="work",
        calendar_id=None,
        html_link=None,
        organizer=PersonRef(id=0, email="me@example.com", name="Me Self", is_self=True),
        attendees=[
            PersonRef(
                id=0, email="alice@example.com", name="Alice Example", is_self=False
            )
        ],
    )
    ew = EventWithLinks(
        event=ev,
        thread=ThreadHit(
            kind="email",
            source_id="gmail://acct/thread/T1",
            title="Q2 planning",
            last_ts=_iso(_NOW - timedelta(days=1)),
        ),
    )
    brief = assemble_event_brief(ew, driver=driver)
    assert brief.title == "Q2 sync"
    assert brief.description == "Plan the quarter"
    assert "agenda.pdf" in brief.attachments
    assert any("Q2 planning" in m.snippet for m in brief.thread_messages)
