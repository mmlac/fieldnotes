"""Tests for the ``fieldnotes person --summary`` LLM brief.

Mixes pure-unit tests (prompt shape, prebrief formatting, MCP payload
plumbing) with Neo4j-gated integration tests that exercise the
seven-block assembler and the Rich profile rendering.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import pytest
from neo4j import Driver, GraphDatabase

from worker.cli.person import run_person
from worker.cli.person_brief_prompt import (
    SYSTEM_PROMPT,
    build_brief_request,
)
from worker.config import Config, MeConfig, Neo4jConfig, SourceConfig
from worker.models.base import CompletionResponse
from worker.query.person import Person
from worker.query.person_brief import (
    MeetingContext,
    PreBrief,
    assemble_prebrief,
    format_prebrief,
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


def _slack_ts(dt: datetime) -> str:
    return f"{dt.timestamp():.6f}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(*, with_me: bool = True, vault_path: str | None = None) -> Config:
    cfg = Config(
        neo4j=Neo4jConfig(uri=_NEO4J_URI, user=_NEO4J_USER, password=_NEO4J_PASSWORD)
    )
    if with_me:
        cfg.me = MeConfig(emails=["self@example.com"], name="Me")
    if vault_path:
        cfg.sources["obsidian"] = SourceConfig(
            name="obsidian", settings={"vault_paths": [vault_path]}
        )
    return cfg


def _mock_registry(text: str = "- [Open OmniFocus tasks] follow up\n") -> MagicMock:
    """A ModelRegistry whose ``for_role('completion')`` returns a stub."""
    registry = MagicMock()
    resolved = MagicMock()
    resolved.complete.return_value = CompletionResponse(text=text)
    registry.for_role.return_value = resolved
    return registry


# ---------------------------------------------------------------------------
# Pure-unit: prompt and formatting
# ---------------------------------------------------------------------------


class TestBriefPrompt:
    def test_summary_prompt_contains_no_invent_instruction(self) -> None:
        """Acceptance: golden — system prompt must forbid invention."""
        assert "Do not invent items not present in the context" in SYSTEM_PROMPT

    def test_prompt_has_four_groupings(self) -> None:
        for heading in (
            "Decisions awaited from them",
            "Decisions awaited from you",
            "Open threads to close",
            "Background",
        ):
            assert heading in SYSTEM_PROMPT

    def test_request_low_temperature(self) -> None:
        prebrief = PreBrief(
            identity_name="Alice",
            identity_email="alice@example.com",
            source_count=1,
        )
        req = build_brief_request(prebrief, since_label="30d")
        assert req.temperature == 0.2
        assert req.system == SYSTEM_PROMPT
        assert req.messages[0]["role"] == "user"
        assert "Alice" in req.messages[0]["content"]


class TestFormatPrebrief:
    def test_renders_seven_block_skeleton(self) -> None:
        """Even with empty data, all section headers render so the LLM
        sees the schema it must follow."""
        prebrief = PreBrief(
            identity_name="Alice",
            identity_email="alice@example.com",
            source_count=2,
        )
        text = format_prebrief(prebrief, since_label="30d")
        assert "[Identity]" in text
        assert "[Open OmniFocus tasks" in text
        assert "[Outstanding email threads" in text
        assert "[Unresolved Slack mentions of you" in text
        assert "[Top active topics" in text
        # blocks 6/7 only render when data present
        assert "[Obsidian People note]" not in text
        assert "[Upcoming meeting context]" not in text

    def test_meeting_block_renders_when_present(self) -> None:
        prebrief = PreBrief(
            identity_name="Alice",
            identity_email="alice@example.com",
            source_count=2,
            meeting=MeetingContext(
                event_id="cal://1",
                summary="Q2 planning",
                start_time="2026-05-01T09:00:00Z",
                location="Zoom",
                description="Plan the quarter",
                attendees=["alice@example.com", "bob@example.com"],
                attachments=["agenda.pdf"],
            ),
        )
        text = format_prebrief(prebrief, since_label="30d")
        assert "[Upcoming meeting context]" in text
        assert "Q2 planning" in text
        assert "Zoom" in text
        assert "agenda.pdf" in text
        assert "alice@example.com" in text

    def test_obsidian_note_renders_when_present(self) -> None:
        prebrief = PreBrief(
            identity_name="Alice",
            identity_email="alice@example.com",
            source_count=1,
            obsidian_note="Long-time collaborator on the search project.",
        )
        text = format_prebrief(prebrief, since_label="30d")
        assert "[Obsidian People note]" in text
        assert "Long-time collaborator" in text


# ---------------------------------------------------------------------------
# run_person: LLM-call gating with mocks
# ---------------------------------------------------------------------------


def _patched_run(
    *,
    cfg: Config,
    registry: MagicMock | None = None,
    seeded_person: Person | None = None,
    profile_func: Any = None,
    **kwargs: Any,
) -> tuple[int, str, str]:
    """Run ``run_person`` with mocked config + driver + profile builder."""
    out = io.StringIO()
    err = io.StringIO()
    with (
        patch("worker.cli.person.load_config", return_value=cfg),
        patch("worker.cli.person._open_driver") as mock_drv,
        patch(
            "worker.cli.person._resolve_identifier",
            return_value=seeded_person,
        ),
        patch(
            "worker.cli.person._build_profile",
            side_effect=profile_func,
        ),
        patch(
            "worker.cli.person._total_interactions",
            return_value=0,
        ),
    ):
        mock_drv.return_value = MagicMock()
        with redirect_stdout(out), redirect_stderr(err):
            code = run_person(registry=registry, **kwargs)
    return code, out.getvalue(), err.getvalue()


def _empty_profile(driver: Any, person: Person, **_: Any) -> Any:
    from worker.query.person import PersonProfile

    return PersonProfile(person=person)


def _alice() -> Person:
    return Person(id=42, email="alice@example.com", name="Alice Example")


class TestSummaryGating:
    def test_summary_omitted_makes_no_llm_call(self) -> None:
        """Acceptance: without --summary, the LLM is not invoked."""
        cfg = _make_config()
        registry = _mock_registry()
        code, _, _ = _patched_run(
            cfg=cfg,
            registry=registry,
            seeded_person=_alice(),
            profile_func=_empty_profile,
            identifier="alice@example.com",
            summary=False,
            json_output=True,
        )
        assert code == 0
        registry.for_role.assert_not_called()

    def test_summary_uses_completion_role(self) -> None:
        """Acceptance: --summary resolves the 'completion' role."""
        cfg = _make_config()
        registry = _mock_registry(text="- [Open OmniFocus tasks] follow up")
        with patch("worker.cli.person.assemble_prebrief") as mock_asm:
            mock_asm.return_value = PreBrief(
                identity_name="Alice",
                identity_email="alice@example.com",
                source_count=1,
            )
            code, out, err = _patched_run(
                cfg=cfg,
                registry=registry,
                seeded_person=_alice(),
                profile_func=_empty_profile,
                identifier="alice@example.com",
                summary=True,
                json_output=True,
            )
        assert code == 0, err
        registry.for_role.assert_called_once_with("completion")
        registry.for_role.return_value.complete.assert_called_once()

    def test_missing_completion_role_errors_before_call(self) -> None:
        """Acceptance: misconfigured role surfaces a doctor-style error."""
        cfg = _make_config()
        registry = MagicMock()
        registry.for_role.side_effect = KeyError("completion")
        with patch("worker.cli.person.assemble_prebrief") as mock_asm:
            mock_asm.return_value = PreBrief(
                identity_name="Alice",
                identity_email="alice@example.com",
                source_count=1,
            )
            code, _, err = _patched_run(
                cfg=cfg,
                registry=registry,
                seeded_person=_alice(),
                profile_func=_empty_profile,
                identifier="alice@example.com",
                summary=True,
            )
        assert code == 2
        assert "completion" in err
        assert "doctor" in err

    def test_summary_renders_in_profile_view(self) -> None:
        """Acceptance: brief surfaces under a 'NEXT-MEETING BRIEF' section."""
        cfg = _make_config()
        brief_text = "- [Open OmniFocus tasks] reply to Alice"
        registry = _mock_registry(text=brief_text)
        with patch("worker.cli.person.assemble_prebrief") as mock_asm:
            mock_asm.return_value = PreBrief(
                identity_name="Alice",
                identity_email="alice@example.com",
                source_count=1,
            )
            code, out, _ = _patched_run(
                cfg=cfg,
                registry=registry,
                seeded_person=_alice(),
                profile_func=_empty_profile,
                identifier="alice@example.com",
                summary=True,
                json_output=False,
            )
        assert code == 0
        assert "NEXT-MEETING BRIEF" in out
        assert "reply to Alice" in out

    def test_summary_emits_next_brief_in_json(self) -> None:
        cfg = _make_config()
        brief_text = "- [Open OmniFocus tasks] reply to Alice"
        registry = _mock_registry(text=brief_text)
        with patch("worker.cli.person.assemble_prebrief") as mock_asm:
            mock_asm.return_value = PreBrief(
                identity_name="Alice",
                identity_email="alice@example.com",
                source_count=1,
            )
            code, out, _ = _patched_run(
                cfg=cfg,
                registry=registry,
                seeded_person=_alice(),
                profile_func=_empty_profile,
                identifier="alice@example.com",
                summary=True,
                json_output=True,
            )
        assert code == 0
        payload = json.loads(out)
        assert payload["next_brief"] == brief_text

    def test_no_summary_omits_next_brief_in_json(self) -> None:
        cfg = _make_config()
        registry = _mock_registry()
        code, out, _ = _patched_run(
            cfg=cfg,
            registry=registry,
            seeded_person=_alice(),
            profile_func=_empty_profile,
            identifier="alice@example.com",
            summary=False,
            json_output=True,
        )
        assert code == 0
        payload = json.loads(out)
        assert "next_brief" not in payload


# ---------------------------------------------------------------------------
# Neo4j-gated integration: pre-brief assembler
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
def seeded(driver: Driver) -> dict[str, int]:
    """Seed Alice (target) + Self (you) with varied edges in horizon."""
    t_now = _NOW
    t_y = t_now - timedelta(days=1)
    t_2d = t_now - timedelta(days=2)
    t_3d = t_now - timedelta(days=3)
    t_5d = t_now - timedelta(days=5)
    t_3mo = t_now - timedelta(days=90)

    cypher = """
    MERGE (alice:Person {email: 'alice@example.com'})
      SET alice.name = 'Alice Example'
    MERGE (self:Person {email: 'self@example.com'})
      SET self.name = 'Me', self.is_self = true

    // Email thread Alice replied to last (you owe her)
    MERGE (th_open:Thread {source_id: 'gmail://acct/thread/T1'})
      SET th_open.subject = 'Need your sign-off'
    MERGE (em_open:Email {source_id: 'gmail://1'})
      SET em_open.subject = 'Re: Need your sign-off', em_open.date = $d_y
    MERGE (em_open)-[:PART_OF]->(th_open)
    MERGE (alice)-[:SENT]->(em_open)
    MERGE (em_open)-[:TO]->(self)

    // Email thread you replied to last (closed — should NOT appear)
    MERGE (th_closed:Thread {source_id: 'gmail://acct/thread/T2'})
      SET th_closed.subject = 'Old chat'
    MERGE (em_old:Email {source_id: 'gmail://2'})
      SET em_old.subject = 'Re: Old chat', em_old.date = $d_2d
    MERGE (em_old)-[:PART_OF]->(th_closed)
    MERGE (self)-[:SENT]->(em_old)
    MERGE (em_old)-[:TO]->(alice)

    // Email beyond horizon (should NOT appear when horizon=7d)
    MERGE (th_far:Thread {source_id: 'gmail://acct/thread/T3'})
      SET th_far.subject = 'Ancient'
    MERGE (em_far:Email {source_id: 'gmail://3'})
      SET em_far.subject = 'Re: Ancient', em_far.date = $d_3mo
    MERGE (em_far)-[:PART_OF]->(th_far)
    MERGE (alice)-[:SENT]->(em_far)

    // Slack mention of self by Alice with no later reply from self
    MERGE (sl_open:SlackMessage {source_id: 'slack://1'})
      SET sl_open.channel_name = 'eng', sl_open.channel_id = 'C-eng',
          sl_open.first_ts = $sl_y, sl_open.last_ts = $sl_y,
          sl_open.text_preview = 'hey @me have you got a sec'
    MERGE (sl_open)-[:SENT_BY]->(alice)
    MERGE (sl_open)-[:MENTIONS]->(self)

    // Slack mention of self by Alice but YOU replied later (should drop)
    MERGE (sl_closed:SlackMessage {source_id: 'slack://2'})
      SET sl_closed.channel_name = 'random', sl_closed.channel_id = 'C-rand',
          sl_closed.first_ts = $sl_3d, sl_closed.last_ts = $sl_3d
    MERGE (sl_closed)-[:SENT_BY]->(alice)
    MERGE (sl_closed)-[:MENTIONS]->(self)
    MERGE (sl_reply:SlackMessage {source_id: 'slack://3'})
      SET sl_reply.channel_name = 'random', sl_reply.channel_id = 'C-rand',
          sl_reply.first_ts = $sl_2d, sl_reply.last_ts = $sl_2d
    MERGE (sl_reply)-[:SENT_BY]->(self)

    // Open OmniFocus task mentioning Alice
    MERGE (t_open:Task {source_id: 'of://1'})
      SET t_open.name = 'Email Alice about Q2',
          t_open.status = 'active',
          t_open.flagged = true,
          t_open.due_date = $d_y
    MERGE (t_open)-[:MENTIONS]->(alice)

    // Topic from a recent doc
    MERGE (em_open)-[:TAGGED]->(:Topic {name: 'Sign-off'})

    // Calendar event for --meeting
    MERGE (cal:CalendarEvent {source_id: 'cal://meet1'})
      SET cal.summary = '1:1 with Alice',
          cal.start_time = $d_now,
          cal.location = 'Zoom',
          cal.description = 'Discuss roadmap'
    MERGE (cal)-[:ATTENDED_BY]->(alice)
    MERGE (cal)-[:ORGANIZED_BY]->(self)
    MERGE (att:Attachment {source_id: 'att://1'})
      SET att.title = 'agenda.pdf'
    MERGE (att)-[:ATTACHED_TO]->(cal)

    RETURN id(alice) AS alice_id
    """
    params = {
        "d_now": _iso(t_now),
        "d_y": _iso(t_y),
        "d_2d": _iso(t_2d),
        "d_3d": _iso(t_3d),
        "d_5d": _iso(t_5d),
        "d_3mo": _iso(t_3mo),
        "sl_y": _slack_ts(t_y),
        "sl_2d": _slack_ts(t_2d),
        "sl_3d": _slack_ts(t_3d),
    }
    with driver.session() as s:
        rec = s.run(cypher, **params).single()
        assert rec is not None
        return {"alice_id": int(rec["alice_id"])}


@_NEEDS_NEO4J
class TestAssembleSeven:
    def test_summary_assembles_all_seven_input_blocks(
        self, driver: Driver, seeded: dict[str, int]
    ) -> None:
        """Acceptance: pre-brief includes all populated input blocks."""
        alice = Person(
            id=seeded["alice_id"],
            email="alice@example.com",
            name="Alice Example",
        )
        prebrief = assemble_prebrief(
            alice,
            driver=driver,
            since=_NOW - timedelta(days=7),
            self_emails=["self@example.com"],
        )
        # 1. Identity
        assert prebrief.identity_name == "Alice Example"
        assert prebrief.identity_email == "alice@example.com"
        assert prebrief.source_count >= 4  # gmail, slack, OF, calendar
        # 2. Open tasks
        assert any("Q2" in t.title for t in prebrief.open_tasks)
        # 3. Email threads — Alice replied last
        subjects = [t.subject for t in prebrief.email_threads]
        assert any("sign-off" in s.lower() for s in subjects)
        assert all("ancient" not in s.lower() for s in subjects)
        # 4. Slack mentions — open one survives, closed one filtered
        channels = [m.channel for m in prebrief.slack_mentions]
        assert "eng" in channels
        assert "random" not in channels
        # 5. Top topics
        topic_names = [t.topic_name for t in prebrief.top_topics]
        assert "Sign-off" in topic_names

    def test_summary_meeting_id_adds_event_block(
        self, driver: Driver, seeded: dict[str, int]
    ) -> None:
        """Acceptance: valid meeting_id pulls calendar context."""
        alice = Person(
            id=seeded["alice_id"],
            email="alice@example.com",
            name="Alice Example",
        )
        prebrief = assemble_prebrief(
            alice,
            driver=driver,
            since=_NOW - timedelta(days=30),
            self_emails=["self@example.com"],
            meeting_id="cal://meet1",
        )
        assert prebrief.meeting is not None
        assert prebrief.meeting.summary == "1:1 with Alice"
        assert prebrief.meeting.location == "Zoom"
        assert "agenda.pdf" in prebrief.meeting.attachments
        # attendees should include both
        emails = set(prebrief.meeting.attendees)
        assert "alice@example.com" in emails

    def test_summary_meeting_id_invalid_errors_cleanly(
        self, driver: Driver, seeded: dict[str, int]
    ) -> None:
        """Acceptance: bad meeting_id raises a clear ValueError."""
        alice = Person(
            id=seeded["alice_id"],
            email="alice@example.com",
            name="Alice Example",
        )
        with pytest.raises(ValueError, match="meeting_id"):
            assemble_prebrief(
                alice,
                driver=driver,
                since=_NOW - timedelta(days=30),
                meeting_id="cal://bogus",
            )


# ---------------------------------------------------------------------------
# MCP plumbing: presence/absence of next_brief
# ---------------------------------------------------------------------------


class TestMcpPersonSummary:
    def test_mcp_person_summary_false_omits_next_brief(self) -> None:
        from worker.mcp_server import _person_profile_to_dict
        from worker.query.person import PersonProfile

        profile = PersonProfile(
            person=Person(id=1, email="alice@example.com", name="Alice Example")
        )
        payload = _person_profile_to_dict("alice@example.com", profile)
        assert "next_brief" not in payload

    def test_mcp_person_summary_true_returns_next_brief(self) -> None:
        from worker.mcp_server import _person_profile_to_dict
        from worker.query.person import PersonProfile

        profile = PersonProfile(
            person=Person(id=1, email="alice@example.com", name="Alice Example")
        )
        payload = _person_profile_to_dict(
            "alice@example.com", profile, brief="- [Open OmniFocus tasks] x"
        )
        assert payload["next_brief"] == "- [Open OmniFocus tasks] x"
