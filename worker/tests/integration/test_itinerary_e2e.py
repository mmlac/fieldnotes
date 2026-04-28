"""End-to-end integration test for the Itinerary feature (fn-wbc).

Round-trips the full feature path on a seeded Neo4j instance:

* CLI ``fieldnotes itinerary --day <iso> --json`` (subprocess)
* MCP ``itinerary`` tool over stdio (subprocess + ``mcp`` client)
* CLI default mode with a stub completion provider — asserts the
  per-event ``next_brief`` is grounded in the seeded graph.
* CLI ``--brief`` mode — asserts ``next_brief`` is ``null`` on every
  event and the stub provider received zero calls.

Skipped automatically when Neo4j isn't reachable — same pattern as
``tests/integration/test_person_profile_e2e.py``.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from datetime import timezone
from pathlib import Path
from typing import Any, Generator
from unittest.mock import patch

import pytest
from neo4j import Driver, GraphDatabase

from worker.config import Config, Neo4jConfig
from worker.models.base import CompletionRequest, CompletionResponse


_NEO4J_URI = os.environ.get("NEO4J_TEST_URI", "bolt://localhost:7687")
_NEO4J_USER = os.environ.get("NEO4J_TEST_USER", "neo4j")
_NEO4J_PASSWORD = os.environ.get("NEO4J_TEST_PASSWORD", "testpassword")


def _neo4j_available() -> bool:
    try:
        with GraphDatabase.driver(
            _NEO4J_URI, auth=(_NEO4J_USER, _NEO4J_PASSWORD)
        ) as drv:
            drv.verify_connectivity()
        return True
    except Exception:
        return False


_NEEDS_NEO4J = pytest.mark.skipif(
    not _neo4j_available(),
    reason=f"Neo4j not available at {_NEO4J_URI}",
)


_SCHEMA_PATH = Path(__file__).parent / "itinerary_schema.json"
# Fixed UTC day so the test is timezone-stable: it asks the CLI/MCP for
# a specific ISO date rather than relying on "today".
_DAY_ISO = "2026-04-29"
_EVENT_START = "2026-04-29T16:00:00Z"
_EVENT_END = "2026-04-29T17:00:00Z"
_EMAIL_LATEST = "2026-04-28T09:00:00Z"
_EMAIL_OLDER = "2026-04-26T09:00:00Z"


def _required_keys(schema: dict[str, Any]) -> set[str]:
    return set(schema.get("required", []))


# ---------------------------------------------------------------------------
# Seed: one CalendarEvent on _DAY_ISO with two attendees, an open OmniFocus
# task mentioning one attendee, and an Email Thread covering both attendees.
# ---------------------------------------------------------------------------


_SEED_CYPHER = """
MERGE (me:Person {email: 'me@example.com'})
  SET me.name = 'Self', me.is_self = true
MERGE (alice:Person {email: 'alice@example.com'})
  SET alice.name = 'Alice Example'
MERGE (bob:Person {email: 'bob@example.com'})
  SET bob.name = 'Bob Builder'

MERGE (cal:CalendarEvent {source_id: 'google_calendar.work:itin-q2'})
  SET cal.summary = 'Q2 sync',
      cal.description = 'Walk through Q2 plan with Alice and Bob.',
      cal.start_time = $start,
      cal.end_time = $end,
      cal.account = 'work',
      cal.calendar_id = 'me@example.com',
      cal.location = 'Zoom',
      cal.html_link = 'https://calendar.example.com/itin-q2'
MERGE (cal)-[:ORGANIZED_BY]->(me)
MERGE (cal)-[:ATTENDED_BY]->(alice)
MERGE (cal)-[:ATTENDED_BY]->(bob)

MERGE (proj:Project {source_id: 'omnifocus-project:Work'})
  SET proj.name = 'Work', proj.source = 'omnifocus'
MERGE (t_open:Task {source_id: 'of://itin-open-1'})
  SET t_open.name = 'Email Alice about Q2 plan',
      t_open.status = 'active',
      t_open.flagged = true,
      t_open.due_date = $due,
      t_open.modification_date = $due
MERGE (t_open)-[:IN_PROJECT]->(proj)
MERGE (t_open)-[:MENTIONS]->(alice)

MERGE (thread:Thread {source_id: 'gmail://thread/itin-q2'})
  SET thread.subject = 'Q2 planning thread'
MERGE (em_latest:Email {source_id: 'gmail://email/itin-2'})
  SET em_latest.subject = 'Re: Q2 planning thread', em_latest.date = $email_latest
MERGE (em_older:Email {source_id: 'gmail://email/itin-1'})
  SET em_older.subject = 'Q2 planning thread', em_older.date = $email_older
MERGE (em_latest)-[:PART_OF]->(thread)
MERGE (em_older)-[:PART_OF]->(thread)
MERGE (bob)-[:SENT]->(em_latest)
MERGE (em_latest)-[:TO]->(alice)
MERGE (em_latest)-[:TO]->(me)
MERGE (alice)-[:SENT]->(em_older)
MERGE (em_older)-[:TO]->(bob)
MERGE (em_older)-[:TO]->(me)

RETURN id(cal) AS cal_id
"""


def _seed_params() -> dict[str, str]:
    return {
        "start": _EVENT_START,
        "end": _EVENT_END,
        "due": "2026-04-28",
        "email_latest": _EMAIL_LATEST,
        "email_older": _EMAIL_OLDER,
    }


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
    with driver.session() as s:
        rec = s.run(_SEED_CYPHER, **_seed_params()).single()
        assert rec is not None
        return {k: int(v) for k, v in rec.data().items()}


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    """Write a minimal config.toml that points fieldnotes at the test Neo4j."""
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        f"""
[core]
data_dir = "{tmp_path}/data"
log_level = "warning"

[neo4j]
uri = "{_NEO4J_URI}"
user = "{_NEO4J_USER}"
password = "{_NEO4J_PASSWORD}"

[qdrant]
host = "localhost"
port = 6333
collection = "fieldnotes_e2e_itinerary"
""".strip()
        + "\n"
    )
    (tmp_path / "data").mkdir(exist_ok=True)
    return cfg


# ---------------------------------------------------------------------------
# Schema fixture — used by every assertion to keep README / fixture / tests
# aligned.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def schema() -> dict[str, Any]:
    return json.loads(_SCHEMA_PATH.read_text())


def _assert_payload_matches_schema(
    payload: dict[str, Any], schema: dict[str, Any]
) -> None:
    """Lightweight structural check (no jsonschema dep): top-level keys + per-event keys."""
    required = _required_keys(schema)
    missing = required - set(payload.keys())
    assert not missing, f"payload missing required top-level keys: {missing}"

    assert isinstance(payload["events"], list), "events must be a list"
    event_schema = schema["properties"]["events"]["items"]
    event_required = _required_keys(event_schema)
    for ev in payload["events"]:
        ev_missing = event_required - set(ev.keys())
        assert not ev_missing, f"event missing required keys: {ev_missing}"
        linked = ev["linked"]
        linked_required = _required_keys(event_schema["properties"]["linked"])
        assert linked_required <= set(linked.keys())
        # Per-row shape on populated lists.
        if linked["tasks"]:
            tk_required = _required_keys(
                event_schema["properties"]["linked"]["properties"]["tasks"]["items"]
            )
            assert tk_required <= set(linked["tasks"][0].keys())
        if linked["notes"]:
            nt_required = _required_keys(
                event_schema["properties"]["linked"]["properties"]["notes"]["items"]
            )
            assert nt_required <= set(linked["notes"][0].keys())
        if linked["thread"] is not None:
            th_required = _required_keys(
                event_schema["properties"]["linked"]["properties"]["thread"]
            )
            assert th_required <= set(linked["thread"].keys())


# ---------------------------------------------------------------------------
# 1. CLI --json round-trip via subprocess
# ---------------------------------------------------------------------------


def _run_cli(config: Path, *args: str) -> tuple[int, str, str]:
    """Run ``python -m worker.cli`` as a subprocess, return (rc, stdout, stderr)."""
    proc = subprocess.run(
        [sys.executable, "-m", "worker.cli", "-c", str(config), *args],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return proc.returncode, proc.stdout, proc.stderr


@_NEEDS_NEO4J
def test_e2e_itinerary_cli_json_matches_documented_schema(
    seeded: dict[str, int],
    config_file: Path,
    schema: dict[str, Any],
) -> None:
    """``fieldnotes itinerary --day <iso> --json --brief`` matches the schema fixture."""
    rc, out, err = _run_cli(
        config_file, "itinerary", "--day", _DAY_ISO, "--brief", "--json"
    )
    assert rc == 0, f"CLI exited {rc}; stderr={err}"

    payload = json.loads(out)
    _assert_payload_matches_schema(payload, schema)

    assert payload["day"] == _DAY_ISO
    assert isinstance(payload["timezone"], str) and payload["timezone"]
    # Exactly one seeded event lands on the requested day.
    assert len(payload["events"]) == 1, payload["events"]

    ev = payload["events"][0]
    assert ev["title"] == "Q2 sync"
    assert ev["account"] == "work"
    assert ev["source_id"] == "google_calendar.work:itin-q2"
    assert ev["start"] == _EVENT_START
    assert ev["end"] == _EVENT_END
    assert ev["next_brief"] is None  # --brief was set

    # Organizer + attendees pulled through the graph.
    assert ev["organizer"] == {"name": "Self", "email": "me@example.com"}
    attendee_emails = {a["email"] for a in ev["attendees"]}
    assert attendee_emails == {"alice@example.com", "bob@example.com"}

    # Linked tasks: exactly the open OmniFocus task we seeded.
    tasks = ev["linked"]["tasks"]
    assert len(tasks) == 1, tasks
    assert tasks[0]["title"] == "Email Alice about Q2 plan"
    assert tasks[0]["flagged"] is True
    assert tasks[0]["project"] == "Work"
    assert tasks[0]["source_id"] == "of://itin-open-1"

    # Linked thread: the Email Thread covering both attendees, latest email.
    thread = ev["linked"]["thread"]
    assert thread is not None
    assert thread["kind"] == "email"
    assert thread["source_id"] == "gmail://thread/itin-q2"
    assert thread["last_ts"] == _EMAIL_LATEST
    # ``last_from`` is the sender of the most recent message → bob.
    assert thread["last_from"] in {"bob@example.com", "Bob Builder"}

    # Notes: no Qdrant seed → empty (caught exception path).
    assert ev["linked"]["notes"] == []


# ---------------------------------------------------------------------------
# 2. MCP itinerary tool → CLI parity
# ---------------------------------------------------------------------------


async def _call_mcp_itinerary(config: Path) -> dict[str, Any]:
    """Spawn ``fieldnotes serve --mcp`` and call the ``itinerary`` tool over stdio."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "worker.cli", "-c", str(config), "serve", "--mcp"],
        env={**os.environ},
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "itinerary",
                {"day": _DAY_ISO, "brief": True},
            )
            assert result.content, "MCP itinerary tool returned no content"
            text_block = result.content[0]
            assert hasattr(text_block, "text"), text_block
            return json.loads(text_block.text)


@_NEEDS_NEO4J
def test_e2e_itinerary_mcp_payload_matches_cli_json(
    seeded: dict[str, int],
    config_file: Path,
    schema: dict[str, Any],
) -> None:
    """The MCP ``itinerary`` tool must emit the same payload shape the CLI does."""
    # CLI payload — ground truth.
    rc, out, err = _run_cli(
        config_file, "itinerary", "--day", _DAY_ISO, "--brief", "--json"
    )
    assert rc == 0, f"CLI exited {rc}; stderr={err}"
    cli_payload = json.loads(out)

    # MCP payload — must validate against the same schema and match content.
    mcp_payload = asyncio.run(_call_mcp_itinerary(config_file))
    _assert_payload_matches_schema(mcp_payload, schema)

    assert mcp_payload["day"] == cli_payload["day"]
    assert mcp_payload["timezone"] == cli_payload["timezone"]
    assert len(mcp_payload["events"]) == len(cli_payload["events"])

    cli_ev = cli_payload["events"][0]
    mcp_ev = mcp_payload["events"][0]

    # Top-level event identity.
    for key in (
        "event_id",
        "source_id",
        "title",
        "start",
        "end",
        "account",
        "calendar_id",
        "location",
        "html_link",
        "next_brief",
    ):
        assert mcp_ev[key] == cli_ev[key], f"event field {key!r} differs"

    # Organizer + attendees identity (order-stable).
    assert mcp_ev["organizer"] == cli_ev["organizer"]
    assert mcp_ev["attendees"] == cli_ev["attendees"]

    # Linked sub-shapes — same length and same row dicts.
    assert len(mcp_ev["linked"]["tasks"]) == len(cli_ev["linked"]["tasks"])
    assert mcp_ev["linked"]["tasks"][0] == cli_ev["linked"]["tasks"][0]
    assert len(mcp_ev["linked"]["notes"]) == len(cli_ev["linked"]["notes"])
    # Thread identity (kind, source_id, last_ts, last_from).
    assert mcp_ev["linked"]["thread"] == cli_ev["linked"]["thread"]


# ---------------------------------------------------------------------------
# 3 + 4. Default vs --brief: stub completion provider verifies grounding +
# zero-LLM behavior.
# ---------------------------------------------------------------------------


class _RecordingResolved:
    """Fake ResolvedModel: records every CompletionRequest and returns canned text."""

    def __init__(self, canned: str) -> None:
        self.requests: list[CompletionRequest] = []
        self._canned = canned

    def complete(
        self, req: CompletionRequest, *, task: str = "unknown"
    ) -> CompletionResponse:
        self.requests.append(req)
        return CompletionResponse(text=self._canned, input_tokens=1, output_tokens=1)


class _StubRegistry:
    """Stub registry that only knows the ``completion`` role.

    ``for_role('embed')`` raises ``KeyError`` so the notes path in
    :func:`worker.query.itinerary.get_itinerary` falls into its
    swallowed-exception branch and returns an empty notes list — keeping
    this test independent of Qdrant.
    """

    def __init__(self, resolved: _RecordingResolved) -> None:
        self._resolved = resolved

    def for_role(self, role: str) -> _RecordingResolved:
        if role != "completion":
            raise KeyError(role)
        return self._resolved


def _run_in_process(
    *,
    brief: bool,
    registry: Any,
) -> dict[str, Any]:
    """Invoke ``run_itinerary`` in-process, return the parsed JSON payload."""
    from worker.cli.itinerary import run_itinerary

    cfg = Config(
        neo4j=Neo4jConfig(uri=_NEO4J_URI, user=_NEO4J_USER, password=_NEO4J_PASSWORD)
    )

    out = io.StringIO()
    err = io.StringIO()
    with patch("worker.cli.itinerary.load_config", return_value=cfg):
        with redirect_stdout(out), redirect_stderr(err):
            code = run_itinerary(
                day=_DAY_ISO,
                json_output=True,
                brief=brief,
                registry=registry,
            )
    assert code == 0, (
        f"run_itinerary exited {code}; stdout={out.getvalue()!r} stderr={err.getvalue()!r}"
    )
    return json.loads(out.getvalue())


@_NEEDS_NEO4J
def test_e2e_itinerary_summary_brief_is_grounded_in_seed_data(
    seeded: dict[str, int],
) -> None:
    """Default mode: per-event ``next_brief`` is built from the seeded graph only."""
    canned_brief = (
        "- Walk through Q2 plan with Alice and Bob\n"
        "- Email Alice about Q2 plan is still open and flagged"
    )
    resolved = _RecordingResolved(canned_brief)
    registry = _StubRegistry(resolved)

    # Ensure local TZ resolves the seeded UTC event onto _DAY_ISO.
    with _force_utc_tz():
        payload = _run_in_process(brief=False, registry=registry)

    assert payload["day"] == _DAY_ISO
    assert len(payload["events"]) == 1
    ev = payload["events"][0]
    # next_brief is present and matches the canned brief on every event.
    assert ev["next_brief"] == canned_brief
    # The completion provider was invoked exactly once per event.
    assert len(resolved.requests) == 1, [r.system for r in resolved.requests]

    # Inspect the prompt body — it must reference seeded items only.
    request = resolved.requests[0]
    body_blob = (
        (request.system or "")
        + "\n"
        + "\n".join(str(m.get("content", "")) for m in request.messages)
    )
    assert "Q2 sync" in body_blob
    assert "Email Alice about Q2 plan" in body_blob
    # Anything we did not seed must not appear — guards against accidental
    # leakage of other graph contents into the prompt.
    assert "Acme Corp" not in body_blob
    assert "ghost@nowhere.invalid" not in body_blob


@_NEEDS_NEO4J
def test_e2e_itinerary_brief_flag_makes_zero_llm_calls(
    seeded: dict[str, int],
) -> None:
    """``--brief`` keeps ``next_brief`` null AND never resolves the completion role."""
    resolved = _RecordingResolved("should-not-be-emitted")
    registry = _StubRegistry(resolved)

    with _force_utc_tz():
        payload = _run_in_process(brief=True, registry=registry)

    assert payload["events"]
    for ev in payload["events"]:
        assert ev["next_brief"] is None, ev
    assert resolved.requests == [], (
        f"completion provider received {len(resolved.requests)} call(s) "
        "with --brief set"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _force_utc_tz:
    """Force ``_local_tz(None)`` to return UTC for the duration of the block.

    The seeded event has a UTC start; on hosts whose local TZ shifts the
    event off ``_DAY_ISO`` the test would silently see zero events.
    Patching the resolver to UTC keeps the assertions stable across CI
    environments.
    """

    def __enter__(self) -> "_force_utc_tz":
        from worker.query import itinerary as itinerary_q

        self._patch = patch.object(itinerary_q, "_local_tz", lambda _tz: timezone.utc)
        self._patch.start()
        from worker.cli import itinerary as itinerary_cli

        self._patch2 = patch.object(itinerary_cli, "_local_tz", lambda _tz: timezone.utc)
        self._patch2.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self._patch.stop()
        self._patch2.stop()
