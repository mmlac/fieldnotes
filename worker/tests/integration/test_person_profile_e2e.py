"""End-to-end integration test for the Person profile feature (fn-364).

Round-trips the full feature path on a seeded Neo4j instance:

* CLI ``fieldnotes person <id> --json`` (subprocess)
* MCP ``person`` tool over stdio (subprocess + ``mcp`` client)
* CLI ``fieldnotes person <id> --summary`` (in-process with a stub
  completion provider)

The test seeds a multi-source graph (Gmail + Calendar + Slack +
OmniFocus + Obsidian + Topic) and verifies the documented JSON schema
matches what the CLI prints, what the MCP tool returns, and what the
``--summary`` brief was grounded in.

Skipped automatically when Neo4j isn't reachable — same pattern as
``tests/cli/test_person.py``.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Generator
from unittest.mock import patch

import pytest
from neo4j import Driver, GraphDatabase

from worker.config import Config, MeConfig, Neo4jConfig
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


_SCHEMA_PATH = Path(__file__).parent / "person_profile_schema.json"
_NOW = datetime(2026, 4, 27, 12, 0, 0, tzinfo=timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _slack_ts(dt: datetime) -> str:
    return f"{dt.timestamp():.6f}"


def _required_keys(schema: dict[str, Any]) -> set[str]:
    return set(schema.get("required", []))


# ---------------------------------------------------------------------------
# Seed: Alice with edges across every people-aware source.
# ---------------------------------------------------------------------------


_SEED_CYPHER = """
MERGE (p_main:Person {email: 'alice@example.com'})
  SET p_main.name = 'Alice Example',
      p_main.slack_user_id = 'U-ALICE',
      p_main.team_id = 'T-TEAM'
MERGE (p_alt:Person {email: 'alice.alt@example.com'})
  SET p_alt.name = 'Alice Example'
MERGE (p_main)-[r1:SAME_AS]->(p_alt)
  SET r1.match_type = 'fuzzy_name',
      r1.confidence = 0.97,
      r1.cross_source = true

MERGE (p_bob:Person {email: 'bob@example.com'})
  SET p_bob.name = 'Bob Builder'

MERGE (e1:Email {source_id: 'gmail://1'})
  SET e1.subject = 'Project kickoff', e1.date = $d_now
MERGE (e2:Email {source_id: 'gmail://2'})
  SET e2.subject = 'Re: weekly sync', e2.date = $d_y
MERGE (p_main)-[:SENT]->(e1)
MERGE (e2)-[:TO]->(p_main)
MERGE (e1)-[:TO]->(p_bob)

MERGE (cal1:CalendarEvent {source_id: 'cal://q2-planning'})
  SET cal1.summary = 'Q2 planning', cal1.start_time = $d_2d
MERGE (cal1)-[:ORGANIZED_BY]->(p_main)

MERGE (s1:SlackMessage {source_id: 'slack://1'})
  SET s1.channel_name = 'eng', s1.first_ts = $sl_y, s1.last_ts = $sl_y
MERGE (s1)-[:SENT_BY]->(p_main)

MERGE (t_open:Task {source_id: 'of://open-1'})
  SET t_open.name = 'Email Alice about Q2',
      t_open.status = 'active',
      t_open.flagged = true,
      t_open.due_date = $d_y
MERGE (proj:Project {source_id: 'omnifocus-project:Work'})
  SET proj.name = 'Work', proj.source = 'omnifocus'
MERGE (t_open)-[:IN_PROJECT]->(proj)
MERGE (t_open)-[:MENTIONS]->(p_main)

MERGE (topic:Topic {name: 'Q2 planning'}) SET topic.source = 'user'
MERGE (e1)-[:TAGGED]->(topic)
MERGE (cal1)-[:TAGGED]->(topic)
MERGE (s1)-[:TAGGED]->(topic)

MERGE (f1:File {source_id: '/notes/alice.md'})
  SET f1.path = '/notes/alice.md',
      f1.modified_at = $d_now,
      f1.source = 'obsidian'
MERGE (f1)-[:MENTIONS]->(p_main)

RETURN id(p_main) AS p_main_id
"""


def _seed_params() -> dict[str, str]:
    return {
        "d_now": _iso(_NOW),
        "d_y": _iso(_NOW - timedelta(days=1)),
        "d_2d": _iso(_NOW - timedelta(days=2)),
        "sl_y": _slack_ts(_NOW - timedelta(days=1)),
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
collection = "fieldnotes_e2e"
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
    """Lightweight structural check (no jsonschema dep): top-level keys + per-row keys."""
    required = _required_keys(schema)
    missing = required - set(payload.keys())
    assert not missing, f"payload missing required top-level keys: {missing}"

    props = schema["properties"]
    # resolved
    resolved_required = _required_keys(props["resolved"])
    assert resolved_required <= set(payload["resolved"].keys())

    # array sections
    for key in (
        "recent_interactions",
        "top_topics",
        "related_people",
        "open_tasks",
        "files_mentioning",
        "identity_cluster",
    ):
        assert isinstance(payload[key], list), f"{key} must be a list"
        item_schema = props[key].get("items")
        if item_schema and payload[key]:
            row_required = _required_keys(item_schema)
            sample_keys = set(payload[key][0].keys())
            missing_row = row_required - sample_keys
            assert not missing_row, f"{key}[0] missing keys: {missing_row}"


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
def test_e2e_cli_json_matches_documented_schema(
    seeded: dict[str, int],
    config_file: Path,
    schema: dict[str, Any],
) -> None:
    rc, out, err = _run_cli(
        config_file, "person", "alice@example.com", "--since", "365d", "--json"
    )
    assert rc == 0, f"CLI exited {rc}; stderr={err}"

    payload = json.loads(out)
    _assert_payload_matches_schema(payload, schema)

    # Expected counts pulled from the seed graph.
    assert payload["resolved"]["email"] == "alice@example.com"
    assert payload["resolved"]["name"] == "Alice Example"
    assert payload["resolved"]["is_self"] is False

    sources = set(payload["sources_present"])
    # gmail (sent + received), calendar (organized), slack (sent), omnifocus
    # (open task → MENTIONS), obsidian/file mention.
    assert {"gmail", "omnifocus"} <= sources
    assert payload["total_interactions"] >= 5

    # Every recent interaction's source must be one we seeded.
    seeded_sources = {"gmail", "calendar", "slack", "file"}
    for ix in payload["recent_interactions"]:
        assert ix["source_type"] in seeded_sources, ix

    # Topic + related people + open task all reflect the seeded graph.
    assert any(t["topic_name"] == "Q2 planning" for t in payload["top_topics"])
    assert any(r["email"] == "bob@example.com" for r in payload["related_people"])
    assert any(t["title"] == "Email Alice about Q2" for t in payload["open_tasks"])

    # SAME_AS alias surfaces on the cluster row.
    cluster_members = {m["member"] for m in payload["identity_cluster"]}
    assert "alice.alt@example.com" in cluster_members


# ---------------------------------------------------------------------------
# 2. MCP person tool → CLI parity
# ---------------------------------------------------------------------------


async def _call_mcp_person(config: Path) -> dict[str, Any]:
    """Spawn ``fieldnotes serve --mcp`` and call the ``person`` tool over stdio."""
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
                "person",
                {"identifier": "alice@example.com", "since": "365d"},
            )
            assert result.content, "MCP person tool returned no content"
            text_block = result.content[0]
            assert hasattr(text_block, "text"), text_block
            return json.loads(text_block.text)


@_NEEDS_NEO4J
def test_e2e_mcp_payload_matches_cli_json(
    seeded: dict[str, int],
    config_file: Path,
    schema: dict[str, Any],
) -> None:
    # CLI payload — ground truth.
    rc, out, err = _run_cli(
        config_file, "person", "alice@example.com", "--since", "365d", "--json"
    )
    assert rc == 0, f"CLI exited {rc}; stderr={err}"
    cli_payload = json.loads(out)

    # MCP payload — must match shape and content.
    mcp_payload = asyncio.run(_call_mcp_person(config_file))
    _assert_payload_matches_schema(mcp_payload, schema)

    # Top-level identity matches.
    assert mcp_payload["resolved"] == cli_payload["resolved"]
    assert mcp_payload["identifier"].strip() == cli_payload["identifier"].strip()
    assert mcp_payload["total_interactions"] == cli_payload["total_interactions"]
    assert sorted(mcp_payload["sources_present"]) == sorted(
        cli_payload["sources_present"]
    )

    # Every list section has the same length (parity guarantee).
    for key in (
        "recent_interactions",
        "top_topics",
        "related_people",
        "open_tasks",
        "files_mentioning",
        "identity_cluster",
    ):
        assert len(mcp_payload[key]) == len(cli_payload[key]), (
            f"section {key!r} length differs between CLI and MCP"
        )


# ---------------------------------------------------------------------------
# 3. --summary brief grounded in the seed dataset
# ---------------------------------------------------------------------------


class _RecordingResolved:
    """Fake ResolvedModel: records the CompletionRequest and returns canned text."""

    def __init__(self, canned: str) -> None:
        self.last_request: CompletionRequest | None = None
        self._canned = canned

    def complete(
        self, req: CompletionRequest, *, task: str = "unknown"
    ) -> CompletionResponse:
        self.last_request = req
        return CompletionResponse(text=self._canned, input_tokens=1, output_tokens=1)


class _StubRegistry:
    def __init__(self, resolved: _RecordingResolved) -> None:
        self._resolved = resolved

    def for_role(self, role: str) -> _RecordingResolved:
        if role != "completion":
            raise KeyError(role)
        return self._resolved


@_NEEDS_NEO4J
def test_e2e_summary_brief_is_grounded_in_seed_data(
    seeded: dict[str, int],
    config_file: Path,
) -> None:
    """`--summary` must produce a grounded `next_brief` and only feed seeded data to the LLM."""
    from worker.cli.person import run_person

    canned_brief = (
        "- [Open OmniFocus tasks] Email Alice about Q2 — flagged\n"
        "- [Recent interactions] Q2 planning topic surfaced via Gmail and Slack"
    )
    resolved = _RecordingResolved(canned_brief)
    registry = _StubRegistry(resolved)

    cfg = Config(
        neo4j=Neo4jConfig(uri=_NEO4J_URI, user=_NEO4J_USER, password=_NEO4J_PASSWORD)
    )
    cfg.me = MeConfig(emails=["self@example.com"], name="Self")

    out = io.StringIO()
    err = io.StringIO()
    with patch("worker.cli.person.load_config", return_value=cfg):
        with redirect_stdout(out), redirect_stderr(err):
            code = run_person(
                identifier="alice@example.com",
                since="365d",
                json_output=True,
                summary=True,
                horizon="365d",
                registry=registry,
            )

    assert code == 0, f"run_person exited {code}; stderr={err.getvalue()}"
    payload = json.loads(out.getvalue())

    # next_brief is present and contains the canned text.
    assert "next_brief" in payload
    assert payload["next_brief"] == canned_brief
    assert "Email Alice about Q2" in payload["next_brief"]

    # The completion request must have been built from the seed data only —
    # i.e. the prebrief assembly fed real graph rows to the LLM.  We assert
    # this by checking the request body references seeded items and contains
    # no items that weren't seeded.
    assert resolved.last_request is not None
    body_blob = (
        resolved.last_request.system
        + "\n"
        + "\n".join(str(m.get("content", "")) for m in resolved.last_request.messages)
    )
    # Seeded names must appear somewhere in the prompt context.
    assert "Email Alice about Q2" in body_blob
    # Anything we didn't seed must not — guard against accidental leakage of
    # other graph contents into the prompt.
    assert "Acme Corp" not in body_blob
    assert "ghost@nowhere.invalid" not in body_blob
