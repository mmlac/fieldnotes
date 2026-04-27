"""Unit tests for the person-curation surface (split, confirm, merge, inspect).

Neo4j is mocked — these test the Cypher shape, the audit log, and the
identifier-parsing behaviour.  Integration with the real graph is covered
indirectly by the existing reconcile tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from worker.curation import (
    AuditLog,
    CurationError,
    PersonCurator,
    parse_identifier,
)
from worker.curation.persons import _match_person_clause


# ---------------------------------------------------------------------------
# parse_identifier
# ---------------------------------------------------------------------------


class TestParseIdentifier:
    def test_email(self):
        ref = parse_identifier("Alice@Example.com")
        assert ref.kind == "email"
        assert ref.email == "alice@example.com"
        assert ref.slack is None
        assert ref.name is None

    def test_slack_short_prefix(self):
        ref = parse_identifier("slack:T123/U456")
        assert ref.kind == "slack"
        assert ref.slack == ("T123", "U456")

    def test_slack_long_prefix(self):
        ref = parse_identifier("slack-user:T_X/U_Y")
        assert ref.slack == ("T_X", "U_Y")

    def test_invalid_slack_missing_user(self):
        with pytest.raises(CurationError):
            parse_identifier("slack:T123/")

    def test_name_fallback(self):
        ref = parse_identifier("Alice Smith")
        assert ref.kind == "name"
        assert ref.name == "Alice Smith"

    def test_strips_whitespace(self):
        ref = parse_identifier("  alice@example.com  ")
        assert ref.email == "alice@example.com"

    def test_empty_raises(self):
        with pytest.raises(CurationError):
            parse_identifier("")
        with pytest.raises(CurationError):
            parse_identifier("   ")


# ---------------------------------------------------------------------------
# _match_person_clause builder
# ---------------------------------------------------------------------------


class TestMatchPersonClause:
    def test_email_clause_uses_lower(self):
        clause, params = _match_person_clause("p", parse_identifier("a@b.com"))
        assert "toLower(p.email) = $p_email" in clause
        assert params == {"p_email": "a@b.com"}

    def test_slack_clause_binds_team_and_uid(self):
        clause, params = _match_person_clause(
            "x", parse_identifier("slack:T1/U2")
        )
        assert "x.team_id = $x_team" in clause
        assert "x.slack_user_id = $x_uid" in clause
        assert params == {"x_team": "T1", "x_uid": "U2"}

    def test_name_clause_matches_person_or_entity(self):
        clause, params = _match_person_clause("a", parse_identifier("Alice"))
        # case-insensitive by name, on Person OR Entity
        assert "a:Person OR a:Entity" in clause
        assert "toLower(a.name) = toLower($a_name)" in clause
        assert params == {"a_name": "Alice"}


# ---------------------------------------------------------------------------
# Mock-driver helpers
# ---------------------------------------------------------------------------


def _make_driver(run_responses: list):
    """Return (driver, session_mock).

    ``run_responses[i]`` is the .single() return for the i-th run() call.
    Use ``"iter:"`` followed by a list to make a result iterable instead.
    """
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(return_value=session)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)

    results: list = []
    for resp in run_responses:
        result = MagicMock()
        if isinstance(resp, dict) or resp is None:
            result.single.return_value = resp
        elif isinstance(resp, list):
            # Iterable result
            result.__iter__ = MagicMock(return_value=iter(resp))
        else:
            raise AssertionError(f"unsupported response: {resp!r}")
        results.append(result)
    session.run = MagicMock(side_effect=results)
    return driver, session


# ---------------------------------------------------------------------------
# inspect()
# ---------------------------------------------------------------------------


class TestInspect:
    def test_unknown_identifier_raises(self):
        driver, session = _make_driver([None])
        curator = PersonCurator(driver)
        with pytest.raises(CurationError):
            curator.inspect("ghost@example.com")

    def test_returns_focal_node_and_edges(self):
        focal_record = {
            "labels": ["Person"],
            "name": "Alice Smith",
            "email": "alice@example.com",
            "slack_user_id": None,
            "team_id": None,
            "is_self": False,
        }

        edge_records = [
            {
                "rtype": "SAME_AS",
                "outgoing": True,
                "olabels": ["Person"],
                "oname": "alice smith",
                "oemail": None,
                "oslack": None,
                "oteam": None,
                "match_type": "fuzzy_name",
                "confidence": 0.96,
                "cross_source": True,
            },
            {
                "rtype": "NEVER_SAME_AS",
                "outgoing": True,
                "olabels": ["Person"],
                "oname": "Alice Sommers",
                "oemail": None,
                "oslack": None,
                "oteam": None,
                "match_type": None,
                "confidence": None,
                "cross_source": None,
            },
        ]

        # Convert dicts into mocks supporting __getitem__
        def _to_record(d):
            rec = MagicMock()
            rec.__getitem__ = lambda self, key, _d=d: _d[key]
            return rec

        focal_mock = _to_record(focal_record)
        edge_mocks = [_to_record(r) for r in edge_records]

        driver = MagicMock()
        session = MagicMock()
        driver.session.return_value.__enter__ = MagicMock(return_value=session)
        driver.session.return_value.__exit__ = MagicMock(return_value=False)

        focal_result = MagicMock()
        focal_result.single.return_value = focal_mock

        edges_result = MagicMock()
        edges_result.__iter__ = MagicMock(return_value=iter(edge_mocks))

        session.run = MagicMock(side_effect=[focal_result, edges_result])

        curator = PersonCurator(driver)
        result = curator.inspect("alice@example.com")

        assert result.focal is not None
        assert result.focal.name == "Alice Smith"
        assert result.focal.label == "Person"
        assert len(result.same_as) == 1
        assert result.same_as[0].match_type == "fuzzy_name"
        assert result.same_as[0].confidence == pytest.approx(0.96)
        assert len(result.never_same_as) == 1
        assert result.never_same_as[0].other_name == "Alice Sommers"


# ---------------------------------------------------------------------------
# split()
# ---------------------------------------------------------------------------


class TestSplit:
    def test_unknown_endpoint_raises(self):
        # Two count() probes; the second returns 0
        driver, session = _make_driver([{"cnt": 1}, {"cnt": 0}])
        curator = PersonCurator(driver)
        with pytest.raises(CurationError):
            curator.split("alice@example.com", "ghost@example.com")

    def test_removes_same_as_and_creates_block(self, tmp_path: Path):
        # Sequence: count(a)=1, count(b)=1, removed=1, block created
        driver, session = _make_driver(
            [
                {"cnt": 1},
                {"cnt": 1},
                {"removed": 1},
                {"created_at": 12345},
            ]
        )
        audit_path = tmp_path / "audit.jsonl"
        audit = AuditLog(path=audit_path)
        curator = PersonCurator(driver, audit=audit)

        result = curator.split("alice@example.com", "alice@work.com")
        assert result.action == "split"
        assert result.detail["same_as_removed"] == 1
        assert result.detail["never_same_as_created_at"] == 12345

        # Verify the third query was a DELETE r and the fourth was a MERGE NEVER_SAME_AS
        third_call = session.run.call_args_list[2]
        assert "DELETE r" in third_call[0][0]
        fourth_call = session.run.call_args_list[3]
        assert "MERGE (a)-[r:NEVER_SAME_AS]->(b)" in fourth_call[0][0]
        assert "ON CREATE SET" in fourth_call[0][0]

        # Audit log captured the action
        assert audit_path.exists()
        line = audit_path.read_text().strip()
        record = json.loads(line)
        assert record["action"] == "split"
        assert record["actor"] == "user"
        assert record["args"] == {
            "identifier": "alice@example.com",
            "member": "alice@work.com",
        }


# ---------------------------------------------------------------------------
# confirm() / merge()
# ---------------------------------------------------------------------------


class TestConfirmMerge:
    def test_confirm_writes_user_confirmed_edge(self, tmp_path: Path):
        # count(a)=1, count(b)=1, blocked-check returns False, MERGE returns ts.
        driver, session = _make_driver(
            [
                {"cnt": 1},
                {"cnt": 1},
                {"blocked": False, "same_node": False},
                {"confirmed_at": 9999},
            ]
        )
        audit = AuditLog(path=tmp_path / "audit.jsonl")
        curator = PersonCurator(driver, audit=audit)

        result = curator.confirm("alice@example.com", "alice@work.com")
        assert result.action == "confirm"
        assert result.detail["match_type"] == "user_confirmed"
        assert result.detail["confidence"] == 1.0
        assert result.detail["confirmed_at"] == 9999

        # The MERGE query sets match_type='user_confirmed' / confidence=1.0
        merge_call = session.run.call_args_list[3]
        cypher = merge_call[0][0]
        assert "MERGE (a)-[r:SAME_AS]->(b)" in cypher
        assert "match_type = 'user_confirmed'" in cypher
        assert "confidence = 1.0" in cypher

    def test_merge_uses_same_implementation(self, tmp_path: Path):
        driver, session = _make_driver(
            [
                {"cnt": 1},
                {"cnt": 1},
                {"blocked": False, "same_node": False},
                {"confirmed_at": 1},
            ]
        )
        audit = AuditLog(path=tmp_path / "audit.jsonl")
        curator = PersonCurator(driver, audit=audit)

        result = curator.merge("alice@example.com", "manual@x.com")
        assert result.action == "merge"
        assert result.detail["match_type"] == "user_confirmed"

    def test_blocked_pair_refuses_confirm(self):
        # blocked = True → CurationError
        driver, session = _make_driver(
            [
                {"cnt": 1},
                {"cnt": 1},
                {"blocked": True, "same_node": False},
            ]
        )
        curator = PersonCurator(driver)
        with pytest.raises(CurationError, match="NEVER_SAME_AS"):
            curator.confirm("alice@example.com", "alice@work.com")

    def test_self_merge_refused(self):
        driver, session = _make_driver(
            [
                {"cnt": 1},
                {"cnt": 1},
                {"blocked": False, "same_node": True},
            ]
        )
        curator = PersonCurator(driver)
        with pytest.raises(CurationError, match="cannot confirm a node with itself"):
            curator.confirm("alice@example.com", "alice@example.com")


# ---------------------------------------------------------------------------
# AuditLog
# ---------------------------------------------------------------------------


class TestAuditLog:
    def test_appends_one_jsonl_record_per_call(self, tmp_path: Path):
        log = AuditLog.from_data_dir(tmp_path)
        log.append("split", args={"a": "1"}, result={"ok": True})
        log.append("confirm", args={"a": "1", "b": "2"}, result={"confidence": 1.0})

        lines = log.path.read_text().strip().split("\n")
        assert len(lines) == 2
        first = json.loads(lines[0])
        second = json.loads(lines[1])
        assert first["action"] == "split"
        assert second["action"] == "confirm"
        assert first["actor"] == "user"
        assert "ts" in first

    def test_creates_data_dir_if_missing(self, tmp_path: Path):
        target = tmp_path / "subdir" / "data"
        log = AuditLog.from_data_dir(target)
        assert target.exists()
        assert log.path == target / "curation_audit.jsonl"
