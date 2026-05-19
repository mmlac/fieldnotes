"""NEVER_SAME_AS guards in the reconcile pipeline.

These tests verify that every reconcile step's Cypher refuses to create
a SAME_AS edge between Person/Entity nodes connected by a user-installed
NEVER_SAME_AS block.  The check is done by inspecting the rendered query
strings — running the actual Cypher requires a live Neo4j and is covered
by the integration suite.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from worker.config import MeConfig
from worker.pipeline.writer import Writer


def _writer_with_session():
    writer = object.__new__(Writer)
    writer._neo4j_driver = MagicMock()
    session = MagicMock()
    writer._neo4j_driver.session.return_value.__enter__ = MagicMock(
        return_value=session
    )
    writer._neo4j_driver.session.return_value.__exit__ = MagicMock(return_value=False)
    return writer, session


def _all_cypher(session) -> str:
    """Concatenate all Cypher strings session.run was called with."""
    return "\n".join(call.args[0] for call in session.run.call_args_list)


# ---------------------------------------------------------------------------
# Email-based reconcile
# ---------------------------------------------------------------------------


def test_email_reconcile_skips_never_same_as():
    writer, session = _writer_with_session()
    # First call returns updated count for the rename pass; second call is
    # the SAME_AS MERGE — no .single() needed.
    rename = MagicMock()
    rename.single.return_value = {"updated": 0}
    same_as = MagicMock()
    session.run.side_effect = [rename, same_as]

    writer._reconcile_persons_neo4j()

    cypher = _all_cypher(session)
    assert "NOT (a)-[:NEVER_SAME_AS]-(b)" in cypher


# ---------------------------------------------------------------------------
# Slack-id reconcile
# ---------------------------------------------------------------------------


def test_slack_reconcile_skips_never_same_as():
    writer, session = _writer_with_session()
    result = MagicMock()
    result.single.return_value = {"cnt": 0}
    session.run.return_value = result

    writer._reconcile_persons_by_slack_user_neo4j()

    cypher = _all_cypher(session)
    assert "NOT (a)-[:NEVER_SAME_AS]-(b)" in cypher


# ---------------------------------------------------------------------------
# Fuzzy name reconcile
# ---------------------------------------------------------------------------


def test_fuzzy_name_reconcile_skips_never_same_as():
    writer, session = _writer_with_session()

    # First call returns names; with two close names we'll produce a pair
    rec_a = MagicMock()
    rec_a.__getitem__ = lambda self, key: "Alice Johnson"
    rec_b = MagicMock()
    rec_b.__getitem__ = lambda self, key: "Alice Johnso"
    name_result = MagicMock()
    name_result.__iter__ = MagicMock(return_value=iter([rec_a, rec_b]))

    create_result = MagicMock()
    create_result.single.return_value = {"cnt": 0}
    session.run.side_effect = [name_result, create_result]

    writer._reconcile_persons_by_name_neo4j()

    cypher = _all_cypher(session)
    assert "NOT (a)-[:NEVER_SAME_AS]-(b)" in cypher


# ---------------------------------------------------------------------------
# Entity → Person bridge
# ---------------------------------------------------------------------------


def test_entity_person_bridge_skips_never_same_as():
    writer, session = _writer_with_session()

    def _rec(name):
        r = MagicMock()
        r.__getitem__ = lambda self, key, _n=name: _n
        return r

    entities = MagicMock()
    entities.__iter__ = MagicMock(return_value=iter([_rec("Alice Smith")]))
    persons = MagicMock()
    persons.__iter__ = MagicMock(return_value=iter([_rec("Alice Smith")]))

    create_result = MagicMock()
    create_result.single.return_value = {"cnt": 0}

    session.run.side_effect = [entities, persons, create_result]

    writer._bridge_entity_persons_neo4j()

    cypher = _all_cypher(session)
    assert "NOT (e)-[:NEVER_SAME_AS]-(p)" in cypher


# ---------------------------------------------------------------------------
# Cross-source resolution
# ---------------------------------------------------------------------------


def test_cross_source_resolve_skips_never_same_as(monkeypatch):
    writer, session = _writer_with_session()

    # Build two source-type buckets so resolve_cross_source has matches.
    src_record = MagicMock()
    src_record.__getitem__ = lambda self, key: {
        "src_type": "gmail",
        "entities": [{"name": "Alice", "type": "Person", "confidence": 1.0}],
    }[key]
    src_result = MagicMock()
    src_result.__iter__ = MagicMock(return_value=iter([src_record]))

    person_result = MagicMock()
    person_result.__iter__ = MagicMock(return_value=iter([]))

    # Patch resolve_cross_source to return one match so the MERGE query runs.
    from worker.pipeline import writer as writer_mod

    fake_match = MagicMock()
    fake_match.entity_a = "Alice"
    fake_match.entity_b = "alice"
    fake_match.confidence = 0.99
    fake_match.match_type = "fuzzy_name"

    monkeypatch.setattr(
        writer_mod, "resolve_cross_source", lambda by_source: [fake_match]
    )

    # Ensure entities_by_source has 2+ source types so the function proceeds.
    src_record_2 = MagicMock()
    src_record_2.__getitem__ = lambda self, key: {
        "src_type": "obsidian",
        "entities": [{"name": "alice", "type": "Person", "confidence": 1.0}],
    }[key]
    src_result.__iter__ = MagicMock(
        return_value=iter([src_record, src_record_2])
    )

    create_result = MagicMock()
    create_result.single.return_value = {"cnt": 0}

    session.run.side_effect = [src_result, person_result, create_result]

    writer._resolve_cross_source_neo4j()

    cypher = _all_cypher(session)
    assert "NOT (a)-[:NEVER_SAME_AS]-(b)" in cypher


# -- regression: review finding fn-29g block 2 --
# Entity nodes are MERGE'd on name.  Previously they had no source_id, so
# `AND a.source_id <> b.source_id` evaluated to null/false for Entity pairs,
# silently preventing all cross-source SAME_AS edges.
#
# The fix (Option A per Mayor direction 2026-05-19): Entity creation paths
# (_upsert_entity, _batch_upsert_entities, _merge_entity_edge) now set
# source_id = coalesce(source_id, 'entity:' + name) on every MERGE.  The
# SAME_AS guard uses the direct `a.source_id <> b.source_id` comparison —
# no coalesce fallback needed.


def test_cross_source_entity_reconciliation_uses_direct_source_id(monkeypatch):
    """SAME_AS MERGE guard uses direct source_id comparison for Entity nodes.

    Entity creation paths guarantee every Entity has source_id set, so
    ``a.source_id <> b.source_id`` is safe and no coalesce fallback is needed.
    """
    writer, session = _writer_with_session()

    from worker.pipeline import writer as writer_mod

    fake_match = MagicMock()
    fake_match.entity_a = "Python"
    fake_match.entity_b = "python"
    fake_match.confidence = 0.95
    fake_match.match_type = "fuzzy_name"

    monkeypatch.setattr(
        writer_mod, "resolve_cross_source", lambda by_source: [fake_match]
    )

    src_record_1 = MagicMock()
    src_record_1.__getitem__ = lambda self, key: {
        "src_type": "gmail",
        "entities": [{"name": "Python", "type": "Concept", "confidence": 1.0}],
    }[key]
    src_record_2 = MagicMock()
    src_record_2.__getitem__ = lambda self, key: {
        "src_type": "obsidian",
        "entities": [{"name": "python", "type": "Concept", "confidence": 1.0}],
    }[key]
    src_result = MagicMock()
    src_result.__iter__ = MagicMock(return_value=iter([src_record_1, src_record_2]))

    person_result = MagicMock()
    person_result.__iter__ = MagicMock(return_value=iter([]))

    create_result = MagicMock()
    create_result.single.return_value = {"cnt": 1}

    session.run.side_effect = [src_result, person_result, create_result]

    writer._resolve_cross_source_neo4j()

    merge_cypher = session.run.call_args_list[2][0][0]
    assert "a.source_id <> b.source_id" in merge_cypher
    assert "elementId" not in merge_cypher


def test_cross_source_entity_reconciliation_returns_edge_count(monkeypatch):
    """Cross-source entity reconciliation returns the count of SAME_AS edges created.

    Regression guard: when Entity nodes lack source_id the WHERE guard evaluates
    to null/false and 0 edges are returned silently.  With source_id backfilled
    by creation paths, the count flows through correctly.
    """
    writer, session = _writer_with_session()

    from worker.pipeline import writer as writer_mod

    fake_match = MagicMock()
    fake_match.entity_a = "Python"
    fake_match.entity_b = "python"
    fake_match.confidence = 0.95
    fake_match.match_type = "fuzzy_name"

    monkeypatch.setattr(
        writer_mod, "resolve_cross_source", lambda by_source: [fake_match]
    )

    src_record_1 = MagicMock()
    src_record_1.__getitem__ = lambda self, key: {
        "src_type": "gmail",
        "entities": [{"name": "Python", "type": "Concept", "confidence": 1.0}],
    }[key]
    src_record_2 = MagicMock()
    src_record_2.__getitem__ = lambda self, key: {
        "src_type": "obsidian",
        "entities": [{"name": "python", "type": "Concept", "confidence": 1.0}],
    }[key]
    src_result = MagicMock()
    src_result.__iter__ = MagicMock(return_value=iter([src_record_1, src_record_2]))

    person_result = MagicMock()
    person_result.__iter__ = MagicMock(return_value=iter([]))

    create_result = MagicMock()
    create_result.single.return_value = {"cnt": 2}

    session.run.side_effect = [src_result, person_result, create_result]

    count = writer._resolve_cross_source_neo4j()
    assert count == 2


# ---------------------------------------------------------------------------
# Self-identity reconcile
# ---------------------------------------------------------------------------


def test_self_identity_skips_never_same_as():
    writer, session = _writer_with_session()

    phase1 = MagicMock()
    phase1.single.return_value = {"names": ["Me"]}
    phase2 = MagicMock()  # display_name update — no .single inspection
    phase3 = MagicMock()
    phase3.single.return_value = {"cnt": 0}
    phase4 = MagicMock()
    phase4.single.return_value = {"cnt": 0}

    session.run.side_effect = [phase1, phase2, phase3, phase4]

    writer._reconcile_self_person_neo4j(
        MeConfig(emails=["me@a.com", "me@b.com"])
    )

    cypher = _all_cypher(session)
    # Pairwise + closure both guarded
    assert cypher.count("NOT (a)-[:NEVER_SAME_AS]-(b)") >= 2


# ---------------------------------------------------------------------------
# Transitive closure
# ---------------------------------------------------------------------------


def test_transitive_closure_respects_never_same_as_endpoints():
    writer, session = _writer_with_session()
    result = MagicMock()
    result.single.return_value = {"cnt": 0}
    session.run.return_value = result

    writer._close_same_as_transitive_neo4j()

    cypher = _all_cypher(session)
    # The endpoints must be checked for NEVER_SAME_AS
    assert "NOT (a)-[:NEVER_SAME_AS]-(b)" in cypher
