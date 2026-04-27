"""Tests for ``[me]`` self-identity reconciliation.

Covers ``Writer.reconcile_self_person()`` (mocked Neo4j session) and the
pipeline wiring that registers the step at the end of the reconcile
chain when ``cfg.me`` is set.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from worker.config import MeConfig
from worker.parsers.base import ParsedDocument
from worker.pipeline import Pipeline
from worker.pipeline.writer import Writer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_writer_with_responses(
    responses: list[dict | None],
) -> tuple[Writer, MagicMock]:
    """Writer with a mocked Neo4j session returning one response per ``run`` call.

    ``responses[i]`` is the dict yielded by ``session.run(...).single()``
    for the i-th call.  ``None`` means "no .single() call expected".
    """
    writer = object.__new__(Writer)
    writer._neo4j_driver = MagicMock()

    mock_session = MagicMock()
    writer._neo4j_driver.session.return_value.__enter__ = MagicMock(
        return_value=mock_session
    )
    writer._neo4j_driver.session.return_value.__exit__ = MagicMock(return_value=False)

    run_results: list[MagicMock] = []
    for resp in responses:
        result = MagicMock()
        if resp is None:
            result.single.return_value = None
        else:
            result.single.return_value = resp
        run_results.append(result)
    mock_session.run = MagicMock(side_effect=run_results)

    return writer, mock_session


# ---------------------------------------------------------------------------
# reconcile_self_person — Writer-level tests (mocked Neo4j)
# ---------------------------------------------------------------------------


class TestReconcileSelfPersonWriter:
    def test_empty_emails_is_noop(self):
        """``MeConfig.emails == []`` is a true no-op (no Neo4j queries)."""
        writer, session = _make_writer_with_responses([])
        result = writer.reconcile_self_person(MeConfig(emails=[]))
        assert result == 0
        session.run.assert_not_called()

    def test_single_email_no_same_as_edges(self):
        """One email: MERGE the Person, set is_self, but skip SAME_AS step."""
        writer, session = _make_writer_with_responses(
            [
                {"names": ["Alice Existing"]},  # phase 1: MERGE + collect names
                # phase 2 is a session.run with no .single() inspection — pass
                # a placeholder result so side_effect doesn't run dry.
                None,
            ]
        )
        n = writer.reconcile_self_person(MeConfig(emails=["alice@personal.com"]))
        assert n == 0
        # Two queries: phase 1 MERGE + phase 2 display_name update.
        # No SAME_AS query because there's only one email.
        assert session.run.call_count == 2
        merged_args = session.run.call_args_list[0]
        assert "MERGE (p:Person {email: email})" in merged_args[0][0]
        assert "is_self = true" in merged_args[0][0]
        display_args = session.run.call_args_list[1]
        assert "display_name = $chosen" in display_args[0][0]
        assert display_args.kwargs["chosen"] == "Alice Existing"

    def test_multi_email_creates_pairwise_same_as(self):
        """Two emails → one SAME_AS edge with self_identity metadata.

        Closure runs after pairwise but adds nothing here (only pairwise
        endpoints and no legacy is_self=true Persons in the graph).
        """
        writer, session = _make_writer_with_responses(
            [
                {"names": ["Alice", "Alice Smith"]},  # phase 1
                None,  # phase 2 (display_name update)
                {"cnt": 1},  # phase 3 (pairwise SAME_AS)
                {"cnt": 0},  # phase 4 (closure — no new edges)
            ]
        )
        n = writer.reconcile_self_person(
            MeConfig(emails=["alice@personal.com", "alice@work.com"])
        )
        # n is pairwise + closure = 1 + 0
        assert n == 1
        assert session.run.call_count == 4

        # Display-name update should choose the longest existing name.
        display_args = session.run.call_args_list[1]
        assert display_args.kwargs["chosen"] == "Alice Smith"

        # Pairwise SAME_AS query carries the right metadata.
        same_as_args = session.run.call_args_list[2]
        cypher = same_as_args[0][0]
        assert "match_type = 'self_identity'" in cypher
        assert "confidence = 1.0" in cypher
        assert "cross_source = true" in cypher

        # Closure query is scoped to is_self=true on both endpoints and
        # along the path.
        closure_args = session.run.call_args_list[3]
        closure_cypher = closure_args[0][0]
        assert "is_self: true" in closure_cypher
        assert "SAME_AS*2..4" in closure_cypher
        assert "ALL(node IN nodes(path) WHERE node.is_self = true)" in closure_cypher

    def test_explicit_name_overrides_longest_rule(self):
        """``me.name`` always wins, even when shorter than existing names."""
        writer, session = _make_writer_with_responses(
            [
                {"names": ["Alice Smithington-Wellington"]},
                None,
            ]
        )
        writer.reconcile_self_person(MeConfig(emails=["alice@personal.com"], name="Al"))
        display_args = session.run.call_args_list[1]
        assert display_args.kwargs["chosen"] == "Al"

    def test_no_existing_names_skips_display_name_update(self):
        """If no Person had a name, no ``display_name`` update query runs."""
        writer, session = _make_writer_with_responses(
            [
                {"names": [None, None]},  # all Persons name-less
                {"cnt": 1},  # pairwise SAME_AS
                {"cnt": 0},  # closure — no new edges
            ]
        )
        n = writer.reconcile_self_person(MeConfig(emails=["a@x.com", "b@x.com"]))
        assert n == 1
        # phase 1 + pairwise + closure, no display_name update phase
        assert session.run.call_count == 3
        cyphers = [c[0][0] for c in session.run.call_args_list]
        assert not any("display_name = $chosen" in c for c in cyphers)

    def test_dedupes_post_canonicalization(self):
        """Identical canonicalized emails collapse to a single Person.

        ``[me]`` parsing canonicalises ``@googlemail.com`` to ``@gmail.com``,
        so a config like ``['me@googlemail.com', 'me@gmail.com']`` arrives
        here as ``['me@gmail.com', 'me@gmail.com']``.  After dedupe there
        is one email, so no SAME_AS edges are created.
        """
        writer, session = _make_writer_with_responses(
            [
                {"names": ["Me"]},
                None,
            ]
        )
        n = writer.reconcile_self_person(
            MeConfig(emails=["me@gmail.com", "me@gmail.com"])
        )
        assert n == 0
        # No SAME_AS query.
        assert session.run.call_count == 2

        # Phase 1 was called with the deduped list.
        merge_args = session.run.call_args_list[0]
        assert merge_args.kwargs["emails"] == ["me@gmail.com"]

    def test_idempotent_second_run_returns_zero(self):
        """A second run finds all SAME_AS edges already present."""
        writer, session = _make_writer_with_responses(
            [
                {"names": ["Me"]},
                None,
                {"cnt": 0},  # pairwise — no new edges
                {"cnt": 0},  # closure — no new edges
            ]
        )
        n = writer.reconcile_self_person(MeConfig(emails=["me@a.com", "me@b.com"]))
        assert n == 0

    def test_self_transitive_closure_after_late_addition(self):
        """Closure links a legacy is_self=true Person no longer in [me].emails.

        Scenario: an earlier run with ``[me]=[A, B]`` flagged both as
        is_self=true and linked them with SAME_AS.  The current config
        is ``[me]=[B, C]`` (A dropped, C added).  Pairwise creates B↔C
        but A↔C is missing — closure walks the SAME_AS chain through
        is_self=true intermediates and adds it.
        """
        writer, session = _make_writer_with_responses(
            [
                {"names": ["Me", "Me"]},  # phase 1: B, C names
                None,  # phase 2: display_name on [B, C]
                {"cnt": 1},  # pairwise: B↔C added
                {"cnt": 1},  # closure: A↔C added via A↔B↔C
                {"names": ["Me", "Me", "Me Old"]},  # phase 5 collect (graph-wide)
                None,  # phase 6: display_name graph-wide
            ]
        )
        n = writer.reconcile_self_person(MeConfig(emails=["me@b.com", "me@c.com"]))
        # Returned count is pairwise + closure.
        assert n == 2
        assert session.run.call_count == 6

        # The closure pass ran with is_self=true scoping on endpoints
        # AND traversal — protecting against false bridging via
        # cross-source SAME_AS to non-self Persons.
        closure_cypher = session.run.call_args_list[3][0][0]
        assert "is_self: true" in closure_cypher
        assert "SAME_AS*2..4" in closure_cypher
        assert "ALL(node IN nodes(path) WHERE node.is_self = true)" in closure_cypher
        assert "match_type = 'self_identity'" in closure_cypher

        # Phase 6 applies the longest survivor name graph-wide so legacy
        # is_self=true Persons get the canonical display_name.
        graph_wide_set = session.run.call_args_list[5]
        assert "MATCH (p:Person {is_self: true})" in graph_wide_set[0][0]
        assert "SET p.display_name = $chosen" in graph_wide_set[0][0]
        assert graph_wide_set.kwargs["chosen"] == "Me Old"

    def test_self_survivor_unchanged_after_closure(self):
        """``me_cfg.name`` survivor stays canonical before AND after closure.

        When closure adds new edges, the post-closure survivor re-run
        applies the SAME chosen display_name graph-wide (rather than
        electing a new survivor based on the wider name set).
        """
        writer, session = _make_writer_with_responses(
            [
                {"names": ["Me One", "Me Two"]},  # phase 1
                None,  # phase 2: display_name on [me].emails
                {"cnt": 1},  # pairwise
                {"cnt": 1},  # closure adds an edge
                None,  # phase 6: display_name graph-wide (no collect because me_cfg.name set)
            ]
        )
        n = writer.reconcile_self_person(
            MeConfig(
                emails=["me@a.com", "me@b.com"],
                name="Canonical Me",
            )
        )
        assert n == 2
        # me_cfg.name short-circuits the graph-wide name collection,
        # so phase 5 collect query is skipped.
        assert session.run.call_count == 5

        # Initial display_name uses the explicit override.
        initial_set = session.run.call_args_list[1]
        assert initial_set.kwargs["chosen"] == "Canonical Me"

        # Post-closure graph-wide set keeps the SAME survivor name —
        # the override wins over any longer name discovered via closure.
        post_closure_set = session.run.call_args_list[4]
        assert "MATCH (p:Person {is_self: true})" in post_closure_set[0][0]
        assert post_closure_set.kwargs["chosen"] == "Canonical Me"

    def test_self_closure_with_cross_source_same_as(self):
        """Closure does NOT bridge self ↔ non-self via cross-source SAME_AS.

        A self-Person ``A`` may have a SAME_AS edge to non-self ``X``
        from a cross-source merge (e.g., Slack→Gmail by email).  The
        self-scoped closure must:
          - only create edges between is_self=true endpoints;
          - only traverse paths whose intermediate nodes are also
            is_self=true.

        This test inspects the closure cypher to verify both
        constraints are encoded so that A↔non-self edges are never
        synthesised.
        """
        writer, session = _make_writer_with_responses(
            [
                {"names": ["Me", "Me"]},
                None,
                {"cnt": 1},  # pairwise
                {"cnt": 0},  # closure — no new edges (X not is_self)
            ]
        )
        writer.reconcile_self_person(MeConfig(emails=["me@a.com", "me@b.com"]))

        closure_cypher = session.run.call_args_list[3][0][0]

        # Endpoints must both be is_self=true.
        assert "(a:Person {is_self: true})" in closure_cypher
        assert "(b:Person {is_self: true})" in closure_cypher

        # Intermediate nodes on the SAME_AS path must also be is_self=true.
        assert "ALL(node IN nodes(path) WHERE node.is_self = true)" in closure_cypher

        # The MERGE creates SAME_AS edges only — no new is_self flag is
        # propagated, so non-self Persons are never marked self.
        assert "MERGE (a)-[r:SAME_AS]->(b)" in closure_cypher
        assert "is_self = true" not in closure_cypher.split("MERGE")[1]


# ---------------------------------------------------------------------------
# Pipeline wiring — me_config is None vs set
# ---------------------------------------------------------------------------


def _doc(**overrides) -> ParsedDocument:
    defaults = dict(
        source_type="file",
        source_id="notes/test.md",
        operation="created",
        text="Hello world.",
        node_label="File",
        node_props={"name": "test.md", "path": "notes/test.md"},
    )
    defaults.update(overrides)
    return ParsedDocument(**defaults)


def _make_pipeline(me_config: MeConfig | None) -> tuple[Pipeline, MagicMock]:
    registry = MagicMock()
    writer = MagicMock(spec=Writer)
    writer.indexed_source_ids.return_value = set()
    writer.get_content_hashes.return_value = {}
    writer.fetch_existing_entities.return_value = []
    writer.fetch_candidate_entities.return_value = []
    pipeline = Pipeline(registry=registry, writer=writer, me_config=me_config)
    return pipeline, writer


class TestPipelineWiring:
    def test_me_absent_skips_self_reconcile(self):
        """``cfg.me is None`` → ``reconcile_self_person`` is never called."""
        pipeline, writer = _make_pipeline(me_config=None)
        with patch.object(pipeline, "process"):
            pipeline.process_batch([_doc()])
        writer.reconcile_self_person.assert_not_called()

    def test_me_present_calls_self_reconcile_at_end(self):
        """``cfg.me`` set → call comes after the existing reconcile chain."""
        me = MeConfig(emails=["me@a.com", "me@b.com"], name="Me")
        pipeline, writer = _make_pipeline(me_config=me)
        with patch.object(pipeline, "process"):
            pipeline.process_batch([_doc()])
        writer.reconcile_self_person.assert_called_once_with(me)

        # Ensure self-identity runs strictly after the chain — i.e. after
        # the transitive closure step.  Compare call ordering on the writer.
        method_order = [
            c[0]
            for c in writer.method_calls
            if c[0].startswith("reconcile")
            or c[0].startswith("close")
            or c[0].startswith("bridge")
            or c[0].startswith("resolve")
        ]
        assert method_order.index("close_same_as_transitive") < method_order.index(
            "reconcile_self_person"
        )

    def test_self_reconcile_failure_is_swallowed(self):
        """Self-identity errors must not break batch processing."""
        me = MeConfig(emails=["me@a.com"])
        pipeline, writer = _make_pipeline(me_config=me)
        writer.reconcile_self_person.side_effect = RuntimeError("neo4j down")
        with patch.object(pipeline, "process"):
            failed = pipeline.process_batch([_doc()])
        # The batch itself succeeded; self-reconcile failure is logged.
        assert failed == []


# ---------------------------------------------------------------------------
# Email canonicalisation through the config layer
# ---------------------------------------------------------------------------


class TestMeEmailCanonicalization:
    """End-to-end check: ``[me]`` parsing canonicalises ``@googlemail.com``.

    This guards the behaviour the writer relies on (it does NOT re-run
    canonicalisation; it trusts the config loader).
    """

    def test_googlemail_collapses_to_gmail(self):
        from worker.config import _parse_me_config

        me = _parse_me_config({"emails": ["me@googlemail.com", "me@gmail.com"]})
        assert me.emails == ["me@gmail.com", "me@gmail.com"]

    def test_me_block_with_plus_tags_dedupes_to_gmail(self):
        """``me@gmail.com`` and ``me+inbox@gmail.com`` collapse to the same
        canonical mailbox so the writer reconciles them onto a single Person.
        """
        from worker.config import _parse_me_config

        me = _parse_me_config({"emails": ["me@gmail.com", "me+inbox@gmail.com"]})
        assert me.emails == ["me@gmail.com", "me@gmail.com"]

    def test_me_block_preserves_plus_tag_on_non_gmail(self):
        """Plus-addressing is preserved on non-Google providers — they have
        their own mailbox semantics and may treat ``+`` as literal."""
        from worker.config import _parse_me_config

        me = _parse_me_config({"emails": ["me@personal.com", "me+inbox@personal.com"]})
        assert me.emails == ["me@personal.com", "me+inbox@personal.com"]
