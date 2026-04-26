"""Cross-source Person merging: Slack <-> Gmail/Calendar/Obsidian parity.

These tests verify that Slack-emitted Person nodes merge cleanly with
Person nodes from other source types via the existing entity resolution
chain, and that the new ``reconcile_persons_by_slack_user`` step closes
the no-email-fallback gap (slack-user-keyed Person upgraded to
email-keyed Person via SAME_AS once email becomes known).

The reconcile pass itself is exercised against a mocked Neo4j driver —
we only need to verify the Cypher dispatch, not Neo4j semantics.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from worker.parsers.gmail import GmailParser
from worker.parsers.slack import SlackParser


# ---------------------------------------------------------------------------
# Test 1 — Email-based parity: Gmail @gmail.com + Slack @googlemail.com
# resolve to the same Person node id.
# ---------------------------------------------------------------------------


def _gmail_event_from_alice() -> dict:
    return {
        "source_type": "gmail",
        "source_id": "gmail:msg-1",
        "operation": "created",
        "text": "Hello from Alice",
        "mime_type": "message/rfc822",
        "meta": {
            "message_id": "msg-1",
            "thread_id": "t-1",
            "subject": "Hi from Gmail",
            "sender_email": "Alice <alice@gmail.com>",
            "recipients": ["bob@example.com"],
            "date": "2026-04-26T10:00:00Z",
        },
    }


def _slack_event_from_alice_googlemail() -> dict:
    """A Slack burst-window where the author's profile email uses the
    Gmail alias domain ``@googlemail.com``."""
    msgs = [{"ts": "1700000000.000000", "user": "U1", "text": "hi from slack"}]
    return {
        "id": "ev-alice",
        "source_type": "slack",
        "source_id": "slack://T1/C1/window/1700000000.000000",
        "operation": "created",
        "text": "",
        "mime_type": "text/plain",
        "meta": {
            "team_id": "T1",
            "channel_id": "C1",
            "channel_name": "general",
            "is_im": False,
            "is_mpim": False,
            "is_private": False,
            "is_archived": False,
            "kind": "window",
            "first_ts": msgs[0]["ts"],
            "last_ts": msgs[-1]["ts"],
            "messages": msgs,
            "users_info": {
                "U1": {
                    "id": "U1",
                    "name": "alice",
                    "real_name": "Alice Example",
                    "profile": {
                        "real_name": "Alice Example",
                        "display_name": "alice",
                        "email": "alice@googlemail.com",
                    },
                },
            },
        },
    }


def test_gmail_and_slack_share_canonical_person_id() -> None:
    """A Gmail Person hint with @gmail.com and a Slack Person hint with
    @googlemail.com must point to the same Person node id, so the
    existing Step-1 email reconcile collapses them in Neo4j."""
    [gmail_doc] = GmailParser().parse(_gmail_event_from_alice())
    [slack_doc] = SlackParser().parse(_slack_event_from_alice_googlemail())

    def _person_node_ids(doc, email: str) -> set[str]:
        """Person nodes appear as either subject or object in graph hints."""
        ids: set[str] = set()
        for h in doc.graph_hints:
            if h.subject_label == "Person" and h.subject_props.get("email") == email:
                ids.add(h.subject_id)
            if h.object_label == "Person" and h.object_props.get("email") == email:
                ids.add(h.object_id)
        return ids

    gmail_person_ids = _person_node_ids(gmail_doc, "alice@gmail.com")
    slack_person_ids = _person_node_ids(slack_doc, "alice@gmail.com")

    assert gmail_person_ids == {"person:alice@gmail.com"}
    assert slack_person_ids == {"person:alice@gmail.com"}
    # Both use the email merge key — Neo4j will MERGE on (Person {email}).
    for h in gmail_doc.graph_hints + slack_doc.graph_hints:
        if (
            h.subject_label == "Person"
            and h.subject_props.get("email") == "alice@gmail.com"
        ):
            assert h.subject_merge_key == "email"
        if (
            h.object_label == "Person"
            and h.object_props.get("email") == "alice@gmail.com"
        ):
            assert h.object_merge_key == "email"


def test_slack_email_keyed_person_carries_slack_identity() -> None:
    """The email-keyed Slack Person must also expose ``slack_user_id`` +
    ``team_id`` so the new reconcile step can link it to any earlier
    slack-user-keyed Person for the same human."""
    [slack_doc] = SlackParser().parse(_slack_event_from_alice_googlemail())

    [hint] = [
        h
        for h in slack_doc.graph_hints
        if h.predicate == "SENT_BY" and h.object_label == "Person"
    ]
    assert hint.object_id == "person:alice@gmail.com"
    assert hint.object_merge_key == "email"
    assert hint.object_props["slack_user_id"] == "U1"
    assert hint.object_props["team_id"] == "T1"


# ---------------------------------------------------------------------------
# Test 2 — No-email fallback upgrade: a slack-user-keyed Person and an
# email-keyed Person sharing the same (team_id, slack_user_id) get linked
# by SAME_AS during reconcile.
# ---------------------------------------------------------------------------


def _slack_event_user_without_email() -> dict:
    """First sighting: user U1 has no profile email."""
    msgs = [{"ts": "1699999000.000000", "user": "U1", "text": "before email known"}]
    return {
        "id": "ev-no-email",
        "source_type": "slack",
        "source_id": "slack://T1/C1/window/1699999000.000000",
        "operation": "created",
        "text": "",
        "mime_type": "text/plain",
        "meta": {
            "team_id": "T1",
            "channel_id": "C1",
            "channel_name": "general",
            "is_im": False,
            "is_mpim": False,
            "is_private": False,
            "is_archived": False,
            "kind": "window",
            "first_ts": msgs[0]["ts"],
            "last_ts": msgs[-1]["ts"],
            "messages": msgs,
            "users_info": {
                "U1": {
                    "id": "U1",
                    "name": "alice",
                    "real_name": "Alice Example",
                    # no profile email yet
                    "profile": {},
                },
            },
        },
    }


def test_no_email_fallback_then_email_known_emits_two_distinct_person_hints() -> None:
    """Sanity check on the parser side of the gap: the first sighting
    yields a slack-user-keyed Person, the later one (after email becomes
    known) yields an email-keyed Person. The reconcile step is what
    bridges them; the parser only has to expose the slack identity on
    both nodes."""
    [first] = SlackParser().parse(_slack_event_user_without_email())
    [second] = SlackParser().parse(_slack_event_from_alice_googlemail())

    [first_person] = [
        h
        for h in first.graph_hints
        if h.predicate == "SENT_BY" and h.object_label == "Person"
    ]
    [second_person] = [
        h
        for h in second.graph_hints
        if h.predicate == "SENT_BY" and h.object_label == "Person"
    ]

    # Without email: keyed on slack-user; carries (slack_user_id, team_id).
    assert first_person.object_merge_key == "source_id"
    assert first_person.object_id == "slack-user:T1/U1"
    assert first_person.object_props["slack_user_id"] == "U1"
    assert first_person.object_props["team_id"] == "T1"

    # With email: keyed on email; ALSO carries (slack_user_id, team_id).
    assert second_person.object_merge_key == "email"
    assert second_person.object_id == "person:alice@gmail.com"
    assert second_person.object_props["slack_user_id"] == "U1"
    assert second_person.object_props["team_id"] == "T1"

    # The two hints on their own do not link the nodes — the reconcile
    # step is what produces the SAME_AS edge.
    assert first_person.object_id != second_person.object_id


def _make_writer_with_mock_session():
    """Build a Writer instance whose Neo4j session is a MagicMock."""
    from worker.pipeline.writer import Writer

    writer = object.__new__(Writer)
    writer._neo4j_driver = MagicMock()

    mock_session = MagicMock()
    writer._neo4j_driver.session.return_value.__enter__.return_value = mock_session
    writer._neo4j_driver.session.return_value.__exit__.return_value = False
    return writer, mock_session


def test_reconcile_persons_by_slack_user_runs_expected_cypher() -> None:
    """The reconcile step issues a single Cypher query that groups Person
    nodes by (team_id, slack_user_id) and creates SAME_AS edges between
    members of each group, tagged ``match_type='slack_user_id'``."""
    writer, session = _make_writer_with_mock_session()

    mock_result = MagicMock()
    mock_result.single.return_value = {"cnt": 1}
    session.run.return_value = mock_result

    created = writer._reconcile_persons_by_slack_user_neo4j()
    assert created == 1
    assert session.run.call_count == 1

    cypher = session.run.call_args.args[0]
    # The query groups by (team_id, slack_user_id) and creates SAME_AS.
    assert "slack_user_id" in cypher
    assert "team_id" in cypher
    assert "SAME_AS" in cypher
    # And it tags the edge so the test in the bead's acceptance criteria
    # ("source='slack' and source='gmail'") can assert the link is the
    # one created by this step.
    assert "match_type" in cypher
    assert "slack_user_id" in cypher  # match_type value


def test_reconcile_persons_by_slack_user_zero_matches_returns_zero() -> None:
    writer, session = _make_writer_with_mock_session()
    mock_result = MagicMock()
    mock_result.single.return_value = {"cnt": 0}
    session.run.return_value = mock_result

    assert writer._reconcile_persons_by_slack_user_neo4j() == 0


# ---------------------------------------------------------------------------
# Test 3 — The pipeline wires the new step into the reconcile chain.
# ---------------------------------------------------------------------------


def test_pipeline_invokes_reconcile_persons_by_slack_user_after_email_step() -> None:
    """The pipeline must call ``reconcile_persons_by_slack_user`` after
    the email-based ``reconcile_persons`` so that any Person node lifted
    to an email key in this batch is available for the slack-identity
    pass."""
    import inspect

    from worker.pipeline import Pipeline

    src = inspect.getsource(Pipeline.process_batch)
    email_idx = src.index("reconcile_persons(")
    slack_idx = src.index("reconcile_persons_by_slack_user(")
    name_idx = src.index("reconcile_persons_by_name(")

    assert email_idx < slack_idx < name_idx, (
        "reconcile_persons_by_slack_user must run after reconcile_persons "
        "and before reconcile_persons_by_name"
    )
