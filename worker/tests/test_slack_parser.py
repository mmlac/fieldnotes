"""Tests for the Slack thread/window parser.

Covers text rendering, GraphHint emission, cross-source Person merging
via canonical emails, mention resolution (with and without users.info),
and the empty/system-message metadata-only path.  No live Slack calls —
fixtures construct IngestEvent dicts directly.
"""

from __future__ import annotations

from typing import Any

from worker.parsers import slack  # ensure registration runs
from worker.parsers.registry import get as get_parser
from worker.parsers.slack import NODE_LABEL, SlackParser


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


_TEAM = "T1"
_CHANNEL_ID = "C42"
_CHANNEL_NAME = "general"


def _users_info(**overrides: dict[str, Any]) -> dict[str, Any]:
    """Standard users.info-style lookup used across most tests."""
    base: dict[str, Any] = {
        "U1": {
            "id": "U1",
            "name": "alice",
            "real_name": "Alice Smith",
            "profile": {
                "real_name": "Alice Smith",
                "display_name": "alice",
                "email": "alice@example.com",
            },
        },
        "U2": {
            "id": "U2",
            "name": "bob",
            "real_name": "Bob Jones",
            "profile": {
                "real_name": "Bob Jones",
                "display_name": "bob",
                "email": "bob@googlemail.com",  # canonicalises to gmail.com
            },
        },
    }
    base.update(overrides)
    return base


def _msg(
    ts: float,
    user: str = "U1",
    text: str = "hello",
    *,
    thread_ts: float | None = None,
    subtype: str | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "ts": f"{ts:.6f}",
        "user": user,
        "text": text,
    }
    if thread_ts is not None:
        out["thread_ts"] = f"{thread_ts:.6f}"
    if subtype is not None:
        out["subtype"] = subtype
    return out


def _thread_event(
    *,
    parent: dict[str, Any],
    replies: list[dict[str, Any]],
    users_info: dict[str, Any] | None = None,
    channel_name: str = _CHANNEL_NAME,
) -> dict[str, Any]:
    msgs = [parent, *replies]
    return {
        "id": "ev1",
        "source_type": "slack",
        "source_id": (f"slack://{_TEAM}/{_CHANNEL_ID}/thread/{parent['ts']}"),
        "operation": "created",
        "text": "",  # parser re-renders
        "mime_type": "text/plain",
        "meta": {
            "team_id": _TEAM,
            "channel_id": _CHANNEL_ID,
            "channel_name": channel_name,
            "is_im": False,
            "is_mpim": False,
            "is_private": False,
            "is_archived": False,
            "kind": "thread",
            "parent_ts": parent["ts"],
            "last_ts": msgs[-1]["ts"],
            "message_ts": [m["ts"] for m in msgs],
            "users": sorted({m["user"] for m in msgs if m.get("user")}),
            "reply_count": len(replies),
            "messages": msgs,
            "users_info": users_info if users_info is not None else _users_info(),
        },
    }


def _window_event(
    *,
    messages: list[dict[str, Any]],
    users_info: dict[str, Any] | None = None,
    is_private: bool = False,
    is_im: bool = False,
    is_mpim: bool = False,
    channel_name: str = _CHANNEL_NAME,
) -> dict[str, Any]:
    first_ts = messages[0]["ts"]
    last_ts = messages[-1]["ts"]
    return {
        "id": "ev2",
        "source_type": "slack",
        "source_id": (f"slack://{_TEAM}/{_CHANNEL_ID}/window/{first_ts}-{last_ts}"),
        "operation": "created",
        "text": "",
        "mime_type": "text/plain",
        "meta": {
            "team_id": _TEAM,
            "channel_id": _CHANNEL_ID,
            "channel_name": channel_name,
            "is_im": is_im,
            "is_mpim": is_mpim,
            "is_private": is_private,
            "is_archived": False,
            "kind": "window",
            "first_ts": first_ts,
            "last_ts": last_ts,
            "message_ts": [m["ts"] for m in messages],
            "users": sorted({m["user"] for m in messages if m.get("user")}),
            "message_count": len(messages),
            "messages": messages,
            "users_info": users_info if users_info is not None else _users_info(),
        },
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_parser_registered_under_slack_key() -> None:
    instance = get_parser("slack")
    assert isinstance(instance, SlackParser)


# ---------------------------------------------------------------------------
# Thread render + hints
# ---------------------------------------------------------------------------


def test_thread_render_indents_replies_and_emits_channel_and_authors() -> None:
    parent = _msg(1700000000.0, user="U1", text="kickoff")
    replies = [
        _msg(1700000060.0, user="U2", text="on it", thread_ts=1700000000.0),
        _msg(1700000120.0, user="U1", text="thanks", thread_ts=1700000000.0),
        _msg(1700000180.0, user="U2", text="done", thread_ts=1700000000.0),
    ]
    event = _thread_event(parent=parent, replies=replies)
    [doc] = SlackParser().parse(event)

    # Text format: parent flush, replies indented two spaces.
    lines = doc.text.split("\n")
    assert len(lines) == 4
    assert lines[0].startswith("[")
    assert " Alice Smith (@alice): kickoff" in lines[0]
    assert lines[1].startswith("  [")
    assert "Bob Jones (@bob): on it" in lines[1]
    assert "Alice Smith (@alice): thanks" in lines[2]
    assert "Bob Jones (@bob): done" in lines[3]

    # Document node carries Slack frontmatter the writer needs.
    assert doc.node_label == NODE_LABEL
    assert doc.node_props["team_id"] == _TEAM
    assert doc.node_props["channel_id"] == _CHANNEL_ID
    assert doc.node_props["conversation_type"] == "public"
    assert doc.node_props["has_thread"] is True
    assert doc.node_props["message_count"] == 4
    assert doc.source_metadata["source_type"] == "slack"
    assert doc.source_metadata["conversation_type"] == "public"

    # GraphHints: 1 IN_CHANNEL + 1 SENT_BY per distinct author (U1, U2).
    in_channel = [h for h in doc.graph_hints if h.predicate == "IN_CHANNEL"]
    sent_by = [h for h in doc.graph_hints if h.predicate == "SENT_BY"]
    assert len(in_channel) == 1
    assert in_channel[0].object_id == f"slack-channel:{_TEAM}/{_CHANNEL_ID}"
    assert in_channel[0].object_label == "Channel"
    assert in_channel[0].object_props["type"] == "public"
    assert len(sent_by) == 2
    assert {h.object_id for h in sent_by} == {
        "person:alice@example.com",
        "person:bob@gmail.com",  # googlemail → gmail canonicalisation
    }
    for h in sent_by:
        assert h.object_label == "Person"
        assert h.object_merge_key == "email"


def test_email_canonicalization_matches_gmail_person_node() -> None:
    """A Slack profile with @googlemail.com produces the same person:
    node id as a Gmail message with @gmail.com."""
    msg_a = _msg(1700000000.0, user="U2", text="hi from slack")
    event = _window_event(messages=[msg_a])
    [doc] = SlackParser().parse(event)

    sent_by = [h for h in doc.graph_hints if h.predicate == "SENT_BY"]
    assert len(sent_by) == 1
    assert sent_by[0].object_id == "person:bob@gmail.com"
    assert sent_by[0].object_props["email"] == "bob@gmail.com"
    assert sent_by[0].object_merge_key == "email"


# ---------------------------------------------------------------------------
# Mentions
# ---------------------------------------------------------------------------


def test_mention_of_user_without_email_uses_slack_user_fallback() -> None:
    users = _users_info(
        U999={
            "id": "U999",
            "name": "ghost",
            "real_name": "G. Host",
            "profile": {"display_name": "ghost", "real_name": "G. Host"},
            # no email
        }
    )
    msg = _msg(1700000000.0, user="U1", text="cc <@U999> please")
    event = _window_event(messages=[msg], users_info=users)
    [doc] = SlackParser().parse(event)

    mentions = [h for h in doc.graph_hints if h.predicate == "MENTIONS"]
    user_mention = [h for h in mentions if h.object_label == "Person"]
    assert len(user_mention) == 1
    h = user_mention[0]
    assert h.object_id == f"slack-user:{_TEAM}/U999"
    assert h.object_merge_key == "source_id"
    assert h.object_props["slack_user_id"] == "U999"
    assert h.object_props["name"] == "G. Host"


def test_inline_email_in_body_yields_mentions_edge_to_canonical_person() -> None:
    msg = _msg(
        1700000000.0,
        user="U1",
        text="contact alice@example.com or alice@googlemail.com",
    )
    event = _window_event(messages=[msg])
    [doc] = SlackParser().parse(event)

    mention_targets = {
        h.object_id
        for h in doc.graph_hints
        if h.predicate == "MENTIONS" and h.object_label == "Person"
    }
    # alice@example.com is the author already (SENT_BY) → suppressed
    # from MENTIONS.  alice@googlemail.com canonicalises to gmail.com
    # and becomes a MENTIONS Person node.
    assert "person:alice@gmail.com" in mention_targets
    assert "person:alice@example.com" not in mention_targets


def test_channel_mention_resolves_to_slack_channel_node() -> None:
    msg = _msg(1700000000.0, user="U1", text="see <#C9999|other> for details")
    event = _window_event(messages=[msg])
    [doc] = SlackParser().parse(event)

    channel_mentions = [
        h
        for h in doc.graph_hints
        if h.predicate == "MENTIONS" and h.object_label == "Channel"
    ]
    assert len(channel_mentions) == 1
    assert channel_mentions[0].object_id == f"slack-channel:{_TEAM}/C9999"


# ---------------------------------------------------------------------------
# Burst window: per-author SENT_BY de-dup
# ---------------------------------------------------------------------------


def test_burst_window_emits_one_sent_by_per_distinct_author() -> None:
    msgs = [
        _msg(1700000000.0, user="U1", text="one"),
        _msg(1700000010.0, user="U2", text="two"),
        _msg(1700000020.0, user="U1", text="three"),
        _msg(1700000030.0, user="U2", text="four"),
        _msg(1700000040.0, user="U1", text="five"),
    ]
    event = _window_event(messages=msgs)
    [doc] = SlackParser().parse(event)

    sent_by = [h for h in doc.graph_hints if h.predicate == "SENT_BY"]
    assert len(sent_by) == 2
    assert {h.object_id for h in sent_by} == {
        "person:alice@example.com",
        "person:bob@gmail.com",
    }
    assert doc.node_props["has_thread"] is False
    assert doc.node_props["conversation_type"] == "public"


# ---------------------------------------------------------------------------
# Conversation types
# ---------------------------------------------------------------------------


def test_conversation_type_for_dm_and_private_channel() -> None:
    msg = _msg(1700000000.0, user="U1", text="just us")

    im = SlackParser().parse(_window_event(messages=[msg], is_im=True))[0]
    assert im.node_props["conversation_type"] == "im"
    in_channel = [h for h in im.graph_hints if h.predicate == "IN_CHANNEL"][0]
    assert in_channel.object_props["type"] == "im"

    priv = SlackParser().parse(_window_event(messages=[msg], is_private=True))[0]
    assert priv.node_props["conversation_type"] == "private"

    mp = SlackParser().parse(_window_event(messages=[msg], is_mpim=True))[0]
    assert mp.node_props["conversation_type"] == "mpim"


# ---------------------------------------------------------------------------
# System / empty messages → metadata-only chunk
# ---------------------------------------------------------------------------


def test_system_only_messages_produce_metadata_only_chunk() -> None:
    sys_msgs = [
        _msg(1700000000.0, user="U1", text="", subtype="channel_join"),
        _msg(1700000060.0, user="U2", text="", subtype="channel_leave"),
    ]
    event = _window_event(messages=sys_msgs)
    [doc] = SlackParser().parse(event)

    # Body is short and metadata-shaped, not a normal render.
    assert "system message" in doc.text
    assert doc.node_props["message_count"] == 2
    # Channel hint still emitted so the document anchors in the graph.
    in_channel = [h for h in doc.graph_hints if h.predicate == "IN_CHANNEL"]
    assert len(in_channel) == 1


# ---------------------------------------------------------------------------
# Delete passthrough
# ---------------------------------------------------------------------------


def test_delete_event_returns_deleted_parsed_doc() -> None:
    event = {
        "id": "ev3",
        "source_type": "slack",
        "source_id": f"slack://{_TEAM}/{_CHANNEL_ID}/thread/1700000000.000000",
        "operation": "deleted",
        "text": "",
        "mime_type": "text/plain",
        "meta": {"team_id": _TEAM, "channel_id": _CHANNEL_ID},
    }
    [doc] = SlackParser().parse(event)
    assert doc.operation == "deleted"
    assert doc.text == ""
    assert doc.graph_hints == []


# ---------------------------------------------------------------------------
# Unknown user fallback in render
# ---------------------------------------------------------------------------


def test_unknown_user_renders_as_at_uid_in_text() -> None:
    msg = _msg(1700000000.0, user="U777", text="who am I")
    event = _window_event(messages=[msg], users_info={})
    [doc] = SlackParser().parse(event)
    assert "@U777" in doc.text


# Sanity: the slack module imports the canonicaliser, not its own copy.
def test_canonicalize_is_shared_with_base() -> None:
    from worker.parsers import base

    assert slack.canonicalize_email is base.canonicalize_email
