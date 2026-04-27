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


def test_parser_defensive_filter() -> None:
    """An empty bot_message that slips past the source-side filter
    (e.g., a queue entry written before fn-dge shipped) must still be
    stripped from the rendered text.  The parser keeps a backstop for
    forward-compat / replay safety."""
    real = _msg(1700000000.0, user="U1", text="hello world")
    empty_bot = {
        "ts": "1700000005.000000",
        "user": "",
        "bot_id": "B0",
        "text": "",
        "subtype": "bot_message",
    }
    join = _msg(1700000010.0, user="U2", text="", subtype="channel_join")
    real2 = _msg(1700000020.0, user="U2", text="goodbye world")
    event = _window_event(messages=[real, empty_bot, join, real2])

    [doc] = SlackParser().parse(event)
    lines = [ln for ln in doc.text.split("\n") if ln.strip()]
    # Only the two real messages render — the empty bot_message and
    # channel_join are stripped by the defensive filter.
    assert len(lines) == 2
    assert "hello world" in doc.text
    assert "goodbye world" in doc.text
    assert "B0" not in doc.text
    assert "channel_join" not in doc.text


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


# ---------------------------------------------------------------------------
# Attachments (fn-bu3)
# ---------------------------------------------------------------------------


_TEAM_DOMAIN = "acme"


def _attachment(
    *,
    file_id: str,
    name: str,
    mime: str,
    size: int,
    ts: str,
    user: str = "U1",
    url: str = "https://files.slack.com/private/url",
    filetype: str = "",
) -> dict[str, Any]:
    return {
        "id": file_id,
        "name": name,
        "mimetype": mime,
        "filetype": filetype or mime.split("/")[-1],
        "size": size,
        "url_private_download": url,
        "ts": ts,
        "user": user,
    }


def _attach_meta(
    event: dict[str, Any],
    *,
    attachments: list[dict[str, Any]],
    download_attachments: bool = False,
    indexable: list[str] | None = None,
    max_size_mb: int = 25,
    team_domain: str = _TEAM_DOMAIN,
) -> dict[str, Any]:
    """Augment a window/thread event with attachment-related meta."""
    event = dict(event)
    meta = dict(event["meta"])
    meta["attachments"] = attachments
    meta["download_attachments"] = download_attachments
    meta["attachment_indexable_mimetypes"] = (
        indexable if indexable is not None else ["application/pdf", "text/plain"]
    )
    meta["attachment_max_size_mb"] = max_size_mb
    meta["team_domain"] = team_domain
    event["meta"] = meta
    return event


class TestAttachmentRenderMarkers:
    def test_window_inserts_file_marker_under_owning_message(self) -> None:
        m1 = _msg(1700000000.0, user="U1", text="hello")
        m2 = _msg(1700000060.0, user="U2", text="see attached")
        m3 = _msg(1700000120.0, user="U1", text="bye")
        event = _attach_meta(
            _window_event(messages=[m1, m2, m3]),
            attachments=[
                _attachment(
                    file_id="F1",
                    name="report.pdf",
                    mime="application/pdf",
                    size=1024,
                    ts=m2["ts"],
                ),
            ],
        )
        [parent, _attachment_doc] = SlackParser().parse(event)
        lines = parent.text.split("\n")
        assert len(lines) == 4
        # Marker appears immediately after m2's render line.
        assert lines[2].startswith("[file] report.pdf")
        assert "application/pdf" in lines[2]
        assert lines[3].startswith("[")  # m3's normal line follows

    def test_filename_with_unicode_separator_does_not_inject_lines(self) -> None:
        """A name containing U+2028 / U+2029 must not break the [file]
        marker into multiple lines (Slack's whole-message-overlap chunker
        is sensitive to boundary corruption)."""
        m1 = _msg(1700000000.0, user="U1", text="hello")
        m2 = _msg(1700000060.0, user="U2", text="see attached")
        evil = "ok.pdf injected line"  # U+2028 LINE SEPARATOR
        event = _attach_meta(
            _window_event(messages=[m1, m2]),
            attachments=[
                _attachment(
                    file_id="Fevil",
                    name=evil,
                    mime="application/pdf",
                    size=1024,
                    ts=m2["ts"],
                ),
            ],
        )
        docs = SlackParser().parse(event)
        parent = docs[0]
        marker_lines = [ln for ln in parent.text.split("\n") if "[file]" in ln]
        assert len(marker_lines) == 1
        assert " " not in parent.text
        # The Attachment Document keeps the original (unsanitized) name.
        att = next(d for d in docs if d.node_label == "Attachment")
        assert att.node_props["name"] == evil

    def test_filename_with_newline_does_not_break_chunk_boundaries(self) -> None:
        m = _msg(1700000000.0, user="U1", text="see file")
        evil = "ok.pdf\n[file] secret.pdf (application/pdf, 1.0 KB)"
        event = _attach_meta(
            _window_event(messages=[m]),
            attachments=[
                _attachment(
                    file_id="Fevil",
                    name=evil,
                    mime="application/pdf",
                    size=1024,
                    ts=m["ts"],
                ),
            ],
        )
        docs = SlackParser().parse(event)
        parent = docs[0]
        # Exactly one [file] marker line — no injected second marker.
        marker_lines = [ln for ln in parent.text.split("\n") if "[file]" in ln]
        assert len(marker_lines) == 1

    def test_thread_reply_marker_is_indented(self) -> None:
        parent = _msg(1700000000.0, user="U1", text="kickoff")
        reply = _msg(1700000060.0, user="U2", text="see this", thread_ts=1700000000.0)
        event = _attach_meta(
            _thread_event(parent=parent, replies=[reply]),
            attachments=[
                _attachment(
                    file_id="F2",
                    name="diagram.png",
                    mime="image/png",
                    size=2048,
                    ts=reply["ts"],
                    user="U2",
                )
            ],
        )
        [doc, _att] = SlackParser().parse(event)
        marker_lines = [ln for ln in doc.text.split("\n") if "[file]" in ln]
        assert len(marker_lines) == 1
        assert marker_lines[0].startswith("  [file] diagram.png")


class TestAttachmentDocsMetadataOnly:
    """download_attachments=False → 2 metadata-only Documents, markers in text."""

    def test_two_files_emit_two_metadata_only_docs(self) -> None:
        m = _msg(1700000000.0, user="U1", text="see files")
        attachments = [
            _attachment(
                file_id="Fpdf",
                name="report.pdf",
                mime="application/pdf",
                size=4096,
                ts=m["ts"],
            ),
            _attachment(
                file_id="Fdoc",
                name="notes.docx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "wordprocessingml.document"
                ),
                size=8192,
                ts=m["ts"],
            ),
        ]
        event = _attach_meta(
            _window_event(messages=[m]),
            attachments=attachments,
            download_attachments=False,
        )
        docs = SlackParser().parse(event)
        # 1 message doc + 2 attachment docs.
        assert len(docs) == 3
        att_docs = [d for d in docs if d.node_label == "Attachment"]
        assert len(att_docs) == 2
        for d in att_docs:
            assert d.text == ""
            assert d.node_props["indexed"] is False

        # Render carries [file] markers for both files.
        msg_text = docs[0].text
        assert "[file] report.pdf" in msg_text
        assert "[file] notes.docx" in msg_text


class TestAttachmentDocsDownloadAndIndex:
    """download_attachments=True with PDF allowlist: PDF fetched, .docx skipped."""

    def test_pdf_only_indexable(self) -> None:
        m = _msg(1700000000.0, user="U1", text="files attached")
        attachments = [
            _attachment(
                file_id="Fpdf",
                name="report.pdf",
                mime="application/pdf",
                size=4096,
                ts=m["ts"],
                url="https://files.slack.com/pdf",
            ),
            _attachment(
                file_id="Fdoc",
                name="notes.docx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "wordprocessingml.document"
                ),
                size=8192,
                ts=m["ts"],
            ),
        ]
        event = _attach_meta(
            _window_event(messages=[m]),
            attachments=attachments,
            download_attachments=True,
            indexable=["application/pdf"],
        )

        # Build a single-page PDF in memory so pymupdf can extract text.
        import pymupdf

        pdf = pymupdf.open()
        page = pdf.new_page()
        page.insert_text((72, 72), "indexed slack pdf")
        pdf_bytes = pdf.tobytes()
        pdf.close()

        fetched_urls: list[str] = []

        def fake_fetch(url: str) -> bytes:
            fetched_urls.append(url)
            return pdf_bytes

        parser = SlackParser()
        parser._fetcher = fake_fetch  # type: ignore[assignment]
        docs = parser.parse(event)

        att_docs = [d for d in docs if d.node_label == "Attachment"]
        assert len(att_docs) == 2

        pdf_doc = next(d for d in att_docs if d.source_id.endswith("/file/Fpdf"))
        docx_doc = next(d for d in att_docs if d.source_id.endswith("/file/Fdoc"))

        # PDF was fetched + parsed.
        assert "indexed slack pdf" in pdf_doc.text
        assert pdf_doc.node_props["indexed"] is True
        assert fetched_urls == ["https://files.slack.com/pdf"]

        # .docx (not in indexable allowlist) → metadata-only, not fetched.
        assert docx_doc.text == ""
        assert docx_doc.node_props["indexed"] is False


class TestAttachmentEdges:
    def test_attached_to_points_at_parent_doc(self) -> None:
        m = _msg(1700000000.0, user="U1", text="here")
        att = _attachment(
            file_id="F1",
            name="x.pdf",
            mime="application/pdf",
            size=1024,
            ts=m["ts"],
        )
        event = _attach_meta(_window_event(messages=[m]), attachments=[att])
        [parent_doc, att_doc] = SlackParser().parse(event)

        attached = [h for h in att_doc.graph_hints if h.predicate == "ATTACHED_TO"]
        assert len(attached) == 1
        assert attached[0].subject_id == att_doc.source_id
        assert attached[0].object_id == parent_doc.source_id
        assert attached[0].object_label == "SlackMessage"

    def test_in_channel_edge_emitted(self) -> None:
        m = _msg(1700000000.0, user="U1", text="here")
        att = _attachment(
            file_id="F1",
            name="x.pdf",
            mime="application/pdf",
            size=1024,
            ts=m["ts"],
        )
        event = _attach_meta(_window_event(messages=[m]), attachments=[att])
        [_parent, att_doc] = SlackParser().parse(event)

        in_chan = [h for h in att_doc.graph_hints if h.predicate == "IN_CHANNEL"]
        assert len(in_chan) == 1
        assert in_chan[0].object_id == f"slack-channel:{_TEAM}/{_CHANNEL_ID}"

    def test_attachment_doc_id_format(self) -> None:
        m = _msg(1700000000.0, user="U1", text="here")
        att = _attachment(
            file_id="F1",
            name="x.pdf",
            mime="application/pdf",
            size=1024,
            ts=m["ts"],
        )
        event = _attach_meta(_window_event(messages=[m]), attachments=[att])
        [_parent, att_doc] = SlackParser().parse(event)
        assert att_doc.source_id == (f"slack://{_TEAM}/{_CHANNEL_ID}/{m['ts']}/file/F1")

    def test_parent_url_uses_team_domain(self) -> None:
        m = _msg(1700000000.0, user="U1", text="here")
        att = _attachment(
            file_id="F1",
            name="x.pdf",
            mime="application/pdf",
            size=1024,
            ts=m["ts"],
        )
        event = _attach_meta(
            _window_event(messages=[m]),
            attachments=[att],
            team_domain="acme",
        )
        [_parent, att_doc] = SlackParser().parse(event)
        # build_parent_url('slack', ...) drops the dot in ts when forming
        # the permalink.  Its exact format is exercised by attachments util
        # tests; here we only confirm it landed on the node.
        assert "acme.slack.com" in att_doc.node_props["parent_url"]


class TestAttachmentBotUploaderFallback:
    """A bot uploader without an email yields the slack-user fallback Person."""

    def test_uploader_without_email_falls_back(self) -> None:
        users = _users_info(
            B1={
                "id": "B1",
                "name": "deploy-bot",
                "real_name": "Deploy Bot",
                "profile": {"display_name": "deploy-bot", "real_name": "Deploy Bot"},
                # no email
            }
        )
        m = _msg(1700000000.0, user="U1", text="bot uploaded a thing")
        att = _attachment(
            file_id="F1",
            name="release.pdf",
            mime="application/pdf",
            size=1024,
            ts=m["ts"],
            user="B1",  # uploader is the bot, not the message author
        )
        event = _attach_meta(
            _window_event(messages=[m], users_info=users),
            attachments=[att],
        )
        [_parent, att_doc] = SlackParser().parse(event)

        sent_by = [
            h
            for h in att_doc.graph_hints
            if h.predicate == "SENT_BY" and h.subject_id == att_doc.source_id
        ]
        assert len(sent_by) == 1
        h = sent_by[0]
        assert h.object_id == f"slack-user:{_TEAM}/B1"
        assert h.object_merge_key == "source_id"
        assert h.subject_label == "Attachment"


class TestAttachmentFailedFetchFallsBack:
    """A fetch error downgrades to metadata-only, never crashes the parent."""

    def test_failed_fetch_downgrades(self) -> None:
        m = _msg(1700000000.0, user="U1", text="here")
        att = _attachment(
            file_id="F1",
            name="report.pdf",
            mime="application/pdf",
            size=1024,
            ts=m["ts"],
        )
        event = _attach_meta(
            _window_event(messages=[m]),
            attachments=[att],
            download_attachments=True,
            indexable=["application/pdf"],
        )

        def boom(url: str) -> bytes:
            raise RuntimeError("network down")

        parser = SlackParser()
        parser._fetcher = boom  # type: ignore[assignment]
        docs = parser.parse(event)

        att_doc = next(d for d in docs if d.node_label == "Attachment")
        assert att_doc.text == ""
        assert att_doc.node_props["indexed"] is False


class TestAttachmentSnippetIndexedAsText:
    """Slack 'snippet' files are text/plain — indexed verbatim."""

    def test_snippet_full_text_indexed(self) -> None:
        m = _msg(1700000000.0, user="U1", text="snippet attached")
        att = _attachment(
            file_id="Fsnip",
            name="pasted.txt",
            mime="text/plain",
            size=42,
            ts=m["ts"],
            filetype="text",
        )
        event = _attach_meta(
            _window_event(messages=[m]),
            attachments=[att],
            download_attachments=True,
            indexable=["text/plain"],
        )

        snippet_body = b"line one\nline two\nline three\n"

        def fake_fetch(url: str) -> bytes:
            return snippet_body

        parser = SlackParser()
        parser._fetcher = fake_fetch  # type: ignore[assignment]
        docs = parser.parse(event)
        att_doc = next(d for d in docs if d.node_label == "Attachment")
        assert att_doc.text == snippet_body.decode("utf-8")
        assert att_doc.node_props["indexed"] is True


class TestThreadAttachmentParentingMatchesChunkOwner:
    """Threaded reply file → ATTACHED_TO points at the same Document the
    chunks are written under (the thread Document, not the reply)."""

    def test_thread_reply_file_attaches_to_thread_doc(self) -> None:
        parent = _msg(1700000000.0, user="U1", text="kickoff")
        reply = _msg(1700000060.0, user="U2", text="here", thread_ts=1700000000.0)
        att = _attachment(
            file_id="F1",
            name="diagram.png",
            mime="image/png",
            size=4096,
            ts=reply["ts"],
            user="U2",
        )
        event = _attach_meta(
            _thread_event(parent=parent, replies=[reply]),
            attachments=[att],
        )
        docs = SlackParser().parse(event)
        thread_doc = docs[0]
        att_doc = docs[1]
        assert thread_doc.node_label == "SlackMessage"
        assert thread_doc.node_props["has_thread"] is True

        attached = [h for h in att_doc.graph_hints if h.predicate == "ATTACHED_TO"]
        assert len(attached) == 1
        # ATTACHED_TO points at the thread Document, not the per-reply ts.
        assert attached[0].object_id == thread_doc.source_id


class TestConfigureSlackParser:
    def test_configure_sets_module_default_fetcher(self) -> None:
        from worker.parsers.slack import SlackParser, configure_slack_parser

        sentinel = lambda url: b"ok"  # noqa: E731
        try:
            configure_slack_parser(fetcher=sentinel)
            # Newly constructed instances pick up the module default.
            assert SlackParser()._fetcher is sentinel
        finally:
            # Cleanup so other tests aren't affected.
            configure_slack_parser(fetcher=None)
        assert SlackParser()._fetcher is None
