"""Pytest bootstrap: ensure the tests directory is on sys.path so that
sibling helper modules (e.g. ``_fake_queue``) can be imported by test
files.  This is needed because the tests directory has no
``__init__.py`` and pytest's default import mode does not automatically
add it to ``sys.path``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)


_SLACK_FIXTURES_DIR = Path(_TESTS_DIR) / "fixtures" / "slack"
_GMAIL_FIXTURES_DIR = Path(_TESTS_DIR) / "fixtures" / "gmail"
_CALENDAR_FIXTURES_DIR = Path(_TESTS_DIR) / "fixtures" / "calendar"
_ATTACHMENT_FIXTURES_DIR = Path(_TESTS_DIR) / "fixtures" / "attachments"


@pytest.fixture
def fake_slack_client() -> MagicMock:
    """A canned-fixture-backed Slack ``WebClient`` mock.

    Reads JSON fixtures from ``tests/fixtures/slack/`` and routes the
    relevant ``WebClient`` methods (``conversations_list``,
    ``conversations_history``, ``conversations_replies``, ``users_info``,
    ``auth_test``) per-channel. Tests that need to exercise the Slack
    source against a deterministic fixture corpus depend on this
    fixture instead of constructing a client themselves.
    """
    if not _SLACK_FIXTURES_DIR.is_dir():
        pytest.skip(f"Slack fixtures missing at {_SLACK_FIXTURES_DIR}")

    def _load(name: str) -> dict[str, Any]:
        return json.loads((_SLACK_FIXTURES_DIR / name).read_text())

    history_eng = _load("conversations_history_engineering.json")
    history_dm = _load("conversations_history_dm.json")
    replies_eng = _load("conversations_replies_engineering.json")
    users = _load("users_info.json")
    channels = _load("conversations_list.json")

    client = MagicMock()
    client.auth_test.return_value = {"team_id": "T-TEST", "ok": True}
    client.conversations_list.return_value = channels

    def _history(**kwargs: Any) -> dict[str, Any]:
        cid = kwargs.get("channel", "")
        if cid == "C-ENG":
            return history_eng
        if cid == "D-AB":
            return history_dm
        return {"ok": True, "messages": [], "response_metadata": {"next_cursor": ""}}

    def _replies(**kwargs: Any) -> dict[str, Any]:
        cid = kwargs.get("channel", "")
        if cid == "C-ENG":
            return replies_eng
        return {"ok": True, "messages": [], "response_metadata": {"next_cursor": ""}}

    def _users_info(**kwargs: Any) -> dict[str, Any]:
        uid = kwargs.get("user", "")
        return {"ok": True, "user": users.get(uid, {})}

    client.conversations_history.side_effect = _history
    client.conversations_replies.side_effect = _replies
    client.users_info.side_effect = _users_info
    # The Slack source caches users.list at startup so the parser can
    # resolve <@Uxxx> mentions and message authors to real names + emails
    # without a per-event API call.  The fake routes the canned user
    # directory through users.list shape (members[] + paginated cursor).
    client.users_list.return_value = {
        "ok": True,
        "members": list(users.values()),
        "response_metadata": {"next_cursor": ""},
    }
    return client


def _load_gmail_fixture(account: str) -> dict[str, Any]:
    path = _GMAIL_FIXTURES_DIR / f"account_{account}.json"
    return json.loads(path.read_text())


def _load_calendar_fixture(account: str) -> dict[str, Any]:
    path = _CALENDAR_FIXTURES_DIR / f"account_{account}.json"
    return json.loads(path.read_text())


def make_fake_gmail_messages_api(account: str) -> MagicMock:
    """Build a fake ``service.users().messages()`` API for *account*.

    Backed by ``tests/fixtures/gmail/account_<account>.json``.  Routes the
    ``list``/``get`` calls used by ``GmailSource._backfill`` so the source's
    cursor + per-message paths run against canned data without touching
    the real Gmail API.
    """
    fixture = _load_gmail_fixture(account)
    list_result = fixture["list"]
    messages = fixture["messages"]

    api = MagicMock()

    list_req = MagicMock()
    list_req.execute.return_value = list_result
    api.list.return_value = list_req

    def _get(**kwargs: Any) -> MagicMock:
        msg_id = kwargs["id"]
        req = MagicMock()
        req.execute.return_value = messages[msg_id]
        return req

    api.get.side_effect = _get
    return api


def make_fake_calendar_service(account: str) -> MagicMock:
    """Build a fake Google Calendar ``service`` for *account*.

    The returned mock exposes ``.events().list(...).execute()`` returning
    the canned response for the account, suitable for driving
    ``GoogleCalendarSource._poll_calendar``.
    """
    fixture = _load_calendar_fixture(account)
    events_response = fixture["events"]

    list_req = MagicMock()
    list_req.execute.return_value = events_response

    events_api = MagicMock()
    events_api.list.return_value = list_req

    service = MagicMock()
    service.events.return_value = events_api
    return service


@pytest.fixture
def fake_gmail_clients() -> dict[str, MagicMock]:
    """Per-account fake Gmail messages APIs (``personal``, ``work``)."""
    if not _GMAIL_FIXTURES_DIR.is_dir():
        pytest.skip(f"Gmail fixtures missing at {_GMAIL_FIXTURES_DIR}")
    return {
        "personal": make_fake_gmail_messages_api("personal"),
        "work": make_fake_gmail_messages_api("work"),
    }


@pytest.fixture
def fake_calendar_services() -> dict[str, MagicMock]:
    """Per-account fake Google Calendar services (``personal``, ``shared``)."""
    if not _CALENDAR_FIXTURES_DIR.is_dir():
        pytest.skip(f"Calendar fixtures missing at {_CALENDAR_FIXTURES_DIR}")
    return {
        "personal": make_fake_calendar_service("personal"),
        "shared": make_fake_calendar_service("shared"),
    }


def attachment_bytes(name: str) -> bytes:
    """Read raw bytes from ``tests/fixtures/attachments/<name>``."""
    return (_ATTACHMENT_FIXTURES_DIR / name).read_bytes()


def make_fake_drive_service(file_sizes: dict[str, int]) -> MagicMock:
    """Build a fake Drive ``service`` exposing ``files().get(fileId=, fields=)``.

    Used by :class:`GoogleCalendarSource._enrich_attachment_sizes` to fetch
    the per-file ``size`` field before the parser decides whether to index
    or fall back to metadata-only.  *file_sizes* maps Drive file_id to the
    integer byte count the fake should return.
    """
    drive = MagicMock()

    def _get(fileId: str, fields: str = "size") -> MagicMock:  # noqa: N803
        req = MagicMock()
        req.execute.return_value = {"size": str(file_sizes.get(fileId, 0))}
        return req

    drive.files.return_value.get.side_effect = _get
    return drive
