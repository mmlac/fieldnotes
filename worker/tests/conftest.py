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
    return client
