"""Tests for conversation history path traversal protection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from worker.cli.history import (
    Conversation,
    TurnRecord,
    _conversation_path,
    load_conversation,
    prune_old_conversations,
    save_conversation,
)


# ---------------------------------------------------------------------------
# Path safety: _conversation_path
# ---------------------------------------------------------------------------

class TestConversationPathSafety:
    def test_normal_id_resolves_inside_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("worker.cli.history._CONVERSATIONS_DIR", tmp_path)
        path = _conversation_path("abc123def456")
        assert path.parent == tmp_path.resolve()
        assert path.name == "abc123def456.json"

    def test_dotdot_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("worker.cli.history._CONVERSATIONS_DIR", tmp_path)
        with pytest.raises(ValueError, match="Invalid conversation ID"):
            _conversation_path("../evil")

    def test_absolute_style_traversal_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("worker.cli.history._CONVERSATIONS_DIR", tmp_path)
        with pytest.raises(ValueError, match="Invalid conversation ID"):
            _conversation_path("../../etc/passwd")

    def test_slash_in_id_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("worker.cli.history._CONVERSATIONS_DIR", tmp_path)
        with pytest.raises(ValueError, match="Invalid conversation ID"):
            _conversation_path("subdir/evil")


# ---------------------------------------------------------------------------
# load_conversation: returns None instead of escaping
# ---------------------------------------------------------------------------

class TestLoadConversationSafety:
    def test_traversal_id_returns_none(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
        monkeypatch.setattr("worker.cli.history._CONVERSATIONS_DIR", tmp_path)
        result = load_conversation("../../../etc/passwd")
        assert result is None

    def test_valid_id_not_found_returns_none(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("worker.cli.history._CONVERSATIONS_DIR", tmp_path)
        result = load_conversation("nonexistentid")
        assert result is None

    def test_valid_id_loads_correctly(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("worker.cli.history._CONVERSATIONS_DIR", tmp_path)
        conv = Conversation(id="abc123def456")
        conv.add_turn(TurnRecord(question="hello", answer="world"))
        save_conversation(conv)

        loaded = load_conversation("abc123def456")
        assert loaded is not None
        assert loaded.id == "abc123def456"
        assert len(loaded.turns) == 1


# ---------------------------------------------------------------------------
# save_conversation: refuses unsafe IDs
# ---------------------------------------------------------------------------

class TestSaveConversationSafety:
    def test_traversal_id_not_written(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("worker.cli.history._CONVERSATIONS_DIR", tmp_path)
        evil_id = "../evil"
        conv = Conversation(id=evil_id)
        save_conversation(conv)
        # The file must NOT have been written outside the conversations dir.
        assert not (tmp_path.parent / "evil.json").exists()

    def test_normal_save_creates_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("worker.cli.history._CONVERSATIONS_DIR", tmp_path)
        conv = Conversation(id="safe000id001")
        save_conversation(conv)
        assert (tmp_path / "safe000id001.json").exists()


# ---------------------------------------------------------------------------
# prune_old_conversations: skips unsafe IDs loaded from disk
# ---------------------------------------------------------------------------

class TestPruneConversationSafety:
    def test_malicious_id_in_file_not_deleted_elsewhere(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("worker.cli.history._CONVERSATIONS_DIR", tmp_path)

        # Write a conversation file whose embedded `id` contains a traversal path.
        evil_data = {
            "id": "../../outside",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "turns": [],
        }
        conv_file = tmp_path / "legit_filename.json"
        conv_file.write_text(json.dumps(evil_data), encoding="utf-8")

        # Create a sentinel file one level up to detect if it gets deleted.
        sentinel = tmp_path.parent / "outside.json"
        sentinel.write_text("{}", encoding="utf-8")

        # prune with max_keep=0 so it tries to delete everything
        prune_old_conversations(max_keep=0)

        # The sentinel must still be there — unsafe id was ignored.
        assert sentinel.exists(), "prune_old_conversations deleted a file outside the conversations dir"
        sentinel.unlink(missing_ok=True)
