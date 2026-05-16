"""Tests for Gmail and Google Calendar setup in ``fieldnotes init``."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from worker.init import (
    _append_calendar_config,
    _append_gmail_config,
    _list_calendars,
    _list_gmail_labels,
    _prompt_multi_select,
    _prompt_path,
    _prompt_yes_no,
)


# ── _prompt_yes_no ──────────────────────────────────────────────────


class TestPromptYesNo:
    def test_yes(self) -> None:
        with patch("builtins.input", return_value="y"):
            assert _prompt_yes_no("Enable?") is True

    def test_no(self) -> None:
        with patch("builtins.input", return_value="n"):
            assert _prompt_yes_no("Enable?") is False

    def test_empty_default_false(self) -> None:
        with patch("builtins.input", return_value=""):
            assert _prompt_yes_no("Enable?") is False

    def test_empty_default_true(self) -> None:
        with patch("builtins.input", return_value=""):
            assert _prompt_yes_no("Enable?", default=True) is True

    def test_yes_full_word(self) -> None:
        with patch("builtins.input", return_value="yes"):
            assert _prompt_yes_no("Enable?") is True

    def test_no_full_word(self) -> None:
        with patch("builtins.input", return_value="no"):
            assert _prompt_yes_no("Enable?") is False


# ── _prompt_path ────────────────────────────────────────────────────


class TestPromptPath:
    """Tests for _prompt_path (readline tab-completion wrapper)."""

    def test_returns_user_input(self) -> None:
        with patch("builtins.input", return_value="/tmp/mydir"):
            assert _prompt_path("Pick dir") == "/tmp/mydir"

    def test_returns_default_on_empty(self) -> None:
        with patch("builtins.input", return_value=""):
            assert _prompt_path("Pick dir", "~/Documents") == "~/Documents"

    def test_restores_readline_state(self) -> None:
        import readline

        sentinel = lambda text, state: None  # noqa: E731
        readline.set_completer(sentinel)
        orig_delims = readline.get_completer_delims()

        with patch("builtins.input", return_value="/tmp"):
            _prompt_path("Dir")

        assert readline.get_completer() is sentinel
        assert readline.get_completer_delims() == orig_delims

    def test_completer_matches_real_paths(self, tmp_path: Path) -> None:
        """The internal completer should produce glob matches."""
        # Create some dirs/files under tmp_path
        (tmp_path / "alpha").mkdir()
        (tmp_path / "alpha_file.txt").touch()
        (tmp_path / "beta").mkdir()
        import readline

        with patch("builtins.input", return_value=str(tmp_path)):
            _prompt_path("Dir")

        # Completer was restored already — re-install to test it
        # Instead, we test the completer directly by importing the function
        # Use a fresh call to get the completer while it's active
        captured_completer = None

        def capture_input(prompt: str) -> str:
            nonlocal captured_completer
            captured_completer = readline.get_completer()
            return str(tmp_path)

        with patch("builtins.input", side_effect=capture_input):
            _prompt_path("Dir")

        assert captured_completer is not None
        # Should match "alpha" and "alpha_file.txt" for the partial "alph"
        prefix = str(tmp_path) + "/alph"
        m0 = captured_completer(prefix, 0)
        m1 = captured_completer(prefix, 1)
        assert m0 is not None
        assert m1 is not None
        # One should be a dir (ends with /), one a file
        results = {m0, m1}
        assert any(r.endswith("/") for r in results)
        assert any(r.endswith(".txt") for r in results)
        # state=2 should be None (only 2 matches)
        assert captured_completer(prefix, 2) is None

    def test_completer_expands_tilde(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """When user types ~/..., completions should preserve the ~ prefix."""
        import readline

        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / "Docs").mkdir()

        captured_completer = None

        def capture_input(prompt: str) -> str:
            nonlocal captured_completer
            captured_completer = readline.get_completer()
            return "~/Docs"

        with patch("builtins.input", side_effect=capture_input):
            _prompt_path("Dir")

        m0 = captured_completer("~/D", 0)
        assert m0 is not None
        assert m0.startswith("~/"), f"Expected ~/ prefix, got {m0!r}"

    def test_falls_back_when_readline_missing(self) -> None:
        """If readline can't be imported, plain _prompt is used."""
        import builtins

        real_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name == "readline":
                raise ImportError("no readline")
            return real_import(name, *args, **kwargs)

        with patch("builtins.input", return_value="/fallback"), \
             patch("builtins.__import__", side_effect=blocked_import):
            assert _prompt_path("Dir") == "/fallback"


# ── _prompt_multi_select ────────────────────────────────────────────


class TestPromptMultiSelect:
    ITEMS = [
        {"id": "INBOX", "name": "Inbox"},
        {"id": "SENT", "name": "Sent"},
        {"id": "STARRED", "name": "Starred"},
    ]

    def test_single_selection(self) -> None:
        with patch("builtins.input", return_value="2"):
            result = _prompt_multi_select(
                "Pick:", self.ITEMS, "id", "name", defaults=[1],
            )
        assert result == ["SENT"]

    def test_multiple_selection(self) -> None:
        with patch("builtins.input", return_value="1,3"):
            result = _prompt_multi_select(
                "Pick:", self.ITEMS, "id", "name", defaults=[1],
            )
        assert result == ["INBOX", "STARRED"]

    def test_default_on_empty(self) -> None:
        # Empty input → _prompt returns the default string "1"
        with patch("builtins.input", return_value=""):
            result = _prompt_multi_select(
                "Pick:", self.ITEMS, "id", "name", defaults=[1],
            )
        assert result == ["INBOX"]

    def test_out_of_range_ignored(self) -> None:
        with patch("builtins.input", return_value="1,99"):
            result = _prompt_multi_select(
                "Pick:", self.ITEMS, "id", "name", defaults=[1],
            )
        assert result == ["INBOX"]

    def test_deduplicates(self) -> None:
        with patch("builtins.input", return_value="2,2"):
            result = _prompt_multi_select(
                "Pick:", self.ITEMS, "id", "name",
            )
        assert result == ["SENT"]

    def test_spaces_in_input(self) -> None:
        with patch("builtins.input", return_value=" 1 , 2 "):
            result = _prompt_multi_select(
                "Pick:", self.ITEMS, "id", "name",
            )
        assert result == ["INBOX", "SENT"]

    def test_non_numeric_ignored(self) -> None:
        with patch("builtins.input", return_value="1,abc,3"):
            result = _prompt_multi_select(
                "Pick:", self.ITEMS, "id", "name",
            )
        assert result == ["INBOX", "STARRED"]


# ── _list_gmail_labels ──────────────────────────────────────────────


class TestListGmailLabels:
    def _mock_labels_api(self, raw_labels: list[dict]) -> MagicMock:
        service = MagicMock()
        service.users.return_value.labels.return_value.list.return_value.execute.return_value = {
            "labels": raw_labels,
        }
        return service

    def test_system_labels_ordered(self) -> None:
        raw = [
            {"id": "SENT", "name": "SENT", "type": "system"},
            {"id": "INBOX", "name": "INBOX", "type": "system"},
            {"id": "TRASH", "name": "TRASH", "type": "system"},
        ]
        with (
            patch("worker.sources.gmail_auth.get_credentials", return_value=MagicMock()),
            patch("googleapiclient.discovery.build", return_value=self._mock_labels_api(raw)),
        ):
            labels = _list_gmail_labels(Path("/fake/creds.json"))

        ids = [lb["id"] for lb in labels]
        assert ids == ["INBOX", "SENT", "TRASH"]
        # Friendly names
        assert labels[0]["name"] == "Inbox"
        assert labels[1]["name"] == "Sent"

    def test_user_labels_sorted_alpha(self) -> None:
        raw = [
            {"id": "Label_2", "name": "Zeta", "type": "user"},
            {"id": "Label_1", "name": "Alpha", "type": "user"},
        ]
        with (
            patch("worker.sources.gmail_auth.get_credentials", return_value=MagicMock()),
            patch("googleapiclient.discovery.build", return_value=self._mock_labels_api(raw)),
        ):
            labels = _list_gmail_labels(Path("/fake/creds.json"))

        assert [lb["name"] for lb in labels] == ["Alpha", "Zeta"]

    def test_system_before_user(self) -> None:
        raw = [
            {"id": "Label_1", "name": "MyLabel", "type": "user"},
            {"id": "INBOX", "name": "INBOX", "type": "system"},
        ]
        with (
            patch("worker.sources.gmail_auth.get_credentials", return_value=MagicMock()),
            patch("googleapiclient.discovery.build", return_value=self._mock_labels_api(raw)),
        ):
            labels = _list_gmail_labels(Path("/fake/creds.json"))

        assert labels[0]["id"] == "INBOX"
        assert labels[1]["id"] == "Label_1"

    def test_skips_internal_labels(self) -> None:
        raw = [
            {"id": "INBOX", "name": "INBOX", "type": "system"},
            {"id": "UNREAD", "name": "UNREAD", "type": "system"},
            {"id": "CHAT", "name": "CHAT", "type": "system"},
        ]
        with (
            patch("worker.sources.gmail_auth.get_credentials", return_value=MagicMock()),
            patch("googleapiclient.discovery.build", return_value=self._mock_labels_api(raw)),
        ):
            labels = _list_gmail_labels(Path("/fake/creds.json"))

        ids = {lb["id"] for lb in labels}
        assert "UNREAD" not in ids
        assert "CHAT" not in ids
        assert "INBOX" in ids

    def test_empty_returns_empty(self) -> None:
        with (
            patch("worker.sources.gmail_auth.get_credentials", return_value=MagicMock()),
            patch("googleapiclient.discovery.build", return_value=self._mock_labels_api([])),
        ):
            labels = _list_gmail_labels(Path("/fake/creds.json"))

        assert labels == []


# ── _list_calendars ─────────────────────────────────────────────────


class TestListCalendars:
    def _mock_calendar_api(self, items: list[dict]) -> MagicMock:
        service = MagicMock()
        service.calendarList.return_value.list.return_value.execute.return_value = {
            "items": items,
        }
        return service

    def test_primary_first(self) -> None:
        items = [
            {"id": "other@group.calendar.google.com", "summary": "Birthdays"},
            {
                "id": "user@gmail.com",
                "summary": "My Calendar",
                "primary": True,
            },
        ]
        with (
            patch("worker.sources.calendar_auth.get_credentials", return_value=MagicMock()),
            patch("googleapiclient.discovery.build", return_value=self._mock_calendar_api(items)),
        ):
            cals = _list_calendars(Path("/fake/creds.json"))

        assert cals[0]["id"] == "primary"
        assert cals[0]["name"] == "My Calendar"
        assert cals[1]["name"] == "Birthdays"

    def test_others_sorted_alpha(self) -> None:
        items = [
            {"id": "z@group.calendar.google.com", "summary": "Zzz"},
            {"id": "a@group.calendar.google.com", "summary": "Aaa"},
        ]
        with (
            patch("worker.sources.calendar_auth.get_credentials", return_value=MagicMock()),
            patch("googleapiclient.discovery.build", return_value=self._mock_calendar_api(items)),
        ):
            cals = _list_calendars(Path("/fake/creds.json"))

        assert [c["name"] for c in cals] == ["Aaa", "Zzz"]

    def test_summary_override_used(self) -> None:
        items = [
            {
                "id": "x@group.calendar.google.com",
                "summaryOverride": "My Work",
            },
        ]
        with (
            patch("worker.sources.calendar_auth.get_credentials", return_value=MagicMock()),
            patch("googleapiclient.discovery.build", return_value=self._mock_calendar_api(items)),
        ):
            cals = _list_calendars(Path("/fake/creds.json"))

        assert cals[0]["name"] == "My Work"

    def test_fallback_to_id(self) -> None:
        items = [{"id": "x@group.calendar.google.com"}]
        with (
            patch("worker.sources.calendar_auth.get_credentials", return_value=MagicMock()),
            patch("googleapiclient.discovery.build", return_value=self._mock_calendar_api(items)),
        ):
            cals = _list_calendars(Path("/fake/creds.json"))

        assert cals[0]["name"] == "x@group.calendar.google.com"

    def test_empty_returns_empty(self) -> None:
        with (
            patch("worker.sources.calendar_auth.get_credentials", return_value=MagicMock()),
            patch("googleapiclient.discovery.build", return_value=self._mock_calendar_api([])),
        ):
            cals = _list_calendars(Path("/fake/creds.json"))

        assert cals == []


# ── Config generation helpers ───────────────────────────────────────


class TestAppendGmailConfig:
    def test_single_label(self) -> None:
        result = _append_gmail_config("# config", "~/.fn/creds.json", ["INBOX"])
        assert '[sources.gmail]' in result
        assert 'label_filter = ["INBOX"]' in result
        assert 'client_secrets_path = "~/.fn/creds.json"' in result
        assert "poll_interval_seconds = 300" in result

    def test_multiple_labels(self) -> None:
        result = _append_gmail_config(
            "", "~/c.json", ["INBOX", "SENT", "Label_1"],
        )
        assert 'label_filter = ["INBOX", "SENT", "Label_1"]' in result

    def test_special_chars_escaped(self) -> None:
        result = _append_gmail_config("", 'C:\\Users\\me\\creds.json', ["INBOX"])
        assert 'C:\\\\Users\\\\me\\\\creds.json' in result


class TestAppendCalendarConfig:
    def test_single_calendar(self) -> None:
        result = _append_calendar_config("# config", "~/c.json", ["primary"])
        assert "[sources.google_calendar]" in result
        assert 'calendar_ids = ["primary"]' in result
        assert "max_initial_days = 90" in result

    def test_multiple_calendars(self) -> None:
        result = _append_calendar_config(
            "", "~/c.json", ["primary", "work@group.calendar.google.com"],
        )
        assert (
            'calendar_ids = ["primary", "work@group.calendar.google.com"]'
            in result
        )


# ── Integration: interactive config with Gmail/Calendar ─────────────


class TestInteractiveGmailCalendar:
    """Test the _interactive_config flow with Gmail/Calendar prompts."""

    def _run_interactive(
        self,
        inputs: list[str],
        *,
        gmail_labels: list[dict] | None = None,
        calendars: list[dict] | None = None,
        creds_exist: bool = True,
    ) -> str:
        from contextlib import ExitStack

        from worker.init import _interactive_config

        input_iter = iter(inputs)
        base_config = (
            '[neo4j]\npassword = ""\n'
            '[sources.files]\nwatch_paths = ["~/Documents"]\n'
            '[sources.obsidian]\nvault_paths = ["~/obsidian-vault"]\n'
        )

        orig_exists = Path.exists

        def fake_exists(p: Path) -> bool:
            if p.name.endswith(".json"):
                return creds_exist
            return orig_exists(p)

        patches = [
            patch("builtins.input", side_effect=input_iter),
            patch("getpass.getpass", return_value="testpass1234"),
            patch.object(Path, "exists", fake_exists),
        ]
        if gmail_labels is not None:
            patches.append(
                patch("worker.init._list_gmail_labels", return_value=gmail_labels)
            )
        if calendars is not None:
            patches.append(
                patch("worker.init._list_calendars", return_value=calendars)
            )

        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            return _interactive_config(base_config)

    def test_gmail_with_label_selection(self) -> None:
        labels = [
            {"id": "INBOX", "name": "Inbox"},
            {"id": "SENT", "name": "Sent"},
            {"id": "Label_1", "name": "Work"},
        ]
        inputs = [
            "",      # provider → ollama
            "",      # documents dir → ~/Documents
            "",      # obsidian vault → default
            "y",     # set up gmail?
            "",      # credentials path → default
            "1,3",   # select labels: Inbox, Work
            "n",     # set up calendar?
        ]
        result = self._run_interactive(
            inputs, gmail_labels=labels, creds_exist=True,
        )
        assert '[sources.gmail]' in result
        assert 'label_filter = ["INBOX", "Label_1"]' in result

    def test_gmail_declined(self) -> None:
        inputs = [
            "",      # provider
            "",      # documents
            "",      # obsidian
            "n",     # gmail? no
            "n",     # calendar? no
        ]
        result = self._run_interactive(inputs)
        assert "[sources.gmail]" not in result

    def test_calendar_with_selection(self) -> None:
        cals = [
            {"id": "primary", "name": "My Calendar"},
            {"id": "work@group.calendar.google.com", "name": "Work"},
        ]
        inputs = [
            "",      # provider
            "",      # documents
            "",      # obsidian
            "n",     # gmail? no
            "y",     # calendar? yes
            "",      # credentials path → default
            "1,2",   # select both calendars
        ]
        result = self._run_interactive(
            inputs, calendars=cals, creds_exist=True,
        )
        assert "[sources.google_calendar]" in result
        assert 'calendar_ids = ["primary", "work@group.calendar.google.com"]' in result

    def test_gmail_creds_missing_falls_back(self) -> None:
        inputs = [
            "",      # provider
            "",      # documents
            "",      # obsidian
            "y",     # gmail? yes
            "",      # credentials path → default
            "n",     # calendar? no
        ]
        result = self._run_interactive(inputs, creds_exist=False)
        assert '[sources.gmail]' in result
        assert 'label_filter = ["INBOX"]' in result

    def test_calendar_reuses_gmail_creds_path(self) -> None:
        labels = [{"id": "INBOX", "name": "Inbox"}]
        cals = [{"id": "primary", "name": "Main"}]
        inputs = [
            "",      # provider
            "",      # documents
            "",      # obsidian
            "y",     # gmail? yes
            "~/my/creds.json",  # credentials path
            "1",     # select INBOX
            "y",     # calendar? yes
            # no credentials path prompt — reuses gmail path
            "1",     # select primary
        ]
        result = self._run_interactive(
            inputs, gmail_labels=labels, calendars=cals, creds_exist=True,
        )
        assert "[sources.gmail]" in result
        assert "[sources.google_calendar]" in result
        # Both should reference the same creds path
        assert result.count("~/my/creds.json") == 2
