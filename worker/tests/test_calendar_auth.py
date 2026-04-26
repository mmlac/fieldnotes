"""Tests for Calendar OAuth2 authentication (calendar_auth.py).

Mirrors test_gmail_auth — same shape, different module/scopes/path.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from google.oauth2.credentials import Credentials

from worker.sources.calendar_auth import (
    SCOPES,
    get_credentials,
    token_path_for_account,
)


def _fake_creds(
    valid: bool = True,
    expired: bool = False,
    refresh_token: str | None = "refresh-tok",
) -> MagicMock:
    creds = MagicMock(spec=Credentials)
    creds.valid = valid
    creds.expired = expired
    creds.refresh_token = refresh_token
    creds.to_json.return_value = json.dumps({"token": "t", "refresh_token": "r"})
    return creds


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect Path.home() to a tmp dir so token writes are sandboxed."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    return tmp_path


class TestTokenPathForAccount:
    def test_personal_account_path(self, home: Path) -> None:
        assert token_path_for_account("personal") == (
            home / ".fieldnotes" / "data" / "calendar_token-personal.json"
        )

    def test_work_account_path(self, home: Path) -> None:
        assert token_path_for_account("work-123") == (
            home / ".fieldnotes" / "data" / "calendar_token-work-123.json"
        )


class TestGetCredentials:
    def test_returns_cached_valid_token(self, home: Path) -> None:
        token = token_path_for_account("default")
        token.parent.mkdir(parents=True, exist_ok=True)
        token.write_text('{"token": "good"}')

        mock_creds = _fake_creds(valid=True)
        with patch.object(
            Credentials, "from_authorized_user_file", return_value=mock_creds
        ):
            result = get_credentials("unused_secrets.json", account="default")

        assert result is mock_creds

    def test_refreshes_expired_token(self, home: Path) -> None:
        token = token_path_for_account("default")
        token.parent.mkdir(parents=True, exist_ok=True)
        token.write_text('{"token": "old"}')

        mock_creds = _fake_creds(valid=False, expired=True, refresh_token="rt")

        with (
            patch.object(
                Credentials, "from_authorized_user_file", return_value=mock_creds
            ),
            patch("worker.sources.calendar_auth.Request") as MockRequest,
        ):
            result = get_credentials("unused.json", account="default")

        mock_creds.refresh.assert_called_once_with(MockRequest())
        assert result is mock_creds
        assert token.exists()

    def test_runs_oauth_flow_when_no_token_file(
        self, home: Path, tmp_path: Path
    ) -> None:
        secrets = tmp_path / "secrets.json"
        secrets.write_text('{"installed": {}}')

        mock_flow = MagicMock()
        mock_new_creds = _fake_creds(valid=True)
        mock_flow.run_local_server.return_value = mock_new_creds

        with patch(
            "worker.sources.calendar_auth.InstalledAppFlow.from_client_secrets_file",
            return_value=mock_flow,
        ) as mock_from_file:
            result = get_credentials(secrets, account="default")

        mock_from_file.assert_called_once_with(str(secrets), SCOPES)
        mock_flow.run_local_server.assert_called_once_with(port=0)
        assert result is mock_new_creds
        assert token_path_for_account("default").exists()

    def test_token_saved_with_restricted_permissions(
        self, home: Path, tmp_path: Path
    ) -> None:
        secrets = tmp_path / "secrets.json"
        secrets.write_text('{"installed": {}}')

        mock_flow = MagicMock()
        mock_new_creds = _fake_creds(valid=True)
        mock_flow.run_local_server.return_value = mock_new_creds

        with patch(
            "worker.sources.calendar_auth.InstalledAppFlow.from_client_secrets_file",
            return_value=mock_flow,
        ):
            get_credentials(secrets, account="default")

        token = token_path_for_account("default")
        assert token.exists()
        mode = token.stat().st_mode & 0o777
        assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"

    def test_chmod_tightens_existing_loose_perms(
        self, home: Path, tmp_path: Path
    ) -> None:
        token = token_path_for_account("default")
        token.parent.mkdir(parents=True, exist_ok=True)
        token.write_text('{"token": "old"}')
        token.chmod(0o644)

        mock_creds = _fake_creds(valid=False, expired=True, refresh_token="rt")

        with (
            patch.object(
                Credentials, "from_authorized_user_file", return_value=mock_creds
            ),
            patch("worker.sources.calendar_auth.Request"),
        ):
            get_credentials("unused.json", account="default")

        mode = token.stat().st_mode & 0o777
        assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"

    def test_creates_parent_directories(
        self, home: Path, tmp_path: Path
    ) -> None:
        secrets = tmp_path / "secrets.json"
        secrets.write_text('{"installed": {}}')

        mock_flow = MagicMock()
        mock_new_creds = _fake_creds(valid=True)
        mock_flow.run_local_server.return_value = mock_new_creds

        token = token_path_for_account("default")
        assert not token.parent.exists()

        with patch(
            "worker.sources.calendar_auth.InstalledAppFlow.from_client_secrets_file",
            return_value=mock_flow,
        ):
            get_credentials(secrets, account="default")

        assert token.exists()
        assert token.parent.is_dir()

    def test_two_accounts_no_collision(
        self, home: Path, tmp_path: Path
    ) -> None:
        secrets = tmp_path / "secrets.json"
        secrets.write_text('{"installed": {}}')

        creds_a = _fake_creds(valid=True)
        creds_a.to_json.return_value = json.dumps({"token": "A"})
        creds_b = _fake_creds(valid=True)
        creds_b.to_json.return_value = json.dumps({"token": "B"})

        flow_a = MagicMock()
        flow_a.run_local_server.return_value = creds_a
        flow_b = MagicMock()
        flow_b.run_local_server.return_value = creds_b

        with patch(
            "worker.sources.calendar_auth.InstalledAppFlow.from_client_secrets_file",
            side_effect=[flow_a, flow_b],
        ):
            get_credentials(secrets, account="personal")
            get_credentials(secrets, account="work")

        path_a = token_path_for_account("personal")
        path_b = token_path_for_account("work")

        assert path_a.exists()
        assert path_b.exists()
        assert path_a != path_b
        assert json.loads(path_a.read_text())["token"] == "A"
        assert json.loads(path_b.read_text())["token"] == "B"

    def test_round_trip_save_then_load(
        self, home: Path, tmp_path: Path
    ) -> None:
        secrets = tmp_path / "secrets.json"
        secrets.write_text('{"installed": {}}')

        new_creds = _fake_creds(valid=True)
        new_creds.to_json.return_value = json.dumps(
            {"token": "tk", "refresh_token": "rt"}
        )
        flow = MagicMock()
        flow.run_local_server.return_value = new_creds

        with patch(
            "worker.sources.calendar_auth.InstalledAppFlow.from_client_secrets_file",
            return_value=flow,
        ):
            first = get_credentials(secrets, account="default")

        cached = _fake_creds(valid=True)
        with (
            patch.object(
                Credentials, "from_authorized_user_file", return_value=cached
            ) as mock_load,
            patch(
                "worker.sources.calendar_auth.InstalledAppFlow.from_client_secrets_file"
            ) as mock_flow_factory,
        ):
            second = get_credentials(secrets, account="default")

        assert first is new_creds
        assert second is cached
        mock_load.assert_called_once()
        mock_flow_factory.assert_not_called()

    def test_account_is_required(self) -> None:
        """Calling without account raises TypeError (no silent default)."""
        with pytest.raises(TypeError):
            get_credentials("unused.json")  # type: ignore[call-arg]
