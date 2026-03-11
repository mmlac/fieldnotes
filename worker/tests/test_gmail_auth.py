"""Tests for Gmail OAuth2 authentication (gmail_auth.py).

Covers: token refresh, expired credentials, missing client secrets,
and file permission handling — all with mocked google-auth.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from google.oauth2.credentials import Credentials

from worker.sources.gmail_auth import SCOPES, get_credentials


def _fake_creds(
    valid: bool = True,
    expired: bool = False,
    refresh_token: str | None = "refresh-tok",
) -> MagicMock:
    """Build a mock Credentials object."""
    creds = MagicMock(spec=Credentials)
    creds.valid = valid
    creds.expired = expired
    creds.refresh_token = refresh_token
    creds.to_json.return_value = json.dumps({"token": "t", "refresh_token": "r"})
    return creds


class TestGetCredentials:
    """Unit tests for get_credentials()."""

    def test_returns_cached_valid_token(self, tmp_path: Path) -> None:
        """If token file exists and creds are valid, return immediately."""
        token = tmp_path / "token.json"
        token.write_text('{"token": "good"}')

        mock_creds = _fake_creds(valid=True)
        with patch.object(
            Credentials, "from_authorized_user_file", return_value=mock_creds
        ):
            result = get_credentials("unused_secrets.json", token_path=token)

        assert result is mock_creds

    def test_refreshes_expired_token(self, tmp_path: Path) -> None:
        """Expired creds with a refresh_token are refreshed via Request()."""
        token = tmp_path / "token.json"
        token.write_text('{"token": "old"}')

        mock_creds = _fake_creds(valid=False, expired=True, refresh_token="rt")

        with (
            patch.object(
                Credentials, "from_authorized_user_file", return_value=mock_creds
            ),
            patch("worker.sources.gmail_auth.Request") as MockRequest,
        ):
            result = get_credentials("unused.json", token_path=token)

        mock_creds.refresh.assert_called_once_with(MockRequest())
        assert result is mock_creds
        # Token was persisted
        assert token.exists()

    def test_runs_oauth_flow_when_no_token_file(self, tmp_path: Path) -> None:
        """When no token file exists, runs the OAuth2 consent flow."""
        token = tmp_path / "token.json"
        secrets = tmp_path / "secrets.json"
        secrets.write_text('{"installed": {}}')

        mock_flow = MagicMock()
        mock_new_creds = _fake_creds(valid=True)
        mock_flow.run_local_server.return_value = mock_new_creds

        with patch(
            "worker.sources.gmail_auth.InstalledAppFlow.from_client_secrets_file",
            return_value=mock_flow,
        ) as mock_from_file:
            result = get_credentials(secrets, token_path=token)

        mock_from_file.assert_called_once_with(str(secrets), SCOPES)
        mock_flow.run_local_server.assert_called_once_with(port=0)
        assert result is mock_new_creds
        # Token persisted
        assert token.exists()

    def test_runs_oauth_flow_when_no_refresh_token(self, tmp_path: Path) -> None:
        """Expired creds without refresh_token trigger full OAuth flow."""
        token = tmp_path / "token.json"
        token.write_text('{"token": "stale"}')
        secrets = tmp_path / "secrets.json"
        secrets.write_text('{"installed": {}}')

        stale_creds = _fake_creds(valid=False, expired=True, refresh_token=None)

        mock_flow = MagicMock()
        mock_new_creds = _fake_creds(valid=True)
        mock_flow.run_local_server.return_value = mock_new_creds

        with (
            patch.object(
                Credentials, "from_authorized_user_file", return_value=stale_creds
            ),
            patch(
                "worker.sources.gmail_auth.InstalledAppFlow.from_client_secrets_file",
                return_value=mock_flow,
            ),
        ):
            result = get_credentials(secrets, token_path=token)

        assert result is mock_new_creds

    def test_token_saved_with_restricted_permissions(self, tmp_path: Path) -> None:
        """Saved token file has 0o600 permissions (owner-only)."""
        token = tmp_path / "token.json"
        secrets = tmp_path / "secrets.json"
        secrets.write_text('{"installed": {}}')

        mock_flow = MagicMock()
        mock_new_creds = _fake_creds(valid=True)
        mock_flow.run_local_server.return_value = mock_new_creds

        with patch(
            "worker.sources.gmail_auth.InstalledAppFlow.from_client_secrets_file",
            return_value=mock_flow,
        ):
            get_credentials(secrets, token_path=token)

        assert token.exists()
        mode = token.stat().st_mode & 0o777
        assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """Token parent dirs are created if they don't exist."""
        token = tmp_path / "deep" / "nested" / "token.json"
        secrets = tmp_path / "secrets.json"
        secrets.write_text('{"installed": {}}')

        mock_flow = MagicMock()
        mock_new_creds = _fake_creds(valid=True)
        mock_flow.run_local_server.return_value = mock_new_creds

        with patch(
            "worker.sources.gmail_auth.InstalledAppFlow.from_client_secrets_file",
            return_value=mock_flow,
        ):
            get_credentials(secrets, token_path=token)

        assert token.exists()
        assert token.parent.is_dir()

    def test_refresh_failure_propagates(self, tmp_path: Path) -> None:
        """If token refresh raises, the error propagates to caller."""
        token = tmp_path / "token.json"
        token.write_text('{"token": "old"}')

        mock_creds = _fake_creds(valid=False, expired=True, refresh_token="rt")
        mock_creds.refresh.side_effect = Exception("refresh failed")

        with (
            patch.object(
                Credentials, "from_authorized_user_file", return_value=mock_creds
            ),
            patch("worker.sources.gmail_auth.Request"),
            pytest.raises(Exception, match="refresh failed"),
        ):
            get_credentials("unused.json", token_path=token)
