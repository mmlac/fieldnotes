"""Tests for Gmail OAuth2 authentication (gmail_auth.py).

Covers per-account token paths, token refresh, expired credentials,
file permission handling, and the ban on the legacy single-account
call signature — all with mocked google-auth.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from google.oauth2.credentials import Credentials

from worker.sources.gmail_auth import (
    SCOPES,
    get_credentials,
    token_path_for_account,
)


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


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect Path.home() to a tmp dir so token writes are sandboxed."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    return tmp_path


class TestTokenPathForAccount:
    """Token path is derived from the account label."""

    def test_personal_account_path(self, home: Path) -> None:
        assert token_path_for_account("personal") == (
            home / ".fieldnotes" / "data" / "gmail_token-personal.json"
        )

    def test_work_account_path(self, home: Path) -> None:
        assert token_path_for_account("work-123") == (
            home / ".fieldnotes" / "data" / "gmail_token-work-123.json"
        )


class TestGetCredentials:
    """Unit tests for get_credentials()."""

    def test_returns_cached_valid_token(self, home: Path) -> None:
        """If token file exists and creds are valid, return immediately."""
        token = token_path_for_account("default")
        token.parent.mkdir(parents=True, exist_ok=True)
        token.write_text('{"token": "good"}')

        mock_creds = _fake_creds(valid=True)
        with patch.object(
            Credentials, "from_authorized_user_info", return_value=mock_creds
        ):
            result = get_credentials("unused_secrets.json", account="default")

        assert result is mock_creds

    def test_refreshes_expired_token(self, home: Path) -> None:
        """Expired creds with a refresh_token are refreshed via Request()."""
        token = token_path_for_account("default")
        token.parent.mkdir(parents=True, exist_ok=True)
        token.write_text('{"token": "old"}')

        mock_creds = _fake_creds(valid=False, expired=True, refresh_token="rt")

        with (
            patch.object(
                Credentials, "from_authorized_user_info", return_value=mock_creds
            ),
            patch("worker.sources.gmail_auth.Request") as MockRequest,
        ):
            result = get_credentials("unused.json", account="default")

        mock_creds.refresh.assert_called_once_with(MockRequest())
        assert result is mock_creds
        # Token was persisted
        assert token.exists()

    def test_runs_oauth_flow_when_no_token_file(
        self, home: Path, tmp_path: Path
    ) -> None:
        """When no token file exists, runs the OAuth2 consent flow."""
        secrets = tmp_path / "secrets.json"
        secrets.write_text('{"installed": {}}')

        mock_flow = MagicMock()
        mock_new_creds = _fake_creds(valid=True)
        mock_flow.run_local_server.return_value = mock_new_creds

        with patch(
            "worker.sources.gmail_auth.InstalledAppFlow.from_client_secrets_file",
            return_value=mock_flow,
        ) as mock_from_file:
            result = get_credentials(secrets, account="default")

        mock_from_file.assert_called_once_with(str(secrets), SCOPES)
        mock_flow.run_local_server.assert_called_once_with(port=0)
        assert result is mock_new_creds
        # Token persisted at the per-account path
        assert token_path_for_account("default").exists()

    def test_runs_oauth_flow_when_no_refresh_token(
        self, home: Path, tmp_path: Path
    ) -> None:
        """Expired creds without refresh_token trigger full OAuth flow."""
        token = token_path_for_account("default")
        token.parent.mkdir(parents=True, exist_ok=True)
        token.write_text('{"token": "stale"}')
        secrets = tmp_path / "secrets.json"
        secrets.write_text('{"installed": {}}')

        stale_creds = _fake_creds(valid=False, expired=True, refresh_token=None)

        mock_flow = MagicMock()
        mock_new_creds = _fake_creds(valid=True)
        mock_flow.run_local_server.return_value = mock_new_creds

        with (
            patch.object(
                Credentials, "from_authorized_user_info", return_value=stale_creds
            ),
            patch(
                "worker.sources.gmail_auth.InstalledAppFlow.from_client_secrets_file",
                return_value=mock_flow,
            ),
        ):
            result = get_credentials(secrets, account="default")

        assert result is mock_new_creds

    def test_token_saved_with_restricted_permissions(
        self, home: Path, tmp_path: Path
    ) -> None:
        """Saved token file has 0o600 permissions (owner-only)."""
        secrets = tmp_path / "secrets.json"
        secrets.write_text('{"installed": {}}')

        mock_flow = MagicMock()
        mock_new_creds = _fake_creds(valid=True)
        mock_flow.run_local_server.return_value = mock_new_creds

        with patch(
            "worker.sources.gmail_auth.InstalledAppFlow.from_client_secrets_file",
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
        """If a token file already exists with loose perms, save tightens to 0o600."""
        token = token_path_for_account("default")
        token.parent.mkdir(parents=True, exist_ok=True)
        token.write_text('{"token": "old"}')
        token.chmod(0o644)

        mock_creds = _fake_creds(valid=False, expired=True, refresh_token="rt")

        with (
            patch.object(
                Credentials, "from_authorized_user_info", return_value=mock_creds
            ),
            patch("worker.sources.gmail_auth.Request"),
        ):
            get_credentials("unused.json", account="default")

        mode = token.stat().st_mode & 0o777
        assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"

    def test_creates_parent_directories(self, home: Path, tmp_path: Path) -> None:
        """Token parent dirs are created if they don't exist."""
        secrets = tmp_path / "secrets.json"
        secrets.write_text('{"installed": {}}')

        mock_flow = MagicMock()
        mock_new_creds = _fake_creds(valid=True)
        mock_flow.run_local_server.return_value = mock_new_creds

        # Parent dirs do not yet exist under the sandboxed home
        token = token_path_for_account("default")
        assert not token.parent.exists()

        with patch(
            "worker.sources.gmail_auth.InstalledAppFlow.from_client_secrets_file",
            return_value=mock_flow,
        ):
            get_credentials(secrets, account="default")

        assert token.exists()
        assert token.parent.is_dir()

    def test_refresh_failure_propagates(self, home: Path) -> None:
        """If token refresh raises, the error propagates to caller."""
        token = token_path_for_account("default")
        token.parent.mkdir(parents=True, exist_ok=True)
        token.write_text('{"token": "old"}')

        mock_creds = _fake_creds(valid=False, expired=True, refresh_token="rt")
        mock_creds.refresh.side_effect = Exception("refresh failed")

        with (
            patch.object(
                Credentials, "from_authorized_user_info", return_value=mock_creds
            ),
            patch("worker.sources.gmail_auth.Request"),
            pytest.raises(Exception, match="refresh failed"),
        ):
            get_credentials("unused.json", account="default")

    def test_two_accounts_no_collision(self, home: Path, tmp_path: Path) -> None:
        """Saving account A then account B leaves both files intact."""
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
            "worker.sources.gmail_auth.InstalledAppFlow.from_client_secrets_file",
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

    def test_round_trip_save_then_load(self, home: Path, tmp_path: Path) -> None:
        """First call saves; second call loads the same file."""
        secrets = tmp_path / "secrets.json"
        secrets.write_text('{"installed": {}}')

        new_creds = _fake_creds(valid=True)
        new_creds.to_json.return_value = json.dumps(
            {"token": "tk", "refresh_token": "rt"}
        )
        flow = MagicMock()
        flow.run_local_server.return_value = new_creds

        with patch(
            "worker.sources.gmail_auth.InstalledAppFlow.from_client_secrets_file",
            return_value=flow,
        ):
            first = get_credentials(secrets, account="default")

        # Second call: file now exists, so flow must NOT run.
        cached = _fake_creds(valid=True)
        with (
            patch.object(
                Credentials, "from_authorized_user_info", return_value=cached
            ) as mock_load,
            patch(
                "worker.sources.gmail_auth.InstalledAppFlow.from_client_secrets_file"
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


class TestTokenSymlinkSafety:
    """Symlink and atomicity invariants on the persisted token file."""

    def test_token_save_refuses_symlink(self, home: Path, tmp_path: Path) -> None:
        """If the token path is a symlink, save raises before writing."""
        token = token_path_for_account("default")
        token.parent.mkdir(parents=True, exist_ok=True)
        attacker = tmp_path / "attacker.json"
        attacker.write_text('{"sentinel": true}')
        token.symlink_to(attacker)

        secrets = tmp_path / "secrets.json"
        secrets.write_text('{"installed": {}}')
        mock_flow = MagicMock()
        mock_flow.run_local_server.return_value = _fake_creds(valid=True)

        with patch(
            "worker.sources.gmail_auth.InstalledAppFlow.from_client_secrets_file",
            return_value=mock_flow,
        ):
            with pytest.raises(OSError, match="symlink"):
                get_credentials(secrets, account="default")

        # Attacker target untouched; no creds written
        assert attacker.read_text() == '{"sentinel": true}'

    def test_token_load_refuses_symlink(self, home: Path) -> None:
        """Pre-existing symlink at the token path must not be silently followed."""
        token = token_path_for_account("default")
        token.parent.mkdir(parents=True, exist_ok=True)
        attacker = token.parent / "attacker.json"
        attacker.write_text('{"token": "evil"}')
        token.symlink_to(attacker)

        with pytest.raises(OSError, match="symlink"):
            get_credentials("unused.json", account="default")

    def test_token_save_refuses_symlinked_parent(
        self, home: Path, tmp_path: Path
    ) -> None:
        """Save bails when the token's parent dir is a symlink."""
        real = tmp_path / "real_data"
        real.mkdir()
        # Replace the data dir with a symlink to a different real dir
        token = token_path_for_account("default")
        token.parent.parent.mkdir(parents=True, exist_ok=True)
        token.parent.symlink_to(real, target_is_directory=True)

        secrets = tmp_path / "secrets.json"
        secrets.write_text('{"installed": {}}')
        mock_flow = MagicMock()
        mock_flow.run_local_server.return_value = _fake_creds(valid=True)

        with patch(
            "worker.sources.gmail_auth.InstalledAppFlow.from_client_secrets_file",
            return_value=mock_flow,
        ):
            with pytest.raises(OSError, match="symlink"):
                get_credentials(secrets, account="default")

        # Nothing landed under the real dir either
        assert list(real.iterdir()) == []

    def test_token_save_atomic_under_crash(
        self, home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A write interrupted before rename leaves the prior token intact."""
        token = token_path_for_account("default")
        token.parent.mkdir(parents=True, exist_ok=True)
        token.write_text(json.dumps({"token": "orig"}))
        token.chmod(0o600)
        original_text = token.read_text()

        # Force the OAuth flow path: existing creds are invalid + no refresh.
        stale = _fake_creds(valid=False, expired=True, refresh_token=None)
        mock_flow = MagicMock()
        mock_flow.run_local_server.return_value = _fake_creds(valid=True)

        import worker.sources._token_io as token_io

        def boom(*args: object, **kwargs: object) -> None:
            raise RuntimeError("simulated crash before rename")

        monkeypatch.setattr(token_io.os, "rename", boom)

        with (
            patch.object(Credentials, "from_authorized_user_info", return_value=stale),
            patch(
                "worker.sources.gmail_auth.InstalledAppFlow.from_client_secrets_file",
                return_value=mock_flow,
            ),
            pytest.raises(RuntimeError, match="simulated crash"),
        ):
            get_credentials("unused.json", account="default")

        assert token.read_text() == original_text

    def test_token_save_mode_0600_survives_rename(
        self, home: Path, tmp_path: Path
    ) -> None:
        """0o600 holds even when overwriting a stale loose-mode file."""
        token = token_path_for_account("default")
        token.parent.mkdir(parents=True, exist_ok=True)
        token.write_text('{"token": "old"}')
        token.chmod(0o644)

        stale = _fake_creds(valid=False, expired=True, refresh_token=None)
        mock_flow = MagicMock()
        mock_flow.run_local_server.return_value = _fake_creds(valid=True)

        with (
            patch.object(Credentials, "from_authorized_user_info", return_value=stale),
            patch(
                "worker.sources.gmail_auth.InstalledAppFlow.from_client_secrets_file",
                return_value=mock_flow,
            ),
        ):
            get_credentials("unused.json", account="default")

        mode = token.stat().st_mode & 0o777
        assert mode == 0o600
