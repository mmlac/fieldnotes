"""Tests for Calendar OAuth2 authentication (calendar_auth.py).

Mirrors test_gmail_auth — same shape, different module/scopes/path.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
from google.oauth2.credentials import Credentials

from worker.sources.calendar_auth import (
    CALENDAR_SCOPE,
    DRIVE_SCOPE,
    SCOPES,
    ReauthRequiredError,
    check_calendar_auth,
    get_credentials,
    get_scopes,
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
            Credentials, "from_authorized_user_info", return_value=mock_creds
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
                Credentials, "from_authorized_user_info", return_value=mock_creds
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
                Credentials, "from_authorized_user_info", return_value=mock_creds
            ),
            patch("worker.sources.calendar_auth.Request"),
        ):
            get_credentials("unused.json", account="default")

        mode = token.stat().st_mode & 0o777
        assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"

    def test_creates_parent_directories(self, home: Path, tmp_path: Path) -> None:
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

    def test_two_accounts_no_collision(self, home: Path, tmp_path: Path) -> None:
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

    def test_round_trip_save_then_load(self, home: Path, tmp_path: Path) -> None:
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
                Credentials, "from_authorized_user_info", return_value=cached
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


class TestGetScopes:
    """Scope set toggles between calendar-only and calendar+drive."""

    def test_default_excludes_drive(self) -> None:
        scopes = get_scopes(False)
        assert CALENDAR_SCOPE in scopes
        assert DRIVE_SCOPE not in scopes
        # Backward-compat alias still names the narrower scope set.
        assert SCOPES == [CALENDAR_SCOPE]

    def test_download_attachments_adds_drive(self) -> None:
        scopes = get_scopes(True)
        assert CALENDAR_SCOPE in scopes
        assert DRIVE_SCOPE in scopes

    def test_get_credentials_uses_widened_scopes(
        self, home: Path, tmp_path: Path
    ) -> None:
        """download_attachments=True flows the wider scope set into
        InstalledAppFlow so Drive scope is requested at consent time."""
        secrets = tmp_path / "secrets.json"
        secrets.write_text('{"installed": {}}')

        mock_flow = MagicMock()
        mock_flow.run_local_server.return_value = _fake_creds(valid=True)

        with patch(
            "worker.sources.calendar_auth.InstalledAppFlow.from_client_secrets_file",
            return_value=mock_flow,
        ) as mock_from_file:
            get_credentials(secrets, account="default", download_attachments=True)

        passed_scopes = mock_from_file.call_args[0][1]
        assert DRIVE_SCOPE in passed_scopes


class TestCheckCalendarAuth:
    """check_calendar_auth raises when scope drift would silently break runtime."""

    def test_silent_when_download_attachments_false(self, home: Path) -> None:
        # No token on disk + narrower scope requested = nothing to verify.
        check_calendar_auth("default", download_attachments=False)

    def test_silent_when_token_missing(self, home: Path) -> None:
        # Install flow handles missing tokens; do not raise here.
        check_calendar_auth("default", download_attachments=True)

    def test_silent_when_drive_scope_present(self, home: Path) -> None:
        token = token_path_for_account("default")
        token.parent.mkdir(parents=True, exist_ok=True)
        token.write_text(json.dumps({"scopes": [CALENDAR_SCOPE, DRIVE_SCOPE]}))
        check_calendar_auth("default", download_attachments=True)

    def test_raises_when_drive_scope_missing(self, home: Path) -> None:
        token = token_path_for_account("default")
        token.parent.mkdir(parents=True, exist_ok=True)
        # Token only has the narrower calendar scope.
        token.write_text(json.dumps({"scopes": [CALENDAR_SCOPE]}))
        with pytest.raises(ReauthRequiredError):
            check_calendar_auth("default", download_attachments=True)

    def test_handles_scopes_as_string(self, home: Path) -> None:
        """Older google-auth tokens persist scopes as a space-separated string."""
        token = token_path_for_account("default")
        token.parent.mkdir(parents=True, exist_ok=True)
        token.write_text(json.dumps({"scopes": f"{CALENDAR_SCOPE} {DRIVE_SCOPE}"}))
        check_calendar_auth("default", download_attachments=True)


class TestMultiAccountTokenIsolation:
    """Account A enabling drive scope must not contaminate account B's token."""

    def test_independent_scope_sets_per_account(
        self, home: Path, tmp_path: Path
    ) -> None:
        secrets = tmp_path / "secrets.json"
        secrets.write_text('{"installed": {}}')

        creds_a = _fake_creds(valid=True)
        creds_a.to_json.return_value = json.dumps(
            {"token": "A", "scopes": [CALENDAR_SCOPE, DRIVE_SCOPE]}
        )
        creds_b = _fake_creds(valid=True)
        creds_b.to_json.return_value = json.dumps(
            {"token": "B", "scopes": [CALENDAR_SCOPE]}
        )

        flow_a = MagicMock()
        flow_a.run_local_server.return_value = creds_a
        flow_b = MagicMock()
        flow_b.run_local_server.return_value = creds_b

        scope_calls: list[list[str]] = []

        def factory(_secrets: str, scopes: list[str]) -> MagicMock:
            scope_calls.append(list(scopes))
            return flow_a if len(scope_calls) == 1 else flow_b

        with patch(
            "worker.sources.calendar_auth.InstalledAppFlow.from_client_secrets_file",
            side_effect=factory,
        ):
            get_credentials(secrets, account="work", download_attachments=True)
            get_credentials(secrets, account="home", download_attachments=False)

        assert DRIVE_SCOPE in scope_calls[0]
        assert DRIVE_SCOPE not in scope_calls[1]

        # Validate account B's token does not silently pass the drive
        # scope check just because account A authorised drive.
        with pytest.raises(ReauthRequiredError):
            check_calendar_auth("home", download_attachments=True)
        check_calendar_auth("work", download_attachments=True)


class TestTokenSymlinkSafety:
    """Symlink and atomicity invariants on the persisted token file."""

    def test_token_save_refuses_symlink(self, home: Path, tmp_path: Path) -> None:
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
            "worker.sources.calendar_auth.InstalledAppFlow.from_client_secrets_file",
            return_value=mock_flow,
        ):
            with pytest.raises(OSError, match="symlink"):
                get_credentials(secrets, account="default")

        assert attacker.read_text() == '{"sentinel": true}'

    def test_token_load_refuses_symlink(self, home: Path) -> None:
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
        real = tmp_path / "real_data"
        real.mkdir()
        token = token_path_for_account("default")
        token.parent.parent.mkdir(parents=True, exist_ok=True)
        token.parent.symlink_to(real, target_is_directory=True)

        secrets = tmp_path / "secrets.json"
        secrets.write_text('{"installed": {}}')
        mock_flow = MagicMock()
        mock_flow.run_local_server.return_value = _fake_creds(valid=True)

        with patch(
            "worker.sources.calendar_auth.InstalledAppFlow.from_client_secrets_file",
            return_value=mock_flow,
        ):
            with pytest.raises(OSError, match="symlink"):
                get_credentials(secrets, account="default")

        assert list(real.iterdir()) == []

    def test_token_save_atomic_under_crash(
        self, home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        token = token_path_for_account("default")
        token.parent.mkdir(parents=True, exist_ok=True)
        token.write_text(json.dumps({"token": "orig"}))
        token.chmod(0o600)
        original_text = token.read_text()

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
                "worker.sources.calendar_auth.InstalledAppFlow.from_client_secrets_file",
                return_value=mock_flow,
            ),
            pytest.raises(RuntimeError, match="simulated crash"),
        ):
            get_credentials("unused.json", account="default")

        assert token.read_text() == original_text

    def test_token_save_mode_0600_survives_rename(
        self, home: Path, tmp_path: Path
    ) -> None:
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
                "worker.sources.calendar_auth.InstalledAppFlow.from_client_secrets_file",
                return_value=mock_flow,
            ),
        ):
            get_credentials("unused.json", account="default")

        mode = token.stat().st_mode & 0o777
        assert mode == 0o600


class TestRemoteScopeVerification:
    """Tokeninfo introspection: the local scope claim must agree with what
    Google says is granted at the account level.  Without this, a user who
    revokes Drive access from the Google granted-apps page would still
    pass the local-only check and the daemon would only learn at runtime
    via a 403."""

    @pytest.fixture(autouse=True)
    def _clear_cache(self) -> None:
        from worker.sources import calendar_auth

        calendar_auth._TOKENINFO_CACHE.clear()
        yield
        calendar_auth._TOKENINFO_CACHE.clear()

    def _write_token_with_drive(self, account: str = "default") -> Path:
        token = token_path_for_account(account)
        token.parent.mkdir(parents=True, exist_ok=True)
        token.write_text(
            json.dumps(
                {
                    "token": "access-tok",
                    "scopes": [CALENDAR_SCOPE, DRIVE_SCOPE],
                }
            )
        )
        return token

    def test_token_scope_introspection_match(self, home: Path) -> None:
        """tokeninfo confirms drive scope: check returns silently."""
        self._write_token_with_drive()
        with (
            patch(
                "worker.sources.calendar_auth._refresh_access_token",
                return_value="access-tok",
            ),
            patch(
                "worker.sources.calendar_auth._fetch_remote_scopes",
                return_value=frozenset([CALENDAR_SCOPE, DRIVE_SCOPE]),
            ),
        ):
            check_calendar_auth("default", download_attachments=True)

    def test_token_scope_revoked_remotely(self, home: Path) -> None:
        """tokeninfo lacks drive scope: invalidate token + raise."""
        token = self._write_token_with_drive()
        with (
            patch(
                "worker.sources.calendar_auth._refresh_access_token",
                return_value="access-tok",
            ),
            patch(
                "worker.sources.calendar_auth._fetch_remote_scopes",
                return_value=frozenset([CALENDAR_SCOPE]),
            ),
            pytest.raises(ReauthRequiredError, match="revoked"),
        ):
            check_calendar_auth("default", download_attachments=True)
        # Stale local claim deleted so a subsequent run cannot pass the
        # local-only check on the same on-disk file.
        assert not token.exists()

    def test_tokeninfo_cached(
        self, home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two calls within 1h hit tokeninfo once; a call past 1h refetches."""
        self._write_token_with_drive()

        fetch_calls = 0

        def fake_fetch(_token: str) -> frozenset[str]:
            nonlocal fetch_calls
            fetch_calls += 1
            return frozenset([CALENDAR_SCOPE, DRIVE_SCOPE])

        clock = [1000.0]

        def fake_now() -> float:
            return clock[0]

        with (
            patch(
                "worker.sources.calendar_auth._refresh_access_token",
                return_value="access-tok",
            ),
            patch(
                "worker.sources.calendar_auth._fetch_remote_scopes",
                side_effect=fake_fetch,
            ),
            patch(
                "worker.sources.calendar_auth._now_monotonic",
                side_effect=fake_now,
            ),
        ):
            check_calendar_auth("default", download_attachments=True)
            assert fetch_calls == 1

            clock[0] = 1000.0 + 600  # 10 min later — well within TTL
            check_calendar_auth("default", download_attachments=True)
            assert fetch_calls == 1, "second call within 1h should hit cache"

            clock[0] = 1000.0 + 3601  # past 1h TTL
            check_calendar_auth("default", download_attachments=True)
            assert fetch_calls == 2, "call past 1h should refetch"

    def test_tokeninfo_network_failure_graceful(self, home: Path) -> None:
        """Transport / HTTP errors must not block the daemon."""
        self._write_token_with_drive()

        def boom(*_args: object, **_kwargs: object) -> object:
            raise httpx.ConnectTimeout("simulated timeout")

        with (
            patch(
                "worker.sources.calendar_auth._refresh_access_token",
                return_value="access-tok",
            ),
            patch("worker.sources.calendar_auth.httpx.get", side_effect=boom),
        ):
            # Must not raise — graceful degrade on transient network blip.
            check_calendar_auth("default", download_attachments=True)
