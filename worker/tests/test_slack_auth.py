"""Tests for Slack OAuth flow and token persistence (slack_auth.py).

Covers: fresh install, token reuse, revoked-token error path, token
file mode 0600, and authed_user_id continuity across re-installs — all
with mocked Slack network calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from slack_sdk.errors import SlackApiError

from worker.sources.slack_auth import (
    BOT_SCOPES,
    DEFAULT_REDIRECT_PORT,
    REAUTH_ERRORS,
    STATE_TTL_SECONDS,
    ReauthRequiredError,
    SlackToken,
    StateExpiredError,
    StateLedger,
    StateReplayError,
    UnknownStateError,
    _build_authorize_url,
    _load_token,
    _save_token,
    get_slack_client,
    install_slack,
)


def _oauth_response(
    *,
    bot_token: str = "xoxb-bot",
    user_token: str | None = "xoxp-user",
    team_id: str = "T123",
    bot_user_id: str = "UBOT",
    authed_user_id: str = "UHUMAN",
    scope: str = "channels:read",
) -> dict:
    """Build a fake oauth.v2.access response payload."""
    return {
        "ok": True,
        "access_token": bot_token,
        "token_type": "bot",
        "scope": scope,
        "bot_user_id": bot_user_id,
        "app_id": "AAPP",
        "team": {"id": team_id, "name": "Acme"},
        "authed_user": {
            "id": authed_user_id,
            "scope": "users:read",
            "access_token": user_token,
            "token_type": "user",
        },
    }


def _slack_api_error(error_code: str) -> SlackApiError:
    """Build a SlackApiError with the given Slack ``error`` field."""
    response = MagicMock()
    response.__getitem__ = lambda self, key: {"error": error_code, "ok": False}[key]
    response.get = lambda key, default=None: {
        "error": error_code,
        "ok": False,
    }.get(key, default)
    return SlackApiError(message=error_code, response=response)


class TestSlackTokenSerialization:
    def test_round_trip_via_dict(self) -> None:
        token = SlackToken(
            bot_token="xoxb",
            user_token="xoxp",
            team_id="T1",
            bot_user_id="UB",
            authed_user_id="UH",
            scope="a,b",
        )
        assert SlackToken.from_dict(token.to_dict()) == token

    def test_from_oauth_response_extracts_authed_user_id(self) -> None:
        token = SlackToken.from_oauth_response(_oauth_response(authed_user_id="UABC"))
        assert token.authed_user_id == "UABC"
        assert token.bot_token == "xoxb-bot"
        assert token.user_token == "xoxp-user"

    def test_from_oauth_response_handles_missing_user_token(self) -> None:
        resp = _oauth_response(user_token=None)
        token = SlackToken.from_oauth_response(resp)
        assert token.user_token is None


class TestBuildAuthorizeUrl:
    def test_includes_client_id_scopes_and_state(self) -> None:
        url = _build_authorize_url(
            client_id="CID",
            bot_scopes=["channels:read", "users:read"],
            user_scopes=[],
            redirect_uri="http://localhost:3000/oauth/callback",
            state="STATE123",
        )
        assert "client_id=CID" in url
        assert "state=STATE123" in url
        assert "channels%3Aread%2Cusers%3Aread" in url
        # All required scopes documented in module are baked into BOT_SCOPES
        for scope in (
            "channels:history",
            "channels:read",
            "im:history",
            "users:read.email",
        ):
            assert scope in BOT_SCOPES


class TestSaveAndLoadToken:
    def test_save_creates_file_with_mode_0600(self, tmp_path: Path) -> None:
        path = tmp_path / "slack_token.json"
        token = SlackToken("xoxb", "xoxp", "T", "UB", "UH", "s")
        _save_token(path, token)
        assert path.exists()
        mode = path.stat().st_mode & 0o777
        assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"

    def test_save_chmod_tightens_existing_file(self, tmp_path: Path) -> None:
        """Even if a stale file exists with looser perms, save enforces 0600."""
        path = tmp_path / "slack_token.json"
        path.write_text("{}")
        path.chmod(0o644)
        token = SlackToken("xoxb", None, "T", "UB", "UH", "s")
        _save_token(path, token)
        mode = path.stat().st_mode & 0o777
        assert mode == 0o600

    def test_save_creates_parent_directories(self, tmp_path: Path) -> None:
        path = tmp_path / "deep" / "nested" / "slack_token.json"
        token = SlackToken("xoxb", None, "T", "UB", "UH", "s")
        _save_token(path, token)
        assert path.exists()

    def test_load_returns_none_when_missing(self, tmp_path: Path) -> None:
        assert _load_token(tmp_path / "nope.json") is None

    def test_load_round_trips_save(self, tmp_path: Path) -> None:
        path = tmp_path / "tok.json"
        token = SlackToken("xoxb", "xoxp", "T1", "UB", "UH", "s")
        _save_token(path, token)
        loaded = _load_token(path)
        assert loaded == token

    def test_token_save_refuses_symlink(self, tmp_path: Path) -> None:
        """Pre-existing symlink at the target path makes save raise."""
        attacker = tmp_path / "attacker.json"
        attacker.write_text('{"sentinel": true}')
        path = tmp_path / "slack_token.json"
        path.symlink_to(attacker)

        token = SlackToken("xoxb", None, "T", "UB", "UH", "s")
        with pytest.raises(OSError, match="symlink"):
            _save_token(path, token)
        # Attacker file untouched
        assert attacker.read_text() == '{"sentinel": true}'

    def test_token_load_refuses_symlink(self, tmp_path: Path) -> None:
        """Symlinked token path makes load raise rather than read attacker."""
        attacker = tmp_path / "attacker.json"
        attacker.write_text(
            json.dumps(SlackToken("xoxb-evil", None, "T", "UB", "UH", "s").to_dict())
        )
        path = tmp_path / "slack_token.json"
        path.symlink_to(attacker)

        with pytest.raises(OSError, match="symlink"):
            _load_token(path)

    def test_token_save_refuses_symlinked_parent(self, tmp_path: Path) -> None:
        """Symlinked parent dir is rejected before any write."""
        real_parent = tmp_path / "real"
        real_parent.mkdir()
        link_parent = tmp_path / "link"
        link_parent.symlink_to(real_parent, target_is_directory=True)

        path = link_parent / "slack_token.json"
        token = SlackToken("xoxb", None, "T", "UB", "UH", "s")
        with pytest.raises(OSError, match="symlink"):
            _save_token(path, token)
        # No file was written under the real parent either
        assert not (real_parent / "slack_token.json").exists()

    def test_token_save_atomic_under_crash(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the rename never happens, the prior token is untouched and tmp cleaned."""
        path = tmp_path / "slack_token.json"
        original = SlackToken("xoxb-orig", None, "T", "UB", "UH", "s")
        _save_token(path, original)
        original_text = path.read_text()

        import worker.sources._token_io as token_io

        def boom(*args: object, **kwargs: object) -> None:
            raise RuntimeError("simulated crash before rename")

        monkeypatch.setattr(token_io.os, "rename", boom)

        with pytest.raises(RuntimeError, match="simulated crash"):
            _save_token(path, SlackToken("xoxb-new", None, "T", "UB", "UH", "s"))

        # Original token preserved
        assert path.read_text() == original_text
        # tmp file may exist but did not clobber the target
        assert path.is_file() and not path.is_symlink()

    def test_token_save_mode_0600_survives_rename(self, tmp_path: Path) -> None:
        """Mode invariant: 0o600 holds across the rename, not just initial create."""
        path = tmp_path / "slack_token.json"
        # Pre-existing loose-mode file gets replaced via rename
        path.write_text("{}")
        path.chmod(0o644)

        token = SlackToken("xoxb", None, "T", "UB", "UH", "s")
        _save_token(path, token)
        mode = path.stat().st_mode & 0o777
        assert mode == 0o600


class TestInstallSlack:
    def test_fresh_install_persists_tokens_with_mode_0600(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "slack_token.json"
        with (
            patch(
                "worker.sources.slack_auth._wait_for_callback",
                return_value={"code": "CODE", "state": "STATE_XYZ"},
            ),
            patch(
                "worker.sources.slack_auth._secrets.token_urlsafe",
                return_value="STATE_XYZ",
            ),
            patch(
                "worker.sources.slack_auth._exchange_code",
                return_value=_oauth_response(),
            ) as mock_exchange,
        ):
            token = install_slack(
                client_id="CID",
                client_secret="CSECRET",
                redirect_port=4321,
                token_path=path,
            )

        # Returned token reflects oauth.v2.access response.
        assert token.bot_token == "xoxb-bot"
        assert token.authed_user_id == "UHUMAN"
        # Persisted to disk.
        assert path.exists()
        on_disk = json.loads(path.read_text())
        assert on_disk["bot_token"] == "xoxb-bot"
        # Mode 0600.
        mode = path.stat().st_mode & 0o777
        assert mode == 0o600
        # Install URL was printed.
        printed = capsys.readouterr().out
        assert "slack.com/oauth/v2/authorize" in printed
        assert "client_id=CID" in printed
        # Code was exchanged with the matching redirect_uri.
        mock_exchange.assert_called_once()
        kwargs = mock_exchange.call_args.kwargs
        assert kwargs["code"] == "CODE"
        assert kwargs["redirect_uri"] == "http://localhost:4321/oauth/callback"
        assert kwargs["client_id"] == "CID"

    def test_state_is_one_shot(self) -> None:
        """A state validated once cannot be validated again."""
        ledger = StateLedger()
        state = ledger.issue()
        ledger.consume(state)
        with pytest.raises(StateReplayError):
            ledger.consume(state)

    def test_state_expires_after_ttl(self) -> None:
        """A state validated past its TTL raises StateExpiredError."""
        clock = [1000.0]
        ledger = StateLedger(ttl_seconds=STATE_TTL_SECONDS, now=lambda: clock[0])
        state = ledger.issue()
        # Eleven minutes later — past the 10-minute TTL
        clock[0] = 1000.0 + (11 * 60)
        with pytest.raises(StateExpiredError):
            ledger.consume(state)

    def test_unknown_state_raises(self) -> None:
        """A state never issued by the ledger is rejected."""
        ledger = StateLedger()
        with pytest.raises(UnknownStateError, match="state mismatch"):
            ledger.consume("never-issued-by-anyone")

    def test_concurrent_install_attempts_isolated(self) -> None:
        """Two install attempts in parallel each only accept their own state.

        Each attempt creates its own ledger, so a state issued by attempt A
        is unknown to attempt B and vice versa.
        """
        ledger_a = StateLedger()
        ledger_b = StateLedger()
        state_a = ledger_a.issue()
        state_b = ledger_b.issue()

        with pytest.raises(UnknownStateError):
            ledger_a.consume(state_b)
        with pytest.raises(UnknownStateError):
            ledger_b.consume(state_a)

        # Each ledger still accepts its own state after the cross-attempts.
        ledger_a.consume(state_a)
        ledger_b.consume(state_b)

    def test_state_mismatch_raises(self, tmp_path: Path) -> None:
        with (
            patch(
                "worker.sources.slack_auth._wait_for_callback",
                return_value={"code": "CODE", "state": "WRONG"},
            ),
            patch(
                "worker.sources.slack_auth._secrets.token_urlsafe",
                return_value="EXPECTED",
            ),
            pytest.raises(RuntimeError, match="state mismatch"),
        ):
            install_slack(
                client_id="CID",
                client_secret="S",
                redirect_port=4321,
                token_path=tmp_path / "t.json",
            )

    def test_callback_error_propagates(self, tmp_path: Path) -> None:
        with (
            patch(
                "worker.sources.slack_auth._wait_for_callback",
                return_value={"error": "access_denied", "state": "S"},
            ),
            patch(
                "worker.sources.slack_auth._secrets.token_urlsafe",
                return_value="S",
            ),
            pytest.raises(RuntimeError, match="access_denied"),
        ):
            install_slack(
                client_id="CID",
                client_secret="S",
                redirect_port=4321,
                token_path=tmp_path / "t.json",
            )

    def test_oauth_v2_access_failure_raises(self, tmp_path: Path) -> None:
        with (
            patch(
                "worker.sources.slack_auth._wait_for_callback",
                return_value={"code": "C", "state": "S"},
            ),
            patch(
                "worker.sources.slack_auth._secrets.token_urlsafe",
                return_value="S",
            ),
            patch(
                "worker.sources.slack_auth._exchange_code",
                return_value={"ok": False, "error": "invalid_code"},
            ),
            pytest.raises(RuntimeError, match="invalid_code"),
        ):
            install_slack(
                client_id="CID",
                client_secret="S",
                redirect_port=4321,
                token_path=tmp_path / "t.json",
            )

    def test_reinstall_preserves_authed_user_id(self, tmp_path: Path) -> None:
        """Re-running install over an existing token preserves user_id continuity.

        Slack assigns stable workspace user ids per human, so a re-install
        flow returns the same authed_user.id.  The on-disk authed_user_id
        therefore matches before and after, and downstream Person nodes
        keep their identity.
        """
        path = tmp_path / "slack_token.json"
        # Initial install
        with (
            patch(
                "worker.sources.slack_auth._wait_for_callback",
                return_value={"code": "C1", "state": "S1"},
            ),
            patch(
                "worker.sources.slack_auth._secrets.token_urlsafe",
                return_value="S1",
            ),
            patch(
                "worker.sources.slack_auth._exchange_code",
                return_value=_oauth_response(
                    authed_user_id="UALICE", bot_token="xoxb-old"
                ),
            ),
        ):
            install_slack(
                client_id="CID",
                client_secret="S",
                redirect_port=4321,
                token_path=path,
            )
        first_user_id = _load_token(path).authed_user_id

        # Re-install — Slack returns the same authed_user_id
        with (
            patch(
                "worker.sources.slack_auth._wait_for_callback",
                return_value={"code": "C2", "state": "S2"},
            ),
            patch(
                "worker.sources.slack_auth._secrets.token_urlsafe",
                return_value="S2",
            ),
            patch(
                "worker.sources.slack_auth._exchange_code",
                return_value=_oauth_response(
                    authed_user_id="UALICE", bot_token="xoxb-new"
                ),
            ),
        ):
            install_slack(
                client_id="CID",
                client_secret="S",
                redirect_port=4321,
                token_path=path,
            )
        second = _load_token(path)
        assert second.authed_user_id == first_user_id == "UALICE"
        # New bot token is persisted (rotation works)
        assert second.bot_token == "xoxb-new"
        # Mode still 0600 after re-install over existing file
        mode = path.stat().st_mode & 0o777
        assert mode == 0o600


class TestGetSlackClient:
    def test_reuses_existing_token_when_valid(self, tmp_path: Path) -> None:
        path = tmp_path / "slack_token.json"
        _save_token(
            path,
            SlackToken("xoxb-saved", None, "T", "UB", "UH", "s"),
        )

        mock_web = MagicMock()
        mock_web.auth_test.return_value = {"ok": True}

        with patch(
            "worker.sources.slack_auth.WebClient", return_value=mock_web
        ) as mock_cls:
            client = get_slack_client({}, token_path=path)

        # WebClient was constructed with the saved bot token.
        mock_cls.assert_called_once_with(token="xoxb-saved")
        mock_web.auth_test.assert_called_once()
        assert client is mock_web

    def test_revoked_token_raises_reauth_required(self, tmp_path: Path) -> None:
        path = tmp_path / "slack_token.json"
        _save_token(path, SlackToken("xoxb-revoked", None, "T", "UB", "UH", "s"))

        mock_web = MagicMock()
        mock_web.auth_test.side_effect = _slack_api_error("token_revoked")

        with (
            patch("worker.sources.slack_auth.WebClient", return_value=mock_web),
            pytest.raises(ReauthRequiredError, match="token_revoked"),
        ):
            get_slack_client({}, token_path=path)

    @pytest.mark.parametrize("err", sorted(REAUTH_ERRORS))
    def test_all_auth_errors_trigger_reauth(self, tmp_path: Path, err: str) -> None:
        path = tmp_path / "slack_token.json"
        _save_token(path, SlackToken("xoxb", None, "T", "UB", "UH", "s"))

        mock_web = MagicMock()
        mock_web.auth_test.side_effect = _slack_api_error(err)
        with (
            patch("worker.sources.slack_auth.WebClient", return_value=mock_web),
            pytest.raises(ReauthRequiredError),
        ):
            get_slack_client({}, token_path=path)

    def test_other_slack_errors_propagate(self, tmp_path: Path) -> None:
        path = tmp_path / "slack_token.json"
        _save_token(path, SlackToken("xoxb", None, "T", "UB", "UH", "s"))

        mock_web = MagicMock()
        mock_web.auth_test.side_effect = _slack_api_error("ratelimited")
        with (
            patch("worker.sources.slack_auth.WebClient", return_value=mock_web),
            pytest.raises(SlackApiError),
        ):
            get_slack_client({}, token_path=path)

    def test_runs_install_flow_when_no_token(self, tmp_path: Path) -> None:
        path = tmp_path / "slack_token.json"

        installed = SlackToken("xoxb-fresh", None, "T", "UB", "UH", "s")
        mock_web = MagicMock()
        mock_web.auth_test.return_value = {"ok": True}

        with (
            patch(
                "worker.sources.slack_auth.install_slack", return_value=installed
            ) as mock_install,
            patch("worker.sources.slack_auth.WebClient", return_value=mock_web),
        ):
            get_slack_client(
                {"client_id": "CID", "client_secret": "S"},
                token_path=path,
            )

        mock_install.assert_called_once()
        kwargs = mock_install.call_args.kwargs
        assert kwargs["client_id"] == "CID"
        assert kwargs["client_secret"] == "S"
        assert kwargs["redirect_port"] == DEFAULT_REDIRECT_PORT
        assert kwargs["token_path"] == path

    def test_missing_client_id_raises_when_no_token(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="client_id"):
            get_slack_client({}, token_path=tmp_path / "missing.json")
