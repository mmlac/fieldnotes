"""Slack OAuth flow and token persistence.

Mirrors gmail_auth.py / calendar_auth.py: runs the OAuth install flow on
first use, persists the resulting tokens to ~/.fieldnotes/data/slack_token.json
(mode 0600), and validates saved tokens via auth.test on subsequent runs.

Required Slack OAuth scopes (configure these on your Slack app's
"OAuth & Permissions" page):

  Bot scopes:
    - channels:history
    - channels:read
    - groups:history
    - groups:read
    - im:history
    - im:read
    - mpim:history
    - mpim:read
    - users:read
    - users:read.email
"""

from __future__ import annotations

import http.server
import json
import logging
import secrets as _secrets
import time
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from worker.log_sanitizer import redact_home_path
from worker.sources._token_io import read_token_safe, write_token_atomic

logger = logging.getLogger(__name__)

BOT_SCOPES: list[str] = [
    "channels:history",
    "channels:read",
    "groups:history",
    "groups:read",
    "im:history",
    "im:read",
    "mpim:history",
    "mpim:read",
    "users:read",
    "users:read.email",
]

# Currently no user-token-only scopes are required; flow still requests an
# authed_user.id so we can pin a stable Person identity across re-installs.
USER_SCOPES: list[str] = []

DEFAULT_TOKEN_PATH = Path.home() / ".fieldnotes" / "data" / "slack_token.json"

AUTHORIZE_URL = "https://slack.com/oauth/v2/authorize"

# Slack's OAuth redirect URI must exactly match a value registered in the
# app's "Redirect URLs" — so we use a fixed default port rather than a
# random one (unlike Google's flow).
DEFAULT_REDIRECT_PORT = 3000

REAUTH_ERRORS: frozenset[str] = frozenset(
    {"invalid_auth", "token_revoked", "account_inactive", "not_authed"}
)

# OAuth state values are valid for 10 minutes after issuance — long enough
# for the user to authorize in their browser, short enough that a captured
# state cannot be replayed days later.
STATE_TTL_SECONDS = 600


class ReauthRequiredError(RuntimeError):
    """Raised when a saved Slack token is rejected by auth.test.

    The caller (or the user via ``fieldnotes doctor``) should delete the
    token file and re-run the install flow.
    """


class UnknownStateError(RuntimeError):
    """Raised when a callback presents a state value never issued by this ledger.

    Subclasses ``RuntimeError`` so callers that catch the broader type
    (e.g. tests asserting ``state mismatch``) continue to work.
    """


class StateReplayError(RuntimeError):
    """Raised when a callback presents a state value that has already been consumed."""


class StateExpiredError(RuntimeError):
    """Raised when a callback presents a state value past its TTL."""


class StateLedger:
    """In-memory, single-use, time-bounded OAuth state store.

    A fresh ledger is created per install attempt, which provides the
    "keyed by install-attempt UUID" isolation cheaply: state values issued
    by one ledger instance are unknown to any other instance.

    The ledger lives only in process memory by design — a daemon restart
    mid-install drops it and forces the user to re-initiate (fail-closed).

    The clock is injectable to keep tests deterministic without pulling in
    a time-travel dependency.
    """

    def __init__(
        self,
        *,
        ttl_seconds: int = STATE_TTL_SECONDS,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl_seconds = ttl_seconds
        self._now = now
        self._issued: dict[str, float] = {}
        self._consumed: set[str] = set()

    def issue(self) -> str:
        """Generate an unguessable state value and record its issuance time."""
        state = _secrets.token_urlsafe(32)
        self._issued[state] = self._now()
        return state

    def consume(self, state: str) -> None:
        """Validate a callback state. Single-use; raises a typed error on misuse.

        Order of checks: replay (already consumed) → unknown (never issued)
        → expired (past TTL). On success, the state is removed from the
        issued set and recorded as consumed.
        """
        if state in self._consumed:
            raise StateReplayError(
                "OAuth state mismatch: state already consumed (replay)"
            )
        if state not in self._issued:
            raise UnknownStateError(
                "OAuth state mismatch: state was not issued (possible CSRF)"
            )
        issued_at = self._issued.pop(state)
        if (self._now() - issued_at) > self._ttl_seconds:
            raise StateExpiredError(
                f"OAuth state mismatch: state expired after {self._ttl_seconds}s TTL"
            )
        self._consumed.add(state)


@dataclass
class SlackToken:
    """Persisted Slack OAuth token bundle.

    ``authed_user_id`` is the Slack workspace user id of the human who
    authorized the install.  We persist it so that re-installing over an
    existing token preserves Person-node continuity downstream — Slack
    returns the same user id for the same human in the same workspace,
    so the file's ``authed_user_id`` is stable across re-installs.
    """

    bot_token: str
    user_token: str | None
    team_id: str
    bot_user_id: str
    authed_user_id: str
    scope: str

    @classmethod
    def from_oauth_response(cls, resp: dict[str, Any]) -> SlackToken:
        authed_user = resp.get("authed_user") or {}
        team = resp.get("team") or {}
        return cls(
            bot_token=resp["access_token"],
            user_token=authed_user.get("access_token"),
            team_id=team.get("id", ""),
            bot_user_id=resp.get("bot_user_id", ""),
            authed_user_id=authed_user.get("id", ""),
            scope=resp.get("scope", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "bot_token": self.bot_token,
            "user_token": self.user_token,
            "team_id": self.team_id,
            "bot_user_id": self.bot_user_id,
            "authed_user_id": self.authed_user_id,
            "scope": self.scope,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SlackToken:
        return cls(
            bot_token=d["bot_token"],
            user_token=d.get("user_token"),
            team_id=d.get("team_id", ""),
            bot_user_id=d.get("bot_user_id", ""),
            authed_user_id=d.get("authed_user_id", ""),
            scope=d.get("scope", ""),
        )


def _load_token(token_path: str | Path) -> SlackToken | None:
    p = Path(token_path)
    raw = read_token_safe(p)
    if raw is None:
        return None
    return SlackToken.from_dict(json.loads(raw))


def _save_token(token_path: str | Path, token: SlackToken) -> None:
    p = Path(token_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    write_token_atomic(p, json.dumps(token.to_dict(), indent=2))


def _build_authorize_url(
    *,
    client_id: str,
    bot_scopes: list[str],
    user_scopes: list[str],
    redirect_uri: str,
    state: str,
) -> str:
    params = {
        "client_id": client_id,
        "scope": ",".join(bot_scopes),
        "user_scope": ",".join(user_scopes),
        "redirect_uri": redirect_uri,
        "state": state,
    }
    return f"{AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    """Minimal one-shot handler that captures the OAuth callback query."""

    def do_GET(self) -> None:  # noqa: N802 — required by BaseHTTPRequestHandler
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        # Stash first-value-per-key on the server so the caller can read it.
        self.server.received_query = {  # type: ignore[attr-defined]
            k: v[0] for k, v in qs.items()
        }
        body = b"Slack auth complete. You can close this window.\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # Suppress default stderr access logging.
        return


def _wait_for_callback(port: int) -> dict[str, str]:
    """Bind a one-shot localhost listener and return the callback query."""
    server = http.server.HTTPServer(("127.0.0.1", port), _CallbackHandler)
    server.received_query = {}  # type: ignore[attr-defined]
    try:
        server.handle_request()
    finally:
        server.server_close()
    return server.received_query  # type: ignore[attr-defined]


def _exchange_code(
    *,
    client_id: str,
    client_secret: str,
    code: str,
    redirect_uri: str,
) -> dict[str, Any]:
    """Exchange an OAuth code for tokens via slack.com/api/oauth.v2.access."""
    response = WebClient().oauth_v2_access(
        client_id=client_id,
        client_secret=client_secret,
        code=code,
        redirect_uri=redirect_uri,
    )
    return dict(response.data)


def install_slack(
    *,
    client_id: str,
    client_secret: str,
    redirect_port: int = DEFAULT_REDIRECT_PORT,
    token_path: str | Path = DEFAULT_TOKEN_PATH,
) -> SlackToken:
    """Run the Slack OAuth install flow and persist the resulting tokens.

    Prints the install URL to stdout, listens on ``http://localhost:<port>``
    for Slack's redirect, exchanges the code for bot+user tokens, and
    writes them to *token_path* with mode 0600.
    """
    redirect_uri = f"http://localhost:{redirect_port}/oauth/callback"
    ledger = StateLedger()
    state = ledger.issue()
    url = _build_authorize_url(
        client_id=client_id,
        bot_scopes=BOT_SCOPES,
        user_scopes=USER_SCOPES,
        redirect_uri=redirect_uri,
        state=state,
    )
    print("\nOpen this URL in your browser to install Slack:\n")
    print(f"  {url}\n")
    print(f"Listening for callback on {redirect_uri} ...")

    query = _wait_for_callback(redirect_port)
    ledger.consume(query.get("state", ""))
    if "code" not in query:
        err = query.get("error", "unknown error")
        raise RuntimeError(f"Slack OAuth callback returned error: {err}")

    resp = _exchange_code(
        client_id=client_id,
        client_secret=client_secret,
        code=query["code"],
        redirect_uri=redirect_uri,
    )
    if not resp.get("ok"):
        raise RuntimeError(f"oauth.v2.access failed: {resp.get('error')}")

    token = SlackToken.from_oauth_response(resp)
    _save_token(token_path, token)
    logger.info("Slack token saved to %s", redact_home_path(str(token_path)))
    return token


def _slack_error_code(exc: SlackApiError) -> str:
    """Extract the ``error`` field from a SlackApiError response."""
    try:
        return (exc.response or {}).get("error", "") or ""
    except Exception:
        return ""


def get_slack_client(
    config: dict[str, Any],
    *,
    token_path: str | Path = DEFAULT_TOKEN_PATH,
) -> WebClient:
    """Return an authenticated Slack ``WebClient`` for the bot token.

    On first call (no token file) runs the OAuth install flow using
    ``config['client_id']`` and ``config['client_secret']``.  On subsequent
    calls loads the persisted token and validates it via ``auth.test``.

    Raises :class:`ReauthRequiredError` if the saved token is rejected by
    Slack — the caller should delete the token file and re-run the
    install flow.
    """
    token = _load_token(token_path)
    if token is None:
        client_id = config.get("client_id")
        client_secret = config.get("client_secret")
        if not client_id or not client_secret:
            raise RuntimeError(
                "Slack OAuth requires 'client_id' and 'client_secret' in "
                "config to run the install flow"
            )
        token = install_slack(
            client_id=client_id,
            client_secret=client_secret,
            redirect_port=int(config.get("redirect_port", DEFAULT_REDIRECT_PORT)),
            token_path=token_path,
        )

    web = WebClient(token=token.bot_token)
    try:
        web.auth_test()
    except SlackApiError as exc:
        err = _slack_error_code(exc)
        if err in REAUTH_ERRORS:
            raise ReauthRequiredError(
                f"Slack token rejected ({err}); delete "
                f"{redact_home_path(str(token_path))} and re-run the install flow"
            ) from exc
        raise
    return web
