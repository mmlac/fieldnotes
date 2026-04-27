"""Google Calendar OAuth2 authentication and token persistence.

Reuses the same OAuth2 flow as Gmail but with Calendar-specific scopes.
If the Gmail credentials.json already has Calendar scopes enabled,
the same client secrets file can be shared.

When a per-account ``download_attachments`` knob is True, the OAuth flow
also requests ``drive.readonly`` so attachments stored on Drive can be
fetched.  Pre-existing tokens with the narrower scope still work for
read-only event ingestion; flipping ``download_attachments`` on later is
detected by :func:`check_calendar_auth`, which raises
:class:`ReauthRequiredError` so the user knows to re-run the install
flow.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import httpx
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from worker.log_sanitizer import redact_home_path
from worker.sources._token_io import read_token_safe, write_token_atomic

logger = logging.getLogger(__name__)

CALENDAR_SCOPE = "https://www.googleapis.com/auth/calendar.events.readonly"
DRIVE_SCOPE = "https://www.googleapis.com/auth/drive.readonly"

TOKENINFO_URL = "https://oauth2.googleapis.com/tokeninfo"
TOKENINFO_CACHE_TTL_SECONDS = 3600

# Per-token cache of (expiry_monotonic, granted_scopes) returned by
# Google's tokeninfo endpoint.  Keyed by the on-disk token path so two
# accounts using the same module instance keep independent cache entries.
_TOKENINFO_CACHE: dict[Path, tuple[float, frozenset[str]]] = {}

# Backwards-compatible alias — the read-only event scope alone, without
# Drive.  Callers that need the dynamic scope set should use
# :func:`get_scopes` instead.
SCOPES = [CALENDAR_SCOPE]


class ReauthRequiredError(RuntimeError):
    """Raised when a saved Calendar token lacks a scope the caller now needs.

    Typically surfaced when ``download_attachments`` is flipped on for an
    account whose existing token only carries ``calendar.events.readonly``.
    The caller (or the user via ``fieldnotes doctor``) should delete the
    token file and re-run the install flow.
    """


def get_scopes(download_attachments: bool) -> list[str]:
    """Return the OAuth scope set required by the caller.

    Always includes ``calendar.events.readonly``; adds ``drive.readonly``
    when *download_attachments* is True so the same credentials can drive
    both event polling and attachment fetches.
    """
    if download_attachments:
        return [CALENDAR_SCOPE, DRIVE_SCOPE]
    return [CALENDAR_SCOPE]


def token_path_for_account(account: str) -> Path:
    """Derive the on-disk token path for an account label."""
    return Path.home() / ".fieldnotes" / "data" / f"calendar_token-{account}.json"


def get_credentials(
    client_secrets_path: str | Path,
    account: str,
    *,
    download_attachments: bool = False,
) -> Credentials:
    """Obtain valid Google Calendar OAuth2 credentials for *account*.

    Loads cached credentials from the per-account token file if available,
    refreshes them when expired, or runs the interactive OAuth2 consent
    flow using *client_secrets_path* for initial authorization.

    When *download_attachments* is True, the requested scope set also
    includes ``drive.readonly`` so attachments stored on Drive can be
    fetched.

    Returns a ``google.oauth2.credentials.Credentials`` instance ready for
    use with the Calendar (and optionally Drive) APIs.
    """
    scopes = get_scopes(download_attachments)
    token_path = token_path_for_account(account)
    creds: Credentials | None = None

    raw = read_token_safe(token_path)
    if raw is not None:
        creds = Credentials.from_authorized_user_info(json.loads(raw), scopes)

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        logger.info("Refreshing expired Calendar token for account=%s", account)
        creds.refresh(Request())
    else:
        logger.info(
            "Starting Calendar OAuth2 authorization flow for account=%s", account
        )
        flow = InstalledAppFlow.from_client_secrets_file(
            str(client_secrets_path), scopes
        )
        creds = flow.run_local_server(port=0)

    # Persist for next run via the symlink-safe atomic helper: writes go to
    # a tmp file with O_NOFOLLOW + 0o600, then rename onto the final path.
    token_path.parent.mkdir(parents=True, exist_ok=True)
    write_token_atomic(token_path, creds.to_json())
    logger.info(
        "Calendar token saved for account=%s at %s",
        account,
        redact_home_path(str(token_path)),
    )

    return creds


def _token_scopes(token_path: Path) -> list[str]:
    """Return the scope list the persisted token was issued with.

    Returns an empty list if the file is missing, unreadable, or carries
    no ``scopes`` field — the caller should treat that as "scope unknown"
    rather than "no scopes granted".
    """
    try:
        raw = read_token_safe(token_path)
    except OSError:
        return []
    if raw is None:
        return []
    try:
        data = json.loads(raw)
    except ValueError:
        return []
    raw = data.get("scopes")
    if isinstance(raw, list):
        return [str(s) for s in raw]
    if isinstance(raw, str):
        # Older tokens persisted scopes as a space-separated string.
        return raw.split()
    return []


def _refresh_access_token(token_path: Path) -> str | None:
    """Load creds from *token_path*, refreshing if expired, return access token.

    Returns ``None`` when the file is missing/unreadable, the JSON is
    malformed, the token cannot be refreshed, or no access token is set
    after load.  Refresh failures are swallowed so the caller can degrade
    gracefully — get_credentials will surface a hard error if it sees the
    same problem when it actually needs the token.
    """
    raw = read_token_safe(token_path)
    if raw is None:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    try:
        creds = Credentials.from_authorized_user_info(
            data, [CALENDAR_SCOPE, DRIVE_SCOPE]
        )
    except (ValueError, KeyError):
        return None
    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except Exception as exc:
            logger.warning(
                "Calendar token refresh failed at %s: %s",
                redact_home_path(str(token_path)),
                exc,
            )
            return None
        try:
            write_token_atomic(token_path, creds.to_json())
        except OSError as exc:
            logger.warning(
                "Calendar token persist after refresh failed at %s: %s",
                redact_home_path(str(token_path)),
                exc,
            )
    return creds.token if creds.token else None


def _fetch_remote_scopes(access_token: str) -> frozenset[str]:
    """Call Google's tokeninfo endpoint and return the granted scope set.

    Raises ``httpx.HTTPError`` on transport errors and HTTP 4xx/5xx — the
    caller decides whether to fail loud or degrade.  Google returns the
    scopes as a space-separated string under the ``scope`` key.
    """
    resp = httpx.get(
        TOKENINFO_URL,
        params={"access_token": access_token},
        timeout=httpx.Timeout(connect=5.0, read=5.0, write=5.0, pool=5.0),
    )
    resp.raise_for_status()
    data = resp.json()
    raw = data.get("scope", "")
    if isinstance(raw, list):
        return frozenset(str(s) for s in raw)
    return frozenset(str(raw).split())


def _now_monotonic() -> float:
    """Indirection for tests to advance the cache clock without sleeping."""
    return time.monotonic()


def _cached_remote_scopes(token_path: Path) -> frozenset[str] | None:
    entry = _TOKENINFO_CACHE.get(token_path)
    if entry is None:
        return None
    expiry, scopes = entry
    if _now_monotonic() >= expiry:
        _TOKENINFO_CACHE.pop(token_path, None)
        return None
    return scopes


def _store_remote_scopes(token_path: Path, scopes: frozenset[str]) -> None:
    _TOKENINFO_CACHE[token_path] = (
        _now_monotonic() + TOKENINFO_CACHE_TTL_SECONDS,
        scopes,
    )


def _verify_remote_drive_scope(account: str, token_path: Path) -> None:
    """Confirm Google still grants ``drive.readonly`` for the saved token.

    On mismatch (drive scope missing from tokeninfo's response) the local
    token is deleted and :class:`ReauthRequiredError` is raised so the
    user gets an actionable re-auth prompt instead of an opaque 403 at
    runtime.  Network and HTTP errors degrade gracefully — a transient
    blip should never block the daemon.
    """
    scopes = _cached_remote_scopes(token_path)
    if scopes is None:
        access_token = _refresh_access_token(token_path)
        if access_token is None:
            return
        try:
            scopes = _fetch_remote_scopes(access_token)
        except httpx.HTTPError as exc:
            logger.warning(
                "tokeninfo probe failed for account=%s, skipping remote "
                "scope check: %s",
                account,
                exc,
            )
            return
        _store_remote_scopes(token_path, scopes)

    if DRIVE_SCOPE in scopes:
        return

    # Stale local claim — drop the cache entry and the on-disk token so a
    # subsequent run cannot pass the local-only check on the same file.
    _TOKENINFO_CACHE.pop(token_path, None)
    try:
        token_path.unlink()
    except OSError:
        pass
    raise ReauthRequiredError(
        f"Calendar token for account={account!r} no longer carries "
        f"{DRIVE_SCOPE!r} — scope was revoked at the Google account level. "
        f"Re-run the install flow with download_attachments=true to grant "
        f"it again."
    )


def check_calendar_auth(
    account: str,
    *,
    download_attachments: bool = False,
) -> None:
    """Verify the persisted token covers the scopes the caller needs.

    Performs a two-stage check when ``download_attachments`` is True:

    1. Local: the saved token's ``scopes`` claim must include
       ``drive.readonly``.  Catches the case where attachment indexing was
       flipped on after the token was issued with the narrower scope set.
    2. Remote: Google's tokeninfo endpoint is consulted (cached for
       :data:`TOKENINFO_CACHE_TTL_SECONDS`) to confirm the scope is still
       granted at the account level.  If a user revoked the app's Drive
       access from Google's granted-apps page, the local claim is stale
       and runtime would fail with a 403 — surfacing it here as
       :class:`ReauthRequiredError` is much more actionable.

    Raises :class:`ReauthRequiredError` on either mismatch.  Returns
    silently otherwise (including when the token file is missing — that's
    a "not yet installed" condition the install flow handles, and when
    tokeninfo is unreachable, since a transient blip should not block the
    daemon).
    """
    if not download_attachments:
        return
    token_path = token_path_for_account(account)
    if not token_path.exists():
        # No token yet — install flow will request the right scopes.
        return
    scopes = _token_scopes(token_path)
    if DRIVE_SCOPE not in scopes:
        raise ReauthRequiredError(
            f"Calendar token for account={account!r} lacks {DRIVE_SCOPE!r}; "
            f"delete {redact_home_path(str(token_path))} and re-run the "
            f"install flow with download_attachments=true to grant it."
        )
    _verify_remote_drive_scope(account, token_path)
