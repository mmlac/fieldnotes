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

import logging
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from worker.log_sanitizer import redact_home_path

logger = logging.getLogger(__name__)

CALENDAR_SCOPE = "https://www.googleapis.com/auth/calendar.events.readonly"
DRIVE_SCOPE = "https://www.googleapis.com/auth/drive.readonly"

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

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), scopes)

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

    # Persist for next run.  Create with restrictive mode, then re-chmod
    # after write to enforce 0600 even if the file already existed with
    # looser perms.
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.touch(mode=0o600, exist_ok=True)
    token_path.write_text(creds.to_json())
    token_path.chmod(0o600)
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
    if not token_path.exists():
        return []
    try:
        import json

        data = json.loads(token_path.read_text())
    except (OSError, ValueError):
        return []
    raw = data.get("scopes")
    if isinstance(raw, list):
        return [str(s) for s in raw]
    if isinstance(raw, str):
        # Older tokens persisted scopes as a space-separated string.
        return raw.split()
    return []


def check_calendar_auth(
    account: str,
    *,
    download_attachments: bool = False,
) -> None:
    """Verify the persisted token covers the scopes the caller needs.

    Raises :class:`ReauthRequiredError` when *download_attachments* is True
    but the persisted token was issued without ``drive.readonly``.  Returns
    silently otherwise (including when the token file is missing — that's
    a "not yet installed" condition the install flow handles).
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
