"""Gmail OAuth2 authentication and token persistence.

Handles the OAuth2 authorization flow for Gmail API access, persists
tokens to disk (one file per account), and automatically refreshes
expired credentials.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import google.auth.exceptions
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from worker.log_sanitizer import redact_home_path
from worker.sources._token_io import read_token_safe, write_token_atomic

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


class ReauthRequiredError(RuntimeError):
    """Raised when the Gmail refresh token is revoked and no TTY is available.

    Typically surfaces when the OAuth app's Testing-mode refresh token expires
    (Google invalidates them after 7 days) and the daemon runs headless.  The
    user should delete the stale token file and re-run the install flow.
    """


def token_path_for_account(account: str) -> Path:
    """Derive the on-disk token path for an account label."""
    return Path.home() / ".fieldnotes" / "data" / f"gmail_token-{account}.json"


def get_credentials(
    client_secrets_path: str | Path,
    account: str,
) -> Credentials:
    """Obtain valid Gmail OAuth2 credentials for *account*.

    Loads cached credentials from the per-account token file if available,
    refreshes them when expired, or runs the interactive OAuth2 consent
    flow using *client_secrets_path* for initial authorization.

    Returns a ``google.oauth2.credentials.Credentials`` instance ready for
    use with the Gmail API.
    """
    token_path = token_path_for_account(account)
    creds: Credentials | None = None

    raw = read_token_safe(token_path)
    if raw is not None:
        creds = Credentials.from_authorized_user_info(json.loads(raw), SCOPES)

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        logger.info("Refreshing expired Gmail token for account=%s", account)
        try:
            creds.refresh(Request())
        except google.auth.exceptions.RefreshError as exc:
            logger.error(
                "Gmail token refresh failed for account=%s (%s); "
                "deleting stale token",
                account,
                exc,
            )
            try:
                token_path.unlink()
            except OSError:
                pass
            if sys.stdin.isatty():
                logger.info(
                    "Starting Gmail OAuth2 authorization flow for account=%s", account
                )
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(client_secrets_path), SCOPES
                )
                creds = flow.run_local_server(port=0)
            else:
                raise ReauthRequiredError(
                    f"Gmail OAuth2 refresh token for account={account!r} has been "
                    f"revoked (invalid_grant). Delete "
                    f"{redact_home_path(str(token_path))} and re-run the install "
                    f"flow to re-authorize."
                ) from exc
    else:
        logger.info("Starting Gmail OAuth2 authorization flow for account=%s", account)
        flow = InstalledAppFlow.from_client_secrets_file(
            str(client_secrets_path), SCOPES
        )
        creds = flow.run_local_server(port=0)

    # Persist for next run via the symlink-safe atomic helper: writes go to
    # a tmp file with O_NOFOLLOW + 0o600, then rename onto the final path.
    token_path.parent.mkdir(parents=True, exist_ok=True)
    write_token_atomic(token_path, creds.to_json())
    logger.info(
        "Gmail token saved for account=%s at %s",
        account,
        redact_home_path(str(token_path)),
    )

    return creds
