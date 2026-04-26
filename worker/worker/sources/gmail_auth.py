"""Gmail OAuth2 authentication and token persistence.

Handles the OAuth2 authorization flow for Gmail API access, persists
tokens to disk (one file per account), and automatically refreshes
expired credentials.
"""

from __future__ import annotations

import logging
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from worker.log_sanitizer import redact_home_path

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


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

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        logger.info("Refreshing expired Gmail token for account=%s", account)
        creds.refresh(Request())
    else:
        logger.info(
            "Starting Gmail OAuth2 authorization flow for account=%s", account
        )
        flow = InstalledAppFlow.from_client_secrets_file(
            str(client_secrets_path), SCOPES
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
        "Gmail token saved for account=%s at %s",
        account,
        redact_home_path(str(token_path)),
    )

    return creds
