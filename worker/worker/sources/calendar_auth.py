"""Google Calendar OAuth2 authentication and token persistence.

Reuses the same OAuth2 flow as Gmail but with Calendar-specific scopes.
If the Gmail credentials.json already has Calendar scopes enabled,
the same client secrets file can be shared.
"""

from __future__ import annotations

import logging
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from worker.log_sanitizer import redact_home_path

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/calendar.events.readonly"]

DEFAULT_TOKEN_PATH = Path.home() / ".fieldnotes" / "data" / "calendar_token.json"


def get_credentials(
    client_secrets_path: str | Path,
    token_path: str | Path = DEFAULT_TOKEN_PATH,
) -> Credentials:
    """Obtain valid Google Calendar OAuth2 credentials.

    Loads cached credentials from *token_path* if available, refreshes them
    when expired, or runs the interactive OAuth2 consent flow using
    *client_secrets_path* for initial authorization.

    Returns a ``google.oauth2.credentials.Credentials`` instance ready for
    use with the Calendar API.
    """
    token_path = Path(token_path)
    creds: Credentials | None = None

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        logger.info("Refreshing expired Calendar token")
        creds.refresh(Request())
    else:
        logger.info("Starting Calendar OAuth2 authorization flow")
        flow = InstalledAppFlow.from_client_secrets_file(
            str(client_secrets_path), SCOPES
        )
        creds = flow.run_local_server(port=0)

    # Persist for next run
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.touch(mode=0o600, exist_ok=True)
    token_path.write_text(creds.to_json())
    logger.info("Calendar token saved to %s", redact_home_path(str(token_path)))

    return creds
