"""Centralized log sanitization to prevent sensitive data leakage.

Provides utilities for redacting credentials, filesystem paths, and
other sensitive information from log output and exception tracebacks.
"""

from __future__ import annotations

import logging
import os
import re
from urllib.parse import urlparse, urlunparse


# Compiled once at import time.
_HOME_DIR: str = os.path.expanduser("~")

# Matches common secret patterns in exception messages / log strings.
_SECRET_PATTERNS: list[re.Pattern[str]] = [
    # password=..., token=..., api_key=..., secret=... in key=value contexts
    re.compile(
        r"(password|token|api_key|secret|authorization|credential)"
        r"(\s*[:=]\s*)"
        r"(\S+)",
        re.IGNORECASE,
    ),
]


def redact_uri(uri: str) -> str:
    """Strip credentials from a connection URI.

    ``bolt://neo4j:s3cret@host:7687`` → ``bolt://***@host:7687``

    Returns the URI unchanged if it contains no userinfo.
    """
    try:
        parsed = urlparse(uri)
    except Exception:
        return "<unparseable-uri>"
    if parsed.username or parsed.password:
        # Rebuild without credentials
        replaced = parsed._replace(
            netloc=f"***@{parsed.hostname}"
            + (f":{parsed.port}" if parsed.port else "")
        )
        return urlunparse(replaced)
    return uri


def redact_home_path(text: str) -> str:
    """Replace the user's home directory with ``~`` in *text*."""
    if _HOME_DIR and _HOME_DIR != "/":
        return text.replace(_HOME_DIR, "~")
    return text


def sanitize_exception(exc: BaseException) -> str:
    """Return a sanitized single-line representation of an exception.

    Redacts filesystem paths and known secret patterns from the
    exception string so it is safe for log output.
    """
    msg = str(exc)
    msg = redact_home_path(msg)
    for pat in _SECRET_PATTERNS:
        msg = pat.sub(r"\1\2<REDACTED>", msg)
    return msg


class SanitizingFormatter(logging.Formatter):
    """Log formatter that redacts home-directory paths from tracebacks.

    In production (non-DEBUG) mode, full filesystem paths in exception
    tracebacks are replaced with ``~/…`` relative paths.  At DEBUG level
    the original paths are preserved for developer convenience.
    """

    def __init__(
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
        production: bool = True,
    ) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)
        self._production = production

    def formatException(self, ei: tuple) -> str:  # noqa: N802 — overrides stdlib
        tb_text = super().formatException(ei)
        if self._production:
            tb_text = redact_home_path(tb_text)
            for pat in _SECRET_PATTERNS:
                tb_text = pat.sub(r"\1\2<REDACTED>", tb_text)
        return tb_text
