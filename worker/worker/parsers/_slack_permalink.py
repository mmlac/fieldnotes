"""Slack web permalink → slack:// source_id resolver.

Slack web permalinks look like:
    https://{workspace}.slack.com/archives/{channel_id}/p{ts_packed}

The canonical source_id form used by fieldnotes is:
    slack://{team_id}/{channel_id}/{ts}

Resolving requires a workspace→team_id map populated by the Slack source
at startup (strategy A).  When the map is empty or the workspace is unknown
the fallback is the empty-team form ``slack:///{channel_id}/{ts}`` so
the reference is still captured even if team_id resolution is unavailable.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_WORKSPACE_MAP_PATH: Path = (
    Path.home() / ".fieldnotes" / "data" / "slack_workspace_map.json"
)

# Matches Slack web permalink URLs (case-insensitive for subdomain robustness).
# Groups: workspace, channel, ts_packed.  Optional query string is consumed
# so trailing characters (e.g. thread_ts param) don't affect the match.
_SLACK_PERMALINK_RE = re.compile(
    r"https://(?P<workspace>[\w-]+)\.slack\.com/archives/"
    r"(?P<channel>[A-Z0-9]+)/p(?P<ts_packed>\d+)"
    r"(?:\?[^\s\]>)\"']*)?",
    re.IGNORECASE,
)


def ts_packed_to_ts(ts_packed: str) -> str:
    """Convert a packed Slack URL ts to canonical dotted form.

    Slack's permalink encodes the ts as a plain integer (microseconds
    since epoch, no decimal point) by dropping the dot:
        1715800000.123456  →  1715800000123456

    The last six digits become the fractional part; the remainder is the
    integer seconds.  Leading zeros in the fractional part are preserved.
    """
    if len(ts_packed) <= 6:
        return ts_packed
    return f"{ts_packed[:-6]}.{ts_packed[-6:]}"


def load_workspace_team_map(path: Path | None = None) -> dict[str, str]:
    """Load the workspace→team_id cache from disk.

    Returns an empty dict on any error so callers degrade gracefully.
    """
    p = path or DEFAULT_WORKSPACE_MAP_PATH
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.debug("Cannot read Slack workspace map at %s: %s", p, exc)
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items() if k and v}


def save_workspace_team_map(
    workspace: str,
    team_id: str,
    path: Path | None = None,
) -> None:
    """Persist a workspace→team_id entry to the map file (write-once MVP).

    Existing entries are preserved; missing parent directories are created.
    No-op when the entry is already current.  Errors are logged at debug
    level and silently ignored so a filesystem hiccup never crashes ingest.
    """
    if not workspace or not team_id:
        return
    p = path or DEFAULT_WORKSPACE_MAP_PATH
    existing = load_workspace_team_map(p)
    if existing.get(workspace) == team_id:
        return
    existing[workspace] = team_id
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(existing, indent=2))
    except OSError as exc:
        logger.debug("Cannot write Slack workspace map to %s: %s", p, exc)


def slack_permalink_to_source_id(
    url: str,
    workspace_map: dict[str, str],
) -> str | None:
    """Resolve a Slack web permalink URL to a canonical slack:// source_id.

    Parameters
    ----------
    url:
        A Slack web permalink, e.g.
        ``https://terra2.slack.com/archives/C09ABCDEF/p1715800000123456``.
        Optional query parameters (``?thread_ts=...&cid=...``) are ignored.
    workspace_map:
        A workspace-subdomain → team_id dict,
        e.g. ``{"terra2": "T012XYZ"}``.

    Returns
    -------
    str | None
        ``slack://{team_id}/{channel}/{ts}`` when the workspace is known,
        ``slack:///{channel}/{ts}`` as a fallback when the team_id cannot be
        resolved (single-workspace setups still get a usable reference).
        ``None`` only when the URL does not match the Slack permalink pattern.
    """
    m = _SLACK_PERMALINK_RE.match(url)
    if m is None:
        return None
    workspace = m.group("workspace").lower()
    channel = m.group("channel")
    ts_packed = m.group("ts_packed")
    ts = ts_packed_to_ts(ts_packed)
    team_id = workspace_map.get(workspace, "")
    if not team_id:
        logger.debug(
            "No team_id for Slack workspace %r; using fallback source_id "
            "(cross-workspace references will not merge perfectly)",
            workspace,
        )
        return f"slack:///{channel}/{ts}"
    return f"slack://{team_id}/{channel}/{ts}"
