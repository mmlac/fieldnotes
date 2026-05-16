"""Slack permalink URL → ``slack://`` source_id translator.

Converts web-form Slack permalinks
(``https://<workspace>.slack.com/archives/<channel>/p<ts>``)
to the canonical ``slack://{team_id}/{channel_id}/{ts}`` source_id used
throughout fieldnotes so the cross-reference machinery can emit
``REFERENCES`` edges to Slack messages.

The main obstacle: permalinks carry ``<workspace>`` (human subdomain
like ``terra2``) but the canonical source_id uses ``<team_id>``
(e.g., ``T012XYZ``).  This module persists the workspace→team_id
mapping in a small JSON file written by the Slack source at startup.

Strategy A (write-once cache):
  On first ingest, ``SlackSource`` calls :func:`persist_workspace_map`
  with the subdomain and team_id it learns from ``auth.test``.  The
  cache is written to ``~/.fieldnotes/data/slack_workspace_map.json``.
  :func:`resolve_slack_permalink` reads that file at call time.

Fallback:
  If the workspace is not in the cache, the function returns ``None``
  and logs a debug message.  No exception is thrown.  Callers that need
  a best-effort ID can pass ``allow_partial=True`` to get the channel-
  only form ``slack:///channel_id/ts`` (empty team_id segment).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_WORKSPACE_MAP_PATH = Path.home() / ".fieldnotes" / "data" / "slack_workspace_map.json"

# https://<workspace>.slack.com/archives/<channel>/p<ts>
# Optional trailing thread_ts parameter is captured in a named group for
# future use but is otherwise ignored.
_PERMALINK_RE = re.compile(
    r"https://(?P<workspace>[\w-]+)\.slack\.com"
    r"/archives/(?P<channel>[A-Z0-9]+)"
    r"/p(?P<ts_packed>\d+)"
    r"(?:[?&]thread_ts=(?P<thread_ts>[\d.]+))?"
)


def _ts_from_packed(ts_packed: str) -> str:
    """Convert a packed Slack ts to canonical ``{secs}.{micros}`` form.

    Slack URLs encode the timestamp without the dot and zero-pad the
    microsecond field to 6 digits.  For example, ``p1715800000123456``
    encodes ``1715800000.123456``.
    """
    if len(ts_packed) < 7:
        return ts_packed
    secs = ts_packed[:-6]
    micros = ts_packed[-6:]
    return f"{secs}.{micros}"


def load_workspace_map(path: Path = DEFAULT_WORKSPACE_MAP_PATH) -> dict[str, str]:
    """Return the ``{subdomain: team_id}`` mapping from the cache file.

    Returns an empty dict if the file does not exist or is malformed.
    """
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items() if k and v}
    except (json.JSONDecodeError, OSError) as exc:
        logger.debug("Failed to read workspace map at %s: %s", path, exc)
    return {}


def persist_workspace_map(
    subdomain: str,
    team_id: str,
    path: Path = DEFAULT_WORKSPACE_MAP_PATH,
) -> None:
    """Persist or update the ``subdomain → team_id`` entry in the cache.

    Safe to call multiple times; existing entries are preserved.  Silently
    skips when either argument is empty.
    """
    if not subdomain or not team_id:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_workspace_map(path)
    if existing.get(subdomain) == team_id:
        return  # nothing to write
    existing[subdomain] = team_id
    try:
        path.write_text(json.dumps(existing, indent=2))
    except OSError as exc:
        logger.warning("Failed to persist workspace map to %s: %s", path, exc)


def resolve_slack_permalink(
    url: str,
    workspace_map: dict[str, str] | None = None,
    *,
    allow_partial: bool = False,
    workspace_map_path: Path = DEFAULT_WORKSPACE_MAP_PATH,
) -> str | None:
    """Translate a Slack permalink URL to a canonical ``slack://`` source_id.

    Parameters
    ----------
    url:
        A Slack permalink such as
        ``https://terra2.slack.com/archives/C09ABCDEF/p1715800000123456``.
    workspace_map:
        Pre-loaded ``{subdomain: team_id}`` dict.  When *None* the cache
        file at *workspace_map_path* is read on every call.
    allow_partial:
        When ``True`` and the workspace can't be resolved, return
        ``slack:///channel_id/ts`` (empty team_id) instead of ``None``.
        Useful for single-workspace setups where the team_id is implicitly
        known from context.
    workspace_map_path:
        Path to the JSON cache file.  Defaults to
        ``~/.fieldnotes/data/slack_workspace_map.json``.

    Returns
    -------
    str or None:
        Canonical ``slack://{team_id}/{channel_id}/{ts}`` on success, a
        partial ``slack:///{channel_id}/{ts}`` if *allow_partial* and the
        team_id is unknown, or ``None`` if the URL doesn't match the Slack
        permalink pattern at all.
    """
    m = _PERMALINK_RE.search(url)
    if not m:
        return None

    workspace = m.group("workspace")
    channel = m.group("channel")
    ts = _ts_from_packed(m.group("ts_packed"))

    if workspace_map is None:
        workspace_map = load_workspace_map(workspace_map_path)

    team_id = workspace_map.get(workspace, "")
    if not team_id:
        logger.debug(
            "Slack workspace %r not in workspace map; cannot resolve team_id for %s",
            workspace,
            url,
        )
        if allow_partial:
            return f"slack:///{channel}/{ts}"
        return None

    return f"slack://{team_id}/{channel}/{ts}"


def find_slack_permalink_source_ids(
    text: str,
    workspace_map: dict[str, str],
) -> list[str]:
    """Return unique resolved ``slack://`` source_ids for all Slack permalinks in text.

    Scans *text* for all Slack permalink URLs and resolves each via
    *workspace_map*.  URLs whose workspace is not in the map are silently
    skipped (debug-logged by :func:`resolve_slack_permalink`).
    Deduplicates by resolved source_id so the same link appearing twice
    produces one entry.
    """
    seen: set[str] = set()
    results: list[str] = []
    for m in _PERMALINK_RE.finditer(text):
        source_id = resolve_slack_permalink(m.group(0), workspace_map)
        if source_id and source_id not in seen:
            seen.add(source_id)
            results.append(source_id)
    return results
