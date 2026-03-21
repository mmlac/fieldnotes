"""LLM-based description generator for macOS applications.

Generates short descriptions for non-Homebrew apps using LLM inference.
Descriptions are cached in ~/.fieldnotes/state/app_descriptions.json keyed
by bundle_id. Brew descriptions always take priority over LLM-generated ones.

Batch processing: apps without cached descriptions are collected and sent
to the LLM in batches of 10 to reduce call overhead.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from worker.models.base import CompletionRequest
from worker.models.resolver import ModelRegistry

logger = logging.getLogger(__name__)

DESCRIBE_ROLE = "describe"
FALLBACK_ROLE = "extract"
BATCH_SIZE = 10
LLM_TIMEOUT = 60.0

SYSTEM_PROMPT = """\
You are a macOS application identifier. For each application, generate a \
1-2 sentence description focusing on what it does and what it's used for.

If you don't recognize an application, respond with exactly \
"Unknown application" for that entry. Do not guess or hallucinate."""

DESCRIBE_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_descriptions",
        "description": "Submit generated descriptions for macOS applications.",
        "parameters": {
            "type": "object",
            "properties": {
                "descriptions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "bundle_id": {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "required": ["bundle_id", "description"],
                    },
                },
            },
            "required": ["descriptions"],
        },
    },
}

DEFAULT_CACHE_PATH = Path.home() / ".fieldnotes" / "state" / "app_descriptions.json"


@dataclass
class AppInfo:
    """Metadata for an app needing a description."""

    bundle_id: str
    display_name: str
    category: str
    version: str


class AppDescriptionCache:
    """Persistent JSON cache for LLM-generated app descriptions.

    Cache entries store the description and the app version at generation time.
    A cached "Unknown application" result is regenerated if the app name changes
    (via the display_name field stored alongside).
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or DEFAULT_CACHE_PATH
        self._data: dict[str, dict[str, str]] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text())
            except (json.JSONDecodeError, OSError):
                logger.warning(
                    "Corrupt app description cache at %s — starting fresh", self._path
                )
                self._data = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, indent=2))

    def get(self, bundle_id: str, display_name: str, version: str) -> str | None:
        """Return cached description or None if cache miss.

        Cache miss conditions:
        - No entry for this bundle_id
        - Major version bump (first component changed)
        - Entry is "Unknown application" but display_name changed
        """
        entry = self._data.get(bundle_id)
        if entry is None:
            return None

        cached_desc = entry.get("description", "")
        cached_version = entry.get("version", "")
        cached_name = entry.get("display_name", "")

        # Regenerate on major version bump
        if _major_version(version) != _major_version(cached_version):
            return None

        # Regenerate "Unknown application" if name changed
        if cached_desc == "Unknown application" and cached_name != display_name:
            return None

        return cached_desc

    def put(
        self, bundle_id: str, display_name: str, version: str, description: str
    ) -> None:
        """Store a description in the cache (does not persist until save)."""
        self._data[bundle_id] = {
            "description": description,
            "version": version,
            "display_name": display_name,
        }

    def save(self) -> None:
        """Persist cache to disk."""
        self._save()


def _major_version(version: str) -> str:
    """Extract major version component."""
    return version.split(".")[0] if version else ""


def describe_apps(
    apps: list[AppInfo],
    registry: ModelRegistry,
    cache: AppDescriptionCache,
) -> dict[str, str]:
    """Generate descriptions for apps missing from cache.

    Returns a mapping of bundle_id → description for all input apps
    (from cache or freshly generated). Persists new entries to cache.
    """
    results: dict[str, str] = {}
    need_llm: list[AppInfo] = []

    for app in apps:
        cached = cache.get(app.bundle_id, app.display_name, app.version)
        if cached is not None:
            results[app.bundle_id] = cached
        else:
            need_llm.append(app)

    if not need_llm:
        return results

    # Process in batches
    for i in range(0, len(need_llm), BATCH_SIZE):
        batch = need_llm[i : i + BATCH_SIZE]
        batch_results = _describe_batch(batch, registry)
        for app in batch:
            desc = batch_results.get(app.bundle_id)
            if desc:
                # LLM returned a real description (including "Unknown application")
                results[app.bundle_id] = desc
                cache.put(app.bundle_id, app.display_name, app.version, desc)
            else:
                # LLM failed entirely for this app — don't cache a false
                # "Unknown application" that would overwrite valid descriptions
                # on retry.  Leave the app out of results so callers know it
                # wasn't described.
                logger.warning(
                    "No LLM description returned for %s (%s) — skipping",
                    app.bundle_id,
                    app.display_name,
                )

    cache.save()
    logger.info(
        "Generated %d app descriptions (%d from cache, %d from LLM)",
        len(results),
        len(results) - len(need_llm),
        len(need_llm),
    )
    return results


def _describe_batch(
    apps: list[AppInfo],
    registry: ModelRegistry,
) -> dict[str, str]:
    """Send a batch of apps to the LLM for description generation."""
    # Build the user prompt listing all apps
    lines = []
    for app in apps:
        lines.append(
            f"- App name: {app.display_name}\n"
            f"  Bundle ID: {app.bundle_id}\n"
            f"  Category: {app.category or 'unknown'}\n"
            f"  Version: {app.version or 'unknown'}"
        )
    user_text = (
        "Generate a 1-2 sentence description for each of the following "
        "macOS applications. If you don't recognize an application, say "
        '"Unknown application" and nothing else for that entry.\n\n'
        + "\n\n".join(lines)
    )

    req = CompletionRequest(
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_text}],
        tools=[DESCRIBE_TOOL],
        temperature=0.0,
        timeout=LLM_TIMEOUT,
    )

    # Try describe role, fall back to extract role
    try:
        model = registry.for_role(DESCRIBE_ROLE)
    except KeyError:
        try:
            model = registry.for_role(FALLBACK_ROLE)
        except KeyError:
            logger.warning(
                "No model configured for '%s' or '%s' role — skipping app descriptions",
                DESCRIBE_ROLE,
                FALLBACK_ROLE,
            )
            return {}

    try:
        resp = model.complete(req, task="describe_apps")
    except Exception:
        logger.warning(
            "LLM call failed for app descriptions — skipping batch", exc_info=True
        )
        return {}

    return _parse_response(resp, apps)


def _parse_response(resp: Any, apps: list[AppInfo]) -> dict[str, str]:
    """Extract bundle_id → description mapping from LLM response."""
    raw_args: dict[str, Any] | None = None

    # Try tool_calls first
    if resp.tool_calls:
        for tc in resp.tool_calls:
            if tc["function"]["name"] == "submit_descriptions":
                args = tc["function"]["arguments"]
                if isinstance(args, str):
                    try:
                        raw_args = json.loads(args)
                    except json.JSONDecodeError:
                        pass
                else:
                    raw_args = args
                break

    # Fallback: try parsing text as JSON
    if raw_args is None and resp.text:
        try:
            raw_args = json.loads(resp.text)
        except json.JSONDecodeError:
            pass

    if not raw_args or "descriptions" not in raw_args:
        logger.warning("Could not parse LLM response for app descriptions")
        return {}

    result: dict[str, str] = {}
    valid_ids = {a.bundle_id for a in apps}
    for entry in raw_args["descriptions"]:
        bid = entry.get("bundle_id", "")
        desc = entry.get("description", "").strip()
        if bid in valid_ids and desc:
            result[bid] = desc

    return result
