"""Configure fieldnotes as an MCP server for Claude Desktop.

Detects the Claude Desktop config file, adds the fieldnotes MCP server
entry, and prints instructions for restarting Claude Desktop.
"""

from __future__ import annotations

import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


def _default_config_path() -> Path:
    """Return the default Claude Desktop config path for the current OS."""
    system = platform.system()
    if system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    if system == "Windows":
        appdata = Path.home() / "AppData" / "Roaming" / "Claude"
        return appdata / "claude_desktop_config.json"
    # Linux / other — XDG convention
    xdg = Path.home() / ".config" / "claude"
    return xdg / "claude_desktop_config.json"


_MCP_ENTRY = {
    "command": "fieldnotes",
    "args": ["serve", "--mcp"],
    "env": {},
}


def _validate_fieldnotes_on_path() -> str | None:
    """Return the resolved path if ``fieldnotes`` is on PATH, else None."""
    return shutil.which("fieldnotes")


def setup_claude(config_path: Path | None = None) -> int:
    """Add fieldnotes MCP server to Claude Desktop config.

    Returns an exit code (0 = success).
    """
    # 1. Validate fieldnotes is on PATH
    fn_path = _validate_fieldnotes_on_path()
    if fn_path is None:
        print(
            "error: 'fieldnotes' not found on PATH.\n"
            "Install it first (e.g. `pipx install fieldnotes` or "
            "`uv tool install fieldnotes`).",
            file=sys.stderr,
        )
        return 1

    # 2. Resolve config location
    cfg_path = config_path or _default_config_path()

    # 3. Load existing config (or start fresh)
    if cfg_path.exists():
        try:
            existing = json.loads(cfg_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            print(f"error: cannot read {cfg_path}: {exc}", file=sys.stderr)
            return 1

        if not isinstance(existing, dict):
            print(f"error: {cfg_path} is not a JSON object", file=sys.stderr)
            return 1

        # Check if already configured
        servers = existing.get("mcpServers", {})
        if "fieldnotes" in servers:
            print(f"fieldnotes MCP server is already configured in {cfg_path}")
            return 0

        # 4. Back up existing config
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_path = cfg_path.with_suffix(f".backup-{ts}.json")
        try:
            shutil.copy2(cfg_path, backup_path)
            print(f"Backed up existing config to {backup_path}")
        except OSError as exc:
            print(f"error: cannot back up {cfg_path}: {exc}", file=sys.stderr)
            return 1
    else:
        existing = {}

    # 5. Add fieldnotes MCP server entry
    if "mcpServers" not in existing:
        existing["mcpServers"] = {}
    existing["mcpServers"]["fieldnotes"] = _MCP_ENTRY

    # 6. Write updated config
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(existing, indent=2) + "\n")
    cfg_path.chmod(0o600)

    print(f"Added fieldnotes MCP server to {cfg_path}")
    print()
    print("Restart Claude Desktop to activate the fieldnotes tools.")
    print(f"Using fieldnotes at: {fn_path}")
    return 0
