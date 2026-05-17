"""GasTown integration helpers for fieldnotes MCP server.

Detects whether fieldnotes is running inside a GasTown rig and
configures the MCP server entry in the rig's ``.mcp.json`` so that
all polecats inherit access to fieldnotes search tools.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

from qdrant_client import QdrantClient

from worker.config import Config, load_config
from worker.neo4j_driver import build_driver

logger = logging.getLogger(__name__)


def detect_rig_root() -> Path | None:
    """Return the GasTown rig root, or *None* if not inside a rig.

    Detection heuristic (in priority order):
    1. ``GT_RIG_ROOT`` environment variable (set by ``gt`` tooling).
    2. Walk up from cwd looking for a directory that contains both a
       ``polecats/`` subdirectory and a ``.beads/`` directory — the
       canonical rig layout.
    """
    env_root = os.environ.get("GT_RIG_ROOT")
    if env_root:
        p = Path(env_root)
        if p.is_dir():
            logger.debug("Rig root from GT_RIG_ROOT: %s", p)
            return p

    cwd = Path.cwd()
    for directory in (cwd, *cwd.parents):
        if (directory / "polecats").is_dir() and (directory / ".beads").is_dir():
            logger.debug("Rig root detected at: %s", directory)
            return directory
        # Stop at filesystem root.
        if directory == directory.parent:
            break

    return None


def _fieldnotes_command(config_path: Path | None = None) -> list[str]:
    """Build the command list to start the fieldnotes MCP server."""
    fieldnotes_bin = shutil.which("fieldnotes")
    if fieldnotes_bin is None:
        # Fall back to module invocation.
        cmd = ["python", "-m", "worker.cli"]
    else:
        cmd = [fieldnotes_bin]

    if config_path is not None:
        cmd.extend(["-c", str(config_path)])

    cmd.extend(["serve", "--mcp"])
    return cmd


def write_mcp_config(
    rig_root: Path,
    *,
    config_path: Path | None = None,
) -> Path:
    """Write or update ``.mcp.json`` in *rig_root* with a fieldnotes entry.

    Returns the path to the written file.
    """
    mcp_json_path = rig_root / ".mcp.json"

    existing: dict = {}
    if mcp_json_path.exists():
        try:
            existing = json.loads(mcp_json_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not parse existing .mcp.json, overwriting: %s", exc)

    cmd = _fieldnotes_command(config_path)
    servers = existing.setdefault("mcpServers", {})
    servers["fieldnotes"] = {
        "command": cmd[0],
        "args": cmd[1:],
    }

    mcp_json_path.write_text(json.dumps(existing, indent=2) + "\n")
    logger.info("Wrote MCP config to %s", mcp_json_path)
    return mcp_json_path


def validate_connectivity(
    cfg: Config,
) -> dict[str, str]:
    """Validate Neo4j and Qdrant connectivity.  Returns health dict."""
    health: dict[str, str] = {}

    # Neo4j
    try:
        driver = build_driver(cfg.neo4j.uri, cfg.neo4j.user, cfg.neo4j.password)
        try:
            driver.verify_connectivity()
            health["neo4j"] = "ok"
        finally:
            driver.close()
    except Exception as exc:
        health["neo4j"] = f"error: {type(exc).__name__}"

    # Qdrant
    try:
        client = QdrantClient(host=cfg.qdrant.host, port=cfg.qdrant.port)
        try:
            client.get_collection(cfg.qdrant.collection)
            health["qdrant"] = "ok"
        finally:
            client.close()
    except Exception as exc:
        health["qdrant"] = f"error: {type(exc).__name__}"

    return health


def setup_gastown(
    *,
    config_path: Path | None = None,
    rig_root: Path | None = None,
) -> int:
    """Run the full GasTown setup flow.  Returns exit code (0 = ok)."""
    # 1. Detect rig.
    if rig_root is None:
        rig_root = detect_rig_root()

    if rig_root is None:
        print("error: not inside a GasTown rig")
        print("  Set GT_RIG_ROOT or run from within a rig directory.")
        return 1

    print(f"Detected rig root: {rig_root}")

    # 2. Load config and validate connectivity.
    try:
        cfg = load_config(config_path)
    except Exception as exc:
        print(f"error: failed to load config: {exc}")
        return 1

    print("Validating Neo4j / Qdrant connectivity...")
    health = validate_connectivity(cfg)

    for service, status in health.items():
        symbol = "ok" if status == "ok" else "FAIL"
        print(f"  {service}: {symbol}")
        if status != "ok":
            print(f"    {status}")

    if any(s != "ok" for s in health.values()):
        print("\nwarning: some services are unreachable (MCP config written anyway)")

    # 3. Write MCP config.
    mcp_path = write_mcp_config(rig_root, config_path=config_path)
    print(f"MCP config written to: {mcp_path}")
    print("\nPolecats in this rig can now use fieldnotes tools via MCP.")

    return 0
