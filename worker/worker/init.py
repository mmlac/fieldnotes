"""``fieldnotes init`` — bootstrap ~/.fieldnotes/ with a default config."""

from __future__ import annotations

import shutil
import sys
from importlib import resources
from pathlib import Path


_FN_DIR = Path.home() / ".fieldnotes"
_CONFIG_PATH = _FN_DIR / "config.toml"
_DATA_DIR = _FN_DIR / "data"


def _ollama_available() -> bool:
    """Return True if the ``ollama`` binary is on PATH."""
    return shutil.which("ollama") is not None


def init() -> int:
    """Create ~/.fieldnotes/ and generate a default config.toml.

    Returns an exit code (0 = success).
    """
    # Create directory structure
    _FN_DIR.mkdir(parents=True, exist_ok=True)
    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    if _CONFIG_PATH.exists():
        print(f"Config already exists at {_CONFIG_PATH}")
        print("To regenerate, remove or rename the existing file first.")
        return 0

    # Read bundled config template
    template = resources.files("worker").joinpath("config.toml.example")
    config_text = template.read_text(encoding="utf-8")

    _CONFIG_PATH.write_text(config_text)
    print(f"Created {_CONFIG_PATH}")
    print(f"Created {_DATA_DIR}/")

    # Detect Ollama
    if _ollama_available():
        print("Detected ollama on PATH.")
    else:
        print(
            "Note: ollama not found on PATH. Install it for local embeddings,\n"
            "or configure an alternative provider in config.toml.",
            file=sys.stderr,
        )

    print()
    print("Next steps:")
    print(f"  1. Edit {_CONFIG_PATH} with your settings")
    print("  2. Start Neo4j and Qdrant (or configure remote instances)")
    print("  3. Run: fieldnotes serve --daemon")
    return 0
