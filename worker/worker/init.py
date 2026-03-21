"""``fieldnotes init`` — bootstrap ~/.fieldnotes/ with a default config."""

from __future__ import annotations

import getpass
import os
import secrets
import shutil
import subprocess
import sys
from importlib import resources
from pathlib import Path


_FN_DIR = Path.home() / ".fieldnotes"
_CONFIG_PATH = _FN_DIR / "config.toml"
_DATA_DIR = _FN_DIR / "data"


def _ollama_available() -> bool:
    """Return True if the ``ollama`` binary is on PATH."""
    return shutil.which("ollama") is not None


def _prompt(prompt: str, default: str = "") -> str:
    """Prompt the user with a default value."""
    suffix = f" [{default}]" if default else ""
    response = input(f"{prompt}{suffix}: ").strip()
    return response or default


def _prompt_choice(prompt: str, choices: list[str], default: str) -> str:
    """Prompt the user to pick from a list."""
    print(f"{prompt} ({'/'.join(choices)}) [{default}]: ", end="", flush=True)
    response = input().strip().lower()
    if response in choices:
        return response
    return default


def _prompt_password(prompt: str) -> str:
    """Prompt for a password without echoing."""
    return getpass.getpass(f"{prompt}: ")


def _interactive_config(config_text: str) -> str:
    """Walk the user through key configuration choices."""
    print("\n── fieldnotes setup ──\n")

    # 1. Neo4j password
    neo4j_pw = os.environ.get("NEO4J_PASSWORD", "")
    if neo4j_pw:
        print("Neo4j password: using NEO4J_PASSWORD from environment")
    else:
        neo4j_pw = _prompt_password("Neo4j password (will be written to config)")
    if neo4j_pw:
        config_text = config_text.replace(
            'password = ""', f'password = "{neo4j_pw}"', 1
        )

    # 2. Model provider
    print()
    provider = _prompt_choice(
        "Primary model provider",
        ["ollama", "openai", "anthropic"],
        "ollama",
    )
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            api_key = _prompt_password("OpenAI API key")
        # Uncomment the OpenAI provider block
        config_text = config_text.replace(
            '# [modelproviders.openai]\n# type = "openai"\n# api_key = ""',
            f'[modelproviders.openai]\ntype = "openai"\napi_key = "{api_key}"',
        )
        # Update role bindings to use openai models
        config_text = config_text.replace(
            '[models.local_chat]\nprovider = "ollama"\nmodel = "llama3.2"',
            '[models.local_chat]\nprovider = "openai"\nmodel = "gpt-4o-mini"',
        )
    elif provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            api_key = _prompt_password("Anthropic API key")
        config_text = config_text.replace(
            '# [modelproviders.anthropic]\n# type = "anthropic"\n# api_key = ""',
            f'[modelproviders.anthropic]\ntype = "anthropic"\napi_key = "{api_key}"',
        )
        config_text = config_text.replace(
            '[models.local_chat]\nprovider = "ollama"\nmodel = "llama3.2"',
            '[models.local_chat]\nprovider = "anthropic"\n'
            'model = "claude-sonnet-4-20250514"',
        )

    # 3. Source watch paths
    print()
    watch = _prompt(
        "Documents directory to index",
        "~/Documents",
    )
    config_text = config_text.replace(
        'watch_paths = ["~/Documents"]',
        f'watch_paths = ["{watch}"]',
    )

    # 4. Obsidian vault
    default_vault = "~/obsidian-vault"
    vault = _prompt("Obsidian vault path (leave empty to skip)", default_vault)
    config_text = config_text.replace(
        f'vault_path = "{default_vault}"',
        f'vault_path = "{vault}"',
    )

    print()
    return config_text


def _generate_env_file(neo4j_pw: str) -> Path:
    """Write a .env file with required passwords for Docker Compose."""
    env_path = Path.cwd() / ".env"
    if env_path.exists():
        print(f".env already exists at {env_path} — skipping generation")
        return env_path

    grafana_pw = os.environ.get("GRAFANA_PASSWORD", "")
    if not grafana_pw:
        grafana_pw = secrets.token_urlsafe(16)

    if not neo4j_pw:
        neo4j_pw = secrets.token_urlsafe(16)

    lines = [
        f"NEO4J_PASSWORD={neo4j_pw}",
        f"GRAFANA_PASSWORD={grafana_pw}",
    ]
    env_path.write_text("\n".join(lines) + "\n")
    env_path.chmod(0o600)
    print(f"Created {env_path} (mode 0600)")
    return env_path


def init(
    *,
    with_docker: bool = False,
    non_interactive: bool = False,
) -> int:
    """Create ~/.fieldnotes/ and generate a default config.toml.

    Returns an exit code (0 = success).
    """
    # Create directory structure
    _FN_DIR.mkdir(parents=True, exist_ok=True)
    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    if _CONFIG_PATH.exists():
        print(f"Config already exists at {_CONFIG_PATH}")
        print("To regenerate, remove or rename the existing file first.")
        if not with_docker:
            return 0

    if not _CONFIG_PATH.exists():
        # Read bundled config template
        template = resources.files("worker").joinpath("config.toml.example")
        config_text = template.read_text(encoding="utf-8")

        interactive = sys.stdin.isatty() and sys.stdout.isatty() and not non_interactive

        if interactive:
            config_text = _interactive_config(config_text)
        else:
            # Non-interactive: inject NEO4J_PASSWORD from env if available
            neo4j_pw = os.environ.get("NEO4J_PASSWORD", "")
            if neo4j_pw:
                config_text = config_text.replace(
                    'password = ""', f'password = "{neo4j_pw}"', 1
                )

        _CONFIG_PATH.write_text(config_text)
        _CONFIG_PATH.chmod(0o600)
        print(f"Created {_CONFIG_PATH}")
        print(f"Created {_DATA_DIR}/")

        # Detect Ollama
        if _ollama_available():
            print("\nDetected ollama on PATH.")
            print("  Pull the default models with:")
            print("    ollama pull nomic-embed-text")
            print("    ollama pull llama3.2")
        else:
            print(
                "\nNote: ollama not found on PATH. Install it for local "
                "embeddings,\nor configure an alternative provider in "
                "config.toml.",
                file=sys.stderr,
            )

    # ── --with-docker ───────────────────────────────────────────────
    if with_docker:
        # Read neo4j password from the config we just wrote
        import tomllib

        raw = tomllib.loads(_CONFIG_PATH.read_text())
        neo4j_pw = raw.get("neo4j", {}).get("password", "")

        _generate_env_file(neo4j_pw)

        if shutil.which("docker"):
            print("\nStarting Docker infrastructure...")
            result = subprocess.run(
                ["docker", "compose", "up", "-d"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print("Docker services started successfully.")
            else:
                print(
                    f"docker compose failed (exit {result.returncode}):",
                    file=sys.stderr,
                )
                print(result.stderr, file=sys.stderr)
                return 1
        else:
            print(
                "\ndocker not found on PATH. Start infrastructure manually:\n"
                "  docker compose up -d",
                file=sys.stderr,
            )

    # ── Next steps ──────────────────────────────────────────────────
    print()
    if with_docker:
        print("Next step:")
        print("  fieldnotes serve --daemon")
    else:
        neo4j_pw = os.environ.get("NEO4J_PASSWORD", "")
        print("Next steps:")
        if not neo4j_pw:
            print(
                f"  1. Set your Neo4j password in {_CONFIG_PATH}\n"
                "     (or export NEO4J_PASSWORD before running the daemon)"
            )
            step = 2
        else:
            print("  \u2713 Neo4j password set from NEO4J_PASSWORD env var")
            step = 1
        print(
            f"  {step}. Start infrastructure:\n"
            "     export NEO4J_PASSWORD=<your-password>\n"
            "     export GRAFANA_PASSWORD=<your-password>\n"
            "     docker compose up -d"
        )
        print(f"  {step + 1}. Run: fieldnotes serve --daemon")
    print("  Run 'fieldnotes doctor' to verify your setup.")
    return 0
