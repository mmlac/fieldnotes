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
_INFRA_DIR = _FN_DIR / "infrastructure"


def _escape_toml_string(value: str) -> str:
    """Escape a value for safe embedding in a TOML basic (double-quoted) string."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


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
            'password = ""',
            f'password = "{_escape_toml_string(neo4j_pw)}"',
            1,
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
            f'[modelproviders.openai]\ntype = "openai"\n'
            f'api_key = "{_escape_toml_string(api_key)}"',
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
            f'[modelproviders.anthropic]\ntype = "anthropic"\n'
            f'api_key = "{_escape_toml_string(api_key)}"',
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


def _generate_env_file(neo4j_pw: str, env_dir: Path) -> Path:
    """Write a .env file with required passwords for Docker Compose."""
    env_path = env_dir / ".env"
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
    fd = os.open(str(env_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Created {env_path} (mode 0600)")
    return env_path


def _extract_infrastructure() -> Path:
    """Copy bundled Docker infrastructure files to ~/.fieldnotes/infrastructure/.

    Returns the infrastructure directory path.  Files are only written if the
    directory doesn't already exist so that user customisations are preserved.
    """
    if _INFRA_DIR.exists():
        print(f"Infrastructure directory already exists at {_INFRA_DIR}")
        return _INFRA_DIR

    infra_pkg = resources.files("worker").joinpath("infrastructure")

    def _copy_tree(src: resources.abc.Traversable, dst: Path) -> None:
        dst.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            target = dst / item.name
            if item.is_file():
                target.write_bytes(item.read_bytes())
            else:
                _copy_tree(item, target)

    _copy_tree(infra_pkg, _INFRA_DIR)
    print(f"Extracted Docker infrastructure to {_INFRA_DIR}")
    return _INFRA_DIR


def _ensure_data_dirs() -> None:
    """Create bind-mount data directories so Docker doesn't create them as root."""
    for subdir in ("neo4j", "qdrant", "prometheus", "grafana"):
        (_DATA_DIR / subdir).mkdir(parents=True, exist_ok=True)


def init(
    *,
    with_docker: bool = False,
    non_interactive: bool = False,
    compose_file: Path | None = None,
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
                    'password = ""',
                    f'password = "{_escape_toml_string(neo4j_pw)}"',
                    1,
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
    if compose_file is not None:
        with_docker = True
    if with_docker:
        # Read neo4j password from the config we just wrote
        import tomllib

        raw = tomllib.loads(_CONFIG_PATH.read_text())
        neo4j_pw = raw.get("neo4j", {}).get("password", "")

        # Determine which compose file to use
        if compose_file is not None:
            compose_path = compose_file.resolve()
            if not compose_path.exists():
                print(
                    f"Compose file not found: {compose_path}",
                    file=sys.stderr,
                )
                return 1
            env_dir = compose_path.parent
        else:
            # Extract bundled infrastructure to ~/.fieldnotes/infrastructure/
            infra_dir = _extract_infrastructure()
            compose_path = infra_dir / "docker-compose.yml"
            env_dir = infra_dir

        _ensure_data_dirs()
        _generate_env_file(neo4j_pw, env_dir)

        if shutil.which("docker"):
            print("\nStarting Docker infrastructure...")
            result = subprocess.run(
                [
                    "docker",
                    "compose",
                    "-f",
                    str(compose_path),
                    "--env-file",
                    str(env_dir / ".env"),
                    "up",
                    "-d",
                ],
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
                f"  docker compose -f {compose_path} up -d",
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
