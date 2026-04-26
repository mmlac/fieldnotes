"""``fieldnotes init`` — bootstrap ~/.fieldnotes/ with a default config."""

from __future__ import annotations

import getpass
import logging
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


def _prompt_path(prompt: str, default: str = "") -> str:
    """Prompt for a filesystem path with Tab completion.

    Enables ``readline`` filename completion while the prompt is active,
    then restores the previous completer so other prompts are unaffected.
    Handles ``~`` expansion for completion but returns the raw user input
    (with ``~`` preserved) so it can be written to config as-is.
    """
    try:
        import readline
    except ImportError:
        # readline unavailable (rare) — fall back to plain prompt
        return _prompt(prompt, default)

    # Save current readline state
    prev_completer = readline.get_completer()
    prev_delims = readline.get_completer_delims()

    def _path_completer(text: str, state: int) -> str | None:
        # Expand ~ for matching but keep it in the output
        expanded = os.path.expanduser(text)
        if os.path.isdir(expanded) and not expanded.endswith(os.sep):
            expanded += os.sep
        import glob

        matches = glob.glob(expanded + "*")
        # Re-insert ~ prefix if the user typed it
        if text.startswith("~") and not expanded.startswith("~"):
            home = os.path.expanduser("~")
            matches = [
                "~" + m[len(home):] if m.startswith(home) else m
                for m in matches
            ]
        # Append / to directories so the user can keep tabbing deeper
        matches = [
            m + os.sep if os.path.isdir(os.path.expanduser(m)) and not m.endswith(os.sep) else m
            for m in matches
        ]
        return matches[state] if state < len(matches) else None

    readline.set_completer(_path_completer)
    readline.set_completer_delims(" \t\n")
    # macOS ships libedit disguised as readline
    if "libedit" in (getattr(readline, "__doc__", None) or ""):
        readline.parse_and_bind("bind ^I rl_complete")
    else:
        readline.parse_and_bind("tab: complete")

    try:
        return _prompt(prompt, default)
    finally:
        readline.set_completer(prev_completer)
        readline.set_completer_delims(prev_delims)


def _prompt_choice(prompt: str, choices: list[str], default: str) -> str:
    """Prompt the user to pick from a list."""
    print(f"{prompt} ({'/'.join(choices)}) [{default}]: ", end="", flush=True)
    response = input().strip().lower()
    if response in choices:
        return response
    return default


def _prompt_yes_no(prompt: str, default: bool = False) -> bool:
    """Prompt with a yes/no question."""
    suffix = "[Y/n]" if default else "[y/N]"
    response = input(f"{prompt} {suffix}: ").strip().lower()
    if response in ("y", "yes"):
        return True
    if response in ("n", "no"):
        return False
    return default


def _prompt_multi_select(
    prompt: str,
    items: list[dict[str, str]],
    id_key: str,
    label_key: str,
    defaults: list[int] | None = None,
) -> list[str]:
    """Present numbered items and let the user pick by comma-separated numbers.

    *defaults* is a list of 1-based indices pre-selected when the user
    presses Enter without typing anything.
    """
    print(f"\n{prompt}")
    default_set = set(defaults or [])
    for i, item in enumerate(items, 1):
        marker = " *" if i in default_set else ""
        label = item[label_key]
        item_id = item[id_key]
        if label != item_id:
            print(f"  {i:>3}. {label} ({item_id}){marker}")
        else:
            print(f"  {i:>3}. {label}{marker}")

    if default_set:
        default_str = ",".join(str(i) for i in sorted(default_set))
        print("  (* = default)")
    else:
        default_str = "1"

    response = _prompt("Enter numbers (comma-separated)", default_str)
    selected: list[str] = []
    for part in response.split(","):
        part = part.strip()
        if part.isdigit():
            idx = int(part) - 1
            if 0 <= idx < len(items):
                item_id = items[idx][id_key]
                if item_id not in selected:
                    selected.append(item_id)
    return selected or [items[i - 1][id_key] for i in sorted(default_set) if 0 < i <= len(items)]


def _prompt_password(prompt: str, *, min_length: int = 0) -> str:
    """Prompt for a password without echoing.

    When *min_length* > 0 the user is re-prompted until the password meets
    the requirement (or they enter an empty string to skip).
    """
    while True:
        pw = getpass.getpass(f"{prompt}: ")
        if not pw or min_length <= 0 or len(pw) >= min_length:
            return pw
        print(f"  Password must be at least {min_length} characters.", file=sys.stderr)


logger = logging.getLogger(__name__)

# Well-known Gmail system labels in a sensible display order.
_GMAIL_SYSTEM_LABELS = [
    ("INBOX", "Inbox"),
    ("SENT", "Sent"),
    ("DRAFT", "Drafts"),
    ("SPAM", "Spam"),
    ("TRASH", "Trash"),
    ("STARRED", "Starred"),
    ("IMPORTANT", "Important"),
    ("CATEGORY_PERSONAL", "Category: Personal"),
    ("CATEGORY_SOCIAL", "Category: Social"),
    ("CATEGORY_PROMOTIONS", "Category: Promotions"),
    ("CATEGORY_UPDATES", "Category: Updates"),
    ("CATEGORY_FORUMS", "Category: Forums"),
]

# System label IDs we skip (internal UI labels, not useful for filtering).
_GMAIL_SKIP_LABELS = frozenset({
    "UNREAD", "CHAT", "CATEGORY_PERSONAL",
})


def _list_gmail_labels(client_secrets_path: Path) -> list[dict[str, str]]:
    """Authenticate with Gmail and return available labels.

    Returns a list of ``{"id": ..., "name": ...}`` dicts sorted with
    well-known system labels first, then user labels alphabetically.
    """
    from googleapiclient.discovery import build

    from worker.sources.gmail_auth import get_credentials

    creds = get_credentials(client_secrets_path, account="default")
    service = build("gmail", "v1", credentials=creds)
    result = service.users().labels().list(userId="me").execute()
    raw_labels = result.get("labels", [])

    system_order = {lid: i for i, (lid, _) in enumerate(_GMAIL_SYSTEM_LABELS)}
    system_names = dict(_GMAIL_SYSTEM_LABELS)

    system: list[dict[str, str]] = []
    user: list[dict[str, str]] = []

    for lb in raw_labels:
        lid = lb.get("id", "")
        if lid in _GMAIL_SKIP_LABELS:
            continue
        name = system_names.get(lid) or lb.get("name", lid)
        entry = {"id": lid, "name": name}
        if lb.get("type") == "system" and lid in system_order:
            system.append(entry)
        elif lb.get("type") != "system":
            user.append(entry)
        # skip unknown system labels

    system.sort(key=lambda e: system_order.get(e["id"], 999))
    user.sort(key=lambda e: e["name"].lower())
    return system + user


def _list_calendars(client_secrets_path: Path) -> list[dict[str, str]]:
    """Authenticate with Google Calendar and return available calendars.

    Returns a list of ``{"id": ..., "name": ...}`` dicts with the
    primary calendar first, then others sorted alphabetically.
    """
    from googleapiclient.discovery import build

    from worker.sources.calendar_auth import get_credentials

    creds = get_credentials(client_secrets_path, account="default")
    service = build("calendar", "v3", credentials=creds)
    result = service.calendarList().list().execute()
    raw = result.get("items", [])

    primary: list[dict[str, str]] = []
    others: list[dict[str, str]] = []
    for cal in raw:
        cal_id = cal.get("id", "")
        summary = cal.get("summary") or cal.get("summaryOverride") or cal_id
        entry = {"id": cal_id, "name": summary}
        if cal.get("primary"):
            # Show "primary" as the id for the primary calendar since
            # the Calendar API accepts both the real id and "primary".
            entry["id"] = "primary"
            primary.append(entry)
        else:
            others.append(entry)

    others.sort(key=lambda e: e["name"].lower())
    return primary + others


def _interactive_config(config_text: str) -> str:
    """Walk the user through key configuration choices."""
    print("\n── fieldnotes setup ──\n")

    # 1. Neo4j password
    neo4j_pw = os.environ.get("NEO4J_PASSWORD", "")
    if neo4j_pw:
        print("Neo4j password: using NEO4J_PASSWORD from environment")
    else:
        neo4j_pw = _prompt_password(
            "Neo4j password (will be written to config)", min_length=8,
        )
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
    watch = _prompt_path(
        "Documents directory to index",
        "~/Documents",
    )
    config_text = config_text.replace(
        'watch_paths = ["~/Documents"]',
        f'watch_paths = ["{watch}"]',
    )

    # 4. Obsidian vault
    default_vault = "~/obsidian-vault"
    vault = _prompt_path("Obsidian vault path (leave empty to skip)", default_vault)
    config_text = config_text.replace(
        f'vault_paths = ["{default_vault}"]',
        f'vault_paths = ["{vault}"]',
    )

    # 5. Gmail
    print()
    gmail_secrets_path: Path | None = None
    if _prompt_yes_no("Set up Gmail indexing?"):
        secrets_path = _prompt_path(
            "Path to Google Cloud OAuth credentials JSON",
            "~/.fieldnotes/credentials.json",
        )
        gmail_secrets_path = Path(secrets_path).expanduser().resolve()
        if not gmail_secrets_path.exists():
            print(
                f"  Credentials file not found at {gmail_secrets_path}\n"
                "  Download it from the Google Cloud Console (OAuth client ID → Desktop app).\n"
                "  See README → Gmail OAuth Setup for details.\n"
                "  Gmail will be configured but label selection will be skipped.",
            )
            config_text = _append_gmail_config(
                config_text, secrets_path, ["INBOX"],
            )
        else:
            print("  Authenticating with Gmail (this will open your browser)...")
            try:
                labels = _list_gmail_labels(gmail_secrets_path)
                if not labels:
                    print("  No labels found — defaulting to INBOX.")
                    config_text = _append_gmail_config(
                        config_text, secrets_path, ["INBOX"],
                    )
                else:
                    # Find the index of INBOX for the default selection
                    inbox_indices = [
                        i + 1
                        for i, lb in enumerate(labels)
                        if lb["id"] == "INBOX"
                    ]
                    selected = _prompt_multi_select(
                        "Select Gmail labels to index:",
                        labels,
                        id_key="id",
                        label_key="name",
                        defaults=inbox_indices or [1],
                    )
                    config_text = _append_gmail_config(
                        config_text, secrets_path, selected,
                    )
                    print(f"  ✓ Gmail configured with labels: {', '.join(selected)}")
            except Exception as exc:
                logger.debug("Gmail label listing failed", exc_info=True)
                print(f"  Could not list labels: {exc}")
                print("  Gmail will be configured with default label INBOX.")
                config_text = _append_gmail_config(
                    config_text, secrets_path, ["INBOX"],
                )

    # 6. Google Calendar
    print()
    if _prompt_yes_no("Set up Google Calendar indexing?"):
        if gmail_secrets_path is not None:
            secrets_path = str(gmail_secrets_path).replace(
                str(Path.home()), "~"
            )
        else:
            secrets_path = _prompt_path(
                "Path to Google Cloud OAuth credentials JSON",
                "~/.fieldnotes/credentials.json",
            )
        cal_secrets = Path(secrets_path).expanduser().resolve()
        if not cal_secrets.exists():
            print(
                f"  Credentials file not found at {cal_secrets}\n"
                "  Calendar will be configured but calendar selection will be skipped.",
            )
            config_text = _append_calendar_config(
                config_text, secrets_path, ["primary"],
            )
        else:
            print(
                "  Authenticating with Google Calendar (this will open your browser)..."
            )
            try:
                calendars = _list_calendars(cal_secrets)
                if not calendars:
                    print("  No calendars found — defaulting to primary.")
                    config_text = _append_calendar_config(
                        config_text, secrets_path, ["primary"],
                    )
                else:
                    primary_indices = [
                        i + 1
                        for i, c in enumerate(calendars)
                        if c["id"] == "primary"
                    ]
                    selected = _prompt_multi_select(
                        "Select calendars to index:",
                        calendars,
                        id_key="id",
                        label_key="name",
                        defaults=primary_indices or [1],
                    )
                    config_text = _append_calendar_config(
                        config_text, secrets_path, selected,
                    )
                    names = [
                        next(
                            (c["name"] for c in calendars if c["id"] == s), s
                        )
                        for s in selected
                    ]
                    print(f"  ✓ Calendar configured: {', '.join(names)}")
            except Exception as exc:
                logger.debug("Calendar listing failed", exc_info=True)
                print(f"  Could not list calendars: {exc}")
                print(
                    "  Calendar will be configured with default calendar (primary)."
                )
                config_text = _append_calendar_config(
                    config_text, secrets_path, ["primary"],
                )

    print()
    return config_text


def _append_gmail_config(
    config_text: str,
    credentials_path: str,
    labels: list[str],
) -> str:
    """Append a ``[sources.gmail]`` TOML section to *config_text*."""
    labels_toml = ", ".join(f'"{_escape_toml_string(lb)}"' for lb in labels)
    section = (
        "\n[sources.gmail]\n"
        f'client_secrets_path = "{_escape_toml_string(credentials_path)}"\n'
        f"label_filter = [{labels_toml}]\n"
        "poll_interval_seconds = 300\n"
        "max_initial_threads = 500\n"
    )
    return config_text + section


def _append_calendar_config(
    config_text: str,
    credentials_path: str,
    calendar_ids: list[str],
) -> str:
    """Append a ``[sources.google_calendar]`` TOML section to *config_text*."""
    ids_toml = ", ".join(f'"{_escape_toml_string(c)}"' for c in calendar_ids)
    section = (
        "\n[sources.google_calendar]\n"
        f'client_secrets_path = "{_escape_toml_string(credentials_path)}"\n'
        f"calendar_ids = [{ids_toml}]\n"
        "poll_interval_seconds = 300\n"
        "max_initial_days = 90\n"
    )
    return config_text + section


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


def _copy_infra_tree(
    src: resources.abc.Traversable,
    dst: Path,
    *,
    skip: set[str] | None = None,
) -> list[Path]:
    """Recursively copy *src* into *dst*, returning written paths.

    Files whose names appear in *skip* are not overwritten.
    """
    dst.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for item in src.iterdir():
        if skip and item.name in skip:
            continue
        target = dst / item.name
        if item.is_file():
            target.write_bytes(item.read_bytes())
            written.append(target)
        else:
            written.extend(_copy_infra_tree(item, target, skip=skip))
    return written


def _extract_infrastructure() -> Path:
    """Copy bundled Docker infrastructure files to ~/.fieldnotes/infrastructure/.

    Returns the infrastructure directory path.  Files are only written if the
    directory doesn't already exist so that user customisations are preserved.
    """
    if _INFRA_DIR.exists():
        print(f"Infrastructure directory already exists at {_INFRA_DIR}")
        return _INFRA_DIR

    infra_pkg = resources.files("worker").joinpath("infrastructure")
    _copy_infra_tree(infra_pkg, _INFRA_DIR)
    print(f"Extracted Docker infrastructure to {_INFRA_DIR}")
    return _INFRA_DIR


def update_infrastructure() -> int:
    """Overwrite infrastructure files from the bundled package.

    The ``.env`` file is preserved so credentials are not lost.
    Returns an exit code (0 = success).
    """
    if not _INFRA_DIR.exists():
        print(
            "Infrastructure directory does not exist. "
            "Run 'fieldnotes init --with-docker' first.",
            file=sys.stderr,
        )
        return 1

    infra_pkg = resources.files("worker").joinpath("infrastructure")
    written = _copy_infra_tree(infra_pkg, _INFRA_DIR, skip={".env"})
    for p in sorted(written):
        print(f"  updated {p.relative_to(_INFRA_DIR)}")
    print(f"Updated {len(written)} infrastructure file(s) in {_INFRA_DIR}")
    return 0


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
