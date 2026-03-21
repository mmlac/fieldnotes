"""Docker Compose infrastructure management for fieldnotes."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


_FN_DIR = Path.home() / ".fieldnotes"
_INFRA_DIR = _FN_DIR / "infrastructure"
_DEFAULT_COMPOSE = _INFRA_DIR / "docker-compose.yml"
_DEFAULT_ENV = _INFRA_DIR / ".env"


def _resolve_compose(compose_file: Path | None) -> tuple[Path, Path]:
    """Return (compose_path, env_file) for the given or default compose file.

    Raises SystemExit if the compose file cannot be found.
    """
    if compose_file is not None:
        compose_path = compose_file.resolve()
        env_file = compose_path.parent / ".env"
    else:
        compose_path = _DEFAULT_COMPOSE
        env_file = _DEFAULT_ENV

    if not compose_path.exists():
        print(
            f"Compose file not found: {compose_path}\n"
            "Run 'fieldnotes init --with-docker' first, or pass --compose-file.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    return compose_path, env_file


def _compose_cmd(
    compose_path: Path,
    env_file: Path,
    *args: str,
) -> list[str]:
    """Build a docker compose command list."""
    cmd = ["docker", "compose", "-f", str(compose_path)]
    if env_file.exists():
        cmd += ["--env-file", str(env_file)]
    cmd += list(args)
    return cmd


def infra_up(compose_file: Path | None = None) -> int:
    """Start Docker infrastructure (``docker compose up -d``)."""
    if not shutil.which("docker"):
        print("docker not found on PATH.", file=sys.stderr)
        return 1

    compose_path, env_file = _resolve_compose(compose_file)
    cmd = _compose_cmd(compose_path, env_file, "up", "-d")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("Docker services started.")
    else:
        print(f"docker compose up failed (exit {result.returncode}):", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
    return result.returncode


def infra_stop(compose_file: Path | None = None) -> int:
    """Stop Docker containers without removing them (``docker compose stop``)."""
    if not shutil.which("docker"):
        print("docker not found on PATH.", file=sys.stderr)
        return 1

    compose_path, env_file = _resolve_compose(compose_file)
    cmd = _compose_cmd(compose_path, env_file, "stop")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("Docker services stopped.")
    else:
        print(
            f"docker compose stop failed (exit {result.returncode}):", file=sys.stderr
        )
        print(result.stderr, file=sys.stderr)
    return result.returncode


def infra_down(compose_file: Path | None = None) -> int:
    """Tear down Docker infrastructure (``docker compose down``)."""
    if not shutil.which("docker"):
        print("docker not found on PATH.", file=sys.stderr)
        return 1

    compose_path, env_file = _resolve_compose(compose_file)
    cmd = _compose_cmd(compose_path, env_file, "down")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("Docker services removed.")
    else:
        print(
            f"docker compose down failed (exit {result.returncode}):", file=sys.stderr
        )
        print(result.stderr, file=sys.stderr)
    return result.returncode
