"""Linux systemd backend for fieldnotes service management."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from string import Template

UNIT_NAME = "fieldnotes.service"

_TEMPLATES = Path(__file__).resolve().parent.parent / "templates"


def _fieldnotes_executable() -> list[str]:
    """Return the command parts needed to invoke ``fieldnotes``."""
    exe = shutil.which("fieldnotes")
    if exe:
        return [exe]
    return [sys.executable, "-m", "worker.cli"]


def _log_dir() -> Path:
    return Path.home() / ".fieldnotes" / "logs"


def _render_template(name: str, variables: dict[str, str]) -> str:
    raw = (_TEMPLATES / name).read_text()
    return Template(raw.replace("{{", "${").replace("}}", "}")).substitute(variables)


class SystemdBackend:
    """Manage a systemd user-level service (no root required)."""

    def __init__(self) -> None:
        self._unit_dir = (
            Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
            / "systemd"
            / "user"
        )
        self._unit_path = self._unit_dir / UNIT_NAME
        self._log_dir = _log_dir()

    def install(self) -> None:
        self._unit_dir.mkdir(parents=True, exist_ok=True)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        content = _render_template(
            "fieldnotes.service",
            {
                "EXECUTABLE": " ".join(_fieldnotes_executable()),
                "LOG_DIR": str(self._log_dir),
            },
        )
        self._unit_path.write_text(content)
        self._unit_path.chmod(0o644)
        print(f"Wrote {self._unit_path}")

        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        subprocess.run(
            ["systemctl", "--user", "enable", "--now", UNIT_NAME],
            check=True,
        )
        print("Service enabled and started via systemd")
        print(f"Logs: journalctl --user -u {UNIT_NAME}")

    def uninstall(self) -> None:
        subprocess.run(
            ["systemctl", "--user", "disable", "--now", UNIT_NAME],
            check=False,
        )
        if self._unit_path.exists():
            self._unit_path.unlink()
            subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
            print(f"Removed {self._unit_path}")
        else:
            print("Systemd unit not found — nothing to remove.")

    def start(self) -> None:
        if not self._unit_path.exists():
            raise SystemExit(
                "error: service not installed — run 'fieldnotes service install' first"
            )
        subprocess.run(
            ["systemctl", "--user", "start", UNIT_NAME],
            check=True,
        )
        print("Service started")

    def stop(self) -> None:
        subprocess.run(
            ["systemctl", "--user", "stop", UNIT_NAME],
            check=False,
        )
        print("Service stopped")

    def status(self) -> None:
        result = subprocess.run(
            ["systemctl", "--user", "status", UNIT_NAME],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(result.stdout)
        else:
            output = result.stdout.strip() or result.stderr.strip()
            if output:
                print(output)
            else:
                print("Service is not installed.")
