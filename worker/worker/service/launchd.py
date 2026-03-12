"""macOS launchd backend for fieldnotes service management."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from string import Template

PLIST_LABEL = "com.fieldnotes.daemon"

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


class LaunchdBackend:
    """Manage a launchd user agent."""

    def __init__(self) -> None:
        self._plist_dir = Path.home() / "Library" / "LaunchAgents"
        self._plist_path = self._plist_dir / f"{PLIST_LABEL}.plist"
        self._log_dir = _log_dir()
        self._log_path = self._log_dir / "daemon.log"

    def install(self) -> None:
        self._plist_dir.mkdir(parents=True, exist_ok=True)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        exe_parts = _fieldnotes_executable()
        exe_strings = "\n        ".join(
            f"<string>{part}</string>" for part in [*exe_parts, "serve", "--daemon"]
        )
        content = _render_template(
            "com.fieldnotes.daemon.plist",
            {
                "PROGRAM_ARGUMENTS": exe_strings,
                "LOG_PATH": str(self._log_path),
            },
        )
        self._plist_path.write_text(content)
        self._plist_path.chmod(0o644)
        print(f"Wrote {self._plist_path}")

        subprocess.run(
            ["launchctl", "load", "-w", str(self._plist_path)],
            check=True,
        )
        print("Service loaded via launchctl")
        print(f"Logs: {self._log_path}")

    def uninstall(self) -> None:
        if self._plist_path.exists():
            subprocess.run(
                ["launchctl", "unload", str(self._plist_path)],
                check=False,
            )
            self._plist_path.unlink()
            print(f"Removed {self._plist_path}")
        else:
            print("Service plist not found — nothing to remove.")

    def start(self) -> None:
        if not self._plist_path.exists():
            raise SystemExit("error: service not installed — run 'fieldnotes service install' first")
        subprocess.run(
            ["launchctl", "load", "-w", str(self._plist_path)],
            check=True,
        )
        print("Service started")

    def stop(self) -> None:
        subprocess.run(
            ["launchctl", "unload", str(self._plist_path)],
            check=False,
        )
        print("Service stopped")

    def status(self) -> None:
        result = subprocess.run(
            ["launchctl", "list", PLIST_LABEL],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(f"Service is loaded ({PLIST_LABEL})")
            for line in result.stdout.strip().splitlines():
                print(f"  {line}")
        else:
            print("Service is not loaded.")

        if self._log_path.exists():
            print(f"Logs: {self._log_path}")
