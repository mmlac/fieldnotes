"""macOS launchd backend for fieldnotes service management."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from string import Template

from worker.infra import infra_stop, infra_up, wait_for_docker

PLIST_LABEL = "com.fieldnotes.daemon"

_TEMPLATES = Path(__file__).resolve().parent.parent / "templates"
_BIN_DIR = Path.home() / ".fieldnotes" / "bin"
_WRAPPER_PATH = _BIN_DIR / "fieldnotes-daemon-wrapper.sh"


def _fieldnotes_executable() -> list[str]:
    """Return the command parts needed to invoke ``fieldnotes``."""
    exe = shutil.which("fieldnotes")
    if exe:
        return [exe]
    return [sys.executable, "-m", "worker.cli"]


def _log_dir() -> Path:
    return Path.home() / ".fieldnotes" / "logs"


def _render_template(name: str, variables: dict[str, str]) -> str:
    target = (_TEMPLATES / name).resolve()
    if not str(target).startswith(str(_TEMPLATES.resolve())):
        raise ValueError(f"Invalid template name: {name}")
    raw = target.read_text()
    return Template(raw.replace("{{", "${").replace("}}", "}")).substitute(variables)


class LaunchdBackend:
    """Manage a launchd user agent."""

    def __init__(self) -> None:
        self._plist_dir = Path.home() / "Library" / "LaunchAgents"
        self._plist_path = self._plist_dir / f"{PLIST_LABEL}.plist"
        self._log_dir = _log_dir()
        self._log_path = self._log_dir / "daemon.log"
        self._wrapper_path = _WRAPPER_PATH

    def install(self) -> None:
        self._plist_dir.mkdir(parents=True, exist_ok=True)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        # Write the wrapper script that waits for Docker then starts infra + daemon.
        self._wrapper_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnotes_cmd = " ".join(_fieldnotes_executable())
        wrapper_content = _render_template(
            "fieldnotes-daemon-wrapper.sh",
            {"FIELDNOTES_CMD": fieldnotes_cmd},
        )
        self._wrapper_path.write_text(wrapper_content)
        self._wrapper_path.chmod(0o755)
        print(f"Wrote {self._wrapper_path}")

        exe_strings = f"<string>{self._wrapper_path}</string>"
        content = _render_template(
            "com.fieldnotes.daemon.plist",
            {
                "PROGRAM_ARGUMENTS": exe_strings,
                "LOG_PATH": str(self._log_path),
            },
        )
        self._plist_path.write_text(content)
        self._plist_path.chmod(0o600)
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

        if self._wrapper_path.exists():
            self._wrapper_path.unlink()

    def start(self) -> None:
        if not self._plist_path.exists():
            raise SystemExit(
                "error: service not installed — run 'fieldnotes service install' first"
            )
        wait_for_docker()
        infra_up()
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
        infra_stop()
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
