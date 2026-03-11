"""Daemon lifecycle management for fieldnotes.

Supports macOS (launchd) and Linux (systemd user units).  The public API is
three functions — ``install``, ``uninstall``, and ``status`` — plus a
``platform_backend`` helper that picks the right backend for the current OS.

Usage::

    fieldnotes daemon install    # install + start
    fieldnotes daemon uninstall  # stop + remove
    fieldnotes daemon status     # print daemon state
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from string import Template

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PLIST_LABEL = "com.fieldnotes.daemon"
_SYSTEMD_UNIT = "fieldnotes.service"

_TEMPLATES = Path(__file__).parent / "templates"


def _fieldnotes_executable() -> list[str]:
    """Return the command parts needed to invoke ``fieldnotes``."""
    exe = shutil.which("fieldnotes")
    if exe:
        return [exe]
    # Fall back to the running interpreter's entry point.
    return [sys.executable, "-m", "worker.cli"]


def _render_template(name: str, variables: dict[str, str]) -> str:
    """Read a template from the *templates* directory and substitute vars."""
    raw = (_TEMPLATES / name).read_text()
    return Template(raw.replace("{{", "${").replace("}}", "}")).substitute(variables)


# ---------------------------------------------------------------------------
# macOS / launchd
# ---------------------------------------------------------------------------


class _LaunchdBackend:
    """Manage a launchd user agent."""

    def __init__(self) -> None:
        self._plist_dir = Path.home() / "Library" / "LaunchAgents"
        self._plist_path = self._plist_dir / f"{_PLIST_LABEL}.plist"
        self._log_path = Path.home() / "Library" / "Logs" / "fieldnotes.log"

    def install(self) -> None:
        self._plist_dir.mkdir(parents=True, exist_ok=True)

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
        print(f"Wrote {self._plist_path}")

        subprocess.run(
            ["launchctl", "load", "-w", str(self._plist_path)],
            check=True,
        )
        print("Daemon loaded via launchctl")

    def uninstall(self) -> None:
        if self._plist_path.exists():
            subprocess.run(
                ["launchctl", "unload", str(self._plist_path)],
                check=False,
            )
            self._plist_path.unlink()
            print(f"Removed {self._plist_path}")
        else:
            print("Daemon plist not found — nothing to remove.")

    def start(self) -> None:
        if not self._plist_path.exists():
            raise SystemExit("error: daemon not installed — run 'fieldnotes daemon install' first")
        subprocess.run(
            ["launchctl", "load", "-w", str(self._plist_path)],
            check=True,
        )
        print("Daemon started")

    def stop(self) -> None:
        subprocess.run(
            ["launchctl", "unload", str(self._plist_path)],
            check=False,
        )
        print("Daemon stopped")

    def status(self) -> None:
        result = subprocess.run(
            ["launchctl", "list", _PLIST_LABEL],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(f"Daemon is loaded ({_PLIST_LABEL})")
            for line in result.stdout.strip().splitlines():
                print(f"  {line}")
        else:
            print("Daemon is not loaded.")

        if self._log_path.exists():
            print(f"Log file: {self._log_path}")


# ---------------------------------------------------------------------------
# Linux / systemd
# ---------------------------------------------------------------------------


class _SystemdBackend:
    """Manage a systemd user-level service (no root required)."""

    def __init__(self) -> None:
        self._unit_dir = (
            Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
            / "systemd"
            / "user"
        )
        self._unit_path = self._unit_dir / _SYSTEMD_UNIT

    def install(self) -> None:
        self._unit_dir.mkdir(parents=True, exist_ok=True)

        content = _render_template(
            "fieldnotes.service",
            {"EXECUTABLE": " ".join(_fieldnotes_executable())},
        )
        self._unit_path.write_text(content)
        print(f"Wrote {self._unit_path}")

        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        subprocess.run(
            ["systemctl", "--user", "enable", "--now", _SYSTEMD_UNIT],
            check=True,
        )
        print("Daemon enabled and started via systemd")

    def uninstall(self) -> None:
        subprocess.run(
            ["systemctl", "--user", "disable", "--now", _SYSTEMD_UNIT],
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
            raise SystemExit("error: daemon not installed — run 'fieldnotes daemon install' first")
        subprocess.run(
            ["systemctl", "--user", "start", _SYSTEMD_UNIT],
            check=True,
        )
        print("Daemon started")

    def stop(self) -> None:
        subprocess.run(
            ["systemctl", "--user", "stop", _SYSTEMD_UNIT],
            check=False,
        )
        print("Daemon stopped")

    def status(self) -> None:
        result = subprocess.run(
            ["systemctl", "--user", "status", _SYSTEMD_UNIT],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(result.stdout)
        else:
            # systemctl returns non-zero for inactive services too.
            output = result.stdout.strip() or result.stderr.strip()
            if output:
                print(output)
            else:
                print("Daemon is not installed.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def platform_backend() -> _LaunchdBackend | _SystemdBackend:
    """Return the daemon backend appropriate for the current platform."""
    system = platform.system()
    if system == "Darwin":
        return _LaunchdBackend()
    if system == "Linux":
        return _SystemdBackend()
    raise SystemExit(f"error: unsupported platform for daemon management: {system}")


def install() -> int:
    """Install and start the daemon. Returns exit code."""
    try:
        platform_backend().install()
        return 0
    except subprocess.CalledProcessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def uninstall() -> int:
    """Stop and remove the daemon. Returns exit code."""
    try:
        platform_backend().uninstall()
        return 0
    except subprocess.CalledProcessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def status() -> int:
    """Print daemon status. Returns exit code."""
    try:
        platform_backend().status()
        return 0
    except subprocess.CalledProcessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def start() -> int:
    """Start the daemon (must already be installed). Returns exit code."""
    try:
        platform_backend().start()
        return 0
    except subprocess.CalledProcessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def stop() -> int:
    """Stop the daemon. Returns exit code."""
    try:
        platform_backend().stop()
        return 0
    except subprocess.CalledProcessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
