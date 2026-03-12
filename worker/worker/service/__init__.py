"""Service management for fieldnotes background daemon.

Platform-specific backends live in ``launchd`` (macOS) and ``systemd`` (Linux).
"""

from __future__ import annotations

import platform
import subprocess
import sys

from worker.service.launchd import LaunchdBackend
from worker.service.systemd import SystemdBackend


def platform_backend() -> LaunchdBackend | SystemdBackend:
    """Return the service backend appropriate for the current platform."""
    system = platform.system()
    if system == "Darwin":
        return LaunchdBackend()
    if system == "Linux":
        return SystemdBackend()
    raise SystemExit(f"error: unsupported platform for service management: {system}")


def install() -> int:
    """Install and start the service. Returns exit code."""
    try:
        platform_backend().install()
        return 0
    except subprocess.CalledProcessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def uninstall() -> int:
    """Stop and remove the service. Returns exit code."""
    try:
        platform_backend().uninstall()
        return 0
    except subprocess.CalledProcessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def status() -> int:
    """Print service status. Returns exit code."""
    try:
        platform_backend().status()
        return 0
    except subprocess.CalledProcessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def start() -> int:
    """Start the service (must already be installed). Returns exit code."""
    try:
        platform_backend().start()
        return 0
    except subprocess.CalledProcessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def stop() -> int:
    """Stop the service. Returns exit code."""
    try:
        platform_backend().stop()
        return 0
    except subprocess.CalledProcessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
