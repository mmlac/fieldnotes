"""Backup and restore for fieldnotes data and configuration."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from string import Template

_FN_DIR = Path.home() / ".fieldnotes"
_BACKUPS_DIR = _FN_DIR / "backups"

# Paths relative to _FN_DIR that are included in every backup.
_BACKUP_ITEMS = [
    "config.toml",
    "credentials.json",
    "data",
    "state",
]

_TEMPLATES = Path(__file__).resolve().parent / "templates"

PLIST_LABEL = "com.fieldnotes.backup"


def _fieldnotes_executable() -> list[str]:
    """Return the command needed to invoke ``fieldnotes``."""
    exe = shutil.which("fieldnotes")
    if exe:
        return [exe]
    return [sys.executable, "-m", "worker.cli"]


def _docker_running() -> bool:
    """Return True if Docker containers are running for fieldnotes."""
    if not shutil.which("docker"):
        return False
    infra_compose = _FN_DIR / "infrastructure" / "docker-compose.yml"
    if not infra_compose.exists():
        return False
    result = subprocess.run(
        ["docker", "compose", "-f", str(infra_compose), "ps", "-q"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def _stop_containers() -> bool:
    """Stop Docker containers if running. Returns whether they were running."""
    if not _docker_running():
        return False
    from worker.infra import infra_stop

    infra_stop()
    return True


def _start_containers() -> None:
    """Restart Docker containers."""
    from worker.infra import infra_up

    infra_up()


def _prune_backups(keep: int) -> None:
    """Delete oldest backups so that only *keep* most-recent remain."""
    if keep < 1:
        raise ValueError("keep must be at least 1")
    archives = sorted(_BACKUPS_DIR.glob("fieldnotes-*.tar.gz"))
    to_delete = archives[: len(archives) - keep]
    for path in to_delete:
        path.unlink()
        print(f"Pruned old backup: {path.name}")


# ── backup ──────────────────────────────────────────────────────────


def backup(*, keep: int | None = None) -> int:
    """Create a compressed backup of all fieldnotes data.

    Stops Docker containers for consistency, archives data and config
    to ``~/.fieldnotes/backups/``, then restarts containers.

    If *keep* is set, only the most recent *keep* backups are retained.

    Returns an exit code (0 = success).
    """
    if not _FN_DIR.exists():
        print(
            "~/.fieldnotes/ does not exist — nothing to back up.\n"
            "Run 'fieldnotes init' first.",
            file=sys.stderr,
        )
        return 1

    _BACKUPS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S")
    archive_name = f"fieldnotes-{timestamp}.tar.gz"
    archive_path = _BACKUPS_DIR / archive_name

    # Stop containers for a consistent snapshot.
    was_running = _stop_containers()
    if was_running:
        print("Stopped Docker containers for consistent backup.")

    try:
        with tarfile.open(archive_path, "w:gz") as tar:
            for item_name in _BACKUP_ITEMS:
                item_path = _FN_DIR / item_name
                if item_path.exists():
                    tar.add(str(item_path), arcname=item_name)
            # Include .env from infrastructure if present.
            env_file = _FN_DIR / "infrastructure" / ".env"
            if env_file.exists():
                tar.add(str(env_file), arcname="infrastructure/.env")

        size_mb = archive_path.stat().st_size / (1024 * 1024)
        print(f"Backup created: {archive_path} ({size_mb:.1f} MB)")

        if keep is not None:
            _prune_backups(keep)
    finally:
        if was_running:
            _start_containers()
            print("Restarted Docker containers.")

    return 0


# ── list ────────────────────────────────────────────────────────────


def list_backups() -> int:
    """List existing backups with timestamps and sizes.

    Returns an exit code (0 = success).
    """
    if not _BACKUPS_DIR.exists():
        print("No backups found.")
        return 0

    archives = sorted(_BACKUPS_DIR.glob("fieldnotes-*.tar.gz"))
    if not archives:
        print("No backups found.")
        return 0

    print(f"{'Backup':<40} {'Size':>10}  {'Created'}")
    print("-" * 70)
    for path in archives:
        size = path.stat().st_size
        if size >= 1024 * 1024:
            size_str = f"{size / (1024 * 1024):.1f} MB"
        elif size >= 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size} B"
        mtime = datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc
        ).strftime("%Y-%m-%d %H:%M UTC")
        print(f"{path.name:<40} {size_str:>10}  {mtime}")

    return 0


# ── restore ─────────────────────────────────────────────────────────


def restore(backup_path: Path) -> int:
    """Restore fieldnotes data from a backup archive.

    Stops Docker containers, extracts the archive over the existing
    data directory, then restarts containers.

    Returns an exit code (0 = success).
    """
    if not backup_path.exists():
        # Check if it's just a filename in the backups directory.
        candidate = _BACKUPS_DIR / backup_path
        if candidate.exists():
            backup_path = candidate
        else:
            print(f"Backup not found: {backup_path}", file=sys.stderr)
            return 1

    if not tarfile.is_tarfile(backup_path):
        print(f"Not a valid tar archive: {backup_path}", file=sys.stderr)
        return 1

    # Safety: verify archive only contains expected paths.
    with tarfile.open(backup_path, "r:gz") as tar:
        for member in tar.getmembers():
            # Block symlinks, hardlinks, and device files.
            if member.issym() or member.islnk():
                print(
                    f"Refusing to extract link: {member.name}",
                    file=sys.stderr,
                )
                return 1
            if not (member.isfile() or member.isdir()):
                print(
                    f"Refusing to extract special entry: {member.name}",
                    file=sys.stderr,
                )
                return 1
            # Block path-traversal attacks via resolved path check.
            resolved = (_FN_DIR / member.name).resolve()
            if not str(resolved).startswith(str(_FN_DIR.resolve())):
                print(
                    f"Refusing to extract unsafe path: {member.name}",
                    file=sys.stderr,
                )
                return 1

    was_running = _stop_containers()
    if was_running:
        print("Stopped Docker containers for restore.")

    try:
        with tarfile.open(backup_path, "r:gz") as tar:
            tar.extractall(path=str(_FN_DIR))  # noqa: S202
        print(f"Restored from {backup_path.name}")
    finally:
        if was_running:
            _start_containers()
            print("Restarted Docker containers.")

    return 0


# ── schedule ────────────────────────────────────────────────────────


def schedule_backup(*, remove: bool = False, keep: int | None = None) -> int:
    """Install or remove a daily scheduled backup.

    On macOS uses launchd; on Linux uses a systemd user timer.
    If *keep* is set, the scheduled command will include ``--keep``.

    Returns an exit code (0 = success).
    """
    system = platform.system()
    if system == "Darwin":
        return _schedule_launchd(remove=remove, keep=keep)
    if system == "Linux":
        return _schedule_systemd(remove=remove, keep=keep)
    print(f"Scheduled backups not supported on {system}.", file=sys.stderr)
    return 1


def _render_template(name: str, variables: dict[str, str]) -> str:
    target = (_TEMPLATES / name).resolve()
    if not str(target).startswith(str(_TEMPLATES.resolve())):
        raise ValueError(f"Invalid template name: {name}")
    raw = target.read_text()
    return Template(raw.replace("{{", "${").replace("}}", "}")).substitute(variables)


def _schedule_launchd(*, remove: bool, keep: int | None = None) -> int:
    """Install/remove a launchd calendar-interval job for daily backups."""
    plist_dir = Path.home() / "Library" / "LaunchAgents"
    plist_path = plist_dir / f"{PLIST_LABEL}.plist"
    log_dir = _FN_DIR / "logs"

    if remove:
        if plist_path.exists():
            subprocess.run(
                ["launchctl", "unload", str(plist_path)], check=False
            )
            plist_path.unlink()
            print(f"Removed scheduled backup ({plist_path})")
        else:
            print("No scheduled backup found — nothing to remove.")
        return 0

    plist_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    exe_parts = _fieldnotes_executable()
    cmd_parts = [*exe_parts, "backup", "create"]
    if keep is not None:
        cmd_parts += ["--keep", str(keep)]
    exe_strings = "\n        ".join(
        f"<string>{part}</string>" for part in cmd_parts
    )
    content = _render_template(
        "com.fieldnotes.backup.plist",
        {
            "PROGRAM_ARGUMENTS": exe_strings,
            "LOG_PATH": str(log_dir / "backup.log"),
        },
    )
    plist_path.write_text(content)
    plist_path.chmod(0o600)

    subprocess.run(
        ["launchctl", "load", "-w", str(plist_path)], check=True
    )
    print(f"Scheduled daily backup (02:00 local time)")
    print(f"  Plist: {plist_path}")
    print(f"  Logs:  {log_dir / 'backup.log'}")
    return 0


def _schedule_systemd(*, remove: bool, keep: int | None = None) -> int:
    """Install/remove a systemd user timer for daily backups."""
    unit_dir = Path.home() / ".config" / "systemd" / "user"
    service_path = unit_dir / "fieldnotes-backup.service"
    timer_path = unit_dir / "fieldnotes-backup.timer"

    if remove:
        subprocess.run(
            ["systemctl", "--user", "disable", "--now", "fieldnotes-backup.timer"],
            check=False,
        )
        for p in (service_path, timer_path):
            if p.exists():
                p.unlink()
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
        print("Removed scheduled backup.")
        return 0

    unit_dir.mkdir(parents=True, exist_ok=True)
    log_dir = _FN_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    exe_parts = _fieldnotes_executable()
    cmd_parts = [*exe_parts, "backup", "create"]
    if keep is not None:
        cmd_parts += ["--keep", str(keep)]
    service_content = _render_template(
        "fieldnotes-backup.service",
        {"EXEC_START": " ".join(cmd_parts)},
    )
    timer_content = _render_template("fieldnotes-backup.timer", {})

    service_path.write_text(service_content)
    timer_path.write_text(timer_content)

    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(
        ["systemctl", "--user", "enable", "--now", "fieldnotes-backup.timer"],
        check=True,
    )
    print("Scheduled daily backup (02:00 local time)")
    print(f"  Check status: systemctl --user status fieldnotes-backup.timer")
    return 0
