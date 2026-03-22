"""Tests for backup, restore, and related CLI commands."""

from __future__ import annotations

import tarfile
from pathlib import Path
from unittest.mock import patch

import pytest

from worker.backup import (
    _BACKUP_ITEMS,
    _BACKUPS_DIR,
    _FN_DIR,
    _prune_backups,
    backup,
    list_backups,
    restore,
)
from worker.cli import _build_parser, main


# ── helpers ─────────────────────────────────────────────────────────


def _populate_fieldnotes(root: Path) -> None:
    """Create a minimal ~/.fieldnotes layout under *root*."""
    (root / "config.toml").write_text("[general]\n")
    (root / "credentials.json").write_text("{}")
    data = root / "data"
    data.mkdir()
    (data / "neo4j").mkdir()
    (data / "neo4j" / "dummy.db").write_bytes(b"\x00" * 64)
    (data / "qdrant").mkdir()
    (data / "qdrant" / "collection").write_bytes(b"\x01" * 32)
    state = root / "state"
    state.mkdir()
    (state / "cursors.json").write_text("{}")
    infra = root / "infrastructure"
    infra.mkdir()
    (infra / ".env").write_text("NEO4J_PASSWORD=secret\n")


def _create_test_archive(backups_dir: Path, fn_dir: Path, name: str) -> Path:
    """Create a valid backup archive for testing restore."""
    backups_dir.mkdir(parents=True, exist_ok=True)
    archive = backups_dir / name
    with tarfile.open(archive, "w:gz") as tar:
        for item in _BACKUP_ITEMS:
            p = fn_dir / item
            if p.exists():
                tar.add(str(p), arcname=item)
        env = fn_dir / "infrastructure" / ".env"
        if env.exists():
            tar.add(str(env), arcname="infrastructure/.env")
    return archive


# ── parser tests ────────────────────────────────────────────────────


class TestBackupParser:
    def test_backup_no_subcommand(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["backup"])
        assert args.command == "backup"
        assert args.backup_command is None

    def test_backup_create(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["backup", "create"])
        assert args.command == "backup"
        assert args.backup_command == "create"

    def test_backup_list(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["backup", "list"])
        assert args.command == "backup"
        assert args.backup_command == "list"

    def test_backup_schedule(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["backup", "schedule"])
        assert args.command == "backup"
        assert args.backup_command == "schedule"
        assert args.remove is False

    def test_backup_schedule_remove(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["backup", "schedule", "--remove"])
        assert args.remove is True

    def test_restore(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["restore", "fieldnotes-20260320-120000.tar.gz"])
        assert args.command == "restore"
        assert args.backup_name == "fieldnotes-20260320-120000.tar.gz"

    def test_backup_keep_on_create(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["backup", "create", "--keep", "5"])
        assert args.keep == 5

    def test_backup_keep_on_bare(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["backup", "--keep", "3"])
        assert args.keep == 3
        assert args.backup_command is None

    def test_schedule_keep(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["backup", "schedule", "--keep", "7"])
        assert args.keep == 7
        assert args.backup_command == "schedule"


# ── backup() tests ──────────────────────────────────────────────────


class TestBackup:
    @patch("worker.backup._stop_containers", return_value=False)
    def test_backup_creates_archive(
        self, _stop: object, tmp_path: Path
    ) -> None:
        fn = tmp_path / ".fieldnotes"
        fn.mkdir()
        _populate_fieldnotes(fn)

        with (
            patch("worker.backup._FN_DIR", fn),
            patch("worker.backup._BACKUPS_DIR", fn / "backups"),
        ):
            rc = backup()

        assert rc == 0
        archives = list((fn / "backups").glob("fieldnotes-*.tar.gz"))
        assert len(archives) == 1

        with tarfile.open(archives[0], "r:gz") as tar:
            names = tar.getnames()
        assert "config.toml" in names
        assert "data/neo4j/dummy.db" in names

    @patch("worker.backup._stop_containers", return_value=True)
    @patch("worker.backup._start_containers")
    def test_backup_restarts_containers(
        self, start: object, _stop: object, tmp_path: Path
    ) -> None:
        fn = tmp_path / ".fieldnotes"
        fn.mkdir()
        _populate_fieldnotes(fn)

        with (
            patch("worker.backup._FN_DIR", fn),
            patch("worker.backup._BACKUPS_DIR", fn / "backups"),
        ):
            rc = backup()

        assert rc == 0
        start.assert_called_once()  # type: ignore[attr-defined]

    @patch("worker.backup._stop_containers", return_value=False)
    def test_backup_no_fieldnotes_dir(
        self, _stop: object, tmp_path: Path
    ) -> None:
        fn = tmp_path / ".fieldnotes"  # does not exist
        with patch("worker.backup._FN_DIR", fn):
            rc = backup()
        assert rc == 1


# ── _prune_backups() tests ─────────────────────────────────────────


class TestPruneBackups:
    def test_prune_keeps_n_most_recent(self, tmp_path: Path) -> None:
        names = [
            "fieldnotes-20260101-000000.tar.gz",
            "fieldnotes-20260102-000000.tar.gz",
            "fieldnotes-20260103-000000.tar.gz",
            "fieldnotes-20260104-000000.tar.gz",
            "fieldnotes-20260105-000000.tar.gz",
        ]
        for n in names:
            (tmp_path / n).write_bytes(b"x")

        with patch("worker.backup._BACKUPS_DIR", tmp_path):
            _prune_backups(3)

        remaining = sorted(p.name for p in tmp_path.glob("fieldnotes-*.tar.gz"))
        assert remaining == names[2:]  # only last 3 kept

    def test_prune_noop_when_fewer(self, tmp_path: Path) -> None:
        names = ["fieldnotes-20260101-000000.tar.gz", "fieldnotes-20260102-000000.tar.gz"]
        for n in names:
            (tmp_path / n).write_bytes(b"x")

        with patch("worker.backup._BACKUPS_DIR", tmp_path):
            _prune_backups(5)

        remaining = sorted(p.name for p in tmp_path.glob("fieldnotes-*.tar.gz"))
        assert remaining == names

    def test_prune_rejects_zero_keep(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            _prune_backups(0)

    def test_prune_rejects_negative_keep(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            _prune_backups(-1)

    @patch("worker.backup._stop_containers", return_value=False)
    def test_backup_with_keep_prunes(
        self, _stop: object, tmp_path: Path
    ) -> None:
        fn = tmp_path / ".fieldnotes"
        fn.mkdir()
        _populate_fieldnotes(fn)
        backups_dir = fn / "backups"
        backups_dir.mkdir()

        # Pre-create 3 old backups
        for i in range(3):
            (backups_dir / f"fieldnotes-2026010{i}-000000.tar.gz").write_bytes(b"old")

        with (
            patch("worker.backup._FN_DIR", fn),
            patch("worker.backup._BACKUPS_DIR", backups_dir),
        ):
            rc = backup(keep=2)

        assert rc == 0
        # 3 old + 1 new = 4 total, keep=2 means only 2 remain
        remaining = list(backups_dir.glob("fieldnotes-*.tar.gz"))
        assert len(remaining) == 2

    @patch("worker.backup._stop_containers", return_value=False)
    def test_backup_without_keep_no_prune(
        self, _stop: object, tmp_path: Path
    ) -> None:
        fn = tmp_path / ".fieldnotes"
        fn.mkdir()
        _populate_fieldnotes(fn)
        backups_dir = fn / "backups"
        backups_dir.mkdir()

        for i in range(3):
            (backups_dir / f"fieldnotes-2026010{i}-000000.tar.gz").write_bytes(b"old")

        with (
            patch("worker.backup._FN_DIR", fn),
            patch("worker.backup._BACKUPS_DIR", backups_dir),
        ):
            rc = backup()

        assert rc == 0
        remaining = list(backups_dir.glob("fieldnotes-*.tar.gz"))
        assert len(remaining) == 4  # 3 old + 1 new, nothing pruned


# ── list_backups() tests ────────────────────────────────────────────


class TestListBackups:
    def test_list_no_backups(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with patch("worker.backup._BACKUPS_DIR", tmp_path / "nope"):
            rc = list_backups()
        assert rc == 0
        assert "No backups found" in capsys.readouterr().out

    @patch("worker.backup._stop_containers", return_value=False)
    def test_list_shows_archives(
        self, _stop: object, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fn = tmp_path / ".fieldnotes"
        fn.mkdir()
        _populate_fieldnotes(fn)

        backups_dir = fn / "backups"
        with (
            patch("worker.backup._FN_DIR", fn),
            patch("worker.backup._BACKUPS_DIR", backups_dir),
        ):
            backup()  # create one

        with patch("worker.backup._BACKUPS_DIR", backups_dir):
            rc = list_backups()

        assert rc == 0
        out = capsys.readouterr().out
        assert "fieldnotes-" in out


# ── restore() tests ─────────────────────────────────────────────────


class TestRestore:
    @patch("worker.backup._stop_containers", return_value=False)
    def test_restore_from_archive(
        self, _stop: object, tmp_path: Path
    ) -> None:
        fn = tmp_path / ".fieldnotes"
        fn.mkdir()
        _populate_fieldnotes(fn)

        backups_dir = fn / "backups"
        archive = _create_test_archive(
            backups_dir, fn, "fieldnotes-20260101-000000.tar.gz"
        )

        # Delete a file, then restore
        (fn / "config.toml").unlink()
        assert not (fn / "config.toml").exists()

        with (
            patch("worker.backup._FN_DIR", fn),
            patch("worker.backup._BACKUPS_DIR", backups_dir),
        ):
            rc = restore(archive)

        assert rc == 0
        assert (fn / "config.toml").exists()

    @patch("worker.backup._stop_containers", return_value=False)
    def test_restore_by_name(
        self, _stop: object, tmp_path: Path
    ) -> None:
        fn = tmp_path / ".fieldnotes"
        fn.mkdir()
        _populate_fieldnotes(fn)

        backups_dir = fn / "backups"
        _create_test_archive(
            backups_dir, fn, "fieldnotes-20260101-000000.tar.gz"
        )

        with (
            patch("worker.backup._FN_DIR", fn),
            patch("worker.backup._BACKUPS_DIR", backups_dir),
        ):
            rc = restore(Path("fieldnotes-20260101-000000.tar.gz"))

        assert rc == 0

    def test_restore_missing_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with patch("worker.backup._BACKUPS_DIR", tmp_path):
            rc = restore(Path("nonexistent.tar.gz"))
        assert rc == 1
        assert "not found" in capsys.readouterr().err

    @patch("worker.backup._stop_containers", return_value=False)
    def test_restore_blocks_path_traversal(
        self, _stop: object, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fn = tmp_path / ".fieldnotes"
        fn.mkdir()
        backups_dir = fn / "backups"
        backups_dir.mkdir(parents=True)

        # Create an archive with a malicious path
        evil_archive = backups_dir / "evil.tar.gz"
        with tarfile.open(evil_archive, "w:gz") as tar:
            import io

            data = b"pwned"
            info = tarfile.TarInfo(name="../../../etc/evil")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

        with (
            patch("worker.backup._FN_DIR", fn),
            patch("worker.backup._BACKUPS_DIR", backups_dir),
        ):
            rc = restore(evil_archive)

        assert rc == 1
        assert "unsafe path" in capsys.readouterr().err

    @patch("worker.backup._stop_containers", return_value=False)
    def test_restore_blocks_symlinks(
        self, _stop: object, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fn = tmp_path / ".fieldnotes"
        fn.mkdir()
        backups_dir = fn / "backups"
        backups_dir.mkdir(parents=True)

        evil_archive = backups_dir / "symlink.tar.gz"
        with tarfile.open(evil_archive, "w:gz") as tar:
            info = tarfile.TarInfo(name="evil_link")
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            tar.addfile(info)

        with (
            patch("worker.backup._FN_DIR", fn),
            patch("worker.backup._BACKUPS_DIR", backups_dir),
        ):
            rc = restore(evil_archive)

        assert rc == 1
        assert "link" in capsys.readouterr().err.lower()

    @patch("worker.backup._stop_containers", return_value=True)
    @patch("worker.backup._start_containers")
    def test_restore_restarts_containers(
        self, start: object, _stop: object, tmp_path: Path
    ) -> None:
        fn = tmp_path / ".fieldnotes"
        fn.mkdir()
        _populate_fieldnotes(fn)

        backups_dir = fn / "backups"
        archive = _create_test_archive(
            backups_dir, fn, "fieldnotes-20260101-000000.tar.gz"
        )

        with (
            patch("worker.backup._FN_DIR", fn),
            patch("worker.backup._BACKUPS_DIR", backups_dir),
        ):
            rc = restore(archive)

        assert rc == 0
        start.assert_called_once()  # type: ignore[attr-defined]

    @patch("worker.backup._stop_containers", return_value=False)
    def test_backup_excludes_daemon_log(
        self, _stop: object, tmp_path: Path
    ) -> None:
        fn = tmp_path / ".fieldnotes"
        fn.mkdir()
        _populate_fieldnotes(fn)
        # Add a daemon log file inside data/
        (fn / "data" / "daemon.log").write_text("log line\n")
        (fn / "data" / "daemon.log.1").write_text("old log\n")

        with (
            patch("worker.backup._FN_DIR", fn),
            patch("worker.backup._BACKUPS_DIR", fn / "backups"),
        ):
            rc = backup()

        assert rc == 0
        archives = list((fn / "backups").glob("fieldnotes-*.tar.gz"))
        with tarfile.open(archives[0], "r:gz") as tar:
            names = tar.getnames()
        assert "data/neo4j/dummy.db" in names
        assert "data/daemon.log" not in names
        assert "data/daemon.log.1" not in names


# ── CLI dispatch tests ──────────────────────────────────────────────


class TestCLIDispatch:
    @patch("worker.backup.backup", return_value=0)
    def test_cli_backup_no_subcommand(self, mock_backup: object) -> None:
        rc = main(["backup"])
        assert rc == 0
        mock_backup.assert_called_once_with(keep=None)  # type: ignore[attr-defined]

    @patch("worker.backup.backup", return_value=0)
    def test_cli_backup_create(self, mock_backup: object) -> None:
        rc = main(["backup", "create"])
        assert rc == 0
        mock_backup.assert_called_once_with(keep=None)  # type: ignore[attr-defined]

    @patch("worker.backup.backup", return_value=0)
    def test_cli_backup_create_with_keep(self, mock_backup: object) -> None:
        rc = main(["backup", "create", "--keep", "5"])
        assert rc == 0
        mock_backup.assert_called_once_with(keep=5)  # type: ignore[attr-defined]

    @patch("worker.backup.backup", return_value=0)
    def test_cli_backup_bare_with_keep(self, mock_backup: object) -> None:
        rc = main(["backup", "--keep", "3"])
        assert rc == 0
        mock_backup.assert_called_once_with(keep=3)  # type: ignore[attr-defined]

    @patch("worker.backup.list_backups", return_value=0)
    def test_cli_backup_list(self, mock_list: object) -> None:
        rc = main(["backup", "list"])
        assert rc == 0
        mock_list.assert_called_once()  # type: ignore[attr-defined]

    @patch("worker.backup.schedule_backup", return_value=0)
    def test_cli_backup_schedule(self, mock_sched: object) -> None:
        rc = main(["backup", "schedule"])
        assert rc == 0
        mock_sched.assert_called_once_with(remove=False, keep=None)  # type: ignore[attr-defined]

    @patch("worker.backup.schedule_backup", return_value=0)
    def test_cli_backup_schedule_remove(self, mock_sched: object) -> None:
        rc = main(["backup", "schedule", "--remove"])
        assert rc == 0
        mock_sched.assert_called_once_with(remove=True, keep=None)  # type: ignore[attr-defined]

    @patch("worker.backup.schedule_backup", return_value=0)
    def test_cli_backup_schedule_with_keep(self, mock_sched: object) -> None:
        rc = main(["backup", "schedule", "--keep", "7"])
        assert rc == 0
        mock_sched.assert_called_once_with(remove=False, keep=7)  # type: ignore[attr-defined]

    @patch("worker.backup.restore", return_value=0)
    def test_cli_restore(self, mock_restore: object) -> None:
        rc = main(["restore", "fieldnotes-20260101-000000.tar.gz"])
        assert rc == 0
        mock_restore.assert_called_once()  # type: ignore[attr-defined]
