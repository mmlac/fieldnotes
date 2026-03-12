"""Tests for daemon/service lifecycle management (launchd / systemd)."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from worker.service import (
    platform_backend,
    install,
    uninstall,
    status,
    start,
    stop,
)
from worker.service.launchd import LaunchdBackend, _fieldnotes_executable, _render_template
from worker.service.systemd import SystemdBackend


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------


class TestFieldnotesExecutable:
    @patch("worker.service.launchd.shutil.which", return_value="/usr/local/bin/fieldnotes")
    def test_found_on_path(self, _mock: object) -> None:
        assert _fieldnotes_executable() == ["/usr/local/bin/fieldnotes"]

    @patch("worker.service.launchd.shutil.which", return_value=None)
    def test_fallback_to_interpreter(self, _mock: object) -> None:
        exe = _fieldnotes_executable()
        assert isinstance(exe, list)
        assert len(exe) == 3
        assert exe[1] == "-m"
        assert exe[2] == "worker.cli"


class TestRenderTemplate:
    def test_plist_template(self) -> None:
        exe_strings = "\n        ".join(
            f"<string>{p}</string>" for p in ["/usr/bin/fieldnotes", "serve", "--daemon"]
        )
        content = _render_template(
            "com.fieldnotes.daemon.plist",
            {"PROGRAM_ARGUMENTS": exe_strings, "LOG_PATH": "/tmp/fn.log"},
        )
        assert "com.fieldnotes.daemon" in content
        assert "<string>/usr/bin/fieldnotes</string>" in content
        assert "<string>serve</string>" in content
        assert "/tmp/fn.log" in content

    def test_systemd_template(self) -> None:
        from worker.service.systemd import _render_template as systemd_render

        content = systemd_render(
            "fieldnotes.service",
            {"EXECUTABLE": "/usr/bin/fieldnotes", "LOG_DIR": "/tmp/logs"},
        )
        assert "fieldnotes" in content
        assert "/usr/bin/fieldnotes" in content
        assert "ExecStart" in content


# ------------------------------------------------------------------
# Platform detection
# ------------------------------------------------------------------


class TestPlatformBackend:
    @patch("worker.service.platform.system", return_value="Darwin")
    def test_darwin_returns_launchd(self, _mock: object) -> None:
        backend = platform_backend()
        assert isinstance(backend, LaunchdBackend)

    @patch("worker.service.platform.system", return_value="Linux")
    def test_linux_returns_systemd(self, _mock: object) -> None:
        backend = platform_backend()
        assert isinstance(backend, SystemdBackend)

    @patch("worker.service.platform.system", return_value="Windows")
    def test_unsupported_raises(self, _mock: object) -> None:
        with pytest.raises(SystemExit, match="unsupported platform"):
            platform_backend()


# ------------------------------------------------------------------
# LaunchdBackend
# ------------------------------------------------------------------


class TestLaunchdBackend:
    @patch("worker.service.launchd.subprocess.run")
    @patch("worker.service.launchd._fieldnotes_executable", return_value=["/usr/bin/fieldnotes"])
    def test_install(
        self, _mock_exe: object, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        backend = LaunchdBackend()
        backend._plist_dir = tmp_path
        backend._plist_path = tmp_path / "com.fieldnotes.daemon.plist"
        backend._log_dir = tmp_path / "logs"
        backend._log_path = tmp_path / "logs" / "daemon.log"

        backend.install()

        assert backend._plist_path.exists()
        content = backend._plist_path.read_text()
        assert "com.fieldnotes.daemon" in content
        assert "<string>/usr/bin/fieldnotes</string>" in content
        assert "<string>serve</string>" in content
        assert "<string>--daemon</string>" in content
        assert backend._plist_path.stat().st_mode & 0o777 == 0o644
        mock_run.assert_called_once()
        assert "launchctl" in mock_run.call_args[0][0]

    @patch("worker.service.launchd.subprocess.run")
    def test_uninstall_removes_plist(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        backend = LaunchdBackend()
        plist_path = tmp_path / "com.fieldnotes.daemon.plist"
        plist_path.write_text("<plist>test</plist>")
        backend._plist_path = plist_path

        backend.uninstall()

        assert not plist_path.exists()
        mock_run.assert_called_once()

    @patch("worker.service.launchd.subprocess.run")
    def test_uninstall_missing_plist(
        self, mock_run: MagicMock, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        backend = LaunchdBackend()
        backend._plist_path = tmp_path / "nonexistent.plist"

        backend.uninstall()

        assert "not found" in capsys.readouterr().out
        mock_run.assert_not_called()

    @patch("worker.service.launchd.subprocess.run")
    def test_start_not_installed_raises(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        backend = LaunchdBackend()
        backend._plist_path = tmp_path / "nonexistent.plist"

        with pytest.raises(SystemExit, match="not installed"):
            backend.start()

    @patch("worker.service.launchd.subprocess.run")
    def test_start_installed(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        backend = LaunchdBackend()
        plist_path = tmp_path / "com.fieldnotes.daemon.plist"
        plist_path.write_text("<plist/>")
        backend._plist_path = plist_path

        backend.start()
        mock_run.assert_called_once()

    @patch("worker.service.launchd.subprocess.run")
    def test_stop(self, mock_run: MagicMock, tmp_path: Path) -> None:
        backend = LaunchdBackend()
        backend._plist_path = tmp_path / "com.fieldnotes.daemon.plist"

        backend.stop()
        mock_run.assert_called_once()

    @patch("worker.service.launchd.subprocess.run")
    def test_status_loaded(
        self, mock_run: MagicMock, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        mock_run.return_value = MagicMock(returncode=0, stdout="PID: 1234\nStatus: running")
        backend = LaunchdBackend()
        backend._plist_path = tmp_path / "test.plist"
        backend._log_path = tmp_path / "nonexistent.log"

        backend.status()

        out = capsys.readouterr().out
        assert "loaded" in out

    @patch("worker.service.launchd.subprocess.run")
    def test_status_not_loaded(
        self, mock_run: MagicMock, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        backend = LaunchdBackend()
        backend._plist_path = tmp_path / "test.plist"
        backend._log_path = tmp_path / "nonexistent.log"

        backend.status()

        out = capsys.readouterr().out
        assert "not loaded" in out


# ------------------------------------------------------------------
# SystemdBackend
# ------------------------------------------------------------------


class TestSystemdBackend:
    @patch("worker.service.systemd.subprocess.run")
    @patch("worker.service.systemd._fieldnotes_executable", return_value=["/usr/bin/fieldnotes"])
    def test_install(
        self, _mock_exe: object, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        backend = SystemdBackend()
        backend._unit_dir = tmp_path
        backend._unit_path = tmp_path / "fieldnotes.service"
        backend._log_dir = tmp_path / "logs"

        backend.install()

        assert backend._unit_path.exists()
        content = backend._unit_path.read_text()
        assert "ExecStart" in content
        assert "/usr/bin/fieldnotes" in content
        assert backend._unit_path.stat().st_mode & 0o777 == 0o644
        # Should call daemon-reload and enable --now
        assert mock_run.call_count == 2

    @patch("worker.service.systemd.subprocess.run")
    def test_uninstall_removes_unit(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        backend = SystemdBackend()
        unit_path = tmp_path / "fieldnotes.service"
        unit_path.write_text("[Unit]\ntest")
        backend._unit_path = unit_path

        backend.uninstall()

        assert not unit_path.exists()

    @patch("worker.service.systemd.subprocess.run")
    def test_uninstall_missing_unit(
        self, mock_run: MagicMock, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        backend = SystemdBackend()
        backend._unit_path = tmp_path / "nonexistent.service"

        backend.uninstall()

        out = capsys.readouterr().out
        assert "not found" in out

    @patch("worker.service.systemd.subprocess.run")
    def test_start_not_installed_raises(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        backend = SystemdBackend()
        backend._unit_path = tmp_path / "nonexistent.service"

        with pytest.raises(SystemExit, match="not installed"):
            backend.start()

    @patch("worker.service.systemd.subprocess.run")
    def test_start_installed(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        backend = SystemdBackend()
        unit_path = tmp_path / "fieldnotes.service"
        unit_path.write_text("[Unit]")
        backend._unit_path = unit_path

        backend.start()
        mock_run.assert_called_once()

    @patch("worker.service.systemd.subprocess.run")
    def test_stop(self, mock_run: MagicMock) -> None:
        backend = SystemdBackend()
        backend.stop()
        mock_run.assert_called_once()

    @patch("worker.service.systemd.subprocess.run")
    def test_status_active(
        self, mock_run: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        mock_run.return_value = MagicMock(returncode=0, stdout="active (running)")
        backend = SystemdBackend()

        backend.status()

        out = capsys.readouterr().out
        assert "active" in out

    @patch("worker.service.systemd.subprocess.run")
    def test_status_not_installed(
        self, mock_run: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        mock_run.return_value = MagicMock(returncode=4, stdout="", stderr="")
        backend = SystemdBackend()

        backend.status()

        out = capsys.readouterr().out
        assert "not installed" in out


# ------------------------------------------------------------------
# Public API wrappers
# ------------------------------------------------------------------


class TestPublicAPI:
    @patch("worker.service.platform_backend")
    def test_install_success(self, mock_backend_fn: MagicMock) -> None:
        mock_backend = MagicMock()
        mock_backend_fn.return_value = mock_backend

        rc = install()

        assert rc == 0
        mock_backend.install.assert_called_once()

    @patch("worker.service.platform_backend")
    def test_install_failure(self, mock_backend_fn: MagicMock) -> None:
        import subprocess

        mock_backend = MagicMock()
        mock_backend.install.side_effect = subprocess.CalledProcessError(1, "cmd")
        mock_backend_fn.return_value = mock_backend

        rc = install()

        assert rc == 1

    @patch("worker.service.platform_backend")
    def test_uninstall_success(self, mock_backend_fn: MagicMock) -> None:
        mock_backend = MagicMock()
        mock_backend_fn.return_value = mock_backend

        rc = uninstall()

        assert rc == 0

    @patch("worker.service.platform_backend")
    def test_status_success(self, mock_backend_fn: MagicMock) -> None:
        mock_backend = MagicMock()
        mock_backend_fn.return_value = mock_backend

        rc = status()

        assert rc == 0

    @patch("worker.service.platform_backend")
    def test_start_success(self, mock_backend_fn: MagicMock) -> None:
        mock_backend = MagicMock()
        mock_backend_fn.return_value = mock_backend

        rc = start()

        assert rc == 0

    @patch("worker.service.platform_backend")
    def test_stop_success(self, mock_backend_fn: MagicMock) -> None:
        mock_backend = MagicMock()
        mock_backend_fn.return_value = mock_backend

        rc = stop()

        assert rc == 0


# ------------------------------------------------------------------
# Backwards compatibility: worker.daemon re-exports
# ------------------------------------------------------------------


class TestDaemonCompat:
    """Ensure ``from worker.daemon import ...`` still works."""

    def test_daemon_reexports(self) -> None:
        from worker.daemon import (
            install,
            uninstall,
            status,
            start,
            stop,
            platform_backend,
        )
        # Just verify they're callable
        assert callable(install)
        assert callable(platform_backend)
