"""Tests for daemon lifecycle management (launchd / systemd)."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from worker.daemon import (
    _LaunchdBackend,
    _SystemdBackend,
    _fieldnotes_executable,
    _render_template,
    platform_backend,
    install,
    uninstall,
    status,
    start,
    stop,
)


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------


class TestFieldnotesExecutable:
    @patch("worker.daemon.shutil.which", return_value="/usr/local/bin/fieldnotes")
    def test_found_on_path(self, _mock: object) -> None:
        assert _fieldnotes_executable() == "/usr/local/bin/fieldnotes"

    @patch("worker.daemon.shutil.which", return_value=None)
    def test_fallback_to_interpreter(self, _mock: object) -> None:
        exe = _fieldnotes_executable()
        assert "-m worker.cli" in exe


class TestRenderTemplate:
    def test_plist_template(self) -> None:
        content = _render_template(
            "com.fieldnotes.daemon.plist",
            {"EXECUTABLE": "/usr/bin/fieldnotes", "LOG_PATH": "/tmp/fn.log"},
        )
        assert "com.fieldnotes.daemon" in content
        assert "/usr/bin/fieldnotes" in content
        assert "/tmp/fn.log" in content

    def test_systemd_template(self) -> None:
        content = _render_template(
            "fieldnotes.service",
            {"EXECUTABLE": "/usr/bin/fieldnotes"},
        )
        assert "fieldnotes" in content
        assert "/usr/bin/fieldnotes" in content
        assert "ExecStart" in content


# ------------------------------------------------------------------
# Platform detection
# ------------------------------------------------------------------


class TestPlatformBackend:
    @patch("worker.daemon.platform.system", return_value="Darwin")
    def test_darwin_returns_launchd(self, _mock: object) -> None:
        backend = platform_backend()
        assert isinstance(backend, _LaunchdBackend)

    @patch("worker.daemon.platform.system", return_value="Linux")
    def test_linux_returns_systemd(self, _mock: object) -> None:
        backend = platform_backend()
        assert isinstance(backend, _SystemdBackend)

    @patch("worker.daemon.platform.system", return_value="Windows")
    def test_unsupported_raises(self, _mock: object) -> None:
        with pytest.raises(SystemExit, match="unsupported platform"):
            platform_backend()


# ------------------------------------------------------------------
# LaunchdBackend
# ------------------------------------------------------------------


class TestLaunchdBackend:
    @patch("worker.daemon.subprocess.run")
    @patch("worker.daemon._fieldnotes_executable", return_value="/usr/bin/fieldnotes")
    def test_install(
        self, _mock_exe: object, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        backend = _LaunchdBackend()
        backend._plist_dir = tmp_path
        backend._plist_path = tmp_path / "com.fieldnotes.daemon.plist"
        backend._log_path = tmp_path / "fieldnotes.log"

        backend.install()

        assert backend._plist_path.exists()
        content = backend._plist_path.read_text()
        assert "com.fieldnotes.daemon" in content
        assert "/usr/bin/fieldnotes" in content
        mock_run.assert_called_once()
        assert "launchctl" in mock_run.call_args[0][0]

    @patch("worker.daemon.subprocess.run")
    def test_uninstall_removes_plist(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        backend = _LaunchdBackend()
        plist_path = tmp_path / "com.fieldnotes.daemon.plist"
        plist_path.write_text("<plist>test</plist>")
        backend._plist_path = plist_path

        backend.uninstall()

        assert not plist_path.exists()
        mock_run.assert_called_once()

    @patch("worker.daemon.subprocess.run")
    def test_uninstall_missing_plist(
        self, mock_run: MagicMock, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        backend = _LaunchdBackend()
        backend._plist_path = tmp_path / "nonexistent.plist"

        backend.uninstall()

        assert "not found" in capsys.readouterr().out
        mock_run.assert_not_called()

    @patch("worker.daemon.subprocess.run")
    def test_start_not_installed_raises(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        backend = _LaunchdBackend()
        backend._plist_path = tmp_path / "nonexistent.plist"

        with pytest.raises(SystemExit, match="not installed"):
            backend.start()

    @patch("worker.daemon.subprocess.run")
    def test_start_installed(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        backend = _LaunchdBackend()
        plist_path = tmp_path / "com.fieldnotes.daemon.plist"
        plist_path.write_text("<plist/>")
        backend._plist_path = plist_path

        backend.start()
        mock_run.assert_called_once()

    @patch("worker.daemon.subprocess.run")
    def test_stop(self, mock_run: MagicMock, tmp_path: Path) -> None:
        backend = _LaunchdBackend()
        backend._plist_path = tmp_path / "com.fieldnotes.daemon.plist"

        backend.stop()
        mock_run.assert_called_once()

    @patch("worker.daemon.subprocess.run")
    def test_status_loaded(
        self, mock_run: MagicMock, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        mock_run.return_value = MagicMock(returncode=0, stdout="PID: 1234\nStatus: running")
        backend = _LaunchdBackend()
        backend._plist_path = tmp_path / "test.plist"
        backend._log_path = tmp_path / "nonexistent.log"

        backend.status()

        out = capsys.readouterr().out
        assert "loaded" in out

    @patch("worker.daemon.subprocess.run")
    def test_status_not_loaded(
        self, mock_run: MagicMock, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        backend = _LaunchdBackend()
        backend._plist_path = tmp_path / "test.plist"
        backend._log_path = tmp_path / "nonexistent.log"

        backend.status()

        out = capsys.readouterr().out
        assert "not loaded" in out


# ------------------------------------------------------------------
# SystemdBackend
# ------------------------------------------------------------------


class TestSystemdBackend:
    @patch("worker.daemon.subprocess.run")
    @patch("worker.daemon._fieldnotes_executable", return_value="/usr/bin/fieldnotes")
    def test_install(
        self, _mock_exe: object, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        backend = _SystemdBackend()
        backend._unit_dir = tmp_path
        backend._unit_path = tmp_path / "fieldnotes.service"

        backend.install()

        assert backend._unit_path.exists()
        content = backend._unit_path.read_text()
        assert "ExecStart" in content
        assert "/usr/bin/fieldnotes" in content
        # Should call daemon-reload and enable --now
        assert mock_run.call_count == 2

    @patch("worker.daemon.subprocess.run")
    def test_uninstall_removes_unit(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        backend = _SystemdBackend()
        unit_path = tmp_path / "fieldnotes.service"
        unit_path.write_text("[Unit]\ntest")
        backend._unit_path = unit_path

        backend.uninstall()

        assert not unit_path.exists()

    @patch("worker.daemon.subprocess.run")
    def test_uninstall_missing_unit(
        self, mock_run: MagicMock, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        backend = _SystemdBackend()
        backend._unit_path = tmp_path / "nonexistent.service"

        backend.uninstall()

        out = capsys.readouterr().out
        assert "not found" in out

    @patch("worker.daemon.subprocess.run")
    def test_start_not_installed_raises(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        backend = _SystemdBackend()
        backend._unit_path = tmp_path / "nonexistent.service"

        with pytest.raises(SystemExit, match="not installed"):
            backend.start()

    @patch("worker.daemon.subprocess.run")
    def test_start_installed(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        backend = _SystemdBackend()
        unit_path = tmp_path / "fieldnotes.service"
        unit_path.write_text("[Unit]")
        backend._unit_path = unit_path

        backend.start()
        mock_run.assert_called_once()

    @patch("worker.daemon.subprocess.run")
    def test_stop(self, mock_run: MagicMock) -> None:
        backend = _SystemdBackend()
        backend.stop()
        mock_run.assert_called_once()

    @patch("worker.daemon.subprocess.run")
    def test_status_active(
        self, mock_run: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        mock_run.return_value = MagicMock(returncode=0, stdout="active (running)")
        backend = _SystemdBackend()

        backend.status()

        out = capsys.readouterr().out
        assert "active" in out

    @patch("worker.daemon.subprocess.run")
    def test_status_not_installed(
        self, mock_run: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        mock_run.return_value = MagicMock(returncode=4, stdout="", stderr="")
        backend = _SystemdBackend()

        backend.status()

        out = capsys.readouterr().out
        assert "not installed" in out


# ------------------------------------------------------------------
# Public API wrappers
# ------------------------------------------------------------------


class TestPublicAPI:
    @patch("worker.daemon.platform_backend")
    def test_install_success(self, mock_backend_fn: MagicMock) -> None:
        mock_backend = MagicMock()
        mock_backend_fn.return_value = mock_backend

        rc = install()

        assert rc == 0
        mock_backend.install.assert_called_once()

    @patch("worker.daemon.platform_backend")
    def test_install_failure(self, mock_backend_fn: MagicMock) -> None:
        import subprocess

        mock_backend = MagicMock()
        mock_backend.install.side_effect = subprocess.CalledProcessError(1, "cmd")
        mock_backend_fn.return_value = mock_backend

        rc = install()

        assert rc == 1

    @patch("worker.daemon.platform_backend")
    def test_uninstall_success(self, mock_backend_fn: MagicMock) -> None:
        mock_backend = MagicMock()
        mock_backend_fn.return_value = mock_backend

        rc = uninstall()

        assert rc == 0

    @patch("worker.daemon.platform_backend")
    def test_status_success(self, mock_backend_fn: MagicMock) -> None:
        mock_backend = MagicMock()
        mock_backend_fn.return_value = mock_backend

        rc = status()

        assert rc == 0

    @patch("worker.daemon.platform_backend")
    def test_start_success(self, mock_backend_fn: MagicMock) -> None:
        mock_backend = MagicMock()
        mock_backend_fn.return_value = mock_backend

        rc = start()

        assert rc == 0

    @patch("worker.daemon.platform_backend")
    def test_stop_success(self, mock_backend_fn: MagicMock) -> None:
        mock_backend = MagicMock()
        mock_backend_fn.return_value = mock_backend

        rc = stop()

        assert rc == 0
