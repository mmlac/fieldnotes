"""Tests for the setup-claude command (Claude Desktop MCP configuration)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from worker.setup import setup_claude, _default_config_path, _MCP_ENTRY


# ------------------------------------------------------------------
# Config path detection
# ------------------------------------------------------------------


class TestDefaultConfigPath:
    @patch("worker.setup.platform.system", return_value="Darwin")
    def test_macos_path(self, _mock: object) -> None:
        p = _default_config_path()
        assert "Library" in str(p)
        assert "Application Support" in str(p)
        assert p.name == "claude_desktop_config.json"

    @patch("worker.setup.platform.system", return_value="Windows")
    def test_windows_path(self, _mock: object) -> None:
        p = _default_config_path()
        assert "AppData" in str(p)
        assert p.name == "claude_desktop_config.json"

    @patch("worker.setup.platform.system", return_value="Linux")
    def test_linux_path(self, _mock: object) -> None:
        p = _default_config_path()
        assert ".config" in str(p)
        assert p.name == "claude_desktop_config.json"


# ------------------------------------------------------------------
# setup_claude
# ------------------------------------------------------------------


class TestSetupClaude:
    @patch("worker.setup.shutil.which", return_value=None)
    def test_fieldnotes_not_on_path_returns_1(
        self, _mock: object, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = setup_claude()
        assert rc == 1
        err = capsys.readouterr().err
        assert "not found on PATH" in err

    @patch("worker.setup.shutil.which", return_value="/usr/local/bin/fieldnotes")
    def test_fresh_install(
        self, _mock: object, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cfg_path = tmp_path / "claude_desktop_config.json"

        rc = setup_claude(config_path=cfg_path)

        assert rc == 0
        assert cfg_path.exists()
        data = json.loads(cfg_path.read_text())
        assert "mcpServers" in data
        assert "fieldnotes" in data["mcpServers"]
        assert data["mcpServers"]["fieldnotes"] == _MCP_ENTRY

    @patch("worker.setup.shutil.which", return_value="/usr/local/bin/fieldnotes")
    def test_existing_config_preserved(
        self, _mock: object, tmp_path: Path
    ) -> None:
        cfg_path = tmp_path / "claude_desktop_config.json"
        existing = {"mcpServers": {"other_tool": {"command": "other"}}, "customKey": 42}
        cfg_path.write_text(json.dumps(existing))

        rc = setup_claude(config_path=cfg_path)

        assert rc == 0
        data = json.loads(cfg_path.read_text())
        assert data["customKey"] == 42
        assert "other_tool" in data["mcpServers"]
        assert "fieldnotes" in data["mcpServers"]

    @patch("worker.setup.shutil.which", return_value="/usr/local/bin/fieldnotes")
    def test_already_configured_noop(
        self, _mock: object, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cfg_path = tmp_path / "claude_desktop_config.json"
        existing = {"mcpServers": {"fieldnotes": _MCP_ENTRY}}
        cfg_path.write_text(json.dumps(existing))

        rc = setup_claude(config_path=cfg_path)

        assert rc == 0
        out = capsys.readouterr().out
        assert "already configured" in out

    @patch("worker.setup.shutil.which", return_value="/usr/local/bin/fieldnotes")
    def test_backup_created(
        self, _mock: object, tmp_path: Path
    ) -> None:
        cfg_path = tmp_path / "claude_desktop_config.json"
        existing = {"mcpServers": {"other": {"command": "x"}}}
        cfg_path.write_text(json.dumps(existing))

        setup_claude(config_path=cfg_path)

        backups = list(tmp_path.glob("*.backup-*.json"))
        assert len(backups) == 1
        backup_data = json.loads(backups[0].read_text())
        assert backup_data == existing

    @patch("worker.setup.shutil.which", return_value="/usr/local/bin/fieldnotes")
    def test_invalid_json_returns_1(
        self, _mock: object, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cfg_path = tmp_path / "claude_desktop_config.json"
        cfg_path.write_text("not valid json {{{")

        rc = setup_claude(config_path=cfg_path)

        assert rc == 1
        err = capsys.readouterr().err
        assert "cannot read" in err

    @patch("worker.setup.shutil.which", return_value="/usr/local/bin/fieldnotes")
    def test_non_object_json_returns_1(
        self, _mock: object, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cfg_path = tmp_path / "claude_desktop_config.json"
        cfg_path.write_text('"just a string"')

        rc = setup_claude(config_path=cfg_path)

        assert rc == 1
        err = capsys.readouterr().err
        assert "not a JSON object" in err

    @patch("worker.setup.shutil.which", return_value="/usr/local/bin/fieldnotes")
    def test_creates_parent_directories(
        self, _mock: object, tmp_path: Path
    ) -> None:
        cfg_path = tmp_path / "deep" / "nested" / "claude_desktop_config.json"

        rc = setup_claude(config_path=cfg_path)

        assert rc == 0
        assert cfg_path.exists()
