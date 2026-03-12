"""Tests for worker.gastown — GasTown integration helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from worker.gastown import (
    detect_rig_root,
    write_mcp_config,
    validate_connectivity,
    _fieldnotes_command,
    setup_gastown,
)


# ------------------------------------------------------------------
# detect_rig_root
# ------------------------------------------------------------------


class TestDetectRigRoot:
    def test_from_env_variable(self, tmp_path: Path) -> None:
        with patch.dict(os.environ, {"GT_RIG_ROOT": str(tmp_path)}):
            result = detect_rig_root()
        assert result == tmp_path

    def test_env_variable_nonexistent_dir(self, tmp_path: Path) -> None:
        # Nonexistent dir falls through to directory walk; mock cwd to avoid
        # finding a rig layout in the real filesystem.
        with (
            patch.dict(os.environ, {"GT_RIG_ROOT": "/no/such/path/xyzzy"}),
            patch("worker.gastown.Path.cwd", return_value=tmp_path),
        ):
            result = detect_rig_root()
        assert result is None

    def test_detects_rig_layout(self, tmp_path: Path) -> None:
        (tmp_path / "polecats").mkdir()
        (tmp_path / ".beads").mkdir()

        with (
            patch.dict(os.environ, {}, clear=False),
            patch("worker.gastown.Path.cwd", return_value=tmp_path),
        ):
            os.environ.pop("GT_RIG_ROOT", None)
            result = detect_rig_root()

        assert result == tmp_path

    def test_walks_up_to_find_rig(self, tmp_path: Path) -> None:
        (tmp_path / "polecats").mkdir()
        (tmp_path / ".beads").mkdir()
        child = tmp_path / "polecats" / "worker"
        child.mkdir(parents=True)

        with (
            patch.dict(os.environ, {}, clear=False),
            patch("worker.gastown.Path.cwd", return_value=child),
        ):
            os.environ.pop("GT_RIG_ROOT", None)
            result = detect_rig_root()

        assert result == tmp_path

    def test_returns_none_when_not_in_rig(self, tmp_path: Path) -> None:
        with (
            patch.dict(os.environ, {}, clear=False),
            patch("worker.gastown.Path.cwd", return_value=tmp_path),
        ):
            os.environ.pop("GT_RIG_ROOT", None)
            result = detect_rig_root()

        assert result is None


# ------------------------------------------------------------------
# _fieldnotes_command
# ------------------------------------------------------------------


class TestFieldnotesCommand:
    @patch("worker.gastown.shutil.which", return_value="/usr/bin/fieldnotes")
    def test_uses_binary_when_found(self, _mock: object) -> None:
        cmd = _fieldnotes_command()
        assert cmd == ["/usr/bin/fieldnotes", "serve", "--mcp"]

    @patch("worker.gastown.shutil.which", return_value=None)
    def test_falls_back_to_module(self, _mock: object) -> None:
        cmd = _fieldnotes_command()
        assert cmd == ["python", "-m", "worker.cli", "serve", "--mcp"]

    @patch("worker.gastown.shutil.which", return_value="/usr/bin/fieldnotes")
    def test_includes_config_path(self, _mock: object) -> None:
        cmd = _fieldnotes_command(config_path=Path("/etc/fn.toml"))
        assert cmd == ["/usr/bin/fieldnotes", "-c", "/etc/fn.toml", "serve", "--mcp"]


# ------------------------------------------------------------------
# write_mcp_config
# ------------------------------------------------------------------


class TestWriteMcpConfig:
    @patch("worker.gastown.shutil.which", return_value="/usr/bin/fieldnotes")
    def test_creates_new_mcp_json(self, _mock: object, tmp_path: Path) -> None:
        result = write_mcp_config(tmp_path)

        assert result == tmp_path / ".mcp.json"
        data = json.loads(result.read_text())
        assert "fieldnotes" in data["mcpServers"]
        assert data["mcpServers"]["fieldnotes"]["command"] == "/usr/bin/fieldnotes"
        assert data["mcpServers"]["fieldnotes"]["args"] == ["serve", "--mcp"]

    @patch("worker.gastown.shutil.which", return_value="/usr/bin/fieldnotes")
    def test_preserves_existing_servers(self, _mock: object, tmp_path: Path) -> None:
        mcp_json = tmp_path / ".mcp.json"
        mcp_json.write_text(json.dumps({
            "mcpServers": {"other": {"command": "other-cmd", "args": []}}
        }))

        write_mcp_config(tmp_path)

        data = json.loads(mcp_json.read_text())
        assert "other" in data["mcpServers"]
        assert "fieldnotes" in data["mcpServers"]

    @patch("worker.gastown.shutil.which", return_value="/usr/bin/fieldnotes")
    def test_overwrites_corrupt_json(self, _mock: object, tmp_path: Path) -> None:
        mcp_json = tmp_path / ".mcp.json"
        mcp_json.write_text("{not valid json")

        write_mcp_config(tmp_path)

        data = json.loads(mcp_json.read_text())
        assert "fieldnotes" in data["mcpServers"]

    @patch("worker.gastown.shutil.which", return_value="/usr/bin/fieldnotes")
    def test_with_config_path(self, _mock: object, tmp_path: Path) -> None:
        write_mcp_config(tmp_path, config_path=Path("/etc/fn.toml"))

        data = json.loads((tmp_path / ".mcp.json").read_text())
        assert data["mcpServers"]["fieldnotes"]["args"] == [
            "-c", "/etc/fn.toml", "serve", "--mcp"
        ]


# ------------------------------------------------------------------
# validate_connectivity
# ------------------------------------------------------------------


class TestValidateConnectivity:
    @patch("worker.gastown.QdrantClient")
    @patch("worker.gastown.GraphDatabase")
    def test_both_healthy(
        self,
        mock_gdb: MagicMock,
        mock_qdrant_cls: MagicMock,
    ) -> None:
        from worker.config import Config, Neo4jConfig, QdrantConfig

        driver = MagicMock()
        mock_gdb.driver.return_value = driver

        client = MagicMock()
        mock_qdrant_cls.return_value = client

        cfg = Config(
            neo4j=Neo4jConfig(uri="bolt://test:7687", user="neo4j", password="pw"),
            qdrant=QdrantConfig(host="localhost", port=6333),
        )
        health = validate_connectivity(cfg)

        assert health["neo4j"] == "ok"
        assert health["qdrant"] == "ok"
        driver.verify_connectivity.assert_called_once()
        driver.close.assert_called_once()
        client.close.assert_called_once()

    @patch("worker.gastown.QdrantClient")
    @patch("worker.gastown.GraphDatabase")
    def test_neo4j_down(
        self,
        mock_gdb: MagicMock,
        mock_qdrant_cls: MagicMock,
    ) -> None:
        from worker.config import Config, Neo4jConfig, QdrantConfig

        mock_gdb.driver.side_effect = ConnectionError("refused")
        client = MagicMock()
        mock_qdrant_cls.return_value = client

        cfg = Config(
            neo4j=Neo4jConfig(uri="bolt://test:7687", user="neo4j", password="pw"),
            qdrant=QdrantConfig(host="localhost", port=6333),
        )
        health = validate_connectivity(cfg)

        assert "error" in health["neo4j"]
        assert health["qdrant"] == "ok"

    @patch("worker.gastown.QdrantClient")
    @patch("worker.gastown.GraphDatabase")
    def test_qdrant_down(
        self,
        mock_gdb: MagicMock,
        mock_qdrant_cls: MagicMock,
    ) -> None:
        from worker.config import Config, Neo4jConfig, QdrantConfig

        driver = MagicMock()
        mock_gdb.driver.return_value = driver
        mock_qdrant_cls.return_value = MagicMock(
            get_collection=MagicMock(side_effect=ConnectionError("down"))
        )

        cfg = Config(
            neo4j=Neo4jConfig(uri="bolt://test:7687", user="neo4j", password="pw"),
            qdrant=QdrantConfig(host="localhost", port=6333),
        )
        health = validate_connectivity(cfg)

        assert health["neo4j"] == "ok"
        assert "error" in health["qdrant"]

    @patch("worker.gastown.QdrantClient")
    @patch("worker.gastown.GraphDatabase")
    def test_neo4j_driver_closed_on_verify_failure(
        self,
        mock_gdb: MagicMock,
        mock_qdrant_cls: MagicMock,
    ) -> None:
        from worker.config import Config, Neo4jConfig, QdrantConfig

        driver = MagicMock()
        driver.verify_connectivity.side_effect = RuntimeError("timeout")
        mock_gdb.driver.return_value = driver
        mock_qdrant_cls.return_value = MagicMock()

        cfg = Config(
            neo4j=Neo4jConfig(uri="bolt://test:7687", user="neo4j", password="pw"),
            qdrant=QdrantConfig(host="localhost", port=6333),
        )
        health = validate_connectivity(cfg)

        assert "error" in health["neo4j"]
        driver.close.assert_called_once()

    @patch("worker.gastown.QdrantClient")
    @patch("worker.gastown.GraphDatabase")
    def test_qdrant_client_closed_on_collection_failure(
        self,
        mock_gdb: MagicMock,
        mock_qdrant_cls: MagicMock,
    ) -> None:
        from worker.config import Config, Neo4jConfig, QdrantConfig

        driver = MagicMock()
        mock_gdb.driver.return_value = driver

        client = MagicMock()
        client.get_collection.side_effect = RuntimeError("fail")
        mock_qdrant_cls.return_value = client

        cfg = Config(
            neo4j=Neo4jConfig(uri="bolt://test:7687", user="neo4j", password="pw"),
            qdrant=QdrantConfig(host="localhost", port=6333),
        )
        health = validate_connectivity(cfg)

        assert "error" in health["qdrant"]
        client.close.assert_called_once()


# ------------------------------------------------------------------
# setup_gastown
# ------------------------------------------------------------------


class TestSetupGastown:
    @patch("worker.gastown.write_mcp_config")
    @patch("worker.gastown.validate_connectivity", return_value={"neo4j": "ok", "qdrant": "ok"})
    @patch("worker.gastown.load_config")
    def test_success(
        self,
        mock_load: MagicMock,
        mock_validate: MagicMock,
        mock_write: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_load.return_value = MagicMock()
        mock_write.return_value = tmp_path / ".mcp.json"

        rc = setup_gastown(rig_root=tmp_path)
        assert rc == 0
        mock_write.assert_called_once()

    def test_no_rig_root_returns_1(self) -> None:
        with patch("worker.gastown.detect_rig_root", return_value=None):
            rc = setup_gastown()
        assert rc == 1

    @patch("worker.gastown.load_config", side_effect=RuntimeError("bad config"))
    def test_config_load_failure_returns_1(
        self, _mock: object, tmp_path: Path
    ) -> None:
        rc = setup_gastown(rig_root=tmp_path)
        assert rc == 1

    @patch("worker.gastown.write_mcp_config")
    @patch("worker.gastown.validate_connectivity", return_value={"neo4j": "error: down", "qdrant": "ok"})
    @patch("worker.gastown.load_config")
    def test_partial_health_still_succeeds(
        self,
        mock_load: MagicMock,
        mock_validate: MagicMock,
        mock_write: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_load.return_value = MagicMock()
        mock_write.return_value = tmp_path / ".mcp.json"

        rc = setup_gastown(rig_root=tmp_path)
        assert rc == 0
