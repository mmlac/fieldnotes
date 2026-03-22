"""Tests for ``fieldnotes init``."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from worker.init import init, update_infrastructure


@pytest.fixture()
def fn_home(tmp_path: Path) -> Path:
    """Return a temporary ~/.fieldnotes stand-in."""
    return tmp_path / ".fieldnotes"


class TestInit:
    def test_creates_dir_and_config(
        self,
        fn_home: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        config_path = fn_home / "config.toml"
        data_dir = fn_home / "data"

        with (
            patch("worker.init._FN_DIR", fn_home),
            patch("worker.init._CONFIG_PATH", config_path),
            patch("worker.init._DATA_DIR", data_dir),
        ):
            rc = init()

        assert rc == 0
        assert fn_home.is_dir()
        assert data_dir.is_dir()
        assert config_path.exists()
        # Verify it contains TOML content from the example
        text = config_path.read_text()
        assert "[core]" in text
        assert "[neo4j]" in text

        out = capsys.readouterr().out
        assert "Created" in out

    def test_existing_config_not_overwritten(
        self,
        fn_home: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        config_path = fn_home / "config.toml"
        data_dir = fn_home / "data"
        fn_home.mkdir(parents=True)
        config_path.write_text("existing")

        with (
            patch("worker.init._FN_DIR", fn_home),
            patch("worker.init._CONFIG_PATH", config_path),
            patch("worker.init._DATA_DIR", data_dir),
        ):
            rc = init()

        assert rc == 0
        assert config_path.read_text() == "existing"
        assert "already exists" in capsys.readouterr().out

    def test_ollama_detected(
        self,
        fn_home: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        config_path = fn_home / "config.toml"
        data_dir = fn_home / "data"

        with (
            patch("worker.init._FN_DIR", fn_home),
            patch("worker.init._CONFIG_PATH", config_path),
            patch("worker.init._DATA_DIR", data_dir),
            patch("worker.init._ollama_available", return_value=True),
        ):
            init()

        assert "Detected ollama" in capsys.readouterr().out

    def test_ollama_not_found(
        self,
        fn_home: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        config_path = fn_home / "config.toml"
        data_dir = fn_home / "data"

        with (
            patch("worker.init._FN_DIR", fn_home),
            patch("worker.init._CONFIG_PATH", config_path),
            patch("worker.init._DATA_DIR", data_dir),
            patch("worker.init._ollama_available", return_value=False),
        ):
            init()

        assert "ollama not found" in capsys.readouterr().err


class TestUpdateInfrastructure:
    def test_fails_when_infra_dir_missing(
        self,
        fn_home: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        infra_dir = fn_home / "infrastructure"

        with patch("worker.init._INFRA_DIR", infra_dir):
            rc = update_infrastructure()

        assert rc == 1
        assert "does not exist" in capsys.readouterr().err

    def test_overwrites_existing_files(
        self,
        fn_home: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        infra_dir = fn_home / "infrastructure"
        infra_dir.mkdir(parents=True)
        # Write a stale file that should be overwritten
        (infra_dir / "docker-compose.yml").write_text("old")

        with patch("worker.init._INFRA_DIR", infra_dir):
            rc = update_infrastructure()

        assert rc == 0
        # docker-compose.yml should now have the bundled content
        text = (infra_dir / "docker-compose.yml").read_text()
        assert text != "old"
        assert "services:" in text  # real compose content

        out = capsys.readouterr().out
        assert "docker-compose.yml" in out
        assert "Updated" in out

    def test_preserves_env_file(
        self,
        fn_home: Path,
    ) -> None:
        infra_dir = fn_home / "infrastructure"
        infra_dir.mkdir(parents=True)
        env_file = infra_dir / ".env"
        env_file.write_text("NEO4J_PASSWORD=secret\n")

        with patch("worker.init._INFRA_DIR", infra_dir):
            rc = update_infrastructure()

        assert rc == 0
        assert env_file.read_text() == "NEO4J_PASSWORD=secret\n"

    def test_cli_dispatches_update(self) -> None:
        from worker.cli import main

        with patch("worker.init.update_infrastructure", return_value=0) as mock:
            rc = main(["update"])

        assert rc == 0
        mock.assert_called_once()
