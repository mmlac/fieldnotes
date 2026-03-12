"""Tests for ``fieldnotes init``."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from worker.init import init


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
