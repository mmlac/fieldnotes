"""Tests for the worker.main entrypoint.

Covers: _setup_logging, _build_sources, _check_neo4j, _check_qdrant, _run
(event processing loop, shutdown/signal handling, error propagation), and
the main() CLI entrypoint with health-check retry logic.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import MagicMock, patch

import pytest

from worker.config import (
    Config,
    CoreConfig,
    Neo4jConfig,
    QdrantConfig,
    SourceConfig,
    ClusteringConfig,
)
from worker.main import (
    _setup_logging,
    _build_sources,
    _check_neo4j,
    _check_qdrant,
    _run,
    main,
    SOURCE_CLASSES,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _cfg(**overrides) -> Config:
    """Build a minimal Config for testing."""
    cfg = Config(
        core=CoreConfig(log_level="warning"),
        neo4j=Neo4jConfig(uri="bolt://test:7687", user="neo4j", password="pw"),
        qdrant=QdrantConfig(host="localhost", port=6333),
        clustering=ClusteringConfig(enabled=False),
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ------------------------------------------------------------------
# _setup_logging
# ------------------------------------------------------------------


class TestSetupLogging:
    def test_sets_level_from_string(self):
        with patch("logging.getLogger") as mock_get:
            root = MagicMock()
            mock_get.return_value = root
            _setup_logging("debug")
            root.setLevel.assert_called_once_with(logging.DEBUG)

    def test_defaults_to_info_on_invalid(self):
        with patch("logging.getLogger") as mock_get:
            root = MagicMock()
            mock_get.return_value = root
            _setup_logging("not_a_level")
            root.setLevel.assert_called_once_with(logging.INFO)

    def test_case_insensitive(self):
        with patch("logging.getLogger") as mock_get:
            root = MagicMock()
            mock_get.return_value = root
            _setup_logging("WARNING")
            root.setLevel.assert_called_once_with(logging.WARNING)

    def test_clears_existing_handlers_to_prevent_duplicates(self):
        """_setup_logging must replace existing handlers, not accumulate them.

        When the CLI calls logging.basicConfig() before run_daemon() calls
        _setup_logging(), the root logger already has a handler.  Without the
        clear step every log message would be emitted twice.
        """
        with patch("logging.getLogger") as mock_get:
            root = MagicMock()
            mock_get.return_value = root
            _setup_logging("info")
            root.handlers.clear.assert_called_once()

    def test_handler_uses_stderr(self, capsys):
        """All log output must go to stderr — never stdout.

        stdout is reserved for the MCP stdio transport (``fieldnotes serve
        --mcp``).  Any log line written to stdout would corrupt the
        JSON-RPC stream consumed by the MCP client.
        """
        import sys
        import logging as _logging

        # Use a real root logger isolated to this test.
        real_root = _logging.getLogger()
        original_handlers = real_root.handlers[:]
        original_level = real_root.level
        try:
            real_root.handlers.clear()
            _setup_logging("warning")
            real_root.warning("test-stderr-routing")
            captured = capsys.readouterr()
            assert "test-stderr-routing" in captured.err
            assert "test-stderr-routing" not in captured.out
        finally:
            real_root.handlers.clear()
            real_root.handlers.extend(original_handlers)
            real_root.setLevel(original_level)


# ------------------------------------------------------------------
# _check_neo4j
# ------------------------------------------------------------------


class TestCheckNeo4j:
    @patch("worker.main.GraphDatabase")
    def test_success(self, mock_gdb):
        driver = MagicMock()
        mock_gdb.driver.return_value = driver

        cfg = _cfg()
        _check_neo4j(cfg)

        mock_gdb.driver.assert_called_once_with(
            cfg.neo4j.uri, auth=(cfg.neo4j.user, cfg.neo4j.password)
        )
        driver.verify_connectivity.assert_called_once()
        driver.close.assert_called_once()

    @patch("worker.main.GraphDatabase")
    def test_close_called_on_failure(self, mock_gdb):
        driver = MagicMock()
        driver.verify_connectivity.side_effect = ConnectionError("refused")
        mock_gdb.driver.return_value = driver

        with pytest.raises(ConnectionError):
            _check_neo4j(_cfg())

        driver.close.assert_called_once()


# ------------------------------------------------------------------
# _check_qdrant
# ------------------------------------------------------------------


class TestCheckQdrant:
    @patch("worker.main.QdrantClient")
    def test_success(self, mock_qc_cls):
        client = MagicMock()
        mock_qc_cls.return_value = client

        cfg = _cfg()
        _check_qdrant(cfg)

        mock_qc_cls.assert_called_once_with(
            host=cfg.qdrant.host, port=cfg.qdrant.port
        )
        client.get_collections.assert_called_once()
        client.close.assert_called_once()

    @patch("worker.main.QdrantClient")
    def test_close_called_on_failure(self, mock_qc_cls):
        client = MagicMock()
        client.get_collections.side_effect = ConnectionError("refused")
        mock_qc_cls.return_value = client

        with pytest.raises(ConnectionError):
            _check_qdrant(_cfg())

        client.close.assert_called_once()


# ------------------------------------------------------------------
# _build_sources
# ------------------------------------------------------------------


class TestBuildSources:
    def test_known_sources_instantiated(self):
        fake_source = MagicMock()
        fake_source.name.return_value = "files"
        fake_cls = MagicMock(return_value=fake_source)

        settings = {"watch_dir": "/tmp"}
        cfg = _cfg(
            sources={"files": SourceConfig(name="files", settings=settings)}
        )

        with patch.dict(SOURCE_CLASSES, {"files": fake_cls}):
            result = _build_sources(cfg)

        assert len(result) == 1
        fake_cls.assert_called_once()
        fake_source.configure.assert_called_once_with(settings)

    def test_unknown_source_skipped(self):
        cfg = _cfg(
            sources={"unknown_src": SourceConfig(name="unknown_src", settings={})}
        )
        result = _build_sources(cfg)
        assert result == []

    def test_empty_sources(self):
        cfg = _cfg(sources={})
        assert _build_sources(cfg) == []

    def test_multiple_sources(self):
        fake_files = MagicMock()
        fake_files.name.return_value = "files"
        fake_obsidian = MagicMock()
        fake_obsidian.name.return_value = "obsidian"

        cfg = _cfg(
            sources={
                "files": SourceConfig(name="files", settings={"dir": "/a"}),
                "obsidian": SourceConfig(name="obsidian", settings={"vault": "/b"}),
            }
        )

        with patch.dict(
            SOURCE_CLASSES,
            {"files": MagicMock(return_value=fake_files),
             "obsidian": MagicMock(return_value=fake_obsidian)},
        ):
            result = _build_sources(cfg)

        assert len(result) == 2


# ------------------------------------------------------------------
# _run — event processing loop
# ------------------------------------------------------------------


class TestRun:
    @pytest.mark.asyncio
    @patch("worker.main.Pipeline")
    @patch("worker.main.Writer")
    @patch("worker.main.ModelRegistry")
    @patch("worker.main._build_sources")
    async def test_no_sources_returns_early(
        self, mock_build, mock_reg, mock_writer, mock_pipeline
    ):
        mock_build.return_value = []
        pipeline_inst = mock_pipeline.return_value

        await _run(_cfg())

        pipeline_inst.close.assert_called_once()

    @pytest.mark.asyncio
    @patch("worker.main.get_parser")
    @patch("worker.main.Pipeline")
    @patch("worker.main.Writer")
    @patch("worker.main.ModelRegistry")
    @patch("worker.main._build_sources")
    async def test_processes_events_from_queue(
        self, mock_build, mock_reg, mock_writer, mock_pipeline, mock_get_parser
    ):
        """Events put on the queue are parsed and processed through the pipeline."""
        fake_source = MagicMock()
        fake_source.name.return_value = "files"

        # Source pushes one event then the stop signal fires
        async def fake_start(queue):
            await queue.put({
                "source_type": "file",
                "source_id": "test.md",
                "operation": "created",
            })
            # Give the consumer a moment to process
            await asyncio.sleep(0.05)

        fake_source.start = fake_start
        mock_build.return_value = [fake_source]

        parser = MagicMock()
        parser.parse.return_value = [MagicMock()]
        mock_get_parser.return_value = parser

        pipeline_inst = mock_pipeline.return_value

        task = asyncio.create_task(_run(_cfg()))
        # Let it process, then stop
        await asyncio.sleep(0.15)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        mock_get_parser.assert_called_with("file")
        parser.parse.assert_called_once()
        pipeline_inst.process.assert_called_once()

    @pytest.mark.asyncio
    @patch("worker.main.get_parser")
    @patch("worker.main.Pipeline")
    @patch("worker.main.Writer")
    @patch("worker.main.ModelRegistry")
    @patch("worker.main._build_sources")
    async def test_parser_error_does_not_crash_loop(
        self, mock_build, mock_reg, mock_writer, mock_pipeline, mock_get_parser
    ):
        """A parsing error on one event should not stop processing."""
        fake_source = MagicMock()
        fake_source.name.return_value = "files"

        async def fake_start(queue):
            await queue.put({
                "source_type": "file",
                "source_id": "bad.md",
                "operation": "created",
            })
            await queue.put({
                "source_type": "file",
                "source_id": "good.md",
                "operation": "created",
            })
            await asyncio.sleep(0.1)

        fake_source.start = fake_start
        mock_build.return_value = [fake_source]

        parser = MagicMock()
        parser.parse.side_effect = [
            ValueError("bad parse"),
            [MagicMock()],  # second event succeeds
        ]
        mock_get_parser.return_value = parser

        pipeline_inst = mock_pipeline.return_value

        task = asyncio.create_task(_run(_cfg()))
        await asyncio.sleep(0.3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # Both events were attempted
        assert parser.parse.call_count == 2
        # Only the good event made it to the pipeline
        pipeline_inst.process.assert_called_once()

    @pytest.mark.asyncio
    @patch("worker.main.Pipeline")
    @patch("worker.main.Writer")
    @patch("worker.main.ModelRegistry")
    @patch("worker.main._build_sources")
    async def test_shutdown_cancels_sources_and_closes_pipeline(
        self, mock_build, mock_reg, mock_writer, mock_pipeline
    ):
        """On shutdown, source tasks are cancelled and pipeline is closed."""
        fake_source = MagicMock()
        fake_source.name.return_value = "files"

        async def fake_start(queue):
            await asyncio.sleep(999)  # run forever until cancelled

        fake_source.start = fake_start
        mock_build.return_value = [fake_source]

        pipeline_inst = mock_pipeline.return_value

        task = asyncio.create_task(_run(_cfg()))
        await asyncio.sleep(0.05)
        # Simulate SIGINT by cancelling the task
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        pipeline_inst.close.assert_called_once()

    @pytest.mark.asyncio
    @patch("worker.main.clustering_loop")
    @patch("worker.main.Pipeline")
    @patch("worker.main.Writer")
    @patch("worker.main.ModelRegistry")
    @patch("worker.main._build_sources")
    async def test_clustering_enabled_starts_background_task(
        self, mock_build, mock_reg, mock_writer, mock_pipeline, mock_cluster
    ):
        """When clustering is enabled, clustering_loop is started as a task."""
        fake_source = MagicMock()
        fake_source.name.return_value = "files"

        async def fake_start(queue):
            await asyncio.sleep(999)

        fake_source.start = fake_start
        mock_build.return_value = [fake_source]

        async def fake_clustering(*args):
            await asyncio.sleep(999)

        mock_cluster.side_effect = fake_clustering

        cfg = _cfg(clustering=ClusteringConfig(enabled=True, cron="0 * * * *"))

        task = asyncio.create_task(_run(cfg))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        mock_cluster.assert_called_once()

    @pytest.mark.asyncio
    @patch("worker.main.clustering_loop")
    @patch("worker.main.Pipeline")
    @patch("worker.main.Writer")
    @patch("worker.main.ModelRegistry")
    @patch("worker.main._build_sources")
    async def test_clustering_disabled_no_background_task(
        self, mock_build, mock_reg, mock_writer, mock_pipeline, mock_cluster
    ):
        """When clustering is disabled, clustering_loop is not called."""
        mock_build.return_value = []  # triggers early return
        mock_pipeline.return_value = MagicMock()

        cfg = _cfg(clustering=ClusteringConfig(enabled=False))
        await _run(cfg)

        mock_cluster.assert_not_called()


# ------------------------------------------------------------------
# main() — CLI entrypoint with health-check retries
# ------------------------------------------------------------------


class TestMain:
    @patch("worker.main.asyncio.run")
    @patch("worker.main._check_qdrant")
    @patch("worker.main._check_neo4j")
    @patch("worker.main._setup_logging")
    @patch("worker.main.load_config")
    def test_happy_path(
        self, mock_load, mock_logging, mock_neo4j, mock_qdrant, mock_arun
    ):
        mock_load.return_value = _cfg()

        with patch("sys.argv", ["worker.main"]):
            main()

        mock_load.assert_called_once()
        mock_logging.assert_called_once_with("warning")
        mock_neo4j.assert_called_once()
        mock_qdrant.assert_called_once()
        mock_arun.assert_called_once()

    @patch("worker.main.time.sleep")
    @patch("worker.main.asyncio.run")
    @patch("worker.main._check_qdrant")
    @patch("worker.main._check_neo4j")
    @patch("worker.main._setup_logging")
    @patch("worker.main.load_config")
    def test_neo4j_retry_then_success(
        self, mock_load, mock_logging, mock_neo4j, mock_qdrant, mock_arun, mock_sleep
    ):
        """Neo4j health check retries on failure then succeeds."""
        mock_load.return_value = _cfg()
        mock_neo4j.side_effect = [ConnectionError("down"), None]

        with patch("sys.argv", ["worker.main"]):
            main()

        assert mock_neo4j.call_count == 2
        mock_sleep.assert_called_once_with(2)  # 2^1

    @patch("worker.main.time.sleep")
    @patch("worker.main._check_neo4j")
    @patch("worker.main._setup_logging")
    @patch("worker.main.load_config")
    def test_neo4j_exhausts_retries(
        self, mock_load, mock_logging, mock_neo4j, mock_sleep
    ):
        """Neo4j health check exits after max retries."""
        mock_load.return_value = _cfg()
        mock_neo4j.side_effect = ConnectionError("down")

        with patch("sys.argv", ["worker.main"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

        assert mock_neo4j.call_count == 5

    @patch("worker.main.time.sleep")
    @patch("worker.main.asyncio.run")
    @patch("worker.main._check_qdrant")
    @patch("worker.main._check_neo4j")
    @patch("worker.main._setup_logging")
    @patch("worker.main.load_config")
    def test_qdrant_retry_then_success(
        self, mock_load, mock_logging, mock_neo4j, mock_qdrant, mock_arun, mock_sleep
    ):
        """Qdrant health check retries on failure then succeeds."""
        mock_load.return_value = _cfg()
        mock_qdrant.side_effect = [ConnectionError("down"), None]

        with patch("sys.argv", ["worker.main"]):
            main()

        assert mock_qdrant.call_count == 2

    @patch("worker.main.time.sleep")
    @patch("worker.main._check_qdrant")
    @patch("worker.main._check_neo4j")
    @patch("worker.main._setup_logging")
    @patch("worker.main.load_config")
    def test_qdrant_exhausts_retries(
        self, mock_load, mock_logging, mock_neo4j, mock_qdrant, mock_sleep
    ):
        """Qdrant health check exits after max retries."""
        mock_load.return_value = _cfg()
        mock_qdrant.side_effect = ConnectionError("down")

        with patch("sys.argv", ["worker.main"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

        assert mock_qdrant.call_count == 5

    @patch("worker.main.asyncio.run")
    @patch("worker.main._check_qdrant")
    @patch("worker.main._check_neo4j")
    @patch("worker.main._setup_logging")
    @patch("worker.main.load_config")
    def test_config_path_forwarded(
        self, mock_load, mock_logging, mock_neo4j, mock_qdrant, mock_arun
    ):
        """--config flag is forwarded to load_config."""
        mock_load.return_value = _cfg()

        with patch("sys.argv", ["worker.main", "--config", "/tmp/test.toml"]):
            main()

        from pathlib import Path
        mock_load.assert_called_once_with(Path("/tmp/test.toml"))
