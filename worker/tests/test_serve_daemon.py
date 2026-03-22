"""Tests for worker.serve_daemon — background ingest daemon."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from worker.config import (
    Config,
    ClusteringConfig,
    CoreConfig,
    Neo4jConfig,
    QdrantConfig,
)
from worker.serve_daemon import _run_daemon, run_daemon


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _cfg(**overrides) -> Config:
    cfg = Config(
        core=CoreConfig(log_level="warning"),
        neo4j=Neo4jConfig(uri="bolt://test:7687", user="neo4j", password="pw"),
        qdrant=QdrantConfig(host="localhost", port=6333),
        clustering=ClusteringConfig(enabled=False),
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# Patch targets use the actual module paths because _run_daemon uses
# local imports (inside the function body).
_P_PARSER = "worker.parsers.registry.get"
_P_PIPELINE = "worker.pipeline.Pipeline"
_P_WRITER = "worker.pipeline.writer.Writer"
_P_REGISTRY = "worker.models.resolver.ModelRegistry"
_P_BUILD = "worker.serve_daemon._build_sources"
_P_CLUSTER = "worker.clustering.scheduler.clustering_loop"


# ------------------------------------------------------------------
# _run_daemon — async core
# ------------------------------------------------------------------


class TestRunDaemon:
    @pytest.mark.asyncio
    @patch(_P_PARSER)
    @patch(_P_PIPELINE)
    @patch(_P_WRITER)
    @patch(_P_REGISTRY)
    @patch(_P_BUILD)
    async def test_no_sources_runs_idle(
        self,
        mock_build,
        mock_reg,
        mock_writer,
        mock_pipeline,
        mock_parser,
    ) -> None:
        """Daemon with no sources configured stays alive (idle ingest loop)."""
        mock_build.return_value = []

        pipeline_inst = mock_pipeline.return_value

        task = asyncio.create_task(_run_daemon(_cfg()))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        pipeline_inst.close.assert_called_once()

    @pytest.mark.asyncio
    @patch(_P_PARSER)
    @patch(_P_PIPELINE)
    @patch(_P_WRITER)
    @patch(_P_REGISTRY)
    @patch(_P_BUILD)
    async def test_processes_events_through_pipeline(
        self,
        mock_build,
        mock_reg,
        mock_writer,
        mock_pipeline,
        mock_parser,
    ) -> None:
        fake_source = MagicMock()
        fake_source.name.return_value = "files"

        async def fake_start(queue):
            await queue.put(
                {
                    "source_type": "file",
                    "source_id": "test.md",
                    "operation": "created",
                }
            )
            await asyncio.sleep(999)

        fake_source.start = fake_start
        mock_build.return_value = [fake_source]

        parser = MagicMock()
        parser.parse.return_value = [MagicMock()]
        mock_parser.return_value = parser

        pipeline_inst = mock_pipeline.return_value

        task = asyncio.create_task(_run_daemon(_cfg()))
        await asyncio.sleep(0.15)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        mock_parser.assert_called_with("file")
        parser.parse.assert_called_once()
        pipeline_inst.process.assert_called_once()

    @pytest.mark.asyncio
    @patch(_P_PARSER)
    @patch(_P_PIPELINE)
    @patch(_P_WRITER)
    @patch(_P_REGISTRY)
    @patch(_P_BUILD)
    async def test_shutdown_cancels_all_tasks(
        self,
        mock_build,
        mock_reg,
        mock_writer,
        mock_pipeline,
        mock_parser,
    ) -> None:
        fake_source = MagicMock()
        fake_source.name.return_value = "files"

        async def fake_start(queue):
            await asyncio.sleep(999)

        fake_source.start = fake_start
        mock_build.return_value = [fake_source]

        pipeline_inst = mock_pipeline.return_value

        task = asyncio.create_task(_run_daemon(_cfg()))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        pipeline_inst.close.assert_called_once()

    @pytest.mark.asyncio
    @patch(_P_CLUSTER)
    @patch(_P_PARSER)
    @patch(_P_PIPELINE)
    @patch(_P_WRITER)
    @patch(_P_REGISTRY)
    @patch(_P_BUILD)
    async def test_clustering_enabled_starts_task(
        self,
        mock_build,
        mock_reg,
        mock_writer,
        mock_pipeline,
        mock_parser,
        mock_cluster,
    ) -> None:
        mock_build.return_value = []

        async def fake_clustering(*args):
            await asyncio.sleep(999)

        mock_cluster.side_effect = fake_clustering

        cfg = _cfg(clustering=ClusteringConfig(enabled=True, cron="0 * * * *"))

        task = asyncio.create_task(_run_daemon(cfg))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        mock_cluster.assert_called_once()

    @pytest.mark.asyncio
    @patch(_P_PARSER)
    @patch(_P_PIPELINE)
    @patch(_P_WRITER)
    @patch(_P_REGISTRY)
    @patch(_P_BUILD)
    async def test_parser_error_does_not_crash(
        self,
        mock_build,
        mock_reg,
        mock_writer,
        mock_pipeline,
        mock_parser,
    ) -> None:
        fake_source = MagicMock()
        fake_source.name.return_value = "files"

        async def fake_start(queue):
            await queue.put(
                {
                    "source_type": "file",
                    "source_id": "bad.md",
                    "operation": "created",
                }
            )
            await queue.put(
                {
                    "source_type": "file",
                    "source_id": "good.md",
                    "operation": "created",
                }
            )
            await asyncio.sleep(999)

        fake_source.start = fake_start
        mock_build.return_value = [fake_source]

        parser = MagicMock()
        parser.parse.side_effect = [ValueError("bad"), [MagicMock()]]
        mock_parser.return_value = parser

        pipeline_inst = mock_pipeline.return_value

        task = asyncio.create_task(_run_daemon(_cfg()))
        await asyncio.sleep(0.3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert parser.parse.call_count == 2
        pipeline_inst.process.assert_called_once()

    @pytest.mark.asyncio
    @patch(_P_PARSER)
    @patch(_P_PIPELINE)
    @patch(_P_WRITER)
    @patch(_P_REGISTRY)
    @patch(_P_BUILD)
    async def test_mcp_server_not_started_in_daemon_mode(
        self,
        mock_build,
        mock_reg,
        mock_writer,
        mock_pipeline,
        mock_parser,
    ) -> None:
        """Daemon mode must NOT start the MCP stdio server.

        The MCP server uses stdio transport which conflicts with service
        managers (launchd, systemd) that redirect stdout to log files.
        JSON-RPC output would corrupt the log file and stdin is /dev/null
        so no client can connect anyway.
        """
        mock_build.return_value = []

        with patch("worker.mcp_server.FieldnotesServer") as mock_server_cls:
            task = asyncio.create_task(_run_daemon(_cfg()))
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        mock_server_cls.assert_not_called()


# ------------------------------------------------------------------
# run_daemon — synchronous entrypoint with retry logic
# ------------------------------------------------------------------


class TestRunDaemonSync:
    @patch("worker.serve_daemon.asyncio.run")
    @patch("worker.serve_daemon._check_qdrant")
    @patch("worker.serve_daemon._check_neo4j")
    @patch("worker.serve_daemon._setup_logging")
    @patch("worker.serve_daemon.load_config")
    def test_happy_path(
        self, mock_load, mock_logging, mock_neo4j, mock_qdrant, mock_arun
    ) -> None:
        mock_load.return_value = _cfg()
        run_daemon()
        mock_neo4j.assert_called_once()
        mock_qdrant.assert_called_once()
        mock_arun.assert_called_once()

    @patch("worker.serve_daemon.time.sleep")
    @patch("worker.serve_daemon.asyncio.run")
    @patch("worker.serve_daemon._check_qdrant")
    @patch("worker.serve_daemon._check_neo4j")
    @patch("worker.serve_daemon._setup_logging")
    @patch("worker.serve_daemon.load_config")
    def test_neo4j_retry_then_success(
        self, mock_load, mock_logging, mock_neo4j, mock_qdrant, mock_arun, mock_sleep
    ) -> None:
        mock_load.return_value = _cfg()
        mock_neo4j.side_effect = [ConnectionError("down"), None]

        run_daemon()

        assert mock_neo4j.call_count == 2
        mock_sleep.assert_called_once_with(2)

    @patch("worker.serve_daemon.time.sleep")
    @patch("worker.serve_daemon._check_neo4j")
    @patch("worker.serve_daemon._setup_logging")
    @patch("worker.serve_daemon.load_config")
    def test_neo4j_exhausts_retries(
        self, mock_load, mock_logging, mock_neo4j, mock_sleep
    ) -> None:
        mock_load.return_value = _cfg()
        mock_neo4j.side_effect = ConnectionError("down")

        with pytest.raises(SystemExit):
            run_daemon()

        assert mock_neo4j.call_count == 5

    @patch("worker.serve_daemon.time.sleep")
    @patch("worker.serve_daemon._check_qdrant")
    @patch("worker.serve_daemon._check_neo4j")
    @patch("worker.serve_daemon._setup_logging")
    @patch("worker.serve_daemon.load_config")
    def test_qdrant_exhausts_retries(
        self, mock_load, mock_logging, mock_neo4j, mock_qdrant, mock_sleep
    ) -> None:
        mock_load.return_value = _cfg()
        mock_qdrant.side_effect = ConnectionError("down")

        with pytest.raises(SystemExit):
            run_daemon()

        assert mock_qdrant.call_count == 5

    @patch("worker.serve_daemon.asyncio.run", side_effect=KeyboardInterrupt)
    @patch("worker.serve_daemon._check_qdrant")
    @patch("worker.serve_daemon._check_neo4j")
    @patch("worker.serve_daemon._setup_logging")
    @patch("worker.serve_daemon.load_config")
    def test_keyboard_interrupt_handled(
        self, mock_load, mock_logging, mock_neo4j, mock_qdrant, mock_arun
    ) -> None:
        mock_load.return_value = _cfg()
        # Should not raise — KeyboardInterrupt is caught gracefully
        run_daemon()
        mock_arun.assert_called_once()

    @patch("worker.serve_daemon.asyncio.run")
    @patch("worker.serve_daemon._check_qdrant")
    @patch("worker.serve_daemon._check_neo4j")
    @patch("worker.serve_daemon.load_config")
    def test_log_file_passed_to_setup_logging(
        self, mock_load, mock_neo4j, mock_qdrant, mock_arun
    ) -> None:
        from worker.serve_daemon import _LOG_FILE

        mock_load.return_value = _cfg()
        with patch("worker.serve_daemon._setup_logging") as mock_logging:
            run_daemon()
        mock_logging.assert_called_once_with(
            _cfg().core.log_level, log_file=_LOG_FILE,
        )
