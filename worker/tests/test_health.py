"""Tests for worker.health — HTTP health-check endpoint."""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from worker.config import (
    Config,
    ClusteringConfig,
    CoreConfig,
    HealthConfig,
    Neo4jConfig,
    QdrantConfig,
)
from worker.health import (
    HealthServer,
    _build_health_payload,
    _check_neo4j_health,
    _check_qdrant_health,
)


def _cfg(**overrides) -> Config:
    cfg = Config(
        core=CoreConfig(log_level="warning"),
        neo4j=Neo4jConfig(uri="bolt://test:7687", user="neo4j", password="pw"),
        qdrant=QdrantConfig(host="localhost", port=6333),
        clustering=ClusteringConfig(enabled=False),
        health=HealthConfig(enabled=True, port=0, bind="127.0.0.1"),
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ------------------------------------------------------------------
# Component checks
# ------------------------------------------------------------------


class TestNeo4jHealthCheck:
    @pytest.mark.asyncio
    @patch("worker.health.asyncio.get_running_loop")
    async def test_returns_ok_on_success(self, mock_loop) -> None:
        mock_driver = MagicMock()
        mock_driver.verify_connectivity.return_value = None
        mock_driver.close.return_value = None

        async def fake_executor(executor, fn):
            return fn()

        mock_loop.return_value.run_in_executor = fake_executor

        with patch("neo4j.GraphDatabase.driver", return_value=mock_driver):
            result = await _check_neo4j_health(_cfg())

        assert result["status"] == "ok"

    @pytest.mark.asyncio
    @patch("worker.health.asyncio.get_running_loop")
    async def test_returns_unhealthy_on_failure(self, mock_loop) -> None:
        mock_driver = MagicMock()
        mock_driver.verify_connectivity.side_effect = ConnectionError("down")
        mock_driver.close.return_value = None

        async def fake_executor(executor, fn):
            return fn()

        mock_loop.return_value.run_in_executor = fake_executor

        with patch("neo4j.GraphDatabase.driver", return_value=mock_driver):
            result = await _check_neo4j_health(_cfg())

        assert result["status"] == "unhealthy"
        assert result["error"] == "ConnectionError"

    @pytest.mark.asyncio
    @patch("worker.health.asyncio.get_running_loop")
    async def test_reuses_provided_driver(self, mock_loop) -> None:
        """Provided driver is used directly — no new GraphDatabase.driver() call."""
        mock_driver = MagicMock()
        mock_driver.verify_connectivity.return_value = None

        async def fake_executor(executor, fn):
            return fn()

        mock_loop.return_value.run_in_executor = fake_executor

        with patch("neo4j.GraphDatabase.driver") as mock_factory:
            result = await _check_neo4j_health(_cfg(), driver=mock_driver)

        assert result["status"] == "ok"
        mock_factory.assert_not_called()
        mock_driver.verify_connectivity.assert_called_once()
        mock_driver.close.assert_not_called()

    @pytest.mark.asyncio
    @patch("worker.health.asyncio.get_running_loop")
    async def test_reuses_provided_driver_on_failure(self, mock_loop) -> None:
        mock_driver = MagicMock()
        mock_driver.verify_connectivity.side_effect = ConnectionError("down")

        async def fake_executor(executor, fn):
            return fn()

        mock_loop.return_value.run_in_executor = fake_executor

        with patch("neo4j.GraphDatabase.driver") as mock_factory:
            result = await _check_neo4j_health(_cfg(), driver=mock_driver)

        assert result["status"] == "unhealthy"
        assert result["error"] == "ConnectionError"
        mock_factory.assert_not_called()
        mock_driver.close.assert_not_called()


class TestQdrantHealthCheck:
    @pytest.mark.asyncio
    @patch("worker.health.asyncio.get_running_loop")
    async def test_returns_ok_on_success(self, mock_loop) -> None:
        mock_client = MagicMock()
        mock_client.get_collections.return_value = []
        mock_client.close.return_value = None

        async def fake_executor(executor, fn):
            return fn()

        mock_loop.return_value.run_in_executor = fake_executor

        with patch("qdrant_client.QdrantClient", return_value=mock_client):
            result = await _check_qdrant_health(_cfg())

        assert result["status"] == "ok"

    @pytest.mark.asyncio
    @patch("worker.health.asyncio.get_running_loop")
    async def test_returns_unhealthy_on_failure(self, mock_loop) -> None:
        mock_client = MagicMock()
        mock_client.get_collections.side_effect = ConnectionError("down")
        mock_client.close.return_value = None

        async def fake_executor(executor, fn):
            return fn()

        mock_loop.return_value.run_in_executor = fake_executor

        with patch("qdrant_client.QdrantClient", return_value=mock_client):
            result = await _check_qdrant_health(_cfg())

        assert result["status"] == "unhealthy"
        assert result["error"] == "ConnectionError"

    @pytest.mark.asyncio
    @patch("worker.health.asyncio.get_running_loop")
    async def test_reuses_provided_client(self, mock_loop) -> None:
        """Provided client is used directly — no new QdrantClient() call."""
        mock_client = MagicMock()
        mock_client.get_collections.return_value = []

        async def fake_executor(executor, fn):
            return fn()

        mock_loop.return_value.run_in_executor = fake_executor

        with patch("qdrant_client.QdrantClient") as mock_factory:
            result = await _check_qdrant_health(_cfg(), client=mock_client)

        assert result["status"] == "ok"
        mock_factory.assert_not_called()
        mock_client.get_collections.assert_called_once()
        mock_client.close.assert_not_called()

    @pytest.mark.asyncio
    @patch("worker.health.asyncio.get_running_loop")
    async def test_reuses_provided_client_on_failure(self, mock_loop) -> None:
        mock_client = MagicMock()
        mock_client.get_collections.side_effect = ConnectionError("down")

        async def fake_executor(executor, fn):
            return fn()

        mock_loop.return_value.run_in_executor = fake_executor

        with patch("qdrant_client.QdrantClient") as mock_factory:
            result = await _check_qdrant_health(_cfg(), client=mock_client)

        assert result["status"] == "unhealthy"
        assert result["error"] == "ConnectionError"
        mock_factory.assert_not_called()
        mock_client.close.assert_not_called()


# ------------------------------------------------------------------
# Health payload
# ------------------------------------------------------------------


class TestBuildHealthPayload:
    @pytest.mark.asyncio
    @patch("worker.health._check_qdrant_health", return_value={"status": "ok"})
    @patch("worker.health._check_neo4j_health", return_value={"status": "ok"})
    async def test_all_ok(self, mock_neo4j, mock_qdrant) -> None:
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        start = time.monotonic() - 60.0

        payload = await _build_health_payload(_cfg(), queue, start)

        assert payload["status"] == "ok"
        assert payload["components"]["neo4j"]["status"] == "ok"
        assert payload["components"]["qdrant"]["status"] == "ok"
        assert payload["components"]["pipeline_queue"]["depth"] == 0
        assert payload["components"]["pipeline_queue"]["max"] == 100
        assert payload["components"]["uptime_seconds"] >= 60.0

    @pytest.mark.asyncio
    @patch("worker.health._check_qdrant_health", return_value={"status": "ok"})
    @patch("worker.health._check_neo4j_health", return_value={"status": "unhealthy", "error": "ConnectionError"})
    async def test_degraded_when_component_unhealthy(self, mock_neo4j, mock_qdrant) -> None:
        payload = await _build_health_payload(_cfg(), None, time.monotonic())

        assert payload["status"] == "degraded"
        assert payload["components"]["neo4j"]["status"] == "unhealthy"

    @pytest.mark.asyncio
    @patch("worker.health._check_qdrant_health", return_value={"status": "ok"})
    @patch("worker.health._check_neo4j_health", return_value={"status": "ok"})
    async def test_no_queue_omits_queue_info(self, mock_neo4j, mock_qdrant) -> None:
        payload = await _build_health_payload(_cfg(), None, time.monotonic())

        assert "pipeline_queue" not in payload["components"]


# ------------------------------------------------------------------
# HealthServer — integration via real TCP
# ------------------------------------------------------------------


class TestHealthServer:
    @pytest.mark.asyncio
    @patch("worker.health._check_qdrant_health", return_value={"status": "ok"})
    @patch("worker.health._check_neo4j_health", return_value={"status": "ok"})
    async def test_get_health_returns_200(self, mock_neo4j, mock_qdrant) -> None:
        cfg = _cfg(health=HealthConfig(enabled=True, port=0, bind="127.0.0.1"))
        server = HealthServer(cfg)
        await server.start()
        assert server._server is not None

        # Get the actual bound port
        addr = server._server.sockets[0].getsockname()
        port = addr[1]

        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n")
        await writer.drain()
        data = await asyncio.wait_for(reader.read(4096), timeout=5.0)
        writer.close()
        await writer.wait_closed()

        await server.stop()

        response = data.decode()
        assert "200 OK" in response
        body = json.loads(response.split("\r\n\r\n", 1)[1])
        assert body["status"] == "ok"
        assert "neo4j" in body["components"]
        assert "qdrant" in body["components"]

    @pytest.mark.asyncio
    @patch("worker.health._check_qdrant_health", return_value={"status": "ok"})
    @patch("worker.health._check_neo4j_health", return_value={"status": "unhealthy", "error": "timeout"})
    async def test_degraded_returns_503(self, mock_neo4j, mock_qdrant) -> None:
        cfg = _cfg(health=HealthConfig(enabled=True, port=0, bind="127.0.0.1"))
        server = HealthServer(cfg)
        await server.start()
        assert server._server is not None

        addr = server._server.sockets[0].getsockname()
        port = addr[1]

        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n")
        await writer.drain()
        data = await asyncio.wait_for(reader.read(4096), timeout=5.0)
        writer.close()
        await writer.wait_closed()

        await server.stop()

        response = data.decode()
        assert "503" in response
        body = json.loads(response.split("\r\n\r\n", 1)[1])
        assert body["status"] == "degraded"

    @pytest.mark.asyncio
    async def test_unknown_path_returns_404(self) -> None:
        cfg = _cfg(health=HealthConfig(enabled=True, port=0, bind="127.0.0.1"))
        server = HealthServer(cfg)
        await server.start()
        assert server._server is not None

        addr = server._server.sockets[0].getsockname()
        port = addr[1]

        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(b"GET /unknown HTTP/1.1\r\nHost: localhost\r\n\r\n")
        await writer.drain()
        data = await asyncio.wait_for(reader.read(4096), timeout=5.0)
        writer.close()
        await writer.wait_closed()

        await server.stop()

        assert "404" in data.decode()


# ------------------------------------------------------------------
# Config parsing
# ------------------------------------------------------------------


class TestHealthConfig:
    def test_defaults(self) -> None:
        cfg = HealthConfig()
        assert cfg.enabled is False
        assert cfg.port == 9100
        assert cfg.bind == "127.0.0.1"

    def test_config_parse_health_section(self) -> None:
        from worker.config import _parse

        raw = {
            "neo4j": {"password": "test"},
            "health": {"enabled": True, "port": 8080, "bind": "0.0.0.0"},
        }
        cfg = _parse(raw)
        assert cfg.health.enabled is True
        assert cfg.health.port == 8080
        assert cfg.health.bind == "0.0.0.0"

    def test_config_parse_no_health_section(self) -> None:
        from worker.config import _parse

        raw = {"neo4j": {"password": "test"}}
        cfg = _parse(raw)
        assert cfg.health.enabled is False
        assert cfg.health.port == 9100
