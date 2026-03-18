"""Lightweight HTTP health-check endpoint for process managers.

Exposes ``GET /health`` on a configurable port (default 9100, disabled by
default).  Returns JSON with per-component status so that systemd
``WatchdogSec``, Kubernetes liveness probes, or monitoring tools can poll
daemon health without an MCP tool call.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from worker.config import Config

logger = logging.getLogger(__name__)

_HTTP_200 = "HTTP/1.1 200 OK\r\n"
_HTTP_503 = "HTTP/1.1 503 Service Unavailable\r\n"
_HTTP_404 = "HTTP/1.1 404 Not Found\r\n"
_HEADERS = "Content-Type: application/json\r\nConnection: close\r\n"


def _json_response(status_line: str, body: dict[str, Any]) -> bytes:
    payload = json.dumps(body).encode()
    return (
        f"{status_line}{_HEADERS}Content-Length: {len(payload)}\r\n\r\n"
    ).encode() + payload


async def _check_neo4j_health(cfg: Config, driver: Any = None) -> dict[str, Any]:
    """Check Neo4j connectivity.  Runs in a thread to avoid blocking.

    If *driver* is provided (an existing ``neo4j.Driver`` instance) it is
    reused — no new connection is opened.  Otherwise a short-lived driver is
    created and closed after the probe.
    """
    def _probe() -> dict[str, Any]:
        if driver is not None:
            try:
                driver.verify_connectivity()
                return {"status": "ok"}
            except Exception as exc:
                return {"status": "unhealthy", "error": type(exc).__name__}
        from neo4j import GraphDatabase
        d = GraphDatabase.driver(
            cfg.neo4j.uri,
            auth=(cfg.neo4j.user, cfg.neo4j.password),
        )
        try:
            d.verify_connectivity()
            return {"status": "ok"}
        except Exception as exc:
            return {"status": "unhealthy", "error": type(exc).__name__}
        finally:
            d.close()

    try:
        return await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(None, _probe),
            timeout=5.0,
        )
    except asyncio.TimeoutError:
        return {"status": "unhealthy", "error": "timeout"}


async def _check_qdrant_health(cfg: Config, client: Any = None) -> dict[str, Any]:
    """Check Qdrant connectivity.  Runs in a thread to avoid blocking.

    If *client* is provided (an existing ``QdrantClient`` instance) it is
    reused — no new connection is opened.  Otherwise a short-lived client is
    created and closed after the probe.
    """
    def _probe() -> dict[str, Any]:
        if client is not None:
            try:
                client.get_collections()
                return {"status": "ok"}
            except Exception as exc:
                return {"status": "unhealthy", "error": type(exc).__name__}
        from qdrant_client import QdrantClient
        c = QdrantClient(host=cfg.qdrant.host, port=cfg.qdrant.port)
        try:
            c.get_collections()
            return {"status": "ok"}
        except Exception as exc:
            return {"status": "unhealthy", "error": type(exc).__name__}
        finally:
            c.close()

    try:
        return await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(None, _probe),
            timeout=5.0,
        )
    except asyncio.TimeoutError:
        return {"status": "unhealthy", "error": "timeout"}


async def _build_health_payload(
    cfg: Config,
    queue: asyncio.Queue[Any] | None,
    start_time: float,
    neo4j_driver: Any = None,
    qdrant_client: Any = None,
) -> dict[str, Any]:
    """Assemble the full health-check response payload."""
    neo4j_result, qdrant_result = await asyncio.gather(
        _check_neo4j_health(cfg, driver=neo4j_driver),
        _check_qdrant_health(cfg, client=qdrant_client),
    )

    components: dict[str, Any] = {
        "neo4j": neo4j_result,
        "qdrant": qdrant_result,
    }

    if queue is not None:
        components["pipeline_queue"] = {
            "depth": queue.qsize(),
            "max": queue.maxsize if queue.maxsize > 0 else None,
        }

    components["uptime_seconds"] = round(time.monotonic() - start_time, 1)

    all_ok = all(
        c.get("status") == "ok"
        for c in components.values()
        if isinstance(c, dict) and "status" in c
    )

    return {
        "status": "ok" if all_ok else "degraded",
        "components": components,
    }


class HealthServer:
    """Minimal asyncio TCP server that speaks just enough HTTP to serve
    ``GET /health`` as JSON."""

    def __init__(
        self,
        cfg: Config,
        queue: asyncio.Queue[Any] | None = None,
        start_time: float | None = None,
        neo4j_driver: Any = None,
        qdrant_client: Any = None,
    ) -> None:
        self._cfg = cfg
        self._queue = queue
        self._start_time = start_time or time.monotonic()
        self._server: asyncio.Server | None = None
        self._neo4j_driver = neo4j_driver
        self._qdrant_client = qdrant_client

    async def _handle(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            request_line = await asyncio.wait_for(
                reader.readline(), timeout=5.0,
            )
            parts = request_line.decode(errors="replace").split()
            method = parts[0] if parts else ""
            path = parts[1] if len(parts) > 1 else ""

            if method == "GET" and path == "/health":
                payload = await _build_health_payload(
                    self._cfg, self._queue, self._start_time,
                    neo4j_driver=self._neo4j_driver,
                    qdrant_client=self._qdrant_client,
                )
                status_line = (
                    _HTTP_200 if payload["status"] == "ok" else _HTTP_503
                )
                writer.write(_json_response(status_line, payload))
            else:
                writer.write(
                    _json_response(_HTTP_404, {"error": "not found"})
                )
        except Exception:
            logger.debug("Health endpoint request error", exc_info=True)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def start(self) -> None:
        bind = self._cfg.health.bind
        port = self._cfg.health.port
        self._server = await asyncio.start_server(
            self._handle, bind, port,
        )
        logger.info("Health endpoint listening on %s:%d", bind, port)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            logger.info("Health endpoint stopped")

    async def run(self) -> None:
        """Start and serve until cancelled."""
        await self.start()
        assert self._server is not None
        async with self._server:
            await self._server.serve_forever()
