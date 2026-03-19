"""Tests for MCP server initialization, tool listing, and lifecycle."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import anyio
import pytest
from mcp.shared.session import SessionMessage
from mcp.types import JSONRPCMessage

from worker.config import Config, McpConfig, Neo4jConfig, QdrantConfig
from worker.mcp_server import (
    FieldnotesServer,
    TOOLS,
    _PrefixedReceiveStream,
    _extract_auth_token,
    _make_error_response,
)


# ------------------------------------------------------------------
# Tool definitions
# ------------------------------------------------------------------


class TestToolDefinitions:
    """Verify the TOOLS list has the expected entries and schemas."""

    def test_all_tools_present(self) -> None:
        names = {t.name for t in TOOLS}
        assert names == {"search", "ask", "list_topics", "show_topic", "topic_gaps", "ingest_status", "timeline", "suggest_connections", "digest"}

    def test_search_schema_requires_query(self) -> None:
        search = next(t for t in TOOLS if t.name == "search")
        assert "query" in search.inputSchema["required"]

    def test_search_schema_has_top_k(self) -> None:
        search = next(t for t in TOOLS if t.name == "search")
        assert "top_k" in search.inputSchema["properties"]

    def test_search_schema_has_source_type_enum(self) -> None:
        search = next(t for t in TOOLS if t.name == "search")
        source_type = search.inputSchema["properties"]["source_type"]
        assert "enum" in source_type
        assert "file" in source_type["enum"]

    def test_show_topic_requires_name(self) -> None:
        st = next(t for t in TOOLS if t.name == "show_topic")
        assert "name" in st.inputSchema["required"]

    def test_list_topics_source_enum(self) -> None:
        lt = next(t for t in TOOLS if t.name == "list_topics")
        source = lt.inputSchema["properties"]["source"]
        assert set(source["enum"]) == {"all", "cluster", "user"}

    def test_topic_gaps_no_required(self) -> None:
        tg = next(t for t in TOOLS if t.name == "topic_gaps")
        assert "required" not in tg.inputSchema

    def test_ingest_status_no_required(self) -> None:
        ist = next(t for t in TOOLS if t.name == "ingest_status")
        assert "required" not in ist.inputSchema


# ------------------------------------------------------------------
# Server lifecycle
# ------------------------------------------------------------------


def _make_cfg() -> Config:
    return Config(
        neo4j=Neo4jConfig(uri="bolt://localhost:7687", user="neo4j", password="test"),
        qdrant=QdrantConfig(host="localhost", port=6333, collection="fieldnotes"),
    )


class TestFieldnotesServerInit:
    def test_init_creates_server(self) -> None:
        cfg = _make_cfg()
        server = FieldnotesServer(cfg)
        assert server._graph_querier is None
        assert server._vector_querier is None

    @patch("worker.mcp_server.VectorQuerier")
    @patch("worker.mcp_server.GraphQuerier")
    @patch("worker.mcp_server.ModelRegistry")
    def test_connect_initialises_queriers(
        self,
        mock_registry: MagicMock,
        mock_gq: MagicMock,
        mock_vq: MagicMock,
    ) -> None:
        cfg = _make_cfg()
        server = FieldnotesServer(cfg)
        server._connect()
        mock_registry.assert_called_once_with(cfg)
        mock_gq.assert_called_once()
        mock_vq.assert_called_once()
        assert server._graph_querier is not None
        assert server._vector_querier is not None

    @patch("worker.mcp_server.VectorQuerier")
    @patch("worker.mcp_server.GraphQuerier")
    @patch("worker.mcp_server.ModelRegistry")
    def test_disconnect_closes_queriers(
        self,
        mock_registry: MagicMock,
        mock_gq: MagicMock,
        mock_vq: MagicMock,
    ) -> None:
        cfg = _make_cfg()
        server = FieldnotesServer(cfg)
        server._connect()
        gq = server._graph_querier
        vq = server._vector_querier
        server._disconnect()
        gq.close.assert_called_once()
        vq.close.assert_called_once()
        assert server._graph_querier is None
        assert server._vector_querier is None

    @patch("worker.mcp_server.VectorQuerier")
    @patch("worker.mcp_server.GraphQuerier")
    @patch("worker.mcp_server.ModelRegistry")
    def test_disconnect_idempotent(
        self,
        mock_registry: MagicMock,
        mock_gq: MagicMock,
        mock_vq: MagicMock,
    ) -> None:
        cfg = _make_cfg()
        server = FieldnotesServer(cfg)
        # disconnect without connect should not raise
        server._disconnect()
        assert server._graph_querier is None


class TestListTools:
    @pytest.mark.asyncio
    async def test_list_tools_returns_all(self) -> None:
        cfg = _make_cfg()
        server = FieldnotesServer(cfg)
        result = await server._list_tools()
        assert result == TOOLS
        assert len(result) == 9


class TestCallToolErrors:
    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self) -> None:
        cfg = _make_cfg()
        server = FieldnotesServer(cfg)
        result = await server._call_tool("nonexistent", {})
        assert len(result) == 1
        assert "error" in result[0].text


# ------------------------------------------------------------------
# Auth helpers
# ------------------------------------------------------------------


def _make_init_message(auth_token: str | None = None) -> SessionMessage:
    """Build a minimal MCP ``initialize`` JSON-RPC request."""
    meta: dict = {}
    if auth_token is not None:
        meta["auth_token"] = auth_token
    params: dict = {
        "protocolVersion": "2025-03-26",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "0.1"},
    }
    if meta:
        params["_meta"] = meta
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": params,
    }
    return SessionMessage(JSONRPCMessage.model_validate(payload))


def _make_ping_message() -> SessionMessage:
    """Build a JSON-RPC ``ping`` request (not initialize)."""
    payload = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "ping",
    }
    return SessionMessage(JSONRPCMessage.model_validate(payload))


class TestExtractAuthToken:
    def test_extracts_token_from_initialize(self) -> None:
        msg = _make_init_message(auth_token="secret-123")
        assert _extract_auth_token(msg) == "secret-123"

    def test_returns_none_when_no_meta(self) -> None:
        msg = _make_init_message(auth_token=None)
        assert _extract_auth_token(msg) is None

    def test_returns_none_for_non_initialize(self) -> None:
        msg = _make_ping_message()
        assert _extract_auth_token(msg) is None


class TestMakeErrorResponse:
    def test_error_response_structure(self) -> None:
        resp = _make_error_response(42)
        root = resp.message.root
        assert root.id == 42
        assert root.error.code == -32001
        assert "Unauthorized" in root.error.message

    def test_error_response_with_string_id(self) -> None:
        resp = _make_error_response("req-abc")
        assert resp.message.root.id == "req-abc"


class TestPrefixedReceiveStream:
    @pytest.mark.asyncio
    async def test_yields_prefix_then_inner(self) -> None:
        prefix = _make_init_message("tok")
        second = _make_ping_message()

        send, recv = anyio.create_memory_object_stream[SessionMessage | Exception](8)
        await send.send(second)
        await send.aclose()

        stream = _PrefixedReceiveStream(prefix, recv)
        first_out = await stream.receive()
        assert first_out is prefix
        second_out = await stream.receive()
        assert second_out is second

    @pytest.mark.asyncio
    async def test_aclose_delegates(self) -> None:
        prefix = _make_init_message("tok")
        send, recv = anyio.create_memory_object_stream[SessionMessage | Exception](1)
        stream = _PrefixedReceiveStream(prefix, recv)
        await stream.aclose()
        # Inner stream should be closed — sending should fail.
        with pytest.raises((anyio.ClosedResourceError, anyio.BrokenResourceError)):
            await send.send(prefix)


class TestAuthGate:
    @pytest.mark.asyncio
    async def test_valid_token_returns_stream(self) -> None:
        msg = _make_init_message(auth_token="good-token")
        send_r, recv_r = anyio.create_memory_object_stream[SessionMessage | Exception](8)
        send_w, recv_w = anyio.create_memory_object_stream[SessionMessage](8)

        await send_r.send(msg)
        await send_r.aclose()

        result = await FieldnotesServer._auth_gate(recv_r, send_w, "good-token")
        # First message from the returned stream should be the original init.
        first = await result.receive()
        assert first is msg

    @pytest.mark.asyncio
    async def test_wrong_token_raises_and_sends_error(self) -> None:
        msg = _make_init_message(auth_token="wrong")
        send_r, recv_r = anyio.create_memory_object_stream[SessionMessage | Exception](8)
        send_w, recv_w = anyio.create_memory_object_stream[SessionMessage](8)

        await send_r.send(msg)

        with pytest.raises(PermissionError, match="invalid or missing token"):
            await FieldnotesServer._auth_gate(recv_r, send_w, "correct-token")

        # An error response should have been sent to the write stream.
        err = await recv_w.receive()
        assert err.message.root.error.code == -32001

    @pytest.mark.asyncio
    async def test_missing_token_raises(self) -> None:
        msg = _make_init_message(auth_token=None)
        send_r, recv_r = anyio.create_memory_object_stream[SessionMessage | Exception](8)
        send_w, recv_w = anyio.create_memory_object_stream[SessionMessage](8)

        await send_r.send(msg)

        with pytest.raises(PermissionError):
            await FieldnotesServer._auth_gate(recv_r, send_w, "expected")

    @pytest.mark.asyncio
    async def test_exception_on_stream_propagates(self) -> None:
        send_r, recv_r = anyio.create_memory_object_stream[SessionMessage | Exception](8)
        send_w, recv_w = anyio.create_memory_object_stream[SessionMessage](8)

        await send_r.send(RuntimeError("connection lost"))

        with pytest.raises(RuntimeError, match="connection lost"):
            await FieldnotesServer._auth_gate(recv_r, send_w, "token")


class TestConfigAuthToken:
    def test_mcp_config_default_none(self) -> None:
        cfg = McpConfig()
        assert cfg.auth_token is None

    def test_mcp_config_with_token(self) -> None:
        cfg = McpConfig(auth_token="my-secret")
        assert cfg.auth_token == "my-secret"


# ------------------------------------------------------------------
# Thread pool executor (thread pool exhaustion fix)
# ------------------------------------------------------------------


class TestExecutorLifecycle:
    @patch("worker.mcp_server.VectorQuerier")
    @patch("worker.mcp_server.GraphQuerier")
    @patch("worker.mcp_server.ModelRegistry")
    def test_connect_creates_executor(
        self,
        mock_registry: MagicMock,
        mock_gq: MagicMock,
        mock_vq: MagicMock,
    ) -> None:
        cfg = _make_cfg()
        server = FieldnotesServer(cfg)
        assert server._executor is None
        server._connect()
        assert server._executor is not None

    @patch("worker.mcp_server.VectorQuerier")
    @patch("worker.mcp_server.GraphQuerier")
    @patch("worker.mcp_server.ModelRegistry")
    def test_disconnect_shuts_down_executor(
        self,
        mock_registry: MagicMock,
        mock_gq: MagicMock,
        mock_vq: MagicMock,
    ) -> None:
        cfg = _make_cfg()
        server = FieldnotesServer(cfg)
        server._connect()
        executor = server._executor
        server._disconnect()
        assert server._executor is None
        # The executor should have been shut down.
        assert executor._shutdown  # type: ignore[union-attr]

    def test_disconnect_without_connect_does_not_raise(self) -> None:
        cfg = _make_cfg()
        server = FieldnotesServer(cfg)
        # No connect — executor is None. Should not raise.
        server._disconnect()
        assert server._executor is None


# ------------------------------------------------------------------
# Shutdown signal cancels task, not double-disconnect (race fix)
# ------------------------------------------------------------------


class TestShutdownSignalHandler:
    @patch("worker.mcp_server.VectorQuerier")
    @patch("worker.mcp_server.GraphQuerier")
    @patch("worker.mcp_server.ModelRegistry")
    @patch("worker.mcp_server.stdio_server")
    def test_disconnect_called_once_on_cancellation(
        self,
        mock_stdio: MagicMock,
        mock_registry: MagicMock,
        mock_gq: MagicMock,
        mock_vq: MagicMock,
    ) -> None:
        """Signal-driven cancellation must not cause a double _disconnect call."""
        import asyncio
        from unittest.mock import AsyncMock, patch as _patch

        cfg = _make_cfg()
        server = FieldnotesServer(cfg)

        disconnect_calls = []
        original_disconnect = server._disconnect

        def counting_disconnect():
            disconnect_calls.append(1)
            original_disconnect()

        server._disconnect = counting_disconnect  # type: ignore[method-assign]

        # stdio_server context manager that immediately cancels the current task.
        class _CancellingStdio:
            async def __aenter__(self):
                asyncio.current_task().cancel()
                read_s, _ = anyio.create_memory_object_stream(1)
                write_s, _ = anyio.create_memory_object_stream(1)
                return read_s, write_s

            async def __aexit__(self, *args):
                pass

        mock_stdio.return_value = _CancellingStdio()

        asyncio.run(server.run())

        assert disconnect_calls == [1], (
            f"_disconnect should be called exactly once, got {len(disconnect_calls)}"
        )


# ------------------------------------------------------------------
# Input validation
# ------------------------------------------------------------------


class TestInputValidation:
    @pytest.mark.asyncio
    async def test_search_empty_query_returns_error(self) -> None:
        cfg = _make_cfg()
        server = FieldnotesServer(cfg)
        result = await server._call_tool("search", {"query": ""})
        assert len(result) == 1
        assert "error" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_search_whitespace_query_returns_error(self) -> None:
        cfg = _make_cfg()
        server = FieldnotesServer(cfg)
        result = await server._call_tool("search", {"query": "   "})
        assert len(result) == 1
        assert "error" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_search_invalid_source_type_returns_error(self) -> None:
        cfg = _make_cfg()
        server = FieldnotesServer(cfg)
        result = await server._call_tool("search", {"query": "hello", "source_type": "bogus"})
        assert len(result) == 1
        assert "error" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_ask_empty_question_returns_error(self) -> None:
        cfg = _make_cfg()
        server = FieldnotesServer(cfg)
        result = await server._call_tool("ask", {"question": ""})
        assert len(result) == 1
        assert "error" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_ask_invalid_source_type_returns_error(self) -> None:
        cfg = _make_cfg()
        server = FieldnotesServer(cfg)
        result = await server._call_tool("ask", {"question": "what?", "source_type": "unknown"})
        assert len(result) == 1
        assert "error" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_show_topic_empty_name_returns_error(self) -> None:
        cfg = _make_cfg()
        server = FieldnotesServer(cfg)
        result = await server._call_tool("show_topic", {"name": ""})
        assert len(result) == 1
        assert "error" in result[0].text.lower()

    @pytest.mark.asyncio
    @patch("worker.mcp_server.VectorQuerier")
    @patch("worker.mcp_server.GraphQuerier")
    @patch("worker.mcp_server.ModelRegistry")
    @patch("worker.mcp_server.is_corpus_empty", return_value=False)
    async def test_search_top_k_zero_clamped_to_one(
        self,
        mock_empty: MagicMock,
        mock_registry: MagicMock,
        mock_gq_cls: MagicMock,
        mock_vq_cls: MagicMock,
    ) -> None:
        from worker.query.graph import GraphQueryResult
        from worker.query.vector import VectorQueryResult

        cfg = Config(
            neo4j=Neo4jConfig(uri="bolt://localhost:7687", user="neo4j", password="test"),
            qdrant=QdrantConfig(host="localhost", port=6333, collection="fieldnotes"),
        )
        server = FieldnotesServer(cfg)
        server._connect()

        server._graph_querier.query = MagicMock(
            return_value=GraphQueryResult(question="q", cypher="", raw_results=[])
        )
        server._vector_querier.query = MagicMock(
            return_value=VectorQueryResult(question="q", results=[])
        )

        # top_k=0 must be clamped to >= 1 and not raise.
        await server._call_tool("search", {"query": "test", "top_k": 0})
        call_args = server._vector_querier.query.call_args
        # top_k is passed as keyword argument
        top_k_used = call_args.kwargs.get("top_k")
        assert top_k_used is not None and top_k_used >= 1
