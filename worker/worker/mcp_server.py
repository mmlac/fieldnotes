"""MCP server exposing fieldnotes query capabilities over stdio transport.

Provides ``search`` and ``topics`` tools for LLM agents via the Model
Context Protocol.  Startup initialises Neo4j + Qdrant connections through
the existing ModelRegistry; shutdown closes them gracefully.

Usage::

    fieldnotes serve --mcp          # starts stdio MCP server
    fieldnotes -c path.toml serve --mcp
"""

from __future__ import annotations

import asyncio
import logging
import signal
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from worker.config import Config, load_config
from worker.models.resolver import ModelRegistry

# Ensure provider registration side-effects run.
import worker.models.providers.ollama  # noqa: F401

from worker.query.graph import GraphQuerier
from worker.query.hybrid import merge
from worker.query.vector import VectorQuerier
from worker.query.topics import (
    TopicQuerier,
    format_topic_detail,
    format_topic_gaps,
    format_topics_list,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS: list[Tool] = [
    Tool(
        name="search",
        description=(
            "Hybrid graph + vector search across the user's personal knowledge "
            "graph.  Returns structured context from both graph traversal and "
            "semantic similarity."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language search query",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum vector results (default 10)",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="list_topics",
        description="List all topics in the knowledge graph with document counts.",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="show_topic",
        description="Show details and linked documents for a specific topic.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Topic name to look up",
                },
            },
            "required": ["name"],
        },
    ),
    Tool(
        name="topic_gaps",
        description=(
            "Find cluster-discovered topics that are missing from the user's "
            "manual taxonomy."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
]


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


class FieldnotesServer:
    """Wraps the MCP server and manages query-layer connections."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._app = Server("fieldnotes")
        self._registry: ModelRegistry | None = None
        self._graph_querier: GraphQuerier | None = None
        self._vector_querier: VectorQuerier | None = None

        # Register handlers.
        self._app.list_tools()(self._list_tools)
        self._app.call_tool()(self._call_tool)

    # -- connection management ----------------------------------------------

    def _connect(self) -> None:
        """Initialise ModelRegistry, GraphQuerier, and VectorQuerier."""
        self._registry = ModelRegistry(self._cfg)
        self._graph_querier = GraphQuerier(self._registry, self._cfg.neo4j)
        self._vector_querier = VectorQuerier(self._registry, self._cfg.qdrant)
        logger.info("Fieldnotes MCP server: connections initialised")

    def _disconnect(self) -> None:
        """Close all connections gracefully."""
        if self._graph_querier is not None:
            self._graph_querier.close()
            self._graph_querier = None
        if self._vector_querier is not None:
            self._vector_querier.close()
            self._vector_querier = None
        self._registry = None
        logger.info("Fieldnotes MCP server: connections closed")

    # -- MCP handlers -------------------------------------------------------

    async def _list_tools(self) -> list[Tool]:
        return TOOLS

    async def _call_tool(
        self,
        name: str,
        arguments: dict,
    ) -> list[TextContent]:
        try:
            if name == "search":
                return self._handle_search(arguments)
            if name == "list_topics":
                return self._handle_list_topics()
            if name == "show_topic":
                return self._handle_show_topic(arguments)
            if name == "topic_gaps":
                return self._handle_topic_gaps()
            raise ValueError(f"Unknown tool: {name}")
        except Exception as exc:
            logger.exception("Tool %s failed", name)
            return [TextContent(type="text", text=f"error: {exc}")]

    # -- tool implementations -----------------------------------------------

    def _handle_search(self, arguments: dict) -> list[TextContent]:
        query = arguments["query"]
        top_k = arguments.get("top_k", 10)

        assert self._graph_querier is not None
        assert self._vector_querier is not None

        graph_result = self._graph_querier.query(query)
        vector_result = self._vector_querier.query(query, top_k=top_k)
        hybrid = merge(query, graph_result, vector_result)

        parts: list[str] = []
        if hybrid.errors:
            parts.append("Warnings: " + "; ".join(hybrid.errors))
        if hybrid.context:
            parts.append(hybrid.context)
        else:
            parts.append("No results found.")

        return [TextContent(type="text", text="\n\n".join(parts))]

    def _handle_list_topics(self) -> list[TextContent]:
        with TopicQuerier(self._cfg.neo4j) as querier:
            topics = querier.list_topics()
            text = format_topics_list(topics, use_json=True)
        return [TextContent(type="text", text=text)]

    def _handle_show_topic(self, arguments: dict) -> list[TextContent]:
        topic_name = arguments["name"]
        with TopicQuerier(self._cfg.neo4j) as querier:
            detail = querier.show_topic(topic_name)
            text = format_topic_detail(detail, use_json=True)
        return [TextContent(type="text", text=text)]

    def _handle_topic_gaps(self) -> list[TextContent]:
        with TopicQuerier(self._cfg.neo4j) as querier:
            gaps = querier.topic_gaps()
            text = format_topic_gaps(gaps, use_json=True)
        return [TextContent(type="text", text=text)]

    # -- run ----------------------------------------------------------------

    async def run(self) -> None:
        """Start the MCP server over stdio transport."""
        self._connect()

        loop = asyncio.get_running_loop()

        def _shutdown_handler() -> None:
            logger.info("Received shutdown signal")
            self._disconnect()

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, _shutdown_handler)

        try:
            async with stdio_server() as (read_stream, write_stream):
                await self._app.run(
                    read_stream,
                    write_stream,
                    self._app.create_initialization_options(),
                )
        finally:
            self._disconnect()


def run_server(config_path: Path | None = None) -> None:
    """Entry point: load config and run the MCP server."""
    cfg = load_config(config_path)
    server = FieldnotesServer(cfg)
    asyncio.run(server.run())
