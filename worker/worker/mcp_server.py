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
import json
import logging
import signal
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from neo4j import GraphDatabase
from qdrant_client import QdrantClient

from worker.config import Config, load_config
from worker.models.resolver import ModelRegistry

# Ensure provider registration side-effects run.
import worker.models.providers.ollama  # noqa: F401

from worker.query.graph import GraphQuerier, GraphQueryResult
from worker.query.hybrid import merge
from worker.query.vector import VectorQuerier, VectorQueryResult
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
            "Search the fieldnotes knowledge graph using natural language. "
            "Combines graph traversal (Neo4j) with semantic vector search "
            "(Qdrant) and returns structured context."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 10,
                },
                "source_type": {
                    "type": "string",
                    "description": "Filter results by source type",
                    "enum": ["file", "email", "obsidian", "repositories"],
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="list_topics",
        description=(
            "List all topics in the knowledge graph with document counts. "
            "Topics come from two sources: cluster-discovered (automatic) "
            "and user-tagged (from Obsidian #tags)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "enum": ["all", "cluster", "user"],
                    "default": "all",
                    "description": "Filter by topic source",
                },
            },
        },
    ),
    Tool(
        name="show_topic",
        description=(
            "Show detailed information about a specific topic, including "
            "linked documents, entities, and related topics."
        ),
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
            "Find cluster-discovered topics that don't correspond to any "
            "user-defined tag. These represent knowledge areas the system "
            "found but the user hasn't explicitly organized."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="ingest_status",
        description=(
            "Check the health and sync status of the fieldnotes ingestion "
            "pipeline. Returns source counts, last sync times, and any errors."
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
                return await self._handle_search(arguments)
            if name == "list_topics":
                return self._handle_list_topics(arguments)
            if name == "show_topic":
                return self._handle_show_topic(arguments)
            if name == "topic_gaps":
                return self._handle_topic_gaps()
            if name == "ingest_status":
                return self._handle_ingest_status()
            raise ValueError(f"Unknown tool: {name}")
        except Exception as exc:
            logger.exception("Tool %s failed", name)
            return [TextContent(type="text", text=f"error: {exc}")]

    # -- tool implementations -----------------------------------------------

    async def _handle_search(self, arguments: dict) -> list[TextContent]:
        query = arguments["query"]
        top_k = arguments.get("top_k", 10)
        source_type: str | None = arguments.get("source_type")

        assert self._graph_querier is not None
        assert self._vector_querier is not None

        loop = asyncio.get_running_loop()

        # Run graph and vector queries concurrently in the thread pool.
        graph_future = loop.run_in_executor(
            None, self._graph_querier.query, query
        )
        vector_future = loop.run_in_executor(
            None,
            lambda: self._vector_querier.query(
                query, top_k=top_k, source_type=source_type
            ),
        )

        graph_result, vector_result = await asyncio.gather(
            graph_future, vector_future, return_exceptions=True
        )

        # Handle partial failures: if one query raised, build an error result.
        if isinstance(graph_result, BaseException):
            logger.exception("Graph query raised", exc_info=graph_result)
            graph_result = GraphQueryResult(
                question=query, cypher="", error=str(graph_result)
            )
        if isinstance(vector_result, BaseException):
            logger.exception("Vector query raised", exc_info=vector_result)
            vector_result = VectorQueryResult(
                question=query, error=str(vector_result)
            )

        hybrid = merge(query, graph_result, vector_result)

        # Build structured response with [Graph context], [Semantic context],
        # [Answer], and source_ids sections.
        parts: list[str] = []

        if hybrid.errors:
            parts.append("Warnings: " + "; ".join(hybrid.errors))

        if hybrid.context:
            parts.append(hybrid.context)

        # [Answer] section from graph chain synthesis.
        if graph_result.answer:
            parts.append(f"[Answer]\n{graph_result.answer}")

        # [Sources] section for traceability.
        source_ids: list[str] = []
        for row in hybrid.graph_results:
            for value in row.values():
                if isinstance(value, dict):
                    sid = value.get("source_id")
                    if sid:
                        source_ids.append(str(sid))
            sid = row.get("source_id")
            if sid:
                source_ids.append(str(sid))
        for vr in hybrid.vector_results:
            if vr.source_id:
                source_ids.append(vr.source_id)

        if source_ids:
            # Deduplicate while preserving order.
            seen: set[str] = set()
            unique_ids: list[str] = []
            for sid in source_ids:
                if sid not in seen:
                    seen.add(sid)
                    unique_ids.append(sid)
            parts.append("[Sources]\n" + "\n".join(unique_ids))

        if not parts or (len(parts) == 1 and parts[0].startswith("Warnings")):
            parts.append("No results found.")

        return [TextContent(type="text", text="\n\n".join(parts))]

    def _handle_list_topics(self, arguments: dict) -> list[TextContent]:
        source = arguments.get("source", "all")
        with TopicQuerier(self._cfg.neo4j) as querier:
            topics = querier.list_topics()
            if source != "all":
                topics = [t for t in topics if t.source == source]
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

    def _handle_ingest_status(self) -> list[TextContent]:
        result: dict = {}

        # --- Neo4j stats ---
        neo4j_cfg = self._cfg.neo4j
        neo4j_health = "ok"
        try:
            driver = GraphDatabase.driver(
                neo4j_cfg.uri,
                auth=(neo4j_cfg.user, neo4j_cfg.password),
            )
            try:
                with driver.session() as session:
                    sources: dict = {}
                    for label in (
                        "File", "Email", "Commit", "Entity",
                        "Topic", "Chunk", "Image", "Repository",
                    ):
                        row = session.run(
                            f"MATCH (n:`{label}`) "
                            "RETURN count(n) AS cnt, "
                            "max(n.modified_at) AS last_sync"
                        ).single()
                        sources[label.lower()] = {
                            "count": row["cnt"],
                            "last_sync": str(row["last_sync"]) if row["last_sync"] else None,
                        }

                    # Entity counts by type
                    entity_rows = session.run(
                        "MATCH (e:Entity) "
                        "RETURN e.type AS type, count(e) AS cnt "
                        "ORDER BY cnt DESC"
                    ).data()
                    entities_by_type = {
                        r["type"]: r["cnt"] for r in entity_rows
                    }
                    entity_total = sum(entities_by_type.values())

                    # Topic counts by source (cluster vs user)
                    topic_rows = session.run(
                        "MATCH (t:Topic) "
                        "RETURN t.source AS source, count(t) AS cnt"
                    ).data()
                    topics_by_source = {
                        r["source"]: r["cnt"] for r in topic_rows
                    }
            finally:
                driver.close()
        except Exception as exc:
            logger.exception("ingest_status: Neo4j query failed")
            neo4j_health = f"error: {exc}"
            sources = {}
            entities_by_type = {}
            entity_total = 0
            topics_by_source = {}

        result["sources"] = sources
        result["entities"] = {
            "total": entity_total,
            "by_type": entities_by_type,
        }
        result["topics"] = topics_by_source

        # --- Qdrant stats ---
        qdrant_cfg = self._cfg.qdrant
        qdrant_health = "ok"
        try:
            client = QdrantClient(host=qdrant_cfg.host, port=qdrant_cfg.port)
            try:
                info = client.get_collection(qdrant_cfg.collection)
                result["vectors"] = {
                    "count": info.points_count,
                    "collection": qdrant_cfg.collection,
                }
            finally:
                client.close()
        except Exception as exc:
            logger.exception("ingest_status: Qdrant query failed")
            qdrant_health = f"error: {exc}"
            result["vectors"] = {"count": 0, "collection": qdrant_cfg.collection}

        result["health"] = {
            "neo4j": neo4j_health,
            "qdrant": qdrant_health,
        }

        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

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
