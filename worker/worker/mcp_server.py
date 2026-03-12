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
import re
import signal
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from worker.circuit_breaker import all_breakers
from worker.config import Config, load_config
from worker.models.base import CompletionRequest
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
            "Search the user's personal knowledge graph using natural language. "
            "Combines graph traversal with semantic vector search and returns "
            "structured context including matched documents, entities, and "
            "relationships. Use this to answer questions about the user's notes, "
            "emails, code repositories, and Obsidian vault. "
            'Examples: "meetings about project X last month", '
            '"what do I know about machine learning", '
            '"emails from Alice about the budget".'
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Natural language search query. Be specific — include "
                        "names, topics, or timeframes for best results."
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 10)",
                    "default": 10,
                },
                "source_type": {
                    "type": "string",
                    "description": (
                        "Optional filter to restrict results to a specific "
                        "source type"
                    ),
                    "enum": ["file", "email", "obsidian", "repositories"],
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="ask",
        description=(
            "Ask a question about the user's knowledge graph and get a "
            "synthesized answer. Unlike 'search' which returns raw results, "
            "'ask' retrieves relevant context then uses an LLM to compose "
            "a clear, cited answer. Use this when you want a direct answer "
            "rather than documents to read through."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Natural language question to answer",
                },
                "source_type": {
                    "type": "string",
                    "description": "Optional filter to restrict context sources",
                    "enum": ["file", "email", "obsidian", "repositories"],
                },
            },
            "required": ["question"],
        },
    ),
    Tool(
        name="list_topics",
        description=(
            "List all topics in the knowledge graph with their document counts. "
            "Use this to understand what subjects the user has notes about and "
            "how extensively each topic is covered. Topics come from two sources: "
            "cluster-discovered (automatic) and user-tagged (from Obsidian #tags). "
            "Use the source filter to see only one kind."
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
            "Show detailed information about a specific topic including its "
            "linked documents, entities, and related topics. Use this after "
            "list_topics to drill into a particular area of interest. "
            'Example: show_topic("machine learning") to see all notes, '
            "emails, and code related to ML."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": (
                        "Exact topic name as returned by list_topics"
                    ),
                },
            },
            "required": ["name"],
        },
    ),
    Tool(
        name="topic_gaps",
        description=(
            "Find topics that were automatically discovered by clustering but "
            "are not yet in the user's manual taxonomy. These represent areas "
            "where the user has content but hasn't explicitly organized it. "
            "Useful for suggesting new topics to add or surfacing overlooked "
            "themes in the knowledge graph."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="ingest_status",
        description=(
            "Check the health and sync status of the fieldnotes ingestion "
            "pipeline. Returns source counts, last sync times, circuit breaker "
            "states for downstream services (Neo4j, Qdrant, LLM providers), "
            "and any errors."
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
        """Initialise ModelRegistry, GraphQuerier, and VectorQuerier.

        Uses try/except so that a failure in a later connection (e.g. Qdrant)
        cleans up already-opened connections (e.g. Neo4j) instead of leaking
        them.
        """
        self._registry = ModelRegistry(self._cfg)
        try:
            self._graph_querier = GraphQuerier(self._registry, self._cfg.neo4j)
            try:
                self._vector_querier = VectorQuerier(self._registry, self._cfg.qdrant)
            except Exception:
                self._graph_querier.close()
                self._graph_querier = None
                raise
        except Exception:
            self._registry = None
            raise
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
            if name == "ask":
                return await self._handle_ask(arguments)
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
            return [TextContent(type="text", text="error: internal server error")]

    # -- tool implementations -----------------------------------------------

    async def _handle_search(self, arguments: dict) -> list[TextContent]:
        query = arguments["query"]
        top_k = min(arguments.get("top_k", 10), 100)
        source_type: str | None = arguments.get("source_type")

        if self._graph_querier is None or self._vector_querier is None:
            raise RuntimeError("Server not initialised — call _connect() first")

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

    async def _handle_ask(self, arguments: dict) -> list[TextContent]:
        """Retrieve context via hybrid search, then synthesize an LLM answer."""
        question = arguments["question"]
        source_type: str | None = arguments.get("source_type")

        if self._graph_querier is None or self._vector_querier is None:
            raise RuntimeError("Server not initialised — call _connect() first")
        if self._registry is None:
            raise RuntimeError("Registry not initialised")

        # --- 1. Retrieve context (same as _handle_search) ---
        loop = asyncio.get_running_loop()

        graph_future = loop.run_in_executor(
            None, self._graph_querier.query, question
        )
        vector_future = loop.run_in_executor(
            None,
            lambda: self._vector_querier.query(
                question, top_k=20, source_type=source_type
            ),
        )

        graph_result, vector_result = await asyncio.gather(
            graph_future, vector_future, return_exceptions=True
        )

        if isinstance(graph_result, BaseException):
            logger.exception("Graph query raised", exc_info=graph_result)
            graph_result = GraphQueryResult(
                question=question, cypher="", error=str(graph_result)
            )
        if isinstance(vector_result, BaseException):
            logger.exception("Vector query raised", exc_info=vector_result)
            vector_result = VectorQueryResult(
                question=question, error=str(vector_result)
            )

        hybrid = merge(question, graph_result, vector_result)

        # --- 2. Build RAG prompt and call LLM ---
        context_text = hybrid.context
        if not context_text.strip():
            return [TextContent(
                type="text",
                text=(
                    "[Answer]\n"
                    "I don't have enough information in the knowledge graph "
                    "to answer this question.\n\n"
                    "[Sources]\nNone\n\n"
                    "[Confidence]\nno relevant context found"
                ),
            )]

        # Collect source_ids for citation tracking.
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

        # Deduplicate while preserving order.
        seen: set[str] = set()
        unique_ids: list[str] = []
        for sid in source_ids:
            if sid not in seen:
                seen.add(sid)
                unique_ids.append(sid)

        system_prompt = (
            "You are a knowledge assistant answering questions using ONLY "
            "the context provided below. Cite sources by their source_id "
            "when referencing specific information. If the context doesn't "
            "contain enough information to fully answer the question, say so "
            "clearly — do not fabricate information."
        )

        user_prompt = (
            f"Context:\n{context_text}\n\n"
            f"Question: {question}\n\n"
            "Provide a clear, concise answer based on the context above. "
            "Reference source_ids when citing specific facts."
        )

        # Truncate context if too large (reserve ~1k tokens for answer).
        # Rough heuristic: 1 token ≈ 4 chars. Most models handle 4k+ easily.
        max_context_chars = 60_000
        if len(user_prompt) > max_context_chars:
            truncated_context = context_text[: max_context_chars - 500]
            user_prompt = (
                f"Context:\n{truncated_context}\n[... truncated ...]\n\n"
                f"Question: {question}\n\n"
                "Provide a clear, concise answer based on the context above. "
                "Reference source_ids when citing specific facts."
            )

        try:
            model = self._registry.for_role("query")
        except KeyError:
            # Fall back to extraction role if query role not configured.
            model = self._registry.for_role("extraction")

        req = CompletionRequest(
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.2,
            timeout=120.0,
        )

        resp = await loop.run_in_executor(
            None, lambda: model.complete(req, task="ask")
        )

        # --- 3. Format response ---
        parts: list[str] = []

        if hybrid.errors:
            parts.append("Warnings: " + "; ".join(hybrid.errors))

        parts.append(f"[Answer]\n{resp.text}")

        if unique_ids:
            parts.append("[Sources]\n" + "\n".join(unique_ids))

        has_context = bool(hybrid.graph_results or hybrid.vector_results)
        sparse = (len(hybrid.graph_results) + len(hybrid.vector_results)) < 3
        if not has_context:
            parts.append("[Confidence]\nno relevant context found")
        elif sparse:
            parts.append("[Confidence]\nlow — limited context available")

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

        # --- Neo4j stats (reuse existing graph querier connection) ---
        neo4j_health = "ok"
        try:
            if self._graph_querier is None:
                raise RuntimeError("Graph querier not initialised")
            driver = self._graph_querier._graph._driver
            with driver.session() as session:
                sources: dict = {}
                for label in (
                    "File", "Email", "Commit", "Entity",
                    "Topic", "Chunk", "Image", "Repository",
                ):
                    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", label):
                        raise ValueError(f"Unsafe Neo4j label: {label!r}")
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

        # --- Qdrant stats (reuse existing vector querier connection) ---
        qdrant_cfg = self._cfg.qdrant
        qdrant_health = "ok"
        try:
            if self._vector_querier is None:
                raise RuntimeError("Vector querier not initialised")
            client = self._vector_querier._qdrant
            info = client.get_collection(qdrant_cfg.collection)
            result["vectors"] = {
                "count": info.points_count,
                "collection": qdrant_cfg.collection,
            }
        except Exception as exc:
            logger.exception("ingest_status: Qdrant query failed")
            qdrant_health = f"error: {exc}"
            result["vectors"] = {"count": 0, "collection": qdrant_cfg.collection}

        result["health"] = {
            "neo4j": neo4j_health,
            "qdrant": qdrant_health,
        }

        # --- Circuit breaker status ---
        breakers = all_breakers()
        if breakers:
            result["circuit_breakers"] = {
                name: cb.status() for name, cb in sorted(breakers.items())
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
