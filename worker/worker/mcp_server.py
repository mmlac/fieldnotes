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
import concurrent.futures
import hmac
import json
import logging
import re
import signal
from pathlib import Path
from typing import Any

import anyio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from neo4j import GraphDatabase
from mcp.shared.session import SessionMessage
from mcp.types import JSONRPCMessage, TextContent, Tool

from worker.circuit_breaker import all_breakers
from worker.config import Config, load_config
from worker.models.base import CompletionRequest
from worker.models.resolver import ModelRegistry

# Ensure provider registration side-effects run.
import worker.models.providers.ollama  # noqa: F401

from worker.query import EMPTY_CORPUS_MESSAGE, is_corpus_empty
from worker.query.graph import GraphQuerier, GraphQueryResult
from worker.query.hybrid import merge
from worker.query.vector import VectorQuerier, VectorQueryResult
from worker.query.connections import ConnectionQuerier
from worker.query.topics import (
    TopicQuerier,
    format_topic_detail,
    format_topic_gaps,
    format_topics_list,
)
from worker.query.timeline import (
    TimelineQuerier,
    VALID_SOURCE_TYPES as _TIMELINE_SOURCE_TYPES,
)
from worker.query.digest import DigestQuerier
from worker.query._time import parse_relative_time
from worker.query.person import PersonProfile, get_profile
from worker.cli.person import BriefError, generate_brief
from worker.cli.itinerary import (
    BriefError as ItineraryBriefError,
    _ThreadDetail as _ItineraryThreadDetail,
    _fetch_thread_detail as _fetch_itinerary_thread_detail,
    _resolve_completion as _resolve_itinerary_completion,
    generate_event_briefs as _generate_itinerary_briefs,
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
                        "Optional filter to restrict results to a specific source type"
                    ),
                    "enum": ["file", "email", "obsidian", "repositories"],
                },
                "rerank": {
                    "type": "boolean",
                    "description": (
                        "Apply the second-stage cross-encoder reranker to vector "
                        "results (default: true if configured)."
                    ),
                    "default": True,
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
                "rerank": {
                    "type": "boolean",
                    "description": (
                        "Apply the second-stage cross-encoder reranker to retrieved "
                        "context (default: true if configured)."
                    ),
                    "default": True,
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
                    "description": ("Exact topic name as returned by list_topics"),
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
        name="suggest_connections",
        description=(
            "Find documents that are semantically similar but not explicitly "
            "linked in the knowledge graph. Useful for discovering latent "
            "relationships across note types — for example, Obsidian notes "
            "related to OmniFocus tasks, or emails related to code commits."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "source_id": {
                    "type": "string",
                    "description": "Focus on a specific document by its source_id",
                },
                "source_type": {
                    "type": "string",
                    "description": "Focus seeds on a specific source type (e.g. 'file', 'obsidian')",
                },
                "threshold": {
                    "type": "number",
                    "description": "Minimum cosine similarity score (0–1). Default: 0.82",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of suggestions to return. Default: 20",
                },
                "cross_source": {
                    "type": "boolean",
                    "description": (
                        "If true, only show connections between different source types "
                        "(high-value mode). Default: false"
                    ),
                },
            },
        },
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
    Tool(
        name="timeline",
        description=(
            "Show a chronological timeline of activity across all indexed sources "
            "within a time range. Answers 'what was I working on?' by listing "
            "File modifications, Task completions, Emails, and Commits ordered by "
            "time. Use this to get a quick overview of recent activity or to "
            "recall what happened during a specific period. "
            "Examples: 'what did I work on yesterday?', "
            "'show activity from last week in my notes'."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "since": {
                    "type": "string",
                    "description": (
                        "Start of the time window. ISO 8601 or relative: "
                        "'24h', '7d', '2w', '3m', 'yesterday', 'last week'. "
                        "Default: '24h'"
                    ),
                    "default": "24h",
                },
                "until": {
                    "type": "string",
                    "description": "End of the time window. Default: 'now'",
                    "default": "now",
                },
                "source_type": {
                    "type": "string",
                    "description": (
                        "Filter to one source type: "
                        "obsidian, omnifocus, gmail, file, repositories, apps"
                    ),
                    "enum": [
                        "obsidian",
                        "omnifocus",
                        "gmail",
                        "file",
                        "repositories",
                        "apps",
                    ],
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum entries to return (default: 50)",
                    "default": 50,
                },
            },
        },
    ),
    Tool(
        name="persons_inspect",
        description=(
            "Show all SAME_AS / NEVER_SAME_AS edges incident on a Person. "
            "Use this to audit the identity chain before splitting or "
            "confirming a merge. Identifier may be an email "
            "(alice@example.com), a slack id (slack:T123/U456), or an "
            "exact display name."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "identifier": {
                    "type": "string",
                    "description": "Email, slack:T../U.. id, or exact name",
                },
            },
            "required": ["identifier"],
        },
    ),
    Tool(
        name="persons_split",
        description=(
            "Break a SAME_AS edge between a person cluster and one of "
            "its members and install a NEVER_SAME_AS block so the next "
            "reconcile pass does not recreate the merge."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "identifier": {
                    "type": "string",
                    "description": "Identifier of the cluster member to split from",
                },
                "member": {
                    "type": "string",
                    "description": "Identifier of the member to split off",
                },
            },
            "required": ["identifier", "member"],
        },
    ),
    Tool(
        name="persons_confirm",
        description=(
            "Lock a good merge as user-confirmed. Writes a SAME_AS edge "
            "with match_type='user_confirmed' and confidence=1.0 — "
            "future reconcile runs treat it as ground truth and never "
            "overwrite it."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "a": {"type": "string", "description": "First identifier"},
                "b": {"type": "string", "description": "Second identifier"},
            },
            "required": ["a", "b"],
        },
    ),
    Tool(
        name="persons_merge",
        description=(
            "Manually merge two persons the automated chain missed "
            "(different names, no shared email/slack id). Equivalent to "
            "persons_confirm but used when no SAME_AS edge exists yet."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "a": {"type": "string", "description": "First identifier"},
                "b": {"type": "string", "description": "Second identifier"},
            },
            "required": ["a", "b"],
        },
    ),
    Tool(
        name="person",
        description=(
            "Profile a person from the user's knowledge graph. Returns "
            "recent interactions, top topics, related people, open tasks, "
            "files mentioning them, and the person's identity cluster. "
            "Resolution order: email (e.g. 'alice@example.com') → "
            "slack-user prefix (e.g. 'slack-user:<team>/<user>') → "
            "fuzzy name match. Ambiguous name matches return a "
            "disambiguation error listing candidates so the caller can "
            "retry with an email."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "identifier": {
                    "type": "string",
                    "description": (
                        "Email, slack-user:<team>/<user>, or name fragment"
                    ),
                },
                "since": {
                    "type": "string",
                    "description": (
                        "Time window for recent interactions: ISO 8601 or "
                        "relative ('30d', '7d', '24h'). Default: '30d'"
                    ),
                    "default": "30d",
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        "Maximum recent interactions to return (default: 10)"
                    ),
                    "default": 10,
                },
                "summary": {
                    "type": "boolean",
                    "description": (
                        "Include an LLM-generated 'next_brief' field that "
                        "answers 'what do I need to discuss with this "
                        "person next?'. Uses the configured 'completion' "
                        "role; default off so a plain person() call makes "
                        "no LLM request."
                    ),
                    "default": False,
                },
                "meeting_id": {
                    "type": "string",
                    "description": (
                        "Optional source_id of a CalendarEvent whose "
                        "summary, description, location, attendees, and "
                        "linked attachments are added to the brief "
                        "context (only meaningful with summary=true)."
                    ),
                },
                "horizon": {
                    "type": "string",
                    "description": (
                        "Lookback window for brief inputs. ISO 8601 or "
                        "relative ('30d', '7d'). Default: '30d'"
                    ),
                    "default": "30d",
                },
            },
            "required": ["identifier"],
        },
    ),
    Tool(
        name="itinerary",
        description=(
            "Aggregated daily agenda for a single day: calendar events with "
            "their linked open OmniFocus tasks, vector-similar notes "
            "(File / Obsidian / Slack), and the most recent email or Slack "
            "thread covering all attendees. Resolution order for 'day': "
            "'today' or 'tomorrow' (relative to the user's local timezone) "
            "or an explicit ISO date 'YYYY-MM-DD'. When 'brief' is true "
            "the per-event LLM summary is skipped and 'next_brief' is "
            "returned as null on every event."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "day": {
                    "type": "string",
                    "description": (
                        "Day to render: 'today' (default), 'tomorrow', or "
                        "an ISO date 'YYYY-MM-DD'."
                    ),
                    "default": "today",
                },
                "account": {
                    "type": "string",
                    "description": (
                        "Optional google_calendar.<account> to filter to a "
                        "single calendar account. Unknown account names "
                        "return a tool error listing the configured ones."
                    ),
                },
                "brief": {
                    "type": "boolean",
                    "description": (
                        "If true, skip the LLM-generated per-event summary "
                        "(next_brief is always null). Default false."
                    ),
                    "default": False,
                },
                "horizon": {
                    "type": "string",
                    "description": (
                        "Lookback window for linked tasks/notes/threads. "
                        "Relative ('30d', '7d') or ISO 8601. Default '30d'."
                    ),
                    "default": "30d",
                },
            },
        },
    ),
    Tool(
        name="digest",
        description=(
            "Summarize recent activity across all indexed sources in a time window. "
            "Returns aggregate counts (new, modified) per source type with top "
            "highlights, plus cross-source connection and topic discovery counts. "
            "Use this for a daily or weekly overview: 'what changed today?', "
            "'show me a digest of the last 7 days'."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "since": {
                    "type": "string",
                    "description": (
                        "Start of the time window. ISO 8601 or relative: "
                        "'24h', '7d', '2w', '3m', 'yesterday', 'last week'. "
                        "Default: '24h'"
                    ),
                    "default": "24h",
                },
                "until": {
                    "type": "string",
                    "description": "End of the time window. Default: 'now'",
                    "default": "now",
                },
                "summarize": {
                    "type": "boolean",
                    "description": "Include an LLM-generated summary paragraph. Default: false",
                    "default": False,
                },
            },
        },
    ),
]


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------


def _extract_auth_token(message: SessionMessage) -> str | None:
    """Extract ``auth_token`` from the ``_meta`` of an ``initialize`` request.

    Returns *None* when the message is not an initialize request or when the
    client did not supply a token.
    """
    try:
        root = message.message.root
        if root.method != "initialize":  # type: ignore[union-attr]
            return None
        meta = root.params.get("_meta") or {}  # type: ignore[union-attr]
        token = meta.get("auth_token")
        return str(token) if token is not None else None
    except Exception:
        return None


def _make_error_response(request_id: int | str | None) -> SessionMessage:
    """Build a JSON-RPC error response for an auth failure."""
    payload = {
        "jsonrpc": "2.0",
        "id": request_id if request_id is not None else 0,
        "error": {"code": -32001, "message": "Unauthorized: invalid auth token"},
    }
    return SessionMessage(JSONRPCMessage.model_validate(payload))


class _PrefixedReceiveStream:
    """A thin wrapper that yields *prefix* once, then delegates to *inner*.

    Implements the subset of the anyio ``ObjectReceiveStream`` protocol
    that the MCP ``Server.run`` loop uses (``receive``, ``aclose``, and
    the async-iterator protocol).
    """

    def __init__(
        self,
        prefix: SessionMessage | Exception,
        inner: anyio.abc.ObjectReceiveStream[SessionMessage | Exception],
    ) -> None:
        self._prefix: SessionMessage | Exception | None = prefix
        self._inner = inner

    async def receive(self) -> SessionMessage | Exception:
        if self._prefix is not None:
            msg = self._prefix
            self._prefix = None
            return msg
        return await self._inner.receive()

    async def aclose(self) -> None:
        await self._inner.aclose()

    def __aiter__(self):  # type: ignore[no-untyped-def]
        return self

    async def __anext__(self) -> SessionMessage | Exception:
        try:
            return await self.receive()
        except anyio.EndOfStream:
            raise StopAsyncIteration


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
        self._reranker = None
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None

        # Register handlers.
        self._app.list_tools()(self._list_tools)
        self._app.call_tool()(self._call_tool)

    # -- connection management ----------------------------------------------

    def _connect(self) -> None:
        """Initialise ModelRegistry, GraphQuerier, VectorQuerier, and thread pool.

        Uses try/except so that a failure in a later connection (e.g. Qdrant)
        cleans up already-opened connections (e.g. Neo4j) instead of leaking
        them.
        """
        from worker.query.reranker import build_reranker

        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
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
        self._reranker = build_reranker(self._cfg.reranker, self._registry)
        logger.info("Fieldnotes MCP server: connections initialised")

    def _disconnect(self) -> None:
        """Close all connections and thread pool gracefully."""
        if self._graph_querier is not None:
            self._graph_querier.close()
            self._graph_querier = None
        if self._vector_querier is not None:
            self._vector_querier.close()
            self._vector_querier = None
        self._registry = None
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
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
            if name == "timeline":
                return await self._handle_timeline(arguments)
            if name == "suggest_connections":
                return self._handle_suggest_connections(arguments)
            if name == "digest":
                return await self._handle_digest(arguments)
            if name == "itinerary":
                return await self._handle_itinerary(arguments)
            if name == "persons_inspect":
                return self._handle_persons_inspect(arguments)
            if name == "persons_split":
                return self._handle_persons_split(arguments)
            if name == "persons_confirm":
                return self._handle_persons_confirm(arguments)
            if name == "persons_merge":
                return self._handle_persons_merge(arguments)
            if name == "person":
                return self._handle_person(arguments)
            raise ValueError(f"Unknown tool: {name}")
        except Exception:
            logger.exception("Tool %s failed", name)
            return [TextContent(type="text", text="error: internal server error")]

    # -- tool implementations -----------------------------------------------

    _VALID_SOURCE_TYPES = frozenset({"file", "email", "obsidian", "repositories"})

    async def _handle_search(self, arguments: dict) -> list[TextContent]:
        query = arguments.get("query", "")
        if not isinstance(query, str) or not query.strip():
            return [
                TextContent(
                    type="text", text="error: 'query' must be a non-empty string"
                )
            ]
        top_k = max(1, min(int(arguments.get("top_k", 10)), 100))
        source_type: str | None = arguments.get("source_type")
        if source_type is not None and source_type not in self._VALID_SOURCE_TYPES:
            return [
                TextContent(
                    type="text",
                    text=f"error: 'source_type' must be one of {sorted(self._VALID_SOURCE_TYPES)}",
                )
            ]
        rerank_arg = arguments.get("rerank", True)
        use_rerank = (
            bool(rerank_arg)
            and self._cfg.reranker.enabled
            and self._reranker is not None
        )
        vector_top_k = self._cfg.reranker.top_k_pre if use_rerank else top_k

        if self._graph_querier is None or self._vector_querier is None:
            raise RuntimeError("Server not initialised — call _connect() first")

        # Check for empty corpus before running expensive queries.
        loop = asyncio.get_running_loop()
        empty = await loop.run_in_executor(
            self._executor,
            is_corpus_empty,
            self._graph_querier,
            self._vector_querier,
        )
        if empty:
            return [TextContent(type="text", text=EMPTY_CORPUS_MESSAGE)]

        # Run graph and vector queries concurrently in the thread pool.
        graph_future = loop.run_in_executor(
            self._executor, self._graph_querier.query, query
        )
        vector_future = loop.run_in_executor(
            self._executor,
            lambda: self._vector_querier.query(
                query, top_k=vector_top_k, source_type=source_type
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
            vector_result = VectorQueryResult(question=query, error=str(vector_result))

        hybrid = await loop.run_in_executor(
            self._executor,
            lambda: merge(
                query,
                graph_result,
                vector_result,
                reranker=self._reranker if use_rerank else None,
                top_k_post=top_k if use_rerank else None,
            ),
        )

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
        question = arguments.get("question", "")
        if not isinstance(question, str) or not question.strip():
            return [
                TextContent(
                    type="text", text="error: 'question' must be a non-empty string"
                )
            ]
        source_type: str | None = arguments.get("source_type")
        if source_type is not None and source_type not in self._VALID_SOURCE_TYPES:
            return [
                TextContent(
                    type="text",
                    text=f"error: 'source_type' must be one of {sorted(self._VALID_SOURCE_TYPES)}",
                )
            ]
        rerank_arg = arguments.get("rerank", True)
        use_rerank = (
            bool(rerank_arg)
            and self._cfg.reranker.enabled
            and self._reranker is not None
        )
        vector_top_k = self._cfg.reranker.top_k_pre if use_rerank else 20
        rerank_post = self._cfg.reranker.top_k_post

        if self._graph_querier is None or self._vector_querier is None:
            raise RuntimeError("Server not initialised — call _connect() first")
        if self._registry is None:
            raise RuntimeError("Registry not initialised")

        # Check for empty corpus before running expensive queries.
        loop = asyncio.get_running_loop()
        empty = await loop.run_in_executor(
            self._executor,
            is_corpus_empty,
            self._graph_querier,
            self._vector_querier,
        )
        if empty:
            return [
                TextContent(
                    type="text",
                    text=(
                        f"[Answer]\n{EMPTY_CORPUS_MESSAGE}\n\n"
                        "[Sources]\nNone\n\n"
                        "[Confidence]\ncorpus is empty"
                    ),
                )
            ]

        # --- 1. Retrieve context (same as _handle_search) ---

        graph_future = loop.run_in_executor(
            self._executor, self._graph_querier.query, question
        )
        vector_future = loop.run_in_executor(
            self._executor,
            lambda: self._vector_querier.query(
                question, top_k=vector_top_k, source_type=source_type
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

        hybrid = await loop.run_in_executor(
            self._executor,
            lambda: merge(
                question,
                graph_result,
                vector_result,
                reranker=self._reranker if use_rerank else None,
                top_k_post=rerank_post if use_rerank else None,
            ),
        )

        # --- 2. Build RAG prompt and call LLM ---
        context_text = hybrid.context
        if not context_text.strip():
            return [
                TextContent(
                    type="text",
                    text=(
                        "[Answer]\n"
                        "I don't have enough information in the knowledge graph "
                        "to answer this question.\n\n"
                        "[Sources]\nNone\n\n"
                        "[Confidence]\nno relevant context found"
                    ),
                )
            ]

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
            self._executor, lambda: model.complete(req, task="ask")
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
        topic_name = arguments.get("name", "")
        if not isinstance(topic_name, str) or not topic_name.strip():
            return [
                TextContent(
                    type="text", text="error: 'name' must be a non-empty string"
                )
            ]
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
                    "File",
                    "Email",
                    "Commit",
                    "Entity",
                    "Topic",
                    "Chunk",
                    "Image",
                    "Repository",
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
                        "last_sync": str(row["last_sync"])
                        if row["last_sync"]
                        else None,
                    }

                # Entity counts by type
                entity_rows = session.run(
                    "MATCH (e:Entity) "
                    "RETURN e.type AS type, count(e) AS cnt "
                    "ORDER BY cnt DESC"
                ).data()
                entities_by_type = {r["type"]: r["cnt"] for r in entity_rows}
                entity_total = sum(entities_by_type.values())

                # Topic counts by source (cluster vs user)
                topic_rows = session.run(
                    "MATCH (t:Topic) RETURN t.source AS source, count(t) AS cnt"
                ).data()
                topics_by_source = {r["source"]: r["cnt"] for r in topic_rows}
        except Exception as exc:
            logger.exception("ingest_status: Neo4j query failed")
            neo4j_health = f"error: {type(exc).__name__}"
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
            qdrant_health = f"error: {type(exc).__name__}"
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

        return [
            TextContent(type="text", text=json.dumps(result, indent=2, default=str))
        ]

    async def _handle_timeline(self, arguments: dict) -> list[TextContent]:
        """Execute a timeline query and return results as JSON."""
        since = arguments.get("since", "24h")
        until = arguments.get("until", "now")
        source_type: str | None = arguments.get("source_type")
        limit = max(1, min(int(arguments.get("limit", 50)), 500))

        if not isinstance(since, str) or not since:
            since = "24h"
        if not isinstance(until, str) or not until:
            until = "now"
        if source_type is not None and source_type not in _TIMELINE_SOURCE_TYPES:
            return [
                TextContent(
                    type="text",
                    text=f"error: 'source_type' must be one of {sorted(_TIMELINE_SOURCE_TYPES)}",
                )
            ]

        loop = asyncio.get_running_loop()

        def _run() -> str:
            with TimelineQuerier(self._cfg.neo4j, self._cfg.qdrant) as querier:
                result = querier.query(
                    since=since,
                    until=until,
                    source_type=source_type,
                    limit=limit,
                )
            if result.error:
                return json.dumps({"error": result.error}, indent=2)
            return json.dumps(
                {
                    "since": result.since,
                    "until": result.until,
                    "entries": [
                        {
                            "source_type": e.source_type,
                            "source_id": e.source_id,
                            "label": e.label,
                            "title": e.title,
                            "timestamp": e.timestamp,
                            "event_type": e.event_type,
                            "snippet": e.snippet,
                        }
                        for e in result.entries
                    ],
                    "count": len(result.entries),
                },
                indent=2,
                default=str,
            )

        text = await loop.run_in_executor(self._executor, _run)
        return [TextContent(type="text", text=text)]

    async def _handle_digest(self, arguments: dict) -> list[TextContent]:
        """Execute a digest query and return results as JSON."""
        since = arguments.get("since", "24h")
        until = arguments.get("until", "now")

        if not isinstance(since, str) or not since:
            since = "24h"
        if not isinstance(until, str) or not until:
            until = "now"

        loop = asyncio.get_running_loop()

        def _run() -> str:
            with DigestQuerier(self._cfg.neo4j) as querier:
                result = querier.query(since=since, until=until)
            if result.error:
                return json.dumps({"error": result.error}, indent=2)
            sources = []
            for a in result.sources:
                completed = getattr(a, "_completed", 0)
                entry: dict = {
                    "source_type": a.source_type,
                    "created": a.created,
                    "modified": a.modified,
                    "highlights": a.highlights,
                }
                if completed:
                    entry["completed"] = completed
                sources.append(entry)
            return json.dumps(
                {
                    "since": result.since,
                    "until": result.until,
                    "sources": sources,
                    "new_connections": result.new_connections,
                    "new_topics": result.new_topics,
                    "summary": result.summary,
                },
                indent=2,
                default=str,
            )

        text = await loop.run_in_executor(self._executor, _run)
        return [TextContent(type="text", text=text)]

    async def _handle_itinerary(self, arguments: dict) -> list[TextContent]:
        """Aggregate daily agenda and return as JSON.

        Returns a documented dict shape mirroring the ``fieldnotes
        itinerary --json`` payload.  When ``brief`` is false (default),
        a per-event LLM call populates ``next_brief``; when true, no LLM
        is invoked and ``next_brief`` is null on every event.
        """
        day = arguments.get("day", "today")
        if not isinstance(day, str) or not day.strip():
            day = "today"

        account = arguments.get("account") or None
        if account is not None and not isinstance(account, str):
            return [_itinerary_error("'account' must be a string")]

        if account is not None:
            configured = sorted(self._cfg.google_calendar.keys())
            if account not in configured:
                return [
                    _itinerary_error(
                        f"Unknown account {account!r}",
                        configured_accounts=configured,
                    )
                ]

        brief = bool(arguments.get("brief", False))
        horizon_str = arguments.get("horizon", "30d")
        if not isinstance(horizon_str, str) or not horizon_str.strip():
            horizon_str = "30d"

        try:
            horizon_td = _parse_horizon(horizon_str)
        except ValueError as exc:
            return [_itinerary_error(f"Invalid 'horizon': {exc}")]

        # Resolve completion role early (fail-fast) when LLM briefs enabled.
        completion_model: Any = None
        if not brief:
            try:
                completion_model = _resolve_itinerary_completion(
                    self._cfg, self._registry
                )
            except ItineraryBriefError as exc:
                return [_itinerary_error(str(exc))]

        loop = asyncio.get_running_loop()

        def _run() -> dict | str:
            from worker.query.itinerary import get_itinerary

            try:
                itin = get_itinerary(
                    day=day,
                    account=account,
                    horizon=horizon_td,
                    registry=self._registry,
                    qdrant_cfg=self._cfg.qdrant,
                    neo4j_cfg=self._cfg.neo4j,
                )
            except ValueError as exc:
                return f"error:{exc}"

            briefs: dict[int, str] = {}
            thread_details: dict[int, _ItineraryThreadDetail] = {}
            needs_driver = any(ewl.thread is not None for ewl in itin.events) or (
                completion_model is not None and itin.events
            )
            if needs_driver:
                drv = GraphDatabase.driver(
                    self._cfg.neo4j.uri,
                    auth=(self._cfg.neo4j.user, self._cfg.neo4j.password),
                )
                try:
                    for ewl in itin.events:
                        if ewl.thread is not None:
                            thread_details[ewl.event.id] = (
                                _fetch_itinerary_thread_detail(drv, ewl.thread)
                            )
                    if completion_model is not None:
                        briefs = _generate_itinerary_briefs(
                            itin, driver=drv, completion_model=completion_model
                        )
                finally:
                    drv.close()
            return _itinerary_to_dict(
                itin, briefs=briefs, thread_details=thread_details
            )

        result = await loop.run_in_executor(self._executor, _run)
        if isinstance(result, str) and result.startswith("error:"):
            return [_itinerary_error(result[len("error:") :])]
        assert isinstance(result, dict)
        return [TextContent(type="text", text=json.dumps(result, default=str))]

    def _handle_person(self, arguments: dict) -> list[TextContent]:
        identifier = arguments.get("identifier", "")
        if not isinstance(identifier, str) or not identifier.strip():
            return [_person_error("'identifier' must be a non-empty string")]

        since_str = arguments.get("since", "30d")
        if not isinstance(since_str, str) or not since_str.strip():
            since_str = "30d"

        try:
            limit = max(1, min(int(arguments.get("limit", 10)), 1000))
        except (TypeError, ValueError):
            limit = 10

        summary = bool(arguments.get("summary", False))
        meeting_id = arguments.get("meeting_id") or None
        horizon_str = arguments.get("horizon", "30d")
        if not isinstance(horizon_str, str) or not horizon_str.strip():
            horizon_str = "30d"

        try:
            since = parse_relative_time(since_str)
        except ValueError as exc:
            return [_person_error(f"Invalid 'since': {exc}")]

        result = get_profile(identifier, since, limit=limit, neo4j_cfg=self._cfg.neo4j)

        if result is None:
            return [_person_error(f"No Person found for {identifier!r}")]
        if isinstance(result, list):
            candidates = [{"name": p.name, "email": p.email} for p in result]
            return [
                _person_error(
                    f"Ambiguous identifier {identifier!r}: "
                    f"{len(result)} candidates — refine using an email "
                    "or slack-user prefix",
                    candidates=candidates,
                )
            ]

        brief: str | None = None
        if summary:
            try:
                horizon_dt = parse_relative_time(horizon_str)
            except ValueError as exc:
                return [_person_error(f"Invalid 'horizon': {exc}")]

            drv = GraphDatabase.driver(
                self._cfg.neo4j.uri,
                auth=(self._cfg.neo4j.user, self._cfg.neo4j.password),
            )
            try:
                brief, _ = generate_brief(
                    result.person,
                    cfg=self._cfg,
                    driver=drv,
                    since=horizon_dt,
                    since_label=horizon_str,
                    meeting_id=meeting_id,
                )
            except BriefError as exc:
                return [_person_error(str(exc))]
            except ValueError as exc:
                return [_person_error(str(exc))]
            finally:
                drv.close()

        payload = _person_profile_to_dict(identifier.strip(), result, brief=brief)
        return [TextContent(type="text", text=json.dumps(payload, default=str))]

    def _handle_suggest_connections(self, arguments: dict) -> list[TextContent]:
        source_id: str | None = arguments.get("source_id") or None
        source_type: str | None = arguments.get("source_type") or None
        threshold = float(arguments.get("threshold", 0.82))
        limit = int(arguments.get("limit", 20))
        cross_source = bool(arguments.get("cross_source", False))

        threshold = max(0.0, min(1.0, threshold))
        limit = max(1, min(limit, 200))

        with ConnectionQuerier(self._cfg.neo4j, self._cfg.qdrant) as querier:
            result = querier.suggest(
                source_id=source_id,
                source_type=source_type,
                threshold=threshold,
                limit=limit,
                cross_source=cross_source,
            )

        if result.error:
            return [TextContent(type="text", text=f"error: {result.error}")]

        import json as _json

        data = {
            "checked": result.checked,
            "suggestions": [
                {
                    "similarity": s.similarity,
                    "source_a": s.source_a,
                    "source_b": s.source_b,
                    "label_a": s.label_a,
                    "label_b": s.label_b,
                    "title_a": s.title_a,
                    "title_b": s.title_b,
                    "source_type_a": s.source_type_a,
                    "source_type_b": s.source_type_b,
                    "reason": s.reason,
                }
                for s in result.suggestions
            ],
        }
        return [TextContent(type="text", text=_json.dumps(data, indent=2))]

    # -- persons curation ---------------------------------------------------

    def _handle_persons_inspect(self, arguments: dict) -> list[TextContent]:
        identifier = arguments.get("identifier", "")
        if not isinstance(identifier, str) or not identifier.strip():
            return [
                TextContent(
                    type="text",
                    text="error: 'identifier' must be a non-empty string",
                )
            ]
        from dataclasses import asdict
        import json as _json

        from worker.curation import CurationError, PersonCurator

        if self._graph_querier is None:
            raise RuntimeError("Server not initialised — call _connect() first")

        try:
            curator = PersonCurator(self._graph_querier._driver)
            result = curator.inspect(identifier)
        except CurationError as exc:
            return [TextContent(type="text", text=f"error: {exc}")]

        payload = {
            "focal": asdict(result.focal) if result.focal else None,
            "same_as": [asdict(e) for e in result.same_as],
            "never_same_as": [asdict(e) for e in result.never_same_as],
        }
        return [TextContent(type="text", text=_json.dumps(payload, indent=2))]

    def _handle_persons_split(self, arguments: dict) -> list[TextContent]:
        return self._run_persons_mutation(
            arguments,
            required=("identifier", "member"),
            invoke=lambda c, args: c.split(args["identifier"], args["member"]),
        )

    def _handle_persons_confirm(self, arguments: dict) -> list[TextContent]:
        return self._run_persons_mutation(
            arguments,
            required=("a", "b"),
            invoke=lambda c, args: c.confirm(args["a"], args["b"]),
        )

    def _handle_persons_merge(self, arguments: dict) -> list[TextContent]:
        return self._run_persons_mutation(
            arguments,
            required=("a", "b"),
            invoke=lambda c, args: c.merge(args["a"], args["b"]),
        )

    def _run_persons_mutation(
        self,
        arguments: dict,
        *,
        required: tuple[str, ...],
        invoke,
    ) -> list[TextContent]:
        import json as _json

        from worker.curation import AuditLog, CurationError, PersonCurator

        for key in required:
            value = arguments.get(key, "")
            if not isinstance(value, str) or not value.strip():
                return [
                    TextContent(
                        type="text",
                        text=f"error: {key!r} must be a non-empty string",
                    )
                ]

        if self._graph_querier is None:
            raise RuntimeError("Server not initialised — call _connect() first")

        audit = AuditLog.from_data_dir(self._cfg.core.data_dir)
        curator = PersonCurator(
            self._graph_querier._driver,
            audit=audit,
            actor="mcp",
        )
        try:
            result = invoke(curator, arguments)
        except CurationError as exc:
            return [TextContent(type="text", text=f"error: {exc}")]
        return [
            TextContent(
                type="text",
                text=_json.dumps(
                    {"action": result.action, "detail": result.detail},
                    indent=2,
                ),
            )
        ]

    # -- run ----------------------------------------------------------------

    async def run(self) -> None:
        """Start the MCP server over stdio transport."""
        self._connect()

        loop = asyncio.get_running_loop()
        current_task = asyncio.current_task()

        def _shutdown_handler() -> None:
            logger.info("Received shutdown signal")
            if current_task is not None:
                current_task.cancel()

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, _shutdown_handler)

        auth_token = self._cfg.mcp.auth_token

        try:
            async with stdio_server() as (read_stream, write_stream):
                if auth_token:
                    read_stream = await self._auth_gate(
                        read_stream,
                        write_stream,
                        auth_token,
                    )
                await self._app.run(
                    read_stream,
                    write_stream,
                    self._app.create_initialization_options(),
                )
        except asyncio.CancelledError:
            logger.info("MCP server shutting down")
        finally:
            self._disconnect()

    @staticmethod
    async def _auth_gate(
        read_stream: anyio.abc.ObjectReceiveStream[SessionMessage | Exception],
        write_stream: anyio.abc.ObjectSendStream[SessionMessage],
        expected_token: str,
    ) -> anyio.abc.ObjectReceiveStream[SessionMessage | Exception]:
        """Validate the auth token on the first ``initialize`` message.

        Consumes the first message from *read_stream*.  If its
        ``_meta.auth_token`` matches *expected_token*, returns a new
        receive-stream that replays the validated message first and then
        relays all subsequent messages.

        On failure an error response is sent and :class:`PermissionError`
        is raised.
        """
        first = await read_stream.receive()

        if isinstance(first, Exception):
            raise first

        client_token = _extract_auth_token(first)

        if client_token is None or not hmac.compare_digest(
            client_token.encode(), expected_token.encode()
        ):
            try:
                req_id = first.message.root.id  # type: ignore[union-attr]
            except Exception:
                req_id = None
            await write_stream.send(_make_error_response(req_id))
            raise PermissionError("MCP auth: invalid or missing token")

        logger.info("MCP auth: client authenticated successfully")

        # Wrap original stream to replay *first*, then forward everything.
        return _PrefixedReceiveStream(first, read_stream)


def _itinerary_error(message: str, **extra: Any) -> TextContent:
    payload: dict[str, Any] = {"error": True, "message": message}
    payload.update(extra)
    return TextContent(type="text", text=json.dumps(payload, default=str))


def _parse_horizon(s: str):
    """Parse a horizon string like ``'30d'`` into a positive ``timedelta``.

    Reuses :func:`parse_relative_time` (which returns ``now - horizon``)
    and inverts back to a duration.
    """
    from datetime import datetime, timezone

    parsed = parse_relative_time(s)
    delta = datetime.now(timezone.utc) - parsed
    if delta.total_seconds() < 0:
        delta = -delta
    return delta


def _itinerary_to_dict(
    itin: Any,
    *,
    briefs: dict[int, str] | None = None,
    thread_details: dict[int, _ItineraryThreadDetail] | None = None,
) -> dict:
    """Serialize an Itinerary to the documented JSON schema.

    ``next_brief`` is populated from *briefs* (event_id → text); events
    without an entry get ``None``.  ``thread_details`` carries the
    enriched ``last_from`` per event (CLI/MCP parity); events without
    detail fall back to ``None``.
    """
    brief_map = briefs or {}
    detail_map = thread_details or {}

    def _person(p: Any) -> dict | None:
        if p is None:
            return None
        return {"name": p.name, "email": p.email}

    events: list[dict] = []
    for ewl in itin.events:
        ev = ewl.event
        events.append(
            {
                "event_id": str(ev.id),
                "source_id": ev.source_id,
                "title": ev.title,
                "start": ev.start_ts,
                "end": ev.end_ts,
                "account": ev.account,
                "calendar_id": ev.calendar_id,
                "organizer": _person(ev.organizer),
                "attendees": [_person(a) for a in ev.attendees if a is not None],
                "location": ev.location,
                "html_link": ev.html_link,
                "linked": {
                    "tasks": [
                        {
                            "title": t.title,
                            "project": t.project,
                            "tags": list(t.tags),
                            "due": t.due,
                            "defer": t.defer,
                            "flagged": bool(t.flagged),
                            "source_id": t.source_id,
                        }
                        for t in ewl.tasks
                    ],
                    "notes": [
                        {
                            "source_id": n.source_id,
                            "title": n.title,
                            "snippet": n.snippet,
                            "mtime": n.mtime,
                            "attendee_overlap": bool(n.attendee_overlap),
                            "score": float(n.score),
                        }
                        for n in ewl.notes
                    ],
                    "thread": (
                        None
                        if ewl.thread is None
                        else {
                            "kind": ewl.thread.kind,
                            "source_id": ewl.thread.source_id,
                            "title": ewl.thread.title,
                            "last_ts": ewl.thread.last_ts,
                            "last_from": detail_map.get(
                                ev.id, _ItineraryThreadDetail()
                            ).last_from,
                        }
                    ),
                },
                "next_brief": brief_map.get(ev.id),
            }
        )

    return {
        "day": itin.day.isoformat()
        if hasattr(itin.day, "isoformat")
        else str(itin.day),
        "timezone": itin.timezone,
        "events": events,
    }


def _person_error(message: str, **extra: Any) -> TextContent:
    payload: dict[str, Any] = {"error": True, "message": message}
    payload.update(extra)
    return TextContent(type="text", text=json.dumps(payload, default=str))


def _person_profile_to_dict(
    identifier: str,
    profile: PersonProfile,
    *,
    brief: str | None = None,
) -> dict:
    p = profile.person
    last_seen = (
        profile.recent_interactions[0].timestamp
        if profile.recent_interactions
        else None
    )
    sources_present: set[str] = {
        i.source_type for i in profile.recent_interactions if i.source_type
    }
    if profile.open_tasks:
        sources_present.add("omnifocus")
    if profile.files:
        sources_present.add("file")

    payload: dict[str, Any] = {
        "identifier": identifier,
        "resolved": {
            "name": p.name,
            "email": p.email,
            "is_self": bool(p.is_self),
        },
        "sources_present": sorted(sources_present),
        "last_seen": last_seen,
        "total_interactions": len(profile.recent_interactions),
        "recent_interactions": [
            {
                "timestamp": i.timestamp,
                "source_type": i.source_type,
                "title": i.title,
                "snippet": i.snippet,
                "edge_kind": i.edge_kind,
            }
            for i in profile.recent_interactions
        ],
        "top_topics": [
            {"topic_name": t.topic_name, "doc_count": t.doc_count}
            for t in profile.top_topics
        ],
        "related_people": [
            {
                "name": r.name,
                "email": r.email,
                "shared_count": r.shared_count,
            }
            for r in profile.related_people
        ],
        "open_tasks": [
            {
                "title": t.title,
                "project": t.project,
                "tags": list(t.tags),
                "due": t.due,
                "defer": t.defer,
                "flagged": t.flagged,
            }
            for t in profile.open_tasks
        ],
        "files_mentioning": [
            {"path": f.path, "mtime": f.mtime, "source": f.source}
            for f in profile.files
        ],
        "identity_cluster": [
            {
                "member": m.member,
                "match_type": m.match_type,
                "confidence": m.confidence,
                "cross_source": m.cross_source,
            }
            for m in profile.identity_cluster
        ],
    }
    if brief is not None:
        payload["next_brief"] = brief
    return payload


def run_server(config_path: Path | None = None) -> None:
    """Entry point: load config and run the MCP server."""
    cfg = load_config(config_path)
    server = FieldnotesServer(cfg)
    asyncio.run(server.run())
