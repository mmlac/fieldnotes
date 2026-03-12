"""Graph query: NL→Cypher translation via LangChain + Neo4j.

Translates natural-language questions into Cypher queries using
LangChain's GraphCypherQAChain. The chain receives the full Neo4j
schema as context so it can generate structurally valid Cypher.

Uses the 'query' role model from ModelRegistry. Returns structured
results with node properties. Safety: read-only queries only — no
MERGE/CREATE/DELETE/SET/REMOVE/DROP in generated Cypher.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from langchain_community.chains.graph_qa.cypher import (
    GraphCypherQAChain,
    extract_cypher,
)
from langchain_community.graphs import Neo4jGraph
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from neo4j import GraphDatabase

from worker.config import Neo4jConfig
from worker.models.resolver import ModelRegistry, ResolvedModel

logger = logging.getLogger(__name__)

# Cypher keywords that indicate a write operation.
_WRITE_KEYWORDS = re.compile(
    r"\b(MERGE|CREATE|DELETE|DETACH|SET|REMOVE|DROP|FOREACH\s*\(|LOAD\s+CSV|SHOW)\b",
    re.IGNORECASE,
)

# Block CALL subqueries (CALL { ... }) and procedure invocations
# (CALL <name>(...)).  Read-only built-in procedures like
# db.index.fulltext.queryNodes are not expected in LLM-generated Cypher,
# so a blanket block is the safest default.
_CALL_PATTERN = re.compile(
    r"\bCALL\s*(\{|[A-Za-z])",
    re.IGNORECASE,
)

# APOC procedures known to perform writes (kept for specificity in logs).
_WRITE_APOC = re.compile(
    r"\b(apoc\.periodic\.commit|apoc\.periodic\.iterate|apoc\.cypher\.run|apoc\.cypher\.doIt)\b",
    re.IGNORECASE,
)


# Default LIMIT appended to LLM-generated queries that lack one.
_DEFAULT_QUERY_LIMIT = 1000

# Timeout (in seconds) for read-only query transactions (server-side advisory).
_QUERY_TIMEOUT_S = 30

# Client-side timeouts for the Neo4j driver (seconds).
_CONNECTION_TIMEOUT_S = 10
_MAX_TRANSACTION_RETRY_TIME_S = 30

_LIMIT_PATTERN = re.compile(r"\bLIMIT\s+\d+", re.IGNORECASE)


def _ensure_limit(cypher: str, limit: int = _DEFAULT_QUERY_LIMIT) -> str:
    """Append a LIMIT clause if the Cypher query doesn't already have one."""
    if _LIMIT_PATTERN.search(cypher):
        return cypher
    return f"{cypher.rstrip().rstrip(';')} LIMIT {limit}"


@dataclass
class GraphQueryResult:
    """Structured result from a graph query."""

    question: str
    cypher: str
    raw_results: list[dict[str, Any]] = field(default_factory=list)
    answer: str = ""
    error: str | None = None


class _RegistryLLM(BaseChatModel):
    """Adapts a ResolvedModel into a LangChain BaseChatModel.

    This thin wrapper lets us plug the ModelRegistry into LangChain's
    chain machinery without depending on any vendor-specific LangChain
    integration package.
    """

    resolved: Any = None  # ResolvedModel, typed as Any for pydantic compat

    @property
    def _llm_type(self) -> str:
        return f"fieldnotes-{self.resolved.alias}"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        from worker.models.base import CompletionRequest

        lc_messages = [
            {"role": _lc_role(m), "content": m.content} for m in messages
        ]
        # Split system messages from the rest.
        system_parts = [
            m["content"] for m in lc_messages if m["role"] == "system"
        ]
        non_system = [m for m in lc_messages if m["role"] != "system"]

        req = CompletionRequest(
            system="\n".join(system_parts) if system_parts else "",
            messages=non_system or [{"role": "user", "content": ""}],
            temperature=0.0,
        )
        resp = self.resolved.complete(req, task="graph_query")
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=resp.text))]
        )

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop=stop)


def _lc_role(msg: BaseMessage) -> str:
    """Map a LangChain message to a simple role string."""
    if msg.type == "system":
        return "system"
    if msg.type == "ai":
        return "assistant"
    return "user"


def _normalize_cypher_for_validation(cypher: str) -> str:
    """Normalize Cypher text to defeat bypass techniques.

    1. Replace all Unicode whitespace characters (NBSP, em-space, etc.)
       with plain ASCII spaces so regex word boundaries work correctly.
    2. Strip Cypher line comments (// ...) and block comments (/* ... */).
    """
    # Normalize Unicode whitespace to ASCII space
    normalized = "".join(
        " " if unicodedata.category(ch) in ("Zs", "Zl", "Zp") else ch
        for ch in cypher
    )
    # Strip block comments (/* ... */), including nested
    normalized = re.sub(r"/\*.*?\*/", " ", normalized, flags=re.DOTALL)
    # Strip line comments (// ...)
    normalized = re.sub(r"//[^\n]*", " ", normalized)
    return normalized


def _validate_cypher_readonly(cypher: str) -> None:
    """Raise if the Cypher contains write operations.

    Defense-in-depth: this regex blocklist catches common mutation patterns,
    but the real safety boundary is the read-only transaction in GraphQuerier.

    Normalizes Unicode whitespace and strips comments before checking to
    prevent bypass via NBSP word-boundary evasion or comment embedding.
    """
    normalized = _normalize_cypher_for_validation(cypher)
    if (
        _WRITE_KEYWORDS.search(normalized)
        or _CALL_PATTERN.search(normalized)
        or _WRITE_APOC.search(normalized)
    ):
        raise ReadOnlyCypherViolation(cypher)


class ReadOnlyCypherViolation(Exception):
    """Raised when generated Cypher contains write operations."""

    def __init__(self, cypher: str) -> None:
        super().__init__(
            f"Generated Cypher contains write operations and was blocked: {cypher}"
        )
        self.cypher = cypher


class GraphQuerier:
    """Translates natural-language questions to Cypher via LangChain.

    Usage::

        querier = GraphQuerier(registry, neo4j_cfg)
        result = querier.query("Which people are mentioned in my March notes?")
        print(result.answer)
        querier.close()
    """

    def __init__(
        self,
        registry: ModelRegistry,
        neo4j_cfg: Neo4jConfig | None = None,
    ) -> None:
        neo4j_cfg = neo4j_cfg or Neo4jConfig()
        self._resolved = registry.for_role("query")

        self._graph = Neo4jGraph(
            url=neo4j_cfg.uri,
            username=neo4j_cfg.user,
            password=neo4j_cfg.password,
        )
        try:
            self._graph.refresh_schema()

            # Maintain our own driver instance instead of reaching into
            # LangChain internals (_graph._driver) which are not part of the
            # public API and may break across versions.
            self._driver = GraphDatabase.driver(
                neo4j_cfg.uri,
                auth=(neo4j_cfg.user, neo4j_cfg.password),
                connection_timeout=_CONNECTION_TIMEOUT_S,
                max_transaction_retry_time=_MAX_TRANSACTION_RETRY_TIME_S,
            )
        except Exception:
            # Close the already-opened Neo4jGraph to avoid leaking its
            # internal driver connection pool.
            try:
                self._graph._driver.close()
            except Exception:
                pass
            raise

        self._database = getattr(self._graph, "_database", None) or "neo4j"

        self._llm = _RegistryLLM(resolved=self._resolved)
        self._chain = GraphCypherQAChain.from_llm(
            llm=self._llm,
            graph=self._graph,
            verbose=False,
            return_intermediate_steps=True,
            allow_dangerous_requests=True,  # required by LangChain; we validate before execution
        )

    def query(self, question: str) -> GraphQueryResult:
        """Translate *question* to Cypher, execute, and return structured results.

        Safety: generates Cypher first, validates it is read-only, and only
        then executes against Neo4j — so write operations never reach the DB.
        """
        try:
            # Step 1: Generate Cypher (without executing against Neo4j).
            schema = self._graph.get_structured_schema
            raw_cypher = self._chain.cypher_generation_chain.run(
                {"question": question, "schema": schema}
            )
            cypher = extract_cypher(raw_cypher)

            # Step 2: Safety gate — reject write operations BEFORE execution.
            if cypher:
                _validate_cypher_readonly(cypher)

            # Step 3: Execute the validated Cypher in a READ-ONLY transaction.
            # Neo4j enforces this at the protocol level — even if the regex
            # blocklist above is bypassed, the server will reject writes.
            raw_results: list[dict[str, Any]] = []
            if cypher:
                raw_results = self._execute_readonly(cypher)[: self._chain.top_k]

            # Step 4: Generate human-readable answer via the QA chain.
            qa_result = self._chain.qa_chain.invoke(
                {"question": question, "context": raw_results}
            )
            # qa_chain.invoke returns a dict with 'text' key (LLMChain)
            # or may return a string depending on the chain type.
            if isinstance(qa_result, dict):
                answer = qa_result.get("text", str(qa_result))
            else:
                answer = str(qa_result)

            return GraphQueryResult(
                question=question,
                cypher=cypher,
                raw_results=raw_results,
                answer=answer,
            )
        except ReadOnlyCypherViolation:
            raise
        except Exception as exc:
            logger.exception("Graph query failed for: %s", question)
            return GraphQueryResult(
                question=question,
                cypher="",
                error=str(exc),
            )

    def _execute_readonly(self, cypher: str) -> list[dict[str, Any]]:
        """Execute *cypher* inside a Neo4j read-only transaction.

        Enforces a LIMIT clause (appended if missing) and a per-query
        timeout so that runaway queries cannot exhaust server resources.
        Uses session.execute_read() so that Neo4j rejects any write
        operations at the protocol level, regardless of the query content.

        Two timeout layers:
        - ``tx.run(timeout=...)`` — server-side advisory timeout (Neo4j may
          ignore this under load).
        - ``session(fetch_size=...)`` + driver-level
          ``max_transaction_retry_time`` — caps client-side wait so a hung
          server cannot block the worker forever.
        """
        cypher = _ensure_limit(cypher)

        def _work(tx: Any) -> list[dict[str, Any]]:
            result = tx.run(cypher, timeout=_QUERY_TIMEOUT_S)
            return [record.data() for record in result]

        with self._driver.session(
            database=self._database,
            fetch_size=1000,
        ) as session:
            return session.execute_read(_work)

    def refresh_schema(self) -> None:
        """Refresh the Neo4j schema cache (call after schema changes)."""
        self._graph.refresh_schema()

    def __enter__(self) -> GraphQuerier:
        return self

    def __exit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Release the Neo4j connection and close the driver pool."""
        if self._driver is not None:
            try:
                self._driver.close()
            except Exception:
                logger.debug("Error closing Neo4j driver", exc_info=True)
            self._driver = None  # type: ignore[assignment]
            self._graph = None  # type: ignore[assignment]
