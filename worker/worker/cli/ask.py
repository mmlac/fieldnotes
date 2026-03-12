"""Interactive Q&A against the knowledge graph.

Provides both one-shot (``fieldnotes ask "question"``) and interactive REPL
(``fieldnotes ask``) modes, reusing the same RAG+LLM pipeline as the MCP
``ask`` tool.
"""

from __future__ import annotations

import logging
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from worker.config import Config, load_config
from worker.models.base import CompletionRequest
from worker.models.resolver import ModelRegistry
from worker.query.graph import GraphQuerier, GraphQueryResult
from worker.query.hybrid import merge
from worker.query.vector import VectorQuerier, VectorQueryResult

# Ensure provider registration side-effects run.
import worker.models.providers.ollama  # noqa: F401

logger = logging.getLogger("worker.cli.ask")

# REPL slash-commands
_COMMANDS = {"/history", "/clear", "/verbose", "/quit"}

_MAX_CONTEXT_CHARS = 60_000


@dataclass
class _Turn:
    question: str
    context: str
    answer: str


@dataclass
class _Session:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    history: list[_Turn] = field(default_factory=list)
    verbose: bool = False


def _synthesize(
    question: str,
    *,
    registry: ModelRegistry,
    graph_querier: GraphQuerier,
    vector_querier: VectorQuerier,
    session: _Session | None = None,
) -> str:
    """Run hybrid retrieval + LLM synthesis. Returns formatted answer text."""
    # --- 1. Retrieve context ---
    graph_result: GraphQueryResult
    try:
        graph_result = graph_querier.query(question)
    except Exception as exc:
        logger.debug("Graph query failed: %s", exc)
        graph_result = GraphQueryResult(question=question, cypher="", error=str(exc))

    vector_result: VectorQueryResult
    try:
        vector_result = vector_querier.query(question, top_k=20)
    except Exception as exc:
        logger.debug("Vector query failed: %s", exc)
        vector_result = VectorQueryResult(question=question, error=str(exc))

    hybrid = merge(question, graph_result, vector_result)

    # --- 2. Build RAG prompt ---
    context_text = hybrid.context
    if not context_text.strip():
        return (
            "I don't have enough information in the knowledge graph "
            "to answer this question."
        )

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

    # Include conversation history for follow-up resolution.
    history_block = ""
    if session and session.history:
        turns = []
        for turn in session.history[-5:]:  # last 5 turns for context window
            turns.append(f"Q: {turn.question}\nA: {turn.answer}")
        history_block = (
            "Previous conversation:\n"
            + "\n---\n".join(turns)
            + "\n\n"
        )

    user_prompt = (
        f"{history_block}"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        "Provide a clear, concise answer based on the context above. "
        "Reference source_ids when citing specific facts."
    )

    # Truncate if too large.
    if len(user_prompt) > _MAX_CONTEXT_CHARS:
        truncated_context = context_text[: _MAX_CONTEXT_CHARS - 500]
        user_prompt = (
            f"{history_block}"
            f"Context:\n{truncated_context}\n[... truncated ...]\n\n"
            f"Question: {question}\n\n"
            "Provide a clear, concise answer based on the context above. "
            "Reference source_ids when citing specific facts."
        )

    try:
        model = registry.for_role("query")
    except KeyError:
        model = registry.for_role("extraction")

    req = CompletionRequest(
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=0.2,
        timeout=120.0,
    )
    resp = model.complete(req, task="ask")

    # --- 3. Format response ---
    parts: list[str] = []

    if hybrid.errors:
        parts.append("Warnings: " + "; ".join(hybrid.errors))

    parts.append(resp.text)

    if unique_ids and (session is None or session.verbose):
        parts.append("[Sources]\n" + "\n".join(unique_ids))

    has_context = bool(hybrid.graph_results or hybrid.vector_results)
    sparse = (len(hybrid.graph_results) + len(hybrid.vector_results)) < 3
    if not has_context:
        parts.append("[Confidence] no relevant context found")
    elif sparse:
        parts.append("[Confidence] low — limited context available")

    answer = "\n\n".join(parts)

    # Record turn in session history.
    if session is not None:
        session.history.append(_Turn(
            question=question,
            context=context_text,
            answer=resp.text,
        ))

    return answer


def _run_repl(
    *,
    config_path: Path | None,
) -> int:
    """Interactive REPL loop using prompt_toolkit."""
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
        from prompt_toolkit.completion import WordCompleter
        from prompt_toolkit.history import InMemoryHistory
    except ImportError:
        print(
            "error: prompt_toolkit is required for interactive mode.\n"
            "Install it with: pip install prompt_toolkit",
            file=sys.stderr,
        )
        return 1

    cfg = load_config(config_path)
    registry = ModelRegistry(cfg)
    graph_querier = GraphQuerier(registry, cfg.neo4j)
    vector_querier = VectorQuerier(registry, cfg.qdrant)

    session = _Session()
    completer = WordCompleter(sorted(_COMMANDS), sentence=True)
    history = InMemoryHistory()
    prompt_session: PromptSession[str] = PromptSession(
        history=history,
        auto_suggest=AutoSuggestFromHistory(),
        completer=completer,
        multiline=False,
    )

    print(f"fieldnotes interactive mode (session {session.id})")
    print("Type a question, or /quit to exit. /help for commands.\n")

    try:
        while True:
            try:
                text = prompt_session.prompt("fieldnotes> ").strip()
            except KeyboardInterrupt:
                # Ctrl+C cancels current input, continues loop.
                print()
                continue
            except EOFError:
                # Ctrl+D exits.
                print("\nBye.")
                break

            if not text:
                continue

            # Handle commands.
            if text.startswith("/"):
                cmd = text.split()[0].lower()
                if cmd in ("/quit", "/exit", "/q"):
                    print("Bye.")
                    break
                elif cmd == "/history":
                    if not session.history:
                        print("No history yet.")
                    else:
                        for i, turn in enumerate(session.history, 1):
                            print(f"  {i}. {turn.question}")
                    print()
                    continue
                elif cmd == "/clear":
                    session.history.clear()
                    print("Conversation cleared.\n")
                    continue
                elif cmd == "/verbose":
                    session.verbose = not session.verbose
                    state = "on" if session.verbose else "off"
                    print(f"Verbose mode {state}.\n")
                    continue
                elif cmd == "/help":
                    print("Commands:")
                    print("  /history  — Show conversation history")
                    print("  /clear    — Clear conversation history")
                    print("  /verbose  — Toggle source citations")
                    print("  /quit     — Exit")
                    print()
                    continue
                else:
                    print(f"Unknown command: {cmd}")
                    print("Available: /history, /clear, /verbose, /quit\n")
                    continue

            # Ask the question.
            try:
                answer = _synthesize(
                    text,
                    registry=registry,
                    graph_querier=graph_querier,
                    vector_querier=vector_querier,
                    session=session,
                )
                print(f"\n{answer}\n")
            except KeyboardInterrupt:
                print("\n(cancelled)\n")
            except Exception as exc:
                print(f"error: {exc}\n", file=sys.stderr)
    finally:
        graph_querier.close()
        vector_querier.close()

    return 0


def run_ask(
    question: str | None,
    *,
    config_path: Path | None,
) -> int:
    """Entry point for the ``ask`` subcommand.

    If *question* is provided, runs one-shot mode. Otherwise starts the REPL.
    """
    if question is None:
        return _run_repl(config_path=config_path)

    # One-shot mode.
    cfg = load_config(config_path)
    registry = ModelRegistry(cfg)
    graph_querier = GraphQuerier(registry, cfg.neo4j)
    vector_querier = VectorQuerier(registry, cfg.qdrant)

    try:
        answer = _synthesize(
            question,
            registry=registry,
            graph_querier=graph_querier,
            vector_querier=vector_querier,
        )
        print(answer)
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        graph_querier.close()
        vector_querier.close()
