"""Interactive Q&A against the knowledge graph.

Provides both one-shot (``fieldnotes ask "question"``) and interactive REPL
(``fieldnotes ask``) modes, reusing the same RAG+LLM pipeline as the MCP
``ask`` tool.

Streaming is enabled by default when stdout is a TTY. Use ``--no-stream``
for batch mode or ``--json`` for structured output.
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from worker.cli.display import display_progress, spinner
from worker.cli.reformulator import reformulate
from worker.config import load_config
from worker.models.base import CompletionRequest
from worker.models.resolver import ModelRegistry
from worker.query import EMPTY_CORPUS_MESSAGE, is_corpus_empty
from worker.query.graph import GraphQuerier, GraphQueryResult
from worker.query.hybrid import merge
from worker.query.vector import VectorQuerier, VectorQueryResult

from worker.cli.history import (
    Conversation,
    TurnRecord,
    load_conversation,
    load_most_recent,
    list_conversations,
    prune_old_conversations,
    save_conversation,
)

# Ensure provider registration side-effects run.
import worker.models.providers.ollama  # noqa: F401

logger = logging.getLogger("worker.cli.ask")

# REPL slash-commands
_COMMANDS = {"/history", "/clear", "/verbose", "/quit", "/save", "/sessions"}

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
    conversation: Conversation = field(default_factory=Conversation)


@dataclass
class _PreparedContext:
    """Result of retrieval + prompt building, ready for LLM call."""

    system_prompt: str
    user_prompt: str
    source_ids: list[str]
    errors: list[str]
    has_context: bool
    sparse: bool
    context_text: str
    empty: bool = False
    empty_corpus: bool = False


def _prepare_context(
    question: str,
    *,
    registry: ModelRegistry,
    graph_querier: GraphQuerier,
    vector_querier: VectorQuerier,
    session: _Session | None = None,
) -> _PreparedContext:
    """Run hybrid retrieval and build the RAG prompt."""
    verbose = session.verbose if session else False

    # --- 0. Reformulate follow-up questions ---
    search_query = question
    if session and session.history:
        try:
            extraction_model = registry.for_role("extraction")
        except KeyError:
            extraction_model = None
        if extraction_model is not None:
            history_pairs = [(t.question, t.answer) for t in session.history]
            search_query = reformulate(question, history_pairs, extraction_model)
            if search_query != question:
                logger.info("Reformulated: %r -> %r", question, search_query)

    # --- 0b. Check for empty corpus ---
    if is_corpus_empty(graph_querier, vector_querier):
        return _PreparedContext(
            system_prompt="",
            user_prompt="",
            source_ids=[],
            errors=[],
            has_context=False,
            sparse=False,
            context_text="",
            empty=True,
            empty_corpus=True,
        )

    # --- 1. Retrieve context ---
    graph_result: GraphQueryResult
    vector_result: VectorQueryResult

    with spinner("Searching..."):
        try:
            graph_result = graph_querier.query(search_query)
        except Exception as exc:
            logger.debug("Graph query failed: %s", exc)
            graph_result = GraphQueryResult(
                question=search_query, cypher="", error=str(exc)
            )

        try:
            vector_result = vector_querier.query(search_query, top_k=20)
        except Exception as exc:
            logger.debug("Vector query failed: %s", exc)
            vector_result = VectorQueryResult(question=search_query, error=str(exc))

        hybrid = merge(search_query, graph_result, vector_result)

    # Show progress tree with source breakdown.
    display_progress(
        hybrid,
        graph_result=graph_result,
        vector_result=vector_result,
        verbose=verbose,
    )

    context_text = hybrid.context
    if not context_text.strip():
        return _PreparedContext(
            system_prompt="",
            user_prompt="",
            source_ids=[],
            errors=[],
            has_context=False,
            sparse=False,
            context_text="",
            empty=True,
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
        history_block = "Previous conversation:\n" + "\n---\n".join(turns) + "\n\n"

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

    has_context = bool(hybrid.graph_results or hybrid.vector_results)
    sparse = (len(hybrid.graph_results) + len(hybrid.vector_results)) < 3

    return _PreparedContext(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        source_ids=unique_ids,
        errors=hybrid.errors,
        has_context=has_context,
        sparse=sparse,
        context_text=context_text,
    )


def _format_footer(
    ctx: _PreparedContext,
    *,
    show_sources: bool = True,
) -> str:
    """Build the footer text (warnings, sources, confidence)."""
    parts: list[str] = []

    if ctx.errors:
        parts.append("Warnings: " + "; ".join(ctx.errors))

    if ctx.source_ids and show_sources:
        from worker.cli.stream import format_sources

        parts.append(format_sources(ctx.source_ids))

    if not ctx.has_context:
        parts.append("[Confidence] no relevant context found")
    elif ctx.sparse:
        parts.append("[Confidence] low — limited context available")

    return "\n\n".join(parts)


def _record_turn(
    session: _Session,
    question: str,
    answer: str,
    context_text: str,
    source_ids: list[str],
) -> None:
    """Record a turn in session history and persist conversation."""
    session.history.append(
        _Turn(
            question=question,
            context=context_text,
            answer=answer,
        )
    )
    session.conversation.add_turn(
        TurnRecord(
            question=question,
            answer=answer,
            sources_found=len(source_ids),
            source_ids=source_ids,
        )
    )
    save_conversation(session.conversation)


def _synthesize(
    question: str,
    *,
    registry: ModelRegistry,
    graph_querier: GraphQuerier,
    vector_querier: VectorQuerier,
    session: _Session | None = None,
) -> str:
    """Run hybrid retrieval + LLM synthesis (non-streaming). Returns formatted answer text."""
    ctx = _prepare_context(
        question,
        registry=registry,
        graph_querier=graph_querier,
        vector_querier=vector_querier,
        session=session,
    )

    if ctx.empty:
        if ctx.empty_corpus:
            return EMPTY_CORPUS_MESSAGE
        return (
            "I don't have enough information in the knowledge graph "
            "to answer this question."
        )

    try:
        model = registry.for_role("query")
    except KeyError:
        model = registry.for_role("extraction")

    req = CompletionRequest(
        system=ctx.system_prompt,
        messages=[{"role": "user", "content": ctx.user_prompt}],
        temperature=0.2,
        timeout=120.0,
    )
    with spinner("Thinking..."):
        resp = model.complete(req, task="ask")

    # Format response.
    parts: list[str] = []
    if ctx.errors:
        parts.append("Warnings: " + "; ".join(ctx.errors))
    parts.append(resp.text)

    show_sources = session is None or session.verbose
    if ctx.source_ids and show_sources:
        parts.append("[Sources]\n" + "\n".join(ctx.source_ids))

    if not ctx.has_context:
        parts.append("[Confidence] no relevant context found")
    elif ctx.sparse:
        parts.append("[Confidence] low — limited context available")

    answer = "\n\n".join(parts)

    # Record turn in session history.
    if session is not None:
        _record_turn(session, question, resp.text, ctx.context_text, ctx.source_ids)

    return answer


def _synthesize_stream(
    question: str,
    *,
    registry: ModelRegistry,
    graph_querier: GraphQuerier,
    vector_querier: VectorQuerier,
    session: _Session | None = None,
) -> str:
    """Run hybrid retrieval + streaming LLM synthesis. Returns the raw answer text."""
    from worker.cli.stream import render_stream

    ctx = _prepare_context(
        question,
        registry=registry,
        graph_querier=graph_querier,
        vector_querier=vector_querier,
        session=session,
    )

    if ctx.empty:
        if ctx.empty_corpus:
            print(EMPTY_CORPUS_MESSAGE)
            return EMPTY_CORPUS_MESSAGE
        msg = (
            "I don't have enough information in the knowledge graph "
            "to answer this question."
        )
        print(msg)
        return msg

    if ctx.errors:
        print("Warnings: " + "; ".join(ctx.errors))

    try:
        model = registry.for_role("query")
    except KeyError:
        model = registry.for_role("extraction")

    req = CompletionRequest(
        system=ctx.system_prompt,
        messages=[{"role": "user", "content": ctx.user_prompt}],
        temperature=0.2,
        timeout=120.0,
    )

    chunks = model.stream_complete(req, task="ask")
    result = render_stream(chunks)

    # Print footer after streamed answer.
    show_sources = session is None or session.verbose
    footer = _format_footer(ctx, show_sources=show_sources)
    if footer:
        print(f"\n\n{footer}")

    # Record turn in session history.
    if session is not None:
        _record_turn(session, question, result.text, ctx.context_text, ctx.source_ids)

    return result.text


def _run_repl(
    *,
    config_path: Path | None,
    resume_id: str | None = None,
    stream: bool = True,
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

    # Resume a previous conversation if requested.
    if resume_id is not None:
        if resume_id == "":
            # --resume with no ID → most recent
            conv = load_most_recent()
        else:
            conv = load_conversation(resume_id)

        if conv is None:
            print("error: no conversation found to resume.", file=sys.stderr)
            graph_querier.close()
            vector_querier.close()
            return 1

        session.conversation = conv
        session.id = conv.id
        # Restore in-memory history from persisted turns.
        for turn in conv.turns:
            session.history.append(
                _Turn(
                    question=turn.question,
                    context="",
                    answer=turn.answer,
                )
            )
    else:
        session.conversation = Conversation(id=session.id)

    completer = WordCompleter(sorted(_COMMANDS), sentence=True)
    history = InMemoryHistory()
    prompt_session: PromptSession[str] = PromptSession(
        history=history,
        auto_suggest=AutoSuggestFromHistory(),
        completer=completer,
        multiline=False,
    )

    if resume_id is not None and session.history:
        print(f"fieldnotes interactive mode (resumed session {session.id})")
        print(f"  {len(session.history)} previous turn(s) loaded.")
    else:
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
                    # Start a new conversation (old one stays saved).
                    session.conversation = Conversation()
                    session.id = session.conversation.id
                    print("Conversation cleared (new session started).\n")
                    continue
                elif cmd == "/verbose":
                    session.verbose = not session.verbose
                    state = "on" if session.verbose else "off"
                    print(f"Verbose mode {state}.\n")
                    continue
                elif cmd == "/save":
                    save_conversation(session.conversation)
                    print(f"Conversation saved ({session.conversation.id}).\n")
                    continue
                elif cmd == "/sessions":
                    convs = list_conversations(limit=20)
                    if not convs:
                        print("No saved conversations.")
                    else:
                        for c in convs:
                            n_turns = len(c.turns)
                            q = c.first_question
                            if len(q) > 60:
                                q = q[:57] + "..."
                            print(f"  {c.id}  {c.updated_at}  ({n_turns} turns)  {q}")
                    print()
                    continue
                elif cmd == "/help":
                    print("Commands:")
                    print("  /history   — Show this session's questions")
                    print("  /sessions  — List saved conversations")
                    print("  /clear     — Clear history and start fresh")
                    print("  /save      — Force save current conversation")
                    print("  /verbose   — Toggle source citations")
                    print("  /quit      — Exit")
                    print()
                    continue
                else:
                    print(f"Unknown command: {cmd}")
                    print(
                        "Available: /history, /sessions, /clear, /save, /verbose, /quit\n"
                    )
                    continue

            # Ask the question.
            try:
                if stream:
                    print()  # blank line before streamed answer
                    _synthesize_stream(
                        text,
                        registry=registry,
                        graph_querier=graph_querier,
                        vector_querier=vector_querier,
                        session=session,
                    )
                    print()  # blank line after
                else:
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
        # Auto-prune old conversations.
        pruned = prune_old_conversations()
        if pruned:
            logger.debug("Pruned %d old conversation(s).", pruned)
        graph_querier.close()
        vector_querier.close()

    return 0


def run_ask(
    question: str | None,
    *,
    config_path: Path | None,
    verbose: bool = False,
    resume_id: str | None = None,
    stream: bool = True,
    json_output: bool = False,
) -> int:
    """Entry point for the ``ask`` subcommand.

    If *question* is provided, runs one-shot mode. Otherwise starts the REPL.
    *resume_id* triggers conversation resume (empty string = most recent).
    Streaming is on by default when stdout is a TTY; ``--no-stream`` or
    ``--json`` disables it.
    """
    if resume_id is not None or question is None:
        if json_output:
            print("error: --json requires a question argument", file=sys.stderr)
            return 1
        return _run_repl(
            config_path=config_path,
            resume_id=resume_id,
            stream=stream,
        )

    # One-shot mode.
    cfg = load_config(config_path)
    registry = ModelRegistry(cfg)
    graph_querier = GraphQuerier(registry, cfg.neo4j)
    vector_querier = VectorQuerier(registry, cfg.qdrant)

    # One-shot conversations are also persisted.
    session = _Session(verbose=verbose)
    session.conversation = Conversation(id=session.id)

    try:
        if json_output:
            return _run_json(
                question,
                registry=registry,
                graph_querier=graph_querier,
                vector_querier=vector_querier,
            )

        if stream:
            _synthesize_stream(
                question,
                registry=registry,
                graph_querier=graph_querier,
                vector_querier=vector_querier,
                session=session,
            )
            print()  # trailing newline
            return 0
        else:
            answer = _synthesize(
                question,
                registry=registry,
                graph_querier=graph_querier,
                vector_querier=vector_querier,
                session=session,
            )
            print(answer)
            return 0
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        graph_querier.close()
        vector_querier.close()


def run_history() -> int:
    """Entry point for the ``ask --history`` flag. Lists past conversations."""
    convs = list_conversations(limit=50)
    if not convs:
        print("No saved conversations.")
        return 0

    for conv in convs:
        n_turns = len(conv.turns)
        q = conv.first_question
        if len(q) > 70:
            q = q[:67] + "..."
        print(f"  {conv.id}  {conv.updated_at}  ({n_turns} turns)  {q}")
    return 0


def _run_json(
    question: str,
    *,
    registry: ModelRegistry,
    graph_querier: GraphQuerier,
    vector_querier: VectorQuerier,
) -> int:
    """Run a question and output structured JSON."""
    from worker.cli.stream import render_json

    start = time.monotonic()

    ctx = _prepare_context(
        question,
        registry=registry,
        graph_querier=graph_querier,
        vector_querier=vector_querier,
    )

    if ctx.empty:
        answer = (
            EMPTY_CORPUS_MESSAGE
            if ctx.empty_corpus
            else "I don't have enough information in the knowledge graph to answer this question."
        )
        render_json(
            question,
            answer,
            [],
            elapsed=time.monotonic() - start,
        )
        return 0

    try:
        model = registry.for_role("query")
    except KeyError:
        model = registry.for_role("extraction")

    req = CompletionRequest(
        system=ctx.system_prompt,
        messages=[{"role": "user", "content": ctx.user_prompt}],
        temperature=0.2,
        timeout=120.0,
    )
    resp = model.complete(req, task="ask")
    elapsed = time.monotonic() - start

    render_json(
        question,
        resp.text,
        ctx.source_ids,
        elapsed=elapsed,
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
    )
    return 0
