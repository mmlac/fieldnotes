"""Reformulate follow-up questions into standalone queries using conversation history.

When a user asks "Tell me more about that" or "What about Alice?", the
reformulator expands the question by resolving pronouns and implicit
references against the recent conversation context.
"""

from __future__ import annotations

import logging
import re

from worker.models.base import CompletionRequest, CompletionResponse
from worker.models.resolver import ResolvedModel

logger = logging.getLogger("worker.cli.reformulator")

# Patterns that suggest the question references prior conversation context.
_CONTEXTUAL_PATTERNS = re.compile(
    r"\b(?:it|that|this|them|those|these|more|also|"
    r"first|second|third|fourth|fifth|last|previous|above|mentioned)\b",
    re.IGNORECASE,
)

_MAX_HISTORY_TURNS = 3

_REFORMULATION_PROMPT = """\
Given the conversation history below, rewrite the user's latest question \
as a standalone search query. Resolve pronouns, references, and implicit \
context. If the question is already standalone, return it unchanged.

Conversation:
{history}

Latest question: {question}

Standalone query:"""


def _needs_reformulation(question: str, history_len: int) -> bool:
    """Return True if the question likely references prior conversation."""
    if history_len == 0:
        return False
    if _CONTEXTUAL_PATTERNS.search(question):
        return True
    return False


def _format_history(
    turns: list[tuple[str, str]],
) -> str:
    """Format conversation turns (question, answer) for the prompt."""
    parts: list[str] = []
    for q, a in turns[-_MAX_HISTORY_TURNS:]:
        parts.append(f"Q: {q}\nA: {a}")
    return "\n---\n".join(parts)


def reformulate(
    question: str,
    history: list[tuple[str, str]],
    model: ResolvedModel,
) -> str:
    """Reformulate *question* into a standalone query using conversation *history*.

    Parameters
    ----------
    question:
        The user's latest question (may contain pronouns/references).
    history:
        List of ``(question, answer)`` tuples from prior turns.
    model:
        The model to use for reformulation (typically the extraction model).

    Returns
    -------
    str
        A standalone query suitable for hybrid search.
    """
    if not _needs_reformulation(question, len(history)):
        logger.debug("Question is standalone, skipping reformulation: %s", question)
        return question

    formatted = _format_history(history)
    prompt = _REFORMULATION_PROMPT.format(history=formatted, question=question)

    req = CompletionRequest(
        system=(
            "You rewrite follow-up questions into standalone search queries. "
            "Output ONLY the rewritten query, nothing else."
        ),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=256,
    )

    try:
        resp: CompletionResponse = model.complete(req, task="reformulate")
    except Exception as exc:
        logger.warning("Reformulation failed, using original question: %s", exc)
        return question

    result = resp.text.strip()
    if not result:
        return question

    logger.debug("Reformulated: %r -> %r", question, result)
    return result
