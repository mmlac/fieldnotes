"""LLM prompt builder for ``fieldnotes itinerary`` per-meeting briefs.

Constructs the :class:`~worker.models.base.CompletionRequest` that the
itinerary brief generator sends per event.  System prompt enforces a
hard 1–2 sentence shape with at least one ``[Source]`` citation and a
'do not invent' rule; the user message is the rendered pre-brief from
:func:`worker.query.itinerary_brief.format_event_brief`.
"""

from __future__ import annotations

from worker.models.base import CompletionRequest
from worker.query.itinerary_brief import EventBrief, format_event_brief


SYSTEM_PROMPT = (
    "You produce a 1-2 sentence brief for an upcoming meeting.  Cite at "
    "least one source in [brackets] inline (e.g. [Email], "
    "[Slack #channel], [OmniFocus], [Obsidian: path], [Calendar]).  If "
    "there is no relevant context, say so plainly.  Do not invent items "
    "not present in the structured context.  Avoid social-pleasantry "
    "filler."
)

_DEFAULT_MAX_TOKENS = 150
_DEFAULT_TEMPERATURE = 0.2


def build_event_brief_request(
    brief: EventBrief,
    *,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    temperature: float = _DEFAULT_TEMPERATURE,
) -> CompletionRequest:
    """Build the LLM request for one event's *brief*.

    Temperature is low (0.2) so the brief is reproducible run-to-run;
    the schema-constraining system prompt does the heavy lifting.
    """
    user = format_event_brief(brief)
    return CompletionRequest(
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
