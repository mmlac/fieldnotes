"""LLM prompt builder for ``fieldnotes person <id> --summary``.

Constructs the :class:`~worker.models.base.CompletionRequest` that the
brief generator sends.  The system prompt is a structured-summarizer
instruction with a hard 'do not invent' rule and a fixed bullet schema;
the user message is the rendered pre-brief from
:func:`worker.query.person_brief.format_prebrief`.
"""

from __future__ import annotations

from worker.models.base import CompletionRequest
from worker.query.person_brief import PreBrief, format_prebrief


SYSTEM_PROMPT = (
    "You are a structured meeting brief assistant.  Given a structured "
    "snapshot of pending work involving one person, produce 3–7 short "
    "bullets that answer 'what do I need to discuss with this person "
    "next?'.\n\n"
    "Group bullets under these headings (omit a heading if it has no "
    "items):\n"
    "  - Decisions awaited from them\n"
    "  - Decisions awaited from you\n"
    "  - Open threads to close\n"
    "  - Background\n\n"
    "Each bullet must cite the [Source] block it came from "
    "(e.g. '[Open OmniFocus tasks]', '[Outstanding email threads]', "
    "'[Unresolved Slack mentions of you]', '[Top active topics]', "
    "'[Obsidian People note]', '[Upcoming meeting context]').\n\n"
    "Do not invent items not present in the context.  If a section is "
    "empty, do not synthesise items for it."
)


_DEFAULT_MAX_TOKENS = 800
_DEFAULT_TEMPERATURE = 0.2


def build_brief_request(
    prebrief: PreBrief,
    *,
    since_label: str,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    temperature: float = _DEFAULT_TEMPERATURE,
) -> CompletionRequest:
    """Build the LLM request for *prebrief*.

    Temperature is low (0.2) so output is reproducible run-to-run; the
    schema-constraining system prompt does the heavy lifting.
    """
    user = format_prebrief(prebrief, since_label=since_label)
    return CompletionRequest(
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
