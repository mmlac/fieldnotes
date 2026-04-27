"""Shared subtype filters for Slack messages.

The Slack source is the authoritative layer: it filters system events
and empty bot_messages BEFORE window-splitting so the token-budget
accounting that drives chunk boundaries matches what the parser later
renders.  The parser imports the same predicate as a defense-in-depth
backstop in case a stray filtered message slips past the source (older
queue entries, future subtypes, etc.).

Keeping the canonical list here means a single edit covers both layers.
"""

from __future__ import annotations

from typing import Any

# Subtypes whose body text is never indexable on its own.  Slack's
# ``bot_message`` is intentionally NOT in this set — bots routinely
# post substantive messages; only the empty-text variants are dropped
# (see :func:`is_filtered_subtype`).
SLACK_FILTERED_SUBTYPES = frozenset({"channel_join", "channel_leave"})


def is_filtered_subtype(msg: dict[str, Any]) -> bool:
    """True iff *msg* should be excluded from indexing.

    Drops ``channel_join``/``channel_leave`` events outright and any
    ``bot_message`` that carries no body text.  Edit/delete envelopes
    (``message_changed`` / ``message_deleted``) are handled separately
    by the source's edit-and-delete path and are not filtered here.
    """
    subtype = msg.get("subtype")
    if subtype in SLACK_FILTERED_SUBTYPES:
        return True
    if subtype == "bot_message" and not (msg.get("text") or "").strip():
        return True
    return False
