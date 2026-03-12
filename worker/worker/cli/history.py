"""Conversation persistence: save, resume, and browse past Q&A sessions.

Stores conversations as JSON files in ``~/.fieldnotes/conversations/``.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("worker.cli.history")

_CONVERSATIONS_DIR = Path.home() / ".fieldnotes" / "conversations"
_MAX_CONVERSATIONS = 100


@dataclass
class TurnRecord:
    """A single Q&A turn within a conversation."""

    question: str
    answer: str
    sources_found: int = 0
    source_ids: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: _iso_now())


@dataclass
class Conversation:
    """A full conversation with metadata and turns."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: str = field(default_factory=lambda: _iso_now())
    updated_at: str = field(default_factory=lambda: _iso_now())
    turns: list[TurnRecord] = field(default_factory=list)

    def add_turn(self, turn: TurnRecord) -> None:
        self.turns.append(turn)
        self.updated_at = _iso_now()

    @property
    def first_question(self) -> str:
        if self.turns:
            return self.turns[0].question
        return "(empty)"


def _iso_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ensure_dir() -> Path:
    _CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)
    return _CONVERSATIONS_DIR


def _conversation_path(conv_id: str) -> Path:
    return _ensure_dir() / f"{conv_id}.json"


def save_conversation(conv: Conversation) -> None:
    """Persist a conversation to disk as JSON."""
    path = _conversation_path(conv.id)
    data = _serialize(conv)
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        logger.warning("Failed to save conversation %s: %s", conv.id, exc)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def load_conversation(conv_id: str) -> Conversation | None:
    """Load a conversation by ID, or *None* if not found / corrupted."""
    path = _conversation_path(conv_id)
    return _load_from_path(path)


def load_most_recent() -> Conversation | None:
    """Load the most recently updated conversation, or *None*."""
    convs = list_conversations(limit=1)
    if convs:
        return load_conversation(convs[0].id)
    return None


def list_conversations(limit: int = 50) -> list[Conversation]:
    """Return conversations sorted by updated_at (newest first).

    Only loads metadata and the first turn to keep it fast.
    """
    conv_dir = _ensure_dir()
    entries: list[tuple[str, Path]] = []

    for p in conv_dir.glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            updated = data.get("updated_at", "")
            entries.append((updated, p))
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug("Skipping corrupted conversation file %s: %s", p, exc)
            continue

    entries.sort(key=lambda e: e[0], reverse=True)

    result: list[Conversation] = []
    for _, p in entries[:limit]:
        conv = _load_from_path(p)
        if conv is not None:
            result.append(conv)

    return result


def prune_old_conversations(max_keep: int = _MAX_CONVERSATIONS) -> int:
    """Remove oldest conversations beyond *max_keep*. Returns count removed."""
    all_convs = list_conversations(limit=max_keep + 1000)
    if len(all_convs) <= max_keep:
        return 0

    to_remove = all_convs[max_keep:]
    removed = 0
    for conv in to_remove:
        path = _conversation_path(conv.id)
        try:
            path.unlink(missing_ok=True)
            removed += 1
        except OSError as exc:
            logger.debug("Failed to prune %s: %s", conv.id, exc)
    return removed


# ── Serialization helpers ────────────────────────────────────────────


def _serialize(conv: Conversation) -> dict[str, Any]:
    return {
        "id": conv.id,
        "created_at": conv.created_at,
        "updated_at": conv.updated_at,
        "turns": [asdict(t) for t in conv.turns],
    }


def _deserialize(data: dict[str, Any]) -> Conversation:
    turns = []
    for t in data.get("turns", []):
        turns.append(
            TurnRecord(
                question=t.get("question", ""),
                answer=t.get("answer", ""),
                sources_found=t.get("sources_found", 0),
                source_ids=t.get("source_ids", []),
                timestamp=t.get("timestamp", ""),
            )
        )
    return Conversation(
        id=data.get("id", uuid.uuid4().hex[:12]),
        created_at=data.get("created_at", ""),
        updated_at=data.get("updated_at", ""),
        turns=turns,
    )


def _load_from_path(path: Path) -> Conversation | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return _deserialize(data)
    except (json.JSONDecodeError, OSError, KeyError, TypeError) as exc:
        logger.debug("Skipping corrupted conversation file %s: %s", path, exc)
        return None
