"""OmniFocus task parser.

Transforms IngestEvents from OmniFocusSource into ParsedDocuments with
GraphHints that model:

- Task nodes (label ``Task``) with status, dates, and flag properties
- Tag hierarchy as edges: each tag (e.g. ``People/Boss``) becomes a ``Tag``
  node with a ``TAGGED`` edge from the task
- Project membership: ``IN_PROJECT`` edge from task to ``Project`` node
- Parent task: ``SUBTASK_OF`` edge when a task has a parent
- Person mentions: emails and ``@mentions`` found in task name/note, plus
  tags under ``People/`` (e.g. ``People/Alice``)
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .base import BaseParser, GraphHint, ParsedDocument, canonicalize_email
from .registry import register

logger = logging.getLogger(__name__)

# Matches email addresses in free text (task notes, titles)
_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
)

# Matches @mentions like "@Alice" or "@Bob Smith" (word boundary terminated)
_AT_MENTION_RE = re.compile(
    r"@([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)",
)


@register
class OmniFocusParser(BaseParser):
    """Parses OmniFocus task IngestEvents into Task nodes with tag edges."""

    @property
    def source_type(self) -> str:
        return "omnifocus"

    def parse(self, event: dict[str, Any]) -> list[ParsedDocument]:
        source_id: str = event["source_id"]
        operation: str = event.get("operation", "created")
        meta: dict[str, Any] = event.get("meta", {})

        if operation == "deleted":
            return [
                ParsedDocument(
                    source_type=self.source_type,
                    source_id=source_id,
                    operation="deleted",
                    text="",
                )
            ]

        name: str = meta.get("name", "")
        note: str = meta.get("note", "")
        status: str = meta.get("status", "active")
        flagged: bool = meta.get("flagged", False)
        tags: list[str] = meta.get("tags", [])
        project: str = meta.get("project", "")
        parent_task: str = meta.get("parent_task", "")
        parent_task_id: str = meta.get("parent_task_id", "")

        # Dates
        creation_date = meta.get("creation_date")
        modification_date = meta.get("modification_date")
        completion_date = meta.get("completion_date")
        due_date = meta.get("due_date")
        defer_date = meta.get("defer_date")

        # Build content text for embedding
        parts = [name]
        if note:
            parts.append(note)
        if project:
            parts.append(f"Project: {project}")
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")
        if status != "active":
            parts.append(f"Status: {status}")
        text = "\n".join(parts)

        # Node properties
        node_props: dict[str, Any] = {
            "name": name,
            "status": status,
            "flagged": flagged,
        }
        if creation_date:
            node_props["creation_date"] = creation_date
        if modification_date:
            node_props["modification_date"] = modification_date
        if completion_date:
            node_props["completion_date"] = completion_date
        if due_date:
            node_props["due_date"] = due_date
        if defer_date:
            node_props["defer_date"] = defer_date
        if note:
            node_props["note"] = note

        # ── Graph hints ──────────────────────────────────────────
        graph_hints: list[GraphHint] = []

        # Tag edges: each tag becomes a Tag node with TAGGED edge
        for tag in tags:
            tag = tag.strip()
            if not tag:
                continue
            graph_hints.append(
                GraphHint(
                    subject_id=source_id,
                    subject_label="Task",
                    predicate="TAGGED",
                    object_id=f"omnifocus-tag:{tag}",
                    object_label="Tag",
                    object_props={"name": tag, "source": "omnifocus"},
                    subject_merge_key="source_id",
                    object_merge_key="source_id",
                    confidence=1.0,
                )
            )

        # Project edge
        if project:
            graph_hints.append(
                GraphHint(
                    subject_id=source_id,
                    subject_label="Task",
                    predicate="IN_PROJECT",
                    object_id=f"omnifocus-project:{project}",
                    object_label="Project",
                    object_props={"name": project, "source": "omnifocus"},
                    subject_merge_key="source_id",
                    object_merge_key="source_id",
                    confidence=1.0,
                )
            )

        # Parent task edge — only emit when we have the parent's ID so the
        # MERGE key matches the parent task's actual source_id format.
        if parent_task_id:
            graph_hints.append(
                GraphHint(
                    subject_id=source_id,
                    subject_label="Task",
                    predicate="SUBTASK_OF",
                    object_id=f"omnifocus://{parent_task_id}",
                    object_label="Task",
                    object_props={"name": parent_task, "source": "omnifocus"},
                    subject_merge_key="source_id",
                    object_merge_key="source_id",
                    confidence=0.95,
                )
            )

        # ── Person extraction ────────────────────────────────────
        graph_hints.extend(
            self._extract_people(source_id, name, note, tags)
        )

        return [
            ParsedDocument(
                source_type=self.source_type,
                source_id=source_id,
                operation=operation,
                text=text,
                node_label="Task",
                node_props=node_props,
                graph_hints=graph_hints,
                source_metadata={
                    "task_id": meta.get("id", ""),
                    "project": project,
                    "tags": tags,
                },
            )
        ]

    # ------------------------------------------------------------------
    # Person extraction from task text and tags
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_people(
        source_id: str,
        name: str,
        note: str,
        tags: list[str],
    ) -> list[GraphHint]:
        """Extract Person MENTIONS hints from emails, @mentions, and People/ tags."""
        hints: list[GraphHint] = []
        seen_names: set[str] = set()

        combined_text = f"{name}\n{note}" if note else name

        # 1. Email addresses in name/note → Person with email merge key
        for match in _EMAIL_RE.finditer(combined_text):
            email = canonicalize_email(match.group(0))
            if email in seen_names:
                continue
            seen_names.add(email)
            hints.append(
                GraphHint(
                    subject_id=source_id,
                    subject_label="Task",
                    predicate="MENTIONS",
                    object_id=f"person:{email}",
                    object_label="Person",
                    object_props={"email": email},
                    subject_merge_key="source_id",
                    object_merge_key="email",
                    confidence=1.0,
                )
            )

        # 2. @mentions in name/note → Person with name only
        for match in _AT_MENTION_RE.finditer(combined_text):
            person_name = match.group(1).strip()
            if not person_name or person_name.lower() in seen_names:
                continue
            seen_names.add(person_name.lower())
            hints.append(
                GraphHint(
                    subject_id=source_id,
                    subject_label="Task",
                    predicate="MENTIONS",
                    object_id=f"omnifocus-person:{person_name}",
                    object_label="Person",
                    object_props={"name": person_name, "source": "omnifocus"},
                    subject_merge_key="source_id",
                    object_merge_key="source_id",
                    confidence=0.9,
                )
            )

        # 3. Tags under "People/" → Person with name from tag leaf
        for tag in tags:
            tag = tag.strip()
            if not tag.startswith("People/"):
                continue
            person_name = tag[len("People/") :].strip()
            if not person_name or person_name.lower() in seen_names:
                continue
            seen_names.add(person_name.lower())
            hints.append(
                GraphHint(
                    subject_id=source_id,
                    subject_label="Task",
                    predicate="MENTIONS",
                    object_id=f"omnifocus-person:{person_name}",
                    object_label="Person",
                    object_props={"name": person_name, "source": "omnifocus"},
                    subject_merge_key="source_id",
                    object_merge_key="source_id",
                    confidence=0.95,
                )
            )

        return hints
