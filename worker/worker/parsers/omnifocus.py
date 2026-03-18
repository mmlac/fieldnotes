"""OmniFocus task parser.

Transforms IngestEvents from OmniFocusSource into ParsedDocuments with
GraphHints that model:

- Task nodes (label ``Task``) with status, dates, and flag properties
- Tag hierarchy as edges: each tag (e.g. ``People/Boss``) becomes a ``Tag``
  node with a ``TAGGED`` edge from the task
- Project membership: ``IN_PROJECT`` edge from task to ``Project`` node
- Parent task: ``SUBTASK_OF`` edge when a task has a parent
"""

from __future__ import annotations

import logging
from typing import Any

from .base import BaseParser, GraphHint, ParsedDocument
from .registry import register

logger = logging.getLogger(__name__)


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

        # Parent task edge
        if parent_task:
            graph_hints.append(
                GraphHint(
                    subject_id=source_id,
                    subject_label="Task",
                    predicate="SUBTASK_OF",
                    object_id=f"omnifocus-task:{parent_task}",
                    object_label="Task",
                    object_props={"name": parent_task, "source": "omnifocus"},
                    subject_merge_key="source_id",
                    object_merge_key="source_id",
                    confidence=0.95,
                )
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
