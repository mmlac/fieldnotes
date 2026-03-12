"""Parser for macOS application scanner events.

Transforms IngestEvents from the macos_apps source into ParsedDocuments
with Application node labels, plus GraphHints for Category nodes and
CATEGORIZED_AS edges.

Registered for source_type='macos_apps'.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import BaseParser, GraphHint, ParsedDocument
from .registry import register

logger = logging.getLogger(__name__)


@register
class MacOSAppsParser(BaseParser):
    """Parses macOS .app bundle events into Application nodes."""

    @property
    def source_type(self) -> str:
        return "macos_apps"

    def parse(self, event: dict[str, Any]) -> list[ParsedDocument]:
        source_id = event.get("source_id", "")
        operation = event.get("operation", "created")
        meta = event.get("meta", {})

        name = meta.get("name", "")
        bundle_id = meta.get("bundle_id", "")
        version = meta.get("version", "")
        path = meta.get("path", "")
        category = meta.get("category", "")

        # Build text summary for chunking/embedding
        lines = [name]
        if bundle_id:
            lines.append(f"Bundle ID: {bundle_id}")
        if version:
            lines.append(f"Version: {version}")
        if path:
            lines.append(f"Path: {path}")
        if category:
            lines.append(f"Category: {category}")
        text = "\n".join(lines)

        node_props: dict[str, Any] = {
            "name": name,
            "bundle_id": bundle_id,
            "version": version,
            "path": path,
        }
        if category:
            node_props["category"] = category

        hints: list[GraphHint] = []

        # Create Category node and CATEGORIZED_AS edge if category exists
        if category:
            cat_name = _normalize_category(category)
            hints.append(GraphHint(
                subject_id=source_id,
                subject_label="Application",
                predicate="CATEGORIZED_AS",
                object_id=f"category://{cat_name}",
                object_label="Category",
                subject_props={"bundle_id": bundle_id, "name": name},
                object_props={"name": cat_name},
                subject_merge_key="bundle_id",
                object_merge_key="name",
            ))

        return [
            ParsedDocument(
                source_type="macos_apps",
                source_id=source_id,
                operation=operation,
                text=text,
                node_label="Application",
                node_props=node_props,
                graph_hints=hints,
                source_metadata={"date": event.get("source_modified_at", "")},
            )
        ]


def _normalize_category(raw: str) -> str:
    """Normalize an LSApplicationCategoryType to a human-readable name.

    Converts e.g. "public.app-category.developer-tools" to "Developer Tools".
    """
    # Strip common Apple prefix
    name = raw
    if name.startswith("public.app-category."):
        name = name[len("public.app-category."):]

    # Convert kebab-case to title case
    return name.replace("-", " ").title()
