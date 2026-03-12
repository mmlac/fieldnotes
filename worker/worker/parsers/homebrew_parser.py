"""Parser for Homebrew source events.

Transforms IngestEvents from the homebrew source into ParsedDocuments
with Tool node labels, plus GraphHints for Command nodes (PROVIDES edges)
and INSTALLED_VIA edges linking casks to their Application nodes.

Registered for source_type='homebrew'.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import BaseParser, GraphHint, ParsedDocument
from .registry import register

logger = logging.getLogger(__name__)


@register
class HomebrewParser(BaseParser):
    """Parses Homebrew package events into Tool nodes."""

    @property
    def source_type(self) -> str:
        return "homebrew"

    def parse(self, event: dict[str, Any]) -> list[ParsedDocument]:
        source_id = event.get("source_id", "")
        operation = event.get("operation", "created")
        meta = event.get("meta", {})

        name = meta.get("package_name", "")
        kind = meta.get("package_kind", "formula")
        version = meta.get("version", "")
        homepage = meta.get("homepage", "")
        tap = meta.get("tap", "")
        binaries: list[str] = meta.get("binaries", [])
        bundle_id = meta.get("bundle_id", "")

        # Use text from event if available, otherwise build summary
        text = event.get("text", "")
        if not text:
            lines = [f"{name} ({kind})"]
            if version:
                lines.append(f"Version: {version}")
            if homepage:
                lines.append(f"Homepage: {homepage}")
            if binaries:
                lines.append(f"Binaries: {', '.join(binaries)}")
            text = "\n".join(lines)

        node_props: dict[str, Any] = {
            "name": name,
            "kind": kind,
            "version": version,
        }
        if homepage:
            node_props["homepage"] = homepage
        if tap:
            node_props["tap"] = tap

        hints: list[GraphHint] = []

        # Create Command nodes and PROVIDES edges for each binary
        for binary in binaries:
            hints.append(GraphHint(
                subject_id=source_id,
                subject_label="Tool",
                predicate="PROVIDES",
                object_id=f"cmd://{binary}",
                object_label="Command",
                subject_props={"name": name},
                object_props={"name": binary},
                subject_merge_key="name",
                object_merge_key="name",
            ))

        # Link cask to its Application node via INSTALLED_VIA
        if kind == "cask" and bundle_id:
            app_source_id = f"app://{bundle_id}"
            hints.append(GraphHint(
                subject_id=app_source_id,
                subject_label="Application",
                predicate="INSTALLED_VIA",
                object_id=source_id,
                object_label="Tool",
                subject_props={"bundle_id": bundle_id},
                object_props={"name": name},
                subject_merge_key="bundle_id",
                object_merge_key="name",
            ))

        return [
            ParsedDocument(
                source_type="homebrew",
                source_id=source_id,
                operation=operation,
                text=text,
                node_label="Tool",
                node_props=node_props,
                graph_hints=hints,
                source_metadata={"date": event.get("source_modified_at", "")},
            )
        ]
