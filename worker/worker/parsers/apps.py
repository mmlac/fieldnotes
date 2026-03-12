"""App and tool parser for macOS application and Homebrew events.

Transforms IngestEvents from MacOSAppsSource and HomebrewSource into
ParsedDocuments with GraphHints for the knowledge graph.

- macOS app events (source_id ``app://…``) produce Application nodes
- Brew formula events (``brew://formula/…``) produce Tool nodes with
  PROVIDES edges to Command nodes
- Brew cask events (``brew://cask/…``) produce Application nodes and
  link to existing ones when bundle_id matches a known app

Deduplication: when a cask and an app bundle share a bundle_id, the
parser emits a SAME_AS hint so the graph writer can merge them.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import BaseParser, GraphHint, ParsedDocument
from .registry import register

logger = logging.getLogger(__name__)


@register
class MacOSAppsParser(BaseParser):
    """Parses macOS application IngestEvents into Application nodes."""

    @property
    def source_type(self) -> str:
        return "macos_apps"

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
        bundle_id: str = meta.get("bundle_id", "")
        version: str = meta.get("version", "")
        path: str = meta.get("path", "")
        category: str = meta.get("category", "")
        description: str = meta.get("description", "")

        # Build content text — include description if available
        parts = [name]
        if version:
            parts[0] = f"{name} (v{version})"
        if description and description != "Unknown application":
            parts.append(description)
        text = " — ".join(p for p in parts if p)

        node_props: dict[str, Any] = {
            "name": name,
            "bundle_id": bundle_id,
            "version": version,
            "path": path,
        }
        if description:
            node_props["description"] = description

        graph_hints: list[GraphHint] = []

        # Category edge from LSApplicationCategoryType
        if category:
            graph_hints.append(
                GraphHint(
                    subject_id=source_id,
                    subject_label="Application",
                    predicate="CATEGORIZED_AS",
                    object_id=f"category:{category}",
                    object_label="Category",
                    subject_props={},
                    object_props={"name": category},
                    subject_merge_key="source_id",
                    object_merge_key="name",
                )
            )

        return [
            ParsedDocument(
                source_type=self.source_type,
                source_id=source_id,
                operation=operation,
                text=text,
                node_label="Application",
                node_props=node_props,
                graph_hints=graph_hints,
                source_metadata={
                    "bundle_id": bundle_id,
                    "path": path,
                },
            )
        ]


@register
class HomebrewParser(BaseParser):
    """Parses Homebrew IngestEvents into Tool or Application nodes."""

    @property
    def source_type(self) -> str:
        return "homebrew"

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

        if source_id.startswith("brew://formula/"):
            return self._parse_formula(event, source_id, operation, meta)
        if source_id.startswith("brew://cask/"):
            return self._parse_cask(event, source_id, operation, meta)

        # Unknown brew package kind — fall back to generic
        logger.warning("Unknown brew source_id format: %s", source_id)
        return self._parse_formula(event, source_id, operation, meta)

    def _parse_formula(
        self,
        event: dict[str, Any],
        source_id: str,
        operation: str,
        meta: dict[str, Any],
    ) -> list[ParsedDocument]:
        """Parse a Homebrew formula into a Tool node with PROVIDES edges."""
        name: str = meta.get("package_name", "")
        version: str = meta.get("version", "")
        description: str = event.get("text", "")
        tap: str = meta.get("tap", "")
        homepage: str = meta.get("homepage", "")
        binaries: list[str] = meta.get("binaries", [])

        # Build content text
        parts = [name]
        if version:
            parts[0] = f"{name} (v{version})"
        if description:
            parts.append(description)
        text = " — ".join(p for p in parts if p)

        node_props: dict[str, Any] = {
            "name": name,
            "version": version,
            "description": description,
            "tap": tap,
            "homepage": homepage,
        }

        graph_hints: list[GraphHint] = []

        # Tool PROVIDES Command for each binary
        for binary in binaries:
            graph_hints.append(
                GraphHint(
                    subject_id=source_id,
                    subject_label="Tool",
                    predicate="PROVIDES",
                    object_id=f"command:{binary}",
                    object_label="Command",
                    subject_props={},
                    object_props={"name": binary},
                    subject_merge_key="source_id",
                    object_merge_key="name",
                )
            )

        # Tool INSTALLED_VIA Source (brew formula)
        graph_hints.append(
            GraphHint(
                subject_id=source_id,
                subject_label="Tool",
                predicate="INSTALLED_VIA",
                object_id="source:brew_formula",
                object_label="Source",
                subject_props={},
                object_props={"name": "brew_formula"},
                subject_merge_key="source_id",
                object_merge_key="name",
            )
        )

        return [
            ParsedDocument(
                source_type=self.source_type,
                source_id=source_id,
                operation=operation,
                text=text,
                node_label="Tool",
                node_props=node_props,
                graph_hints=graph_hints,
                source_metadata={
                    "package_kind": "formula",
                    "tap": tap,
                },
            )
        ]

    def _parse_cask(
        self,
        event: dict[str, Any],
        source_id: str,
        operation: str,
        meta: dict[str, Any],
    ) -> list[ParsedDocument]:
        """Parse a Homebrew cask into an Application node.

        If bundle_id is present, emit a SAME_AS hint linking the cask
        to the corresponding ``app://`` Application node so the graph
        writer can merge them.
        """
        name: str = meta.get("package_name", "")
        version: str = meta.get("version", "")
        description: str = event.get("text", "")
        tap: str = meta.get("tap", "")
        homepage: str = meta.get("homepage", "")
        bundle_id: str = meta.get("bundle_id", "")

        # Build content text — prefer brew's description (higher quality)
        parts = [name]
        if version:
            parts[0] = f"{name} (v{version})"
        if description:
            parts.append(description)
        text = " — ".join(p for p in parts if p)

        node_props: dict[str, Any] = {
            "name": name,
            "version": version,
            "description": description,
            "tap": tap,
            "homepage": homepage,
            "installed_via": "brew_cask",
        }
        if bundle_id:
            node_props["bundle_id"] = bundle_id

        graph_hints: list[GraphHint] = []

        # Application INSTALLED_VIA Source (brew cask)
        graph_hints.append(
            GraphHint(
                subject_id=source_id,
                subject_label="Application",
                predicate="INSTALLED_VIA",
                object_id="source:brew_cask",
                object_label="Source",
                subject_props={},
                object_props={"name": "brew_cask"},
                subject_merge_key="source_id",
                object_merge_key="name",
            )
        )

        # Link to matching app:// Application node via SAME_AS
        if bundle_id:
            app_source_id = f"app://{bundle_id}"
            graph_hints.append(
                GraphHint(
                    subject_id=source_id,
                    subject_label="Application",
                    predicate="SAME_AS",
                    object_id=app_source_id,
                    object_label="Application",
                    subject_props={},
                    object_props={"bundle_id": bundle_id},
                    subject_merge_key="source_id",
                    object_merge_key="source_id",
                )
            )

        return [
            ParsedDocument(
                source_type=self.source_type,
                source_id=source_id,
                operation=operation,
                text=text,
                node_label="Application",
                node_props=node_props,
                graph_hints=graph_hints,
                source_metadata={
                    "package_kind": "cask",
                    "tap": tap,
                    "bundle_id": bundle_id,
                },
            )
        ]
