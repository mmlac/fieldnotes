"""Repository file parser.

Transforms IngestEvents from the RepositorySource into ParsedDocuments with
GraphHints linking Repository nodes to File nodes via CONTAINS relationships.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import BaseParser, GraphHint, ParsedDocument
from .registry import register

logger = logging.getLogger(__name__)


@register
class RepositoryParser(BaseParser):
    """Parses repository file IngestEvents into ParsedDocuments with graph hints."""

    @property
    def source_type(self) -> str:
        return "repositories"

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

        repo_name: str = meta.get("repo_name", "")
        repo_path: str = meta.get("repo_path", "")
        remote_url: str | None = meta.get("remote_url")
        relative_path: str = meta.get("relative_path", "")

        # Skip binary files — text content will be absent for non-text mime types
        text: str = event.get("text", "")
        mime_type: str = event.get("mime_type", "text/plain")
        if not text and not mime_type.startswith("text/"):
            logger.warning(
                "Skipping binary file %s in repo %s (mime: %s)",
                relative_path,
                repo_name,
                mime_type,
            )
            return []

        # File node properties
        node_props: dict[str, Any] = {
            "path": relative_path,
            "name": relative_path.rsplit("/", 1)[-1] if relative_path else "",
            "ext": _file_ext(relative_path),
        }
        if meta.get("sha256"):
            node_props["sha256"] = meta["sha256"]
        if event.get("source_modified_at"):
            node_props["modified_at"] = event["source_modified_at"]

        # Repository → File CONTAINS relationship
        graph_hints: list[GraphHint] = [
            GraphHint(
                subject_id=repo_path,
                subject_label="Repository",
                predicate="CONTAINS",
                object_id=source_id,
                object_label="File",
                subject_props={
                    "name": repo_name,
                    "path": repo_path,
                    "remote_url": remote_url,
                },
                object_props={},
                subject_merge_key="source_id",
                object_merge_key="source_id",
            ),
        ]

        return [
            ParsedDocument(
                source_type=self.source_type,
                source_id=source_id,
                operation=operation,
                text=text,
                mime_type=mime_type,
                node_label="File",
                node_props=node_props,
                graph_hints=graph_hints,
                source_metadata={
                    "repo_name": repo_name,
                    "repo_path": repo_path,
                    "relative_path": relative_path,
                },
            )
        ]


def _file_ext(path: str) -> str:
    """Extract file extension from a path (e.g. '.md', '.toml')."""
    dot = path.rfind(".")
    if dot == -1 or dot == len(path) - 1:
        return ""
    return path[dot:]
