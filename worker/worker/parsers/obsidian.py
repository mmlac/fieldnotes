"""Obsidian vault parser.

Extracts frontmatter, wikilinks, tags, and image embeds from Obsidian markdown
notes and produces ParsedDocuments with GraphHints for the knowledge graph.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any

import frontmatter

from .base import BaseParser, GraphHint, ParsedDocument
from .registry import register

# [[target]] or [[target|alias]]  — but NOT ![[embed]]
_WIKILINK_RE = re.compile(r"(?<!\!)\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")

# ![[image.png]] or ![[path/to/image.jpg]]
_EMBED_RE = re.compile(r"\!\[\[([^\]]+\.(?:png|jpg|jpeg|gif|svg|webp|bmp))\]\]", re.IGNORECASE)

# #tag but not inside code fences or frontmatter; simplified inline match
_TAG_RE = re.compile(r"(?:^|\s)#([\w][\w/\-]*)", re.MULTILINE)


@register
class ObsidianParser(BaseParser):
    """Parses Obsidian markdown notes into ParsedDocuments."""

    @property
    def source_type(self) -> str:
        return "obsidian"

    def parse(self, event: dict[str, Any]) -> list[ParsedDocument]:
        source_id: str = event["source_id"]
        operation: str = event["operation"]
        text: str = event.get("text", "")
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

        # --- Parse frontmatter ------------------------------------------------
        post = frontmatter.loads(text)
        fm: dict[str, Any] = dict(post.metadata)
        body: str = post.content

        # Detect web clips
        is_web_clip = bool(fm.get("url") or fm.get("source_url"))

        # Build node properties from frontmatter
        node_props: dict[str, Any] = {}
        if fm.get("title"):
            node_props["title"] = fm["title"]
        if fm.get("aliases"):
            node_props["aliases"] = fm["aliases"]
        if is_web_clip:
            node_props["web_clip"] = True
            node_props["source_url"] = fm.get("url") or fm.get("source_url")
        if fm.get("created"):
            node_props["created"] = str(fm["created"])
        if fm.get("updated"):
            node_props["updated"] = str(fm["updated"])

        # --- Extract wikilinks → LINKS_TO GraphHints -------------------------
        graph_hints: list[GraphHint] = []

        for target in _WIKILINK_RE.findall(body):
            target = target.strip()
            if not target:
                continue
            graph_hints.append(
                GraphHint(
                    subject_id=source_id,
                    subject_label="File",
                    predicate="LINKS_TO",
                    object_id=target,
                    object_label="File",
                    confidence=0.95,
                )
            )

        # --- Extract tags → TAGGED_BY_USER GraphHints ------------------------
        for tag in _TAG_RE.findall(body):
            graph_hints.append(
                GraphHint(
                    subject_id=source_id,
                    subject_label="File",
                    predicate="TAGGED_BY_USER",
                    object_id=f"tag:{tag}",
                    object_label="Tag",
                    object_props={"source": "user"},
                    confidence=1.0,
                )
            )

        # Also extract tags declared in frontmatter
        fm_tags = fm.get("tags", [])
        if isinstance(fm_tags, str):
            fm_tags = [fm_tags]
        for tag in fm_tags:
            tag = str(tag).strip().lstrip("#")
            if not tag:
                continue
            graph_hints.append(
                GraphHint(
                    subject_id=source_id,
                    subject_label="File",
                    predicate="TAGGED_BY_USER",
                    object_id=f"tag:{tag}",
                    object_label="Tag",
                    object_props={"source": "user"},
                    confidence=1.0,
                )
            )

        # --- Build the main text ParsedDocument -------------------------------
        docs: list[ParsedDocument] = [
            ParsedDocument(
                source_type=self.source_type,
                source_id=source_id,
                operation=operation,
                text=body,
                node_label="File",
                node_props=node_props,
                graph_hints=graph_hints,
                source_metadata={
                    "frontmatter": fm,
                    "web_clip": is_web_clip,
                    **meta,
                },
            )
        ]

        # --- Handle ![[image.png]] embeds → separate image ParsedDocuments ----
        vault_root = meta.get("vault_root", "")
        for embed_path in _EMBED_RE.findall(body):
            # Build a source_id for the image relative to the vault
            if vault_root:
                image_id = str(PurePosixPath(vault_root) / embed_path)
            else:
                image_id = embed_path

            suffix = PurePosixPath(embed_path).suffix.lower()
            mime_map = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".svg": "image/svg+xml",
                ".webp": "image/webp",
                ".bmp": "image/bmp",
            }

            docs.append(
                ParsedDocument(
                    source_type=self.source_type,
                    source_id=image_id,
                    operation=operation,
                    text="",
                    mime_type=mime_map.get(suffix, "application/octet-stream"),
                    node_label="File",
                    node_props={"embedded_in": source_id},
                    source_metadata={"embed_path": embed_path, **meta},
                )
            )

        return docs
