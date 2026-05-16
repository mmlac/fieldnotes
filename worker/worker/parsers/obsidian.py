"""Obsidian vault parser.

Extracts frontmatter, wikilinks, tags, and image embeds from Obsidian markdown
notes and produces ParsedDocuments with GraphHints for the knowledge graph.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path, PurePosixPath
from typing import Any

import frontmatter

from .base import BaseParser, GraphHint, ParsedDocument, canonicalize_email, extract_source_link_hints
from .registry import register

logger = logging.getLogger(__name__)

# Max image size to load into memory (50 MiB) — prevents OOM on huge files.
_MAX_IMAGE_BYTES = 50 * 1024 * 1024
_MAX_EMBEDS_PER_NOTE = (
    50  # Cap embedded images to prevent loading 50GB+ from a single note
)

# [[target]] or [[target|alias]]  — but NOT ![[embed]]
_WIKILINK_RE = re.compile(r"(?<!\!)\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")

# ![[image.png]] or ![[path/to/image.jpg]]
_EMBED_RE = re.compile(
    r"\!\[\[([^\]]+\.(?:png|jpg|jpeg|gif|svg|webp|bmp))\]\]", re.IGNORECASE
)

# #tag but not inside code fences or frontmatter; simplified inline match
_TAG_RE = re.compile(r"(?:^|\s)#([\w][\w/\-]*)", re.MULTILINE)

# Patterns for stripping code before tag extraction
_FENCED_CODE_RE = re.compile(r"^(`{3,}|~{3,}).*?\n[\s\S]*?\n\1\s*$", re.MULTILINE)
_INLINE_CODE_RE = re.compile(r"`[^`]+`")


def _read_image_bytes(path: Path) -> bytes | None:
    """Read image bytes from *path*, returning ``None`` on failure.

    Skips files that exceed ``_MAX_IMAGE_BYTES`` to prevent OOM.
    """
    try:
        size = path.stat().st_size
        if size > _MAX_IMAGE_BYTES:
            logger.warning(
                "Skipping image %s — exceeds max size (%d > %d bytes)",
                path,
                size,
                _MAX_IMAGE_BYTES,
            )
            return None
        return path.read_bytes()
    except FileNotFoundError:
        logger.debug("Embedded image not found on disk: %s", path)
        return None
    except OSError as exc:
        logger.warning("Failed to read image %s: %s", path, exc)
        return None


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

        # Index-only files: emit metadata with filename, no content parsing.
        if meta.get("index_only"):
            import os

            name = os.path.basename(source_id)
            ext = os.path.splitext(source_id)[1]
            node_props: dict[str, Any] = {"path": source_id, "name": name, "ext": ext}
            if size := meta.get("size_bytes"):
                node_props["size_bytes"] = size
            if "source_modified_at" in event:
                node_props["modified_at"] = event["source_modified_at"]
            desc_text = f"File: {name}"
            directory = os.path.dirname(source_id)
            if directory:
                desc_text += f" in {directory}/"
            return [
                ParsedDocument(
                    source_type=self.source_type,
                    source_id=source_id,
                    operation=operation,
                    text=desc_text,
                    node_label="File",
                    node_props=node_props,
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
        categories_key = meta.get("categories_key", "categories")
        raw_cats = fm.get(categories_key, [])
        if isinstance(raw_cats, str):
            raw_cats = [raw_cats]
        if raw_cats:
            node_props["categories"] = raw_cats

        # --- Extract wikilinks → LINKS_TO GraphHints -------------------------
        graph_hints: list[GraphHint] = []

        for target in _WIKILINK_RE.findall(body):
            target = target.split("#")[0].strip()
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
        # Strip fenced code blocks and inline code so #tags inside them are ignored
        body_no_code = _FENCED_CODE_RE.sub("", body)
        body_no_code = _INLINE_CODE_RE.sub("", body_no_code)
        for tag in _TAG_RE.findall(body_no_code):
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

        # --- Synthesize category/name hierarchical tag hints -----------------
        # Merges onto the same Tag node that OmniFocus creates for this item,
        # so queries against that Tag return both OmniFocus tasks and this file.
        # categories is a list; one hint is emitted per category.
        categories = node_props.get("categories", [])
        name = (
            Path(meta["relative_path"]).stem if meta.get("relative_path") else None
        )
        if categories and name:
            for cat in categories:
                graph_hints.append(
                    GraphHint(
                        subject_id=source_id,
                        subject_label="File",
                        predicate="TAGGED_BY_USER",
                        object_id=f"omnifocus-tag:{cat}/{name}",
                        object_label="Tag",
                        object_props={"name": f"{cat}/{name}", "source": "omnifocus"},
                        confidence=1.0,
                    )
                )

        # --- Parse emails frontmatter → MENTIONS Person GraphHints -----------
        # Supports: emails: "a@b.com, c@d.com" (string) or emails: [a@b.com, c@d.com]
        raw_emails = fm.get("emails")
        if raw_emails:
            email_list: list[str] = []
            if isinstance(raw_emails, str):
                email_list = [e.strip() for e in raw_emails.split(",") if e.strip()]
            elif isinstance(raw_emails, list):
                email_list = [str(e).strip() for e in raw_emails if str(e).strip()]

            for raw_email in email_list:
                norm = canonicalize_email(raw_email)
                if not norm or "@" not in norm:
                    continue
                graph_hints.append(
                    GraphHint(
                        subject_id=source_id,
                        subject_label="File",
                        predicate="MENTIONS",
                        object_id=f"person:{norm}",
                        object_label="Person",
                        object_props={"email": norm},
                        object_merge_key="email",
                        confidence=1.0,
                    )
                )

        link_hints = extract_source_link_hints(body, source_id, "File")
        if link_hints:
            graph_hints.extend(link_hints)

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
        vault_root = meta.get("vault_root") or meta.get("vault_name", "")
        vault_path = meta.get("vault_path", "")
        embed_matches = _EMBED_RE.findall(body)
        if len(embed_matches) > _MAX_EMBEDS_PER_NOTE:
            logger.warning(
                "Note %s has %d image embeds, truncating to %d",
                source_id,
                len(embed_matches),
                _MAX_EMBEDS_PER_NOTE,
            )
            embed_matches = embed_matches[:_MAX_EMBEDS_PER_NOTE]
        for embed_path in embed_matches:
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

            # Load image bytes from disk when vault_path is available
            image_bytes: bytes | None = None
            if vault_path:
                vault_resolved = Path(vault_path).resolve()
                # Check unresolved path first to reject traversal attempts
                # like "../../../etc/passwd" before symlink resolution.
                image_unresolved = Path(vault_path) / embed_path
                if not image_unresolved.is_relative_to(vault_resolved):
                    logger.warning(
                        "Embed path %r escapes vault directory, skipping",
                        embed_path,
                    )
                    continue
                # Resolve symlinks, then re-check to catch symlink escapes.
                image_file = image_unresolved.resolve()
                if not image_file.is_relative_to(vault_resolved):
                    logger.warning(
                        "Embed path %r resolves outside vault directory "
                        "(symlink escape), skipping",
                        embed_path,
                    )
                    continue
                image_bytes = _read_image_bytes(image_file)

            docs.append(
                ParsedDocument(
                    source_type=self.source_type,
                    source_id=image_id,
                    operation=operation,
                    text="",
                    mime_type=mime_map.get(suffix, "application/octet-stream"),
                    node_label="File",
                    node_props={"embedded_in": source_id},
                    image_bytes=image_bytes,
                    source_metadata={"embed_path": embed_path, **meta},
                )
            )

        return docs
