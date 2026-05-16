from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
import os
import re
import urllib.parse
from typing import Any

from worker.parsers._slack_permalink import (
    DEFAULT_WORKSPACE_MAP_PATH,
    _SLACK_PERMALINK_RE,
    load_workspace_team_map,
    slack_permalink_to_source_id,
)

# ---------------------------------------------------------------------------
# Email canonicalization — shared across Gmail, Calendar, and Obsidian parsers
# ---------------------------------------------------------------------------

# Domains that are aliases for gmail.com.  Google treats these as the same
# mailbox, so normalising them avoids duplicate Person nodes.
_GMAIL_ALIASES = frozenset({"googlemail.com"})

# Matches email addresses in free text.
_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
)


def canonicalize_email(raw: str) -> str:
    """Normalise an email address for cross-source Person matching.

    * Strips whitespace, lower-cases.
    * Rewrites ``@googlemail.com`` → ``@gmail.com``.
    * Strips ``+tag`` plus-addressing from the local part on Google
      mailboxes (``@gmail.com`` / ``@googlemail.com``).  Google treats
      ``alice+work@gmail.com`` and ``alice@gmail.com`` as the same
      mailbox.  Plus-addressing is NOT stripped on other domains —
      mailbox semantics vary across providers and some treat ``+`` as a
      literal local-part character.
    """
    email = raw.strip().lower()
    if not email or "@" not in email:
        return email
    local, domain = email.rsplit("@", 1)
    if domain in _GMAIL_ALIASES:
        domain = "gmail.com"
    if domain == "gmail.com":
        local = local.split("+", 1)[0]
    return f"{local}@{domain}"


def extract_email_person_hints(
    text: str,
    source_id: str,
    subject_label: str = "File",
) -> list["GraphHint"]:
    """Extract email addresses from free text and return MENTIONS Person hints.

    Creates a ``subject -[MENTIONS]-> Person`` GraphHint for every unique
    email address found.  Person nodes use the ``email`` merge key so they
    bridge automatically with Gmail, Calendar, and Git Person nodes.
    """
    seen: set[str] = set()
    hints: list[GraphHint] = []
    for match in _EMAIL_RE.finditer(text):
        email = canonicalize_email(match.group(0))
        if email in seen:
            continue
        seen.add(email)
        hints.append(
            GraphHint(
                subject_id=source_id,
                subject_label=subject_label,
                predicate="MENTIONS",
                object_id=f"person:{email}",
                object_label="Person",
                object_props={"email": email},
                subject_merge_key="source_id",
                object_merge_key="email",
                confidence=1.0,
            )
        )
    return hints


# ---------------------------------------------------------------------------
# Source-link extraction — matches bare source-id URLs embedded in text
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# Greedy non-whitespace match; trailing sentence punctuation is stripped after.
# Lookbehind (?<![=&]) skips URLs that appear as query-string values inside
# https:// links (e.g. "?ref=gmail://...").
_SOURCE_URL_RE = re.compile(
    r"(?<![=&])(?:gmail|google-calendar|omnifocus|slack|obsidian)://\S+"
)

_SCHEME_TO_LABEL: dict[str, str] = {
    "gmail": "Email",
    "google-calendar": "CalendarEvent",
    "omnifocus": "Task",
    "slack": "SlackMessage",
    "obsidian": "File",
}

# Sentence-ending punctuation stripped from the right of a matched URL.
_URL_TRAIL_STRIP = ".,;:!?)>]"


def _normalize_omnifocus_url(url: str) -> str:
    """Normalize omnifocus://task/{task_id} to canonical omnifocus://{task_id}."""
    prefix = "omnifocus://task/"
    if url.startswith(prefix):
        return "omnifocus://" + url[len(prefix):]
    return url


def _resolve_obsidian_url(url: str, obsidian_vaults: dict[str, str] | None) -> str:
    """Resolve obsidian:// URL to canonical absolute-path source_id."""
    try:
        params = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        vault_name = params.get("vault", [""])[0]
        rel_path = urllib.parse.unquote(params.get("file", [""])[0])
        if not vault_name or not rel_path:
            return url
        if obsidian_vaults and vault_name in obsidian_vaults:
            return os.path.join(obsidian_vaults[vault_name], rel_path)
        logger.debug(
            "Obsidian vault %r not in vault map; emitting URL as source_id (will dangle)",
            vault_name,
        )
        return url
    except Exception:
        return url


# Module-level vault map used as fallback by extract_source_link_hints.
# Set once at startup via configure_obsidian_vaults() after vault discovery.
_obsidian_vaults: dict[str, str] | None = None


def configure_obsidian_vaults(vaults: dict[str, str]) -> None:
    """Set the vault-name → absolute-path map for obsidian:// URL resolution.

    Call once at startup after vault discovery.  Parsers calling
    extract_source_link_hints without an explicit obsidian_vaults= will use
    this map.
    """
    global _obsidian_vaults
    _obsidian_vaults = vaults


def extract_source_link_hints(
    text: str,
    source_id: str,
    subject_label: str = "File",
    obsidian_vaults: dict[str, str] | None = None,
    *,
    workspace_map: dict[str, str] | None = None,
) -> list["GraphHint"]:
    """Extract embedded source-id URLs and Slack permalink URLs, returning REFERENCES GraphHints.

    Scans *text* for:
      - gmail://{account}/message/{message_id}        → Email
      - google-calendar://{account}/event/{event_id}  → CalendarEvent
      - omnifocus://task/{task_id}                    → Task (normalized to omnifocus://{task_id})
      - slack://{team_id}/{channel_id}/{ts}           → SlackMessage
      - obsidian://open?vault={vault}&file={rel_path} → File
      - https://{workspace}.slack.com/archives/{channel}/p{ts} → SlackMessage

    De-dupes within a single call so the same URL appearing twice produces one edge.

    obsidian_vaults: vault-name → absolute-path map; falls back to the module-level
    global set by configure_obsidian_vaults().  None means obsidian:// URLs pass
    through as-is (will dangle in the graph until a vault map is configured).

    workspace_map: Slack workspace-subdomain → team_id map; loads from the default
    path written by the Slack source on startup when omitted.
    """
    if obsidian_vaults is None:
        obsidian_vaults = _obsidian_vaults
    if workspace_map is None:
        workspace_map = load_workspace_team_map(DEFAULT_WORKSPACE_MAP_PATH)

    seen: set[str] = set()
    hints: list[GraphHint] = []

    for match in _SOURCE_URL_RE.finditer(text):
        url = match.group(0).rstrip(_URL_TRAIL_STRIP)
        scheme = url.split("://", 1)[0]
        object_label = _SCHEME_TO_LABEL.get(scheme, "File")
        if scheme == "omnifocus":
            object_id = _normalize_omnifocus_url(url)
        elif scheme == "obsidian":
            object_id = _resolve_obsidian_url(url, obsidian_vaults)
        else:
            object_id = url
        if object_id in seen:
            continue
        seen.add(object_id)
        hints.append(
            GraphHint(
                subject_id=source_id,
                subject_label=subject_label,
                predicate="REFERENCES",
                object_id=object_id,
                object_label=object_label,
                subject_merge_key="source_id",
                object_merge_key="source_id",
                confidence=1.0,
            )
        )

    for m in _SLACK_PERMALINK_RE.finditer(text):
        target_id = slack_permalink_to_source_id(m.group(0), workspace_map)
        if not target_id or target_id in seen:
            continue
        seen.add(target_id)
        hints.append(
            GraphHint(
                subject_id=source_id,
                subject_label=subject_label,
                predicate="REFERENCES",
                object_id=target_id,
                object_label="SlackMessage",
                subject_merge_key="source_id",
                object_merge_key="source_id",
                confidence=1.0,
            )
        )

    return hints


@dataclass
class GraphHint:
    """A pre-extracted graph fact that bypasses the LLM extractor.

    Used when the adapter has high-confidence structured data (Obsidian wikilinks,
    email headers, git commit metadata) that does not need LLM inference.
    Hints are written to Neo4j directly after the write step, before entity resolution.
    """

    subject_id: str  # source_id of the subject node
    subject_label: str  # Neo4j label: "File", "Email", "Person", etc.
    predicate: str  # relationship type: "LINKS_TO", "SENT", etc.
    object_id: str  # source_id of the object node
    object_label: str
    subject_props: dict[str, Any] = field(default_factory=dict)
    object_props: dict[str, Any] = field(default_factory=dict)
    # Properties stamped onto the relationship itself (e.g. ``account``)
    # so callers can query "edges scoped to a particular Gmail account".
    edge_props: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0  # 1.0 = certain (derived from structure, not LLM)
    # Merge keys: which property to MERGE on for each node.
    # Default "source_id" works for most nodes; email nodes use "email",
    # Thread nodes use "thread_id", etc.
    subject_merge_key: str = "source_id"
    object_merge_key: str = "source_id"


@dataclass
class ParsedDocument:
    """The normalised output of any adapter parser.

    Every parser must produce one of these. The pipeline beyond this point
    is entirely source-agnostic — it only sees ParsedDocuments.
    """

    # Identity
    source_type: str  # matches Source.Name() in the Go daemon
    source_id: str  # stable external identifier (path, message_id, etc.)
    operation: str  # "created" | "modified" | "deleted"

    # Content for the text pipeline (chunker → embedder → LLM extractor)
    text: str  # plain text body; empty for image-only documents
    mime_type: str = "text/plain"

    # Node properties written directly to Neo4j (no LLM involved)
    # Keys become node properties; the writer merges these on upsert.
    node_label: str = "File"
    node_props: dict[str, Any] = field(default_factory=dict)

    # Pre-extracted graph facts (high-confidence, bypass LLM extractor)
    graph_hints: list[GraphHint] = field(default_factory=list)

    # Binary content for the vision pipeline (images only)
    image_bytes: bytes | None = None

    # Source metadata passed through to Qdrant payload
    source_metadata: dict[str, Any] = field(default_factory=dict)



class BaseParser(ABC):
    """Base class for all adapter parsers."""

    @property
    @abstractmethod
    def source_type(self) -> str:
        """Must match the Go Source.Name() for this adapter."""
        ...

    @abstractmethod
    def parse(self, event: dict[str, Any]) -> list[ParsedDocument]:
        """Transform a raw IngestEvent dict into one or more ParsedDocuments.

        Returns a list because some events produce multiple documents —
        an Obsidian note with embedded images yields one text document
        and N image documents. An email thread yields one document per message.
        """
        ...

    def configure(self, cfg: dict[str, Any]) -> None:
        """Optional: receive the source's config section on startup."""
        pass
