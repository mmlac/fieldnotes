from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import re
from typing import Any

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
    """
    email = raw.strip().lower()
    if not email or "@" not in email:
        return email
    local, domain = email.rsplit("@", 1)
    if domain in _GMAIL_ALIASES:
        domain = "gmail.com"
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
