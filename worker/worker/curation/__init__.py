"""Person-merge curation: split, confirm, merge, never-same-as.

Public API exposed for CLI and MCP handlers.
"""

from __future__ import annotations

from worker.curation.audit import AuditLog
from worker.curation.persons import (
    CurationError,
    PersonCurator,
    PersonRef,
    parse_identifier,
)

__all__ = [
    "AuditLog",
    "CurationError",
    "PersonCurator",
    "PersonRef",
    "parse_identifier",
]
