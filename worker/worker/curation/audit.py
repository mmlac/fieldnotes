"""Append-only audit log for person curation actions.

Every ``inspect``-mutating action (``split``, ``confirm``, ``merge``)
appends one JSON line to ``<data_dir>/curation_audit.jsonl`` so reversals
are traceable.  The log is intentionally append-only and never rotated by
the worker itself — operators rotate it manually if it grows too large.

Inspect actions are not logged; they are read-only and would dominate the
log without adding traceability value.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AuditLog:
    """Append-only JSONL audit log."""

    path: Path

    @classmethod
    def from_data_dir(cls, data_dir: str | Path) -> AuditLog:
        """Build an audit log under ``<data_dir>/curation_audit.jsonl``.

        The directory is created if missing.  ``data_dir`` may use a leading
        ``~`` (expanded) or be a relative path (resolved against cwd).
        """
        base = Path(os.fspath(data_dir)).expanduser()
        base.mkdir(parents=True, exist_ok=True)
        return cls(path=base / "curation_audit.jsonl")

    def append(
        self,
        action: str,
        *,
        args: dict[str, Any],
        result: dict[str, Any],
        actor: str = "user",
    ) -> None:
        """Append one record.  Best-effort — log on failure, never raise."""
        record = {
            "ts": _now_iso(),
            "action": action,
            "actor": actor,
            "args": args,
            "result": result,
        }
        try:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")
        except OSError as exc:
            logger.warning("Curation audit log write failed: %s", exc)


def _now_iso() -> str:
    """Return current UTC timestamp in RFC3339 form (seconds resolution)."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
