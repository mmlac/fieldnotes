"""Timeline query: temporal search across all indexed sources.

Returns a chronological list of activity (created, modified, completed events)
across File, Task, Email, and Commit nodes, optionally filtered to a time
range and/or a specific source type.

Usage::

    querier = TimelineQuerier(neo4j_cfg, qdrant_cfg)
    result = querier.query(since="24h", until="now", limit=50)
    for entry in result.entries:
        print(entry.timestamp, entry.label, entry.title)
    querier.close()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from neo4j import GraphDatabase, Driver
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny

from worker.config import Neo4jConfig, QdrantConfig
from worker.query._time import parse_relative_time as _parse_relative_time

logger = logging.getLogger(__name__)

# Max timeline entries returned per query.
_MAX_ENTRIES = 500

# Valid source type filter values.
VALID_SOURCE_TYPES = frozenset(
    {"obsidian", "omnifocus", "gmail", "file", "repositories", "apps"}
)

# Labels that map to source_type values used in Qdrant payloads.
_LABEL_TO_SOURCE: dict[str, str] = {
    "File": "file",
    "Task": "omnifocus",
    "Email": "gmail",
    "Commit": "repositories",
    "Application": "apps",
}


@dataclass
class TimelineEntry:
    """A single activity event in the timeline."""

    source_type: str
    source_id: str
    label: str
    title: str
    timestamp: str  # ISO 8601
    event_type: str  # "created" | "modified" | "completed"
    snippet: str = ""


@dataclass
class TimelineResult:
    """Structured result from a timeline query."""

    entries: list[TimelineEntry] = field(default_factory=list)
    since: str = ""
    until: str = ""
    error: str | None = None


# ---------------------------------------------------------------------------
# Cypher query — one branch per node type, normalising timestamps to a
# common output schema.  Each branch is kept independent (no cross-label
# JOINs) so UNION ALL stays fast even with millions of nodes.
# ---------------------------------------------------------------------------

# Mapping of source_type filter value → set of labels to include.
_SOURCE_TYPE_LABELS: dict[str, list[str]] = {
    "file": ["File"],
    "obsidian": ["File"],
    "omnifocus": ["Task"],
    "gmail": ["Email"],
    "repositories": ["Commit"],
    "apps": ["Application"],
}

_CYPHER_TEMPLATE = """\
MATCH (n:File)
WHERE (n.updated >= $since AND n.updated <= $until)
   OR (n.created >= $since AND n.created <= $until)
   OR (n.indexed_at >= $since AND n.indexed_at <= $until)
WITH n,
     CASE
       WHEN n.updated IS NOT NULL THEN n.updated
       WHEN n.indexed_at IS NOT NULL THEN n.indexed_at
       ELSE n.created
     END AS ts,
     CASE
       WHEN n.created IS NOT NULL AND n.updated IS NOT NULL
            AND n.created = n.updated THEN 'created'
       ELSE 'modified'
     END AS event_type
WHERE ts >= $since AND ts <= $until
RETURN n.source_id AS source_id,
       'File' AS label,
       COALESCE(n.title, n.name, n.source_id) AS title,
       ts,
       event_type

UNION ALL

MATCH (n:Task)
WHERE (n.modification_date >= $since AND n.modification_date <= $until)
   OR (n.completion_date >= $since AND n.completion_date <= $until)
   OR (n.creation_date >= $since AND n.creation_date <= $until)
WITH n,
     CASE
       WHEN n.completion_date IS NOT NULL
            AND n.completion_date >= $since
            AND n.completion_date <= $until THEN n.completion_date
       WHEN n.modification_date IS NOT NULL
            AND n.modification_date >= $since
            AND n.modification_date <= $until THEN n.modification_date
       ELSE n.creation_date
     END AS ts,
     CASE
       WHEN n.status = 'completed' THEN 'completed'
       WHEN n.creation_date IS NOT NULL
            AND n.modification_date IS NOT NULL
            AND n.creation_date = n.modification_date THEN 'created'
       ELSE 'modified'
     END AS event_type
WHERE ts >= $since AND ts <= $until
RETURN n.source_id AS source_id,
       'Task' AS label,
       COALESCE(n.name, n.source_id) AS title,
       ts,
       event_type

UNION ALL

MATCH (n:Email)
WHERE n.date >= $since AND n.date <= $until
RETURN n.source_id AS source_id,
       'Email' AS label,
       COALESCE(n.subject, n.source_id) AS title,
       n.date AS ts,
       'created' AS event_type

UNION ALL

MATCH (n:Commit)
WHERE n.date >= $since AND n.date <= $until
RETURN n.source_id AS source_id,
       'Commit' AS label,
       COALESCE(n.message, n.source_id) AS title,
       n.date AS ts,
       'created' AS event_type

UNION ALL

MATCH (n:Application)
WHERE n.indexed_at >= $since AND n.indexed_at <= $until
RETURN n.source_id AS source_id,
       'Application' AS label,
       COALESCE(n.name, n.source_id) AS title,
       n.indexed_at AS ts,
       'modified' AS event_type

ORDER BY ts DESC
LIMIT $limit
"""

# Per-source-type queries (only the matching branches).
_CYPHER_FILE = """\
MATCH (n:File)
WHERE (n.updated >= $since AND n.updated <= $until)
   OR (n.created >= $since AND n.created <= $until)
   OR (n.indexed_at >= $since AND n.indexed_at <= $until)
WITH n,
     CASE
       WHEN n.updated IS NOT NULL THEN n.updated
       WHEN n.indexed_at IS NOT NULL THEN n.indexed_at
       ELSE n.created
     END AS ts,
     CASE
       WHEN n.created IS NOT NULL AND n.updated IS NOT NULL
            AND n.created = n.updated THEN 'created'
       ELSE 'modified'
     END AS event_type
WHERE ts >= $since AND ts <= $until
RETURN n.source_id AS source_id,
       'File' AS label,
       COALESCE(n.title, n.name, n.source_id) AS title,
       ts,
       event_type
ORDER BY ts DESC
LIMIT $limit
"""

_CYPHER_TASK = """\
MATCH (n:Task)
WHERE (n.modification_date >= $since AND n.modification_date <= $until)
   OR (n.completion_date >= $since AND n.completion_date <= $until)
   OR (n.creation_date >= $since AND n.creation_date <= $until)
WITH n,
     CASE
       WHEN n.completion_date IS NOT NULL
            AND n.completion_date >= $since
            AND n.completion_date <= $until THEN n.completion_date
       WHEN n.modification_date IS NOT NULL
            AND n.modification_date >= $since
            AND n.modification_date <= $until THEN n.modification_date
       ELSE n.creation_date
     END AS ts,
     CASE
       WHEN n.status = 'completed' THEN 'completed'
       WHEN n.creation_date IS NOT NULL
            AND n.modification_date IS NOT NULL
            AND n.creation_date = n.modification_date THEN 'created'
       ELSE 'modified'
     END AS event_type
WHERE ts >= $since AND ts <= $until
RETURN n.source_id AS source_id,
       'Task' AS label,
       COALESCE(n.name, n.source_id) AS title,
       ts,
       event_type
ORDER BY ts DESC
LIMIT $limit
"""

_CYPHER_EMAIL = """\
MATCH (n:Email)
WHERE n.date >= $since AND n.date <= $until
RETURN n.source_id AS source_id,
       'Email' AS label,
       COALESCE(n.subject, n.source_id) AS title,
       n.date AS ts,
       'created' AS event_type
ORDER BY ts DESC
LIMIT $limit
"""

_CYPHER_COMMIT = """\
MATCH (n:Commit)
WHERE n.date >= $since AND n.date <= $until
RETURN n.source_id AS source_id,
       'Commit' AS label,
       COALESCE(n.message, n.source_id) AS title,
       n.date AS ts,
       'created' AS event_type
ORDER BY ts DESC
LIMIT $limit
"""

_CYPHER_APP = """\
MATCH (n:Application)
WHERE n.indexed_at >= $since AND n.indexed_at <= $until
RETURN n.source_id AS source_id,
       'Application' AS label,
       COALESCE(n.name, n.source_id) AS title,
       n.indexed_at AS ts,
       'modified' AS event_type
ORDER BY ts DESC
LIMIT $limit
"""

_CYPHER_BY_SOURCE: dict[str, str] = {
    "file": _CYPHER_FILE,
    "obsidian": _CYPHER_FILE,
    "omnifocus": _CYPHER_TASK,
    "gmail": _CYPHER_EMAIL,
    "repositories": _CYPHER_COMMIT,
    "apps": _CYPHER_APP,
}

# Snippet max length from Qdrant chunk text.
_SNIPPET_MAX = 200


class TimelineQuerier:
    """Queries Neo4j + Qdrant for a chronological timeline of activity.

    Usage::

        with TimelineQuerier(neo4j_cfg, qdrant_cfg) as q:
            result = q.query(since="24h", limit=50)
    """

    def __init__(
        self,
        neo4j_cfg: Neo4jConfig | None = None,
        qdrant_cfg: QdrantConfig | None = None,
    ) -> None:
        neo4j_cfg = neo4j_cfg or Neo4jConfig()
        qdrant_cfg = qdrant_cfg or QdrantConfig()
        self._driver: Driver = GraphDatabase.driver(
            neo4j_cfg.uri,
            auth=(neo4j_cfg.user, neo4j_cfg.password),
        )
        self._qdrant = QdrantClient(host=qdrant_cfg.host, port=qdrant_cfg.port)
        self._collection = qdrant_cfg.collection

    def query(
        self,
        *,
        since: str = "24h",
        until: str = "now",
        source_type: str | None = None,
        limit: int = 50,
    ) -> TimelineResult:
        """Return a timeline of activity within the specified time window.

        Parameters
        ----------
        since:
            Start of the time window (relative or ISO 8601). Default: "24h".
        until:
            End of the time window (relative or ISO 8601). Default: "now".
        source_type:
            Optional filter to one source type (obsidian, omnifocus, gmail,
            file, repositories, apps).
        limit:
            Maximum number of entries to return (capped at _MAX_ENTRIES).
        """
        limit = max(1, min(limit, _MAX_ENTRIES))

        try:
            since_dt = _parse_relative_time(since)
            until_dt = _parse_relative_time(until)
        except ValueError as exc:
            return TimelineResult(error=str(exc))

        since_iso = since_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        until_iso = until_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        if source_type and source_type not in VALID_SOURCE_TYPES:
            return TimelineResult(
                error=f"Invalid source_type {source_type!r}. "
                f"Must be one of: {sorted(VALID_SOURCE_TYPES)}",
            )

        try:
            raw_rows = self._run_neo4j(
                since_iso, until_iso, source_type=source_type, limit=limit
            )
        except Exception as exc:
            logger.exception("Timeline Neo4j query failed")
            return TimelineResult(
                error=f"Neo4j error: {exc}", since=since_iso, until=until_iso
            )

        entries = self._build_entries(raw_rows, source_type)

        # Fetch snippets from Qdrant in a single batch.
        try:
            source_ids = [e.source_id for e in entries if e.source_id]
            snippets = self._fetch_snippets(source_ids)
            for entry in entries:
                entry.snippet = snippets.get(entry.source_id, "")
        except Exception:
            logger.debug("Timeline: Qdrant snippet fetch failed", exc_info=True)
            # Non-fatal — entries are still useful without snippets.

        return TimelineResult(entries=entries, since=since_iso, until=until_iso)

    def _run_neo4j(
        self,
        since_iso: str,
        until_iso: str,
        *,
        source_type: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Execute the appropriate Cypher query and return raw rows."""
        if source_type is not None and source_type in _CYPHER_BY_SOURCE:
            cypher = _CYPHER_BY_SOURCE[source_type]
        else:
            cypher = _CYPHER_TEMPLATE

        with self._driver.session() as session:
            return session.execute_read(
                lambda tx: tx.run(
                    cypher,
                    since=since_iso,
                    until=until_iso,
                    limit=limit,
                ).data()
            )

    def _build_entries(
        self, rows: list[dict[str, Any]], source_type: str | None
    ) -> list[TimelineEntry]:
        entries: list[TimelineEntry] = []
        for row in rows:
            label = str(row.get("label") or "")
            sid = str(row.get("source_id") or "")
            title = str(row.get("title") or sid or "")
            # Truncate commit messages to the first line.
            if label == "Commit" and "\n" in title:
                title = title.split("\n", 1)[0]
            ts_raw = row.get("ts")
            ts = str(ts_raw) if ts_raw else ""
            event_type = str(row.get("event_type") or "modified")
            st = source_type or _LABEL_TO_SOURCE.get(label, label.lower())
            entries.append(
                TimelineEntry(
                    source_type=st,
                    source_id=sid,
                    label=label,
                    title=title,
                    timestamp=ts,
                    event_type=event_type,
                )
            )
        return entries

    def _fetch_snippets(self, source_ids: list[str]) -> dict[str, str]:
        """Batch-fetch chunk_index=0 text from Qdrant for each source_id."""
        if not source_ids:
            return {}

        # Deduplicate while preserving order.
        unique_ids = list(dict.fromkeys(source_ids))

        try:
            results, _ = self._qdrant.scroll(
                collection_name=self._collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="source_id",
                            match=MatchAny(any=unique_ids),
                        ),
                        FieldCondition(
                            key="chunk_index",
                            match={"value": 0},
                        ),
                    ]
                ),
                limit=len(unique_ids),
                with_payload=True,
                with_vectors=False,
            )
        except Exception:
            logger.debug("Qdrant scroll failed", exc_info=True)
            return {}

        snippets: dict[str, str] = {}
        for point in results:
            payload = point.payload or {}
            sid = payload.get("source_id", "")
            text = payload.get("text", "")
            if sid and text:
                snippets[sid] = text[:_SNIPPET_MAX]

        return snippets

    def close(self) -> None:
        """Release Neo4j and Qdrant connections."""
        try:
            self._driver.close()
        except Exception:
            pass
        try:
            self._qdrant.close()
        except Exception:
            pass

    def __enter__(self) -> TimelineQuerier:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
