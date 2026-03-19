"""Daily digest query: summarize recent activity across all indexed sources.

Returns aggregate counts and highlights per source type for a configurable
time window, plus cross-source connection and topic discovery counts.

Usage::

    querier = DigestQuerier(neo4j_cfg)
    result = querier.query(since="24h")
    for source in result.sources:
        print(source.source_type, source.created, source.modified)
    querier.close()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from neo4j import GraphDatabase, Driver

from worker.config import Neo4jConfig
from worker.query._time import parse_relative_time

logger = logging.getLogger(__name__)

# Limit on highlight titles per source type.
_HIGHLIGHTS_LIMIT = 5


@dataclass
class SourceActivity:
    """Activity summary for a single source type."""

    source_type: str         # obsidian, omnifocus, gmail, file, repositories, apps
    created: int = 0         # count of new documents
    modified: int = 0        # count of modified documents
    deleted: int = 0         # count of deleted documents (not yet tracked — always 0)
    highlights: list[str] = field(default_factory=list)  # top titles/names


@dataclass
class DigestResult:
    """Structured result from a digest query."""

    since: str                                  # ISO 8601
    until: str                                  # ISO 8601
    sources: list[SourceActivity] = field(default_factory=list)
    new_connections: int = 0                    # entities newly linked across sources
    new_topics: int = 0                         # topics discovered since last digest
    summary: str | None = None                  # LLM-generated summary (optional)
    error: str | None = None


# ---------------------------------------------------------------------------
# Per-source Cypher queries — each returns:
#   created_count, modified_count, completed_count (tasks only), highlights
# ---------------------------------------------------------------------------

_CYPHER_OBSIDIAN = """\
MATCH (n:File)
WHERE (n.updated >= $since AND n.updated <= $until)
   OR (n.created >= $since AND n.created <= $until)
   OR (n.indexed_at >= $since AND n.indexed_at <= $until)
WITH n,
     CASE
       WHEN n.created IS NOT NULL AND n.updated IS NOT NULL
            AND n.created = n.updated THEN 'created'
       ELSE 'modified'
     END AS event_type
RETURN
  count(CASE WHEN event_type = 'created' THEN 1 END) AS created_count,
  count(CASE WHEN event_type = 'modified' THEN 1 END) AS modified_count,
  0 AS completed_count,
  collect(DISTINCT coalesce(n.title, n.name, n.source_id))[..$limit] AS highlights
"""

_CYPHER_OMNIFOCUS = """\
MATCH (n:Task)
WHERE (n.modification_date >= $since AND n.modification_date <= $until)
   OR (n.completion_date >= $since AND n.completion_date <= $until)
   OR (n.creation_date >= $since AND n.creation_date <= $until)
WITH n,
     CASE
       WHEN n.status = 'completed'
            AND n.completion_date IS NOT NULL
            AND n.completion_date >= $since
            AND n.completion_date <= $until THEN 'completed'
       WHEN n.creation_date IS NOT NULL
            AND n.modification_date IS NOT NULL
            AND n.creation_date = n.modification_date THEN 'created'
       ELSE 'modified'
     END AS event_type
RETURN
  count(CASE WHEN event_type = 'created' THEN 1 END) AS created_count,
  count(CASE WHEN event_type = 'modified' THEN 1 END) AS modified_count,
  count(CASE WHEN event_type = 'completed' THEN 1 END) AS completed_count,
  collect(DISTINCT coalesce(n.name, n.source_id))[..$limit] AS highlights
"""

_CYPHER_GMAIL = """\
MATCH (n:Email)
WHERE n.date >= $since AND n.date <= $until
RETURN
  count(n) AS created_count,
  0 AS modified_count,
  0 AS completed_count,
  collect(DISTINCT coalesce(n.subject, n.source_id))[..$limit] AS highlights
"""

_CYPHER_REPOSITORIES = """\
MATCH (n:Commit)
WHERE n.date >= $since AND n.date <= $until
WITH n,
     CASE
       WHEN n.message CONTAINS '\\n' THEN split(n.message, '\\n')[0]
       ELSE n.message
     END AS title
RETURN
  count(n) AS created_count,
  0 AS modified_count,
  0 AS completed_count,
  collect(DISTINCT coalesce(title, n.source_id))[..$limit] AS highlights
"""

_CYPHER_APPS = """\
MATCH (n:Application)
WHERE n.indexed_at >= $since AND n.indexed_at <= $until
RETURN
  0 AS created_count,
  count(n) AS modified_count,
  0 AS completed_count,
  collect(DISTINCT coalesce(n.name, n.source_id))[..$limit] AS highlights
"""

# Map source_type → (cypher, display_name)
_SOURCE_QUERIES: list[tuple[str, str]] = [
    ("obsidian", _CYPHER_OBSIDIAN),
    ("omnifocus", _CYPHER_OMNIFOCUS),
    ("gmail", _CYPHER_GMAIL),
    ("repositories", _CYPHER_REPOSITORIES),
    ("apps", _CYPHER_APPS),
]

_CYPHER_NEW_CONNECTIONS = """\
MATCH (a)-[r]->(b)
WHERE r.created_at >= $since AND r.created_at <= $until
  AND any(la IN labels(a) WHERE la IN ['File','Task','Email','Commit','Application'])
  AND any(lb IN labels(b) WHERE lb IN ['File','Task','Email','Commit','Application'])
  AND labels(a) <> labels(b)
RETURN count(r) AS new_connections
"""

_CYPHER_NEW_TOPICS = """\
MATCH (t:Topic)
WHERE t.created_at >= $since AND t.created_at <= $until
RETURN count(t) AS new_topics
"""


class DigestQuerier:
    """Queries Neo4j for a digest of activity within a time window.

    Usage::

        with DigestQuerier(neo4j_cfg) as q:
            result = q.query(since="24h")
    """

    def __init__(self, neo4j_cfg: Neo4jConfig | None = None) -> None:
        neo4j_cfg = neo4j_cfg or Neo4jConfig()
        self._driver: Driver = GraphDatabase.driver(
            neo4j_cfg.uri,
            auth=(neo4j_cfg.user, neo4j_cfg.password),
        )

    def query(
        self,
        *,
        since: str = "24h",
        until: str = "now",
    ) -> DigestResult:
        """Return an activity digest for the specified time window.

        Parameters
        ----------
        since:
            Start of the time window (relative or ISO 8601). Default: "24h".
        until:
            End of the time window (relative or ISO 8601). Default: "now".
        """
        try:
            since_dt = parse_relative_time(since)
            until_dt = parse_relative_time(until)
        except ValueError as exc:
            return DigestResult(since="", until="", error=str(exc))

        since_iso = since_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        until_iso = until_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        result = DigestResult(since=since_iso, until=until_iso)

        try:
            result.sources = self._query_sources(since_iso, until_iso)
            result.new_connections = self._query_new_connections(since_iso, until_iso)
            result.new_topics = self._query_new_topics(since_iso, until_iso)
        except Exception as exc:
            logger.exception("Digest query failed")
            result.error = f"Neo4j error: {exc}"

        return result

    def _query_sources(self, since_iso: str, until_iso: str) -> list[SourceActivity]:
        """Run per-source-type aggregate queries."""
        sources: list[SourceActivity] = []
        with self._driver.session() as session:
            for source_type, cypher in _SOURCE_QUERIES:
                try:
                    rows = session.execute_read(
                        lambda tx, q=cypher: tx.run(
                            q,
                            since=since_iso,
                            until=until_iso,
                            limit=_HIGHLIGHTS_LIMIT,
                        ).data()
                    )
                    if rows:
                        row = rows[0]
                        created = int(row.get("created_count") or 0)
                        modified = int(row.get("modified_count") or 0)
                        completed = int(row.get("completed_count") or 0)
                        highlights = [str(h) for h in (row.get("highlights") or [])]
                        # For OmniFocus, completed tasks contribute to "modified" display
                        # but we track them separately in the SourceActivity.
                        # Encode completed in "modified" field for non-omnifocus sources.
                        if source_type == "omnifocus":
                            activity = SourceActivity(
                                source_type=source_type,
                                created=created,
                                modified=modified + completed,
                                highlights=highlights,
                            )
                            # Stash completed count for display purposes
                            activity._completed = completed  # type: ignore[attr-defined]
                        else:
                            activity = SourceActivity(
                                source_type=source_type,
                                created=created,
                                modified=modified,
                                highlights=highlights,
                            )
                        total = created + modified + (completed if source_type == "omnifocus" else 0)
                        if total > 0:
                            sources.append(activity)
                except Exception:
                    logger.debug("Digest: %s query failed", source_type, exc_info=True)
        return sources

    def _query_new_connections(self, since_iso: str, until_iso: str) -> int:
        """Count new cross-source relationships created in the window."""
        try:
            with self._driver.session() as session:
                rows = session.execute_read(
                    lambda tx: tx.run(
                        _CYPHER_NEW_CONNECTIONS,
                        since=since_iso,
                        until=until_iso,
                    ).data()
                )
                if rows:
                    return int(rows[0].get("new_connections") or 0)
        except Exception:
            logger.debug("Digest: new_connections query failed", exc_info=True)
        return 0

    def _query_new_topics(self, since_iso: str, until_iso: str) -> int:
        """Count new Topic nodes created in the window."""
        try:
            with self._driver.session() as session:
                rows = session.execute_read(
                    lambda tx: tx.run(
                        _CYPHER_NEW_TOPICS,
                        since=since_iso,
                        until=until_iso,
                    ).data()
                )
                if rows:
                    return int(rows[0].get("new_topics") or 0)
        except Exception:
            logger.debug("Digest: new_topics query failed", exc_info=True)
        return 0

    def close(self) -> None:
        """Release Neo4j connection."""
        try:
            self._driver.close()
        except Exception:
            pass

    def __enter__(self) -> DigestQuerier:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
