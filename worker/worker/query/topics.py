"""Topic queries: list, show, and gap-analysis against Neo4j.

Provides read-only Cypher queries for Topic nodes:
  - list_topics: all topics grouped by source with document counts
  - show_topic: single topic details with linked documents
  - topic_gaps: cluster-discovered topics missing from user taxonomy
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from neo4j import GraphDatabase, Driver

from worker.config import Neo4jConfig

logger = logging.getLogger(__name__)


@dataclass
class TopicSummary:
    """A topic with its source and linked document count."""

    name: str
    source: str
    description: str
    doc_count: int


@dataclass
class TopicDetail:
    """Full topic info with linked document list."""

    name: str
    source: str
    description: str
    documents: list[dict[str, str]] = field(default_factory=list)


@dataclass
class TopicGap:
    """A cluster-discovered topic with no user-taxonomy counterpart."""

    name: str
    description: str
    doc_count: int


class TopicQuerier:
    """Read-only queries against Topic nodes in Neo4j."""

    def __init__(self, neo4j_cfg: Neo4jConfig | None = None) -> None:
        neo4j_cfg = neo4j_cfg or Neo4jConfig()
        self._driver: Driver = GraphDatabase.driver(
            neo4j_cfg.uri,
            auth=(neo4j_cfg.user, neo4j_cfg.password),
        )

    def list_topics(self) -> list[TopicSummary]:
        """Return all Topic nodes with document counts."""
        with self._driver.session() as session:
            records = session.run(
                """
                MATCH (t:Topic)
                OPTIONAL MATCH (source_node)-[:TAGGED]->(t)
                RETURN t.name AS name,
                       t.source AS source,
                       COALESCE(t.description, '') AS description,
                       count(source_node) AS doc_count
                ORDER BY t.source, t.name
                """
            ).data()
        return [
            TopicSummary(
                name=r["name"],
                source=r["source"],
                description=r["description"],
                doc_count=r["doc_count"],
            )
            for r in records
        ]

    def show_topic(self, name: str) -> TopicDetail | None:
        """Return details for a single topic by name."""
        with self._driver.session() as session:
            records = session.run(
                """
                MATCH (t:Topic {name: $name})
                OPTIONAL MATCH (doc)-[:TAGGED]->(t)
                RETURN t.name AS name,
                       t.source AS source,
                       COALESCE(t.description, '') AS description,
                       labels(doc) AS doc_labels,
                       COALESCE(doc.source_id, '') AS source_id,
                       COALESCE(doc.name, doc.source_id, '') AS doc_name
                ORDER BY doc_name
                """,
                name=name,
            ).data()

        if not records:
            return None

        first = records[0]
        documents: list[dict[str, str]] = []
        for r in records:
            if r["source_id"]:
                doc_labels = [
                    lb for lb in (r["doc_labels"] or []) if lb != "Topic"
                ]
                documents.append(
                    {
                        "source_id": r["source_id"],
                        "name": r["doc_name"],
                        "type": doc_labels[0] if doc_labels else "Unknown",
                    }
                )

        return TopicDetail(
            name=first["name"],
            source=first["source"],
            description=first["description"],
            documents=documents,
        )

    def topic_gaps(self) -> list[TopicGap]:
        """Return cluster-derived topics not present in user taxonomy."""
        with self._driver.session() as session:
            records = session.run(
                """
                MATCH (t:Topic {source: 'cluster'})
                WHERE NOT EXISTS {
                    MATCH (:Topic {name: t.name, source: 'user'})
                }
                OPTIONAL MATCH (doc)-[:TAGGED]->(t)
                RETURN t.name AS name,
                       COALESCE(t.description, '') AS description,
                       count(doc) AS doc_count
                ORDER BY doc_count DESC, t.name
                """
            ).data()
        return [
            TopicGap(
                name=r["name"],
                description=r["description"],
                doc_count=r["doc_count"],
            )
            for r in records
        ]

    def close(self) -> None:
        """Release the Neo4j driver."""
        self._driver.close()

    def __enter__(self) -> TopicQuerier:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def format_topics_list(topics: list[TopicSummary], *, use_json: bool = False) -> str:
    """Format topic list for terminal output."""
    if use_json:
        return json.dumps(
            [
                {
                    "name": t.name,
                    "source": t.source,
                    "description": t.description,
                    "doc_count": t.doc_count,
                }
                for t in topics
            ],
            indent=2,
        )

    if not topics:
        return "No topics found."

    grouped: dict[str, list[TopicSummary]] = {}
    for t in topics:
        grouped.setdefault(t.source, []).append(t)

    lines: list[str] = []
    for source, items in sorted(grouped.items()):
        label = "User-defined" if source == "user" else "Cluster-discovered"
        lines.append(f"\033[1;36m{label} ({len(items)})\033[0m")
        for t in items:
            count_str = f"\033[33m{t.doc_count}\033[0m doc{'s' if t.doc_count != 1 else ''}"
            lines.append(f"  \033[1m{t.name}\033[0m  ({count_str})")
            if t.description:
                lines.append(f"    {t.description}")
        lines.append("")

    return "\n".join(lines).rstrip()


def format_topic_detail(detail: TopicDetail | None, *, use_json: bool = False) -> str:
    """Format single topic detail for terminal output."""
    if detail is None:
        return "Topic not found."

    if use_json:
        return json.dumps(
            {
                "name": detail.name,
                "source": detail.source,
                "description": detail.description,
                "documents": detail.documents,
            },
            indent=2,
        )

    source_label = "user-defined" if detail.source == "user" else "cluster-discovered"
    lines: list[str] = [
        f"\033[1m{detail.name}\033[0m  (\033[36m{source_label}\033[0m)",
    ]
    if detail.description:
        lines.append(f"  {detail.description}")
    lines.append("")

    if not detail.documents:
        lines.append("  No linked documents.")
    else:
        lines.append(
            f"\033[1;36mLinked documents ({len(detail.documents)})\033[0m"
        )
        for doc in detail.documents:
            lines.append(
                f"  \033[33m[{doc['type']}]\033[0m {doc['name']}"
            )

    return "\n".join(lines)


def format_topic_gaps(gaps: list[TopicGap], *, use_json: bool = False) -> str:
    """Format topic gaps for terminal output."""
    if use_json:
        return json.dumps(
            [
                {
                    "name": g.name,
                    "description": g.description,
                    "doc_count": g.doc_count,
                }
                for g in gaps
            ],
            indent=2,
        )

    if not gaps:
        return "No gaps found — all cluster topics have user-defined counterparts."

    lines: list[str] = [
        f"\033[1;36mGaps in your thinking ({len(gaps)} topics)\033[0m",
        "Cluster-discovered topics with no matching user-defined topic:\n",
    ]
    for g in gaps:
        count_str = f"\033[33m{g.doc_count}\033[0m doc{'s' if g.doc_count != 1 else ''}"
        lines.append(f"  \033[1m{g.name}\033[0m  ({count_str})")
        if g.description:
            lines.append(f"    {g.description}")

    return "\n".join(lines)
