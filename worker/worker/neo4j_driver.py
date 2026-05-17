"""Centralised Neo4j driver factory with the worker's notification policy.

Every driver in the worker is built here so that the
``notifications_min_severity`` setting is applied consistently.  INFO-level
notifications (e.g. schema-setup no-ops from ``IF NOT EXISTS``) are suppressed
at the wire level; WARNING and above still surface.
"""

from __future__ import annotations

from neo4j import Driver, GraphDatabase, NotificationMinimumSeverity


def build_driver(uri: str, user: str, password: str, **kwargs: object) -> Driver:
    return GraphDatabase.driver(
        uri,
        auth=(user, password),
        notifications_min_severity=NotificationMinimumSeverity.WARNING,
        **kwargs,
    )
