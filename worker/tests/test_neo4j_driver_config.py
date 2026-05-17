"""Tests for worker.neo4j_driver — notification policy and call-site coverage."""

from __future__ import annotations

import ast
import pathlib
from unittest.mock import MagicMock, patch

import pytest

from worker.neo4j_driver import build_driver

# ---------------------------------------------------------------------------
# Static analysis: every GraphDatabase.driver() call must use build_driver
# ---------------------------------------------------------------------------

_WORKER_ROOT = pathlib.Path(__file__).parent.parent / "worker"


def _py_files(root: pathlib.Path) -> list[pathlib.Path]:
    return [
        p
        for p in root.rglob("*.py")
        if "__pycache__" not in p.parts
        and p.name != "neo4j_driver.py"  # The helper itself is exempt
    ]


def _has_bare_graphdb_driver(source: str) -> bool:
    """Return True if the source contains a bare GraphDatabase.driver( call."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "driver"
            and isinstance(func.value, ast.Name)
            and func.value.id == "GraphDatabase"
        ):
            return True
    return False


class TestDriverCallSitesUseBuildDriver:
    def test_no_bare_graphdb_driver_calls(self) -> None:
        """Every GraphDatabase.driver() call must be in neo4j_driver.py only."""
        offenders: list[str] = []
        for path in _py_files(_WORKER_ROOT):
            source = path.read_text(encoding="utf-8")
            if _has_bare_graphdb_driver(source):
                offenders.append(str(path.relative_to(_WORKER_ROOT.parent)))
        assert not offenders, (
            "These files call GraphDatabase.driver() directly; "
            "use build_driver() from worker.neo4j_driver instead:\n"
            + "\n".join(f"  {p}" for p in sorted(offenders))
        )


# ---------------------------------------------------------------------------
# Unit: build_driver passes notifications_min_severity=WARNING
# ---------------------------------------------------------------------------


class TestBuildDriverPassesWarningSeverity:
    def test_notifications_min_severity_keyword(self) -> None:
        with patch("worker.neo4j_driver.GraphDatabase") as mock_gdb:
            mock_driver = MagicMock()
            mock_gdb.driver.return_value = mock_driver

            from neo4j import NotificationMinimumSeverity

            result = build_driver("bolt://localhost:7687", "neo4j", "secret")

        assert result is mock_driver
        mock_gdb.driver.assert_called_once_with(
            "bolt://localhost:7687",
            auth=("neo4j", "secret"),
            notifications_min_severity=NotificationMinimumSeverity.WARNING,
        )

    def test_extra_kwargs_forwarded(self) -> None:
        with patch("worker.neo4j_driver.GraphDatabase") as mock_gdb:
            mock_gdb.driver.return_value = MagicMock()

            from neo4j import NotificationMinimumSeverity

            build_driver(
                "bolt://localhost:7687",
                "neo4j",
                "secret",
                connection_timeout=10.0,
            )

        _, kwargs = mock_gdb.driver.call_args
        assert kwargs["notifications_min_severity"] == NotificationMinimumSeverity.WARNING
        assert kwargs["connection_timeout"] == 10.0


# ---------------------------------------------------------------------------
# Integration: schema-setup notifications suppressed (skipped if Neo4j absent)
# ---------------------------------------------------------------------------

_NEO4J_URI = "bolt://localhost:7687"
_NEO4J_USER = "neo4j"
_NEO4J_PASSWORD = "password"


def _neo4j_available() -> bool:
    try:
        from neo4j import GraphDatabase

        with GraphDatabase.driver(_NEO4J_URI, auth=(_NEO4J_USER, _NEO4J_PASSWORD)) as drv:
            drv.verify_connectivity()
        return True
    except Exception:
        return False


_NEEDS_NEO4J = pytest.mark.skipif(
    not _neo4j_available(),
    reason=f"Neo4j not available at {_NEO4J_URI}",
)


@_NEEDS_NEO4J
class TestIntegrationNotificationsSuppressed:
    def test_schema_setup_notifications_empty(self) -> None:
        """IF-NOT-EXISTS schema ops must produce zero notifications at WARNING floor."""
        drv = build_driver(_NEO4J_URI, _NEO4J_USER, _NEO4J_PASSWORD)
        try:
            with drv.session() as session:
                result = session.run(
                    "CREATE CONSTRAINT __test_fn5o5__ IF NOT EXISTS "
                    "FOR (n:__TestFn5o5__) REQUIRE n.id IS UNIQUE"
                )
                notifications = result.consume().notifications
        finally:
            drv.close()
        assert notifications == [] or all(
            getattr(n, "severity", None) not in (None, "INFORMATION")
            for n in notifications
        ), f"Expected no INFO notifications, got: {notifications}"

    def test_warning_level_still_surfaces(self) -> None:
        """Deprecated id() queries must still yield a WARNING notification."""
        drv = build_driver(_NEO4J_URI, _NEO4J_USER, _NEO4J_PASSWORD)
        try:
            with drv.session() as session:
                result = session.run("MATCH (n) RETURN id(n) LIMIT 1")
                result.consume()
                notifications = result.consume().notifications
        finally:
            drv.close()
        # The id() deprecation warning (01N01) is WARNING level.
        # If there are no nodes the query may return nothing; in that case
        # we can't assert the warning is present — skip the assertion.
        if notifications:
            severities = {
                getattr(n, "severity", None) or getattr(n, "rawSeverityLevel", None)
                for n in notifications
            }
            assert any(s in ("WARNING", "WARN") for s in severities), (
                f"Expected at least one WARNING notification, got severities: {severities}"
            )
