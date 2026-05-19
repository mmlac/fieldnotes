"""Static regression guard: no production Cypher may use the deprecated id() function.

Neo4j 5.x deprecates ``id()`` (notification code ``01N01``).  This test scans
the production query strings in all affected modules and asserts they don't
contain ``id(`` followed by a single lowercase letter (the node-variable form
that triggers the deprecation).

No Neo4j connection needed — this is a pure source inspection.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

# Production files to scan (relative to the worker package root)
_WORKER_ROOT = Path(__file__).parent.parent / "worker"

_FILES = [
    _WORKER_ROOT / "pipeline" / "writer.py",
    _WORKER_ROOT / "query" / "itinerary.py",
    _WORKER_ROOT / "query" / "itinerary_brief.py",
    _WORKER_ROOT / "query" / "person.py",
    _WORKER_ROOT / "query" / "person_brief.py",
    _WORKER_ROOT / "curation" / "persons.py",
    _WORKER_ROOT / "cli" / "person.py",
]

# Matches id() with a single node-variable letter, e.g. id(a), id(p), id(e)
_ID_PATTERN = re.compile(r"\bid\([a-z]\)")


def _extract_string_literals(source: str) -> list[str]:
    """Return all string literal values found in *source* via AST parsing."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    literals: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            literals.append(node.value)
        elif isinstance(node, (ast.JoinedStr,)):
            # f-strings: skip (no Cypher embedded there in practice)
            pass
    return literals


def test_no_deprecated_id_calls_in_cypher_strings() -> None:
    """All Cypher strings in production modules must be free of id(<var>)."""
    violations: list[str] = []
    for path in _FILES:
        source = path.read_text(encoding="utf-8")
        for literal in _extract_string_literals(source):
            if _ID_PATTERN.search(literal):
                violations.append(
                    f"{path.relative_to(_WORKER_ROOT.parent)}: "
                    f"contains deprecated id() call: {literal[:120]!r}"
                )
    assert not violations, (
        "Found deprecated Neo4j id() calls in Cypher strings:\n"
        + "\n".join(violations)
    )
