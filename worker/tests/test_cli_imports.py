"""Regression test: every CLI module must import cleanly.

This catches stale-venv bugs where ``pyproject.toml`` has been updated
with a new dependency (e.g. ``tomlkit`` for ``cli.migrate``) but the
local venv was not refreshed via ``make install-dev``. Without this
test, the failure surfaces as a single ``ModuleNotFoundError`` during
collection of whichever test happens to import the affected CLI module
first — easy to misdiagnose as a pre-existing failure unrelated to the
current change.

If this test fails with ``ModuleNotFoundError``, run::

    make install-dev

from the repo root to refresh the venv.
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest

import worker.cli as cli_pkg


def _cli_module_names() -> list[str]:
    return sorted(
        f"{cli_pkg.__name__}.{info.name}"
        for info in pkgutil.iter_modules(cli_pkg.__path__)
    )


@pytest.mark.parametrize("module_name", _cli_module_names())
def test_cli_module_imports(module_name: str) -> None:
    """Importing the CLI module must not raise.

    A ``ModuleNotFoundError`` here almost always means the venv is
    stale relative to ``worker/pyproject.toml`` — re-run
    ``make install-dev``.
    """
    importlib.import_module(module_name)
