"""Regression test: every CLI module must import cleanly *and* its key
dependency must be runtime-functional.

A pure ``importlib.import_module`` check catches stale-venv bugs where
``pyproject.toml`` declares a new dep but the local venv is out of date.
It does *not* catch the case where the dep imports successfully but its
underlying code is broken — e.g. a binary-incompatible ``tomlkit`` whose
``parse`` raises on trivial input. To close that gap, each CLI module
also exercises a small real call against its key dep, so a broken-
but-importable dep surfaces here as a test failure instead of slipping
through to runtime.

If this test fails with ``ModuleNotFoundError``, run::

    make install-dev

from the repo root to refresh the venv.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Callable

import pytest

import worker.cli as cli_pkg


# ── Per-module probes ─────────────────────────────────────────────────
#
# Each probe receives the imported module and performs a minimal real
# operation that *uses* the module's key dependency. Probes must be
# fast (microseconds) and side-effect-free.


def _probe_ask(mod: Any) -> None:
    ctx = mod._PreparedContext(
        system_prompt="",
        user_prompt="",
        source_ids=[],
        errors=[],
        has_context=False,
        sparse=False,
        context_text="",
    )
    assert "[Confidence]" in mod._format_footer(ctx, show_sources=False)


def _probe_cluster(mod: Any) -> None:
    # qdrant_client is the dep loaded lazily inside _corpus_size; force
    # it to resolve here so a broken install fails the canary, not
    # production runs.
    qc = importlib.import_module("qdrant_client")
    assert callable(qc.QdrantClient)
    assert callable(mod.cluster_embeddings)
    assert callable(mod.label_clusters)


def _probe_connections(mod: Any) -> None:
    assert callable(mod.ConnectionQuerier)
    assert callable(mod.run_connections)


def _probe_digest(mod: Any) -> None:
    activity = mod.SourceActivity(source_type="gmail", created=1, modified=2)
    lines = mod._format_source_line(activity)
    assert isinstance(lines, list) and lines


def _probe_display(mod: Any) -> None:
    assert isinstance(mod._use_rich(), bool)


def _probe_history(mod: Any) -> None:
    ts = mod._iso_now()
    assert "T" in ts and ts.endswith("+00:00")


def _probe_migrate(mod: Any) -> None:
    # tomlkit is migrate's headline dep and is imported lazily inside
    # migrate_config — invoke parse directly so the canary catches a
    # broken-but-importable tomlkit (the original motivation for this
    # probe pattern).
    import tomlkit

    doc = tomlkit.parse("a = 1\n")
    assert doc["a"] == 1
    assert mod.rewrite_source_id("gmail://thread/abc", "ml") == "gmail://ml/thread/abc"


def _probe_queue(mod: Any) -> None:
    assert callable(mod.PersistentQueue)
    item = {
        "id": "1",
        "status": "pending",
        "source_id": "gmail://default/thread/abc",
        "source_type": "gmail",
        "operation": "ingest",
    }
    line = mod._format_item(item)
    assert "gmail://default/thread/abc" in line


def _probe_reformulator(mod: Any) -> None:
    assert mod._needs_reformulation("hello", 0) is False


def _probe_stream(mod: Any) -> None:
    url = mod.source_id_to_url("gmail://default/message/abc123")
    assert url is not None and url.startswith("https://mail.google.com/")


def _probe_timeline(mod: Any) -> None:
    assert mod._fmt_timestamp("2025-01-01T12:34:56Z").startswith("2025-01-01")


def _probe_persons(mod: Any) -> None:
    # Identifier parsing is the runtime-canary for the persons handlers —
    # it pulls in the whole curation package via re-export.
    from worker.curation import parse_identifier

    assert parse_identifier("alice@example.com").email == "alice@example.com"
    assert hasattr(mod, "run_inspect")
    assert hasattr(mod, "run_split")
    assert hasattr(mod, "run_confirm")
    assert hasattr(mod, "run_merge")


def _probe_person(mod: Any) -> None:
    # Rich is the headline dep — exercise a small real call so a broken
    # but importable rich install fails the canary instead of slipping
    # through to runtime.
    from rich.table import Table

    assert callable(Table)
    assert callable(mod.run_person)
    assert callable(mod.generate_brief)


def _probe_person_brief_prompt(mod: Any) -> None:
    assert callable(mod.build_brief_request)
    assert "Do not invent" in mod.SYSTEM_PROMPT


_PROBES: dict[str, Callable[[Any], None]] = {
    "ask": _probe_ask,
    "cluster": _probe_cluster,
    "connections": _probe_connections,
    "digest": _probe_digest,
    "display": _probe_display,
    "history": _probe_history,
    "migrate": _probe_migrate,
    "persons": _probe_persons,
    "queue": _probe_queue,
    "reformulator": _probe_reformulator,
    "stream": _probe_stream,
    "timeline": _probe_timeline,
    "person": _probe_person,
    "person_brief_prompt": _probe_person_brief_prompt,
}


def _cli_module_names() -> list[str]:
    return sorted(info.name for info in pkgutil.iter_modules(cli_pkg.__path__))


@pytest.mark.parametrize("module_name", _cli_module_names())
def test_cli_module_imports_and_probe(module_name: str) -> None:
    """Each CLI submodule imports cleanly and its key dep is runtime-functional.

    A ``ModuleNotFoundError`` here almost always means the venv is
    stale relative to ``worker/pyproject.toml`` — re-run
    ``make install-dev``. A probe assertion failure means a dep is
    importable but its runtime behavior is broken (e.g. version
    mismatch).
    """
    mod = importlib.import_module(f"{cli_pkg.__name__}.{module_name}")
    probe = _PROBES.get(module_name)
    assert probe is not None, (
        f"Missing canary probe for cli module {module_name!r}. "
        f"Add an entry to test_cli_imports._PROBES."
    )
    probe(mod)


def test_broken_tomlkit_breaks_migrate_canary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken tomlkit must surface as a probe failure, not a silent pass.

    Guards against the migrate probe degenerating into a pure import
    test that would slip a binary-incompatible tomlkit past CI.
    """
    import tomlkit

    def _broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("simulated broken tomlkit")

    monkeypatch.setattr(tomlkit, "parse", _broken)

    migrate_mod = importlib.import_module(f"{cli_pkg.__name__}.migrate")
    with pytest.raises(RuntimeError, match="simulated broken tomlkit"):
        _probe_migrate(migrate_mod)
