"""Tests for the Homebrew source adapter: discovery, state, diffing, events."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from worker.sources.homebrew import (
    DEFAULT_POLL_INTERVAL,
    HomebrewSource,
    _build_cask_snapshot,
    _build_event,
    _build_formula_snapshot,
    _diff_snapshots,
    _find_brew,
    _load_state,
    _save_state,
)


# ── helpers ────────────────────────────────────────────────────────


SAMPLE_FORMULA = {
    "name": "ripgrep",
    "desc": "Search tool like grep and The Silver Searcher",
    "homepage": "https://github.com/BurntSushi/ripgrep",
    "tap": "homebrew/core",
    "versions": {"stable": "14.1.0"},
    "installed": [
        {"version": "14.1.0", "installed_as_dependency": False}
    ],
}

SAMPLE_CASK = {
    "token": "docker",
    "name": ["Docker Desktop"],
    "desc": "App to build and share containerised applications and microservices",
    "homepage": "https://www.docker.com/products/docker-desktop",
    "tap": "homebrew/cask",
    "version": "4.30.0",
    "artifacts": [{"app": ["Docker.app"]}],
}

SAMPLE_BREW_JSON = {
    "formulae": [SAMPLE_FORMULA],
    "casks": [SAMPLE_CASK],
}


async def _collect_events(
    queue: asyncio.Queue[dict[str, Any]], timeout: float = 2.0
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    try:
        while True:
            ev = await asyncio.wait_for(queue.get(), timeout=timeout)
            events.append(ev)
    except (asyncio.TimeoutError, TimeoutError):
        pass
    return events


# ── _find_brew ─────────────────────────────────────────────────────


class TestFindBrew:
    def test_returns_none_when_not_installed(self) -> None:
        with patch("worker.sources.homebrew.Path.is_file", return_value=False), \
             patch("worker.sources.homebrew.shutil.which", return_value=None):
            assert _find_brew() is None

    def test_finds_apple_silicon_path(self) -> None:
        def is_file_side_effect(self: Any = None) -> bool:
            return str(self) == "/opt/homebrew/bin/brew"

        with patch.object(Path, "is_file", is_file_side_effect):
            result = _find_brew()
            assert result == "/opt/homebrew/bin/brew"


# ── _load_state / _save_state ─────────────────────────────────────


class TestStatePersistence:
    def test_load_missing_file(self, tmp_path: Path) -> None:
        assert _load_state(tmp_path / "missing.json") == {}

    def test_save_and_load(self, tmp_path: Path) -> None:
        p = tmp_path / "state.json"
        state = {"brew://formula/rg": {"name": "ripgrep", "version": "14.1.0", "kind": "formula"}}
        _save_state(p, state)
        loaded = _load_state(p)
        assert loaded == state

    def test_load_corrupt_json(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text("not valid json{{{")
        assert _load_state(p) == {}

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        p = tmp_path / "sub" / "dir" / "state.json"
        _save_state(p, {"k": {"v": 1}})
        assert p.exists()

    def test_save_sets_restrictive_permissions(self, tmp_path: Path) -> None:
        p = tmp_path / "state.json"
        _save_state(p, {"k": {"v": 1}})
        assert p.stat().st_mode & 0o777 == 0o600


# ── snapshot builders ──────────────────────────────────────────────


class TestBuildFormulaSnapshot:
    def test_basic_formula(self) -> None:
        sid, snap = _build_formula_snapshot(SAMPLE_FORMULA, None)
        assert sid == "brew://formula/ripgrep"
        assert snap["name"] == "ripgrep"
        assert snap["kind"] == "formula"
        assert snap["version"] == "14.1.0"
        assert snap["description"] == "Search tool like grep and The Silver Searcher"
        assert snap["homepage"] == "https://github.com/BurntSushi/ripgrep"
        assert snap["tap"] == "homebrew/core"

    def test_formula_without_installed(self) -> None:
        f = {**SAMPLE_FORMULA, "installed": []}
        sid, snap = _build_formula_snapshot(f, None)
        assert snap["version"] == "14.1.0"  # falls back to versions.stable


class TestBuildCaskSnapshot:
    def test_basic_cask(self) -> None:
        sid, snap = _build_cask_snapshot(SAMPLE_CASK)
        assert sid == "brew://cask/docker"
        assert snap["name"] == "Docker Desktop"
        assert snap["kind"] == "cask"
        assert snap["version"] == "4.30.0"
        assert snap["installed_via_brew"] is True


# ── _diff_snapshots ────────────────────────────────────────────────


class TestDiffSnapshots:
    def test_new_package(self) -> None:
        prev: dict[str, dict[str, Any]] = {}
        curr = {"brew://formula/rg": {"name": "ripgrep", "version": "14.1.0", "kind": "formula"}}
        changes = _diff_snapshots(prev, curr)
        assert len(changes) == 1
        assert changes[0][2] == "created"

    def test_removed_package(self) -> None:
        prev = {"brew://formula/rg": {"name": "ripgrep", "version": "14.1.0", "kind": "formula"}}
        curr: dict[str, dict[str, Any]] = {}
        changes = _diff_snapshots(prev, curr)
        assert len(changes) == 1
        assert changes[0][2] == "deleted"

    def test_version_change(self) -> None:
        prev = {"brew://formula/rg": {"name": "ripgrep", "version": "14.0.0", "kind": "formula"}}
        curr = {"brew://formula/rg": {"name": "ripgrep", "version": "14.1.0", "kind": "formula"}}
        changes = _diff_snapshots(prev, curr)
        assert len(changes) == 1
        assert changes[0][2] == "modified"

    def test_no_changes(self) -> None:
        state = {"brew://formula/rg": {"name": "ripgrep", "version": "14.1.0", "kind": "formula"}}
        changes = _diff_snapshots(state, dict(state))
        assert changes == []


# ── _build_event ───────────────────────────────────────────────────


class TestBuildEvent:
    def test_event_structure(self) -> None:
        snap = {
            "name": "ripgrep",
            "kind": "formula",
            "version": "14.1.0",
            "description": "A search tool",
            "homepage": "https://example.com",
            "tap": "homebrew/core",
            "binaries": ["rg"],
        }
        event = _build_event("brew://formula/ripgrep", snap, "created")
        assert event["source_type"] == "homebrew"
        assert event["source_id"] == "brew://formula/ripgrep"
        assert event["operation"] == "created"
        assert event["mime_type"] == "text/plain"
        assert "ripgrep" in event["text"]
        assert "A search tool" in event["text"]
        assert event["meta"]["package_name"] == "ripgrep"
        assert event["meta"]["package_kind"] == "formula"
        assert event["meta"]["binaries"] == ["rg"]
        assert "id" in event
        assert "enqueued_at" in event

    def test_cask_event_with_bundle_id(self) -> None:
        snap = {
            "name": "Docker Desktop",
            "kind": "cask",
            "version": "4.30.0",
            "description": "Docker app",
            "homepage": "https://docker.com",
            "tap": "homebrew/cask",
            "bundle_id": "com.docker.docker",
            "installed_via_brew": True,
        }
        event = _build_event("brew://cask/docker", snap, "created")
        assert event["meta"]["bundle_id"] == "com.docker.docker"
        assert event["meta"]["installed_via_brew"] is True


# ── HomebrewSource configure ───────────────────────────────────────


class TestHomebrewSourceConfigure:
    def test_defaults(self) -> None:
        s = HomebrewSource()
        s.configure({})
        assert s._poll_interval == DEFAULT_POLL_INTERVAL
        assert s._include_system is False

    def test_custom_poll_interval(self) -> None:
        s = HomebrewSource()
        s.configure({"poll_interval_seconds": 3600})
        assert s._poll_interval == 3600

    def test_include_system(self) -> None:
        s = HomebrewSource()
        s.configure({"include_system": True})
        assert s._include_system is True

    def test_custom_state_path(self, tmp_path: Path) -> None:
        s = HomebrewSource()
        s.configure({"state_path": str(tmp_path / "custom.json")})
        assert s._state_path == tmp_path / "custom.json"

    def test_name(self) -> None:
        assert HomebrewSource().name() == "homebrew"


# ── HomebrewSource start (integration) ─────────────────────────────


@pytest.mark.asyncio
async def test_start_no_brew_installed() -> None:
    """When brew is not found, source should run idle without crashing."""
    s = HomebrewSource()
    s.configure({})

    with patch("worker.sources.homebrew._find_brew", return_value=None):
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        task = asyncio.create_task(s.start(q))
        events = await _collect_events(q, timeout=1.0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert events == []


@pytest.mark.asyncio
async def test_initial_scan_emits_events(tmp_path: Path) -> None:
    """Initial scan should emit created events for all installed packages."""
    s = HomebrewSource()
    s.configure({"state_path": str(tmp_path / "state.json")})

    with patch("worker.sources.homebrew._find_brew", return_value="/usr/local/bin/brew"), \
         patch("worker.sources.homebrew._collect_installed", return_value=SAMPLE_BREW_JSON), \
         patch("worker.sources.homebrew._brew_prefix", return_value="/usr/local"):
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        task = asyncio.create_task(s.start(q))
        events = await _collect_events(q)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert len(events) == 2  # 1 formula + 1 cask
    source_ids = {e["source_id"] for e in events}
    assert "brew://formula/ripgrep" in source_ids
    assert "brew://cask/docker" in source_ids
    for e in events:
        assert e["operation"] == "created"
        assert e["source_type"] == "homebrew"


@pytest.mark.asyncio
async def test_incremental_scan_detects_changes(tmp_path: Path) -> None:
    """Second scan should detect version changes and new packages."""
    state_path = tmp_path / "state.json"

    # Seed state with an old version of ripgrep
    prev = {"brew://formula/ripgrep": {"name": "ripgrep", "version": "13.0.0", "kind": "formula"}}
    _save_state(state_path, prev)

    # New scan has ripgrep 14.1.0 (version bump) + docker (new cask)
    s = HomebrewSource()
    s.configure({"state_path": str(state_path)})

    with patch("worker.sources.homebrew._find_brew", return_value="/usr/local/bin/brew"), \
         patch("worker.sources.homebrew._collect_installed", return_value=SAMPLE_BREW_JSON), \
         patch("worker.sources.homebrew._brew_prefix", return_value="/usr/local"):
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        task = asyncio.create_task(s.start(q))
        events = await _collect_events(q)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    ops = {e["source_id"]: e["operation"] for e in events}
    assert ops["brew://formula/ripgrep"] == "modified"
    assert ops["brew://cask/docker"] == "created"


@pytest.mark.asyncio
async def test_handles_brew_timeout(tmp_path: Path) -> None:
    """Source should handle brew command timeout gracefully."""
    import subprocess

    s = HomebrewSource()
    s.configure({"state_path": str(tmp_path / "state.json")})

    def timeout_collect(*args: Any, **kwargs: Any) -> None:
        raise subprocess.TimeoutExpired(cmd="brew", timeout=120)

    with patch("worker.sources.homebrew._find_brew", return_value="/usr/local/bin/brew"), \
         patch("worker.sources.homebrew._collect_installed", side_effect=timeout_collect):
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        task = asyncio.create_task(s.start(q))
        events = await _collect_events(q, timeout=1.0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert events == []  # No crash, no events


@pytest.mark.asyncio
async def test_uninstall_detection(tmp_path: Path) -> None:
    """Packages removed between scans should emit deleted events."""
    state_path = tmp_path / "state.json"

    # Seed state with ripgrep installed
    prev = {
        "brew://formula/ripgrep": {"name": "ripgrep", "version": "14.1.0", "kind": "formula"},
    }
    _save_state(state_path, prev)

    # New scan has no formulae (ripgrep was uninstalled) but has docker cask
    brew_json = {"formulae": [], "casks": [SAMPLE_CASK]}

    s = HomebrewSource()
    s.configure({"state_path": str(state_path)})

    with patch("worker.sources.homebrew._find_brew", return_value="/usr/local/bin/brew"), \
         patch("worker.sources.homebrew._collect_installed", return_value=brew_json), \
         patch("worker.sources.homebrew._brew_prefix", return_value="/usr/local"):
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        task = asyncio.create_task(s.start(q))
        events = await _collect_events(q)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    ops = {e["source_id"]: e["operation"] for e in events}
    assert ops["brew://formula/ripgrep"] == "deleted"
    assert ops["brew://cask/docker"] == "created"
