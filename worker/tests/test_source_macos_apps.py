"""Tests for the macOS app scanner source adapter."""

from __future__ import annotations

import asyncio
import plistlib
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from worker.sources.macos_apps import (
    DEFAULT_POLL_INTERVAL,
    MacOSAppsSource,
    _build_event,
    _discover_apps,
    _hash_plist,
    _load_state,
    _parse_info_plist,
    _save_state,
)


# ── helpers ────────────────────────────────────────────────────────


def _create_app(
    base: Path,
    name: str,
    bundle_id: str = "com.example.test",
    version: str = "1.0",
    display_name: str | None = None,
    category: str = "",
) -> Path:
    """Create a minimal .app bundle with Info.plist."""
    app_path = base / f"{name}.app"
    contents = app_path / "Contents"
    contents.mkdir(parents=True)

    plist_data: dict[str, Any] = {
        "CFBundleName": name,
        "CFBundleIdentifier": bundle_id,
        "CFBundleShortVersionString": version,
    }
    if display_name:
        plist_data["CFBundleDisplayName"] = display_name
    if category:
        plist_data["LSApplicationCategoryType"] = category

    with open(contents / "Info.plist", "wb") as f:
        plistlib.dump(plist_data, f)

    return app_path


async def _collect_events(
    queue: asyncio.Queue[dict[str, Any]], timeout: float = 2.0
) -> list[dict[str, Any]]:
    """Drain all events from *queue* until *timeout* elapses."""
    events: list[dict[str, Any]] = []
    try:
        while True:
            ev = await asyncio.wait_for(queue.get(), timeout=timeout)
            events.append(ev)
    except (asyncio.TimeoutError, TimeoutError):
        pass
    return events


# ── _discover_apps ────────────────────────────────────────────────


class TestDiscoverApps:
    def test_finds_direct_apps(self, tmp_path: Path) -> None:
        _create_app(tmp_path, "Foo")
        _create_app(tmp_path, "Bar")
        apps = _discover_apps([tmp_path])
        names = {a.name for a in apps}
        assert "Foo.app" in names
        assert "Bar.app" in names

    def test_recurses_one_level(self, tmp_path: Path) -> None:
        utils = tmp_path / "Utilities"
        utils.mkdir()
        _create_app(utils, "Terminal")
        apps = _discover_apps([tmp_path])
        assert any(a.name == "Terminal.app" for a in apps)

    def test_skips_missing_dir(self, tmp_path: Path) -> None:
        apps = _discover_apps([tmp_path / "nonexistent"])
        assert apps == []

    def test_skips_non_app_dirs(self, tmp_path: Path) -> None:
        (tmp_path / "NotAnApp").mkdir()
        _create_app(tmp_path, "RealApp")
        apps = _discover_apps([tmp_path])
        names = {a.name for a in apps}
        assert "RealApp.app" in names
        assert "NotAnApp" not in names

    def test_sorted_output(self, tmp_path: Path) -> None:
        _create_app(tmp_path, "Zebra")
        _create_app(tmp_path, "Alpha")
        apps = _discover_apps([tmp_path])
        assert apps[0].name == "Alpha.app"
        assert apps[1].name == "Zebra.app"


# ── _parse_info_plist ─────────────────────────────────────────────


class TestParseInfoPlist:
    def test_parses_basic_plist(self, tmp_path: Path) -> None:
        app = _create_app(
            tmp_path, "Test", bundle_id="com.test.app", version="2.1",
            category="public.app-category.developer-tools",
        )
        meta = _parse_info_plist(app)
        assert meta is not None
        assert meta["name"] == "Test"
        assert meta["bundle_id"] == "com.test.app"
        assert meta["version"] == "2.1"
        assert meta["path"] == str(app)
        assert meta["category"] == "public.app-category.developer-tools"

    def test_prefers_display_name(self, tmp_path: Path) -> None:
        app = _create_app(tmp_path, "Internal", display_name="My App")
        meta = _parse_info_plist(app)
        assert meta is not None
        assert meta["name"] == "My App"

    def test_falls_back_to_dir_name(self, tmp_path: Path) -> None:
        app = tmp_path / "FallbackApp.app"
        contents = app / "Contents"
        contents.mkdir(parents=True)
        # Plist with no name keys
        with open(contents / "Info.plist", "wb") as f:
            plistlib.dump({"CFBundleIdentifier": "com.fallback"}, f)

        meta = _parse_info_plist(app)
        assert meta is not None
        assert meta["name"] == "FallbackApp"

    def test_missing_plist(self, tmp_path: Path) -> None:
        app = tmp_path / "NoInfo.app"
        app.mkdir(parents=True)
        assert _parse_info_plist(app) is None

    def test_corrupt_plist(self, tmp_path: Path) -> None:
        app = tmp_path / "Bad.app"
        contents = app / "Contents"
        contents.mkdir(parents=True)
        (contents / "Info.plist").write_text("not a plist")
        assert _parse_info_plist(app) is None


# ── _hash_plist ───────────────────────────────────────────────────


class TestHashPlist:
    def test_returns_sha256(self, tmp_path: Path) -> None:
        app = _create_app(tmp_path, "HashTest")
        h = _hash_plist(app)
        assert h is not None
        assert len(h) == 64  # SHA-256 hex digest

    def test_returns_none_for_missing(self, tmp_path: Path) -> None:
        app = tmp_path / "Missing.app"
        app.mkdir()
        assert _hash_plist(app) is None


# ── state persistence ─────────────────────────────────────────────


class TestStatePersistence:
    def test_load_missing_file(self, tmp_path: Path) -> None:
        assert _load_state(tmp_path / "missing.json") == {}

    def test_save_and_load(self, tmp_path: Path) -> None:
        p = tmp_path / "state.json"
        _save_state(p, {"com.test": "abc123"})
        loaded = _load_state(p)
        assert loaded == {"com.test": "abc123"}

    def test_load_corrupt_json(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text("{{{not json")
        assert _load_state(p) == {}

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        p = tmp_path / "sub" / "dir" / "state.json"
        _save_state(p, {"x": "y"})
        assert p.exists()

    def test_save_sets_restrictive_permissions(self, tmp_path: Path) -> None:
        p = tmp_path / "state.json"
        _save_state(p, {"com.test": "abc"})
        assert p.stat().st_mode & 0o777 == 0o600


# ── _build_event ──────────────────────────────────────────────────


class TestBuildEvent:
    def test_event_structure(self) -> None:
        meta = {
            "name": "Test",
            "bundle_id": "com.test",
            "version": "1.0",
            "path": "/Applications/Test.app",
            "category": "",
        }
        event = _build_event(meta, "created")
        assert event["source_type"] == "macos_apps"
        assert event["source_id"] == "app://com.test"
        assert event["operation"] == "created"
        assert event["meta"] == meta
        assert "id" in event
        assert "enqueued_at" in event

    def test_falls_back_to_name_for_source_id(self) -> None:
        meta = {"name": "NoBundle", "bundle_id": "", "version": "", "path": "", "category": ""}
        event = _build_event(meta, "created")
        assert event["source_id"] == "app://NoBundle"


# ── MacOSAppsSource configure ────────────────────────────────────


class TestMacOSAppsSourceConfigure:
    def test_name(self) -> None:
        assert MacOSAppsSource().name() == "macos_apps"

    def test_default_scan_dirs(self) -> None:
        s = MacOSAppsSource()
        s.configure({})
        assert len(s._scan_dirs) == 2

    def test_custom_scan_dirs(self, tmp_path: Path) -> None:
        s = MacOSAppsSource()
        s.configure({"scan_dirs": [str(tmp_path)]})
        assert len(s._scan_dirs) == 1
        assert s._scan_dirs[0] == tmp_path

    def test_poll_interval(self) -> None:
        s = MacOSAppsSource()
        s.configure({"poll_interval_seconds": 3600})
        assert s._poll_interval == 3600

    def test_default_poll_interval(self) -> None:
        s = MacOSAppsSource()
        s.configure({})
        assert s._poll_interval == DEFAULT_POLL_INTERVAL

    def test_explicit_enabled(self) -> None:
        s = MacOSAppsSource()
        s.configure({"enabled": True})
        assert s._enabled is True

    def test_explicit_disabled(self) -> None:
        s = MacOSAppsSource()
        s.configure({"enabled": False})
        assert s._enabled is False

    @patch("worker.sources.macos_apps._is_macos", return_value=False)
    def test_disabled_on_non_macos(self, _mock: Any) -> None:
        s = MacOSAppsSource()
        s.configure({})
        assert s._enabled is False


# ── MacOSAppsSource start (integration) ──────────────────────────


@pytest.mark.asyncio
async def test_initial_scan_emits_created_events(tmp_path: Path) -> None:
    """Initial scan should emit 'created' events for all discovered apps."""
    _create_app(tmp_path, "App1", bundle_id="com.test.app1")
    _create_app(tmp_path, "App2", bundle_id="com.test.app2")

    s = MacOSAppsSource()
    s.configure({
        "enabled": True,
        "scan_dirs": [str(tmp_path)],
        "state_path": str(tmp_path / "state.json"),
        "poll_interval_seconds": 3600,
    })

    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    task = asyncio.create_task(s.start(q))
    events = await _collect_events(q)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert len(events) == 2
    assert all(e["operation"] == "created" for e in events)
    assert all(e["source_type"] == "macos_apps" for e in events)
    bundle_ids = {e["meta"]["bundle_id"] for e in events}
    assert "com.test.app1" in bundle_ids
    assert "com.test.app2" in bundle_ids


@pytest.mark.asyncio
async def test_second_scan_detects_modifications(tmp_path: Path) -> None:
    """Modified Info.plist should emit 'modified' event on second scan."""
    app = _create_app(tmp_path, "ModApp", bundle_id="com.test.mod")
    state_path = tmp_path / "state.json"

    s = MacOSAppsSource()
    s.configure({
        "enabled": True,
        "scan_dirs": [str(tmp_path)],
        "state_path": str(state_path),
        "poll_interval_seconds": 3600,
    })

    # First scan
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    task = asyncio.create_task(s.start(q))
    events = await _collect_events(q)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert len(events) == 1
    assert events[0]["operation"] == "created"

    # Modify the plist
    plist_path = app / "Contents" / "Info.plist"
    with open(plist_path, "rb") as f:
        data = plistlib.load(f)
    data["CFBundleShortVersionString"] = "2.0"
    with open(plist_path, "wb") as f:
        plistlib.dump(data, f)

    # Second scan
    s2 = MacOSAppsSource()
    s2.configure({
        "enabled": True,
        "scan_dirs": [str(tmp_path)],
        "state_path": str(state_path),
        "poll_interval_seconds": 3600,
    })

    q2: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    task2 = asyncio.create_task(s2.start(q2))
    events2 = await _collect_events(q2)
    task2.cancel()
    try:
        await task2
    except asyncio.CancelledError:
        pass

    assert len(events2) == 1
    assert events2[0]["operation"] == "modified"
    assert events2[0]["meta"]["version"] == "2.0"


@pytest.mark.asyncio
async def test_detects_deleted_apps(tmp_path: Path) -> None:
    """Removed app should emit 'deleted' event."""
    app = _create_app(tmp_path, "GoingAway", bundle_id="com.test.bye")
    state_path = tmp_path / "state.json"

    s = MacOSAppsSource()
    s.configure({
        "enabled": True,
        "scan_dirs": [str(tmp_path)],
        "state_path": str(state_path),
        "poll_interval_seconds": 3600,
    })

    # First scan
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    task = asyncio.create_task(s.start(q))
    await _collect_events(q)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # Remove the app
    import shutil
    shutil.rmtree(app)

    # Second scan
    s2 = MacOSAppsSource()
    s2.configure({
        "enabled": True,
        "scan_dirs": [str(tmp_path)],
        "state_path": str(state_path),
        "poll_interval_seconds": 3600,
    })

    q2: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    task2 = asyncio.create_task(s2.start(q2))
    events2 = await _collect_events(q2)
    task2.cancel()
    try:
        await task2
    except asyncio.CancelledError:
        pass

    assert len(events2) == 1
    assert events2[0]["operation"] == "deleted"
    assert events2[0]["source_id"] == "app://com.test.bye"


@pytest.mark.asyncio
async def test_unchanged_apps_emit_no_events(tmp_path: Path) -> None:
    """Second scan with no changes should emit no events."""
    _create_app(tmp_path, "Stable", bundle_id="com.test.stable")
    state_path = tmp_path / "state.json"

    s = MacOSAppsSource()
    s.configure({
        "enabled": True,
        "scan_dirs": [str(tmp_path)],
        "state_path": str(state_path),
        "poll_interval_seconds": 3600,
    })

    # First scan
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    task = asyncio.create_task(s.start(q))
    await _collect_events(q)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # Second scan
    s2 = MacOSAppsSource()
    s2.configure({
        "enabled": True,
        "scan_dirs": [str(tmp_path)],
        "state_path": str(state_path),
        "poll_interval_seconds": 3600,
    })

    q2: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    task2 = asyncio.create_task(s2.start(q2))
    events2 = await _collect_events(q2)
    task2.cancel()
    try:
        await task2
    except asyncio.CancelledError:
        pass

    assert len(events2) == 0


@pytest.mark.asyncio
async def test_skips_broken_app_bundles(tmp_path: Path) -> None:
    """Broken .app bundles (no Info.plist) should be skipped."""
    broken = tmp_path / "Broken.app"
    broken.mkdir()  # No Contents/Info.plist

    _create_app(tmp_path, "Good", bundle_id="com.test.good")

    s = MacOSAppsSource()
    s.configure({
        "enabled": True,
        "scan_dirs": [str(tmp_path)],
        "state_path": str(tmp_path / "state.json"),
        "poll_interval_seconds": 3600,
    })

    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    task = asyncio.create_task(s.start(q))
    events = await _collect_events(q)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert len(events) == 1
    assert events[0]["meta"]["bundle_id"] == "com.test.good"


@pytest.mark.asyncio
async def test_disabled_source_emits_nothing(tmp_path: Path) -> None:
    """Disabled source should not scan or emit events."""
    _create_app(tmp_path, "App", bundle_id="com.test")

    s = MacOSAppsSource()
    s.configure({
        "enabled": False,
        "scan_dirs": [str(tmp_path)],
        "state_path": str(tmp_path / "state.json"),
        "poll_interval_seconds": 3600,
    })

    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    task = asyncio.create_task(s.start(q))
    events = await _collect_events(q, timeout=1.0)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert len(events) == 0
