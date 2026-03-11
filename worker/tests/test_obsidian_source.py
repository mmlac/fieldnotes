"""Tests for the Obsidian vault source shim."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from worker.sources.obsidian import ObsidianSource, discover_vaults


# ── discover_vaults ─────────────────────────────────────────────────


def test_discover_vaults_finds_direct_vault(tmp_path: Path):
    (tmp_path / ".obsidian").mkdir()
    vaults = discover_vaults([tmp_path])
    assert vaults == [tmp_path]


def test_discover_vaults_finds_nested_vaults(tmp_path: Path):
    (tmp_path / "vault_a" / ".obsidian").mkdir(parents=True)
    (tmp_path / "vault_b" / ".obsidian").mkdir(parents=True)
    (tmp_path / "not_a_vault").mkdir()
    vaults = discover_vaults([tmp_path])
    names = [v.name for v in vaults]
    assert "vault_a" in names
    assert "vault_b" in names
    assert "not_a_vault" not in names


def test_discover_vaults_skips_missing_path(tmp_path: Path):
    missing = tmp_path / "nonexistent"
    vaults = discover_vaults([missing])
    assert vaults == []


# ── ObsidianSource configure ───────────────────────────────────────


def test_obsidian_source_name():
    s = ObsidianSource()
    assert s.name() == "obsidian"


def test_obsidian_source_requires_vault_paths():
    s = ObsidianSource()
    with pytest.raises(ValueError, match="vault_paths"):
        s.configure({})


def test_obsidian_source_configure_basic(tmp_path: Path):
    s = ObsidianSource()
    s.configure({"vault_paths": [str(tmp_path)]})
    assert s._vault_paths == [tmp_path]
    assert s._include_extensions is None
    assert s._exclude_patterns == []
    assert s._recursive is True


def test_obsidian_source_configure_extensions(tmp_path: Path):
    s = ObsidianSource()
    s.configure({
        "vault_paths": [str(tmp_path)],
        "include_extensions": [".md", "canvas"],
    })
    assert s._include_extensions == {".md", ".canvas"}


# ── ObsidianSource watcher integration ─────────────────────────────


@pytest.mark.asyncio
async def test_obsidian_source_detects_create(tmp_path: Path):
    vault = tmp_path / "my_vault"
    vault.mkdir()
    (vault / ".obsidian").mkdir()

    s = ObsidianSource()
    s.configure({"vault_paths": [str(tmp_path)]})
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    task = asyncio.create_task(s.start(q))
    await asyncio.sleep(0.5)

    test_file = vault / "note.md"
    test_file.write_text("# Hello")

    events = []
    try:
        for _ in range(10):
            await asyncio.sleep(0.3)
            while not q.empty():
                events.append(q.get_nowait())
            if events:
                break
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert len(events) >= 1
    ev = events[0]
    assert ev["source_type"] == "obsidian"
    assert ev["operation"] in ("created", "modified")
    assert ev["mime_type"] == "text/markdown"
    assert ev["meta"]["vault_name"] == "my_vault"
    assert ev["meta"]["vault_path"] == str(vault)
    assert ev["meta"]["relative_path"] == "note.md"


@pytest.mark.asyncio
async def test_obsidian_source_skips_dotobsidian_files(tmp_path: Path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / ".obsidian").mkdir()

    s = ObsidianSource()
    s.configure({"vault_paths": [str(tmp_path)]})
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    task = asyncio.create_task(s.start(q))
    await asyncio.sleep(0.5)

    # Write to .obsidian config — should be skipped
    (vault / ".obsidian" / "app.json").write_text("{}")
    await asyncio.sleep(0.5)

    # Write a real note — should be captured
    (vault / "real.md").write_text("content")

    events = []
    try:
        for _ in range(10):
            await asyncio.sleep(0.3)
            while not q.empty():
                events.append(q.get_nowait())
            if any("real.md" in e["source_id"] for e in events):
                break
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    source_ids = [e["source_id"] for e in events]
    assert any("real.md" in sid for sid in source_ids)
    assert not any(".obsidian" in sid for sid in source_ids)


@pytest.mark.asyncio
async def test_obsidian_source_no_vaults_found(tmp_path: Path):
    """Source should start and be cancellable even with no vaults."""
    s = ObsidianSource()
    s.configure({"vault_paths": [str(tmp_path)]})
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    task = asyncio.create_task(s.start(q))
    await asyncio.sleep(0.3)

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert q.empty()
