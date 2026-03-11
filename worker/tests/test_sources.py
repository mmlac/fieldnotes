"""Tests for Python source shims: base interface and file watcher."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any

import pytest

from worker.sources.base import PythonSource
from worker.sources.files import FileSource


# ── PythonSource ABC ────────────────────────────────────────────────


class DummySource(PythonSource):
    def name(self) -> str:
        return "dummy"

    def configure(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg

    async def start(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        await queue.put({"source_type": "dummy"})


def test_python_source_subclass():
    s = DummySource()
    assert s.name() == "dummy"
    s.configure({"key": "val"})
    assert s.cfg == {"key": "val"}


@pytest.mark.asyncio
async def test_dummy_source_emits_event():
    s = DummySource()
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    await s.start(q)
    event = q.get_nowait()
    assert event["source_type"] == "dummy"


def test_cannot_instantiate_abstract():
    with pytest.raises(TypeError):
        PythonSource()  # type: ignore[abstract]


# ── FileSource configure ────────────────────────────────────────────


def test_file_source_name():
    fs = FileSource()
    assert fs.name() == "files"


def test_file_source_requires_watch_paths():
    fs = FileSource()
    with pytest.raises(ValueError, match="watch_paths"):
        fs.configure({})


def test_file_source_configure_basic(tmp_path: Path):
    fs = FileSource()
    fs.configure({"watch_paths": [str(tmp_path)]})
    assert fs._watch_paths == [tmp_path]
    assert fs._include_extensions is None
    assert fs._exclude_patterns == []
    assert fs._recursive is True


def test_file_source_configure_extensions(tmp_path: Path):
    fs = FileSource()
    fs.configure({
        "watch_paths": [str(tmp_path)],
        "include_extensions": [".md", "txt"],
    })
    assert fs._include_extensions == {".md", ".txt"}


def test_file_source_configure_excludes(tmp_path: Path):
    fs = FileSource()
    fs.configure({
        "watch_paths": [str(tmp_path)],
        "exclude_patterns": ["*.pyc", ".git/*"],
        "recursive": False,
    })
    assert fs._exclude_patterns == ["*.pyc", ".git/*"]
    assert fs._recursive is False


# ── FileSource watcher integration ──────────────────────────────────


@pytest.mark.asyncio
async def test_file_source_detects_create(tmp_path: Path):
    fs = FileSource()
    fs.configure({"watch_paths": [str(tmp_path)]})
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    task = asyncio.create_task(fs.start(q))
    await asyncio.sleep(0.5)  # let observer start

    # Create a file
    test_file = tmp_path / "hello.md"
    test_file.write_text("hello world")
    content_hash = hashlib.sha256(b"hello world").hexdigest()

    # Wait for events (watchdog may fire created + modified)
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
    assert ev["source_type"] == "files"
    assert ev["source_id"] == str(test_file)
    assert ev["operation"] in ("created", "modified")
    assert ev["mime_type"] == "text/markdown"
    assert "id" in ev
    assert "enqueued_at" in ev


@pytest.mark.asyncio
async def test_file_source_respects_extension_filter(tmp_path: Path):
    fs = FileSource()
    fs.configure({
        "watch_paths": [str(tmp_path)],
        "include_extensions": [".md"],
    })
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    task = asyncio.create_task(fs.start(q))
    await asyncio.sleep(0.5)

    # Create a .txt file (should be filtered out)
    (tmp_path / "skip.txt").write_text("skip me")
    await asyncio.sleep(0.5)

    # Create a .md file (should pass)
    (tmp_path / "keep.md").write_text("keep me")

    events = []
    try:
        for _ in range(10):
            await asyncio.sleep(0.3)
            while not q.empty():
                events.append(q.get_nowait())
            if any(e["source_id"].endswith("keep.md") for e in events):
                break
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    source_ids = [e["source_id"] for e in events]
    assert any("keep.md" in sid for sid in source_ids)
    assert not any("skip.txt" in sid for sid in source_ids)


@pytest.mark.asyncio
async def test_file_source_respects_exclude_pattern(tmp_path: Path):
    fs = FileSource()
    fs.configure({
        "watch_paths": [str(tmp_path)],
        "exclude_patterns": ["*.pyc"],
    })
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    task = asyncio.create_task(fs.start(q))
    await asyncio.sleep(0.5)

    (tmp_path / "bad.pyc").write_bytes(b"\x00")
    (tmp_path / "good.py").write_text("print('hi')")

    events = []
    try:
        for _ in range(10):
            await asyncio.sleep(0.3)
            while not q.empty():
                events.append(q.get_nowait())
            if any(e["source_id"].endswith("good.py") for e in events):
                break
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    source_ids = [e["source_id"] for e in events]
    assert any("good.py" in sid for sid in source_ids)
    assert not any("bad.pyc" in sid for sid in source_ids)


@pytest.mark.asyncio
async def test_file_source_delete_event(tmp_path: Path):
    # Pre-create a file before watching
    target = tmp_path / "doomed.txt"
    target.write_text("goodbye")

    fs = FileSource()
    fs.configure({"watch_paths": [str(tmp_path)]})
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    task = asyncio.create_task(fs.start(q))
    await asyncio.sleep(0.5)

    target.unlink()

    events = []
    try:
        for _ in range(10):
            await asyncio.sleep(0.3)
            while not q.empty():
                events.append(q.get_nowait())
            if any(e["operation"] == "deleted" for e in events):
                break
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    deleted = [e for e in events if e["operation"] == "deleted"]
    assert len(deleted) >= 1
    assert "doomed.txt" in deleted[0]["source_id"]
