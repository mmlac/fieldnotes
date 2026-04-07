"""Tests for Python source shims: base interface and file watcher.

Uses event-driven synchronization (asyncio.wait_for on queue.get) instead
of sleep-based polling to avoid timing flakiness.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any

import pytest

from _fake_queue import FakeQueue
from worker.sources.base import PythonSource
from worker.sources.files import DEFAULT_MAX_FILE_SIZE, FileSource, _streaming_sha256


# ── Helpers ────────────────────────────────────────────────────────


async def _collect_events(
    q: Any,
    *,
    min_events: int = 1,
    timeout: float = 5.0,
    path_prefix: str | None = None,
    ack: bool = True,
) -> list[dict[str, Any]]:
    """Collect at least *min_events* from the queue, with a hard timeout.

    When *path_prefix* is given, only events whose ``source_id`` starts
    with the prefix count towards *min_events*.  Events outside the
    prefix are silently discarded so that stale filesystem notifications
    from earlier test runs never pollute the result.
    """
    events: list[dict[str, Any]] = []
    try:
        while len(events) < min_events:
            ev = await asyncio.wait_for(q.get(), timeout=timeout)
            if ack:
                cb = ev.get("_on_indexed")
                if cb:
                    cb()
            if path_prefix and not ev.get("source_id", "").startswith(path_prefix):
                continue
            events.append(ev)
    except asyncio.TimeoutError:
        pass
    return events


async def _collect_until(
    q: Any,
    predicate,
    *,
    timeout: float = 5.0,
    path_prefix: str | None = None,
    ack: bool = True,
) -> list[dict[str, Any]]:
    """Collect events until *predicate(events)* is true or timeout.

    Events whose ``source_id`` does not start with *path_prefix* (when
    given) are silently dropped.
    """
    events: list[dict[str, Any]] = []
    deadline = asyncio.get_event_loop().time() + timeout
    try:
        while not predicate(events):
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                break
            ev = await asyncio.wait_for(q.get(), timeout=remaining)
            if ack:
                cb = ev.get("_on_indexed")
                if cb:
                    cb()
            if path_prefix and not ev.get("source_id", "").startswith(path_prefix):
                continue
            events.append(ev)
    except asyncio.TimeoutError:
        pass
    return events


# ── PythonSource ABC ────────────────────────────────────────────────


class DummySource(PythonSource):
    def name(self) -> str:
        return "dummy"

    def configure(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg

    async def start(self, queue: Any, **_kwargs: Any) -> None:
        queue.enqueue({"source_type": "dummy"})


def test_python_source_subclass():
    s = DummySource()
    assert s.name() == "dummy"
    s.configure({"key": "val"})
    assert s.cfg == {"key": "val"}


@pytest.mark.asyncio
async def test_dummy_source_emits_event():
    s = DummySource()
    q = FakeQueue()
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
    assert fs._max_file_size == DEFAULT_MAX_FILE_SIZE


def test_file_source_configure_max_file_size(tmp_path: Path):
    fs = FileSource()
    fs.configure({"watch_paths": [str(tmp_path)], "max_file_size": 1024})
    assert fs._max_file_size == 1024


def test_file_source_configure_extensions(tmp_path: Path):
    fs = FileSource()
    fs.configure(
        {
            "watch_paths": [str(tmp_path)],
            "include_extensions": [".md", "txt"],
        }
    )
    assert fs._include_extensions == {".md", ".txt"}


def test_file_source_configure_excludes(tmp_path: Path):
    fs = FileSource()
    fs.configure(
        {
            "watch_paths": [str(tmp_path)],
            "exclude_patterns": ["*.pyc", ".git/*"],
            "recursive": False,
        }
    )
    assert fs._exclude_patterns == ["*.pyc", ".git/*"]
    assert fs._recursive is False


# ── FileSource watcher integration ──────────────────────────────────


@pytest.mark.asyncio
async def test_file_source_detects_create(tmp_path: Path):
    prefix = str(tmp_path.resolve())
    fs = FileSource()
    fs.configure({"watch_paths": [str(tmp_path)]})
    q = FakeQueue()

    task = asyncio.create_task(fs.start(q))
    # Give the observer thread a moment to attach.
    await asyncio.sleep(0.3)

    test_file = tmp_path / "hello.md"
    test_file.write_text("hello world")

    events = await _collect_events(q, min_events=1, timeout=5.0, path_prefix=prefix)

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert len(events) >= 1
    ev = events[0]
    assert ev["source_type"] == "files"
    assert ev["source_id"] == str(test_file.resolve())
    assert ev["operation"] in ("created", "modified")
    assert ev["mime_type"] == "text/markdown"
    assert "id" in ev
    assert "enqueued_at" in ev


@pytest.mark.asyncio
async def test_file_source_populates_text_for_text_files(tmp_path: Path):
    """Text MIME type files must have their content loaded into event['text']."""
    prefix = str(tmp_path.resolve())
    fs = FileSource()
    fs.configure({"watch_paths": [str(tmp_path)]})
    q = FakeQueue()

    task = asyncio.create_task(fs.start(q))
    await asyncio.sleep(0.3)

    test_file = tmp_path / "note.md"
    test_file.write_text("# Hello\n\nSome content here.")

    events = await _collect_events(q, min_events=1, timeout=5.0, path_prefix=prefix)

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert len(events) >= 1
    ev = events[0]
    assert ev["mime_type"] == "text/markdown"
    assert ev["text"] == "# Hello\n\nSome content here."


@pytest.mark.asyncio
async def test_file_source_no_text_for_binary_files(tmp_path: Path):
    """Binary files should not have a 'text' key in the event."""
    prefix = str(tmp_path.resolve())
    fs = FileSource()
    fs.configure({"watch_paths": [str(tmp_path)], "include_extensions": [".png"]})
    q = FakeQueue()

    task = asyncio.create_task(fs.start(q))
    await asyncio.sleep(0.3)

    test_file = tmp_path / "image.png"
    test_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)

    events = await _collect_events(q, min_events=1, timeout=5.0, path_prefix=prefix)

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert len(events) >= 1
    ev = events[0]
    assert "text" not in ev


@pytest.mark.asyncio
async def test_file_source_respects_extension_filter(tmp_path: Path):
    prefix = str(tmp_path.resolve())
    fs = FileSource()
    fs.configure(
        {
            "watch_paths": [str(tmp_path)],
            "include_extensions": [".md"],
        }
    )
    q = FakeQueue()

    task = asyncio.create_task(fs.start(q))
    await asyncio.sleep(0.3)

    # Create a .txt file (should be filtered out)
    (tmp_path / "skip.txt").write_text("skip me")
    # Create a .md file (should pass)
    (tmp_path / "keep.md").write_text("keep me")

    events = await _collect_until(
        q,
        lambda evs: any(e["source_id"].endswith("keep.md") for e in evs),
        timeout=5.0,
        path_prefix=prefix,
    )

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
    prefix = str(tmp_path.resolve())
    fs = FileSource()
    fs.configure(
        {
            "watch_paths": [str(tmp_path)],
            "exclude_patterns": ["*.pyc"],
        }
    )
    q = FakeQueue()

    task = asyncio.create_task(fs.start(q))
    await asyncio.sleep(0.3)

    (tmp_path / "bad.pyc").write_bytes(b"\x00")
    (tmp_path / "good.py").write_text("print('hi')")

    events = await _collect_until(
        q,
        lambda evs: any(e["source_id"].endswith("good.py") for e in evs),
        timeout=5.0,
        path_prefix=prefix,
    )

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
    prefix = str(tmp_path.resolve())
    # Pre-create a file before watching
    target = tmp_path / "doomed.txt"
    target.write_text("goodbye")

    fs = FileSource()
    fs.configure({"watch_paths": [str(tmp_path)]})
    q = FakeQueue()

    task = asyncio.create_task(fs.start(q))
    await asyncio.sleep(0.3)

    target.unlink()

    events = await _collect_until(
        q,
        lambda evs: any(e["operation"] == "deleted" for e in evs),
        timeout=5.0,
        path_prefix=prefix,
    )

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    deleted = [e for e in events if e["operation"] == "deleted"]
    assert len(deleted) >= 1
    assert "doomed.txt" in deleted[0]["source_id"]


# ── Streaming SHA-256 ──────────────────────────────────────────────


def test_streaming_sha256_small_file(tmp_path: Path):
    p = tmp_path / "small.txt"
    p.write_bytes(b"hello")
    result = _streaming_sha256(p, max_size=1024)
    assert result is not None
    digest, size = result
    assert size == 5
    assert digest == hashlib.sha256(b"hello").hexdigest()


def test_streaming_sha256_exceeds_max_size(tmp_path: Path):
    p = tmp_path / "big.bin"
    p.write_bytes(b"x" * 200)
    result = _streaming_sha256(p, max_size=100)
    assert result is None


# ── Max file size skipping ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_file_source_indexes_oversized_as_metadata(tmp_path: Path):
    prefix = str(tmp_path.resolve())
    fs = FileSource()
    fs.configure(
        {
            "watch_paths": [str(tmp_path)],
            "max_file_size": 50,
        }
    )
    q = FakeQueue()

    task = asyncio.create_task(fs.start(q))
    await asyncio.sleep(0.3)

    # Create an oversized file (should be indexed as metadata only)
    (tmp_path / "big.md").write_bytes(b"x" * 100)
    # Create a small file (should have full content)
    (tmp_path / "small.md").write_text("ok")

    events = await _collect_until(
        q,
        lambda evs: any("small.md" in e["source_id"] for e in evs),
        timeout=5.0,
        path_prefix=prefix,
    )

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    small = [e for e in events if "small.md" in e["source_id"]]
    big = [e for e in events if "big.md" in e["source_id"]]

    assert len(small) >= 1
    assert "text" in small[0]

    assert len(big) >= 1
    assert "text" not in big[0]
    assert big[0]["meta"].get("oversized") is True
    assert big[0]["meta"]["size_bytes"] == 100
