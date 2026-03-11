"""Unit tests for BaseHandler filtering, hashing, event building, and dispatch."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from watchdog.events import (
    DirCreatedEvent,
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
)

from worker.sources._handler import (
    BaseHandler,
    DEFAULT_MAX_FILE_SIZE,
    guess_mime,
    streaming_sha256,
)


# ── Helpers ────────────────────────────────────────────────────────


def _make_handler(
    include_extensions: set[str] | None = None,
    exclude_patterns: list[str] | None = None,
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
) -> BaseHandler:
    loop = asyncio.new_event_loop()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    h = BaseHandler(
        queue=queue,
        loop=loop,
        include_extensions=include_extensions,
        exclude_patterns=exclude_patterns or [],
        max_file_size=max_file_size,
    )
    h._source_type = "test"
    return h


# ── guess_mime ─────────────────────────────────────────────────────


class TestGuessMime:
    def test_markdown(self):
        assert guess_mime("notes/hello.md") == "text/markdown"

    def test_plain_text(self):
        assert guess_mime("readme.txt") == "text/plain"

    def test_pdf(self):
        assert guess_mime("/tmp/doc.pdf") == "application/pdf"

    def test_jpeg_variants(self):
        assert guess_mime("photo.jpg") == "image/jpeg"
        assert guess_mime("photo.jpeg") == "image/jpeg"

    def test_png(self):
        assert guess_mime("img.png") == "image/png"

    def test_gif(self):
        assert guess_mime("anim.gif") == "image/gif"

    def test_svg(self):
        assert guess_mime("icon.svg") == "image/svg+xml"

    def test_json(self):
        assert guess_mime("data.json") == "application/json"

    def test_yaml_variants(self):
        assert guess_mime("cfg.yaml") == "text/yaml"
        assert guess_mime("cfg.yml") == "text/yaml"

    def test_toml(self):
        assert guess_mime("pyproject.toml") == "text/toml"

    def test_html(self):
        assert guess_mime("index.html") == "text/html"

    def test_csv(self):
        assert guess_mime("data.csv") == "text/csv"

    def test_canvas(self):
        assert guess_mime("board.canvas") == "application/json"

    def test_unknown_extension(self):
        assert guess_mime("binary.xyz") == "application/octet-stream"

    def test_no_extension(self):
        assert guess_mime("Makefile") == "application/octet-stream"

    def test_case_insensitive(self):
        assert guess_mime("NOTES.MD") == "text/markdown"
        assert guess_mime("image.PNG") == "image/png"


# ── streaming_sha256 ──────────────────────────────────────────────


class TestStreamingSha256:
    def test_normal_file(self, tmp_path: Path):
        p = tmp_path / "data.bin"
        content = b"hello world"
        p.write_bytes(content)
        result = streaming_sha256(p, max_size=1024)
        assert result is not None
        digest, size = result
        assert size == len(content)
        assert digest == hashlib.sha256(content).hexdigest()

    def test_empty_file(self, tmp_path: Path):
        p = tmp_path / "empty.txt"
        p.write_bytes(b"")
        result = streaming_sha256(p, max_size=1024)
        assert result is not None
        digest, size = result
        assert size == 0
        assert digest == hashlib.sha256(b"").hexdigest()

    def test_exceeds_max_size(self, tmp_path: Path):
        p = tmp_path / "big.bin"
        p.write_bytes(b"x" * 200)
        result = streaming_sha256(p, max_size=100)
        assert result is None

    def test_exactly_at_max_size(self, tmp_path: Path):
        p = tmp_path / "exact.bin"
        content = b"x" * 100
        p.write_bytes(content)
        result = streaming_sha256(p, max_size=100)
        assert result is not None
        digest, size = result
        assert size == 100
        assert digest == hashlib.sha256(content).hexdigest()

    def test_file_not_found(self, tmp_path: Path):
        p = tmp_path / "nonexistent.txt"
        with pytest.raises(FileNotFoundError):
            streaming_sha256(p, max_size=1024)

    def test_multi_chunk_file(self, tmp_path: Path):
        """File larger than the internal chunk size (64 KiB)."""
        p = tmp_path / "large.bin"
        content = b"A" * (128 * 1024)  # 128 KiB — spans 2 chunks
        p.write_bytes(content)
        result = streaming_sha256(p, max_size=256 * 1024)
        assert result is not None
        digest, size = result
        assert size == len(content)
        assert digest == hashlib.sha256(content).hexdigest()


# ── _should_skip ──────────────────────────────────────────────────


class TestShouldSkip:
    def test_no_filters_passes(self):
        h = _make_handler()
        assert h._should_skip("/path/to/file.md") is False

    def test_include_extensions_match(self):
        h = _make_handler(include_extensions={".md", ".txt"})
        assert h._should_skip("/path/file.md") is False
        assert h._should_skip("/path/file.txt") is False

    def test_include_extensions_reject(self):
        h = _make_handler(include_extensions={".md"})
        assert h._should_skip("/path/file.py") is True

    def test_include_extensions_case_insensitive(self):
        h = _make_handler(include_extensions={".md"})
        assert h._should_skip("/path/file.MD") is False

    def test_exclude_pattern_full_path(self):
        h = _make_handler(exclude_patterns=["*/.git/*"])
        assert h._should_skip("/repo/.git/config") is True

    def test_exclude_pattern_filename(self):
        h = _make_handler(exclude_patterns=["*.pyc"])
        assert h._should_skip("/path/to/module.pyc") is True

    def test_exclude_pattern_no_match(self):
        h = _make_handler(exclude_patterns=["*.pyc"])
        assert h._should_skip("/path/to/module.py") is False

    def test_multiple_exclude_patterns(self):
        h = _make_handler(exclude_patterns=["*.pyc", "*.tmp", ".DS_Store"])
        assert h._should_skip("/a/b.pyc") is True
        assert h._should_skip("/a/b.tmp") is True
        assert h._should_skip("/a/.DS_Store") is True
        assert h._should_skip("/a/b.py") is False

    def test_both_include_and_exclude(self):
        h = _make_handler(include_extensions={".md"}, exclude_patterns=["*/drafts/*"])
        # .md but in drafts → excluded
        assert h._should_skip("/notes/drafts/idea.md") is True
        # .md not in drafts → passes
        assert h._should_skip("/notes/final/idea.md") is False
        # .py → excluded by include filter
        assert h._should_skip("/notes/final/code.py") is True

    def test_extra_skip_hook(self):
        h = _make_handler()
        h._extra_skip = lambda path: ".obsidian" in path  # type: ignore[assignment]
        assert h._should_skip("/vault/.obsidian/config") is True
        assert h._should_skip("/vault/notes.md") is False


# ── _operation ────────────────────────────────────────────────────


class TestOperation:
    def test_created(self):
        ev = FileCreatedEvent("/a/b.md")
        assert BaseHandler._operation(ev) == "created"

    def test_modified(self):
        ev = FileModifiedEvent("/a/b.md")
        assert BaseHandler._operation(ev) == "modified"

    def test_deleted(self):
        ev = FileDeletedEvent("/a/b.md")
        assert BaseHandler._operation(ev) == "deleted"

    def test_unknown_event_type(self):
        ev = FileMovedEvent("/a/b.md", "/a/c.md")
        assert BaseHandler._operation(ev) is None


# ── _build_event ──────────────────────────────────────────────────


class TestBuildEvent:
    def test_directory_event_returns_none(self):
        h = _make_handler()
        ev = DirCreatedEvent("/some/dir")
        assert h._build_event(ev) is None

    def test_unsupported_event_type_returns_none(self):
        h = _make_handler()
        ev = FileMovedEvent("/a.md", "/b.md")
        assert h._build_event(ev) is None

    def test_skipped_file_returns_none(self):
        h = _make_handler(include_extensions={".md"})
        ev = FileCreatedEvent("/path/file.py")
        assert h._build_event(ev) is None

    def test_created_event_structure(self, tmp_path: Path):
        p = tmp_path / "note.md"
        p.write_text("hello")
        h = _make_handler()
        ev = FileCreatedEvent(str(p))
        result = h._build_event(ev)
        assert result is not None
        assert result["source_type"] == "test"
        assert result["source_id"] == str(p)
        assert result["operation"] == "created"
        assert result["mime_type"] == "text/markdown"
        assert "id" in result
        assert "enqueued_at" in result
        assert "source_modified_at" in result
        assert result["meta"]["sha256"] == hashlib.sha256(b"hello").hexdigest()
        assert result["meta"]["size_bytes"] == 5
        assert result["text"] == "hello"

    def test_deleted_event_no_content(self, tmp_path: Path):
        p = tmp_path / "gone.md"
        # File doesn't exist for deleted events
        h = _make_handler()
        ev = FileDeletedEvent(str(p))
        result = h._build_event(ev)
        assert result is not None
        assert result["operation"] == "deleted"
        assert "sha256" not in result.get("meta", {})
        assert "text" not in result

    def test_binary_file_no_text_key(self, tmp_path: Path):
        p = tmp_path / "image.png"
        p.write_bytes(b"\x89PNG" + b"\x00" * 20)
        h = _make_handler()
        ev = FileCreatedEvent(str(p))
        result = h._build_event(ev)
        assert result is not None
        assert result["mime_type"] == "image/png"
        assert "text" not in result

    def test_oversized_file_returns_none(self, tmp_path: Path):
        p = tmp_path / "huge.md"
        p.write_bytes(b"x" * 200)
        h = _make_handler(max_file_size=100)
        ev = FileCreatedEvent(str(p))
        assert h._build_event(ev) is None

    def test_extra_meta_included(self, tmp_path: Path):
        p = tmp_path / "note.md"
        p.write_text("content")
        h = _make_handler()
        h._extra_meta = lambda src_path: {"vault": "test-vault"}  # type: ignore[assignment]
        ev = FileCreatedEvent(str(p))
        result = h._build_event(ev)
        assert result is not None
        assert result["meta"]["vault"] == "test-vault"
        # sha256 is merged into the same meta dict
        assert "sha256" in result["meta"]

    def test_oserror_on_read_emits_without_hash(self, tmp_path: Path):
        p = tmp_path / "broken.md"
        p.write_text("data")
        h = _make_handler()
        ev = FileCreatedEvent(str(p))
        with patch(
            "worker.sources._handler.streaming_sha256",
            side_effect=OSError("disk error"),
        ):
            result = h._build_event(ev)
        assert result is not None
        assert "sha256" not in result.get("meta", {})
        assert "source_modified_at" in result

    def test_modified_event(self, tmp_path: Path):
        p = tmp_path / "edited.txt"
        p.write_text("updated")
        h = _make_handler()
        ev = FileModifiedEvent(str(p))
        result = h._build_event(ev)
        assert result is not None
        assert result["operation"] == "modified"
        assert result["text"] == "updated"


# ── _dispatch and run_coroutine_threadsafe ────────────────────────


class TestDispatch:
    def test_dispatch_enqueues_event(self):
        loop = asyncio.new_event_loop()
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        h = BaseHandler(
            queue=queue,
            loop=loop,
            include_extensions=None,
            exclude_patterns=[],
        )
        h._source_type = "test"

        # Create a real file for the event
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            f.write(b"dispatch test")
            tmp_name = f.name

        try:
            ev = FileCreatedEvent(tmp_name)
            # Run dispatch in a thread while the loop is running
            import threading

            def run_loop():
                loop.run_forever()

            t = threading.Thread(target=run_loop, daemon=True)
            t.start()

            h._dispatch(ev)

            # Give the coroutine time to execute
            import time

            time.sleep(0.1)

            loop.call_soon_threadsafe(loop.stop)
            t.join(timeout=2)

            assert not queue.empty()
            item = queue.get_nowait()
            assert item["source_type"] == "test"
            assert item["source_id"] == tmp_name
        finally:
            Path(tmp_name).unlink(missing_ok=True)
            loop.close()

    def test_dispatch_skips_none_event(self):
        loop = asyncio.new_event_loop()
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        h = BaseHandler(
            queue=queue,
            loop=loop,
            include_extensions={".md"},
            exclude_patterns=[],
        )
        h._source_type = "test"

        # .py file should be skipped by include filter
        ev = FileCreatedEvent("/some/file.py")

        import threading

        def run_loop():
            loop.run_forever()

        t = threading.Thread(target=run_loop, daemon=True)
        t.start()

        h._dispatch(ev)

        import time

        time.sleep(0.1)

        loop.call_soon_threadsafe(loop.stop)
        t.join(timeout=2)

        assert queue.empty()
        loop.close()

    def test_on_created_delegates_to_dispatch(self, tmp_path: Path):
        h = _make_handler()
        calls = []
        h._dispatch = lambda ev: calls.append(ev)  # type: ignore[assignment]
        ev = FileCreatedEvent(str(tmp_path / "a.md"))
        h.on_created(ev)
        assert len(calls) == 1

    def test_on_modified_delegates_to_dispatch(self, tmp_path: Path):
        h = _make_handler()
        calls = []
        h._dispatch = lambda ev: calls.append(ev)  # type: ignore[assignment]
        ev = FileModifiedEvent(str(tmp_path / "a.md"))
        h.on_modified(ev)
        assert len(calls) == 1

    def test_on_deleted_delegates_to_dispatch(self, tmp_path: Path):
        h = _make_handler()
        calls = []
        h._dispatch = lambda ev: calls.append(ev)  # type: ignore[assignment]
        ev = FileDeletedEvent(str(tmp_path / "a.md"))
        h.on_deleted(ev)
        assert len(calls) == 1
