"""Tests for the repository source adapter: discovery, filtering, cursors, events."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import git
import pytest

from worker.sources.repositories import (
    DEFAULT_EXCLUDE_PATTERNS,
    DEFAULT_INCLUDE_PATTERNS,
    RepositorySource,
    _discover_repos,
    _load_cursor,
    _matches_any,
    _save_cursor,
)


# ── helpers ────────────────────────────────────────────────────────


class _TestQueue:
    """Thin wrapper around asyncio.Queue that exposes PersistentQueue API."""

    def __init__(self) -> None:
        self._q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._cursors: dict[str, str] = {}
        self._enqueued_ids: set[str] = set()

    def enqueue(
        self,
        event: dict[str, Any],
        cursor_key: str | None = None,
        cursor_value: str | None = None,
    ) -> str:
        self._q.put_nowait(event)
        sid = event.get("source_id", "")
        if sid:
            self._enqueued_ids.add(sid)
        if cursor_key is not None and cursor_value is not None:
            self._cursors[cursor_key] = cursor_value
        return event.get("id", "")

    def is_enqueued(self, source_id: str) -> bool:
        return source_id in self._enqueued_ids

    def load_cursor(self, key: str) -> str | None:
        return self._cursors.get(key)

    def save_cursor(self, key: str, value: str) -> None:
        self._cursors[key] = value

    async def get(self) -> dict[str, Any]:
        return await self._q.get()

    def get_nowait(self) -> dict[str, Any]:
        return self._q.get_nowait()

    def qsize(self) -> int:
        return self._q.qsize()


def _init_repo(path: Path, files: dict[str, str] | None = None) -> git.Repo:
    """Create a non-bare git repo at *path* with an initial commit."""
    path.mkdir(parents=True, exist_ok=True)
    repo = git.Repo.init(path)
    repo.config_writer().set_value("user", "name", "Test").release()
    repo.config_writer().set_value("user", "email", "test@example.com").release()

    if files is None:
        files = {"README.md": "# Hello\n"}

    for name, content in files.items():
        fp = path / name
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        repo.index.add([name])

    repo.index.commit("Initial commit")
    return repo


async def _collect_events(
    queue: _TestQueue, timeout: float = 2.0, ack: bool = True
) -> list[dict[str, Any]]:
    """Drain all events from *queue* until *timeout* elapses with no new events."""
    events: list[dict[str, Any]] = []
    try:
        while True:
            ev = await asyncio.wait_for(queue.get(), timeout=timeout)
            if ack:
                cb = ev.get("_on_indexed")
                if cb:
                    cb()
            events.append(ev)
    except (asyncio.TimeoutError, TimeoutError):
        pass
    return events


# ── _discover_repos ────────────────────────────────────────────────


class TestDiscoverRepos:
    def test_finds_direct_repo(self, tmp_path: Path) -> None:
        _init_repo(tmp_path / "myrepo")
        repos = _discover_repos([tmp_path])
        assert len(repos) == 1
        assert repos[0] == tmp_path / "myrepo"

    def test_finds_root_itself_if_git(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        repos = _discover_repos([tmp_path])
        assert tmp_path in repos

    def test_skips_bare_directories(self, tmp_path: Path) -> None:
        """Bare repos don't have a .git *directory*, they have git files directly."""
        bare_path = tmp_path / "bare.git"
        git.Repo.init(bare_path, bare=True)
        repos = _discover_repos([tmp_path])
        # bare repos don't have .git subdir, so they aren't discovered
        assert bare_path not in repos

    def test_skips_missing_root(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent"
        repos = _discover_repos([missing])
        assert repos == []

    def test_discovers_nested_repos(self, tmp_path: Path) -> None:
        _init_repo(tmp_path / "a")
        _init_repo(tmp_path / "b")
        (tmp_path / "not_a_repo").mkdir()
        repos = _discover_repos([tmp_path])
        names = {r.name for r in repos}
        assert "a" in names
        assert "b" in names
        assert "not_a_repo" not in names


# ── _matches_any ───────────────────────────────────────────────────


class TestMatchesAny:
    def test_matches_basename(self) -> None:
        assert _matches_any("README.md", ["README*"]) is True

    def test_matches_full_path_glob(self) -> None:
        # fnmatch doesn't support ** like glob; test with patterns that work
        assert _matches_any("docs/guide.md", ["docs/*.md"]) is True

    def test_matches_nested_by_basename(self) -> None:
        # _matches_any checks basename too, so *.md matches any .md file
        assert _matches_any("docs/sub/guide.md", ["*.md"]) is True

    def test_no_match(self) -> None:
        assert _matches_any("src/main.py", ["README*"]) is False


# ── _load_cursor / _save_cursor ───────────────────────────────────


class TestCursorPersistence:
    def test_load_missing_file(self, tmp_path: Path) -> None:
        assert _load_cursor(tmp_path / "missing.json") == {}

    def test_save_and_load(self, tmp_path: Path) -> None:
        p = tmp_path / "cursors.json"
        _save_cursor(p, {"repo1": "abc123"})
        loaded = _load_cursor(p)
        assert loaded == {"repo1": "abc123"}

    def test_load_corrupt_json(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text("not valid json{{{")
        assert _load_cursor(p) == {}

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        p = tmp_path / "sub" / "dir" / "cursors.json"
        _save_cursor(p, {"r": "sha"})
        assert p.exists()

    def test_save_sets_restrictive_permissions(self, tmp_path: Path) -> None:
        p = tmp_path / "cursors.json"
        _save_cursor(p, {"repo1": "abc123"})
        assert p.stat().st_mode & 0o777 == 0o600


# ── RepositorySource configure ────────────────────────────────────


class TestRepositorySourceConfigure:
    def test_requires_repo_roots(self) -> None:
        s = RepositorySource()
        with pytest.raises(ValueError, match="repo_roots"):
            s.configure({})

    def test_default_patterns(self, tmp_path: Path) -> None:
        s = RepositorySource()
        s.configure({"repo_roots": [str(tmp_path)]})
        assert s._include_patterns == DEFAULT_INCLUDE_PATTERNS
        assert s._exclude_patterns == DEFAULT_EXCLUDE_PATTERNS

    def test_custom_include_patterns(self, tmp_path: Path) -> None:
        s = RepositorySource()
        s.configure(
            {
                "repo_roots": [str(tmp_path)],
                "include_patterns": ["*.py"],
            }
        )
        assert s._include_patterns == ["*.py"]

    def test_custom_exclude_patterns(self, tmp_path: Path) -> None:
        s = RepositorySource()
        s.configure(
            {
                "repo_roots": [str(tmp_path)],
                "exclude_patterns": ["build/"],
            }
        )
        assert s._exclude_patterns == ["build/"]

    def test_poll_interval_config(self, tmp_path: Path) -> None:
        s = RepositorySource()
        s.configure(
            {
                "repo_roots": [str(tmp_path)],
                "poll_interval_seconds": 60,
            }
        )
        assert s._poll_interval == 60

    def test_name(self) -> None:
        assert RepositorySource().name() == "repositories"


# ── RepositorySource start (integration) ─────────────────────────


@pytest.mark.asyncio
async def test_initial_scan_emits_file_events(tmp_path: Path) -> None:
    """Initial scan should emit events for matching tracked files."""
    repo_path = tmp_path / "myrepo"
    _init_repo(repo_path, {"README.md": "# Test\n"})

    s = RepositorySource()
    cursor_file = tmp_path / "cursor.json"
    s.configure(
        {
            "repo_roots": [str(tmp_path)],
            "cursor_path": str(cursor_file),
            "poll_interval_seconds": 3600,
        }
    )

    q = _TestQueue()
    task = asyncio.create_task(s.start(q))
    events = await _collect_events(q)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # Should have file events and commit events
    file_events = [e for e in events if e["source_id"].startswith("repo:")]
    commit_events = [e for e in events if e["source_id"].startswith("commit:")]

    assert len(file_events) >= 1
    readme_ev = [e for e in file_events if "README.md" in e["source_id"]]
    assert len(readme_ev) == 1
    assert readme_ev[0]["source_type"] == "repositories"
    assert readme_ev[0]["operation"] == "created"
    assert "repo_name" in readme_ev[0]["meta"]
    assert "sha256" in readme_ev[0]["meta"]
    assert "size_bytes" in readme_ev[0]["meta"]

    assert len(commit_events) >= 1
    assert commit_events[0]["meta"]["author_email"] == "test@example.com"


@pytest.mark.asyncio
async def test_incremental_scan_only_new_changes(tmp_path: Path) -> None:
    """After initial scan, incremental scan should only emit diffs."""
    repo_path = tmp_path / "myrepo"
    repo = _init_repo(repo_path, {"README.md": "# v1\n"})

    cursor_file = tmp_path / "cursor.json"
    s = RepositorySource()
    s.configure(
        {
            "repo_roots": [str(tmp_path)],
            "cursor_path": str(cursor_file),
            "poll_interval_seconds": 3600,
        }
    )

    # First scan
    q = _TestQueue()
    task = asyncio.create_task(s.start(q))
    await _collect_events(q)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # Modify a file and add a new commit
    readme = repo_path / "README.md"
    readme.write_text("# v2\n")
    repo.index.add(["README.md"])
    repo.index.commit("Update readme")

    # Second scan (incremental)
    s2 = RepositorySource()
    s2.configure(
        {
            "repo_roots": [str(tmp_path)],
            "cursor_path": str(cursor_file),
            "poll_interval_seconds": 3600,
        }
    )

    q2 = _TestQueue()
    task2 = asyncio.create_task(s2.start(q2))
    events = await _collect_events(q2)
    task2.cancel()
    try:
        await task2
    except asyncio.CancelledError:
        pass

    file_events = [e for e in events if e["source_id"].startswith("repo:")]
    commit_events = [e for e in events if e["source_id"].startswith("commit:")]

    # Should have the modified README event
    readme_evs = [e for e in file_events if "README.md" in e["source_id"]]
    assert len(readme_evs) == 1
    assert readme_evs[0]["operation"] == "modified"

    # Should have 1 new commit (not the initial one)
    assert len(commit_events) == 1
    assert "Update readme" in commit_events[0]["text"]


@pytest.mark.asyncio
async def test_include_patterns_filter_files(tmp_path: Path) -> None:
    """Only files matching include patterns should emit events."""
    repo_path = tmp_path / "myrepo"
    _init_repo(
        repo_path,
        {
            "README.md": "# Hello\n",
            "src/main.py": "print('hi')\n",
        },
    )

    s = RepositorySource()
    s.configure(
        {
            "repo_roots": [str(tmp_path)],
            "cursor_path": str(tmp_path / "cursor.json"),
            "poll_interval_seconds": 3600,
            "include_patterns": ["README*"],  # Only READMEs
        }
    )

    q = _TestQueue()
    task = asyncio.create_task(s.start(q))
    events = await _collect_events(q)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    file_events = [e for e in events if e["source_id"].startswith("repo:")]
    paths = [e["meta"]["relative_path"] for e in file_events]
    assert any("README.md" in p for p in paths)
    assert not any("main.py" in p for p in paths)


@pytest.mark.asyncio
async def test_commit_event_structure(tmp_path: Path) -> None:
    """Commit events should contain sha, author info, and changed files."""
    repo_path = tmp_path / "myrepo"
    _init_repo(repo_path, {"README.md": "content\n"})

    s = RepositorySource()
    s.configure(
        {
            "repo_roots": [str(tmp_path)],
            "cursor_path": str(tmp_path / "cursor.json"),
            "poll_interval_seconds": 3600,
        }
    )

    q = _TestQueue()
    task = asyncio.create_task(s.start(q))
    events = await _collect_events(q)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    commit_events = [e for e in events if e["source_id"].startswith("commit:")]
    assert len(commit_events) >= 1

    ev = commit_events[0]
    assert ev["source_type"] == "repositories"
    assert ev["operation"] == "created"
    assert ev["mime_type"] == "text/plain"
    assert "text" in ev  # commit message
    meta = ev["meta"]
    assert "sha" in meta
    assert "author_name" in meta
    assert "author_email" in meta
    assert "date" in meta
    assert "repo_name" in meta
    assert "changed_files" in meta


@pytest.mark.asyncio
async def test_skips_bare_repo_during_scan(tmp_path: Path) -> None:
    """Bare repos should be skipped silently during scanning."""
    bare_path = tmp_path / "bare.git"
    git.Repo.init(bare_path, bare=True)

    # Also create a normal repo to verify the source works
    _init_repo(tmp_path / "normal", {"README.md": "hi\n"})

    s = RepositorySource()
    s.configure(
        {
            "repo_roots": [str(tmp_path)],
            "cursor_path": str(tmp_path / "cursor.json"),
            "poll_interval_seconds": 3600,
        }
    )

    q = _TestQueue()
    task = asyncio.create_task(s.start(q))
    events = await _collect_events(q)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # No events from the bare repo
    for ev in events:
        assert "bare.git" not in ev.get("source_id", "")


@pytest.mark.asyncio
async def test_handles_invalid_git_repo(tmp_path: Path) -> None:
    """Corrupted .git dirs should be skipped gracefully."""
    bad_repo = tmp_path / "bad"
    bad_repo.mkdir()
    (bad_repo / ".git").write_text("not a valid git dir")

    s = RepositorySource()
    s.configure(
        {
            "repo_roots": [str(tmp_path)],
            "cursor_path": str(tmp_path / "cursor.json"),
            "poll_interval_seconds": 3600,
        }
    )

    q = _TestQueue()
    task = asyncio.create_task(s.start(q))
    # Should not crash, just skip the bad repo
    events = await _collect_events(q, timeout=1.0)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # No events (the only repo was invalid)
    assert events == []
