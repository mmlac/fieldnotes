"""Git repository source adapter with polling-based discovery.

Scans configured directory roots for git repositories, enumerates tracked
files matching configurable include/exclude patterns, and emits one
IngestEvent per matching file.  Also extracts recent git commits, emitting
one IngestEvent per commit with author metadata and changed file lists.
Uses per-repo HEAD commit SHA as cursor for incremental updates.

Config section ``[sources.repositories]``::

    repo_roots = ["~/projects", "~/work"]
    poll_interval_seconds = 300
    include_patterns = ["README*", "CHANGELOG*", "CONTRIBUTING*",
                        "docs/**/*.md", "*.toml", "ADR/**/*.md"]
    exclude_patterns = ["node_modules/", ".git/", "vendor/",
                        "target/", "__pycache__/"]
    cursor_path = "~/.fieldnotes/data/repo_cursor.json"
    max_file_size = 104857600
    max_commits = 200
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import tempfile
import uuid
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import git

from worker.metrics import (
    SOURCE_WATCHER_EVENTS,
    WATCHER_ACTIVE,
    WATCHER_LAST_EVENT_TIMESTAMP,
)

from ._handler import guess_mime, streaming_sha256
from .base import PythonSource

logger = logging.getLogger(__name__)

DEFAULT_POLL_INTERVAL = 300  # seconds
DEFAULT_MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MiB
DEFAULT_MAX_COMMITS = 200
DEFAULT_CURSOR_PATH = Path.home() / ".fieldnotes" / "data" / "repo_cursor.json"

DEFAULT_INCLUDE_PATTERNS: list[str] = [
    "README*",
    "CHANGELOG*",
    "CONTRIBUTING*",
    "docs/**/*.md",
    "*.toml",
    "ADR/**/*.md",
    "*.csproj",
    "*.fsproj",
    "*.vbproj",
    "Directory.Packages.props",
    "packages.config",
]

DEFAULT_EXCLUDE_PATTERNS: list[str] = [
    "node_modules/",
    ".git/",
    "vendor/",
    "target/",
    "__pycache__/",
]


def _load_cursor(path: Path) -> dict[str, str]:
    """Load per-repo HEAD cursors from disk.

    Returns a mapping of ``repo_path -> commit_sha``.
    """
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        logger.warning("Failed to read repo cursor file %s, starting fresh", path)
        return {}


def _save_cursor(path: Path, cursors: dict[str, str]) -> None:
    """Persist per-repo HEAD cursors to disk atomically.

    Writes to a temporary file in the same directory, then renames it
    into place.  On POSIX systems ``os.replace`` is atomic, so a crash
    mid-write can never leave a partially-written cursor file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=".cursor_",
        suffix=".tmp",
        delete=False,
    )
    try:
        fd.write(json.dumps(cursors))
        fd.flush()
        os.fsync(fd.fileno())
        fd.close()
        os.chmod(fd.name, 0o600)
        os.replace(fd.name, path)
    except BaseException:
        fd.close()
        with contextlib.suppress(OSError):
            os.unlink(fd.name)
        raise


def _matches_any(rel_path: str, patterns: list[str]) -> bool:
    """Check if *rel_path* matches any of the glob *patterns*."""
    name = Path(rel_path).name
    for pat in patterns:
        if fnmatch(rel_path, pat) or fnmatch(name, pat):
            return True
    return False


def _get_remote_url(repo: git.Repo) -> str | None:
    """Extract the origin remote URL, or None if no remotes."""
    try:
        return repo.remotes.origin.url
    except (AttributeError, ValueError):
        return None


def _discover_repos(roots: list[Path]) -> list[Path]:
    """Find git repositories under *roots* (non-bare only)."""
    repos: list[Path] = []
    for root in roots:
        if not root.is_dir():
            logger.warning("repo_root %s is not a directory, skipping", root)
            continue
        # Check if root itself is a repo
        if (root / ".git").exists():
            repos.append(root)
        # Walk one level of subdirectories for repos
        for child in sorted(root.iterdir()):
            if child.is_dir() and (child / ".git").exists():
                repos.append(child)
    return repos


def _build_event(
    file_path: Path,
    repo_path: Path,
    repo_name: str,
    remote_url: str | None,
    operation: str,
    max_file_size: int,
) -> dict[str, Any] | None:
    """Build an IngestEvent dict for a single repository file."""
    rel_path = str(file_path.relative_to(repo_path))
    now = datetime.now(timezone.utc).isoformat()

    meta: dict[str, Any] = {
        "repo_name": repo_name,
        "repo_path": str(repo_path),
        "remote_url": remote_url,
        "relative_path": rel_path,
    }

    event: dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "source_type": "repositories",
        "source_id": f"repo:{repo_path}:{rel_path}",
        "operation": operation,
        "mime_type": guess_mime(str(file_path)),
        "meta": meta,
        "enqueued_at": now,
    }

    if file_path.is_file():
        try:
            stat = file_path.stat()
            event["source_modified_at"] = datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).isoformat()
            result = streaming_sha256(file_path, max_file_size)
            if result is None:
                logger.warning(
                    "Skipping %s — exceeds max_file_size (%d bytes)",
                    file_path,
                    max_file_size,
                )
                return None
            digest, size = result
            meta["sha256"] = digest
            meta["size_bytes"] = size

            if event["mime_type"].startswith("text/"):
                try:
                    event["text"] = file_path.read_text(
                        encoding="utf-8", errors="replace"
                    )
                except OSError:
                    logger.warning("Failed to read text content from %s", file_path)
                    event["text"] = ""
        except OSError:
            logger.warning("Failed to stat %s, emitting without content hash", file_path)
            event["source_modified_at"] = now
    else:
        event["source_modified_at"] = now

    return event


def _build_commit_event(
    commit: git.Commit,
    repo_path: Path,
    repo_name: str,
    remote_url: str | None,
    changed_files: list[str],
) -> dict[str, Any]:
    """Build an IngestEvent dict for a single git commit."""
    sha = commit.hexsha
    author_date = datetime.fromtimestamp(
        commit.authored_date, tz=timezone.utc
    ).isoformat()

    meta: dict[str, Any] = {
        "sha": sha,
        "author_name": commit.author.name or "",
        "author_email": commit.author.email or "",
        "date": author_date,
        "repo_name": repo_name,
        "repo_path": str(repo_path),
        "remote_url": remote_url,
        "changed_files": changed_files,
    }

    return {
        "id": str(uuid.uuid4()),
        "source_type": "repositories",
        "source_id": f"commit:{repo_path}:{sha}",
        "operation": "created",
        "mime_type": "text/plain",
        "text": commit.message,
        "meta": meta,
        "source_modified_at": author_date,
        "enqueued_at": datetime.now(timezone.utc).isoformat(),
    }


class RepositorySource(PythonSource):
    """Polls git repositories for doc files and commits, emits IngestEvent dicts.

    Config keys (from ``[sources.repositories]``):
        repo_roots: list[str]             — directories to scan for repos (required)
        poll_interval_seconds: int        — polling interval (default: 300)
        include_patterns: list[str]       — file glob patterns to include
        exclude_patterns: list[str]       — file glob patterns to exclude
        cursor_path: str                  — cursor persistence file (optional)
        max_file_size: int                — max file size in bytes (default: 100 MiB)
        max_commits: int                  — max commits to index per repo (default: 200)
    """

    def __init__(self) -> None:
        self._repo_roots: list[Path] = []
        self._poll_interval: int = DEFAULT_POLL_INTERVAL
        self._include_patterns: list[str] = list(DEFAULT_INCLUDE_PATTERNS)
        self._exclude_patterns: list[str] = list(DEFAULT_EXCLUDE_PATTERNS)
        self._cursor_path: Path = DEFAULT_CURSOR_PATH
        self._max_file_size: int = DEFAULT_MAX_FILE_SIZE
        self._max_commits: int = DEFAULT_MAX_COMMITS

    def name(self) -> str:
        return "repositories"

    def configure(self, cfg: dict[str, Any]) -> None:
        roots = cfg.get("repo_roots")
        if not roots:
            raise ValueError(
                "RepositorySource requires 'repo_roots' in config"
            )
        self._repo_roots = [Path(r).expanduser().resolve() for r in roots]

        self._poll_interval = int(
            cfg.get("poll_interval_seconds", DEFAULT_POLL_INTERVAL)
        )
        if "include_patterns" in cfg:
            self._include_patterns = list(cfg["include_patterns"])
        if "exclude_patterns" in cfg:
            self._exclude_patterns = list(cfg["exclude_patterns"])

        cursor = cfg.get("cursor_path")
        if cursor:
            self._cursor_path = Path(cursor).expanduser().resolve()

        self._max_file_size = int(
            cfg.get("max_file_size", DEFAULT_MAX_FILE_SIZE)
        )
        self._max_commits = int(
            cfg.get("max_commits", DEFAULT_MAX_COMMITS)
        )

    async def start(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        cursors = _load_cursor(self._cursor_path)

        # Initial scan if we have no cursors at all
        if not cursors:
            logger.info("No repo cursors found — performing initial scan")

        WATCHER_ACTIVE.labels(source_type="repositories").set(1)
        try:
            while True:
                repos = _discover_repos(self._repo_roots)
                for repo_path in repos:
                    await self._scan_repo(repo_path, cursors, queue)
                _save_cursor(self._cursor_path, cursors)
                await asyncio.sleep(self._poll_interval)
        except asyncio.CancelledError:
            WATCHER_ACTIVE.labels(source_type="repositories").set(0)
            raise

    async def _scan_repo(
        self,
        repo_path: Path,
        cursors: dict[str, str],
        queue: asyncio.Queue[dict[str, Any]],
    ) -> None:
        """Scan a single repo, emitting events for new/changed files."""
        repo_key = str(repo_path)
        try:
            repo = git.Repo(repo_path)
        except git.InvalidGitRepositoryError:
            logger.warning("Invalid git repo at %s, skipping", repo_path)
            return

        # Skip bare repos
        if repo.bare:
            logger.debug("Skipping bare repo %s", repo_path)
            return

        try:
            head_sha = repo.head.commit.hexsha
        except ValueError:
            # Empty repo with no commits
            logger.debug("Skipping empty repo %s (no commits)", repo_path)
            return

        prev_sha = cursors.get(repo_key)
        if prev_sha == head_sha:
            # No changes since last poll
            return

        repo_name = repo_path.name
        remote_url = _get_remote_url(repo)

        if prev_sha is None:
            # Initial scan: index all matching tracked files
            await self._scan_all_files(repo, repo_path, repo_name, remote_url, queue)
        else:
            # Incremental: find files changed between prev_sha and HEAD
            await self._scan_diff(
                repo, repo_path, repo_name, remote_url, prev_sha, head_sha, queue
            )

        # Scan commits (initial: last N; incremental: since prev cursor)
        await self._scan_commits(
            repo, repo_path, repo_name, remote_url, prev_sha, queue
        )

        cursors[repo_key] = head_sha

    async def _scan_all_files(
        self,
        repo: git.Repo,
        repo_path: Path,
        repo_name: str,
        remote_url: str | None,
        queue: asyncio.Queue[dict[str, Any]],
    ) -> None:
        """Index all tracked files matching include/exclude patterns."""
        count = 0
        loop = asyncio.get_running_loop()
        tracked = await loop.run_in_executor(
            None, lambda: [item.path for item in repo.head.commit.tree.traverse()]
        )

        for rel_path in tracked:
            if not self._matches_include(rel_path):
                continue
            if self._matches_exclude(rel_path):
                continue

            file_path = repo_path / rel_path
            if not file_path.is_file():
                continue

            event = _build_event(
                file_path, repo_path, repo_name, remote_url,
                "created", self._max_file_size,
            )
            if event is not None:
                await queue.put(event)
                SOURCE_WATCHER_EVENTS.labels(
                    source_type="repositories", event_type="created",
                ).inc()
                WATCHER_LAST_EVENT_TIMESTAMP.labels(
                    source_type="repositories",
                ).set_to_current_time()
                count += 1

        logger.info(
            "Initial scan of %s: %d matching files indexed", repo_name, count
        )

    async def _scan_diff(
        self,
        repo: git.Repo,
        repo_path: Path,
        repo_name: str,
        remote_url: str | None,
        prev_sha: str,
        head_sha: str,
        queue: asyncio.Queue[dict[str, Any]],
    ) -> None:
        """Emit events for files changed between two commits."""
        count = 0
        loop = asyncio.get_running_loop()

        try:
            diffs = await loop.run_in_executor(
                None,
                lambda: repo.commit(prev_sha).diff(head_sha),
            )
        except (git.BadName, ValueError):
            # Previous cursor commit no longer exists (force-push, etc.)
            # Fall back to full scan
            logger.warning(
                "Previous cursor %s not found in %s, doing full re-scan",
                prev_sha, repo_name,
            )
            await self._scan_all_files(repo, repo_path, repo_name, remote_url, queue)
            return

        for diff in diffs:
            # Determine the relevant path and operation
            if diff.deleted_file:
                rel_path = diff.a_path
                operation = "deleted"
            elif diff.new_file:
                rel_path = diff.b_path
                operation = "created"
            else:
                rel_path = diff.b_path or diff.a_path
                operation = "modified"

            if not self._matches_include(rel_path):
                continue
            if self._matches_exclude(rel_path):
                continue

            file_path = repo_path / rel_path
            event = _build_event(
                file_path, repo_path, repo_name, remote_url,
                operation, self._max_file_size,
            )
            if event is not None:
                await queue.put(event)
                SOURCE_WATCHER_EVENTS.labels(
                    source_type="repositories", event_type=operation,
                ).inc()
                WATCHER_LAST_EVENT_TIMESTAMP.labels(
                    source_type="repositories",
                ).set_to_current_time()
                count += 1

        if count:
            logger.info(
                "Incremental scan of %s (%s..%s): %d file(s) changed",
                repo_name, prev_sha[:8], head_sha[:8], count,
            )

    async def _scan_commits(
        self,
        repo: git.Repo,
        repo_path: Path,
        repo_name: str,
        remote_url: str | None,
        prev_sha: str | None,
        queue: asyncio.Queue[dict[str, Any]],
    ) -> None:
        """Emit events for recent git commits."""
        loop = asyncio.get_running_loop()

        try:
            if prev_sha is not None:
                # Incremental: commits between cursor and HEAD
                rev = f"{prev_sha}..HEAD"
            else:
                # Initial scan: last N commits
                rev = "HEAD"

            commits = await loop.run_in_executor(
                None,
                lambda: list(
                    repo.iter_commits(rev, max_count=self._max_commits)
                ),
            )
        except (git.GitCommandError, ValueError) as exc:
            logger.warning(
                "Failed to enumerate commits in %s: %s", repo_name, exc
            )
            return

        count = 0
        for commit_obj in commits:
            # Get changed file paths from the commit's diff against its parent
            try:
                changed = await loop.run_in_executor(
                    None,
                    lambda c=commit_obj: [
                        d.b_path or d.a_path
                        for d in (
                            c.diff(c.parents[0]) if c.parents else c.diff(git.NULL_TREE)
                        )
                    ],
                )
            except (git.GitCommandError, ValueError):
                changed = []

            event = _build_commit_event(
                commit_obj, repo_path, repo_name, remote_url, changed
            )
            await queue.put(event)
            SOURCE_WATCHER_EVENTS.labels(
                source_type="repositories", event_type="created",
            ).inc()
            WATCHER_LAST_EVENT_TIMESTAMP.labels(
                source_type="repositories",
            ).set_to_current_time()
            count += 1

        if count:
            logger.info(
                "Indexed %d commit(s) from %s%s",
                count,
                repo_name,
                f" ({prev_sha[:8]}..HEAD)" if prev_sha else " (initial)",
            )

    def _matches_include(self, rel_path: str) -> bool:
        """Check if rel_path matches any include pattern."""
        return _matches_any(rel_path, self._include_patterns)

    def _matches_exclude(self, rel_path: str) -> bool:
        """Check if rel_path matches any exclude pattern."""
        return _matches_any(rel_path, self._exclude_patterns)
