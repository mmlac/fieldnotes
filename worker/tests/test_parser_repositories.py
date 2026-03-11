"""Tests for the repository parser: file events, commit events, graph hints."""

from __future__ import annotations

from typing import Any

import pytest

from worker.parsers.base import GraphHint, ParsedDocument
from worker.parsers.repositories import RepositoryParser


# ── helpers ────────────────────────────────────────────────────────


def _file_event(
    repo_path: str = "/repos/myproject",
    relative_path: str = "README.md",
    operation: str = "created",
    text: str = "# Hello",
    mime_type: str = "text/markdown",
    repo_name: str = "myproject",
    remote_url: str = "https://github.com/user/myproject",
    sha256: str = "abc123",
) -> dict[str, Any]:
    return {
        "id": "test-id",
        "source_type": "repositories",
        "source_id": f"repo:{repo_path}:{relative_path}",
        "operation": operation,
        "mime_type": mime_type,
        "text": text,
        "source_modified_at": "2026-01-01T00:00:00Z",
        "meta": {
            "repo_name": repo_name,
            "repo_path": repo_path,
            "remote_url": remote_url,
            "relative_path": relative_path,
            "sha256": sha256,
            "size_bytes": len(text),
        },
    }


def _commit_event(
    repo_path: str = "/repos/myproject",
    sha: str = "deadbeef",
    author_name: str = "Alice Dev",
    author_email: str = "alice@example.com",
    message: str = "feat: add widget",
    changed_files: list[str] | None = None,
    repo_name: str = "myproject",
    remote_url: str = "https://github.com/user/myproject",
) -> dict[str, Any]:
    if changed_files is None:
        changed_files = ["src/widget.py", "tests/test_widget.py"]
    return {
        "id": "test-commit-id",
        "source_type": "repositories",
        "source_id": f"commit:{repo_path}:{sha}",
        "operation": "created",
        "mime_type": "text/plain",
        "text": message,
        "source_modified_at": "2026-01-01T12:00:00Z",
        "meta": {
            "sha": sha,
            "author_name": author_name,
            "author_email": author_email,
            "date": "2026-01-01T12:00:00Z",
            "repo_name": repo_name,
            "repo_path": repo_path,
            "remote_url": remote_url,
            "changed_files": changed_files,
        },
    }


# ── RepositoryParser basics ───────────────────────────────────────


class TestRepositoryParserBasics:
    def test_source_type(self) -> None:
        assert RepositoryParser().source_type == "repositories"


# ── File event parsing ────────────────────────────────────────────


class TestFileEventParsing:
    def test_basic_file_event(self) -> None:
        parser = RepositoryParser()
        docs = parser.parse(_file_event())
        assert len(docs) == 1

        doc = docs[0]
        assert doc.source_type == "repositories"
        assert doc.source_id == "repo:/repos/myproject:README.md"
        assert doc.operation == "created"
        assert doc.text == "# Hello"
        assert doc.node_label == "File"
        assert doc.node_props["path"] == "README.md"
        assert doc.node_props["name"] == "README.md"
        assert doc.node_props["ext"] == ".md"
        assert doc.node_props["sha256"] == "abc123"

    def test_repository_contains_file_hint(self) -> None:
        parser = RepositoryParser()
        docs = parser.parse(_file_event())
        doc = docs[0]

        contains_hints = [h for h in doc.graph_hints if h.predicate == "CONTAINS"]
        assert len(contains_hints) == 1
        h = contains_hints[0]
        assert h.subject_label == "Repository"
        assert h.object_label == "File"
        assert h.subject_id == "/repos/myproject"
        assert h.subject_props["name"] == "myproject"
        assert h.subject_props["remote_url"] == "https://github.com/user/myproject"

    def test_delete_operation_forwarding(self) -> None:
        parser = RepositoryParser()
        docs = parser.parse(_file_event(operation="deleted"))
        assert len(docs) == 1
        doc = docs[0]
        assert doc.operation == "deleted"
        assert doc.text == ""

    def test_binary_file_skipped(self) -> None:
        """Non-text files with no text content should return empty list."""
        parser = RepositoryParser()
        ev = _file_event(text="", mime_type="application/octet-stream")
        docs = parser.parse(ev)
        assert docs == []

    def test_source_metadata_populated(self) -> None:
        parser = RepositoryParser()
        docs = parser.parse(_file_event())
        doc = docs[0]
        assert doc.source_metadata["repo_name"] == "myproject"
        assert doc.source_metadata["repo_path"] == "/repos/myproject"
        assert doc.source_metadata["relative_path"] == "README.md"


# ── Commit event parsing ─────────────────────────────────────────


class TestCommitEventParsing:
    def test_commit_produces_commit_node(self) -> None:
        parser = RepositoryParser()
        docs = parser.parse(_commit_event())
        assert len(docs) == 1

        doc = docs[0]
        assert doc.node_label == "Commit"
        assert doc.node_props["sha"] == "deadbeef"
        assert doc.node_props["message"] == "feat: add widget"
        assert doc.text == "feat: add widget"
        assert doc.operation == "created"

    def test_person_authored_hint(self) -> None:
        parser = RepositoryParser()
        docs = parser.parse(_commit_event())
        doc = docs[0]

        authored = [h for h in doc.graph_hints if h.predicate == "AUTHORED"]
        assert len(authored) == 1
        h = authored[0]
        assert h.subject_label == "Person"
        assert h.object_label == "Commit"
        assert h.subject_props["name"] == "Alice Dev"
        assert h.subject_props["email"] == "alice@example.com"
        assert h.subject_merge_key == "email"

    def test_person_email_normalized(self) -> None:
        parser = RepositoryParser()
        docs = parser.parse(_commit_event(author_email="  ALICE@Example.COM  "))
        doc = docs[0]

        authored = [h for h in doc.graph_hints if h.predicate == "AUTHORED"]
        assert authored[0].subject_props["email"] == "alice@example.com"
        assert authored[0].subject_id == "alice@example.com"

    def test_commit_part_of_repository(self) -> None:
        parser = RepositoryParser()
        docs = parser.parse(_commit_event())
        doc = docs[0]

        part_of = [h for h in doc.graph_hints if h.predicate == "PART_OF"]
        assert len(part_of) == 1
        h = part_of[0]
        assert h.subject_label == "Commit"
        assert h.object_label == "Repository"
        assert h.object_props["name"] == "myproject"

    def test_commit_modified_files(self) -> None:
        parser = RepositoryParser()
        docs = parser.parse(
            _commit_event(changed_files=["a.py", "b.py", "c.py"])
        )
        doc = docs[0]

        modified = [h for h in doc.graph_hints if h.predicate == "MODIFIED"]
        assert len(modified) == 3
        object_ids = {h.object_id for h in modified}
        assert "repo:/repos/myproject:a.py" in object_ids
        assert "repo:/repos/myproject:b.py" in object_ids

    def test_no_author_email_skips_authored_hint(self) -> None:
        parser = RepositoryParser()
        docs = parser.parse(_commit_event(author_email=""))
        doc = docs[0]

        authored = [h for h in doc.graph_hints if h.predicate == "AUTHORED"]
        assert len(authored) == 0

    def test_commit_source_metadata(self) -> None:
        parser = RepositoryParser()
        docs = parser.parse(_commit_event())
        doc = docs[0]
        assert doc.source_metadata["sha"] == "deadbeef"
        assert doc.source_metadata["repo_name"] == "myproject"
