"""Tests for cross-source Person entity resolution via email dedup."""

from __future__ import annotations

from typing import Any

import pytest

from worker.pipeline.resolver import (
    PersonResolutionResult,
    _normalize_email,
    _prefer_longer_name,
    resolve_persons,
)


# ── helpers ────────────────────────────────────────────────────────


def _person(
    email: str,
    name: str = "",
    source_type: str = "",
    source_id: str = "",
) -> dict[str, Any]:
    d: dict[str, Any] = {"email": email, "name": name}
    if source_type:
        d["source_type"] = source_type
    if source_id:
        d["source_id"] = source_id
    return d


# ── _normalize_email ──────────────────────────────────────────────


class TestNormalizeEmail:
    def test_lowercase(self) -> None:
        assert _normalize_email("Alice@Example.COM") == "alice@example.com"

    def test_strip_whitespace(self) -> None:
        assert _normalize_email("  bob@test.com  ") == "bob@test.com"

    def test_already_normalized(self) -> None:
        assert _normalize_email("user@host.com") == "user@host.com"


# ── _prefer_longer_name ──────────────────────────────────────────


class TestPreferLongerName:
    def test_longer_wins(self) -> None:
        assert _prefer_longer_name("A", "Alice") == "Alice"

    def test_existing_longer(self) -> None:
        assert _prefer_longer_name("Alice Wonderland", "Alice") == "Alice Wonderland"

    def test_empty_existing(self) -> None:
        assert _prefer_longer_name("", "Bob") == "Bob"

    def test_empty_incoming(self) -> None:
        assert _prefer_longer_name("Alice", "") == "Alice"

    def test_equal_length(self) -> None:
        # Existing wins on tie
        assert _prefer_longer_name("Abcd", "Efgh") == "Abcd"


# ── resolve_persons: basic merge ──────────────────────────────────


class TestResolvePersonsMerge:
    def test_single_person_passthrough(self) -> None:
        result = resolve_persons([
            _person("alice@example.com", "Alice"),
        ])
        assert "alice@example.com" in result.merged_persons
        p = result.merged_persons["alice@example.com"]
        assert p["name"] == "Alice"
        assert p["email"] == "alice@example.com"

    def test_merge_by_email_same_source(self) -> None:
        result = resolve_persons([
            _person("alice@example.com", "A"),
            _person("alice@example.com", "Alice Developer"),
        ])
        assert len(result.merged_persons) == 1
        p = result.merged_persons["alice@example.com"]
        assert p["name"] == "Alice Developer"  # longer name wins

    def test_merge_across_git_and_gmail(self) -> None:
        result = resolve_persons([
            _person("alice@example.com", "alice", source_type="git"),
            _person("alice@example.com", "Alice Smith", source_type="gmail"),
        ])
        assert len(result.merged_persons) == 1
        p = result.merged_persons["alice@example.com"]
        assert p["name"] == "Alice Smith"
        assert "git" in p["source_types"]
        assert "gmail" in p["source_types"]

    def test_different_emails_stay_separate(self) -> None:
        result = resolve_persons([
            _person("alice@example.com", "Alice"),
            _person("bob@example.com", "Bob"),
        ])
        assert len(result.merged_persons) == 2
        assert "alice@example.com" in result.merged_persons
        assert "bob@example.com" in result.merged_persons


# ── resolve_persons: case-insensitive email matching ──────────────


class TestCaseInsensitiveEmail:
    def test_different_case_emails_merge(self) -> None:
        result = resolve_persons([
            _person("Alice@Example.COM", "A", source_type="git"),
            _person("alice@example.com", "Alice Smith", source_type="gmail"),
        ])
        assert len(result.merged_persons) == 1
        p = result.merged_persons["alice@example.com"]
        assert p["name"] == "Alice Smith"

    def test_mixed_case_with_whitespace(self) -> None:
        result = resolve_persons([
            _person("  BOB@Test.Com  ", "Bob"),
            _person("bob@test.com", "Robert"),
        ])
        assert len(result.merged_persons) == 1
        assert result.merged_persons["bob@test.com"]["name"] == "Robert"


# ── resolve_persons: name preference ─────────────────────────────


class TestNamePreference:
    def test_longer_name_wins(self) -> None:
        result = resolve_persons([
            _person("dev@co.com", "J"),
            _person("dev@co.com", "Jane"),
            _person("dev@co.com", "Jane Developer"),
        ])
        p = result.merged_persons["dev@co.com"]
        assert p["name"] == "Jane Developer"

    def test_first_non_empty_name_used_if_all_same_length(self) -> None:
        result = resolve_persons([
            _person("x@y.com", "Abcd"),
            _person("x@y.com", "Efgh"),
        ])
        # Existing wins on tie (first one processed becomes existing)
        p = result.merged_persons["x@y.com"]
        assert p["name"] in ("Abcd", "Efgh")


# ── resolve_persons: SAME_AS edges ───────────────────────────────


class TestSameAsEdges:
    def test_cross_source_emits_same_as(self) -> None:
        result = resolve_persons([
            _person("alice@example.com", "A", source_type="git", source_id="git:alice"),
            _person("alice@example.com", "Alice", source_type="gmail", source_id="gmail:alice"),
        ])
        assert len(result.same_as_edges) == 1
        edge = result.same_as_edges[0]
        assert "git:alice" in edge
        assert "gmail:alice" in edge

    def test_single_source_no_same_as(self) -> None:
        result = resolve_persons([
            _person("alice@example.com", "Alice", source_type="git", source_id="git:alice"),
        ])
        assert result.same_as_edges == []

    def test_three_sources_emit_pairwise_edges(self) -> None:
        result = resolve_persons([
            _person("x@y.com", "X", source_type="git", source_id="s1"),
            _person("x@y.com", "X", source_type="gmail", source_id="s2"),
            _person("x@y.com", "X", source_type="slack", source_id="s3"),
        ])
        # canonical (s1) paired with s2 and s3
        assert len(result.same_as_edges) == 2


# ── resolve_persons: edge cases ──────────────────────────────────


class TestResolvePersonsEdgeCases:
    def test_empty_input(self) -> None:
        result = resolve_persons([])
        assert result.merged_persons == {}
        assert result.same_as_edges == []

    def test_empty_email_skipped(self) -> None:
        result = resolve_persons([
            _person("", "Ghost"),
            _person("real@example.com", "Real"),
        ])
        assert len(result.merged_persons) == 1
        assert "real@example.com" in result.merged_persons

    def test_no_source_id_no_same_as_edges(self) -> None:
        """Without source_id, same_as edges aren't emitted even for multi-source."""
        result = resolve_persons([
            _person("a@b.com", "A", source_type="git"),
            _person("a@b.com", "A", source_type="gmail"),
        ])
        # Nodes exist, source_types merged
        assert "git" in result.merged_persons["a@b.com"]["source_types"]
        assert "gmail" in result.merged_persons["a@b.com"]["source_types"]
        # But no source_id → no edges
        assert result.same_as_edges == []

    def test_retroactive_reconciliation_scenario(self) -> None:
        """Simulate git persons arriving first, then gmail persons later.

        Both batches should merge correctly when processed together (as
        resolve_persons sees the full set).
        """
        git_persons = [
            _person("dev@company.com", "jsmith", source_type="git", source_id="git:jsmith"),
            _person("alice@company.com", "alice", source_type="git", source_id="git:alice"),
        ]
        gmail_persons = [
            _person("dev@company.com", "John Smith", source_type="gmail", source_id="gmail:john"),
            _person("alice@company.com", "Alice Johnson", source_type="gmail", source_id="gmail:alice"),
        ]
        result = resolve_persons(git_persons + gmail_persons)

        assert len(result.merged_persons) == 2
        assert result.merged_persons["dev@company.com"]["name"] == "John Smith"
        assert result.merged_persons["alice@company.com"]["name"] == "Alice Johnson"
        assert len(result.same_as_edges) == 2
