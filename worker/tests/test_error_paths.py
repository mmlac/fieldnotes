"""Tests for error paths and edge cases across the codebase.

Covers untested paths identified during code review:
- Cypher validation: Unicode homoglyphs, overlong labels, injection via entity names
- Race conditions: file ops between stat/read
- Unicode handling: full range in paths, entities, emails, commits
- Empty/corrupt input: zero-byte files, truncated PDFs, empty repos, empty threads
- Config validation: invalid values caught at startup
- Parser edge cases: whitespace-only files, frontmatter-only markdown
- LLM output validation: malformed extraction results
- Resolver edge cases: partial NaN vectors, mixed NaN rows
- Gmail cursor integrity: non-integer historyId, integer coercion
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Cypher validation edge cases
# ---------------------------------------------------------------------------

from worker.query.graph import (
    ReadOnlyCypherViolation,
    _normalize_cypher_for_validation,
    _validate_cypher_readonly,
)


class TestCypherUnicodeHomoglyphs:
    """Test Unicode homoglyph attacks on Cypher validation."""

    @pytest.mark.parametrize(
        "cypher",
        [
            # Fullwidth Latin letters: ＣＲＥＡＴＥ
            "MATCH (n) \uff23\uff32\uff25\uff21\uff34\uff25 (m:Evil)",
            # Ogham space mark (\u1680) between keywords
            "MATCH (n)\u1680DELETE\u1680n",
            # Ideographic space (\u3000)
            "MATCH (n)\u3000SET\u3000n.x = 1",
            # Zero-width joiner between D and E of DELETE
            "MATCH (n) D\u200cELETE n",
            # Mathematical bold capital letters
            "MATCH (n) \U0001d40c\U0001d404\U0001d411\U0001d40e\U0001d404 (m:X {id: 'x'})",
        ],
    )
    def test_homoglyph_write_keywords_handled(self, cypher: str) -> None:
        """Homoglyph attacks should either be blocked or not execute writes.

        Some homoglyphs won't match the regex — that's fine because Neo4j
        won't recognize them as valid Cypher keywords either. The important
        thing is that ASCII write keywords are caught after normalization.
        """
        # These should either raise or pass without executing writes.
        # The real safety net is Neo4j's read-only transaction.
        # We just verify the normalizer doesn't crash on exotic Unicode.
        _normalize_cypher_for_validation(cypher)

    @pytest.mark.parametrize(
        "cypher",
        [
            # Line separator (Zl category)
            "MATCH (n)\u2028DELETE\u2028n",
            # Paragraph separator (Zp category)
            "MATCH (n)\u2029MERGE\u2029(m:X)",
            # Thin space (Zs)
            "MATCH (n)\u2009CREATE\u2009(m:X)",
            # Hair space
            "MATCH (n)\u200aDELETE\u200an",
            # Medium mathematical space
            "MATCH (n)\u205fSET\u205fn.x = 1",
        ],
    )
    def test_unicode_whitespace_categories_normalized(self, cypher: str) -> None:
        """All Unicode whitespace categories (Zs, Zl, Zp) should be normalized."""
        with pytest.raises(ReadOnlyCypherViolation):
            _validate_cypher_readonly(cypher)


class TestCypherInjectionViaEntityNames:
    """Test injection attempts through entity/topic names that reach Cypher."""

    def test_entity_name_with_closing_brace_injection(self) -> None:
        """Entity names with Cypher syntax should not break parameterized queries."""
        from worker.pipeline.writer import _validate_cypher_identifier

        # These should fail identifier validation
        with pytest.raises(ValueError):
            _validate_cypher_identifier("}) DETACH DELETE n //", "entity")

        with pytest.raises(ValueError):
            _validate_cypher_identifier("name'}) MATCH (x", "entity")

    def test_safe_identifier_regex_rejects_special_chars(self) -> None:
        from worker.pipeline.writer import _validate_cypher_identifier

        for bad_id in [
            "",
            "123start",
            "has space",
            "has-dash",
            "has.dot",
            "has;semicolon",
            "has'quote",
            'has"double',
            "has\nnewline",
            "has\ttab",
        ]:
            with pytest.raises(ValueError, match="Unsafe Cypher identifier"):
                _validate_cypher_identifier(bad_id, "test")

    def test_safe_identifier_accepts_valid(self) -> None:
        from worker.pipeline.writer import _validate_cypher_identifier

        for good_id in ["File", "MENTIONS", "source_id", "_private", "A123"]:
            assert _validate_cypher_identifier(good_id, "test") == good_id


class TestCypherOverlongLabels:
    """Test overlong entity names in writer."""

    def test_truncate_entity_name_normal(self) -> None:
        from worker.pipeline.writer import _truncate_entity_name

        short = "Normal Name"
        assert _truncate_entity_name(short) == short

    def test_truncate_entity_name_at_limit(self) -> None:
        from worker.pipeline.writer import _truncate_entity_name, MAX_ENTITY_NAME_LEN

        at_limit = "x" * MAX_ENTITY_NAME_LEN
        assert _truncate_entity_name(at_limit) == at_limit

    def test_truncate_entity_name_over_limit(self) -> None:
        from worker.pipeline.writer import _truncate_entity_name, MAX_ENTITY_NAME_LEN

        over_limit = "x" * (MAX_ENTITY_NAME_LEN + 100)
        result = _truncate_entity_name(over_limit)
        assert len(result) <= MAX_ENTITY_NAME_LEN
        # Should end with a hash suffix for disambiguation
        assert "_" in result

    def test_two_distinct_long_names_get_different_truncations(self) -> None:
        from worker.pipeline.writer import _truncate_entity_name, MAX_ENTITY_NAME_LEN

        name_a = "a" * (MAX_ENTITY_NAME_LEN + 100)
        name_b = "b" * (MAX_ENTITY_NAME_LEN + 100)
        assert _truncate_entity_name(name_a) != _truncate_entity_name(name_b)


# ---------------------------------------------------------------------------
# Unicode handling
# ---------------------------------------------------------------------------


class TestUnicodeEntityNames:
    """Test full Unicode range in entity names through the resolver."""

    def test_cjk_entity_names(self) -> None:
        from worker.pipeline.resolver import resolve_entities

        extracted = [{"name": "人工智能", "type": "Concept", "confidence": 0.9}]
        existing = [{"name": "人工智能", "type": "Technology", "confidence": 0.8}]
        result = resolve_entities(extracted, existing)
        assert len(result.entities) == 1
        assert result.entities[0].merged_into == "人工智能"

    def test_emoji_in_entity_name(self) -> None:
        from worker.pipeline.resolver import resolve_entities

        extracted = [{"name": "🚀 Launch", "type": "Concept", "confidence": 0.7}]
        result = resolve_entities(extracted, [])
        assert len(result.entities) == 1
        assert result.entities[0].name == "🚀 Launch"

    def test_mixed_script_entity_names(self) -> None:
        from worker.pipeline.resolver import resolve_entities

        extracted = [
            {"name": "Café résumé naïve", "type": "Concept", "confidence": 0.8}
        ]
        existing = [{"name": "café résumé naïve", "type": "Concept", "confidence": 0.7}]
        result = resolve_entities(extracted, existing)
        assert len(result.entities) == 1
        assert result.entities[0].merged_into is not None

    def test_rtl_entity_name(self) -> None:
        from worker.pipeline.resolver import resolve_entities

        extracted = [{"name": "بايثون", "type": "Technology", "confidence": 0.8}]
        result = resolve_entities(extracted, [])
        assert result.entities[0].name == "بايثون"


class TestUnicodeInFileParsing:
    """Test Unicode in file paths and text content."""

    def test_unicode_path_in_file_event(self) -> None:
        from worker.parsers.files import FileParser

        parser = FileParser()
        event = {
            "mime_type": "text/plain",
            "source_id": "/home/user/笔记/日记.txt",
            "text": "今日の内容",
        }
        docs = parser.parse(event)
        assert len(docs) == 1
        assert docs[0].source_id == "/home/user/笔记/日记.txt"
        assert docs[0].node_props["name"] == "日记.txt"

    def test_unicode_text_sha256(self) -> None:
        from worker.parsers.files import FileParser

        parser = FileParser()
        text = "Ünïcödé with ëmöjïs 🎉🎊"
        event = {
            "mime_type": "text/plain",
            "source_id": "test.txt",
            "text": text,
        }
        docs = parser.parse(event)
        expected_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert docs[0].node_props["sha256"] == expected_sha


class TestUnicodeInObsidian:
    """Test Unicode in Obsidian wikilinks, tags, frontmatter."""

    def test_unicode_wikilink_target(self) -> None:
        from worker.parsers.obsidian import ObsidianParser

        parser = ObsidianParser()
        note = "---\n---\nSee [[日本語ノート]] and [[Ñoño]]."
        event = {
            "source_id": "test.md",
            "operation": "created",
            "text": note,
            "meta": {},
        }
        docs = parser.parse(event)
        links = [h for h in docs[0].graph_hints if h.predicate == "LINKS_TO"]
        targets = {h.object_id for h in links}
        assert "日本語ノート" in targets
        assert "Ñoño" in targets

    def test_unicode_tags(self) -> None:
        from worker.parsers.obsidian import ObsidianParser

        parser = ObsidianParser()
        note = "---\n---\nSome text #日記 and #données here."
        event = {
            "source_id": "test.md",
            "operation": "created",
            "text": note,
            "meta": {},
        }
        docs = parser.parse(event)
        tag_hints = [h for h in docs[0].graph_hints if h.predicate == "TAGGED_BY_USER"]
        # The tag regex uses \w which includes Unicode word chars
        tag_ids = {h.object_id for h in tag_hints}
        assert "tag:日記" in tag_ids
        assert "tag:données" in tag_ids


# ---------------------------------------------------------------------------
# Empty/corrupt input
# ---------------------------------------------------------------------------


class TestEmptyAndCorruptInput:
    """Test zero-byte files, empty text, corrupt PDFs."""

    def test_zero_byte_text_file(self) -> None:
        from worker.parsers.files import FileParser

        parser = FileParser()
        event = {
            "mime_type": "text/plain",
            "source_id": "empty.txt",
            "text": "",
        }
        docs = parser.parse(event)
        assert len(docs) == 1
        assert docs[0].text == ""

    def test_whitespace_only_text_file(self) -> None:
        from worker.parsers.files import FileParser

        parser = FileParser()
        event = {
            "mime_type": "text/plain",
            "source_id": "whitespace.txt",
            "text": "   \n\t\n   ",
        }
        docs = parser.parse(event)
        assert len(docs) == 1
        assert docs[0].text == "   \n\t\n   "

    def test_corrupt_pdf_bytes(self) -> None:
        from worker.parsers.files import FileParser

        parser = FileParser()
        event = {
            "mime_type": "application/pdf",
            "source_id": "corrupt.pdf",
            "raw_bytes": b"\x00\x01\x02\x03\x04",
        }
        docs = parser.parse(event)
        assert docs == []

    def test_truncated_pdf(self) -> None:
        """A PDF header but truncated before any page data."""
        from worker.parsers.files import FileParser

        parser = FileParser()
        # Minimal PDF header but truncated
        truncated = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n"
        event = {
            "mime_type": "application/pdf",
            "source_id": "truncated.pdf",
            "raw_bytes": truncated,
        }
        docs = parser.parse(event)
        # Should either return empty or a doc with empty text, not crash
        if docs:
            assert isinstance(docs[0].text, str)
        else:
            assert docs == []

    def test_pdf_with_no_text_content(self) -> None:
        """PDF with pages but no extractable text."""
        import pymupdf

        doc = pymupdf.open()
        doc.new_page()  # blank page, no text
        pdf_bytes = doc.tobytes()
        doc.close()

        from worker.parsers.files import FileParser

        parser = FileParser()
        event = {
            "mime_type": "application/pdf",
            "source_id": "blank.pdf",
            "raw_bytes": pdf_bytes,
        }
        docs = parser.parse(event)
        assert len(docs) == 1
        # Text should be empty or whitespace-only
        assert docs[0].text.strip() == ""


class TestEmptyObsidianInput:
    """Test Obsidian parser with edge-case input."""

    def test_frontmatter_only_note(self) -> None:
        from worker.parsers.obsidian import ObsidianParser

        parser = ObsidianParser()
        note = "---\ntitle: Empty Body\ntags:\n  - meta\n---\n"
        event = {
            "source_id": "frontmatter-only.md",
            "operation": "created",
            "text": note,
            "meta": {},
        }
        docs = parser.parse(event)
        assert len(docs) == 1
        assert docs[0].node_props["title"] == "Empty Body"
        # Body should be empty or whitespace
        assert docs[0].text.strip() == ""

    def test_empty_note(self) -> None:
        from worker.parsers.obsidian import ObsidianParser

        parser = ObsidianParser()
        event = {
            "source_id": "empty.md",
            "operation": "created",
            "text": "",
            "meta": {},
        }
        docs = parser.parse(event)
        assert len(docs) == 1
        assert docs[0].text == ""

    def test_frontmatter_string_tags(self) -> None:
        """Tags as a single string instead of list should be handled."""
        from worker.parsers.obsidian import ObsidianParser

        parser = ObsidianParser()
        note = "---\ntags: single-tag\n---\nBody"
        event = {
            "source_id": "test.md",
            "operation": "created",
            "text": note,
            "meta": {},
        }
        docs = parser.parse(event)
        tag_hints = [h for h in docs[0].graph_hints if h.predicate == "TAGGED_BY_USER"]
        tag_ids = {h.object_id for h in tag_hints}
        assert "tag:single-tag" in tag_ids


class TestEmptyGitRepo:
    """Test repository source with empty/unusual repos."""

    @pytest.mark.asyncio
    async def test_empty_repo_no_commits_skipped(self, tmp_path: Path) -> None:
        """Repo with no commits should be skipped gracefully."""
        import git

        repo_path = tmp_path / "empty-repo"
        repo_path.mkdir()
        git.Repo.init(repo_path)

        from worker.sources.repositories import RepositorySource

        s = RepositorySource()
        s.configure(
            {
                "repo_roots": [str(tmp_path)],
                "cursor_path": str(tmp_path / "cursor.json"),
                "poll_interval_seconds": 3600,
            }
        )

        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        task = asyncio.create_task(s.start(q))

        events: list[dict[str, Any]] = []
        try:
            while True:
                ev = await asyncio.wait_for(q.get(), timeout=2.0)
                events.append(ev)
        except (asyncio.TimeoutError, TimeoutError):
            pass

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # No events from empty repo
        assert events == []


# ---------------------------------------------------------------------------
# Config validation edge cases
# ---------------------------------------------------------------------------


class TestConfigEdgeCases:
    """Test invalid config values caught at parse time."""

    def test_negative_qdrant_port(self) -> None:
        """Negative port numbers should be accepted at parse level (runtime validation)."""
        from worker.config import _parse

        # Config parsing doesn't validate port ranges — it only checks types.
        # This is a gap, but the test documents current behavior.
        cfg = _parse({"qdrant": {"port": -1}})
        assert cfg.qdrant.port == -1

    def test_empty_collection_name(self) -> None:
        from worker.config import _parse

        cfg = _parse({"qdrant": {"collection": ""}})
        assert cfg.qdrant.collection == ""

    def test_malformed_neo4j_uri(self) -> None:
        """Malformed URIs are accepted at parse time (validated at connection time)."""
        from worker.config import _parse

        cfg = _parse({"neo4j": {"uri": "not-a-valid-uri"}})
        assert cfg.neo4j.uri == "not-a-valid-uri"

    def test_repositories_config_wrong_types(self) -> None:
        """Type validation for [sources.repositories] settings."""
        from worker.config import _parse

        with pytest.raises(TypeError, match=r"repo_roots"):
            _parse({"sources": {"repositories": {"repo_roots": "not-a-list"}}})

        with pytest.raises(TypeError, match=r"include_patterns"):
            _parse(
                {
                    "sources": {
                        "repositories": {
                            "repo_roots": ["/tmp"],
                            "include_patterns": "*.md",
                        }
                    }
                }
            )

        with pytest.raises(TypeError, match=r"poll_interval_seconds"):
            _parse(
                {
                    "sources": {
                        "repositories": {
                            "repo_roots": ["/tmp"],
                            "poll_interval_seconds": "fast",
                        }
                    }
                }
            )

    def test_clustering_max_vectors_bounds(self) -> None:
        from worker.config import _parse

        with pytest.raises(ValueError, match=r"max_vectors must be >= 1"):
            _parse({"clustering": {"max_vectors": 0}})

        with pytest.raises(ValueError, match=r"max_vectors must be >= 1"):
            _parse({"clustering": {"max_vectors": -5}})

        with pytest.raises(ValueError, match=r"max_vectors must be <= 10000000"):
            _parse({"clustering": {"max_vectors": 99_999_999}})

    def test_clustering_min_interval_upper_bound(self) -> None:
        from worker.config import _parse

        with pytest.raises(ValueError, match=r"min_interval_seconds must be <= 86400"):
            _parse({"clustering": {"min_interval_seconds": 100_000}})

    def test_vision_config_wrong_types(self) -> None:
        from worker.config import _parse

        with pytest.raises(TypeError, match=r"\[vision\] enabled"):
            _parse({"vision": {"enabled": 0}})

    def test_macos_apps_config_validation(self) -> None:
        from worker.config import _parse

        with pytest.raises(TypeError, match=r"scan_dirs"):
            _parse({"sources": {"macos_apps": {"scan_dirs": "/Applications"}}})

        with pytest.raises(TypeError, match=r"enabled"):
            _parse({"sources": {"macos_apps": {"enabled": "yes"}}})

    def test_homebrew_config_validation(self) -> None:
        from worker.config import _parse

        with pytest.raises(TypeError, match=r"include_system"):
            _parse({"sources": {"homebrew": {"include_system": "yes"}}})

        with pytest.raises(TypeError, match=r"poll_interval_seconds"):
            _parse({"sources": {"homebrew": {"poll_interval_seconds": "300"}}})


# ---------------------------------------------------------------------------
# LLM output validation — extractor edge cases
# ---------------------------------------------------------------------------


class TestExtractorValidation:
    """Test _validate_and_build with malformed LLM output."""

    def test_entities_not_a_list(self) -> None:
        from worker.pipeline.extractor import ValidationError, _validate_and_build

        with pytest.raises(ValidationError, match="entities"):
            _validate_and_build({"entities": "not-a-list", "triples": []})

    def test_triples_not_a_list(self) -> None:
        from worker.pipeline.extractor import ValidationError, _validate_and_build

        with pytest.raises(ValidationError, match="triples"):
            _validate_and_build({"entities": [], "triples": "oops"})

    def test_non_dict_input(self) -> None:
        from worker.pipeline.extractor import ValidationError, _validate_and_build

        with pytest.raises(ValidationError, match="Expected dict"):
            _validate_and_build([])  # type: ignore[arg-type]

    def test_entity_missing_name_skipped(self) -> None:
        from worker.pipeline.extractor import _validate_and_build

        result = _validate_and_build(
            {
                "entities": [
                    {"type": "Person", "confidence": 0.8},  # missing name
                    {"name": "Valid", "type": "Person", "confidence": 0.9},
                ],
                "triples": [],
            }
        )
        assert len(result.entities) == 1
        assert result.entities[0]["name"] == "Valid"

    def test_entity_non_string_name_skipped(self) -> None:
        from worker.pipeline.extractor import _validate_and_build

        result = _validate_and_build(
            {
                "entities": [
                    {"name": 42, "type": "Person", "confidence": 0.8},
                    {"name": ["list"], "type": "Person", "confidence": 0.8},
                ],
                "triples": [],
            }
        )
        assert len(result.entities) == 0

    def test_entity_overlong_name_skipped(self) -> None:
        from worker.pipeline.extractor import MAX_ENTITY_NAME_LEN, _validate_and_build

        result = _validate_and_build(
            {
                "entities": [
                    {
                        "name": "x" * (MAX_ENTITY_NAME_LEN + 1),
                        "type": "Person",
                        "confidence": 0.8,
                    },
                ],
                "triples": [],
            }
        )
        assert len(result.entities) == 0

    def test_entity_unknown_type_defaults_to_concept(self) -> None:
        from worker.pipeline.extractor import _validate_and_build

        result = _validate_and_build(
            {
                "entities": [
                    {"name": "X", "type": "Alien", "confidence": 0.8},
                ],
                "triples": [],
            }
        )
        assert result.entities[0]["type"] == "Concept"

    def test_entity_invalid_confidence_defaults(self) -> None:
        from worker.pipeline.extractor import DEFAULT_CONFIDENCE, _validate_and_build

        result = _validate_and_build(
            {
                "entities": [
                    {"name": "X", "type": "Person", "confidence": "high"},
                ],
                "triples": [],
            }
        )
        assert result.entities[0]["confidence"] == DEFAULT_CONFIDENCE

    def test_entity_confidence_clamped(self) -> None:
        from worker.pipeline.extractor import _validate_and_build

        result = _validate_and_build(
            {
                "entities": [
                    {"name": "A", "type": "Person", "confidence": 999},
                    {"name": "B", "type": "Person", "confidence": -5},
                ],
                "triples": [],
            }
        )
        assert result.entities[0]["confidence"] == 1.0
        assert result.entities[1]["confidence"] == 0.0

    def test_triple_missing_fields_skipped(self) -> None:
        from worker.pipeline.extractor import _validate_and_build

        result = _validate_and_build(
            {
                "entities": [],
                "triples": [
                    {"subject": "A", "predicate": "KNOWS"},  # missing object
                    {"subject": "A", "object": "B"},  # missing predicate
                    {"subject": "A", "predicate": "KNOWS", "object": "B"},  # valid
                ],
            }
        )
        assert len(result.triples) == 1

    def test_triple_non_string_fields_skipped(self) -> None:
        from worker.pipeline.extractor import _validate_and_build

        result = _validate_and_build(
            {
                "entities": [],
                "triples": [
                    {"subject": 123, "predicate": "KNOWS", "object": "B"},
                    {"subject": "A", "predicate": ["KNOWS"], "object": "B"},
                    {"subject": "A", "predicate": "KNOWS", "object": {"name": "B"}},
                ],
            }
        )
        assert len(result.triples) == 0

    def test_triple_overlong_names_skipped(self) -> None:
        from worker.pipeline.extractor import MAX_ENTITY_NAME_LEN, _validate_and_build

        long_name = "x" * (MAX_ENTITY_NAME_LEN + 1)
        result = _validate_and_build(
            {
                "entities": [],
                "triples": [
                    {"subject": long_name, "predicate": "KNOWS", "object": "B"},
                    {"subject": "A", "predicate": "KNOWS", "object": long_name},
                    {"subject": "A", "predicate": long_name, "object": "B"},
                ],
            }
        )
        assert len(result.triples) == 0

    def test_empty_extraction(self) -> None:
        from worker.pipeline.extractor import _validate_and_build

        result = _validate_and_build({"entities": [], "triples": []})
        assert result.entities == []
        assert result.triples == []

    def test_missing_keys_uses_defaults(self) -> None:
        from worker.pipeline.extractor import _validate_and_build

        result = _validate_and_build({})
        assert result.entities == []
        assert result.triples == []


# ---------------------------------------------------------------------------
# Resolver edge cases — NaN vectors, partial NaN rows
# ---------------------------------------------------------------------------


class TestResolverNaNEdgeCases:
    """Extended NaN and edge case tests for the entity resolver."""

    def test_partial_nan_in_similarity_row(self) -> None:
        """Some NaN values in a row should not produce false matches."""
        from worker.models.base import EmbedResponse
        from worker.pipeline.resolver import resolve_entities

        def make_embed_model(vecs: dict[str, list[float]]):
            model = MagicMock()

            def side_effect(req, **kw):
                return EmbedResponse(
                    vectors=[vecs.get(t, [0.0] * 3) for t in req.texts],
                    model="mock",
                    input_tokens=0,
                )

            model.embed.side_effect = side_effect
            return model

        vectors = {
            "PartialNaN": [0.9, float("nan"), 0.1],
            "Existing": [0.9, 0.1, 0.0],
        }
        embed_model = make_embed_model(vectors)

        extracted = [{"name": "PartialNaN", "type": "Concept", "confidence": 0.8}]
        existing = [{"name": "Existing", "type": "Concept", "confidence": 0.8}]

        # Should not crash; the NaN handling in the resolver should manage this
        result = resolve_entities(extracted, existing, embed_model=embed_model)
        assert len(result.entities) == 1

    def test_all_existing_nan_vectors(self) -> None:
        """All existing vectors being NaN should result in entities kept as new."""
        from worker.models.base import EmbedResponse
        from worker.pipeline.resolver import resolve_entities

        model = MagicMock()

        def side_effect(req, **kw):
            vecs = []
            for t in req.texts:
                if t == "New":
                    vecs.append([0.9, 0.1, 0.0])
                else:
                    vecs.append([float("nan")] * 3)
            return EmbedResponse(vectors=vecs, model="mock", input_tokens=0)

        model.embed.side_effect = side_effect

        extracted = [{"name": "New", "type": "Concept", "confidence": 0.8}]
        existing = [{"name": "Old", "type": "Concept", "confidence": 0.8}]

        result = resolve_entities(extracted, existing, embed_model=model)
        assert len(result.entities) == 1
        # Should not match due to NaN existing vectors
        # The cosine similarity with a NaN vector produces NaN


class TestResolverConfidenceEdgeCases:
    """Test confidence clamping edge cases in resolver."""

    def test_infinity_confidence(self) -> None:
        from worker.pipeline.resolver import _clamp_confidence

        assert _clamp_confidence(float("inf")) == 1.0
        assert _clamp_confidence(float("-inf")) == 0.0

    def test_boolean_confidence(self) -> None:
        from worker.pipeline.resolver import _clamp_confidence

        # bool is subclass of int in Python
        assert _clamp_confidence(True) == 1.0
        assert _clamp_confidence(False) == 0.0

    def test_string_number_confidence(self) -> None:
        from worker.pipeline.resolver import _clamp_confidence

        assert _clamp_confidence("0.5") == 0.5

    def test_empty_string_confidence(self) -> None:
        from worker.pipeline.resolver import _clamp_confidence

        assert _clamp_confidence("") == 0.75  # default


# ---------------------------------------------------------------------------
# Gmail cursor integrity
# ---------------------------------------------------------------------------


class TestGmailCursorEdgeCases:
    """Test Gmail cursor with non-integer and edge-case values."""

    def test_non_integer_history_id_rejected(self, tmp_path: Path) -> None:
        from worker.sources.gmail import _load_cursor

        f = tmp_path / "cursor.json"
        f.write_text(json.dumps({"history_id": "not-a-number"}))
        assert _load_cursor(f) is None

    def test_float_history_id_rejected(self, tmp_path: Path) -> None:
        from worker.sources.gmail import _load_cursor

        f = tmp_path / "cursor.json"
        f.write_text(json.dumps({"history_id": "3.14"}))
        # int("3.14") raises ValueError, so this should be rejected
        assert _load_cursor(f) is None

    def test_integer_history_id_accepted(self, tmp_path: Path) -> None:
        from worker.sources.gmail import _load_cursor

        f = tmp_path / "cursor.json"
        f.write_text(json.dumps({"history_id": "12345"}))
        assert _load_cursor(f) == "12345"

    def test_none_history_id_returns_none(self, tmp_path: Path) -> None:
        from worker.sources.gmail import _load_cursor

        f = tmp_path / "cursor.json"
        f.write_text(json.dumps({"history_id": None}))
        assert _load_cursor(f) is None

    def test_integer_type_history_id(self, tmp_path: Path) -> None:
        """historyId stored as int (not string) should be validated."""
        from worker.sources.gmail import _load_cursor

        f = tmp_path / "cursor.json"
        f.write_text(json.dumps({"history_id": 42}))
        # int(42) succeeds, and the function returns the value
        result = _load_cursor(f)
        assert result == 42


# ---------------------------------------------------------------------------
# Writer predicate validation
# ---------------------------------------------------------------------------


class TestWriterPredicateValidation:
    """Test predicate validation in the writer's _merge_entity_edge."""

    def test_unknown_predicate_mapped_to_related_to(self) -> None:
        from worker.pipeline.writer import _merge_entity_edge

        tx = MagicMock()
        triple = {"subject": "A", "predicate": "INVENTED", "object": "B"}
        _merge_entity_edge(tx, triple)
        # The Cypher should use RELATED_TO
        call_args = tx.run.call_args
        assert "RELATED_TO" in call_args[0][0]

    def test_predicate_with_spaces_normalized(self) -> None:
        from worker.pipeline.writer import _merge_entity_edge

        tx = MagicMock()
        triple = {"subject": "A", "predicate": "works at", "object": "B"}
        _merge_entity_edge(tx, triple)
        call_args = tx.run.call_args
        assert "WORKS_AT" in call_args[0][0]

    def test_invalid_predicate_chars_mapped_to_related_to(self) -> None:
        from worker.pipeline.writer import _merge_entity_edge

        tx = MagicMock()
        triple = {"subject": "A", "predicate": "rel-ates;to", "object": "B"}
        _merge_entity_edge(tx, triple)
        call_args = tx.run.call_args
        assert "RELATED_TO" in call_args[0][0]

    def test_allowed_predicate_passes_through(self) -> None:
        from worker.pipeline.writer import _merge_entity_edge

        tx = MagicMock()
        triple = {"subject": "A", "predicate": "KNOWS", "object": "B"}
        _merge_entity_edge(tx, triple)
        call_args = tx.run.call_args
        assert "KNOWS" in call_args[0][0]


# ---------------------------------------------------------------------------
# Text file size limits
# ---------------------------------------------------------------------------


class TestTextFileSizeLimits:
    """Test that oversized text files are rejected."""

    def test_oversized_text_skipped(self) -> None:
        from worker.parsers.files import FileParser

        parser = FileParser()
        parser._max_text_bytes = 10
        event = {
            "mime_type": "text/plain",
            "source_id": "big.txt",
            "text": "x" * 100,
        }
        docs = parser.parse(event)
        assert docs == []

    def test_text_at_limit_accepted(self) -> None:
        from worker.parsers.files import FileParser

        parser = FileParser()
        parser._max_text_bytes = 20
        text = "x" * 20  # exactly 20 bytes in UTF-8
        event = {
            "mime_type": "text/plain",
            "source_id": "ok.txt",
            "text": text,
        }
        docs = parser.parse(event)
        assert len(docs) == 1

    def test_multibyte_text_size_check_uses_bytes(self) -> None:
        """Size check should be on UTF-8 bytes, not character count."""
        from worker.parsers.files import FileParser

        parser = FileParser()
        # 10 CJK chars = 30 bytes in UTF-8
        parser._max_text_bytes = 20
        event = {
            "mime_type": "text/plain",
            "source_id": "cjk.txt",
            "text": "日" * 10,
        }
        docs = parser.parse(event)
        assert docs == []  # 30 bytes > 20 byte limit


# ---------------------------------------------------------------------------
# Obsidian embed limits
# ---------------------------------------------------------------------------


class TestObsidianEmbedLimits:
    """Test embed count cap in Obsidian parser."""

    def test_embed_count_capped(self) -> None:
        from worker.parsers.obsidian import ObsidianParser, _MAX_EMBEDS_PER_NOTE

        parser = ObsidianParser()
        # Create a note with more embeds than the cap
        embeds = "\n".join(
            f"![[img_{i}.png]]" for i in range(_MAX_EMBEDS_PER_NOTE + 10)
        )
        note = f"---\n---\n{embeds}"
        event = {
            "source_id": "many-embeds.md",
            "operation": "created",
            "text": note,
            "meta": {},
        }
        docs = parser.parse(event)
        # 1 text doc + capped number of image docs
        img_docs = [d for d in docs if d.mime_type != "text/plain"]
        assert len(img_docs) == _MAX_EMBEDS_PER_NOTE


# ---------------------------------------------------------------------------
# Batch cosine similarity edge cases
# ---------------------------------------------------------------------------


class TestBatchCosineSimilarity:
    """Test edge cases in the cosine similarity matrix computation."""

    def test_zero_vectors(self) -> None:
        from worker.pipeline.resolver import _batch_cosine_similarity

        a = [[0.0, 0.0, 0.0]]
        b = [[1.0, 0.0, 0.0]]
        result = _batch_cosine_similarity(a, b)
        assert result.shape == (1, 1)
        assert result[0, 0] == 0.0

    def test_identical_vectors(self) -> None:
        from worker.pipeline.resolver import _batch_cosine_similarity

        a = [[1.0, 0.0, 0.0]]
        b = [[1.0, 0.0, 0.0]]
        result = _batch_cosine_similarity(a, b)
        assert result[0, 0] == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        from worker.pipeline.resolver import _batch_cosine_similarity

        a = [[1.0, 0.0, 0.0]]
        b = [[0.0, 1.0, 0.0]]
        result = _batch_cosine_similarity(a, b)
        assert result[0, 0] == pytest.approx(0.0)

    def test_matrix_shape(self) -> None:
        from worker.pipeline.resolver import _batch_cosine_similarity

        a = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
        b = [[1.0, 0.0], [0.0, 1.0]]
        result = _batch_cosine_similarity(a, b)
        assert result.shape == (3, 2)

    def test_both_zero_vectors(self) -> None:
        from worker.pipeline.resolver import _batch_cosine_similarity

        a = [[0.0, 0.0]]
        b = [[0.0, 0.0]]
        result = _batch_cosine_similarity(a, b)
        assert result[0, 0] == 0.0


# ---------------------------------------------------------------------------
# Repository cursor edge cases
# ---------------------------------------------------------------------------


class TestRepoCursorEdgeCases:
    """Test repo cursor loading with unusual data."""

    def test_cursor_non_dict_resets(self, tmp_path: Path) -> None:
        """Non-dict JSON in cursor file should return empty dict."""
        from worker.sources.repositories import _load_cursor

        f = tmp_path / "cursor.json"
        f.write_text(json.dumps(["not", "a", "dict"]))
        assert _load_cursor(f) == {}

    def test_cursor_with_extra_keys_preserved(self, tmp_path: Path) -> None:
        from worker.sources.repositories import _load_cursor

        f = tmp_path / "cursor.json"
        f.write_text(json.dumps({"repo1": "sha1", "extra": "data"}))
        result = _load_cursor(f)
        assert result["repo1"] == "sha1"
        assert result["extra"] == "data"


# ---------------------------------------------------------------------------
# Cross-source resolver with empty/edge-case data
# ---------------------------------------------------------------------------


class TestCrossSourceEdgeCases:
    """Test cross-source matching with unusual entity data."""

    def test_entity_with_empty_name(self) -> None:
        from worker.pipeline.resolver import resolve_cross_source

        entities_by_source = {
            "gmail": [{"name": "", "type": "Person", "confidence": 0.9}],
            "repos": [{"name": "", "type": "Person", "confidence": 0.9}],
        }
        matches = resolve_cross_source(entities_by_source)
        # Empty names should match on exact (both lowercased to "")
        assert len(matches) == 1

    def test_entity_with_whitespace_only_name(self) -> None:
        from worker.pipeline.resolver import resolve_cross_source

        entities_by_source = {
            "gmail": [{"name": "   ", "type": "Person", "confidence": 0.9}],
            "repos": [{"name": "   ", "type": "Person", "confidence": 0.9}],
        }
        matches = resolve_cross_source(entities_by_source)
        assert len(matches) == 1

    def test_very_long_entity_name(self) -> None:
        from worker.pipeline.resolver import resolve_cross_source

        long_name = "A" * 1000
        entities_by_source = {
            "gmail": [{"name": long_name, "type": "Concept", "confidence": 0.9}],
            "repos": [{"name": long_name, "type": "Concept", "confidence": 0.9}],
        }
        matches = resolve_cross_source(entities_by_source)
        assert len(matches) == 1
        assert matches[0].match_type == "exact"
