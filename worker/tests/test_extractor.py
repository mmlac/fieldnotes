"""Tests for the LLM-based entity and triple extractor.

Uses unittest.mock to stub out model completions so tests run without
a live LLM backend.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from worker.models.base import CompletionResponse
from worker.models.resolver import ResolvedModel
from worker.pipeline.chunker import Chunk
from worker.pipeline.extractor import (
    EXTRACT_ROLE,
    EXTRACTION_TOOL,
    ExtractionResult,
    FALLBACK_ROLE,
    SYSTEM_PROMPT,
    ValidationError,
    _call_and_parse,
    _validate_and_build,
    extract_chunk,
    extract_chunks,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


def _chunk(text: str = "Alice works at Acme Corp.", index: int = 0) -> Chunk:
    return Chunk(text=text, index=index)


def _tool_call_response(entities: list, triples: list) -> CompletionResponse:
    """Build a CompletionResponse with a tool_call containing extraction data."""
    return CompletionResponse(
        text="",
        tool_calls=[
            {
                "function": {
                    "name": "extract_entities_and_triples",
                    "arguments": json.dumps(
                        {"entities": entities, "triples": triples}
                    ),
                }
            }
        ],
    )


def _text_response(data: dict | str) -> CompletionResponse:
    """Build a CompletionResponse with JSON in the text body."""
    text = data if isinstance(data, str) else json.dumps(data)
    return CompletionResponse(text=text, tool_calls=None)


def _mock_model(response: CompletionResponse) -> ResolvedModel:
    """Create a mock ResolvedModel that returns the given response."""
    model = MagicMock(spec=ResolvedModel)
    model.complete.return_value = response
    return model


# ------------------------------------------------------------------
# _validate_and_build
# ------------------------------------------------------------------


class TestValidateAndBuild:
    def test_valid_full_data(self) -> None:
        data = {
            "entities": [
                {"name": "Alice", "type": "Person", "confidence": 0.95},
                {"name": "Acme", "type": "Organization", "confidence": 0.9},
            ],
            "triples": [
                {"subject": "Alice", "predicate": "works_at", "object": "Acme"},
            ],
        }
        result = _validate_and_build(data)
        assert len(result.entities) == 2
        assert result.entities[0]["name"] == "Alice"
        assert result.entities[0]["type"] == "Person"
        assert result.entities[0]["confidence"] == 0.95
        assert len(result.triples) == 1
        assert result.triples[0]["predicate"] == "works_at"

    def test_defaults_for_missing_fields(self) -> None:
        data = {
            "entities": [{"name": "X"}],
            "triples": [],
        }
        result = _validate_and_build(data)
        assert result.entities[0]["type"] == "Concept"
        assert result.entities[0]["confidence"] == 0.75

    def test_skips_entities_without_name(self) -> None:
        data = {
            "entities": [{"type": "Person"}, {"name": "Bob", "type": "Person", "confidence": 0.8}],
            "triples": [],
        }
        result = _validate_and_build(data)
        assert len(result.entities) == 1
        assert result.entities[0]["name"] == "Bob"

    def test_skips_incomplete_triples(self) -> None:
        data = {
            "entities": [],
            "triples": [
                {"subject": "A", "predicate": "knows"},  # missing object
                {"subject": "A", "predicate": "knows", "object": "B"},
            ],
        }
        result = _validate_and_build(data)
        assert len(result.triples) == 1

    def test_non_dict_raises(self) -> None:
        with pytest.raises(ValidationError, match="Expected dict"):
            _validate_and_build([1, 2, 3])  # type: ignore[arg-type]

    def test_entities_not_list_raises(self) -> None:
        with pytest.raises(ValidationError, match="'entities' must be a list"):
            _validate_and_build({"entities": "bad", "triples": []})

    def test_triples_not_list_raises(self) -> None:
        with pytest.raises(ValidationError, match="'triples' must be a list"):
            _validate_and_build({"entities": [], "triples": "bad"})

    def test_empty_data_returns_empty(self) -> None:
        result = _validate_and_build({})
        assert result.entities == []
        assert result.triples == []


# ------------------------------------------------------------------
# _call_and_parse
# ------------------------------------------------------------------


class TestCallAndParse:
    def test_parses_tool_call_response(self) -> None:
        resp = _tool_call_response(
            entities=[{"name": "Neo4j", "type": "Technology", "confidence": 0.9}],
            triples=[],
        )
        model = _mock_model(resp)
        from worker.models.base import CompletionRequest

        req = CompletionRequest(system="", messages=[{"role": "user", "content": "test"}])
        result = _call_and_parse(model, req)
        assert len(result.entities) == 1
        assert result.entities[0]["name"] == "Neo4j"

    def test_parses_text_json_response(self) -> None:
        resp = _text_response({
            "entities": [{"name": "Python", "type": "Technology", "confidence": 0.85}],
            "triples": [],
        })
        model = _mock_model(resp)
        from worker.models.base import CompletionRequest

        req = CompletionRequest(system="", messages=[{"role": "user", "content": "test"}])
        result = _call_and_parse(model, req)
        assert len(result.entities) == 1

    def test_empty_response_returns_empty(self) -> None:
        resp = CompletionResponse(text="", tool_calls=None)
        model = _mock_model(resp)
        from worker.models.base import CompletionRequest

        req = CompletionRequest(system="", messages=[{"role": "user", "content": "test"}])
        result = _call_and_parse(model, req)
        assert result.entities == []
        assert result.triples == []

    def test_malformed_json_raises(self) -> None:
        resp = CompletionResponse(text="not json {{{", tool_calls=None)
        model = _mock_model(resp)
        from worker.models.base import CompletionRequest

        req = CompletionRequest(system="", messages=[{"role": "user", "content": "test"}])
        with pytest.raises(json.JSONDecodeError):
            _call_and_parse(model, req)

    def test_tool_call_with_dict_arguments(self) -> None:
        """Some providers return arguments as a dict, not a JSON string."""
        resp = CompletionResponse(
            text="",
            tool_calls=[
                {
                    "function": {
                        "name": "extract_entities_and_triples",
                        "arguments": {
                            "entities": [{"name": "Go", "type": "Technology", "confidence": 0.9}],
                            "triples": [],
                        },
                    }
                }
            ],
        )
        model = _mock_model(resp)
        from worker.models.base import CompletionRequest

        req = CompletionRequest(system="", messages=[{"role": "user", "content": "test"}])
        result = _call_and_parse(model, req)
        assert result.entities[0]["name"] == "Go"


# ------------------------------------------------------------------
# extract_chunk
# ------------------------------------------------------------------


class TestExtractChunk:
    def test_successful_extraction(self) -> None:
        resp = _tool_call_response(
            entities=[{"name": "Alice", "type": "Person", "confidence": 0.95}],
            triples=[{"subject": "Alice", "predicate": "works_at", "object": "Acme"}],
        )
        model = _mock_model(resp)
        result = extract_chunk(_chunk(), model)
        assert len(result.entities) == 1
        assert len(result.triples) == 1

    def test_falls_back_on_json_error(self) -> None:
        bad_model = _mock_model(CompletionResponse(text="{bad", tool_calls=None))
        good_resp = _tool_call_response(
            entities=[{"name": "Bob", "type": "Person", "confidence": 0.8}],
            triples=[],
        )
        fallback = _mock_model(good_resp)

        result = extract_chunk(_chunk(), bad_model, fallback_model=fallback)
        assert len(result.entities) == 1
        assert result.entities[0]["name"] == "Bob"
        fallback.complete.assert_called_once()

    def test_falls_back_on_validation_error(self) -> None:
        # Return a non-dict JSON value
        bad_model = _mock_model(CompletionResponse(text='"just a string"', tool_calls=None))
        good_resp = _tool_call_response(
            entities=[{"name": "C", "type": "Concept", "confidence": 0.7}],
            triples=[],
        )
        fallback = _mock_model(good_resp)

        result = extract_chunk(_chunk(), bad_model, fallback_model=fallback)
        assert len(result.entities) == 1

    def test_returns_empty_when_both_fail(self) -> None:
        bad = _mock_model(CompletionResponse(text="{bad", tool_calls=None))
        also_bad = _mock_model(CompletionResponse(text="{bad", tool_calls=None))

        result = extract_chunk(_chunk(), bad, fallback_model=also_bad)
        assert result.entities == []
        assert result.triples == []

    def test_returns_empty_when_no_fallback(self) -> None:
        bad = _mock_model(CompletionResponse(text="{bad", tool_calls=None))
        result = extract_chunk(_chunk(), bad)
        assert result.entities == []
        assert result.triples == []

    def test_passes_correct_request(self) -> None:
        resp = _tool_call_response(entities=[], triples=[])
        model = _mock_model(resp)
        chunk = _chunk("Some text about Rust.")
        extract_chunk(chunk, model)

        call_args = model.complete.call_args[0][0]
        assert call_args.system == SYSTEM_PROMPT
        assert call_args.messages[0]["content"] == "Some text about Rust."
        assert call_args.tools == [EXTRACTION_TOOL]
        assert call_args.temperature == 0.0


# ------------------------------------------------------------------
# extract_chunks
# ------------------------------------------------------------------


class TestExtractChunks:
    def test_processes_all_chunks(self) -> None:
        resp = _tool_call_response(
            entities=[{"name": "X", "type": "Concept", "confidence": 0.5}],
            triples=[],
        )
        registry = MagicMock()
        registry.for_role.side_effect = lambda role: (
            _mock_model(resp) if role == EXTRACT_ROLE else MagicMock()
        )
        # Make fallback raise KeyError so it's None
        def for_role_side_effect(role: str):
            if role == EXTRACT_ROLE:
                return _mock_model(resp)
            raise KeyError(f"No model for role {role!r}")

        registry.for_role.side_effect = for_role_side_effect

        chunks = [_chunk("text 1", 0), _chunk("text 2", 1), _chunk("text 3", 2)]
        results = extract_chunks(chunks, registry)
        assert len(results) == 3
        assert all(len(r.entities) == 1 for r in results)

    def test_empty_chunks_returns_empty(self) -> None:
        registry = MagicMock()
        result = extract_chunks([], registry)
        assert result == []
        registry.for_role.assert_not_called()

    def test_uses_fallback_when_configured(self) -> None:
        good_resp = _tool_call_response(
            entities=[{"name": "Y", "type": "Concept", "confidence": 0.6}],
            triples=[],
        )
        primary = _mock_model(CompletionResponse(text="{bad", tool_calls=None))
        fallback = _mock_model(good_resp)

        registry = MagicMock()

        def for_role_side_effect(role: str):
            if role == EXTRACT_ROLE:
                return primary
            if role == FALLBACK_ROLE:
                return fallback
            raise KeyError(f"No model for role {role!r}")

        registry.for_role.side_effect = for_role_side_effect

        results = extract_chunks([_chunk()], registry)
        assert len(results) == 1
        assert results[0].entities[0]["name"] == "Y"


# ------------------------------------------------------------------
# ExtractionResult
# ------------------------------------------------------------------


class TestExtractionResult:
    def test_defaults_empty(self) -> None:
        r = ExtractionResult()
        assert r.entities == []
        assert r.triples == []

    def test_with_data(self) -> None:
        r = ExtractionResult(
            entities=[{"name": "A", "type": "Person", "confidence": 0.9}],
            triples=[{"subject": "A", "predicate": "knows", "object": "B"}],
        )
        assert len(r.entities) == 1
        assert len(r.triples) == 1
