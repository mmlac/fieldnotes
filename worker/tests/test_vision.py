"""Tests for the vision extraction module.

Uses unittest.mock to stub out model completions so tests run without
a live multimodal backend.
"""

from __future__ import annotations

import base64
import json
from unittest.mock import MagicMock

import pytest

from worker.models.base import CompletionResponse
from worker.models.resolver import ResolvedModel
from worker.pipeline.vision import (
    ENTITY_CONFIDENCE,
    SYSTEM_PROMPT,
    VISION_ROLE,
    VisionResult,
    _parse_response,
    extract_image,
    extract_image_from_registry,
    vision_result_to_chunk,
    vision_result_to_entities,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

SAMPLE_IMAGE = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64  # fake PNG header


def _json_response(data: dict) -> CompletionResponse:
    """Build a CompletionResponse with JSON in the text body."""
    return CompletionResponse(text=json.dumps(data), tool_calls=None)


def _mock_model(response: CompletionResponse) -> ResolvedModel:
    """Create a mock ResolvedModel that returns the given response."""
    model = MagicMock(spec=ResolvedModel)
    model.complete.return_value = response
    return model


def _full_response() -> dict:
    """A complete, valid vision response."""
    return {
        "description": "A screenshot of a terminal running Python code.",
        "visible_text": "import json\nprint('hello')",
        "entities": [
            {"name": "Python", "type": "Technology"},
            {"name": "JSON", "type": "Technology"},
        ],
    }


# ------------------------------------------------------------------
# _parse_response
# ------------------------------------------------------------------


class TestParseResponse:
    def test_valid_full_response(self) -> None:
        resp = _json_response(_full_response())
        result = _parse_response(resp)
        assert result.description == "A screenshot of a terminal running Python code."
        assert result.visible_text == "import json\nprint('hello')"
        assert len(result.entities) == 2
        assert result.entities[0]["name"] == "Python"
        assert result.entities[1]["type"] == "Technology"

    def test_empty_text_returns_empty(self) -> None:
        resp = CompletionResponse(text="", tool_calls=None)
        result = _parse_response(resp)
        assert result.description == ""
        assert result.visible_text == ""
        assert result.entities == []

    def test_none_text_returns_empty(self) -> None:
        resp = CompletionResponse(text=None, tool_calls=None)  # type: ignore[arg-type]
        result = _parse_response(resp)
        assert result == VisionResult()

    def test_strips_markdown_code_fences(self) -> None:
        data = _full_response()
        text = "```json\n" + json.dumps(data) + "\n```"
        resp = CompletionResponse(text=text, tool_calls=None)
        result = _parse_response(resp)
        assert result.description == data["description"]
        assert len(result.entities) == 2

    def test_strips_bare_code_fences(self) -> None:
        data = {"description": "test", "visible_text": "", "entities": []}
        text = "```\n" + json.dumps(data) + "\n```"
        resp = CompletionResponse(text=text, tool_calls=None)
        result = _parse_response(resp)
        assert result.description == "test"

    def test_malformed_json_raises(self) -> None:
        resp = CompletionResponse(text="not json {{{", tool_calls=None)
        with pytest.raises(json.JSONDecodeError):
            _parse_response(resp)

    def test_non_dict_raises(self) -> None:
        resp = CompletionResponse(text='["a list"]', tool_calls=None)
        with pytest.raises(ValueError, match="Expected dict"):
            _parse_response(resp)

    def test_missing_fields_default_empty(self) -> None:
        resp = _json_response({})
        result = _parse_response(resp)
        assert result.description == ""
        assert result.visible_text == ""
        assert result.entities == []

    def test_entities_not_list_defaults_empty(self) -> None:
        resp = _json_response({"entities": "bad"})
        result = _parse_response(resp)
        assert result.entities == []

    def test_skips_entities_without_name(self) -> None:
        resp = _json_response(
            {
                "entities": [
                    {"type": "Person"},  # no name
                    {"name": "Alice", "type": "Person"},
                ],
            }
        )
        result = _parse_response(resp)
        assert len(result.entities) == 1
        assert result.entities[0]["name"] == "Alice"

    def test_entity_defaults_type_to_concept(self) -> None:
        resp = _json_response({"entities": [{"name": "X"}]})
        result = _parse_response(resp)
        assert result.entities[0]["type"] == "Concept"


# ------------------------------------------------------------------
# extract_image
# ------------------------------------------------------------------


class TestExtractImage:
    def test_successful_extraction(self) -> None:
        resp = _json_response(_full_response())
        model = _mock_model(resp)
        result = extract_image(SAMPLE_IMAGE, model)

        assert result.description == "A screenshot of a terminal running Python code."
        assert len(result.entities) == 2

    def test_passes_correct_request(self) -> None:
        resp = _json_response(_full_response())
        model = _mock_model(resp)
        extract_image(SAMPLE_IMAGE, model, mime_type="image/jpeg")

        call_args = model.complete.call_args[0][0]
        assert call_args.system == SYSTEM_PROMPT
        assert call_args.temperature == 0.0

        # Verify the message contains image content
        msg = call_args.messages[0]
        assert msg["role"] == "user"
        content = msg["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "image"
        assert content[0]["source"]["media_type"] == "image/jpeg"
        # Verify base64 encoding
        decoded = base64.b64decode(content[0]["source"]["data"])
        assert decoded == SAMPLE_IMAGE

    def test_returns_empty_on_json_error(self) -> None:
        resp = CompletionResponse(text="{bad json", tool_calls=None)
        model = _mock_model(resp)
        result = extract_image(SAMPLE_IMAGE, model)
        assert result == VisionResult()

    def test_returns_empty_on_value_error(self) -> None:
        resp = CompletionResponse(text='"just a string"', tool_calls=None)
        model = _mock_model(resp)
        result = extract_image(SAMPLE_IMAGE, model)
        assert result == VisionResult()


# ------------------------------------------------------------------
# extract_image_from_registry
# ------------------------------------------------------------------


class TestExtractImageFromRegistry:
    def test_resolves_vision_role(self) -> None:
        resp = _json_response(_full_response())
        model = _mock_model(resp)

        registry = MagicMock()
        registry.for_role.return_value = model

        result = extract_image_from_registry(SAMPLE_IMAGE, registry)
        registry.for_role.assert_called_once_with(VISION_ROLE)
        assert result.description == "A screenshot of a terminal running Python code."

    def test_raises_on_missing_vision_role(self) -> None:
        registry = MagicMock()
        registry.for_role.side_effect = KeyError("No model for role 'vision'")

        with pytest.raises(KeyError):
            extract_image_from_registry(SAMPLE_IMAGE, registry)


# ------------------------------------------------------------------
# vision_result_to_chunk
# ------------------------------------------------------------------


class TestVisionResultToChunk:
    def test_both_fields(self) -> None:
        result = VisionResult(
            description="A photo of a cat.",
            visible_text="MEOW",
        )
        chunk = vision_result_to_chunk(result)
        assert chunk is not None
        assert chunk.text == "A photo of a cat.\n\nMEOW"
        assert chunk.index == 0

    def test_description_only(self) -> None:
        result = VisionResult(description="A diagram.")
        chunk = vision_result_to_chunk(result)
        assert chunk is not None
        assert chunk.text == "A diagram."

    def test_visible_text_only(self) -> None:
        result = VisionResult(visible_text="Hello world")
        chunk = vision_result_to_chunk(result)
        assert chunk is not None
        assert chunk.text == "Hello world"

    def test_empty_returns_none(self) -> None:
        result = VisionResult()
        assert vision_result_to_chunk(result) is None


# ------------------------------------------------------------------
# vision_result_to_entities
# ------------------------------------------------------------------


class TestVisionResultToEntities:
    def test_converts_with_fixed_confidence(self) -> None:
        result = VisionResult(
            entities=[
                {"name": "Python", "type": "Technology"},
                {"name": "Alice", "type": "Person"},
            ],
        )
        ents = vision_result_to_entities(result)
        assert len(ents) == 2
        assert ents[0] == {
            "name": "Python",
            "type": "Technology",
            "confidence": ENTITY_CONFIDENCE,
        }
        assert ents[1]["confidence"] == ENTITY_CONFIDENCE

    def test_defaults_type_to_concept(self) -> None:
        result = VisionResult(entities=[{"name": "X"}])
        ents = vision_result_to_entities(result)
        assert ents[0]["type"] == "Concept"

    def test_skips_entities_without_name(self) -> None:
        result = VisionResult(entities=[{"type": "Person"}, {"name": "Bob"}])
        ents = vision_result_to_entities(result)
        assert len(ents) == 1
        assert ents[0]["name"] == "Bob"

    def test_empty_entities(self) -> None:
        result = VisionResult()
        assert vision_result_to_entities(result) == []


# ------------------------------------------------------------------
# VisionResult
# ------------------------------------------------------------------


class TestVisionResult:
    def test_defaults_empty(self) -> None:
        r = VisionResult()
        assert r.description == ""
        assert r.visible_text == ""
        assert r.entities == []

    def test_with_data(self) -> None:
        r = VisionResult(
            description="desc",
            visible_text="text",
            entities=[{"name": "A", "type": "Person"}],
        )
        assert r.description == "desc"
        assert len(r.entities) == 1
