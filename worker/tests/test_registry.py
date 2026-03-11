"""Tests for parsers/registry.py — register and get error paths."""

from __future__ import annotations

from typing import Any

import pytest

from worker.parsers.base import BaseParser, ParsedDocument
from worker.parsers.registry import _registry, get, register


class _FakeParser(BaseParser):
    @property
    def source_type(self) -> str:
        return "fake"

    def parse(self, event: dict[str, Any]) -> list[ParsedDocument]:
        return []


class _AnotherFakeParser(BaseParser):
    @property
    def source_type(self) -> str:
        return "fake"

    def parse(self, event: dict[str, Any]) -> list[ParsedDocument]:
        return []


class TestRegister:
    def test_register_and_get(self) -> None:
        # Clean up after test
        prev = _registry.pop("fake", None)
        try:
            register(_FakeParser)
            parser = get("fake")
            assert isinstance(parser, _FakeParser)
            assert parser.source_type == "fake"
        finally:
            _registry.pop("fake", None)
            if prev is not None:
                _registry["fake"] = prev

    def test_duplicate_registration_overwrites(self) -> None:
        prev = _registry.pop("fake", None)
        try:
            register(_FakeParser)
            register(_AnotherFakeParser)
            parser = get("fake")
            assert isinstance(parser, _AnotherFakeParser)
        finally:
            _registry.pop("fake", None)
            if prev is not None:
                _registry["fake"] = prev


class TestGet:
    def test_unknown_source_type_raises(self) -> None:
        with pytest.raises(ValueError, match="No parser registered for source_type='nonexistent'"):
            get("nonexistent")

    def test_error_lists_known_types(self) -> None:
        with pytest.raises(ValueError, match="Known types:"):
            get("does_not_exist")
