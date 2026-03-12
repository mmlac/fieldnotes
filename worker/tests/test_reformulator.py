"""Tests for the question reformulator."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from worker.cli.reformulator import _needs_reformulation, reformulate
from worker.models.base import CompletionResponse


class TestNeedsReformulation:
    def test_no_history_returns_false(self) -> None:
        assert _needs_reformulation("Tell me more", history_len=0) is False

    def test_standalone_question_no_trigger(self) -> None:
        assert _needs_reformulation("What is Python?", history_len=2) is False

    def test_pronoun_it_triggers(self) -> None:
        assert _needs_reformulation("Tell me more about it", history_len=1) is True

    def test_pronoun_that_triggers(self) -> None:
        assert _needs_reformulation("What about that?", history_len=1) is True

    def test_ordinal_triggers(self) -> None:
        assert _needs_reformulation("What about the second one?", history_len=1) is True

    def test_more_triggers(self) -> None:
        assert _needs_reformulation("Tell me more", history_len=1) is True

    def test_also_triggers(self) -> None:
        assert _needs_reformulation("What is also relevant?", history_len=3) is True

    def test_this_triggers(self) -> None:
        assert _needs_reformulation("Explain this further", history_len=1) is True

    def test_mentioned_triggers(self) -> None:
        assert _needs_reformulation("The tool you mentioned", history_len=2) is True


class TestReformulate:
    def _mock_model(self, response_text: str) -> MagicMock:
        model = MagicMock()
        model.complete.return_value = CompletionResponse(text=response_text)
        return model

    def test_standalone_question_passes_through(self) -> None:
        model = self._mock_model("should not be called")
        result = reformulate("What is Python?", [], model)
        assert result == "What is Python?"
        model.complete.assert_not_called()

    def test_first_question_passes_through(self) -> None:
        model = self._mock_model("should not be called")
        result = reformulate("Tell me more about it", [], model)
        assert result == "Tell me more about it"
        model.complete.assert_not_called()

    def test_followup_reformulated(self) -> None:
        model = self._mock_model("What is Alice's role in the Fieldnotes project?")
        history = [
            ("What is the Fieldnotes project?", "Fieldnotes is a knowledge graph project."),
        ]
        result = reformulate("What about that person Alice mentioned?", history, model)
        assert result == "What is Alice's role in the Fieldnotes project?"
        model.complete.assert_called_once()

    def test_tell_me_more_reformulated(self) -> None:
        model = self._mock_model("Tell me more about the Rust programming language")
        history = [
            ("What is Rust?", "Rust is a systems programming language."),
        ]
        result = reformulate("Tell me more", history, model)
        assert result == "Tell me more about the Rust programming language"

    def test_model_failure_returns_original(self) -> None:
        model = MagicMock()
        model.complete.side_effect = RuntimeError("model unavailable")
        history = [("Q1", "A1")]
        result = reformulate("Tell me more about it", history, model)
        assert result == "Tell me more about it"

    def test_empty_response_returns_original(self) -> None:
        model = self._mock_model("   ")
        history = [("Q1", "A1")]
        result = reformulate("Tell me more about it", history, model)
        assert result == "Tell me more about it"

    def test_history_limited_to_3_turns(self) -> None:
        model = self._mock_model("expanded query")
        history = [
            ("Q1", "A1"),
            ("Q2", "A2"),
            ("Q3", "A3"),
            ("Q4", "A4"),
            ("Q5", "A5"),
        ]
        reformulate("Tell me more about that", history, model)
        call_args = model.complete.call_args[0][0]
        prompt_content = call_args.messages[0]["content"]
        # Should only include last 3 turns
        assert "Q3" in prompt_content
        assert "Q4" in prompt_content
        assert "Q5" in prompt_content
        assert "Q1" not in prompt_content
        assert "Q2" not in prompt_content
