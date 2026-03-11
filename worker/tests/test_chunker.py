"""Tests for pipeline/chunker.py — sentence-aware text splitting."""

import pytest

from worker.pipeline.chunker import (
    Chunk,
    chunk_text,
    _split_sentences,
    _tokenize,
    _detokenize,
    _merge_short_chunks,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    MIN_CHUNK_TOKENS,
)


class TestTokenize:
    def test_splits_on_whitespace(self) -> None:
        assert _tokenize("hello world foo") == ["hello", "world", "foo"]

    def test_empty_string(self) -> None:
        assert _tokenize("") == []

    def test_single_word(self) -> None:
        assert _tokenize("word") == ["word"]


class TestDetokenize:
    def test_joins_with_space(self) -> None:
        assert _detokenize(["hello", "world"]) == "hello world"

    def test_empty_list(self) -> None:
        assert _detokenize([]) == ""


class TestSplitSentences:
    def test_splits_on_period(self) -> None:
        result = _split_sentences("First sentence. Second sentence.")
        assert len(result) == 2
        assert result[0] == "First sentence."
        assert result[1] == "Second sentence."

    def test_splits_on_question_mark(self) -> None:
        result = _split_sentences("What? Really!")
        assert len(result) == 2

    def test_no_sentence_boundary(self) -> None:
        result = _split_sentences("no boundary here")
        assert result == ["no boundary here"]

    def test_empty_string(self) -> None:
        result = _split_sentences("")
        assert result == []


class TestMergeShortChunks:
    def test_merges_short_trailing_chunk(self) -> None:
        chunks = [["a"] * 200, ["b"] * 50]  # second is short
        result = _merge_short_chunks(chunks, 100, chunk_size=512)
        assert len(result) == 1
        assert len(result[0]) == 250

    def test_keeps_long_chunks(self) -> None:
        chunks = [["a"] * 200, ["b"] * 200]
        result = _merge_short_chunks(chunks, 100, chunk_size=512)
        assert len(result) == 2

    def test_merges_short_first_chunk_forward(self) -> None:
        chunks = [["a"] * 50, ["b"] * 200]
        result = _merge_short_chunks(chunks, 100, chunk_size=512)
        assert len(result) == 1
        assert len(result[0]) == 250

    def test_empty_input(self) -> None:
        assert _merge_short_chunks([], 100, chunk_size=512) == []

    def test_single_chunk(self) -> None:
        result = _merge_short_chunks([["a"] * 50], 100, chunk_size=512)
        assert len(result) == 1


class TestChunkText:
    def test_empty_text_returns_empty(self) -> None:
        assert chunk_text("") == []

    def test_whitespace_only_returns_empty(self) -> None:
        assert chunk_text("   \n  ") == []

    def test_short_text_single_chunk(self) -> None:
        text = "Hello world. This is a test."
        chunks = chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0].index == 0
        assert "Hello" in chunks[0].text

    def test_chunk_indices_are_sequential(self) -> None:
        # Generate enough text to produce multiple chunks
        sentences = [f"Sentence number {i} with enough words to matter." for i in range(200)]
        text = " ".join(sentences)
        chunks = chunk_text(text, chunk_size=50, overlap=5, min_chunk_tokens=10)
        assert len(chunks) > 1
        for i, chunk in enumerate(chunks):
            assert chunk.index == i

    def test_overlap_creates_shared_tokens(self) -> None:
        # Create two sentences that together exceed chunk_size
        words_a = " ".join(f"word{i}" for i in range(30))
        words_b = " ".join(f"term{i}" for i in range(30))
        text = f"{words_a}. {words_b}."
        chunks = chunk_text(text, chunk_size=35, overlap=10, min_chunk_tokens=5)
        if len(chunks) >= 2:
            # Last tokens of chunk 0 should appear in start of chunk 1
            tokens_0 = chunks[0].text.split()
            tokens_1 = chunks[1].text.split()
            overlap_tokens = set(tokens_0[-10:]) & set(tokens_1[:10])
            assert len(overlap_tokens) > 0

    def test_returns_chunk_dataclass(self) -> None:
        chunks = chunk_text("Hello world.")
        assert len(chunks) == 1
        assert isinstance(chunks[0], Chunk)
        assert isinstance(chunks[0].text, str)
        assert isinstance(chunks[0].index, int)

    def test_custom_chunk_size(self) -> None:
        sentences = [f"This is sentence {i} with some extra padding words." for i in range(50)]
        text = " ".join(sentences)
        small_chunks = chunk_text(text, chunk_size=20, overlap=2, min_chunk_tokens=5)
        large_chunks = chunk_text(text, chunk_size=200, overlap=10, min_chunk_tokens=5)
        assert len(small_chunks) > len(large_chunks)

    def test_no_overlap(self) -> None:
        sentences = [f"Sentence {i} has words." for i in range(100)]
        text = " ".join(sentences)
        chunks = chunk_text(text, chunk_size=20, overlap=0, min_chunk_tokens=5)
        assert len(chunks) > 1

    def test_min_chunk_tokens_merges_small_chunks(self) -> None:
        # A short trailing sentence should get merged
        text = "A " * 200 + ". B."  # long first sentence + tiny second
        chunks = chunk_text(text, chunk_size=300, overlap=10, min_chunk_tokens=100)
        # The tiny "B." should be merged, not standalone
        for chunk in chunks:
            assert len(chunk.text.split()) >= 2  # no single-word chunks
