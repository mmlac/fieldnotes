"""Tests for pipeline/chunker.py — sentence-aware text splitting."""

from worker.pipeline.chunker import (
    Chunk,
    chunk_text,
    _split_sentences,
    _split_slack_messages,
    _tokenize,
    _detokenize,
    _merge_short_chunks,
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
        sentences = [
            f"Sentence number {i} with enough words to matter." for i in range(200)
        ]
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
        sentences = [
            f"This is sentence {i} with some extra padding words." for i in range(50)
        ]
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


# ── Slack-aware ``message_overlap`` mode ─────────────────────────────────


def _make_slack_message(idx: int, body_tokens: int) -> str:
    """Build a single Slack-formatted message line with a given body length."""
    body = " ".join(f"w{idx}-{j}" for j in range(body_tokens))
    return f"[09:{idx:02d} UTC] User {idx} (@user{idx}): {body}"


def _make_slack_doc(n_messages: int, body_tokens_per_msg: int) -> str:
    return "\n".join(
        _make_slack_message(i, body_tokens_per_msg) for i in range(n_messages)
    )


class TestSplitSlackMessages:
    def test_empty_text(self) -> None:
        assert _split_slack_messages("") == []

    def test_no_headers_returns_single_block(self) -> None:
        assert _split_slack_messages("just some prose") == ["just some prose"]

    def test_splits_on_message_headers(self) -> None:
        text = _make_slack_doc(3, 5)
        blocks = _split_slack_messages(text)
        assert len(blocks) == 3
        assert blocks[0].startswith("[09:00 UTC]")
        assert blocks[1].startswith("[09:01 UTC]")
        assert blocks[2].startswith("[09:02 UTC]")

    def test_handles_indented_thread_replies(self) -> None:
        text = (
            "[09:00 UTC] Parent (@p): hello\n"
            "  [09:01 UTC] Replier (@r): reply one\n"
            "  [09:02 UTC] Replier (@r): reply two\n"
        )
        blocks = _split_slack_messages(text)
        assert len(blocks) == 3
        assert blocks[1].lstrip().startswith("[09:01 UTC]")

    def test_keeps_multiline_message_body_with_header(self) -> None:
        text = (
            "[09:00 UTC] A (@a): line one\nline two of the same message\n"
            "[09:01 UTC] B (@b): next msg\n"
        )
        blocks = _split_slack_messages(text)
        assert len(blocks) == 2
        assert "line two of the same message" in blocks[0]


class TestSlackMessageOverlap:
    def test_short_doc_single_chunk(self) -> None:
        # 5 messages * 20 tokens/msg ≈ ~125 tokens of body, well under 512.
        text = _make_slack_doc(5, 20)
        chunks = chunk_text(
            text,
            chunk_strategy={"mode": "message_overlap", "overlap_messages": 2},
        )
        assert len(chunks) == 1
        # All 5 messages preserved.
        for i in range(5):
            assert f"@user{i}" in chunks[0].text

    def test_overlap_messages_zero(self) -> None:
        # 12 messages * 50 tokens body each → ~600+ tokens total.
        # Each message is ~55 tokens including header; 9 messages fit in 512.
        text = _make_slack_doc(12, 50)
        chunks = chunk_text(
            text,
            chunk_strategy={"mode": "message_overlap", "overlap_messages": 0},
        )
        # No overlap → no shared messages between adjacent chunks.
        if len(chunks) >= 2:
            last_of_first = chunks[0].text.split("\n")[-1]
            first_of_second = chunks[1].text.split("\n")[0]
            assert last_of_first != first_of_second

    def test_consecutive_chunks_share_last_n_messages(self) -> None:
        # Build a 12-message document where each message body is ~120 tokens.
        # Total ≈ 1500 tokens → splits into multiple chunks.
        text = _make_slack_doc(12, 120)
        chunks = chunk_text(
            text,
            chunk_strategy={"mode": "message_overlap", "overlap_messages": 2},
        )
        assert len(chunks) >= 2
        for prev, curr in zip(chunks, chunks[1:]):
            prev_msgs = prev.text.split("\n")
            curr_msgs = curr.text.split("\n")
            # The first 2 messages of `curr` must equal the last 2 of `prev`.
            assert curr_msgs[:2] == prev_msgs[-2:], (
                f"Expected overlap of 2 whole messages, got prev tail "
                f"{prev_msgs[-2:]!r} vs curr head {curr_msgs[:2]!r}"
            )

    def test_no_message_split_in_half(self) -> None:
        # Every chunk's text should consist of complete message blocks —
        # i.e. every line that *looks* like a message header must be a
        # whole message header (no truncated headers, no orphan body).
        text = _make_slack_doc(15, 80)
        chunks = chunk_text(
            text,
            chunk_strategy={"mode": "message_overlap", "overlap_messages": 3},
        )
        import re

        header_re = re.compile(r"^(?:  )?\[\d{2}:\d{2} UTC\] ")
        for chunk in chunks:
            lines = chunk.text.split("\n")
            assert lines[0] == "" or header_re.match(lines[0]), (
                f"Chunk does not start on a message header: {lines[0]!r}"
            )

    def test_overlap_messages_default_is_3(self) -> None:
        # When overlap_messages is omitted from the strategy, default is 3.
        text = _make_slack_doc(15, 100)
        chunks = chunk_text(text, chunk_strategy={"mode": "message_overlap"})
        if len(chunks) >= 2:
            prev_msgs = chunks[0].text.split("\n")
            curr_msgs = chunks[1].text.split("\n")
            assert curr_msgs[:3] == prev_msgs[-3:]

    def test_unknown_mode_falls_back_to_default(self) -> None:
        # An unrecognized strategy mode must use the default sentence path.
        text = "Hello world. This is a test."
        chunks_default = chunk_text(text)
        chunks_unknown = chunk_text(text, chunk_strategy={"mode": "nonsense"})
        assert [c.text for c in chunks_default] == [c.text for c in chunks_unknown]

    def test_oversized_single_message_becomes_own_chunk(self) -> None:
        # One huge message larger than chunk_size must not be split mid-message.
        big = "[09:00 UTC] Big (@big): " + " ".join(f"tok{i}" for i in range(2000))
        small = "[09:01 UTC] Small (@s): tiny"
        text = big + "\n" + small
        chunks = chunk_text(
            text,
            chunk_strategy={"mode": "message_overlap", "overlap_messages": 1},
        )
        # The big message must be present in exactly one chunk and not split.
        big_chunks = [c for c in chunks if "@big" in c.text]
        assert len(big_chunks) == 1
        # Token count of that chunk == big message tokens (no truncation).
        assert "tok1999" in big_chunks[0].text

    def test_chunker_message_overlap_clamps_to_progress(self) -> None:
        # overlap_messages exceeds the total message count. The chunker must
        # terminate (no infinite loop, no negative-stride iteration) and emit
        # a single chunk because both messages fit within chunk_size.
        text = _make_slack_doc(2, 20)
        chunks = chunk_text(
            text,
            chunk_strategy={"mode": "message_overlap", "overlap_messages": 5},
        )
        assert len(chunks) == 1
        assert "@user0" in chunks[0].text
        assert "@user1" in chunks[0].text

    def test_chunker_message_overlap_2_messages_overlap_2(self) -> None:
        # 4 messages sized so that 3 fit per chunk_size=512 but 4 do not.
        # With overlap_messages=2, the chunker must emit 2 chunks whose
        # boundary shares the documented two-message overlap.
        text = _make_slack_doc(4, 145)
        chunks = chunk_text(
            text,
            chunk_strategy={"mode": "message_overlap", "overlap_messages": 2},
        )
        assert len(chunks) == 2
        prev_msgs = chunks[0].text.split("\n")
        curr_msgs = chunks[1].text.split("\n")
        assert curr_msgs[:2] == prev_msgs[-2:]

    def test_chunker_message_overlap_one_short_message(self) -> None:
        # Single-message document — no clamp drama, exactly one chunk.
        text = _make_slack_doc(1, 10)
        chunks = chunk_text(
            text,
            chunk_strategy={"mode": "message_overlap", "overlap_messages": 3},
        )
        assert len(chunks) == 1
        assert "@user0" in chunks[0].text

    def test_chunker_message_overlap_exact_boundary(self) -> None:
        # overlap_messages == message_count. Must terminate with a single
        # chunk (the whole document fits) and never enter an infinite loop.
        text = _make_slack_doc(3, 20)
        chunks = chunk_text(
            text,
            chunk_strategy={"mode": "message_overlap", "overlap_messages": 3},
        )
        assert len(chunks) == 1
        for i in range(3):
            assert f"@user{i}" in chunks[0].text


class TestDefaultModeUnchanged:
    """Regression check: omitting chunk_strategy is byte-identical to before."""

    def test_prose_fixture_byte_identical_with_and_without_none_strategy(self) -> None:
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "Pack my box with five dozen liquor jugs. "
            "How vexingly quick daft zebras jump! "
            "Sphinx of black quartz, judge my vow. "
            "The five boxing wizards jump quickly."
        )
        a = chunk_text(text, chunk_size=20, overlap=5, min_chunk_tokens=3)
        b = chunk_text(
            text,
            chunk_size=20,
            overlap=5,
            min_chunk_tokens=3,
            chunk_strategy=None,
        )
        assert [c.text for c in a] == [c.text for c in b]
        assert [c.index for c in a] == [c.index for c in b]
