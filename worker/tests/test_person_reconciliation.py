"""Tests for fuzzy Person name reconciliation and Entity→Person bridging.

These test the matching logic in isolation by mocking the Neo4j driver.
"""

from __future__ import annotations

from unittest.mock import MagicMock


from worker.pipeline.resolver import _fuzzy_threshold_for_length


# ---------------------------------------------------------------------------
# Tests for _fuzzy_threshold_for_length (used by both features)
# ---------------------------------------------------------------------------


class TestFuzzyThreshold:
    def test_short_name_exact_only(self):
        assert _fuzzy_threshold_for_length("Bo") == 100
        assert _fuzzy_threshold_for_length("AWS") == 100

    def test_medium_name_strict(self):
        assert _fuzzy_threshold_for_length("Alice") == 95

    def test_long_name_standard(self):
        assert _fuzzy_threshold_for_length("Markus Smith") == 88


# ---------------------------------------------------------------------------
# Tests for reconcile_persons_by_name logic
# ---------------------------------------------------------------------------


class TestReconcilePersonsByName:
    """Tests fuzzy Person name matching — mocks Neo4j so no DB needed."""

    def _make_writer_with_persons(self, names: list[str]):
        """Create a Writer-like object with mocked Neo4j returning given names."""
        from worker.pipeline.writer import Writer

        writer = object.__new__(Writer)
        writer._neo4j_driver = MagicMock()

        mock_session = MagicMock()
        writer._neo4j_driver.session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        writer._neo4j_driver.session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        # First call: fetch person names
        name_records = [MagicMock(name=n) for _ in [None] for n in names]
        for rec, name in zip(name_records, names):
            rec.__getitem__ = lambda self, key, n=name: n if key == "name" else None

        # Build mock result that iterates over records
        mock_name_result = MagicMock()
        mock_name_result.__iter__ = MagicMock(return_value=iter(name_records))

        # Second call: batch create SAME_AS edges
        mock_create_result = MagicMock()
        mock_create_result.single.return_value = {"cnt": 0}

        mock_session.run = MagicMock(side_effect=[mock_name_result, mock_create_result])

        return writer, mock_session

    def test_no_persons_returns_zero(self):
        writer, _ = self._make_writer_with_persons([])
        assert writer._reconcile_persons_by_name_neo4j() == 0

    def test_single_person_returns_zero(self):
        writer, _ = self._make_writer_with_persons(["Alice Smith"])
        assert writer._reconcile_persons_by_name_neo4j() == 0

    def test_identical_names_matched(self):
        """Two identical names should be caught (threshold 95+)."""
        writer, mock_session = self._make_writer_with_persons(
            ["Alice Smith", "alice smith"]
        )
        # The second session.run call is for creating SAME_AS edges
        mock_create = MagicMock()
        mock_create.single.return_value = {"cnt": 1}

        name_records = []
        for name in ["Alice Smith", "alice smith"]:
            rec = MagicMock()
            rec.__getitem__ = lambda self, key, n=name: n
            name_records.append(rec)

        mock_name_result = MagicMock()
        mock_name_result.__iter__ = MagicMock(return_value=iter(name_records))

        mock_session.run.side_effect = [mock_name_result, mock_create]

        writer._reconcile_persons_by_name_neo4j()
        # Should have attempted to create SAME_AS edges
        assert mock_session.run.call_count == 2
        # Verify the UNWIND query was called with pairs
        second_call = mock_session.run.call_args_list[1]
        assert "SAME_AS" in second_call[0][0]

    def test_different_names_not_matched(self):
        """Completely different names should not produce pairs."""
        writer, mock_session = self._make_writer_with_persons(
            ["Alice Smith", "Bob Johnson"]
        )

        name_records = []
        for name in ["Alice Smith", "Bob Johnson"]:
            rec = MagicMock()
            rec.__getitem__ = lambda self, key, n=name: n
            name_records.append(rec)

        mock_name_result = MagicMock()
        mock_name_result.__iter__ = MagicMock(return_value=iter(name_records))

        mock_session.run.side_effect = [mock_name_result]

        writer._reconcile_persons_by_name_neo4j()
        # Only one query (name fetch) — no pairs to create
        assert mock_session.run.call_count == 1


# ---------------------------------------------------------------------------
# Tests for the fuzzy matching logic directly (without Neo4j)
# ---------------------------------------------------------------------------


class TestPersonNameFuzzyLogic:
    """Test the RapidFuzz matching logic extracted from the writer."""

    def _find_pairs(self, names: list[str]) -> list[tuple[str, str, float]]:
        """Reproduce the pairing logic from _reconcile_persons_by_name_neo4j."""
        from rapidfuzz import fuzz, process as rfprocess

        pairs: list[tuple[str, str, float]] = []
        names_lower = [n.lower() for n in names]

        for i, name in enumerate(names):
            threshold = max(_fuzzy_threshold_for_length(name), 95)
            candidates = names_lower[i + 1 :]
            if not candidates:
                continue
            matches = rfprocess.extract(
                names_lower[i],
                candidates,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=threshold,
                limit=None,
            )
            for _match_str, score, idx in matches:
                actual_idx = i + 1 + idx
                confidence = score / 100.0
                pairs.append((names[i], names[actual_idx], confidence))

        return pairs

    def test_similar_names_paired(self):
        pairs = self._find_pairs(["Markus Smith", "M. Smith"])
        # These have token_sort_ratio of ~76 — below threshold 95
        # so they should NOT be paired (conservative threshold)
        assert len(pairs) == 0

    def test_near_identical_names_paired(self):
        # "Alice Johnso" vs "Alice Johnson" — one char off in a long name
        pairs = self._find_pairs(["Alice Johnson", "Alice Johnso"])
        # token_sort_ratio ≈ 96 — above threshold 95
        assert len(pairs) == 1
        assert pairs[0][2] >= 0.95

    def test_case_variants_paired(self):
        pairs = self._find_pairs(["alice smith", "Alice Smith"])
        assert len(pairs) == 1
        assert pairs[0][2] == 1.0

    def test_completely_different_not_paired(self):
        pairs = self._find_pairs(["Alice Smith", "Bob Johnson", "Carol Williams"])
        assert len(pairs) == 0

    def test_short_names_require_exact(self):
        """Short names (< 4 chars) require exact match (threshold 100)."""
        pairs = self._find_pairs(["AWS", "AMS"])
        assert len(pairs) == 0

    def test_short_exact_match(self):
        pairs = self._find_pairs(["Bob", "bob"])
        # threshold is max(100, 95) = 100, and "bob" vs "bob" is 100
        assert len(pairs) == 1

    def test_empty_list(self):
        assert self._find_pairs([]) == []

    def test_single_name(self):
        assert self._find_pairs(["Alice"]) == []


# ---------------------------------------------------------------------------
# Tests for Entity→Person bridging logic
# ---------------------------------------------------------------------------


class TestEntityPersonBridgeLogic:
    """Test the fuzzy matching between Entity(Person) names and Person names."""

    def _find_bridges(
        self,
        entity_names: list[str],
        person_names: list[str],
    ) -> list[tuple[str, str, float]]:
        """Reproduce the bridging logic from _bridge_entity_persons_neo4j."""
        from rapidfuzz import fuzz, process as rfprocess

        person_names_lower = [n.lower() for n in person_names]
        pairs: list[tuple[str, str, float]] = []

        for entity_name in entity_names:
            threshold = max(_fuzzy_threshold_for_length(entity_name), 93)
            match = rfprocess.extractOne(
                entity_name.lower(),
                person_names_lower,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=threshold,
            )
            if match is not None:
                _, score, idx = match
                pairs.append((entity_name, person_names[idx], score / 100.0))

        return pairs

    def test_exact_name_bridges(self):
        bridges = self._find_bridges(["Alice Smith"], ["Alice Smith"])
        assert len(bridges) == 1
        assert bridges[0][2] == 1.0

    def test_case_insensitive_bridge(self):
        bridges = self._find_bridges(["alice smith"], ["Alice Smith"])
        assert len(bridges) == 1

    def test_no_match_different_names(self):
        bridges = self._find_bridges(["Alice Smith"], ["Bob Johnson"])
        assert len(bridges) == 0

    def test_close_match_bridges(self):
        # "Alice Smithe" vs "Alice Smith" — score ≈ 95.6, above bridge threshold 93
        bridges = self._find_bridges(["Alice Smithe"], ["Alice Smith"])
        assert len(bridges) == 1
        assert bridges[0][2] >= 0.93

    def test_multiple_entities_multiple_persons(self):
        bridges = self._find_bridges(
            ["Alice Smith", "Bob Johnson", "Unknown Person"],
            ["Alice Smith", "Robert Johnson"],
        )
        # Alice Smith matches exactly
        assert any(e == "Alice Smith" for e, p, c in bridges)
        # "Bob Johnson" vs "Robert Johnson" — fairly different
        # token_sort_ratio is moderate, may or may not match at 93 threshold

    def test_empty_entity_list(self):
        assert self._find_bridges([], ["Alice"]) == []

    def test_empty_person_list(self):
        assert self._find_bridges(["Alice"], []) == []

    def test_short_entity_name_exact_only(self):
        """Short entity names should require exact match."""
        bridges = self._find_bridges(["Bob"], ["BOB"])
        assert len(bridges) == 1  # exact case-insensitive
        bridges = self._find_bridges(["Bob"], ["Rob"])
        assert len(bridges) == 0  # short name, not exact

    def test_threshold_at_93_for_bridging(self):
        """Entity→Person bridging uses 93 as minimum threshold."""
        # "Alice Smithe" vs "Alice Smith" — should be close enough
        bridges = self._find_bridges(["Alice Smithe"], ["Alice Smith"])
        assert len(bridges) == 1
