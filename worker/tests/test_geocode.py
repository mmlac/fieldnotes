"""Tests for reverse geocoding with mocked Nominatim responses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

from worker.pipeline.geocode import (
    GeocodedLocation,
    _CACHE_RADIUS_DEG,
    _create_location_node,
    _find_nearby_location,
    link_image_to_location,
    reverse_geocode,
    reverse_geocode_cached,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeNominatimResult:
    """Mimics a geopy Location object."""

    raw: dict[str, Any]


def _make_nominatim_result(
    city: str = "San Francisco",
    state: str = "California",
    country: str = "United States",
    display_name: str = "San Francisco, California, United States",
) -> FakeNominatimResult:
    return FakeNominatimResult(
        raw={
            "address": {
                "city": city,
                "state": state,
                "country": country,
            },
            "display_name": display_name,
        }
    )


class FakeNeo4jSession:
    """Minimal Neo4j session mock that tracks queries."""

    def __init__(self, records: list[dict[str, Any]] | None = None):
        self._records = records or []
        self.queries: list[tuple[str, dict]] = []

    def run(self, query: str, **params: Any) -> "FakeNeo4jResult":
        self.queries.append((query, params))
        return FakeNeo4jResult(self._records)

    def __enter__(self) -> "FakeNeo4jSession":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class FakeNeo4jResult:
    def __init__(self, records: list[dict[str, Any]]):
        self._records = records

    def single(self) -> dict[str, Any] | None:
        return self._records[0] if self._records else None


def _session_factory(records: list[dict[str, Any]] | None = None):
    """Return a session factory that yields FakeNeo4jSession."""
    session = FakeNeo4jSession(records)

    def factory():
        return session

    return factory, session


# ---------------------------------------------------------------------------
# Tests: reverse_geocode
# ---------------------------------------------------------------------------


class TestReverseGeocode:
    @patch("worker.pipeline.geocode._enforce_rate_limit")
    @patch("worker.pipeline.geocode.Nominatim")
    def test_successful_geocode(self, mock_nominatim_cls, mock_rate):
        mock_geolocator = MagicMock()
        mock_nominatim_cls.return_value = mock_geolocator
        mock_geolocator.reverse.return_value = _make_nominatim_result()

        result = reverse_geocode(37.7749, -122.4194)

        assert result.city == "San Francisco"
        assert result.state == "California"
        assert result.country == "United States"
        assert result.display_name == "San Francisco, California, United States"
        mock_geolocator.reverse.assert_called_once()

    @patch("worker.pipeline.geocode._enforce_rate_limit")
    @patch("worker.pipeline.geocode.Nominatim")
    def test_town_fallback(self, mock_nominatim_cls, mock_rate):
        """City field falls back to town when city is absent."""
        mock_geolocator = MagicMock()
        mock_nominatim_cls.return_value = mock_geolocator
        mock_geolocator.reverse.return_value = FakeNominatimResult(
            raw={
                "address": {
                    "town": "Smallville",
                    "state": "Kansas",
                    "country": "United States",
                },
                "display_name": "Smallville, Kansas, United States",
            }
        )

        result = reverse_geocode(39.0, -95.0)
        assert result.city == "Smallville"

    @patch("worker.pipeline.geocode._enforce_rate_limit")
    @patch("worker.pipeline.geocode.Nominatim")
    def test_none_result(self, mock_nominatim_cls, mock_rate):
        """Nominatim returns None for ocean/unknown locations."""
        mock_geolocator = MagicMock()
        mock_nominatim_cls.return_value = mock_geolocator
        mock_geolocator.reverse.return_value = None

        result = reverse_geocode(0.0, 0.0)
        assert result.city is None
        assert result.display_name is None

    @patch("worker.pipeline.geocode._enforce_rate_limit")
    @patch("worker.pipeline.geocode.Nominatim")
    def test_timeout_returns_empty(self, mock_nominatim_cls, mock_rate):
        from geopy.exc import GeocoderTimedOut

        mock_geolocator = MagicMock()
        mock_nominatim_cls.return_value = mock_geolocator
        mock_geolocator.reverse.side_effect = GeocoderTimedOut("timeout")

        result = reverse_geocode(37.7749, -122.4194)
        assert result.city is None
        assert result.display_name is None

    @patch("worker.pipeline.geocode._enforce_rate_limit")
    @patch("worker.pipeline.geocode.Nominatim")
    def test_service_error_returns_empty(self, mock_nominatim_cls, mock_rate):
        from geopy.exc import GeocoderServiceError

        mock_geolocator = MagicMock()
        mock_nominatim_cls.return_value = mock_geolocator
        mock_geolocator.reverse.side_effect = GeocoderServiceError("503")

        result = reverse_geocode(37.7749, -122.4194)
        assert result.city is None


# ---------------------------------------------------------------------------
# Tests: Neo4j cache layer
# ---------------------------------------------------------------------------


class TestFindNearbyLocation:
    def test_cache_hit(self):
        records = [
            {
                "city": "Paris",
                "state": "Île-de-France",
                "country": "France",
                "display_name": "Paris, France",
            }
        ]
        factory, session = _session_factory(records)

        result = _find_nearby_location(48.8566, 2.3522, factory)

        assert result is not None
        assert result.city == "Paris"
        assert result.country == "France"
        assert len(session.queries) == 1
        query, params = session.queries[0]
        assert "Location" in query
        assert params["lat"] == 48.8566
        assert params["radius"] == _CACHE_RADIUS_DEG

    def test_cache_miss(self):
        factory, _ = _session_factory([])

        result = _find_nearby_location(48.8566, 2.3522, factory)
        assert result is None


class TestCreateLocationNode:
    def test_creates_node(self):
        factory, session = _session_factory()
        loc = GeocodedLocation(
            city="Berlin",
            state="Berlin",
            country="Germany",
            display_name="Berlin, Germany",
        )

        _create_location_node(52.52, 13.405, loc, factory)

        assert len(session.queries) == 1
        query, params = session.queries[0]
        assert "CREATE" in query
        assert "Location" in query
        assert params["city"] == "Berlin"
        assert params["lat"] == 52.52


class TestLinkImageToLocation:
    def test_creates_taken_at_edge(self):
        factory, session = _session_factory()

        link_image_to_location("img-123", 48.8566, 2.3522, factory)

        assert len(session.queries) == 1
        query, params = session.queries[0]
        assert "TAKEN_AT" in query
        assert params["source_id"] == "img-123"


# ---------------------------------------------------------------------------
# Tests: reverse_geocode_cached (integration of cache + API)
# ---------------------------------------------------------------------------


class TestReverseGeocodeCached:
    @patch("worker.pipeline.geocode._enforce_rate_limit")
    @patch("worker.pipeline.geocode.Nominatim")
    def test_cache_miss_calls_api_and_creates_node(self, mock_nominatim_cls, mock_rate):
        """On cache miss, calls Nominatim and stores result in Neo4j."""
        mock_geolocator = MagicMock()
        mock_nominatim_cls.return_value = mock_geolocator
        mock_geolocator.reverse.return_value = _make_nominatim_result(
            city="Tokyo",
            state="Tokyo",
            country="Japan",
            display_name="Tokyo, Japan",
        )

        # Empty cache (no records returned for the MATCH query)
        # But CREATE should succeed
        call_count = 0
        sessions: list[FakeNeo4jSession] = []

        def factory():
            nonlocal call_count
            # First call = cache lookup (no results), second = create
            s = FakeNeo4jSession([] if call_count == 0 else [])
            sessions.append(s)
            call_count += 1
            return s

        result = reverse_geocode_cached(35.6762, 139.6503, factory)

        assert result.city == "Tokyo"
        assert result.country == "Japan"
        mock_geolocator.reverse.assert_called_once()
        # Should have 2 sessions: one for cache lookup, one for create
        assert len(sessions) == 2
        assert "CREATE" in sessions[1].queries[0][0]

    def test_cache_hit_skips_api(self):
        """On cache hit, returns cached result without calling Nominatim."""
        records = [
            {
                "city": "Sydney",
                "state": "New South Wales",
                "country": "Australia",
                "display_name": "Sydney, NSW, Australia",
            }
        ]
        factory, session = _session_factory(records)

        # No Nominatim mock needed — it shouldn't be called
        result = reverse_geocode_cached(-33.8688, 151.2093, factory)

        assert result.city == "Sydney"
        assert result.country == "Australia"
        # Only the cache lookup query, no create
        assert len(session.queries) == 1
        assert "MATCH" in session.queries[0][0]
