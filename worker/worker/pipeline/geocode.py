"""Reverse geocoding for photo GPS coordinates.

Uses Nominatim (OpenStreetMap) via geopy to convert lat/lon into human-readable
location data. Results are cached in Neo4j Location nodes — before calling the
API, checks if a Location node exists within ~1km radius.

Rate limit: 1 request/second per Nominatim ToS.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

logger = logging.getLogger(__name__)

# Nominatim ToS: max 1 request per second
_RATE_LIMIT_INTERVAL = 1.0
_last_request_time = 0.0
_rate_lock = threading.Lock()

# Cache radius for nearby Location dedup.
# 0.005° ≈ 550m at equator, ~275m at 60°N latitude.
# Kept small to avoid matching wrong cities. The Chebyshev (square)
# check means actual match distance is up to sqrt(2) × this value.
_CACHE_RADIUS_DEG = 0.005


@dataclass
class GeocodedLocation:
    """Result of a reverse geocode lookup."""

    city: str | None = None
    state: str | None = None
    country: str | None = None
    display_name: str | None = None


def reverse_geocode(lat: float, lon: float) -> GeocodedLocation:
    """Reverse geocode a lat/lon pair via Nominatim.

    Parameters
    ----------
    lat:
        Latitude in decimal degrees.
    lon:
        Longitude in decimal degrees.

    Returns
    -------
    GeocodedLocation
        Resolved location fields. All fields may be None on failure.
    """
    _enforce_rate_limit()

    try:
        geolocator = Nominatim(
            user_agent="fieldnotes-pipeline/0.1 (https://github.com/mmlac/fieldnotes)"
        )
        location = geolocator.reverse(
            (lat, lon),
            exactly_one=True,
            language="en",
            timeout=10,
        )
    except (GeocoderTimedOut, GeocoderServiceError) as exc:
        logger.warning("Nominatim reverse geocode failed: %s", exc)
        return GeocodedLocation()
    except Exception:
        logger.warning("Unexpected geocoding error", exc_info=True)
        return GeocodedLocation()

    if location is None:
        return GeocodedLocation()

    addr = location.raw.get("address", {})

    city = (
        addr.get("city")
        or addr.get("town")
        or addr.get("village")
        or addr.get("hamlet")
    )
    state = addr.get("state")
    country = addr.get("country")
    display_name = location.raw.get("display_name")

    return GeocodedLocation(
        city=city,
        state=state,
        country=country,
        display_name=display_name,
    )


def _enforce_rate_limit() -> None:
    """Block until at least _RATE_LIMIT_INTERVAL seconds since last request."""
    global _last_request_time
    with _rate_lock:
        now = time.monotonic()
        elapsed = now - _last_request_time
        if elapsed < _RATE_LIMIT_INTERVAL:
            time.sleep(_RATE_LIMIT_INTERVAL - elapsed)
        _last_request_time = time.monotonic()


def reverse_geocode_cached(
    lat: float,
    lon: float,
    neo4j_session_factory: Callable[[], Any],
) -> GeocodedLocation:
    """Reverse geocode with Neo4j Location node cache.

    Before calling the Nominatim API, checks if a Location node exists
    within ~1km radius. If found, returns the cached result. Otherwise
    calls the API, creates a new Location node, and returns the result.

    Parameters
    ----------
    lat:
        Latitude in decimal degrees.
    lon:
        Longitude in decimal degrees.
    neo4j_session_factory:
        Callable returning a Neo4j session context manager.

    Returns
    -------
    GeocodedLocation
        Resolved location data.
    """
    cached = _find_nearby_location(lat, lon, neo4j_session_factory)
    if cached is not None:
        logger.debug(
            "Geocode cache hit for (%.4f, %.4f): %s",
            lat,
            lon,
            cached.display_name,
        )
        return cached

    result = reverse_geocode(lat, lon)

    if result.display_name:
        _create_location_node(lat, lon, result, neo4j_session_factory)
        logger.debug(
            "Geocoded (%.4f, %.4f) → %s",
            lat,
            lon,
            result.display_name,
        )

    return result


def _find_nearby_location(
    lat: float,
    lon: float,
    neo4j_session_factory: Callable[[], Any],
) -> GeocodedLocation | None:
    """Query Neo4j for a Location node matching rounded coordinates.

    First tries an exact match on rounded coords (matches MERGE key),
    then falls back to proximity search within the cache radius.
    """
    rounded_lat = _round_coord(lat)
    rounded_lon = _round_coord(lon)
    with neo4j_session_factory() as session:
        result = session.run(
            """
            MATCH (loc:Location)
            WHERE (loc.latitude = $rounded_lat AND loc.longitude = $rounded_lon)
               OR (abs(loc.latitude - $lat) < $radius
                   AND abs(loc.longitude - $lon) < $radius)
            RETURN loc.city AS city, loc.state AS state,
                   loc.country AS country, loc.display_name AS display_name
            LIMIT 1
            """,
            rounded_lat=rounded_lat,
            rounded_lon=rounded_lon,
            lat=lat,
            lon=lon,
            radius=_CACHE_RADIUS_DEG,
        )
        record = result.single()
        if record is None:
            return None

        return GeocodedLocation(
            city=record["city"],
            state=record["state"],
            country=record["country"],
            display_name=record["display_name"],
        )


def _round_coord(val: float, precision: int = 3) -> float:
    """Round a coordinate to *precision* decimal places (~111m at 3dp)."""
    return round(val, precision)


def _create_location_node(
    lat: float,
    lon: float,
    location: GeocodedLocation,
    neo4j_session_factory: Callable[[], Any],
) -> None:
    """Create or merge a Location node in Neo4j for cache.

    Uses MERGE on rounded coordinates to prevent duplicate Location
    nodes from GPS noise or concurrent threads geocoding nearby points.
    """
    rounded_lat = _round_coord(lat)
    rounded_lon = _round_coord(lon)
    with neo4j_session_factory() as session:
        session.run(
            """
            MERGE (loc:Location {
                latitude: $rounded_lat,
                longitude: $rounded_lon
            })
            ON CREATE SET
                loc.city = $city,
                loc.state = $state,
                loc.country = $country,
                loc.display_name = $display_name
            """,
            rounded_lat=rounded_lat,
            rounded_lon=rounded_lon,
            city=location.city,
            state=location.state,
            country=location.country,
            display_name=location.display_name,
        )


def link_image_to_location(
    source_id: str,
    lat: float,
    lon: float,
    neo4j_session_factory: Callable[[], Any],
) -> None:
    """Link an Image/File node to the nearest Location node via TAKEN_AT.

    Uses rounded coordinates to match the MERGE'd Location node, then
    falls back to proximity search if no exact match.
    """
    rounded_lat = _round_coord(lat)
    rounded_lon = _round_coord(lon)
    with neo4j_session_factory() as session:
        session.run(
            """
            MATCH (img {source_id: $source_id})
            WHERE img:Image OR img:File
            MATCH (loc:Location)
            WHERE loc.latitude = $rounded_lat AND loc.longitude = $rounded_lon
            MERGE (img)-[:TAKEN_AT]->(loc)
            """,
            source_id=source_id,
            rounded_lat=rounded_lat,
            rounded_lon=rounded_lon,
        )
