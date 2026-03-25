"""EXIF GPS and date extraction from image bytes.

Extracts GPS coordinates (latitude, longitude) and EXIF date from image
bytes using Pillow. Handles missing EXIF gracefully — screenshots, web
images, and other non-camera images simply return empty results.

Supports JPEG, TIFF, and HEIC formats (HEIC requires pillow-heif).
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class ExifGpsResult:
    """Extracted GPS and date information from EXIF data."""

    latitude: float | None = None
    longitude: float | None = None
    exif_date: str | None = None


def extract_exif_gps(image_bytes: bytes) -> ExifGpsResult:
    """Extract GPS coordinates and date from image EXIF data.

    Parameters
    ----------
    image_bytes:
        Raw image bytes (JPEG, TIFF, or HEIC).

    Returns
    -------
    ExifGpsResult
        Extracted lat/lon and date. Fields are None if not available.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception:
        logger.debug("Could not open image for EXIF extraction")
        return ExifGpsResult()

    try:
        exif_data = img.getexif()
    except Exception:
        logger.debug("Could not read EXIF data")
        return ExifGpsResult()

    if not exif_data:
        return ExifGpsResult()

    result = ExifGpsResult()

    # Extract date — DateTimeOriginal lives in the Exif sub-IFD (0x8769),
    # DateTime (tag 306) lives in the root IFD.
    exif_ifd = exif_data.get_ifd(0x8769)
    date_original = None
    if exif_ifd:
        date_original = exif_ifd.get(36867)  # DateTimeOriginal
    if not date_original:
        date_original = exif_data.get(306)  # DateTime (root IFD fallback)
    if date_original and isinstance(date_original, str):
        result.exif_date = date_original

    # Extract GPS info
    gps_ifd = exif_data.get_ifd(0x8825)  # GPSInfo IFD
    if gps_ifd:
        result.latitude = _parse_gps_coordinate(
            gps_ifd.get(2),  # GPSLatitude
            gps_ifd.get(1),  # GPSLatitudeRef
        )
        result.longitude = _parse_gps_coordinate(
            gps_ifd.get(4),  # GPSLongitude
            gps_ifd.get(3),  # GPSLongitudeRef
        )

    return result


def _parse_gps_coordinate(
    dms: Any,
    ref: str | None,
) -> float | None:
    """Convert EXIF GPS DMS (degrees, minutes, seconds) to decimal degrees.

    Parameters
    ----------
    dms:
        Tuple of (degrees, minutes, seconds) as IFDRational values.
    ref:
        Reference direction: 'N'/'S' for latitude, 'E'/'W' for longitude.

    Returns
    -------
    float or None
        Decimal degrees, negative for S/W. None if input is invalid.
    """
    if not dms or not ref:
        return None

    try:
        degrees = float(dms[0])
        minutes = float(dms[1])
        seconds = float(dms[2])
    except (TypeError, IndexError, ValueError):
        return None

    decimal = degrees + minutes / 60.0 + seconds / 3600.0

    if ref in ("S", "W"):
        decimal = -decimal

    return decimal


def apply_exif_to_doc(
    image_bytes: bytes,
    node_props: dict[str, Any],
) -> None:
    """Extract EXIF GPS/date from image_bytes and attach to node_props in-place.

    Adds ``latitude``, ``longitude``, and ``exif_date`` keys only when
    values are present. Existing values in node_props are not overwritten.
    """
    result = extract_exif_gps(image_bytes)

    if result.latitude is not None and "latitude" not in node_props:
        node_props["latitude"] = result.latitude
    if result.longitude is not None and "longitude" not in node_props:
        node_props["longitude"] = result.longitude
    if result.exif_date and "exif_date" not in node_props:
        node_props["exif_date"] = result.exif_date
