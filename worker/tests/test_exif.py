"""Tests for EXIF GPS and date extraction."""

import io

import pytest
from PIL import Image

from worker.pipeline.exif import apply_exif_to_doc, extract_exif_gps


def _make_jpeg_with_exif(
    lat: tuple[float, float, float] | None = None,
    lat_ref: str = "N",
    lon: tuple[float, float, float] | None = None,
    lon_ref: str = "E",
    date_original: str | None = None,
) -> bytes:
    """Create a minimal JPEG with EXIF GPS and date data."""
    img = Image.new("RGB", (10, 10), color="red")
    import piexif

    exif_dict: dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}}

    if lat is not None:
        exif_dict["GPS"][piexif.GPSIFD.GPSLatitude] = _to_rational_tuple(lat)
        exif_dict["GPS"][piexif.GPSIFD.GPSLatitudeRef] = lat_ref.encode()

    if lon is not None:
        exif_dict["GPS"][piexif.GPSIFD.GPSLongitude] = _to_rational_tuple(lon)
        exif_dict["GPS"][piexif.GPSIFD.GPSLongitudeRef] = lon_ref.encode()

    if date_original is not None:
        exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = date_original.encode()

    exif_bytes = piexif.dump(exif_dict)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif_bytes)
    return buf.getvalue()


def _to_rational_tuple(
    dms: tuple[float, float, float],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Convert (degrees, minutes, seconds) to EXIF rational format."""
    return (
        (int(dms[0] * 1000), 1000),
        (int(dms[1] * 1000), 1000),
        (int(dms[2] * 10000), 10000),
    )


def _make_png_no_exif() -> bytes:
    """Create a PNG with no EXIF data (like a screenshot)."""
    img = Image.new("RGB", (10, 10), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestExtractExifGps:
    def test_no_exif_returns_empty(self):
        result = extract_exif_gps(_make_png_no_exif())
        assert result.latitude is None
        assert result.longitude is None
        assert result.exif_date is None

    def test_invalid_bytes_returns_empty(self):
        result = extract_exif_gps(b"not an image")
        assert result.latitude is None
        assert result.longitude is None
        assert result.exif_date is None

    def test_empty_bytes_returns_empty(self):
        result = extract_exif_gps(b"")
        assert result.latitude is None


class TestExtractExifGpsWithPiexif:
    """Tests that require piexif for EXIF generation."""

    @pytest.fixture(autouse=True)
    def _check_piexif(self):
        pytest.importorskip("piexif")

    def test_gps_north_east(self):
        # San Francisco: 37°46'30"N, 122°25'10"W
        image_bytes = _make_jpeg_with_exif(
            lat=(37.0, 46.0, 30.0),
            lat_ref="N",
            lon=(122.0, 25.0, 10.0),
            lon_ref="W",
            date_original="2024:06:15 14:30:00",
        )
        result = extract_exif_gps(image_bytes)
        assert result.latitude is not None
        assert abs(result.latitude - 37.775) < 0.001
        assert result.longitude is not None
        assert abs(result.longitude - (-122.4194)) < 0.01
        assert result.exif_date == "2024:06:15 14:30:00"

    def test_gps_south_east(self):
        # Sydney: 33°52'10"S, 151°12'30"E
        image_bytes = _make_jpeg_with_exif(
            lat=(33.0, 52.0, 10.0),
            lat_ref="S",
            lon=(151.0, 12.0, 30.0),
            lon_ref="E",
        )
        result = extract_exif_gps(image_bytes)
        assert result.latitude is not None
        assert result.latitude < 0  # Southern hemisphere
        assert result.longitude is not None
        assert result.longitude > 0  # Eastern hemisphere

    def test_date_only_no_gps(self):
        image_bytes = _make_jpeg_with_exif(
            date_original="2023:12:25 08:00:00",
        )
        result = extract_exif_gps(image_bytes)
        assert result.latitude is None
        assert result.longitude is None
        assert result.exif_date == "2023:12:25 08:00:00"

    def test_gps_only_no_date(self):
        image_bytes = _make_jpeg_with_exif(
            lat=(40.0, 42.0, 46.0),
            lat_ref="N",
            lon=(74.0, 0.0, 22.0),
            lon_ref="W",
        )
        result = extract_exif_gps(image_bytes)
        assert result.latitude is not None
        assert result.exif_date is None


class TestApplyExifToDoc:
    def test_applies_all_fields(self):
        pytest.importorskip("piexif")
        image_bytes = _make_jpeg_with_exif(
            lat=(48.0, 51.0, 24.0),
            lat_ref="N",
            lon=(2.0, 21.0, 7.0),
            lon_ref="E",
            date_original="2024:07:14 10:00:00",
        )
        props: dict = {}
        apply_exif_to_doc(image_bytes, props)
        assert "latitude" in props
        assert "longitude" in props
        assert "exif_date" in props
        assert abs(props["latitude"] - 48.8567) < 0.01
        assert abs(props["longitude"] - 2.3519) < 0.01

    def test_does_not_overwrite_existing(self):
        pytest.importorskip("piexif")
        image_bytes = _make_jpeg_with_exif(
            lat=(48.0, 51.0, 24.0),
            lat_ref="N",
            lon=(2.0, 21.0, 7.0),
            lon_ref="E",
        )
        props = {"latitude": 99.0, "longitude": 88.0}
        apply_exif_to_doc(image_bytes, props)
        assert props["latitude"] == 99.0
        assert props["longitude"] == 88.0

    def test_no_exif_leaves_props_unchanged(self):
        image_bytes = _make_png_no_exif()
        props: dict = {"existing_key": "value"}
        apply_exif_to_doc(image_bytes, props)
        assert "latitude" not in props
        assert "longitude" not in props
        assert "exif_date" not in props
        assert props["existing_key"] == "value"

    def test_invalid_image_leaves_props_unchanged(self):
        props: dict = {}
        apply_exif_to_doc(b"garbage", props)
        assert props == {}
