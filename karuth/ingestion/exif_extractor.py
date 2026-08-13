"""Extract image metadata from embedded JPEG EXIF and/or synthetic sidecars.

Two metadata sources are supported, and both are kept working:

1. **Embedded JPEG EXIF** — parsed locally with Pillow (``PIL.Image``). GPS
   latitude/longitude, capture timestamp (``DateTimeOriginal``), camera
   make/model, and orientation are decoded into the same canonical keys the
   sidecar path uses.
2. **Synthetic sidecar** — the ``<image>.exif.json`` files the synthetic case
   generator writes next to JPEG fixtures. Sidecars carry synthetic-only
   fields (``synthetic_identity``, ``vehicle_plate``, ``camera_serial``,
   ``sensitive_content``, ``activity_interpretation``) that cannot live in
   standard EXIF.

Every result records *provenance* (``embedded`` / ``sidecar`` / ``merged`` /
``conflicting`` / ``unavailable``) and a ``metadata_confidence`` value.
Missing or corrupt metadata is reported as ``None`` (rendered as
"Not scored" in the UI) — never silently as zero confidence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from PIL import Image, ExifTags
    from PIL.ExifTags import GPS

    _PILLOW_AVAILABLE = True
except ImportError:  # pragma: no cover - Pillow is optional for embedded EXIF
    _PILLOW_AVAILABLE = False

# Sidecar values win over embedded values for these shared keys.
_SIDECAR_WINS = {
    "captured_at",
    "gps",
    "camera_make",
    "camera_model",
    "orientation",
}

# Synthetic-only keys that can only come from a sidecar.
_SIDECAR_ONLY = {
    "camera_serial",
    "vehicle_plate",
    "synthetic_identity",
    "sensitive_content",
    "activity_interpretation",
    "landmark",
    "visible_vehicle",
    "plate_alternatives",
    "plate_confidence",
    "interpretation_confidence",
    "image_tag",
}

_EXIF_DATETIME_ORIGINAL = 0x9003  # "YYYY:MM:DD HH:MM:SS"
_EXIF_ORIENTATION = 0x0112
_EXIF_MAKE = 0x010F
_EXIF_MODEL = 0x0110
_EXIF_GPS_IFD = 0x8825


# ---------------------------------------------------------------------------
# hashing / helpers
# ---------------------------------------------------------------------------


def source_hash(image_path: Path) -> str | None:
    """SHA-256 of the image bytes, or ``None`` when unreadable."""
    try:
        return hashlib.sha256(Path(image_path).read_bytes()).hexdigest()
    except OSError:
        return None


def _parse_exif_datetime(raw: str) -> str | None:
    """Convert EXIF ``2026:01:12 09:42:00`` into ISO ``2026-01-12T09:42:00``."""
    try:
        from datetime import datetime

        return datetime.strptime(raw.strip(), "%Y:%m:%d %H:%M:%S").isoformat()
    except (ValueError, TypeError):
        return None


def _gps_to_decimal(value: Any) -> float:
    """EXIF DMS rational ``(deg, min, sec)`` -> decimal degrees."""
    try:
        degrees = float(value[0])
        minutes = float(value[1])
        seconds = float(value[2])
        return degrees + minutes / 60.0 + seconds / 3600.0
    except (TypeError, ValueError, IndexError):
        return 0.0


def _read_gps(exif: Any) -> dict[str, float] | None:
    """Decode the GPS IFD into ``{"lat": float, "lon": float}``."""
    try:
        gps_ifd = exif.get_ifd(_EXIF_GPS_IFD)
    except Exception:  # noqa: BLE001 - malformed GPS IFD must not crash parsing
        return None
    if not gps_ifd or not isinstance(gps_ifd, dict):
        return None
    if _EXIF_GPS_IFD in gps_ifd:  # some writers nest it
        gps_ifd = gps_ifd[_EXIF_GPS_IFD]
    try:
        lat = _gps_to_decimal(gps_ifd[GPS.GPSLatitude])
        lon = _gps_to_decimal(gps_ifd[GPS.GPSLongitude])
        if str(gps_ifd.get(GPS.GPSLatitudeRef, "N")).upper() == "S":
            lat = -lat
        if str(gps_ifd.get(GPS.GPSLongitudeRef, "E")).upper() == "W":
            lon = -lon
    except (KeyError, TypeError):
        return None
    if lat == 0 and lon == 0:
        return None
    return {"lat": round(lat, 6), "lon": round(lon, 6)}


def _read_embedded_exif(image_path: Path) -> dict[str, Any] | None:
    """Parse embedded JPEG EXIF into canonical keys, or ``None``.

    Returns ``None`` for files without EXIF, unreadable files, corrupted
    EXIF blocks, and non-image files — never raises.
    """
    if not _PILLOW_AVAILABLE:
        return None
    try:
        with Image.open(image_path) as image:
            exif = image.getexif()
            if not exif:
                return None
            data: dict[str, Any] = {}
            orientation = exif.get(_EXIF_ORIENTATION)
            if orientation:
                data["orientation"] = int(orientation)
            make = exif.get(_EXIF_MAKE)
            if make:
                data["camera_make"] = str(make)
            model = exif.get(_EXIF_MODEL)
            if model:
                data["camera_model"] = str(model)
            captured_at = exif.get(_EXIF_DATETIME_ORIGINAL)
            if captured_at:
                iso = _parse_exif_datetime(str(captured_at))
                if iso:
                    data["captured_at"] = iso
            gps = _read_gps(exif)
            if gps:
                data["gps"] = gps
            return data or None
    except Exception:  # noqa: BLE001 - corrupt EXIF must degrade, not crash
        return None


def _read_sidecar(image_path: Path) -> dict[str, Any] | None:
    sidecar = Path(image_path).with_suffix(".exif.json")
    if not sidecar.exists():
        return None
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _values_differ(key: str, left: Any, right: Any) -> bool:
    if key == "gps":
        try:
            return not (
                abs(float(left["lat"]) - float(right["lat"])) < 0.0001
                and abs(float(left["lon"]) - float(right["lon"])) < 0.0001
            )
        except (TypeError, ValueError, KeyError):
            return True
    if key == "captured_at":
        from datetime import datetime, timezone

        def _norm(value: Any) -> datetime | None:
            if not isinstance(value, str):
                return None
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
            if parsed.tzinfo is None:  # embedded EXIF timestamps are naive
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed

        left_t, right_t = _norm(left), _norm(right)
        if left_t is None or right_t is None:
            return True
        return abs((left_t - right_t).total_seconds()) > 60
    return str(left) != str(right)


def extract_image_metadata(image_path: Path) -> dict[str, Any]:
    """Extract metadata from embedded JPEG EXIF and/or a synthetic sidecar.

    Returns a dict with ``source_file``, ``provenance``, ``source``,
    ``metadata_confidence``, ``source_hash``, and any merged metadata keys.
    Sidecar values win on conflicts (synthetic determinism), but conflicts are
    recorded in ``conflicts`` and the embedded values in ``embedded_exif`` so
    downstream contradiction hunting can surface them.
    """
    path = Path(image_path)
    result: dict[str, Any] = {
        "file": path.name,
        "source_file": str(path),
        "provenance": "unavailable",
        "source": "unavailable",
        "metadata_confidence": None,
        "source_hash": source_hash(path),
        "conflicts": [],
    }

    embedded = _read_embedded_exif(path)
    sidecar = _read_sidecar(path)

    if embedded:
        result["embedded_exif"] = {key: value for key, value in embedded.items()}
    if sidecar:
        result["sidecar_exif"] = sidecar

    if not embedded and not sidecar:
        return result

    merged: dict[str, Any] = {}
    conflicts: list[str] = []
    if sidecar:
        for key, value in sidecar.items():
            if key in _SIDECAR_WINS and key not in _SIDECAR_ONLY:
                merged[key] = value
            elif key in _SIDECAR_ONLY:
                merged[key] = value
            elif key not in _SIDECAR_WINS:
                merged[key] = value
    if embedded:
        for key, value in embedded.items():
            if key in merged:
                if _values_differ(key, merged[key], value):
                    conflicts.append(key)
                    continue  # sidecar wins
            merged.setdefault(key, value)

    # Keep synthetic-only sidecar fields out of embedded but merged regardless.
    if sidecar:
        for key in _SIDECAR_ONLY:
            if key in sidecar:
                merged[key] = sidecar[key]

    if embedded and sidecar:
        provenance = "conflicting" if conflicts else "merged"
        source = "embedded + synthetic sidecar"
    elif sidecar:
        provenance = "sidecar"
        source = "synthetic sidecar"
    else:
        provenance = "embedded"
        source = "embedded JPEG EXIF"

    if provenance == "conflicting":
        confidence = 0.6
    elif provenance == "merged" or provenance == "sidecar":
        confidence = 0.95
    elif provenance == "embedded":
        confidence = 0.9 if embedded.get("gps") and embedded.get("captured_at") else 0.7
    else:  # pragma: no cover - defensive
        confidence = None

    merged.pop("file", None)
    merged.pop("source_file", None)
    merged.update(
        {
            "file": path.name,
            "source_file": str(path),
            "provenance": provenance,
            "source": source,
            "metadata_confidence": confidence,
            "source_hash": result.get("source_hash"),
            "conflicts": conflicts,
        }
    )
    return merged
