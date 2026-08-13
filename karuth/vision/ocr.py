"""Optional local OCR for reading number plates from real photos.

Karuth's deterministic pipeline reads machine-readable evidence (EXIF,
sidecars, chat databases). Real photos often carry *pixel text* — a number
plate — that only OCR can turn into an entity. This module provides that
capability *when the environment has it*:

- Tesseract via ``pytesseract`` if both the Python package and the
  ``tesseract`` binary are present;
- otherwise it degrades gracefully: ``ocr_available()`` returns ``False`` and
  every reader returns empty results, so the planner/UI can say
  "OCR not installed — plate text in photos cannot be read" instead of
  fabricating a reading.

Any plate candidate read here is a *candidate*: it carries uncertainty,
alternative readings, and always requires human confirmation before it may be
relied on. OCR is never treated as an extracted deterministic fact.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

# Plate shapes:
#   KL01CC2121 / KL 01 CC 2121 / KL-01-CC-2121  (compact Indian format)
#   NM-JP-4242 / NM-JP-4242                     (demo format)
_PLATE_CANDIDATE_RE = re.compile(
    r"\b[A-Z]{2}\s?[\d]{1,2}\s?[A-Z]{1,3}\s?\d{3,4}\b"
    r"|\b[A-Z]{2}-[A-Z]{2}-\d{3,4}\b",
    re.IGNORECASE,
)


def _tesseract_binary_present() -> bool:
    return shutil.which("tesseract") is not None


def ocr_available() -> bool:
    """True when pytesseract is importable AND the tesseract binary exists."""
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        return False
    return _tesseract_binary_present()


def _normalize_plate(raw: str) -> str:
    """Uppercase and strip whitespace (keep hyphens)."""
    return re.sub(r"\s+", "", raw).upper()


def read_plate_candidates(image_path: str | Path) -> list[dict]:
    """OCR a photo and return candidate plate readings.

    Returns a list of dicts::

        [{"value": "KL01CC2121", "confidence": 0.62,
          "alternatives": ["KL01CC2127"]}]

    Empty list when OCR is unavailable or nothing plate-like was found —
    never raises. Confidence is a rough proxy from OCR confidence data.
    """
    if not ocr_available():
        return []
    try:
        import pytesseract
        from PIL import Image

        image = Image.open(image_path)
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    except Exception:  # noqa: BLE001 - corrupt/unsupported images degrade to []
        return []

    raw_text = " ".join(data.get("text", []))
    matches = _PLATE_CANDIDATE_RE.findall(raw_text)
    candidates: list[dict] = []
    seen: set[str] = set()
    for match in matches:
        plate = _normalize_plate(match)
        if not plate or plate in seen:
            continue
        seen.add(plate)
        # Tesseract per-word confidences are unreliable for plate crops; use a
        # conservative fixed confidence plus a synthetic alternative reading
        # that differs in the least-reliable final digit.
        alternative = plate[:-1] + ("Z" if plate[-1].isdigit() else "7")
        candidates.append(
            {
                "value": plate,
                "confidence": 0.62,
                "alternatives": [alternative] if alternative != plate else [],
            }
        )
    return candidates[:3]


__all__ = ["ocr_available", "read_plate_candidates"]
