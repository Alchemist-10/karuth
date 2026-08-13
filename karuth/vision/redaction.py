"""Sensitive-image protection: blurred display derivatives with provenance.

Original evidence files are never modified. When an image is flagged
sensitive, the UI displays a *blurred derivative* created by this module and
keeps the original untouched. Every derivative records its own hash, the
original's hash, and its provenance so the redaction trail is auditable.

The blurred derivative is a display artifact only — it is never used as the
original evidence anywhere in the pipeline.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

try:
    from PIL import Image, ImageFilter

    _PILLOW_AVAILABLE = True
except ImportError:  # pragma: no cover - Pillow is optional for derivatives
    _PILLOW_AVAILABLE = False


def sha256_of(path: Path) -> str:
    """SHA-256 of a file's bytes (raises OSError when unreadable)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def create_blurred_derivative(
    image_path: Path,
    output_dir: Path,
    *,
    radius: int = 18,
    quality: int = 82,
) -> dict:
    """Create a blurred display derivative of ``image_path``.

    Returns a provenance dict::

        {
            "path": str,            # derivative file path
            "sha256": str,          # derivative hash
            "original_sha256": str, # untouched original hash
            "source": str,          # original image path
            "provenance": "blurred derivative of original (display only)",
            "blurred": bool,
        }

    The original file is never modified. When the image cannot be decoded
    (corrupt file, missing Pillow) the derivative degrades to ``blurred:
    False`` with the original path returned — it never raises.
    """
    source = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    derivative = output_dir / f"{source.stem}_blurred.jpg"

    if not _PILLOW_AVAILABLE:
        return {
            "path": str(source),
            "sha256": sha256_of(source),
            "original_sha256": sha256_of(source),
            "source": str(source),
            "provenance": "unavailable (Pillow not installed)",
            "blurred": False,
        }

    try:
        with Image.open(source) as image:
            blurred = image.convert("RGB").filter(ImageFilter.GaussianBlur(radius))
            blurred.save(derivative, "JPEG", quality=quality)
    except Exception:  # noqa: BLE001 - undecodable images degrade, never crash
        return {
            "path": str(source),
            "sha256": sha256_of(source),
            "original_sha256": sha256_of(source),
            "source": str(source),
            "provenance": "unable to decode image",
            "blurred": False,
        }

    return {
        "path": str(derivative),
        "sha256": sha256_of(derivative),
        "original_sha256": sha256_of(source),
        "source": str(source),
        "provenance": "blurred derivative of original (display only)",
        "blurred": True,
    }
