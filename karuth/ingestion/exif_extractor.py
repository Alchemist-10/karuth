"""Extract synthetic image metadata.

The Phase 1 generator writes `.exif.json` sidecars next to JPEG fixtures. This
keeps tests deterministic while still modeling GPS, camera serial, timestamps,
plates, and synthetic face identities.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def extract_image_metadata(image_path: Path) -> dict[str, Any]:
    sidecar = image_path.with_suffix(".exif.json")
    if not sidecar.exists():
        return {"file": image_path.name, "source_file": str(image_path)}
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["source_file"] = str(image_path)
    return payload
