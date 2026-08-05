"""Local face embedding extraction interface.

Phase 2 uses synthetic metadata to create deterministic 512-d embeddings. The
production implementation should swap in a local ONNX execution provider only.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from karuth.entities.schema import Entity
from karuth.ingestion.exif_extractor import extract_image_metadata


def _embedding_for_identity(identity: str) -> list[float]:
    rng = random.Random(hashlib.sha256(identity.encode("utf-8")).hexdigest())
    vector = [rng.uniform(-1.0, 1.0) for _ in range(512)]
    norm = sum(value * value for value in vector) ** 0.5
    return [round(value / norm, 8) for value in vector]


def extract_face_entities(image_path: Path) -> list[Entity]:
    metadata = extract_image_metadata(image_path)
    identity = metadata.get("synthetic_identity")
    if not identity:
        return []
    embedding = _embedding_for_identity(str(identity))
    entity_id = hashlib.sha256(
        f"{image_path}|face_embedding|{identity}".encode("utf-8")
    ).hexdigest()[:24]
    return [
        Entity(
            entity_id=entity_id,
            type="face_embedding",
            value=embedding,
            source_file=str(image_path),
            confidence=0.99,
            metadata={
                "synthetic_identity": identity,
                "bounding_box": [16, 16, 96, 96],
                "embedding_dimensions": len(embedding),
                "raw_metadata": json.dumps(metadata, sort_keys=True),
            },
        )
    ]
