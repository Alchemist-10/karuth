"""Canonical entity schema."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

EntityType = Literal[
    "phone",
    "username",
    "person_name",
    "message",
    "vehicle_plate",
    "camera_serial",
    "location",
    "transaction",
    "document_text",
    "image_tag",
    "face_embedding",
]


class Entity(BaseModel):
    entity_id: str = Field(min_length=1)
    type: EntityType
    value: Any
    source_file: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)
