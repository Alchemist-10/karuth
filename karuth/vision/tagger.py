"""Mockable local vision tagging interface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class VisionTag:
    label: str
    confidence: float


class VisionTagger(Protocol):
    def tag(self, image_path: Path) -> list[VisionTag]:
        """Return local model tags for an image."""


class StaticVisionTagger:
    """Test double and offline fallback for local vision tagging."""

    def __init__(self, tags: list[VisionTag] | None = None, confidence_floor: float = 0.5) -> None:
        self.tags = tags or []
        self.confidence_floor = confidence_floor

    def tag(self, image_path: Path) -> list[VisionTag]:
        return [tag for tag in self.tags if tag.confidence >= self.confidence_floor]
