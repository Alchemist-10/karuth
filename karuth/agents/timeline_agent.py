"""Timeline agent for chronological ordering."""

from __future__ import annotations

from datetime import datetime

from karuth.entities.schema import Entity


def order_chronologically(entities: list[Entity]) -> list[Entity]:
    def key(entity: Entity) -> datetime:
        for field in ("sent_at", "captured_at", "timestamp"):
            value = entity.metadata.get(field)
            if isinstance(value, str):
                try:
                    return datetime.fromisoformat(value)
                except ValueError:
                    pass
        return entity.extracted_at

    return sorted(entities, key=key)
