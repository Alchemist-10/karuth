"""SQLite-backed entity persistence.

Neo4j synchronization is intentionally left for the graph phase; this repository
keeps Phase 2 deterministic and local.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from karuth.entities.schema import Entity


class EntityRepository:
    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS entities (
                    entity_id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    source_file TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    extracted_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type)")

    def upsert(self, entity: Entity) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO entities (
                    entity_id, type, value_json, source_file, confidence,
                    extracted_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(entity_id) DO UPDATE SET
                    type = excluded.type,
                    value_json = excluded.value_json,
                    source_file = excluded.source_file,
                    confidence = excluded.confidence,
                    extracted_at = excluded.extracted_at,
                    metadata_json = excluded.metadata_json
                """,
                (
                    entity.entity_id,
                    entity.type,
                    json.dumps(entity.value, sort_keys=True, ensure_ascii=True),
                    entity.source_file,
                    entity.confidence,
                    entity.extracted_at.isoformat(),
                    json.dumps(entity.metadata, sort_keys=True, ensure_ascii=True),
                ),
            )

    def list(self, entity_type: str | None = None) -> list[Entity]:
        query = "SELECT * FROM entities"
        params: tuple[str, ...] = ()
        if entity_type is not None:
            query += " WHERE type = ?"
            params = (entity_type,)
        query += " ORDER BY entity_id"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_entity(row) for row in rows]

    @staticmethod
    def _row_to_entity(row: sqlite3.Row) -> Entity:
        return Entity(
            entity_id=str(row["entity_id"]),
            type=row["type"],
            value=json.loads(str(row["value_json"])),
            source_file=str(row["source_file"]),
            confidence=float(row["confidence"]),
            extracted_at=datetime.fromisoformat(str(row["extracted_at"])),
            metadata=json.loads(str(row["metadata_json"])),
        )
