"""SQLite persistence for the unified append-only ledger."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable

from karuth.ledger.chain import GENESIS_HASH, LedgerChain
from karuth.ledger.schema import LedgerEntry, LedgerEntryCreate


class LedgerStore:
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
                CREATE TABLE IF NOT EXISTS ledger_entries (
                    sequence INTEGER PRIMARY KEY,
                    previous_hash TEXT NOT NULL,
                    entry_hash TEXT NOT NULL UNIQUE,
                    entry_type TEXT NOT NULL,
                    case_id TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    action TEXT NOT NULL,
                    target TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ledger_case_id ON ledger_entries(case_id)"
            )

    def append(self, entry: LedgerEntryCreate) -> LedgerEntry:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            latest = conn.execute(
                "SELECT sequence, entry_hash FROM ledger_entries ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
            sequence = 1 if latest is None else int(latest["sequence"]) + 1
            previous_hash = GENESIS_HASH if latest is None else str(latest["entry_hash"])
            chained = LedgerChain.build_entry(
                entry,
                sequence=sequence,
                previous_hash=previous_hash,
            )
            conn.execute(
                """
                INSERT INTO ledger_entries (
                    sequence, previous_hash, entry_hash, entry_type, case_id, actor,
                    action, target, payload_json, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chained.sequence,
                    chained.previous_hash,
                    chained.entry_hash,
                    chained.entry_type,
                    chained.case_id,
                    chained.actor,
                    chained.action,
                    chained.target,
                    json.dumps(chained.payload, sort_keys=True, ensure_ascii=True),
                    chained.timestamp.isoformat(),
                ),
            )
            return chained

    def list_entries(self, case_id: str | None = None) -> list[LedgerEntry]:
        query = "SELECT * FROM ledger_entries"
        params: tuple[str, ...] = ()
        if case_id is not None:
            query += " WHERE case_id = ?"
            params = (case_id,)
        query += " ORDER BY sequence ASC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_entry(row) for row in rows]

    def verify(self) -> bool:
        return LedgerChain.verify(self.list_entries())

    def __iter__(self) -> Iterable[LedgerEntry]:
        return iter(self.list_entries())

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> LedgerEntry:
        return LedgerEntry(
            sequence=int(row["sequence"]),
            previous_hash=str(row["previous_hash"]),
            entry_hash=str(row["entry_hash"]),
            entry_type=row["entry_type"],
            case_id=str(row["case_id"]),
            actor=str(row["actor"]),
            action=str(row["action"]),
            target=str(row["target"]),
            payload=json.loads(str(row["payload_json"])),
            timestamp=datetime.fromisoformat(str(row["timestamp"])),
        )
