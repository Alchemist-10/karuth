"""Hash-chain construction and verification."""

from __future__ import annotations

import hashlib
import json
from datetime import timezone
from typing import Iterable

from karuth.ledger.schema import LedgerEntry, LedgerEntryCreate

GENESIS_HASH = "0" * 64


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def hash_entry_material(entry: LedgerEntryCreate, sequence: int, previous_hash: str) -> str:
    material = {
        "sequence": sequence,
        "previous_hash": previous_hash,
        "entry_type": entry.entry_type,
        "case_id": entry.case_id,
        "actor": entry.actor,
        "action": entry.action,
        "target": entry.target,
        "payload": entry.payload,
        "timestamp": entry.timestamp.astimezone(timezone.utc).isoformat(),
    }
    return hashlib.sha256(_canonical_json(material).encode("utf-8")).hexdigest()


class LedgerChain:
    """Pure hash-chain logic, separated from persistence for simple testing."""

    @staticmethod
    def build_entry(
        entry: LedgerEntryCreate,
        *,
        sequence: int,
        previous_hash: str = GENESIS_HASH,
    ) -> LedgerEntry:
        entry_hash = hash_entry_material(entry, sequence, previous_hash)
        return LedgerEntry(
            **entry.model_dump(),
            sequence=sequence,
            previous_hash=previous_hash,
            entry_hash=entry_hash,
        )

    @staticmethod
    def verify(entries: Iterable[LedgerEntry]) -> bool:
        previous_hash = GENESIS_HASH
        expected_sequence = 1
        for entry in entries:
            if entry.sequence != expected_sequence:
                return False
            if entry.previous_hash != previous_hash:
                return False
            create = LedgerEntryCreate(
                entry_type=entry.entry_type,
                case_id=entry.case_id,
                actor=entry.actor,
                action=entry.action,
                target=entry.target,
                payload=entry.payload,
                timestamp=entry.timestamp,
            )
            if hash_entry_material(create, entry.sequence, entry.previous_hash) != entry.entry_hash:
                return False
            previous_hash = entry.entry_hash
            expected_sequence += 1
        return True
