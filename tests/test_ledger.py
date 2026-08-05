from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from karuth.ledger import LedgerEntryCreate, LedgerStore
from karuth.ledger.chain import GENESIS_HASH


def test_append_hash_chains_entries(tmp_path):
    store = LedgerStore(tmp_path / "ledger.sqlite3")

    first = store.append(
        LedgerEntryCreate(
            entry_type="evidence",
            case_id="CASE-1",
            actor="scanner",
            action="hash_file",
            target="image.jpg",
            payload={"sha256": "abc"},
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
    )
    second = store.append(
        LedgerEntryCreate(
            entry_type="agent_action",
            case_id="CASE-1",
            actor="planner",
            action="dispatch",
            target="correlation_agent",
            payload={"goal": "find plate"},
            timestamp=datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc),
        )
    )

    assert first.sequence == 1
    assert first.previous_hash == GENESIS_HASH
    assert second.sequence == 2
    assert second.previous_hash == first.entry_hash
    assert first.entry_hash != second.entry_hash
    assert store.verify()


def test_verify_detects_tampering(tmp_path):
    db_path = tmp_path / "ledger.sqlite3"
    store = LedgerStore(db_path)
    store.append(
        LedgerEntryCreate(
            entry_type="evidence",
            case_id="CASE-1",
            actor="scanner",
            action="hash_file",
            target="document.txt",
            payload={"sha256": "original"},
        )
    )

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE ledger_entries SET payload_json = ? WHERE sequence = 1",
            ('{"sha256": "tampered"}',),
        )

    assert not store.verify()


def test_list_entries_filters_case_id(tmp_path):
    store = LedgerStore(tmp_path / "ledger.sqlite3")
    store.append(
        LedgerEntryCreate(
            entry_type="evidence",
            case_id="CASE-1",
            action="hash_file",
            target="a.txt",
        )
    )
    store.append(
        LedgerEntryCreate(
            entry_type="evidence",
            case_id="CASE-2",
            action="hash_file",
            target="b.txt",
        )
    )

    entries = store.list_entries(case_id="CASE-2")

    assert len(entries) == 1
    assert entries[0].case_id == "CASE-2"
