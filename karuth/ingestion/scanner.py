"""Recursive case-folder scanning and evidence hashing."""

from __future__ import annotations

import hashlib
from pathlib import Path

from karuth.ledger import LedgerEntryCreate, LedgerStore


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_case_folder(case_id: str, folder: Path, ledger: LedgerStore) -> list[str]:
    """Hash files recursively and append evidence entries to the shared ledger."""

    file_hashes: list[str] = []
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        file_hash = sha256_file(path)
        file_hashes.append(file_hash)
        ledger.append(
            LedgerEntryCreate(
                entry_type="evidence",
                case_id=case_id,
                actor="ingestion.scanner",
                action="hash_file",
                target=str(path.relative_to(folder)),
                payload={"sha256": file_hash, "size_bytes": path.stat().st_size},
            )
        )
    return file_hashes
