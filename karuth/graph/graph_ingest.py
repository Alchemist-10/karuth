"""Deterministic graph ingestion: unstructured case content -> Neo4j structure.

The correlation sync pushes *entities and matches* into the graph, but the raw
case content (chat messages, document prose, bank memos) never becomes
structure. This module closes that gap: it reads the unstructured text files
in a case folder and turns them into typed edges over the already-synced
``Entity`` nodes, so the graph answers questions pairwise matching cannot:

- ``SENT_BY``    person entity -> message entity (who sent what, when)
- ``MENTIONS``   message / document / transaction -> entity named in its text
- new ``document_text`` and ``transaction`` ``Entity`` nodes are MERGEd for
  document and bank content that extraction does not produce

Mention matching is *deterministic* substring matching against
already-extracted entity values (never the LLM): a chat message containing the
extracted plate ``KL07CX4242`` gets a ``MENTIONS`` edge to the plate entity,
which also links the chat thread to the image and bank records that share the
same plate. Everything is best-effort enrichment; a down or missing graph is a
cheap no-op and the deterministic pipeline never depends on it.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Iterable

from karuth.entities.schema import Entity
from karuth.graph.neo4j_client import _run_write, get_driver
from karuth.ingestion.normalizer import entity_id

logger = logging.getLogger(__name__)

# Entity types whose values are worth matching inside unstructured text. The
# extracted message/document/face values are sources, never mention targets.
_MENTIONABLE_TYPES = {
    "phone",
    "username",
    "person_name",
    "vehicle_plate",
    "camera_serial",
    "location",
    "transaction",
    "image_tag",
}

_TEXT_SUFFIXES = {".txt", ".md", ".eml"}

# Sanity caps: a hostile or huge case folder must not blow up the graph layer.
_MAX_TEXT_BYTES = 2 * 1024 * 1024  # per document
_MAX_MENTIONS_PER_SOURCE = 50
_MIN_MENTION_LEN = 4  # shorter values match too many substrings to be useful


def _rel(root: Path, path: Path) -> str:
    """Case-folder-relative path with forward slashes (portable entity ids)."""
    return str(path.relative_to(root)).replace("\\", "/")


def _iter_text_documents(case_folder: Path) -> Iterable[tuple[str, str]]:
    """Yield (relative_path, text) for plain-text documents in the case folder."""
    root = Path(case_folder)
    if not root.exists():
        return
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.suffix.lower() not in _TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue  # unreadable/corrupt file: skip, never crash extraction
        if text:
            yield _rel(root, path), text[: _MAX_TEXT_BYTES]


def _iter_bank_records(case_folder: Path) -> Iterable[dict[str, Any]]:
    """Yield transaction dicts from bank CSVs (transaction_id + memo columns)."""
    root = Path(case_folder)
    if not root.exists():
        return
    for path in sorted(p for p in root.rglob("*.csv") if p.is_file()):
        try:
            with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                reader = csv.DictReader(handle)
                fieldnames = {name.strip().lower() for name in (reader.fieldnames or [])}
                if "transaction_id" not in fieldnames or "memo" not in fieldnames:
                    continue
                for row in reader:
                    transaction_id = (row.get("transaction_id") or "").strip()
                    memo = (row.get("memo") or "").strip()
                    if transaction_id and memo:
                        yield {
                            "source_file": _rel(root, path),
                            "transaction_id": transaction_id,
                            "timestamp": (row.get("timestamp") or "").strip(),
                            "memo": memo,
                        }
        except (OSError, csv.Error):
            continue  # corrupt CSV: skip, never crash


def _mention_index(entities: list[Entity]) -> list[tuple[str, str]]:
    """Flatten extracted entities into (entity_id, lowercase value) pairs.

    Values are the deterministic lexicon the mention matcher searches for.
    Face embeddings and raw message bodies are never mention targets.
    """
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for entity in entities:
        if entity.type not in _MENTIONABLE_TYPES:
            continue
        value = str(entity.value).strip().lower()
        if len(value) < _MIN_MENTION_LEN:
            continue
        if entity.entity_id in seen:
            continue
        seen.add(entity.entity_id)
        pairs.append((entity.entity_id, value))
    return pairs


def _mentioned_ids(
    text: str, index: list[tuple[str, str]], exclude: set[str]
) -> list[str]:
    """Entity ids whose extracted value appears verbatim in ``text``."""
    lowered = text.lower()
    hits: list[str] = []
    for entity_id_, value in index:
        if entity_id_ in exclude:
            continue
        if value in lowered:
            hits.append(entity_id_)
            if len(hits) >= _MAX_MENTIONS_PER_SOURCE:
                break
    return hits


def ingest_unstructured(
    case_id: str, case_folder: Path, entities: list[Entity]
) -> dict[str, Any]:
    """Push unstructured case content into the graph as typed edges (best-effort).

    Returns ``{"synced": bool, "messages": n, "documents": n, "transactions": n,
    "sent_by": n, "mentions": n}``. When the graph is unavailable or the case
    folder holds no unstructured text, this is a cheap no-op.
    """
    if get_driver() is None:
        return {
            "synced": False,
            "reason": "graph unavailable",
            "messages": 0,
            "documents": 0,
            "transactions": 0,
            "sent_by": 0,
            "mentions": 0,
        }

    root = Path(case_folder)
    if not root.exists() or not root.is_dir():
        return {
            "synced": False,
            "reason": "case folder missing",
            "messages": 0,
            "documents": 0,
            "transactions": 0,
            "sent_by": 0,
            "mentions": 0,
        }

    index = _mention_index(entities)
    sent_by_rows: list[dict[str, Any]] = []
    mention_rows: list[dict[str, Any]] = []
    new_entity_rows: list[dict[str, Any]] = []

    # 1. Chat messages: SENT_BY (person -> message) and MENTIONS edges.
    message_entities = [entity for entity in entities if entity.type == "message"]
    for message in message_entities:
        message_id = message.entity_id
        contact_name = message.metadata.get("contact_name")
        if contact_name:
            person_entity_id = entity_id(message.source_file, "person_name", contact_name)
            sent_by_rows.append(
                {
                    "person_id": person_entity_id,
                    "message_id": message_id,
                    "case_id": case_id,
                }
            )
        for target_id in _mentioned_ids(
            str(message.value), index, exclude={message_id}
        ):
            mention_rows.append(
                {"source_id": message_id, "target_id": target_id, "case_id": case_id}
            )

    # 2. Plain-text documents: MERGE document_text nodes + MENTIONS edges.
    document_count = 0
    for rel_path, text in _iter_text_documents(root):
        document_count += 1
        document_entity_id = entity_id(rel_path, "document_text", rel_path)
        new_entity_rows.append(
            {
                "entity_id": document_entity_id,
                "type": "document_text",
                "value_preview": text[:120],
                "source_file": rel_path,
                "case_id": case_id,
                "sent_at": None,
                "contact_name": None,
            }
        )
        for target_id in _mentioned_ids(text, index, exclude={document_entity_id}):
            mention_rows.append(
                {
                    "source_id": document_entity_id,
                    "target_id": target_id,
                    "case_id": case_id,
                }
            )

    # 3. Bank transaction memos: MERGE transaction nodes + MENTIONS edges.
    transaction_count = 0
    for record in _iter_bank_records(root):
        transaction_count += 1
        transaction_entity_id = entity_id(
            record["source_file"], "transaction", record["transaction_id"]
        )
        value = f"{record['memo']} ({record['source_file']})"
        new_entity_rows.append(
            {
                "entity_id": transaction_entity_id,
                "type": "transaction",
                "value_preview": value[:120],
                "source_file": record["source_file"],
                "case_id": case_id,
                "sent_at": record["timestamp"] or None,
                "contact_name": None,
            }
        )
        for target_id in _mentioned_ids(
            record["memo"], index, exclude={transaction_entity_id}
        ):
            mention_rows.append(
                {
                    "source_id": transaction_entity_id,
                    "target_id": target_id,
                    "case_id": case_id,
                }
            )

    _run_write(
        "UNWIND $rows AS row "
        "MERGE (e:Entity {entity_id: row.entity_id}) "
        "SET e.type = row.type, e.value_preview = row.value_preview, "
        "    e.source_file = row.source_file, e.case_id = row.case_id, "
        "    e.sent_at = row.sent_at, e.contact_name = row.contact_name",
        new_entity_rows,
    )
    _run_write(
        "UNWIND $rows AS row "
        "MATCH (p:Entity {entity_id: row.person_id}) "
        "MATCH (m:Entity {entity_id: row.message_id}) "
        "MERGE (p)-[:SENT_BY {case_id: row.case_id}]->(m)",
        sent_by_rows,
    )
    _run_write(
        "UNWIND $rows AS row "
        "MATCH (s:Entity {entity_id: row.source_id}) "
        "MATCH (t:Entity {entity_id: row.target_id}) "
        "MERGE (s)-[:MENTIONS {case_id: row.case_id}]->(t)",
        mention_rows,
    )

    return {
        "synced": True,
        "messages": len(message_entities),
        "documents": document_count,
        "transactions": transaction_count,
        "sent_by": len(sent_by_rows),
        "mentions": len(mention_rows),
    }
