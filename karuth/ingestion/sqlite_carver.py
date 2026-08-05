"""Extract chats and contacts from synthetic WhatsApp-style SQLite files."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


def carve_chat_db(db_path: Path) -> dict[str, list[dict[str, Any]]]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        contacts = [dict(row) for row in conn.execute("SELECT * FROM contacts")]
        messages = [dict(row) for row in conn.execute("SELECT * FROM messages")]
    return {"contacts": contacts, "messages": messages, "source_file": str(db_path)}
