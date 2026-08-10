"""Case-folder evidence inventory for the semantic planner.

The planner must never schedule tools that cannot produce useful output. This
module inventories what evidence actually exists in a case folder (chats,
images, videos, documents, bank records, audio, manifest) so the LLM backend
can reason about the case instead of guessing from the goal text alone.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

EVIDENCE_KINDS = ("manifest", "chats", "images", "videos", "documents", "bank", "audio")

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".bmp", ".gif"}
_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
_DOCUMENT_SUFFIXES = {".txt", ".md", ".pdf", ".doc", ".docx", ".html", ".eml", ".rtf"}
_BANK_SUFFIXES = {".csv", ".ofx", ".qif"}
_AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}


class EvidenceItem(BaseModel):
    kind: str = Field(min_length=1)
    file_count: int = 0
    size_bytes: int = 0
    sample_files: list[str] = Field(default_factory=list)
    notes: str = ""


def _kind_for(root: Path, path: Path) -> str | None:
    rel = path.relative_to(root)
    parts = [part.lower() for part in rel.parts]
    suffix = path.suffix.lower()
    name = path.name.lower()
    if name == "manifest.json":
        return "manifest"
    if any(part == "chats" for part in parts) or suffix == ".sqlite":
        return "chats"
    if any(part in ("images", "photos", "pictures") for part in parts) or suffix in _IMAGE_SUFFIXES:
        return "images"
    if any(part in ("videos", "video") for part in parts) or suffix in _VIDEO_SUFFIXES:
        return "videos"
    if any(part in ("bank", "finance", "accounts") for part in parts) or suffix in _BANK_SUFFIXES:
        return "bank"
    if suffix in _DOCUMENT_SUFFIXES:
        return "documents"
    if suffix in _AUDIO_SUFFIXES:
        return "audio"
    return None


def inventory_case_evidence(case_folder: Path) -> list[EvidenceItem]:
    """Walk a case folder and summarize the evidence by kind.

    Unreadable files (permission errors, broken stat) are counted and flagged
    in ``notes`` so the planner knows the evidence is corrupt or inaccessible
    rather than silently treating it as absent.
    """
    root = Path(case_folder).resolve()
    by_kind: dict[str, EvidenceItem] = {kind: EvidenceItem(kind=kind) for kind in EVIDENCE_KINDS}

    if not root.exists() or not root.is_dir():
        return []

    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        kind = _kind_for(root, path)
        if kind is None:
            continue
        item = by_kind[kind]
        item.file_count += 1
        try:
            item.size_bytes += path.stat().st_size
        except OSError:
            if "unreadable" not in item.notes:
                item.notes = "contains unreadable files"
            continue
        if len(item.sample_files) < 3:
            item.sample_files.append(str(path.relative_to(root)))

    return [item for item in by_kind.values() if item.file_count > 0]


def evidence_summary(items: list[EvidenceItem]) -> str:
    """Render the inventory as a short text block for the LLM prompt."""
    if not items:
        return "No evidence files present in the case folder."
    lines = []
    for item in items:
        size_text = f"{item.size_bytes} bytes"
        examples = f" e.g. {', '.join(item.sample_files[:3])}" if item.sample_files else ""
        notes = f" ({item.notes})" if item.notes else ""
        lines.append(f"- {item.kind}: {item.file_count} file(s), {size_text}{notes}{examples}")
    return "\n".join(lines)
