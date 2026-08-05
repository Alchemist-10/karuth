"""Legal document action agent."""

from __future__ import annotations

from pathlib import Path

from karuth.legal.filler import render_section_63_certificate, render_section_94_notice


def fill_legal_document(template_dir: Path, template_name: str, fields: dict[str, object]) -> str:
    if template_name == "section_94_bnss_notice":
        return render_section_94_notice(template_dir, fields)
    if template_name == "section_63_bsa_certificate":
        return render_section_63_certificate(template_dir, fields)
    raise ValueError(f"Unknown legal template: {template_name}")
