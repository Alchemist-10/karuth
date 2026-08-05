"""Template field-mapping legal document filler.

No LLM calls are allowed in this path. Placeholders are replaced only from
explicit caller-provided fields.
"""

from __future__ import annotations

import re
from pathlib import Path

PLACEHOLDER_RE = re.compile(r"{{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*}}")


def render_template(template_path: Path, fields: dict[str, object]) -> str:
    template = template_path.read_text(encoding="utf-8")

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in fields:
            raise KeyError(f"Missing legal template field: {key}")
        return str(fields[key])

    return PLACEHOLDER_RE.sub(replace, template)


def render_section_94_notice(template_dir: Path, fields: dict[str, object]) -> str:
    return render_template(template_dir / "section_94_bnss_notice.md", fields)


def render_section_63_certificate(template_dir: Path, fields: dict[str, object]) -> str:
    return render_template(template_dir / "section_63_bsa_certificate.md", fields)
