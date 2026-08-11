"""LLM-backed image transcription via LM Studio's OpenAI-compatible endpoint.

The planner may schedule this tool so an LLM vision model describes what is
visible in a real image and extracts readable text. This is investigator
assistance only: the transcription is LLM-generated, can hallucinate, and must
never be treated as extracted evidence. It is routed through review like any
other LLM output, and its nature is recorded in the result and the ledger.

The heavy client is lazy-imported so the rest of the package (and the
deterministic pipeline) works without the OpenAI SDK installed.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from karuth.config import (
    LMSTUDIO_BASE_URL,
    LMSTUDIO_MAX_TOKENS,
    LMSTUDIO_MODEL,
    LMSTUDIO_TIMEOUT,
    VISION_MODEL,
)

_MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}

DEFAULT_TRANSCRIPTION_PROMPT = (
    "You are assisting a digital forensics investigator. Describe exactly what is "
    "visible in this image: people, objects, clothing, surroundings, and any readable "
    "text (transcribe text verbatim). Do not speculate about events, intent, or "
    "identity. If nothing is clearly visible, say so."
)

DISCLAIMER = (
    "LLM-generated image description; investigator assistance only. Not extracted "
    "evidence and must be independently verified."
)


class ImageTranscription(BaseModel):
    """Result of transcribing one image with an LLM vision model."""

    image_path: str = Field(min_length=1)
    transcription: str = ""
    extracted_text: str = ""
    model_used: str = ""
    llm_backend: bool = True
    disclaimer: str = DISCLAIMER
    error: str = ""
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class VisionAgentError(Exception):
    """Raised when the LLM vision backend is unavailable for transcription."""


def _mime_for(path: Path) -> str:
    return _MIME_BY_SUFFIX.get(path.suffix.lower(), "image/jpeg")


def _data_uri(path: Path) -> str:
    return f"data:{_mime_for(path)};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


_VISION_HINTS = ("vl", "vision", "llava", "qwen2.5vl", "minicpm", "glm-4v")


def _discover_vision_model(client: object) -> str | None:
    """Pick the first loaded vision-capable model from ``GET /v1/models``.

    Vision-capable ids (``qwen3-vl``, ``llava``, ...) are preferred; if none
    are found, the first non-embedding model is used so a text-only model can
    at least attempt the request (which will fail softly and be logged).
    Returns ``None`` when the server is unreachable.
    """
    try:
        models = client.models.list()
    except Exception:
        return None
    loaded: list[str] = []
    vision: list[str] = []
    for model in models:
        model_id = getattr(model, "id", None)
        if not model_id:
            continue
        lowered = model_id.lower()
        if any(hint in lowered for hint in ("embed", "embedding", "bge-", "nomic")):
            continue
        loaded.append(model_id)
        if any(hint in lowered for hint in _VISION_HINTS):
            vision.append(model_id)
    return (vision or loaded or [None])[0]


def _default_client():
    try:
        import openai  # lazy import: deterministic pipeline never needs it
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise VisionAgentError("OpenAI SDK not installed; cannot reach LM Studio vision endpoint") from exc
    try:
        # LM Studio's OpenAI-compatible endpoint ignores the key, but the SDK
        # requires one to construct a client. max_retries=0 makes an
        # unreachable server fail fast instead of multiplying the timeout by
        # retries, so the planner falls back quickly.
        return openai.OpenAI(
            base_url=LMSTUDIO_BASE_URL,
            api_key="lm-studio",
            timeout=LMSTUDIO_TIMEOUT,
            max_retries=0,
        )
    except Exception as exc:  # pragma: no cover - client construction is environment dependent
        raise VisionAgentError(f"cannot create LM Studio client: {exc}") from exc


def _resolve_model(client: object, model: str | None) -> str:
    """Resolve the vision model: explicit arg, config, or auto-discovery."""
    if model:
        return model
    if VISION_MODEL:
        return VISION_MODEL
    discovered = _discover_vision_model(client)
    if discovered:
        return discovered
    return LMSTUDIO_MODEL or "local-model"


def transcribe_image(
    image_path: str | Path,
    prompt: str | None = None,
    client: object | None = None,
    model: str | None = None,
) -> ImageTranscription:
    """Transcribe one image with the LM Studio vision model.

    ``client`` may be injected for tests; by default a real client is built
    from configuration. Any failure (unreachable server, non-vision model,
    unreadable file) is captured in ``ImageTranscription.error`` rather than
    raised, so the planner can log a soft failure and continue.
    """
    path = Path(image_path)
    try:
        data_uri = _data_uri(path)
    except OSError as exc:
        return ImageTranscription(image_path=str(path), error=f"unreadable image: {exc}")

    try:
        client = client or _default_client()
        resolved_model = _resolve_model(client, model)
        response = client.chat.completions.create(
            model=resolved_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt or DEFAULT_TRANSCRIPTION_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ],
            temperature=0.0,
            max_tokens=LMSTUDIO_MAX_TOKENS,
        )
        content = (response.choices[0].message.content or "").strip()
    except Exception as exc:  # noqa: BLE001 - capture any backend failure as a soft error
        return ImageTranscription(
            image_path=str(path),
            error=f"LM Studio vision request failed: {exc}",
        )

    return ImageTranscription(
        image_path=str(path),
        transcription=content,
        extracted_text=content,
        model_used=resolved_model,
    )


def transcribe_images(
    image_paths: list[str | Path],
    prompt: str | None = None,
    client: object | None = None,
    model: str | None = None,
) -> list[ImageTranscription]:
    """Transcribe several images, capturing per-file failures individually."""
    return [
        transcribe_image(path, prompt=prompt, client=client, model=model)
        for path in image_paths
    ]
