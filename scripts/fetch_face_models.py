"""Fetch the local face-recognition models (YuNet + SFace) into the cache.

Usage:
    python scripts/fetch_face_models.py

Downloads the two small ONNX models from the OpenCV Zoo into the model cache
(``~/.cache/karuth/models`` by default; override with ``KARUTH_MODEL_DIR``)
and verifies both parse as ONNX graphs with OpenCV DNN. Safe to re-run: files
already present and valid are skipped.
"""

from __future__ import annotations

from karuth.config import FACE_MODEL_DIR, SFACE_MODEL_URL, YUNET_MODEL_URL
from karuth.vision.face_extractor import ensure_face_models, model_cache_dir


def main() -> None:
    print(f"Model cache: {model_cache_dir()}")
    print(f"  YuNet detector: {YUNET_MODEL_URL}")
    print(f"  SFace recognizer: {SFACE_MODEL_URL}")
    models = ensure_face_models()
    if models is None:
        raise SystemExit("Failed to obtain valid YuNet/SFace models. Check network access.")
    yunet, sface = models
    print(f"OK: {yunet} ({yunet.stat().st_size} bytes)")
    print(f"OK: {sface} ({sface.stat().st_size} bytes)")
    print("Real face extraction is ready.")


if __name__ == "__main__":
    main()
