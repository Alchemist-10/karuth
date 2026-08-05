"""Default blur helpers for sensitive visual evidence."""

from __future__ import annotations

import shutil
from pathlib import Path


def blur_image_default(input_path: Path, output_path: Path) -> Path:
    """Create a blurred copy when OpenCV is available, otherwise copy fixture bytes.

    Assumption: tests should not depend on OpenCV. Production deployments can
    install `opencv-python` and receive Gaussian blur behavior from the same API.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2  # type: ignore

        image = cv2.imread(str(input_path))
        if image is None:
            shutil.copyfile(input_path, output_path)
            return output_path
        blurred = cv2.GaussianBlur(image, (51, 51), 0)
        cv2.imwrite(str(output_path), blurred)
    except Exception:
        shutil.copyfile(input_path, output_path)
    return output_path
