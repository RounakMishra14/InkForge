"""UI-friendly image preview helpers."""

from __future__ import annotations

import numpy as np
from PIL import Image


def side_by_side(left: Image.Image, right: Image.Image, gap: int = 18, background: int = 255) -> Image.Image:
    """Place two grayscale previews next to each other on a shared white canvas."""

    left_array = np.array(left.convert("L"), dtype=np.uint8)
    right_array = np.array(right.convert("L"), dtype=np.uint8)

    height = max(left_array.shape[0], right_array.shape[0])
    width = left_array.shape[1] + gap + right_array.shape[1]
    canvas = np.full((height, width), background, dtype=np.uint8)

    canvas[: left_array.shape[0], : left_array.shape[1]] = left_array
    offset = left_array.shape[1] + gap
    canvas[: right_array.shape[0], offset : offset + right_array.shape[1]] = right_array
    return Image.fromarray(canvas)
