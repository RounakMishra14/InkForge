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


def stack_horizontal(images: list[Image.Image], gap: int = 18, background: int = 255) -> Image.Image:
    """Lay out multiple grayscale previews horizontally on one white canvas."""

    arrays = [np.array(image.convert("L"), dtype=np.uint8) for image in images if image is not None]
    if not arrays:
        return Image.fromarray(np.full((1, 1), background, dtype=np.uint8))

    height = max(array.shape[0] for array in arrays)
    width = sum(array.shape[1] for array in arrays) + (gap * (len(arrays) - 1))
    canvas = np.full((height, width), background, dtype=np.uint8)

    cursor = 0
    for index, array in enumerate(arrays):
        canvas[: array.shape[0], cursor : cursor + array.shape[1]] = array
        cursor += array.shape[1]
        if index < len(arrays) - 1:
            cursor += gap

    return Image.fromarray(canvas)
