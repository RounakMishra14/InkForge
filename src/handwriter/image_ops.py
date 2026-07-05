"""Small image processing helpers kept dependency-light for the first batch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_grayscale(path: Path) -> np.ndarray:
    """Load an image as a uint8 grayscale numpy array."""

    return np.array(Image.open(path).convert("L"), dtype=np.uint8)


def to_ink_mask(image: np.ndarray, threshold: int = 220) -> np.ndarray:
    """Return a boolean mask where True means foreground ink."""

    return image < threshold


def tight_crop(image: np.ndarray, threshold: int = 220, margin: int = 1) -> np.ndarray:
    """Crop away most background while preserving a small safety margin."""

    mask = to_ink_mask(image, threshold=threshold)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return image.copy()

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0) + 1
    y_min = max(0, y_min - margin)
    x_min = max(0, x_min - margin)
    y_max = min(image.shape[0], y_max + margin)
    x_max = min(image.shape[1], x_max + margin)
    return image[y_min:y_max, x_min:x_max].copy()


def resize_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
    """Resize while preserving aspect ratio."""

    pil_image = Image.fromarray(image)
    width = max(1, round((target_height / pil_image.height) * pil_image.width))
    resized = pil_image.resize((width, target_height), Image.Resampling.LANCZOS)
    return np.array(resized, dtype=np.uint8)


def binary_iou(first: np.ndarray, second: np.ndarray, threshold: int = 220) -> float:
    """Measure overlap between two grayscale handwriting images."""

    first_mask = to_ink_mask(first, threshold=threshold)
    second_mask = to_ink_mask(second, threshold=threshold)
    intersection = np.logical_and(first_mask, second_mask).sum()
    union = np.logical_or(first_mask, second_mask).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def normalized_mae(first: np.ndarray, second: np.ndarray) -> float:
    """Return a 0-1 pixel-distance style score where lower is better."""

    max_delta = 255.0
    diff = np.abs(first.astype(np.float32) - second.astype(np.float32))
    return float(diff.mean() / max_delta)


def pad_to_same_size(first: np.ndarray, second: np.ndarray, fill_value: int = 255) -> tuple[np.ndarray, np.ndarray]:
    """Pad two arrays to the same shape for direct comparison."""

    height = max(first.shape[0], second.shape[0])
    width = max(first.shape[1], second.shape[1])

    def _pad(image: np.ndarray) -> np.ndarray:
        padded = np.full((height, width), fill_value, dtype=np.uint8)
        padded[: image.shape[0], : image.shape[1]] = image
        return padded

    return _pad(first), _pad(second)
