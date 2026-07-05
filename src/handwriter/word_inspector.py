"""Debug helpers for inspecting extracted word samples."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw

from .preview import stack_horizontal
from .words import WordSample


def build_word_contact_sheet(samples: list[WordSample]) -> Image.Image:
    """Create a simple labeled contact sheet for a list of word samples."""

    if not samples:
        return Image.fromarray(np.full((1, 1), 255, dtype=np.uint8))

    tiles: list[Image.Image] = []
    for sample in samples:
        word_image = Image.fromarray(sample.image).convert("L")
        canvas = Image.new("L", (word_image.width, word_image.height + 16), 255)
        canvas.paste(word_image, (0, 0))

        # The default bitmap font keeps the dependency surface light.
        draw = ImageDraw.Draw(canvas)
        draw.text((2, word_image.height + 2), sample.segmentation_mode[:10], fill=0)
        tiles.append(canvas)

    return stack_horizontal(tiles, gap=12)


def export_word_samples(word: str, samples: list[WordSample], output_dir: Path) -> Path:
    """Save a contact sheet for the selected word under artifacts/."""

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_word = _slugify(word)
    output_path = output_dir / f"{safe_word}_samples.png"
    sheet = build_word_contact_sheet(samples)
    sheet.save(output_path)
    return output_path


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value.strip())
    return slug.strip("_") or "word"
