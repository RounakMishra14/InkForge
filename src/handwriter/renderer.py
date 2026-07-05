"""Glyph-based handwritten line renderer."""

from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
from PIL import Image

from .config import RenderConfig
from .dataset import GlyphSample, HandwritingDataset
from .image_ops import load_grayscale, resize_to_height, tight_crop
from .style import StyleProfile


@dataclass(frozen=True)
class RenderedLine:
    """Rendered output and trace metadata for debugging placement decisions."""

    image: Image.Image
    placed_labels: list[str]
    unsupported_labels: list[str]


class HandwritingRenderer:
    """Compose novel text from isolated glyph samples plus style heuristics."""

    def __init__(
        self,
        dataset: HandwritingDataset,
        style_profile: StyleProfile,
        config: RenderConfig | None = None,
    ) -> None:
        self.dataset = dataset
        self.style_profile = style_profile
        self.config = config or RenderConfig()

    def render_text(self, text: str, seed: int = 7) -> RenderedLine:
        """Render a single handwritten line for the provided text."""

        rng = random.Random(seed)
        canvas = np.full(
            (self.config.canvas_height, self.config.canvas_width),
            self.config.output_background,
            dtype=np.uint8,
        )

        placed_labels: list[str] = []
        unsupported: list[str] = []
        cursor_x = self.config.padding_x
        baseline_y = int(
            self.config.padding_y
            + self.style_profile.average_sentence_height
            + rng.randint(-self.config.baseline_jitter, self.config.baseline_jitter)
        )

        for char in text:
            if char == " ":
                cursor_x += max(self.config.default_word_gap, int(round(self.style_profile.average_word_gap)))
                continue

            glyph = self._choose_glyph(char, rng)
            if glyph is None:
                unsupported.append(char)
                cursor_x += max(self.config.default_word_gap, int(round(self.style_profile.average_word_gap / 2)))
                continue

            glyph_image = self._prepare_glyph(glyph)
            cursor_x = self._paste_glyph(canvas, glyph_image, cursor_x, baseline_y, rng)
            placed_labels.append(char)

        final = self._trim_canvas(canvas)
        return RenderedLine(image=Image.fromarray(final), placed_labels=placed_labels, unsupported_labels=unsupported)

    def _choose_glyph(self, char: str, rng: random.Random) -> GlyphSample | None:
        choices = self.dataset.glyphs_for(char)
        if not choices and char == "*":
            choices = self.dataset.glyphs_for("x")
        if not choices:
            return None
        return rng.choice(choices)

    def _prepare_glyph(self, glyph: GlyphSample) -> np.ndarray:
        image = load_grayscale(glyph.path)
        cropped = tight_crop(image, threshold=220, margin=1)

        # Scale the isolated glyphs into the vertical footprint implied by the
        # sentence-level examples so the composition feels closer to the writer.
        target_height = max(24, min(72, self.style_profile.average_sentence_height + 10))
        return resize_to_height(cropped, target_height=target_height)

    def _paste_glyph(
        self,
        canvas: np.ndarray,
        glyph: np.ndarray,
        cursor_x: int,
        baseline_y: int,
        rng: random.Random,
    ) -> int:
        char_gap = max(
            self.config.min_char_gap,
            int(round(self.style_profile.average_char_gap)) + rng.randint(0, self.config.max_char_gap_jitter),
        )
        vertical_jitter = rng.randint(-self.config.glyph_vertical_jitter, self.config.glyph_vertical_jitter)

        top = max(0, baseline_y - glyph.shape[0] + vertical_jitter)
        left = min(cursor_x, canvas.shape[1] - 1)
        right = min(canvas.shape[1], left + glyph.shape[1])
        bottom = min(canvas.shape[0], top + glyph.shape[0])

        writable_glyph = glyph[: bottom - top, : right - left]
        if writable_glyph.size == 0:
            return cursor_x

        # Using np.minimum keeps darker ink while preserving the white paper.
        canvas[top:bottom, left:right] = np.minimum(canvas[top:bottom, left:right], writable_glyph)
        return right + char_gap

    @staticmethod
    def _trim_canvas(canvas: np.ndarray, margin: int = 8) -> np.ndarray:
        mask = canvas < 245
        coords = np.argwhere(mask)
        if coords.size == 0:
            return canvas

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0) + 1
        y_min = max(0, y_min - margin)
        x_min = max(0, x_min - margin)
        y_max = min(canvas.shape[0], y_max + margin)
        x_max = min(canvas.shape[1], x_max + margin)
        return canvas[y_min:y_max, x_min:x_max]
