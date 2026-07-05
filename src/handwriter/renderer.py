"""Glyph-based handwritten line renderer."""

from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
from PIL import Image

from .config import RenderConfig
from .dataset import GlyphSample, HandwritingDataset
from .image_ops import load_grayscale, resize_to_height, tight_crop
from .spacing import SpacingProfile
from .style import StyleProfile
from .words import WordBank, prepare_word_image, split_edge_punctuation


@dataclass(frozen=True)
class RenderedLine:
    """Rendered output and trace metadata for debugging placement decisions."""

    image: Image.Image
    placed_labels: list[str]
    unsupported_labels: list[str]
    used_word_samples: list[str]


class HandwritingRenderer:
    """Compose novel text from isolated glyph samples plus style heuristics."""

    def __init__(
        self,
        dataset: HandwritingDataset,
        style_profile: StyleProfile,
        spacing_profile: SpacingProfile | None = None,
        word_bank: WordBank | None = None,
        config: RenderConfig | None = None,
    ) -> None:
        self.dataset = dataset
        self.style_profile = style_profile
        self.spacing_profile = spacing_profile
        self.word_bank = word_bank
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
        used_word_samples: list[str] = []
        cursor_x = self.config.padding_x
        line_baseline_y = int(
            self.config.padding_y
            + self.style_profile.average_sentence_height
            + rng.randint(-self.config.baseline_jitter, self.config.baseline_jitter)
        )

        tokens = text.split(" ")
        for token_index, token in enumerate(tokens):
            if token:
                token_baseline_y = line_baseline_y + rng.randint(
                    -self._token_baseline_jitter(),
                    self._token_baseline_jitter(),
                )
                token_cursor, token_placed, token_unsupported, token_word_sample = self._render_token(
                    canvas=canvas,
                    token=token,
                    cursor_x=cursor_x,
                    baseline_y=token_baseline_y,
                    rng=rng,
                )
                cursor_x = token_cursor
                placed_labels.extend(token_placed)
                unsupported.extend(token_unsupported)
                if token_word_sample:
                    used_word_samples.append(token_word_sample)
            if token_index < len(tokens) - 1:
                cursor_x += max(self.config.default_word_gap, int(round(self.style_profile.average_word_gap)))

        final = self._trim_canvas(canvas)
        return RenderedLine(
            image=Image.fromarray(final),
            placed_labels=placed_labels,
            unsupported_labels=unsupported,
            used_word_samples=used_word_samples,
        )

    def estimate_text_width(self, text: str, seed: int = 7) -> int:
        """Estimate rendered width using the same sampling path as the line renderer."""

        rng = random.Random(seed)
        cursor_x = self.config.padding_x
        tokens = text.split(" ")
        for token_index, token in enumerate(tokens):
            if token:
                cursor_x = self._estimate_token_width(token=token, cursor_x=cursor_x, rng=rng)
            if token_index < len(tokens) - 1:
                cursor_x += max(self.config.default_word_gap, int(round(self.style_profile.average_word_gap)))
        return max(0, cursor_x - self.config.padding_x)

    def _render_token(
        self,
        canvas: np.ndarray,
        token: str,
        cursor_x: int,
        baseline_y: int,
        rng: random.Random,
    ) -> tuple[int, list[str], list[str], str | None]:
        """Render one token either from the word bank or from glyph composition."""

        if self.config.prefer_word_bank and self.word_bank is not None:
            word_sample = self.word_bank.sample_for(token, rng)
            if word_sample is not None:
                word_image = prepare_word_image(
                    word_sample,
                    target_height=self._target_word_height(word_sample),
                )
                next_cursor = self._paste_glyph(canvas, word_image, cursor_x, baseline_y, rng, gap_after=0)
                return next_cursor, list(token), [], word_sample.text

            leading, core, trailing = split_edge_punctuation(token)
            core_sample = self.word_bank.sample_for(core, rng) if core and core != token else None
            if core_sample is not None:
                cursor_x, placed_labels, unsupported = self._render_char_sequence(
                    canvas=canvas,
                    text=leading,
                    cursor_x=cursor_x,
                    baseline_y=baseline_y,
                    rng=rng,
                    next_char=core[0],
                )
                word_image = prepare_word_image(
                    core_sample,
                    target_height=self._target_word_height(core_sample),
                )
                cursor_x = self._paste_glyph(canvas, word_image, cursor_x, baseline_y, rng, gap_after=0)
                cursor_x, trailing_labels, trailing_unsupported = self._render_char_sequence(
                    canvas=canvas,
                    text=trailing,
                    cursor_x=cursor_x,
                    baseline_y=baseline_y,
                    rng=rng,
                    previous_char=core[-1],
                )
                placed_labels.extend(list(core))
                placed_labels.extend(trailing_labels)
                unsupported.extend(trailing_unsupported)
                return cursor_x, placed_labels, unsupported, core_sample.text

        placed_labels: list[str] = []
        unsupported: list[str] = []
        cursor_x, placed_labels, unsupported = self._render_char_sequence(
            canvas=canvas,
            text=token,
            cursor_x=cursor_x,
            baseline_y=baseline_y,
            rng=rng,
        )

        return cursor_x, placed_labels, unsupported, None

    def _estimate_token_width(
        self,
        token: str,
        cursor_x: int,
        rng: random.Random,
    ) -> int:
        """Estimate token width without creating a line canvas."""

        if self.config.prefer_word_bank and self.word_bank is not None:
            word_sample = self.word_bank.sample_for(token, rng)
            if word_sample is not None:
                word_image = prepare_word_image(
                    word_sample,
                    target_height=self._target_word_height(word_sample),
                )
                return cursor_x + word_image.shape[1]

            leading, core, trailing = split_edge_punctuation(token)
            core_sample = self.word_bank.sample_for(core, rng) if core and core != token else None
            if core_sample is not None:
                cursor_x = self._estimate_char_sequence_width(
                    text=leading,
                    cursor_x=cursor_x,
                    rng=rng,
                    next_char=core[0],
                )
                word_image = prepare_word_image(
                    core_sample,
                    target_height=self._target_word_height(core_sample),
                )
                cursor_x += word_image.shape[1]
                return self._estimate_char_sequence_width(
                    text=trailing,
                    cursor_x=cursor_x,
                    rng=rng,
                    previous_char=core[-1],
                )

        return self._estimate_char_sequence_width(text=token, cursor_x=cursor_x, rng=rng)

    def _choose_glyph(self, char: str, rng: random.Random) -> GlyphSample | None:
        choices = self.dataset.glyphs_for(char)
        if not choices and char == "*":
            choices = self.dataset.glyphs_for("x")
        if not choices:
            return None
        return rng.choice(choices)

    def _render_char_sequence(
        self,
        canvas: np.ndarray,
        text: str,
        cursor_x: int,
        baseline_y: int,
        rng: random.Random,
        next_char: str | None = None,
        previous_char: str | None = None,
    ) -> tuple[int, list[str], list[str]]:
        """Render a short glyph sequence, optionally bridging spacing from adjacent parts."""

        placed_labels: list[str] = []
        unsupported: list[str] = []
        if previous_char is not None and text:
            cursor_x += self._gap_after(previous_char, text[0], rng)

        for index, char in enumerate(text):
            glyph = self._choose_glyph(char, rng)
            if glyph is None:
                unsupported.append(char)
                cursor_x += max(self.config.default_word_gap, int(round(self.style_profile.average_word_gap / 2)))
                continue

            glyph_image = self._prepare_glyph(glyph, target_height=self._target_glyph_height(rng))
            gap_after = 0
            if index < len(text) - 1:
                gap_after = self._gap_after(char, text[index + 1], rng)
            elif next_char is not None:
                gap_after = self._gap_after(char, next_char, rng)
            cursor_x = self._paste_glyph(canvas, glyph_image, cursor_x, baseline_y, rng, gap_after=gap_after)
            placed_labels.append(char)

        return cursor_x, placed_labels, unsupported

    def _estimate_char_sequence_width(
        self,
        text: str,
        cursor_x: int,
        rng: random.Random,
        next_char: str | None = None,
        previous_char: str | None = None,
    ) -> int:
        """Mirror glyph sequence rendering when only width estimation is needed."""

        if previous_char is not None and text:
            cursor_x += self._gap_after(previous_char, text[0], rng)

        for index, char in enumerate(text):
            glyph = self._choose_glyph(char, rng)
            if glyph is None:
                cursor_x += max(self.config.default_word_gap, int(round(self.style_profile.average_word_gap / 2)))
                continue

            glyph_image = self._prepare_glyph(glyph, target_height=self._target_glyph_height(rng))
            cursor_x += glyph_image.shape[1]
            if index < len(text) - 1:
                cursor_x += self._gap_after(char, text[index + 1], rng)
            elif next_char is not None:
                cursor_x += self._gap_after(char, next_char, rng)

        return cursor_x

    def _prepare_glyph(self, glyph: GlyphSample, target_height: int) -> np.ndarray:
        image = load_grayscale(glyph.path)
        cropped = tight_crop(image, threshold=220, margin=1)

        # Scale the isolated glyphs into the vertical footprint implied by the
        # sentence-level examples so the composition feels closer to the writer.
        adjusted_height = self._glyph_target_height(glyph.label, target_height)
        return resize_to_height(cropped, target_height=adjusted_height)

    def _paste_glyph(
        self,
        canvas: np.ndarray,
        glyph: np.ndarray,
        cursor_x: int,
        baseline_y: int,
        rng: random.Random,
        gap_after: int,
    ) -> int:
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
        return right + gap_after

    def _gap_after(self, left: str, right: str, rng: random.Random) -> int:
        if self.config.use_context_spacing and self.spacing_profile is not None:
            base_gap = self.spacing_profile.gap_for(left, right)
            jitter_max = max(1, self.config.max_char_gap_jitter - 1)
            return max(self.config.min_char_gap, int(round(base_gap + rng.randint(0, jitter_max))))

        return max(
            self.config.min_char_gap,
            int(round(self.style_profile.average_char_gap)) + rng.randint(0, self.config.max_char_gap_jitter),
        )

    def _target_glyph_height(self, rng: random.Random) -> int:
        base_height = self._base_glyph_height()
        height_jitter = self._token_height_jitter()
        return max(24, min(72, base_height + rng.randint(-height_jitter, height_jitter)))

    def _base_glyph_height(self) -> int:
        return max(24, min(72, self.style_profile.average_sentence_height + 10))

    def _target_word_height(self, word_sample) -> int:
        """Slightly normalize exact word-bank samples against the bank average."""

        base_height = self._base_glyph_height()
        if self.word_bank is None:
            return base_height

        sample_height = max(1, word_sample.image.shape[0])
        average_height = max(1, self.word_bank.average_word_height)
        correction = (average_height / sample_height) ** 0.25
        correction = min(1.18, max(0.92, correction))
        return max(24, min(72, round(base_height * correction)))

    @staticmethod
    def _glyph_target_height(label: str, target_height: int) -> int:
        """Keep punctuation visually smaller than full letter glyphs."""

        if label == ".":
            return max(8, round(target_height * 0.3))
        if label == ",":
            return max(10, round(target_height * 0.38))
        if label in {"!", "?"}:
            return max(18, round(target_height * 0.78))
        return target_height

    def _token_baseline_jitter(self) -> int:
        if self.spacing_profile is not None:
            return self.spacing_profile.token_baseline_jitter
        return self.config.token_baseline_jitter

    def _token_height_jitter(self) -> int:
        if self.spacing_profile is not None:
            return self.spacing_profile.token_height_jitter
        return self.config.token_height_jitter

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
