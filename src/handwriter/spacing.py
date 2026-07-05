"""Context-aware spacing profile derived from segmented handwritten words."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import mean

from .dataset import GlyphSample, HandwritingDataset
from .image_ops import load_grayscale, resize_to_height, tight_crop
from .style import StyleProfile
from .words import WordBank, WordSample


@dataclass(frozen=True)
class SpacingProfile:
    """Learned spacing overrides layered on top of the global style profile."""

    fallback_gap: float
    pair_gap_overrides: dict[str, float]
    class_gap_overrides: dict[str, float]
    token_baseline_jitter: int
    token_height_jitter: int

    def gap_for(self, left: str, right: str) -> float:
        """Return the best available spacing estimate for a character pair."""

        pair_key = f"{left}{right}"
        if pair_key in self.pair_gap_overrides:
            return self.pair_gap_overrides[pair_key]

        class_key = f"{classify_char(left)}->{classify_char(right)}"
        if class_key in self.class_gap_overrides:
            return self.class_gap_overrides[class_key]

        return self.fallback_gap

    def top_pairs(self, limit: int = 12) -> list[dict[str, object]]:
        """Expose the strongest learned pair overrides for UI inspection."""

        rows = [
            {"pair": pair, "gap": round(gap, 2)}
            for pair, gap in sorted(
                self.pair_gap_overrides.items(),
                key=lambda item: abs(item[1] - self.fallback_gap),
                reverse=True,
            )[:limit]
        ]
        return rows


def build_spacing_profile(
    dataset: HandwritingDataset,
    word_bank: WordBank,
    style_profile: StyleProfile,
) -> SpacingProfile:
    """Estimate pair-specific gaps by reconciling word widths with glyph widths."""

    target_height = max(24, min(72, style_profile.average_sentence_height + 10))
    glyph_widths = _estimate_glyph_widths(dataset, target_height=target_height)
    pair_observations: dict[str, list[float]] = defaultdict(list)
    class_observations: dict[str, list[float]] = defaultdict(list)
    per_word_gap_values: list[float] = []
    token_heights: list[int] = []

    for samples in word_bank.samples_by_word.values():
        for sample in samples:
            if sample.segmentation_mode != "segmented":
                continue
            word_gaps = _estimate_word_pair_gaps(sample, glyph_widths, target_height=target_height)
            if not word_gaps:
                continue

            token_heights.append(sample.image.shape[0])
            for pair_key, gap in word_gaps:
                pair_observations[pair_key].append(gap)
                class_key = f"{classify_char(pair_key[0])}->{classify_char(pair_key[1])}"
                class_observations[class_key].append(gap)
                per_word_gap_values.append(gap)

    fallback_gap = (
        float(mean(per_word_gap_values))
        if per_word_gap_values
        else style_profile.average_char_gap
    )
    pair_gap_overrides = {
        pair: float(mean(values))
        for pair, values in pair_observations.items()
        if len(values) >= 2
    }
    class_gap_overrides = {
        class_key: float(mean(values))
        for class_key, values in class_observations.items()
        if len(values) >= 2
    }

    height_jitter = 2
    if token_heights:
        height_range = max(token_heights) - min(token_heights)
        height_jitter = max(1, min(4, round(height_range / 8)))

    return SpacingProfile(
        fallback_gap=fallback_gap,
        pair_gap_overrides=pair_gap_overrides,
        class_gap_overrides=class_gap_overrides,
        token_baseline_jitter=2,
        token_height_jitter=height_jitter,
    )


def classify_char(char: str) -> str:
    """Bucket characters into coarse handwriting-relevant shape classes."""

    if char.isdigit():
        return "digit"
    if char in ".,!?":
        return "punct"
    if char in "+-*/#":
        return "symbol"
    if char in "iltfj":
        return "tall"
    if char in "mwMW":
        return "wide"
    if char in "aceosqudgpb":
        return "round"
    if char in "ygjpq":
        return "descender"
    if char.isupper():
        return "upper"
    return "lower"


def _estimate_glyph_widths(dataset: HandwritingDataset, target_height: int) -> dict[str, float]:
    widths: dict[str, list[int]] = defaultdict(list)
    for label in dataset.supported_labels():
        if label == " ":
            continue
        for glyph in dataset.glyphs_for(label):
            widths[label].append(_prepared_glyph_width(glyph, target_height=target_height))
    return {
        label: float(mean(label_widths))
        for label, label_widths in widths.items()
        if label_widths
    }


def _prepared_glyph_width(glyph: GlyphSample, target_height: int) -> int:
    image = load_grayscale(glyph.path)
    cropped = tight_crop(image, threshold=220, margin=1)
    resized = resize_to_height(cropped, target_height=target_height)
    return resized.shape[1]


def _estimate_word_pair_gaps(
    sample: WordSample,
    glyph_widths: dict[str, float],
    target_height: int,
) -> list[tuple[str, float]]:
    text = sample.text
    if len(text) < 2:
        return []

    resized_word = resize_to_height(tight_crop(sample.image, threshold=220, margin=1), target_height=target_height)
    observed_width = resized_word.shape[1]
    char_width_sum = 0.0
    for char in text:
        width = glyph_widths.get(char)
        if width is None:
            return []
        char_width_sum += width

    pair_count = len(text) - 1
    if pair_count <= 0:
        return []

    implied_gap = max(1.0, (observed_width - char_width_sum) / pair_count)
    # Clip outliers from noisy segmentation so the profile remains stable.
    implied_gap = max(1.0, min(16.0, implied_gap))
    return [(f"{left}{right}", implied_gap) for left, right in zip(text, text[1:])]
