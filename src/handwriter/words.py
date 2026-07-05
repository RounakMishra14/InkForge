"""Word-level extraction from sentence crops."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import mean
import random

import numpy as np

from .dataset import HandwritingDataset, SentenceSample
from .image_ops import load_grayscale, resize_to_height, tight_crop, to_ink_mask


@dataclass(frozen=True)
class WordSample:
    """A segmented handwritten word extracted from a sentence crop."""

    text: str
    source_sentence: str
    source_file: str
    image: np.ndarray
    segmentation_mode: str
    segment_index: int


@dataclass(frozen=True)
class WordBank:
    """Lookup table for exact handwritten words seen in the sentence dataset."""

    samples_by_word: dict[str, list[WordSample]]
    average_word_height: int

    def available_words(self) -> list[str]:
        return sorted(self.samples_by_word.keys())

    def coverage(self, text: str) -> tuple[list[str], list[str]]:
        """Split requested tokens into retrievable words and fallback words."""

        available: list[str] = []
        missing: list[str] = []
        for token in text.split():
            normalized = token.lower()
            if normalized in self.samples_by_word:
                available.append(token)
            else:
                missing.append(token)
        return available, missing

    def sample_for(self, word: str, rng: random.Random) -> WordSample | None:
        choices = self.samples_by_word.get(word.lower())
        if not choices:
            return None
        return rng.choice(choices)

    def samples_for(self, word: str) -> list[WordSample]:
        """Return all stored samples for a normalized word token."""

        return list(self.samples_by_word.get(word.lower(), []))

    def inventory_rows(self) -> list[dict[str, object]]:
        """Build a compact inventory useful for UI tables and reporting."""

        rows: list[dict[str, object]] = []
        for word, samples in sorted(self.samples_by_word.items()):
            segmented = sum(1 for sample in samples if sample.segmentation_mode == "segmented")
            fallback = sum(1 for sample in samples if sample.segmentation_mode != "segmented")
            rows.append(
                {
                    "word": word,
                    "samples": len(samples),
                    "segmented_samples": segmented,
                    "fallback_samples": fallback,
                }
            )
        return rows


def build_word_bank(
    dataset: HandwritingDataset,
    include_copies: bool = False,
    include_fallback_samples: bool = False,
    min_samples_per_word: int = 3,
) -> WordBank:
    """Extract a reusable bank of handwritten word crops from sentence images."""

    raw_samples_by_word: dict[str, list[WordSample]] = defaultdict(list)

    for sample in dataset.sentences(include_copies=include_copies):
        for word_sample in extract_words_from_sentence(sample):
            if not include_fallback_samples and word_sample.segmentation_mode != "segmented":
                continue
            if not _is_valid_word_sample(word_sample):
                continue
            raw_samples_by_word[word_sample.text.lower()].append(word_sample)

    samples_by_word: dict[str, list[WordSample]] = {}
    heights: list[int] = []
    for word, samples in raw_samples_by_word.items():
        filtered_samples = _filter_word_sample_outliers(samples)
        if len(filtered_samples) < min_samples_per_word:
            continue
        samples_by_word[word] = filtered_samples
        heights.extend(sample.image.shape[0] for sample in filtered_samples)

    return WordBank(
        samples_by_word=dict(samples_by_word),
        average_word_height=int(round(mean(heights))) if heights else 32,
    )


def extract_words_from_sentence(sample: SentenceSample) -> list[WordSample]:
    """Segment a sentence crop into word crops using transcript-aware gap selection."""

    expected_words = sample.text.split()
    if len(expected_words) <= 1:
        return [_single_word_sample(sample, expected_words[0] if expected_words else sample.text, 0)]

    image = load_grayscale(sample.path)
    word_images = _segment_word_images(image=image, expected_word_count=len(expected_words))
    if len(word_images) != len(expected_words):
        # A graceful fallback keeps the pipeline deterministic even when the
        # simple gap-based segmentation misses a boundary.
        return [_single_word_sample(sample, word, index) for index, word in enumerate(expected_words)]

    extracted: list[WordSample] = []
    for index, (word, word_image) in enumerate(zip(expected_words, word_images)):
        extracted.append(
            WordSample(
                text=word,
                source_sentence=sample.text,
                source_file=sample.path.name,
                image=word_image,
                segmentation_mode="segmented",
                segment_index=index,
            )
        )
    return extracted


def _single_word_sample(sample: SentenceSample, word: str, segment_index: int) -> WordSample:
    image = tight_crop(load_grayscale(sample.path), threshold=220, margin=2)
    return WordSample(
        text=word,
        source_sentence=sample.text,
        source_file=sample.path.name,
        image=image,
        segmentation_mode="fallback_sentence_crop",
        segment_index=segment_index,
    )


def _segment_word_images(image: np.ndarray, expected_word_count: int) -> list[np.ndarray]:
    mask = to_ink_mask(image)
    col_sums = mask.sum(axis=0)
    gaps = _find_zero_runs(col_sums == 0)
    if expected_word_count <= 1:
        return [tight_crop(image, threshold=220, margin=2)]

    needed_boundaries = expected_word_count - 1
    if len(gaps) < needed_boundaries:
        return [tight_crop(image, threshold=220, margin=2)]

    chosen_gaps = sorted(gaps, key=lambda gap: gap[1] - gap[0], reverse=True)[:needed_boundaries]
    chosen_gaps = sorted(chosen_gaps, key=lambda gap: gap[0])

    boundaries: list[tuple[int, int]] = []
    start = 0
    for gap_start, gap_end in chosen_gaps:
        boundaries.append((start, gap_start))
        start = gap_end
    boundaries.append((start, image.shape[1]))

    words: list[np.ndarray] = []
    for start_col, end_col in boundaries:
        slice_image = image[:, start_col:end_col]
        cropped = tight_crop(slice_image, threshold=220, margin=2)
        if cropped.size > 0:
            words.append(cropped)
    return words


def prepare_word_image(word_sample: WordSample, target_height: int) -> np.ndarray:
    """Resize a word crop into the shared sentence-height footprint."""

    cropped = tight_crop(word_sample.image, threshold=220, margin=1)
    return resize_to_height(cropped, target_height=target_height)


def _is_valid_word_sample(word_sample: WordSample) -> bool:
    """Filter obviously broken word segments out of the reusable word bank."""

    cropped = tight_crop(word_sample.image, threshold=220, margin=1)
    height, width = cropped.shape
    min_width = max(12, len(word_sample.text) * 4)
    if height < 14 or width < min_width:
        return False
    if height <= 2 or width <= 2:
        return False
    return True


def _filter_word_sample_outliers(samples: list[WordSample]) -> list[WordSample]:
    """Drop obviously stretched or squashed samples within the same word bucket."""

    if len(samples) <= 2:
        return list(samples)

    ratios = []
    for sample in samples:
        cropped = tight_crop(sample.image, threshold=220, margin=1)
        height, width = cropped.shape
        ratios.append(width / max(1, height))

    median_ratio = sorted(ratios)[len(ratios) // 2]
    filtered: list[WordSample] = []
    for sample, ratio in zip(samples, ratios):
        if ratio > median_ratio * 1.7:
            continue
        if ratio < median_ratio * 0.55:
            continue
        filtered.append(sample)

    return filtered if filtered else list(samples)


def _find_zero_runs(is_zero_column: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(is_zero_column.tolist()):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(is_zero_column)))
    return runs
