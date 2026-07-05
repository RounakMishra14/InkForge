"""Sentence-level handwriting style analysis utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean

import numpy as np

from .dataset import HandwritingDataset, SentenceSample
from .image_ops import load_grayscale, to_ink_mask


@dataclass(frozen=True)
class SentenceStyleMetrics:
    """Measurements extracted from one sentence crop."""

    text: str
    path_name: str
    width: int
    height: int
    baseline_row: int
    top_row: int
    ink_density: float
    average_char_gap: float
    average_word_gap: float
    content_width: int


@dataclass(frozen=True)
class StyleProfile:
    """Aggregated style settings used by the renderer."""

    average_sentence_height: int
    median_baseline_ratio: float
    average_char_gap: float
    average_word_gap: float
    average_ink_density: float
    supported_sentence_count: int
    analyzed_samples: list[SentenceStyleMetrics]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["analyzed_samples"] = [asdict(metric) for metric in self.analyzed_samples]
        return payload


def analyze_sentence(sample: SentenceSample) -> SentenceStyleMetrics:
    """Extract a compact set of spacing and baseline statistics from a crop."""

    image = load_grayscale(sample.path)
    mask = to_ink_mask(image)
    rows = np.where(mask.sum(axis=1) > 0)[0]
    cols = np.where(mask.sum(axis=0) > 0)[0]

    if rows.size == 0 or cols.size == 0:
        return SentenceStyleMetrics(
            text=sample.text,
            path_name=sample.path.name,
            width=image.shape[1],
            height=image.shape[0],
            baseline_row=image.shape[0] - 1,
            top_row=0,
            ink_density=0.0,
            average_char_gap=0.0,
            average_word_gap=0.0,
            content_width=0,
        )

    row_sums = mask.sum(axis=1)
    baseline_row = int(row_sums.argmax())
    top_row = int(rows.min())
    content_width = int(cols.max() - cols.min() + 1)
    ink_density = float(mask.mean())

    col_sums = mask.sum(axis=0)
    gaps = _collect_zero_runs(col_sums == 0)

    # Long runs are a practical proxy for word boundaries in this dataset.
    word_gaps = [gap for gap in gaps if gap >= 12]
    char_gaps = [gap for gap in gaps if 1 <= gap < 12]

    return SentenceStyleMetrics(
        text=sample.text,
        path_name=sample.path.name,
        width=image.shape[1],
        height=image.shape[0],
        baseline_row=baseline_row,
        top_row=top_row,
        ink_density=ink_density,
        average_char_gap=float(mean(char_gaps)) if char_gaps else 0.0,
        average_word_gap=float(mean(word_gaps)) if word_gaps else 0.0,
        content_width=content_width,
    )


def build_style_profile(dataset: HandwritingDataset, include_copies: bool = False) -> StyleProfile:
    """Aggregate representative spacing and layout statistics for the writer."""

    analyzed_samples = [
        analyze_sentence(sample)
        for sample in dataset.sentences(include_copies=include_copies)
    ]

    sentence_heights = [metric.height for metric in analyzed_samples]
    baselines = [metric.baseline_row / max(1, metric.height) for metric in analyzed_samples]
    char_gaps = [metric.average_char_gap for metric in analyzed_samples if metric.average_char_gap > 0]
    word_gaps = [metric.average_word_gap for metric in analyzed_samples if metric.average_word_gap > 0]
    densities = [metric.ink_density for metric in analyzed_samples]

    return StyleProfile(
        average_sentence_height=int(round(mean(sentence_heights))) if sentence_heights else 0,
        median_baseline_ratio=float(np.median(baselines)) if baselines else 0.75,
        average_char_gap=float(mean(char_gaps)) if char_gaps else 4.0,
        average_word_gap=float(mean(word_gaps)) if word_gaps else 22.0,
        average_ink_density=float(mean(densities)) if densities else 0.0,
        supported_sentence_count=len({metric.text for metric in analyzed_samples}),
        analyzed_samples=analyzed_samples,
    )


def _collect_zero_runs(is_zero_column: np.ndarray) -> list[int]:
    """Return lengths of contiguous zero-valued projection runs."""

    runs: list[int] = []
    current = 0
    for value in is_zero_column.tolist():
        if value:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs
