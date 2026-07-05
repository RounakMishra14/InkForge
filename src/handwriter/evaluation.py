"""Evaluation helpers for measuring the current baseline renderer."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

import numpy as np
from PIL import Image

from .dataset import HandwritingDataset, SentenceSample
from .image_ops import binary_iou, load_grayscale, normalized_mae, pad_to_same_size
from .renderer import HandwritingRenderer


@dataclass(frozen=True)
class SentenceEvaluation:
    """Per-sample reconstruction metrics for a held-out sentence crop."""

    text: str
    path_name: str
    iou: float
    mae: float
    unsupported_count: int


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregate metrics that act as an initial quality proxy."""

    mean_iou: float
    mean_mae: float
    sample_count: int
    details: list[SentenceEvaluation]


def evaluate_dataset_reconstruction(
    dataset: HandwritingDataset,
    renderer: HandwritingRenderer,
    max_samples_per_sentence: int = 2,
    include_copies: bool = False,
) -> EvaluationResult:
    """Render known transcripts and compare them with held-out sentence crops."""

    details: list[SentenceEvaluation] = []
    for text, samples in dataset.sentence_groups(include_copies=include_copies).items():
        for sample_index, sample in enumerate(samples[:max_samples_per_sentence]):
            seed = _stable_seed(text=text, salt=sample_index)
            rendered = renderer.render_text(text, seed=seed)
            reference = load_grayscale(sample.path)
            candidate = _to_grayscale_array(rendered.image)
            candidate, reference = pad_to_same_size(candidate, reference)

            details.append(
                SentenceEvaluation(
                    text=text,
                    path_name=sample.path.name,
                    iou=binary_iou(candidate, reference),
                    mae=normalized_mae(candidate, reference),
                    unsupported_count=len(rendered.unsupported_labels),
                )
            )

    return EvaluationResult(
        mean_iou=float(mean([detail.iou for detail in details])) if details else 0.0,
        mean_mae=float(mean([detail.mae for detail in details])) if details else 0.0,
        sample_count=len(details),
        details=details,
    )


def _stable_seed(text: str, salt: int) -> int:
    return sum(ord(char) for char in text) + (salt * 97)


def _to_grayscale_array(image: Image.Image):
    return load_pil_grayscale(image)


def load_pil_grayscale(image: Image.Image):
    return np.array(image.convert("L"), dtype=np.uint8)
