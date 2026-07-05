"""Dataset quality and duplication reporting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib

from .dataset import HandwritingDataset


@dataclass(frozen=True)
class DatasetQualityReport:
    """High-level dataset hygiene summary for the current corpus."""

    total_sentence_samples: int
    unique_sentence_prompts: int
    explicit_copy_files: int
    duplicate_glyph_files: int
    duplicate_sentence_files: int
    clean_sentence_samples: int


def build_quality_report(dataset: HandwritingDataset) -> DatasetQualityReport:
    """Summarize duplicate pressure and explicit copy counts in the dataset."""

    all_sentences = dataset.sentences(include_copies=True)
    glyph_paths = [sample.path for label in dataset.supported_labels() for sample in dataset.glyphs_for(label) if label != " "]
    sentence_paths = [sample.path for sample in all_sentences]

    explicit_copy_files = sum(1 for sample in all_sentences if sample.is_explicit_copy)
    duplicate_glyph_files = _count_duplicate_files(glyph_paths)
    duplicate_sentence_files = _count_duplicate_files(sentence_paths)

    return DatasetQualityReport(
        total_sentence_samples=len(all_sentences),
        unique_sentence_prompts=len(dataset.sentence_groups(include_copies=True)),
        explicit_copy_files=explicit_copy_files,
        duplicate_glyph_files=duplicate_glyph_files,
        duplicate_sentence_files=duplicate_sentence_files,
        clean_sentence_samples=len(dataset.sentences(include_copies=False)),
    )


def _count_duplicate_files(paths: list[Path]) -> int:
    """Count files whose image content is duplicated elsewhere in the same list."""

    digest_counts: dict[str, int] = {}
    for path in paths:
        digest = hashlib.sha1(path.read_bytes()).hexdigest()
        digest_counts[digest] = digest_counts.get(digest, 0) + 1
    return sum(count - 1 for count in digest_counts.values() if count > 1)
