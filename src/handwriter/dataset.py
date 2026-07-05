"""Dataset discovery and label normalization helpers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import DatasetPaths

SYMBOL_LABELS = {
    "addition": "+",
    "comma": ",",
    "division": "/",
    "exclamation_point": "!",
    "hashtag": "#",
    "multiplication": "*",
    "period": ".",
    "question_mark": "?",
    "substraction": "-",
}


@dataclass(frozen=True)
class GlyphSample:
    """Single glyph image and the normalized text label it represents."""

    label: str
    path: Path
    source_group: str


@dataclass(frozen=True)
class SentenceSample:
    """Sentence-level handwriting sample with transcript from the folder name."""

    text: str
    path: Path

    @property
    def is_explicit_copy(self) -> bool:
        return "copy" in self.path.stem.lower()


class HandwritingDataset:
    """Catalogs glyph and sentence assets under the project dataset."""

    def __init__(self, paths: DatasetPaths) -> None:
        self.paths = paths
        self._glyph_index = self._build_glyph_index()
        self._sentence_index = self._build_sentence_index()

    def supported_labels(self) -> list[str]:
        labels = list(self._glyph_index.keys()) + [" "]
        return sorted(set(labels))

    def glyphs_for(self, label: str) -> list[GlyphSample]:
        return list(self._glyph_index.get(label, []))

    def sentences(self, include_copies: bool = True) -> list[SentenceSample]:
        if include_copies:
            return list(self._sentence_index)
        return [sample for sample in self._sentence_index if not sample.is_explicit_copy]

    def sentence_groups(self, include_copies: bool = True) -> dict[str, list[SentenceSample]]:
        grouped: dict[str, list[SentenceSample]] = defaultdict(list)
        for sample in self.sentences(include_copies=include_copies):
            grouped[sample.text].append(sample)
        return dict(grouped)

    def coverage_report(self) -> dict[str, int]:
        return {label: len(samples) for label, samples in sorted(self._glyph_index.items())}

    def _build_glyph_index(self) -> dict[str, list[GlyphSample]]:
        glyphs: dict[str, list[GlyphSample]] = defaultdict(list)
        for source in (
            self._scan_character_glyphs(),
            self._scan_number_glyphs(),
            self._scan_symbol_glyphs(),
        ):
            for label, samples in source.items():
                glyphs[label].extend(samples)
        return dict(glyphs)

    def _scan_character_glyphs(self) -> dict[str, list[GlyphSample]]:
        glyphs: dict[str, list[GlyphSample]] = defaultdict(list)
        for folder in sorted(self.paths.characters_dir.iterdir()):
            if not folder.is_dir():
                continue
            label = self._decode_character_folder(folder.name)
            for path in sorted(folder.glob("*.png")):
                glyphs[label].append(GlyphSample(label=label, path=path, source_group=folder.name))
        return dict(glyphs)

    def _scan_number_glyphs(self) -> dict[str, list[GlyphSample]]:
        glyphs: dict[str, list[GlyphSample]] = defaultdict(list)
        for folder in sorted(self.paths.numbers_dir.iterdir()):
            if not folder.is_dir():
                continue
            label = folder.name
            for path in sorted(folder.glob("*.png")):
                glyphs[label].append(GlyphSample(label=label, path=path, source_group=folder.name))
        return dict(glyphs)

    def _scan_symbol_glyphs(self) -> dict[str, list[GlyphSample]]:
        glyphs: dict[str, list[GlyphSample]] = defaultdict(list)
        for folder in sorted(self.paths.symbols_dir.iterdir()):
            if not folder.is_dir():
                continue
            label = SYMBOL_LABELS.get(folder.name)
            if label is None:
                continue
            for path in sorted(folder.glob("*.png")):
                glyphs[label].append(GlyphSample(label=label, path=path, source_group=folder.name))
        return dict(glyphs)

    def _build_sentence_index(self) -> list[SentenceSample]:
        samples: list[SentenceSample] = []
        for folder in sorted(self.paths.sentences_dir.iterdir()):
            if not folder.is_dir():
                continue
            for path in sorted(folder.glob("*.png")):
                samples.append(SentenceSample(text=folder.name, path=path))
        return samples

    @staticmethod
    def _decode_character_folder(folder_name: str) -> str:
        prefix, _, suffix = folder_name.partition("_")
        if prefix == "lowercase":
            return suffix
        if prefix == "uppercase":
            return suffix.upper()
        raise ValueError(f"Unsupported character folder name: {folder_name}")


def iter_unique_sentence_text(dataset: HandwritingDataset) -> Iterable[str]:
    """Yield each sentence transcript once while preserving display order."""

    seen: set[str] = set()
    for sample in dataset.sentences(include_copies=True):
        if sample.text in seen:
            continue
        seen.add(sample.text)
        yield sample.text
