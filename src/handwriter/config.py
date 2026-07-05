"""Configuration primitives used across the project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetPaths:
    """Resolved dataset locations relative to the project root."""

    root: Path

    @property
    def dataset_dir(self) -> Path:
        return self.root / "Dataset"

    @property
    def characters_dir(self) -> Path:
        return self.dataset_dir / "characters"

    @property
    def numbers_dir(self) -> Path:
        return self.dataset_dir / "numbers"

    @property
    def symbols_dir(self) -> Path:
        return self.dataset_dir / "symbols"

    @property
    def sentences_dir(self) -> Path:
        return self.dataset_dir / "sentences"

    @property
    def raw_pages(self) -> list[Path]:
        return sorted(self.dataset_dir.glob("handwritten*.JPG"))


@dataclass(frozen=True)
class RenderConfig:
    """Tunables for line rendering."""

    canvas_width: int = 1400
    canvas_height: int = 220
    padding_x: int = 24
    padding_y: int = 24
    min_char_gap: int = 3
    max_char_gap_jitter: int = 4
    default_word_gap: int = 22
    baseline_jitter: int = 3
    glyph_vertical_jitter: int = 2
    output_background: int = 255
