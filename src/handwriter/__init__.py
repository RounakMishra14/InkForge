"""Core package for the handwritten text synthesis prototype."""

from .config import DatasetPaths, RenderConfig
from .dataset import HandwritingDataset
from .evaluation import EvaluationResult, evaluate_dataset_reconstruction
from .paragraph import ParagraphLayoutConfig, ParagraphRenderer, RenderedParagraph, image_to_png_bytes
from .renderer import HandwritingRenderer, RenderedLine
from .spacing import SpacingProfile, build_spacing_profile
from .style import StyleProfile, build_style_profile
from .words import WordBank, WordSample, build_word_bank

__all__ = [
    "DatasetPaths",
    "EvaluationResult",
    "HandwritingDataset",
    "HandwritingRenderer",
    "ParagraphLayoutConfig",
    "ParagraphRenderer",
    "RenderConfig",
    "RenderedLine",
    "RenderedParagraph",
    "SpacingProfile",
    "StyleProfile",
    "build_spacing_profile",
    "WordBank",
    "WordSample",
    "build_style_profile",
    "build_word_bank",
    "evaluate_dataset_reconstruction",
    "image_to_png_bytes",
]
