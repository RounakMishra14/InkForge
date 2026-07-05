"""Core package for the handwritten text synthesis prototype."""

from .config import DatasetPaths, RenderConfig
from .dataset import HandwritingDataset
from .evaluation import EvaluationResult, evaluate_dataset_reconstruction
from .page_design import PageStyleConfig, image_to_png_bytes, images_to_pdf_bytes
from .paragraph import ParagraphLayoutConfig, ParagraphRenderer, RenderedDocument, RenderedPage
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
    "PageStyleConfig",
    "RenderConfig",
    "RenderedLine",
    "RenderedDocument",
    "RenderedPage",
    "SpacingProfile",
    "StyleProfile",
    "build_spacing_profile",
    "WordBank",
    "WordSample",
    "build_style_profile",
    "build_word_bank",
    "evaluate_dataset_reconstruction",
    "image_to_png_bytes",
    "images_to_pdf_bytes",
]
