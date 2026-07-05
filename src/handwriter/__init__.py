"""Core package for the handwritten text synthesis prototype."""

from .config import DatasetPaths, RenderConfig
from .dataset import HandwritingDataset
from .evaluation import EvaluationResult, evaluate_dataset_reconstruction
from .renderer import HandwritingRenderer, RenderedLine
from .style import StyleProfile, build_style_profile

__all__ = [
    "DatasetPaths",
    "EvaluationResult",
    "HandwritingDataset",
    "HandwritingRenderer",
    "RenderConfig",
    "RenderedLine",
    "StyleProfile",
    "build_style_profile",
    "evaluate_dataset_reconstruction",
]
