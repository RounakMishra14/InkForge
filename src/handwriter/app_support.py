"""Shared app-side helpers for building pipeline state and render outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .config import DatasetPaths, RenderConfig
from .dataset import HandwritingDataset
from .page_design import PageStyleConfig
from .paragraph import ParagraphLayoutConfig, ParagraphRenderer, RenderedDocument
from .preview import stack_horizontal
from .quality import DatasetQualityReport, build_quality_report
from .renderer import HandwritingRenderer
from .spacing import SpacingProfile, build_spacing_profile
from .style import StyleProfile, build_style_profile
from .words import WordBank, build_word_bank


@dataclass(frozen=True)
class PipelineBundle:
    """Long-lived objects shared across Streamlit reruns."""

    dataset: HandwritingDataset
    style_profile: StyleProfile
    spacing_profile: SpacingProfile
    word_bank: WordBank
    context_renderer: HandwritingRenderer
    flat_word_bank_renderer: HandwritingRenderer
    glyph_renderer: HandwritingRenderer
    quality_report: DatasetQualityReport


@dataclass(frozen=True)
class ParagraphRenderBundle:
    """Page-level render outputs used across multiple UI sections."""

    context: RenderedDocument
    flat_word_bank: RenderedDocument
    glyph_only: RenderedDocument
    available_words: list[str]
    missing_words: list[str]

    @property
    def unsupported_labels(self) -> list[str]:
        labels = set(
            self.context.unsupported_labels
            + self.flat_word_bank.unsupported_labels
            + self.glyph_only.unsupported_labels
        )
        return sorted(labels)

    @property
    def page_count(self) -> int:
        return len(self.context.pages)

    def pages_for_mode(self, render_mode: str) -> list:
        if render_mode == "Context-aware":
            return self.context.pages
        if render_mode == "Flat word-bank":
            return self.flat_word_bank.pages
        if render_mode == "Glyph only":
            return self.glyph_only.pages

        max_pages = max(len(self.context.pages), len(self.flat_word_bank.pages), len(self.glyph_only.pages))
        comparison_pages = []
        for index in range(max_pages):
            page_images = [
                _page_image_or_blank(self.context.pages, index),
                _page_image_or_blank(self.flat_word_bank.pages, index),
                _page_image_or_blank(self.glyph_only.pages, index),
            ]
            comparison_pages.append(
                stack_horizontal(page_images, gap=28)
            )
        return comparison_pages


def build_pipeline(project_root: Path) -> PipelineBundle:
    """Create the reusable dataset, style profile, and renderer stack."""

    paths = DatasetPaths(root=project_root)
    dataset = HandwritingDataset(paths=paths)
    style_profile = build_style_profile(dataset=dataset, include_copies=False)
    word_bank = build_word_bank(
        dataset=dataset,
        include_copies=False,
        include_fallback_samples=False,
    )
    spacing_profile = build_spacing_profile(
        dataset=dataset,
        word_bank=word_bank,
        style_profile=style_profile,
    )
    context_renderer = HandwritingRenderer(
        dataset=dataset,
        style_profile=style_profile,
        spacing_profile=spacing_profile,
        word_bank=word_bank,
        config=RenderConfig(use_context_spacing=True),
    )
    flat_word_bank_renderer = HandwritingRenderer(
        dataset=dataset,
        style_profile=style_profile,
        spacing_profile=None,
        word_bank=word_bank,
        config=RenderConfig(use_context_spacing=False),
    )
    glyph_renderer = HandwritingRenderer(
        dataset=dataset,
        style_profile=style_profile,
        spacing_profile=None,
        word_bank=None,
        config=RenderConfig(prefer_word_bank=False),
    )
    quality_report = build_quality_report(dataset)
    return PipelineBundle(
        dataset=dataset,
        style_profile=style_profile,
        spacing_profile=spacing_profile,
        word_bank=word_bank,
        context_renderer=context_renderer,
        flat_word_bank_renderer=flat_word_bank_renderer,
        glyph_renderer=glyph_renderer,
        quality_report=quality_report,
    )


def render_paragraph_bundle(
    pipeline: PipelineBundle,
    input_text: str,
    seed: int,
    layout_config: ParagraphLayoutConfig,
    page_style: PageStyleConfig,
) -> ParagraphRenderBundle:
    """Render all output modes once so the UI can reuse the results."""

    available_words, missing_words = pipeline.word_bank.coverage(input_text)

    context_paragraph_renderer = ParagraphRenderer(
        pipeline.context_renderer,
        layout_config=layout_config,
        page_style=page_style,
    )
    flat_paragraph_renderer = ParagraphRenderer(
        pipeline.flat_word_bank_renderer,
        layout_config=layout_config,
        page_style=page_style,
    )
    glyph_paragraph_renderer = ParagraphRenderer(
        pipeline.glyph_renderer,
        layout_config=layout_config,
        page_style=page_style,
    )

    return ParagraphRenderBundle(
        context=context_paragraph_renderer.render_document(input_text, seed=seed),
        flat_word_bank=flat_paragraph_renderer.render_document(input_text, seed=seed),
        glyph_only=glyph_paragraph_renderer.render_document(input_text, seed=seed),
        available_words=available_words,
        missing_words=missing_words,
    )


def _page_image_or_blank(pages: list, index: int):
    if index < len(pages):
        return pages[index].image
    return Image.new("RGB", (100, 140), "white")
