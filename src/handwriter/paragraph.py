"""Multi-page note layout helpers built on top of the line renderer."""

from __future__ import annotations

from dataclasses import dataclass, replace
import random
import math

import numpy as np
from PIL import Image

from .page_design import (
    PageStyleConfig,
    build_page_background,
    compose_handwriting_layer,
    draw_highlight_band,
)
from .renderer import HandwritingRenderer, RenderedLine


@dataclass(frozen=True)
class ParagraphLayoutConfig:
    """Controls for line wrapping and page-level layout."""

    max_line_width: int = 980
    page_padding_top: int = 88
    page_padding_right: int = 84
    page_padding_bottom: int = 88
    page_padding_left: int = 112
    line_spacing: int = 20
    line_margin_drift: int = 8
    body_scale: float = 1.0
    title_scale: float = 1.2
    title_spacing: int = 26
    treat_first_line_as_title: bool = False
    highlight_prefix: str = "!! "


@dataclass(frozen=True)
class LineLayout:
    """One logical line plus note-taking presentation hints."""

    text: str
    kind: str
    highlight: bool = False


@dataclass(frozen=True)
class RenderedPage:
    """One rendered note page and its line metadata."""

    image: Image.Image
    lines: list[str]
    page_number: int


@dataclass(frozen=True)
class RenderedDocument:
    """Rendered multi-page output plus layout metadata."""

    pages: list[RenderedPage]
    lines: list[str]
    unsupported_labels: list[str]
    used_word_samples: list[str]

    @property
    def primary_image(self) -> Image.Image:
        if self.pages:
            return self.pages[0].image
        return Image.new("RGB", (1, 1), "white")


class ParagraphRenderer:
    """Wrap text and stack rendered lines into shareable notebook-style pages."""

    def __init__(
        self,
        line_renderer: HandwritingRenderer,
        layout_config: ParagraphLayoutConfig | None = None,
        page_style: PageStyleConfig | None = None,
    ) -> None:
        self.line_renderer = line_renderer
        self.layout_config = layout_config or ParagraphLayoutConfig()
        self.page_style = page_style or PageStyleConfig()

    def wrap_text(self, text: str, seed: int = 7) -> list[LineLayout]:
        """Wrap free-form text into note lines using renderer width estimates."""

        logical_lines = self._expand_note_lines(text)
        wrapped_lines: list[LineLayout] = []
        for line_index, line in enumerate(logical_lines):
            normalized = " ".join(line.text.split())
            if not normalized:
                wrapped_lines.append(LineLayout(text="", kind="blank", highlight=False))
                continue

            words = normalized.split(" ")
            current_line = words[0]
            for word_index, word in enumerate(words[1:], start=1):
                candidate = f"{current_line} {word}"
                candidate_seed = seed + (line_index * 1000) + word_index
                if self._estimate_line_width(candidate, line.kind, candidate_seed) <= self.layout_config.max_line_width:
                    current_line = candidate
                else:
                    wrapped_lines.append(
                        LineLayout(
                            text=current_line,
                            kind=line.kind,
                            highlight=line.highlight and not wrapped_lines,
                        )
                    )
                    current_line = word
                    line = LineLayout(text=line.text, kind="body", highlight=False)
            wrapped_lines.append(
                LineLayout(
                    text=current_line,
                    kind=line.kind,
                    highlight=line.highlight,
                )
            )

        return wrapped_lines

    def render_document(self, text: str, seed: int = 7) -> RenderedDocument:
        """Render wrapped text as one or more notebook pages."""

        line_layouts = self.wrap_text(text=text, seed=seed)
        if not line_layouts:
            blank_page = RenderedPage(image=Image.new("RGB", (1, 1), "white"), lines=[], page_number=1)
            return RenderedDocument([blank_page], [], [], [])

        rng = random.Random(seed)
        line_results: list[tuple[LineLayout, RenderedLine, Image.Image]] = []
        unsupported_labels: list[str] = []
        used_word_samples: list[str] = []
        for index, line in enumerate(line_layouts):
            rendered_line = self._render_line_layout(line, seed=seed + (index * 137))
            line_image = rendered_line.image.convert("L")
            line_image = self._scale_line_image(line_image, self._line_scale(line.kind))
            line_results.append((line, rendered_line, line_image))
            unsupported_labels.extend(rendered_line.unsupported_labels)
            used_word_samples.extend(rendered_line.used_word_samples)

        page_limit = self.page_style.page_height - self.layout_config.page_padding_bottom
        pages: list[RenderedPage] = []
        current_items: list[tuple[LineLayout, Image.Image]] = []
        cursor_y = self.layout_config.page_padding_top
        page_number = 1

        for line, _, line_image in line_results:
            consumed_height = self._line_consumed_height(line_image.height, line.kind)
            if current_items and cursor_y + consumed_height > page_limit:
                pages.append(
                    self._compose_page(
                        page_number=page_number,
                        items=current_items,
                        seed=seed + page_number,
                    )
                )
                page_number += 1
                current_items = []
                cursor_y = self.layout_config.page_padding_top

            current_items.append((line, line_image))
            cursor_y += consumed_height

        if current_items or not pages:
            pages.append(
                self._compose_page(
                    page_number=page_number,
                    items=current_items,
                    seed=seed + page_number,
                )
            )

        return RenderedDocument(
            pages=pages,
            lines=[line.text for line in line_layouts],
            unsupported_labels=sorted(set(unsupported_labels)),
            used_word_samples=used_word_samples,
        )

    def _compose_page(
        self,
        page_number: int,
        items: list[tuple[LineLayout, Image.Image]],
        seed: int,
    ) -> RenderedPage:
        background = build_page_background(self.page_style)
        rng = random.Random(seed)
        cursor_y = self.layout_config.page_padding_top
        lines: list[str] = []

        for line, image in items:
            drift = rng.randint(-self.layout_config.line_margin_drift, self.layout_config.line_margin_drift)
            cursor_x = max(0, self.layout_config.page_padding_left + drift)
            line_top = self._line_top(cursor_y, image.height)

            if line.highlight:
                draw_highlight_band(
                    background,
                    top=max(0, line_top + 4),
                    left=max(0, cursor_x - 8),
                    width=min(image.width + 16, background.shape[1] - cursor_x),
                    height=max(28, image.height),
                    color=self.page_style.marker_color,
                    opacity=self.page_style.marker_opacity,
                )

            compose_handwriting_layer(
                background=background,
                handwriting=image,
                top=line_top,
                left=cursor_x,
                ink_color=self.page_style.ink_color,
                boldness=self.page_style.boldness,
            )
            lines.append(line.text)
            cursor_y += self._line_consumed_height(image.height, line.kind)

        return RenderedPage(
            image=Image.fromarray(background),
            lines=lines,
            page_number=page_number,
        )

    def _render_line_layout(self, line: LineLayout, seed: int) -> RenderedLine:
        if not line.text:
            blank = np.full((1, 1), 255, dtype=np.uint8)
            return RenderedLine(Image.fromarray(blank), [], [], [])

        scale = self._line_scale(line.kind)
        if scale >= 0.99:
            return self.line_renderer.render_text(line.text, seed=seed)

        expanded_renderer = self._scaled_line_renderer(scale)
        return expanded_renderer.render_text(line.text, seed=seed)

    def _estimate_line_width(self, text: str, kind: str, seed: int) -> int:
        estimated = self.line_renderer.estimate_text_width(text, seed=seed)
        return int(round(estimated * self._line_scale(kind)))

    def _line_spacing_after(self, kind: str) -> int:
        if kind == "title":
            return self.layout_config.title_spacing
        return self.layout_config.line_spacing

    def _line_consumed_height(self, image_height: int, kind: str) -> int:
        base_height = image_height + self._line_spacing_after(kind)
        if self.page_style.paper_style in {"Ruled", "Grid"}:
            return max(self.page_style.rule_spacing, base_height)
        return base_height

    def _line_top(self, cursor_y: int, image_height: int) -> int:
        if self.page_style.paper_style not in {"Ruled", "Grid"}:
            return cursor_y

        rule_spacing = max(24, self.page_style.rule_spacing)
        band_index = max(0, math.ceil(cursor_y / rule_spacing))
        band_top = band_index * rule_spacing
        centered_top = band_top + max(0, (rule_spacing - image_height) // 2)
        return max(0, centered_top)

    def _expand_note_lines(self, text: str) -> list[LineLayout]:
        blocks = text.splitlines() or [text]
        expanded: list[LineLayout] = []
        first_content_line = True
        for raw_line in blocks:
            stripped = raw_line.strip()
            if not stripped:
                expanded.append(LineLayout(text="", kind="blank"))
                continue

            highlight = False
            if stripped.startswith(self.layout_config.highlight_prefix):
                stripped = stripped[len(self.layout_config.highlight_prefix) :].strip()
                highlight = True

            if stripped.startswith("## "):
                expanded.append(
                    LineLayout(
                        text=self._normalize_note_prefix(stripped[3:].strip()),
                        kind="title",
                        highlight=highlight or self.page_style.title_highlight,
                    )
                )
                first_content_line = False
                continue

            stripped = self._normalize_note_prefix(stripped)
            if first_content_line and self.layout_config.treat_first_line_as_title:
                expanded.append(
                    LineLayout(
                        text=stripped,
                        kind="title",
                        highlight=highlight or self.page_style.title_highlight,
                    )
                )
                first_content_line = False
                continue

            expanded.append(LineLayout(text=stripped, kind="body", highlight=highlight))
            first_content_line = False
        return expanded

    @staticmethod
    def _normalize_note_prefix(text: str) -> str:
        if text.startswith("- "):
            return f"* {text[2:].strip()}"
        if text.startswith("* "):
            return f"* {text[2:].strip()}"
        if text.startswith("> "):
            return f"> {text[2:].strip()}"
        if text.startswith("[ ] "):
            return f"[ ] {text[4:].strip()}"
        if text.lower().startswith("[x] "):
            return f"[x] {text[4:].strip()}"
        return text

    def _line_scale(self, kind: str) -> float:
        if kind == "title":
            return self.layout_config.body_scale * self.layout_config.title_scale
        return self.layout_config.body_scale

    def _scaled_line_renderer(self, scale: float) -> HandwritingRenderer:
        expansion = max(1.0, 1.0 / max(scale, 0.1))
        config = self.line_renderer.config
        expanded_config = replace(
            config,
            canvas_width=max(config.canvas_width, int(round(config.canvas_width * expansion))),
            canvas_height=max(config.canvas_height, int(round(config.canvas_height * expansion))),
            padding_x=max(config.padding_x, int(round(config.padding_x * expansion))),
            padding_y=max(config.padding_y, int(round(config.padding_y * expansion))),
        )
        return HandwritingRenderer(
            dataset=self.line_renderer.dataset,
            style_profile=self.line_renderer.style_profile,
            spacing_profile=self.line_renderer.spacing_profile,
            word_bank=self.line_renderer.word_bank,
            config=expanded_config,
        )

    @staticmethod
    def _scale_line_image(image: Image.Image, scale: float) -> Image.Image:
        if abs(scale - 1.0) < 0.01:
            return image
        width = max(1, round(image.width * scale))
        height = max(1, round(image.height * scale))
        return image.resize((width, height), Image.Resampling.LANCZOS)
