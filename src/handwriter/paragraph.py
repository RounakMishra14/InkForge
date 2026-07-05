"""Multi-line paragraph layout and page rendering helpers."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import random

import numpy as np
from PIL import Image

from .renderer import HandwritingRenderer, RenderedLine


@dataclass(frozen=True)
class ParagraphLayoutConfig:
    """Controls for line wrapping and page-level layout."""

    max_line_width: int = 1100
    page_padding_x: int = 40
    page_padding_y: int = 32
    line_spacing: int = 22
    line_margin_drift: int = 8
    background_value: int = 255


@dataclass(frozen=True)
class RenderedParagraph:
    """Rendered multi-line output plus layout metadata."""

    image: Image.Image
    lines: list[str]
    line_images: list[Image.Image]
    unsupported_labels: list[str]
    used_word_samples: list[str]


class ParagraphRenderer:
    """Wrap text and stack rendered lines into a shareable page image."""

    def __init__(
        self,
        line_renderer: HandwritingRenderer,
        layout_config: ParagraphLayoutConfig | None = None,
    ) -> None:
        self.line_renderer = line_renderer
        self.layout_config = layout_config or ParagraphLayoutConfig()

    def wrap_text(self, text: str, seed: int = 7) -> list[str]:
        """Wrap free-form text into line-sized chunks using renderer width estimates."""

        wrapped_lines: list[str] = []
        paragraph_blocks = text.splitlines() or [text]
        for paragraph_index, block in enumerate(paragraph_blocks):
            normalized = " ".join(block.split())
            if not normalized:
                wrapped_lines.append("")
                continue

            words = normalized.split(" ")
            current_line = words[0]
            for word_index, word in enumerate(words[1:], start=1):
                candidate = f"{current_line} {word}"
                candidate_seed = seed + (paragraph_index * 1000) + word_index
                if self.line_renderer.estimate_text_width(candidate, seed=candidate_seed) <= self.layout_config.max_line_width:
                    current_line = candidate
                else:
                    wrapped_lines.append(current_line)
                    current_line = word
            wrapped_lines.append(current_line)

        return wrapped_lines

    def render_paragraph(self, text: str, seed: int = 7) -> RenderedParagraph:
        """Render wrapped text as a simple page with line drift and spacing."""

        lines = self.wrap_text(text=text, seed=seed)
        if not lines:
            blank = Image.fromarray(np.full((1, 1), self.layout_config.background_value, dtype=np.uint8))
            return RenderedParagraph(blank, [], [], [], [])

        rng = random.Random(seed)
        line_results: list[RenderedLine] = []
        for index, line in enumerate(lines):
            if line:
                line_results.append(self.line_renderer.render_text(line, seed=seed + (index * 137)))
            else:
                blank_line = Image.fromarray(np.full((1, 1), self.layout_config.background_value, dtype=np.uint8))
                line_results.append(RenderedLine(blank_line, [], [], []))

        line_images = [result.image.convert("L") for result in line_results]
        page_width = max(
            self.layout_config.page_padding_x * 2 + self.layout_config.max_line_width,
            max(image.width for image in line_images) + (self.layout_config.page_padding_x * 2),
        )
        page_height = (
            self.layout_config.page_padding_y * 2
            + sum(image.height for image in line_images)
            + (self.layout_config.line_spacing * max(0, len(line_images) - 1))
        )
        canvas = np.full((page_height, page_width), self.layout_config.background_value, dtype=np.uint8)

        cursor_y = self.layout_config.page_padding_y
        unsupported_labels: list[str] = []
        used_word_samples: list[str] = []
        for line_image, line_result in zip(line_images, line_results):
            drift = rng.randint(-self.layout_config.line_margin_drift, self.layout_config.line_margin_drift)
            cursor_x = max(0, self.layout_config.page_padding_x + drift)
            right = min(page_width, cursor_x + line_image.width)
            bottom = min(page_height, cursor_y + line_image.height)
            writable = np.array(line_image, dtype=np.uint8)[: bottom - cursor_y, : right - cursor_x]
            canvas[cursor_y:bottom, cursor_x:right] = np.minimum(canvas[cursor_y:bottom, cursor_x:right], writable)
            cursor_y = bottom + self.layout_config.line_spacing
            unsupported_labels.extend(line_result.unsupported_labels)
            used_word_samples.extend(line_result.used_word_samples)

        return RenderedParagraph(
            image=Image.fromarray(canvas),
            lines=lines,
            line_images=line_images,
            unsupported_labels=unsupported_labels,
            used_word_samples=used_word_samples,
        )


def image_to_png_bytes(image: Image.Image) -> bytes:
    """Serialize a rendered page as PNG bytes for downloads or exports."""

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
