"""Page backgrounds, ink compositing, and export helpers for note-style output."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image, ImageFilter


@dataclass(frozen=True)
class PageStyleConfig:
    """Visual options for note pages and handwriting ink."""

    page_width: int = 1240
    page_height: int = 1754
    paper_style: str = "Blank"
    paper_color: tuple[int, int, int] = (255, 255, 255)
    ink_color: tuple[int, int, int] = (30, 30, 30)
    rule_color: tuple[int, int, int] = (198, 215, 244)
    margin_color: tuple[int, int, int] = (232, 136, 136)
    show_margin_guide: bool = True
    margin_guide_offset: int = 92
    rule_spacing: int = 52
    boldness: int = 1
    title_highlight: bool = False
    title_highlight_color: tuple[int, int, int] = (255, 246, 153)
    marker_color: tuple[int, int, int] = (255, 246, 153)
    marker_opacity: float = 0.42


def build_page_background(style: PageStyleConfig) -> np.ndarray:
    """Create an RGB page background with optional ruling and margin guides."""

    canvas = np.full(
        (style.page_height, style.page_width, 3),
        style.paper_color,
        dtype=np.uint8,
    )

    if style.paper_style == "Ruled":
        for y in range(style.rule_spacing, style.page_height, style.rule_spacing):
            canvas[max(0, y - 1) : min(style.page_height, y + 1), :, :] = style.rule_color
    elif style.paper_style == "Grid":
        for y in range(style.rule_spacing, style.page_height, style.rule_spacing):
            canvas[max(0, y - 1) : min(style.page_height, y + 1), :, :] = style.rule_color
        for x in range(style.rule_spacing, style.page_width, style.rule_spacing):
            canvas[:, max(0, x - 1) : min(style.page_width, x + 1), :] = style.rule_color

    if style.show_margin_guide:
        margin_x = min(style.page_width - 1, max(0, style.margin_guide_offset))
        canvas[:, max(0, margin_x - 1) : min(style.page_width, margin_x + 1), :] = style.margin_color

    return canvas


def compose_handwriting_layer(
    background: np.ndarray,
    handwriting: Image.Image,
    top: int,
    left: int,
    ink_color: tuple[int, int, int],
    boldness: int = 1,
) -> None:
    """Place grayscale handwriting onto a page background using the chosen ink color."""

    line_image = handwriting.convert("L")
    if boldness > 1:
        line_image = _embolden(line_image, passes=boldness - 1)

    grayscale = np.array(line_image, dtype=np.uint8)
    height = min(grayscale.shape[0], background.shape[0] - top)
    width = min(grayscale.shape[1], background.shape[1] - left)
    if height <= 0 or width <= 0:
        return

    grayscale = grayscale[:height, :width]
    ink_mask = 255 - grayscale
    ink_strength = ink_mask.astype(np.float32) / 255.0
    if not ink_strength.any():
        return

    target = background[top : top + height, left : left + width, :].astype(np.float32)
    ink = np.array(ink_color, dtype=np.float32).reshape(1, 1, 3)
    blended = (target * (1.0 - ink_strength[..., None])) + (ink * ink_strength[..., None])
    background[top : top + height, left : left + width, :] = np.clip(blended, 0, 255).astype(np.uint8)


def draw_highlight_band(
    background: np.ndarray,
    top: int,
    left: int,
    width: int,
    height: int,
    color: tuple[int, int, int],
    opacity: float = 0.4,
) -> None:
    """Apply a marker-style highlight band behind a handwritten line."""

    right = min(background.shape[1], left + width)
    bottom = min(background.shape[0], top + height)
    if right <= left or bottom <= top:
        return

    pad_y = max(2, round(height * 0.08))
    pad_x = max(4, round(width * 0.015))
    top = max(0, top - pad_y)
    bottom = min(background.shape[0], bottom + pad_y)
    left = max(0, left - pad_x)
    right = min(background.shape[1], right + pad_x)

    area = background[top:bottom, left:right, :].astype(np.float32)
    highlight = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    blended = (area * (1.0 - opacity)) + (highlight * opacity)
    background[top:bottom, left:right, :] = np.clip(blended, 0, 255).astype(np.uint8)


def image_to_png_bytes(image: Image.Image) -> bytes:
    """Serialize a rendered page as PNG bytes."""

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def images_to_pdf_bytes(images: list[Image.Image]) -> bytes:
    """Serialize multiple RGB pages into a single PDF."""

    if not images:
        blank = Image.new("RGB", (1, 1), "white")
        images = [blank]

    rgb_pages = [image.convert("RGB") for image in images]
    buffer = BytesIO()
    rgb_pages[0].save(
        buffer,
        format="PDF",
        save_all=True,
        append_images=rgb_pages[1:],
        resolution=150.0,
    )
    return buffer.getvalue()


def _embolden(image: Image.Image, passes: int) -> Image.Image:
    """Thicken ink slightly by repeatedly applying a small minimum filter."""

    result = image
    for _ in range(max(0, passes)):
        result = result.filter(ImageFilter.MinFilter(size=3))
    return result
