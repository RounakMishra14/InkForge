"""Synthetic handwritten-like fallbacks for weak or missing symbols."""

from __future__ import annotations

import numpy as np


PREFERRED_SYNTHETIC_SYMBOLS = {
    "-",
    "*",
    "+",
    "=",
    ":",
    ";",
    "_",
    "/",
    "\\",
    "|",
    "<",
    ">",
}
SUPPORTED_SYNTHETIC_SYMBOLS = PREFERRED_SYNTHETIC_SYMBOLS | {
    "(",
    ")",
    "[",
    "]",
    "{",
    "}",
}


def should_prefer_synthetic_symbol(
    char: str,
    previous_char: str | None = None,
    next_char: str | None = None,
) -> bool:
    """Return whether a symbol should bypass the dataset glyph path."""

    if char in {"=", ":", ";", "_", "/", "\\", "|", "<", ">"}:
        return True
    if char == "*" and previous_char is None:
        return True
    if char == "+" and _is_standalone_operator(previous_char, next_char):
        return True
    if char == "-" and (previous_char is None or _is_standalone_operator(previous_char, next_char)):
        return True
    return False


def synthetic_symbol_image(char: str, target_height: int) -> np.ndarray | None:
    """Build a grayscale glyph-like image for common note-taking symbols."""

    height = max(16, target_height)
    width = _symbol_width(char, height)
    canvas = np.full((height, width), 255, dtype=np.uint8)
    stroke = max(2, round(height * 0.08))
    mid_y = height // 2
    mid_x = width // 2
    ink = 28

    if char in {"-", "_"}:
        line_y = mid_y if char == "-" else max(mid_y, round(height * 0.78))
        x_margin = max(2, round(width * 0.12))
        canvas[max(0, line_y - stroke) : min(height, line_y + stroke), x_margin : width - x_margin] = ink
        return canvas

    if char == "=":
        x_margin = max(2, round(width * 0.14))
        offset = max(3, round(height * 0.12))
        for y in (mid_y - offset, mid_y + offset):
            canvas[max(0, y - stroke) : min(height, y + stroke), x_margin : width - x_margin] = ink
        return canvas

    if char == "*":
        radius_y = max(3, round(height * 0.12))
        radius_x = max(3, round(width * 0.18))
        yy, xx = np.ogrid[:height, :width]
        mask = (((yy - mid_y) / max(1, radius_y)) ** 2) + (((xx - mid_x) / max(1, radius_x)) ** 2) <= 1.0
        canvas[mask] = ink
        return canvas

    if char == "+":
        x_margin = max(2, round(width * 0.18))
        y_margin = max(2, round(height * 0.18))
        canvas[max(0, mid_y - stroke) : min(height, mid_y + stroke), x_margin : width - x_margin] = ink
        canvas[y_margin : height - y_margin, max(0, mid_x - stroke) : min(width, mid_x + stroke)] = ink
        return canvas

    if char == ":":
        return _double_dot(canvas, center_offset=max(4, round(height * 0.16)))

    if char == ";":
        canvas = _double_dot(canvas, center_offset=max(4, round(height * 0.16)))
        tail_top = min(height - 2, round(height * 0.62))
        tail_left = mid_x
        for step in range(max(4, round(height * 0.14))):
            y = min(height - 1, tail_top + step)
            x = max(0, tail_left - step // 2)
            canvas[max(0, y - 1) : min(height, y + 1), x : min(width, x + stroke)] = ink
        return canvas

    if char == "/":
        return _diagonal(canvas, rising=True, stroke=stroke, ink=ink)

    if char == "\\":
        return _diagonal(canvas, rising=False, stroke=stroke, ink=ink)

    if char == "|":
        canvas[:, max(0, mid_x - stroke) : min(width, mid_x + stroke)] = ink
        return canvas

    if char == "<":
        return _angle(canvas, left_facing=True, stroke=stroke, ink=ink)

    if char == ">":
        return _angle(canvas, left_facing=False, stroke=stroke, ink=ink)

    if char in {"(", ")", "[", "]", "{", "}"}:
        return _bracket_symbol(char, canvas, stroke=stroke, ink=ink)

    return None


def _symbol_width(char: str, height: int) -> int:
    if char in {"-", "_", "/", "\\", "<", ">"}:
        return max(16, round(height * 0.55))
    if char in {"=", "+"}:
        return max(20, round(height * 0.72))
    if char in {"*", ":", ";", "|"}:
        return max(12, round(height * 0.28))
    if char in {"(", ")", "[", "]"}:
        return max(14, round(height * 0.28))
    if char in {"{", "}"}:
        return max(16, round(height * 0.34))
    return max(14, round(height * 0.4))


def _is_standalone_operator(previous_char: str | None, next_char: str | None) -> bool:
    left_space = previous_char is None or previous_char.isspace()
    right_space = next_char is None or next_char.isspace()
    return left_space or right_space


def _double_dot(canvas: np.ndarray, center_offset: int) -> np.ndarray:
    height, width = canvas.shape
    radius = max(2, round(height * 0.07))
    center_x = width // 2
    for center_y in (height // 2 - center_offset, height // 2 + center_offset):
        yy, xx = np.ogrid[:height, :width]
        mask = (((yy - center_y) / max(1, radius)) ** 2) + (((xx - center_x) / max(1, radius)) ** 2) <= 1.0
        canvas[mask] = 26
    return canvas


def _diagonal(canvas: np.ndarray, rising: bool, stroke: int, ink: int) -> np.ndarray:
    height, width = canvas.shape
    for x in range(width):
        ratio = x / max(1, width - 1)
        y = round((height - 1) * (1.0 - ratio if rising else ratio))
        canvas[max(0, y - stroke) : min(height, y + stroke), max(0, x - stroke) : min(width, x + stroke)] = ink
    return canvas


def _angle(canvas: np.ndarray, left_facing: bool, stroke: int, ink: int) -> np.ndarray:
    height, width = canvas.shape
    pivot_x = round(width * (0.28 if left_facing else 0.72))
    pivot_y = height // 2
    for x in range(width):
        ratio = x / max(1, width - 1)
        if left_facing:
            upper_y = round(pivot_y - (pivot_y - 2) * (ratio if x <= pivot_x else 1.0 - ratio))
            lower_y = round(pivot_y + (height - pivot_y - 2) * (ratio if x <= pivot_x else 1.0 - ratio))
        else:
            upper_y = round(2 + (pivot_y - 2) * ratio)
            lower_y = round((height - 2) - (height - pivot_y - 2) * ratio)
        canvas[max(0, upper_y - stroke) : min(height, upper_y + stroke), max(0, x - stroke) : min(width, x + stroke)] = ink
        canvas[max(0, lower_y - stroke) : min(height, lower_y + stroke), max(0, x - stroke) : min(width, x + stroke)] = ink
    return canvas


def _bracket_symbol(char: str, canvas: np.ndarray, stroke: int, ink: int) -> np.ndarray:
    height, width = canvas.shape
    top = max(1, round(height * 0.12))
    bottom = min(height - 1, round(height * 0.88))
    left = max(1, round(width * 0.25))
    right = min(width - 1, round(width * 0.75))

    if char == "(":
        for y in range(top, bottom):
            x = left + round((y - top) * 0.12) if y <= height // 2 else left + round((bottom - y) * 0.12)
            canvas[y, max(0, x - stroke) : min(width, x + stroke)] = ink
    elif char == ")":
        for y in range(top, bottom):
            x = right - round((y - top) * 0.12) if y <= height // 2 else right - round((bottom - y) * 0.12)
            canvas[y, max(0, x - stroke) : min(width, x + stroke)] = ink
    elif char == "[":
        canvas[top:bottom, max(0, left - stroke) : min(width, left + stroke)] = ink
        canvas[max(0, top - stroke) : min(height, top + stroke), left:right] = ink
        canvas[max(0, bottom - stroke) : min(height, bottom + stroke), left:right] = ink
    elif char == "]":
        canvas[top:bottom, max(0, right - stroke) : min(width, right + stroke)] = ink
        canvas[max(0, top - stroke) : min(height, top + stroke), left:right] = ink
        canvas[max(0, bottom - stroke) : min(height, bottom + stroke), left:right] = ink
    elif char == "{":
        canvas = _bracket_symbol("(", canvas, stroke, ink)
        center = height // 2
        canvas[max(0, center - stroke) : min(height, center + stroke), : width // 2] = ink
    elif char == "}":
        canvas = _bracket_symbol(")", canvas, stroke, ink)
        center = height // 2
        canvas[max(0, center - stroke) : min(height, center + stroke), width // 2 :] = ink
    return canvas
