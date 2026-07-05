"""Primary user-facing Streamlit sections for the handwriting notebook app."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from .app_support import ParagraphRenderBundle
from .page_design import PageStyleConfig, image_to_png_bytes, images_to_pdf_bytes
from .paragraph import ParagraphLayoutConfig

DEFAULT_TEXT = (
    "Physics Notes\n"
    "Newton's laws explain how motion changes.\n"
    "Force equals mass multiplied by acceleration.\n"
    "Revision points:\n"
    "- Objects stay at rest unless acted on.\n"
    "- Every action has an equal and opposite reaction."
)

RENDER_MODES = ("Context-aware", "Flat word-bank", "Glyph only", "Compare all")
PAPER_STYLES = ("Blank", "Ruled", "Grid")


@dataclass(frozen=True)
class MainViewState:
    """User controls for notebook rendering."""

    input_text: str
    seed: int
    layout_config: ParagraphLayoutConfig
    page_style: PageStyleConfig
    render_mode: str


def render_header() -> None:
    """Display the public-facing app title and summary."""

    st.title("InkForge")
    st.caption("Turn typed notes into organized notebook pages using your personalized handwriting style.")


def render_main_controls() -> MainViewState:
    """Collect notebook content and sidebar page settings."""

    input_text = st.text_area("Write your notes", value=DEFAULT_TEXT, height=320)
    with st.sidebar:
        st.subheader("Page Setup")
        seed = st.number_input("Render seed", min_value=0, max_value=9999, value=7, step=1)
        render_mode = st.radio("Preview mode", options=RENDER_MODES, horizontal=False)
        paper_style = st.selectbox("Paper style", options=PAPER_STYLES, index=1)
        page_width = st.slider("Page width", min_value=900, max_value=1600, value=1240, step=20)
        page_height = st.slider("Page height", min_value=1200, max_value=2200, value=1754, step=20)
        max_line_width = st.slider("Writing width", min_value=520, max_value=1300, value=980, step=20)
        top_padding = st.slider("Top margin", min_value=40, max_value=180, value=88, step=4)
        right_padding = st.slider("Right margin", min_value=40, max_value=180, value=84, step=4)
        bottom_padding = st.slider("Bottom margin", min_value=40, max_value=180, value=88, step=4)
        left_padding = st.slider("Left margin", min_value=60, max_value=220, value=112, step=4)
        line_spacing = st.slider("Line spacing", min_value=8, max_value=48, value=20, step=2)
        line_margin_drift = st.slider("Hand drift", min_value=0, max_value=24, value=8, step=1)

        st.subheader("Ink and Styling")
        ink_hex = st.color_picker("Text color", value="#1e1e1e")
        paper_hex = st.color_picker("Paper color", value="#fffdf7")
        rule_hex = st.color_picker("Rule color", value="#c6d7f4")
        margin_hex = st.color_picker("Margin guide color", value="#e88888")
        show_margin_guide = st.toggle("Show margin guide", value=True)
        margin_guide_offset = st.slider("Margin guide position", min_value=40, max_value=180, value=92, step=2)
        boldness = st.slider("Boldness", min_value=1, max_value=4, value=1, step=1)
        title_scale = st.slider("Title size", min_value=1.0, max_value=1.5, value=1.2, step=0.05)
        title_highlight = st.toggle("Highlight first line as title", value=True)
        title_highlight_hex = st.color_picker("Highlight color", value="#fff699")
        rule_spacing = st.slider("Rule spacing", min_value=36, max_value=80, value=52, step=2)

    layout_config = ParagraphLayoutConfig(
        max_line_width=int(max_line_width),
        page_padding_top=int(top_padding),
        page_padding_right=int(right_padding),
        page_padding_bottom=int(bottom_padding),
        page_padding_left=int(left_padding),
        line_spacing=int(line_spacing),
        line_margin_drift=int(line_margin_drift),
        title_scale=float(title_scale),
    )
    page_style = PageStyleConfig(
        page_width=int(page_width),
        page_height=int(page_height),
        paper_style=paper_style,
        paper_color=_hex_to_rgb(paper_hex),
        ink_color=_hex_to_rgb(ink_hex),
        rule_color=_hex_to_rgb(rule_hex),
        margin_color=_hex_to_rgb(margin_hex),
        show_margin_guide=show_margin_guide,
        margin_guide_offset=int(margin_guide_offset),
        rule_spacing=int(rule_spacing),
        boldness=int(boldness),
        title_highlight=title_highlight,
        title_highlight_color=_hex_to_rgb(title_highlight_hex),
    )
    return MainViewState(
        input_text=input_text,
        seed=int(seed),
        layout_config=layout_config,
        page_style=page_style,
        render_mode=render_mode,
    )


def render_preview_section(rendered: ParagraphRenderBundle, render_mode: str) -> None:
    """Show the notebook preview with a simple creator-facing layout."""

    st.subheader("Preview")
    preview_pages = rendered.pages_for_mode(render_mode)
    for index, page in enumerate(preview_pages, start=1):
        image = page.image if hasattr(page, "image") else page
        st.image(image, caption=f"Page {index}", use_container_width=True)

    if rendered.unsupported_labels:
        st.warning(f"Unsupported labels skipped: {rendered.unsupported_labels}")


def render_download_section(rendered: ParagraphRenderBundle) -> None:
    """Expose notebook-friendly PNG and PDF download options."""

    st.subheader("Download")
    st.download_button(
        "Download PDF",
        data=images_to_pdf_bytes([page.image for page in rendered.context.pages]),
        file_name="handwritten_notes.pdf",
        mime="application/pdf",
    )

    for page in rendered.context.pages:
        st.download_button(
            f"Download Page {page.page_number} PNG",
            data=image_to_png_bytes(page.image),
            file_name=f"handwritten_page_{page.page_number}.png",
            mime="image/png",
            key=f"download-page-{page.page_number}",
        )


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))
