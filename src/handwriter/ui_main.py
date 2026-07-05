"""Primary user-facing Streamlit sections for the InkForge notebook app."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from .app_support import ParagraphRenderBundle
from .page_design import PageStyleConfig, image_to_png_bytes, images_to_pdf_bytes
from .paragraph import ParagraphLayoutConfig

DEFAULT_TEXT = (
    "## Physics Notes\n"
    "Newton's laws explain how motion changes.\n"
    "!! Force equals mass multiplied by acceleration.\n"
    "* Objects stay at rest unless acted on.\n"
    "[ ] Revise chapter 3\n"
    "> Every action has an equal and opposite reaction."
)

RENDER_MODES = ("Context-aware", "Flat word-bank", "Glyph only", "Compare all")
PAPER_STYLES = ("Blank", "Ruled", "Grid")

EDITOR_KEY = "inkforge_editor_text"


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

    _inject_modern_styles()
    st.title("InkForge")
    st.caption("Personalized handwritten notes with a cleaner, document-first workflow.")


def render_main_controls() -> MainViewState:
    """Collect notebook content and compact professional controls."""

    _ensure_editor_state()

    top_left, top_right = st.columns([1.55, 0.85], gap="large")
    with top_left:
        st.markdown("### Editor")
        _render_toolbar()
        st.text_area(
            "Compose your note",
            key=EDITOR_KEY,
            height=360,
            label_visibility="collapsed",
        )
        st.caption(
            "Formatting shortcuts: `##` heading, `!!` highlighted line, `*` bullet, `[ ]` task, `[x]` done, `>` quote, `()`, `[]`, `{}` brackets."
        )

    with top_right:
        st.markdown("### Quick Settings")
        quick_one, quick_two = st.columns(2)
        with quick_one:
            render_mode = st.radio("Preview", options=RENDER_MODES, index=0, horizontal=True)
            paper_style = st.radio("Paper", options=PAPER_STYLES, index=1, horizontal=True)
            font_size = st.number_input("Font size", min_value=4, max_value=16, value=8, step=1)
        with quick_two:
            ink_hex = st.color_picker("Ink", value="#161616")
            marker_hex = st.color_picker("Marker", value="#fff2a8")
            paper_hex = st.color_picker("Paper tone", value="#fffdf7")

        with st.expander("Advanced Settings", expanded=False):
            advanced_left, advanced_right = st.columns(2)
            with advanced_left:
                seed = st.number_input("Render seed", min_value=0, max_value=9999, value=7, step=1)
                page_width = st.number_input("Page width", min_value=900, max_value=1600, value=1240, step=20)
                page_height = st.number_input("Page height", min_value=1200, max_value=2400, value=1754, step=20)
                line_spacing = st.number_input("Line spacing", min_value=6, max_value=64, value=20, step=2)
                heading_scale = st.number_input("Heading scale", min_value=1.0, max_value=1.8, value=1.25, step=0.05)
            with advanced_right:
                boldness = st.number_input("Ink weight", min_value=1, max_value=4, value=1, step=1)
                marker_opacity = st.number_input("Marker strength", min_value=0.10, max_value=0.85, value=0.40, step=0.05)
                left_margin = st.number_input("Left margin", min_value=60, max_value=220, value=112, step=4)
                right_margin = st.number_input("Right margin", min_value=40, max_value=180, value=84, step=4)
                top_margin = st.number_input("Top margin", min_value=40, max_value=180, value=88, step=4)
                bottom_margin = st.number_input("Bottom margin", min_value=40, max_value=180, value=88, step=4)

            paper_left, paper_right = st.columns(2)
            with paper_left:
                show_margin_guide = st.toggle("Show margin guide", value=True)
                margin_guide_offset = st.number_input("Margin guide position", min_value=40, max_value=180, value=92, step=2)
                rule_spacing = st.number_input("Rule spacing", min_value=30, max_value=90, value=52, step=2)
            with paper_right:
                rule_hex = st.color_picker("Rule color", value="#c6d7f4")
                margin_hex = st.color_picker("Margin guide color", value="#e88888")
                line_margin_drift = st.number_input("Hand drift", min_value=0, max_value=24, value=8, step=1)

            behavior_left, behavior_right = st.columns(2)
            with behavior_left:
                treat_first_line_as_title = st.toggle("Use first line as heading", value=False)
                highlight_prefix = st.text_input("Highlight prefix", value="!! ")
            with behavior_right:
                heading_highlight = st.toggle("Auto-highlight headings", value=False)
                heading_spacing = st.number_input("Heading spacing", min_value=10, max_value=48, value=26, step=2)

    if "seed" not in locals():
        seed = 7
        page_width = 1240
        page_height = 1754
        line_spacing = 20
        heading_scale = 1.25
        boldness = 1
        marker_opacity = 0.40
        left_margin = 112
        right_margin = 84
        top_margin = 88
        bottom_margin = 88
        show_margin_guide = True
        margin_guide_offset = 92
        rule_spacing = 52
        rule_hex = "#c6d7f4"
        margin_hex = "#e88888"
        line_margin_drift = 8
        treat_first_line_as_title = False
        highlight_prefix = "!! "
        heading_highlight = False
        heading_spacing = 26

    usable_writing_width = max(
        420,
        int(page_width) - int(left_margin) - int(right_margin) - 28,
    )

    layout_config = ParagraphLayoutConfig(
        max_line_width=usable_writing_width,
        page_padding_top=int(top_margin),
        page_padding_right=int(right_margin),
        page_padding_bottom=int(bottom_margin),
        page_padding_left=int(left_margin),
        line_spacing=int(line_spacing),
        line_margin_drift=int(line_margin_drift),
        body_scale=max(0.24, float(font_size) / 22.0),
        title_scale=float(heading_scale),
        title_spacing=int(heading_spacing),
        treat_first_line_as_title=treat_first_line_as_title,
        highlight_prefix=highlight_prefix,
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
        title_highlight=heading_highlight,
        title_highlight_color=_hex_to_rgb(marker_hex),
        marker_color=_hex_to_rgb(marker_hex),
        marker_opacity=float(marker_opacity),
    )
    return MainViewState(
        input_text=st.session_state[EDITOR_KEY],
        seed=int(seed),
        layout_config=layout_config,
        page_style=page_style,
        render_mode=render_mode,
    )


def render_preview_section(rendered: ParagraphRenderBundle, render_mode: str) -> None:
    """Show the notebook preview in a cleaner browser layout."""

    st.markdown("### Preview")
    preview_pages = rendered.pages_for_mode(render_mode)
    page_tabs = st.tabs([f"Page {index}" for index in range(1, len(preview_pages) + 1)])
    for index, (tab, page) in enumerate(zip(page_tabs, preview_pages), start=1):
        with tab:
            image = page.image if hasattr(page, "image") else page
            st.image(image, caption=f"Page {index}", use_container_width=True)

    if rendered.unsupported_labels:
        st.warning(f"Unsupported labels skipped: {rendered.unsupported_labels}")


def render_download_section(rendered: ParagraphRenderBundle) -> None:
    """Expose notebook-friendly PNG and PDF download options."""

    st.markdown("### Export")
    export_left, export_right = st.columns([0.7, 1.3], gap="large")
    with export_left:
        st.download_button(
            "Download PDF",
            data=images_to_pdf_bytes([page.image for page in rendered.context.pages]),
            file_name="inkforge_notes.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    with export_right:
        page_cols = st.columns(min(3, max(1, len(rendered.context.pages))))
        for index, page in enumerate(rendered.context.pages):
            with page_cols[index % len(page_cols)]:
                st.download_button(
                    f"Page {page.page_number} PNG",
                    data=image_to_png_bytes(page.image),
                    file_name=f"inkforge_page_{page.page_number}.png",
                    mime="image/png",
                    key=f"download-page-{page.page_number}",
                    use_container_width=True,
                )


def _ensure_editor_state() -> None:
    if EDITOR_KEY not in st.session_state:
        st.session_state[EDITOR_KEY] = DEFAULT_TEXT


def _render_toolbar() -> None:
    rows = [
        [("Heading", "## "), ("Marker Line", "\n!! "), ("Bullet", "\n* "), ("Quote", "\n> ")],
        [("Checkbox", "\n[ ] "), ("Done", "\n[x] "), ("Equation", " = "), ("Round ()", "()")],
        [("Square []", "[]"), ("Curly {}", "{}"), ("Slash /", "/"), ("Colon :", ":")],
    ]
    for row_index, row in enumerate(rows):
        columns = st.columns(len(row))
        for column, (label, snippet) in zip(columns, row):
            with column:
                st.button(
                    label,
                    key=f"toolbar-{row_index}-{label}",
                    on_click=_append_snippet,
                    args=(snippet,),
                    use_container_width=True,
                )


def _append_snippet(snippet: str) -> None:
    current = st.session_state.get(EDITOR_KEY, "")
    if snippet.startswith("\n"):
        st.session_state[EDITOR_KEY] = f"{current.rstrip()}{snippet}"
    else:
        st.session_state[EDITOR_KEY] = f"{current}{snippet}"


def _inject_modern_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
          background: linear-gradient(180deg, #f4efe5 0%, #efe8db 100%);
          color: #1d1d1f;
        }
        .block-container {
          max-width: 1240px;
          padding-top: 1.8rem;
          padding-bottom: 2rem;
        }
        h1, h2, h3, p, label, div, span {
          color: #1d1d1f;
        }
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] span,
        [data-testid="stWidgetLabel"] *,
        label,
        .stSelectbox label,
        .stColorPicker label,
        .stNumberInput label,
        .stTextArea label,
        .stCaption,
        small {
          color: #2a241c !important;
        }
        div[data-testid="stSidebar"] {
          background: #242733;
        }
        div[data-testid="stSidebar"] * {
          color: #f2f4f8;
        }
        div[data-testid="stTextArea"] textarea {
          border-radius: 18px;
          border: 1px solid rgba(77, 69, 53, 0.18);
          background: rgba(255, 252, 246, 0.96);
          box-shadow: 0 18px 40px rgba(63, 46, 18, 0.08);
          color: #1d1d1f !important;
          font-size: 1rem;
          line-height: 1.55;
        }
        div[data-testid="stRadio"] label {
          color: #1d1d1f !important;
          font-weight: 600;
        }
        div[data-testid="stNumberInput"] input {
          background: #fffaf2 !important;
          color: #1d1d1f !important;
          border: 1px solid rgba(77, 69, 53, 0.18) !important;
        }
        div[data-testid="stButton"] > button,
        div[data-testid="stDownloadButton"] > button {
          border-radius: 14px;
          border: 1px solid rgba(77, 69, 53, 0.12);
          background: #fffaf2;
          color: #1d1d1f;
          box-shadow: 0 10px 24px rgba(63, 46, 18, 0.08);
        }
        div[data-testid="stColorPicker"] {
          color: #1d1d1f;
        }
        .inkforge-stat {
          border-radius: 16px;
          padding: 1rem 1rem;
          background: #fffaf2;
          border: 1px solid rgba(77, 69, 53, 0.14);
          box-shadow: 0 10px 24px rgba(63, 46, 18, 0.05);
        }
        .inkforge-stat span {
          display: block;
          font-size: 0.74rem;
          color: #6b5f4c;
          margin-bottom: 0.3rem;
          text-transform: uppercase;
          letter-spacing: 0.04em;
          font-weight: 700;
        }
        .inkforge-stat strong {
          font-size: 1.05rem;
          color: #171513;
          line-height: 1.25;
        }
        .inkforge-help {
          margin-top: 0.65rem;
          border-radius: 14px;
          padding: 0.85rem 1rem;
          background: rgba(255, 250, 242, 0.96);
          border: 1px solid rgba(77, 69, 53, 0.10);
          color: #2a241c;
          line-height: 1.45;
        }
        .inkforge-help code {
          background: rgba(29, 29, 31, 0.08);
          color: #1d1d1f;
          padding: 0.1rem 0.3rem;
          border-radius: 6px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))
