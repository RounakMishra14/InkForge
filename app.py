"""Streamlit entrypoint for the handwriting synthesis prototype."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from handwriter.app_support import build_pipeline, render_paragraph_bundle  # noqa: E402
from handwriter.ui_debug import render_debug_panels, render_debug_toggle  # noqa: E402
from handwriter.ui_main import (  # noqa: E402
    render_download_section,
    render_header,
    render_main_controls,
    render_preview_section,
)


@st.cache_resource
def bootstrap_pipeline():
    """Create reusable pipeline state once per Streamlit session."""

    return build_pipeline(project_root=PROJECT_ROOT)


def main() -> None:
    st.set_page_config(page_title="InkForge", layout="wide")

    pipeline = bootstrap_pipeline()
    show_debug_tools = render_debug_toggle()

    render_header()
    view_state = render_main_controls()
    rendered = render_paragraph_bundle(
        pipeline=pipeline,
        input_text=view_state.input_text,
        seed=view_state.seed,
        layout_config=view_state.layout_config,
        page_style=view_state.page_style,
    )
    render_preview_section(rendered=rendered, render_mode=view_state.render_mode)
    render_download_section(rendered)

    if show_debug_tools:
        render_debug_panels(pipeline=pipeline, project_root=PROJECT_ROOT)


if __name__ == "__main__":
    main()
