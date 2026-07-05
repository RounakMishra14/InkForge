"""Streamlit entrypoint for the handwriting synthesis prototype."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from handwriter import (  # noqa: E402
    DatasetPaths,
    HandwritingDataset,
    HandwritingRenderer,
    RenderConfig,
    build_style_profile,
    evaluate_dataset_reconstruction,
)


@st.cache_resource
def bootstrap_pipeline():
    """Create the reusable dataset, style profile, and renderer once per session."""

    paths = DatasetPaths(root=PROJECT_ROOT)
    dataset = HandwritingDataset(paths=paths)
    style_profile = build_style_profile(dataset=dataset, include_copies=False)
    renderer = HandwritingRenderer(
        dataset=dataset,
        style_profile=style_profile,
        config=RenderConfig(),
    )
    return dataset, style_profile, renderer


def main() -> None:
    st.set_page_config(page_title="Handwriter Prototype", layout="wide")

    dataset, style_profile, renderer = bootstrap_pipeline()

    st.title("Handwriter Prototype")
    st.caption(
        "Current baseline: isolated glyph composition guided by sentence-level spacing and baseline statistics."
    )

    left, right = st.columns([1.1, 0.9])

    with left:
        input_text = st.text_area(
            "Typed input",
            value="The quick brown fox jumps over the lazy dog",
            height=120,
        )
        seed = st.number_input("Render seed", min_value=0, max_value=9999, value=7, step=1)

        rendered = renderer.render_text(input_text, seed=int(seed))
        st.image(rendered.image, caption="Synthesized handwriting", use_container_width=True)

        if rendered.unsupported_labels:
            st.warning(f"Unsupported labels skipped: {sorted(set(rendered.unsupported_labels))}")
        else:
            st.success("All characters in the input are currently supported by the dataset.")

    with right:
        st.subheader("Style Profile")
        st.json(
            {
                "average_sentence_height": style_profile.average_sentence_height,
                "median_baseline_ratio": round(style_profile.median_baseline_ratio, 4),
                "average_char_gap": round(style_profile.average_char_gap, 2),
                "average_word_gap": round(style_profile.average_word_gap, 2),
                "average_ink_density": round(style_profile.average_ink_density, 4),
                "supported_sentence_count": style_profile.supported_sentence_count,
            }
        )

        st.subheader("Dataset Coverage")
        coverage = dataset.coverage_report()
        st.dataframe(
            pd.DataFrame(
                [{"label": label, "samples": count} for label, count in coverage.items()]
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    st.subheader("Baseline Evaluation")
    st.caption(
        "This is an accuracy proxy, not OCR accuracy. The renderer redraws known sentence transcripts and compares them against held-out dataset crops."
    )

    if st.button("Run Reconstruction Evaluation"):
        result = evaluate_dataset_reconstruction(
            dataset=dataset,
            renderer=renderer,
            max_samples_per_sentence=2,
            include_copies=False,
        )
        metric_one, metric_two, metric_three = st.columns(3)
        metric_one.metric("Samples", result.sample_count)
        metric_two.metric("Mean IoU", f"{result.mean_iou:.4f}")
        metric_three.metric("Mean MAE", f"{result.mean_mae:.4f}")

        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "text": detail.text,
                        "file": detail.path_name,
                        "iou": round(detail.iou, 4),
                        "mae": round(detail.mae, 4),
                        "unsupported_count": detail.unsupported_count,
                    }
                    for detail in result.details
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
