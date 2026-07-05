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
    build_word_bank,
    evaluate_dataset_reconstruction,
)
from handwriter.preview import side_by_side  # noqa: E402
from handwriter.quality import build_quality_report  # noqa: E402


@st.cache_resource
def bootstrap_pipeline():
    """Create the reusable dataset, style profile, and renderer once per session."""

    paths = DatasetPaths(root=PROJECT_ROOT)
    dataset = HandwritingDataset(paths=paths)
    style_profile = build_style_profile(dataset=dataset, include_copies=False)
    word_bank = build_word_bank(dataset=dataset, include_copies=False)
    renderer = HandwritingRenderer(
        dataset=dataset,
        style_profile=style_profile,
        word_bank=word_bank,
        config=RenderConfig(),
    )
    quality_report = build_quality_report(dataset)
    return dataset, style_profile, word_bank, renderer, quality_report


def main() -> None:
    st.set_page_config(page_title="Handwriter Prototype", layout="wide")

    dataset, style_profile, word_bank, renderer, quality_report = bootstrap_pipeline()

    st.title("Handwriter Prototype")
    st.caption(
        "Current batch: glyph composition with exact-word retrieval from segmented sentence crops, plus richer dataset and evaluation reporting."
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
        available_words, missing_words = word_bank.coverage(input_text)

        if rendered.unsupported_labels:
            st.warning(f"Unsupported labels skipped: {sorted(set(rendered.unsupported_labels))}")
        else:
            st.success("All characters in the input are currently supported by the dataset.")

        st.caption(
            f"Word-bank coverage: {len(available_words)} matched words, {len(missing_words)} fallback words."
        )
        if rendered.used_word_samples:
            st.info(f"Used exact handwritten words: {rendered.used_word_samples}")

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

        st.subheader("Dataset Quality")
        st.json(
            {
                "total_sentence_samples": quality_report.total_sentence_samples,
                "clean_sentence_samples": quality_report.clean_sentence_samples,
                "unique_sentence_prompts": quality_report.unique_sentence_prompts,
                "explicit_copy_files": quality_report.explicit_copy_files,
                "duplicate_glyph_files": quality_report.duplicate_glyph_files,
                "duplicate_sentence_files": quality_report.duplicate_sentence_files,
                "word_bank_vocabulary": len(word_bank.available_words()),
            }
        )

    st.divider()
    st.subheader("Baseline Evaluation")
    st.caption(
        "This is still an accuracy proxy, not OCR accuracy. The renderer redraws known transcripts and compares them against held-out dataset crops while tracking exact-word reuse."
    )

    if st.button("Run Reconstruction Evaluation"):
        result = evaluate_dataset_reconstruction(
            dataset=dataset,
            renderer=renderer,
            max_samples_per_sentence=2,
            include_copies=False,
        )
        metric_one, metric_two, metric_three, metric_four = st.columns(4)
        metric_one.metric("Samples", result.sample_count)
        metric_two.metric("Mean IoU", f"{result.mean_iou:.4f}")
        metric_three.metric("Mean MAE", f"{result.mean_mae:.4f}")
        metric_four.metric(
            "Mean Word Reuse",
            f"{(sum(detail.used_word_samples for detail in result.details) / max(1, result.sample_count)):.2f}",
        )

        details_df = pd.DataFrame(
            [
                {
                    "text": detail.text,
                    "file": detail.path_name,
                    "iou": round(detail.iou, 4),
                    "mae": round(detail.mae, 4),
                    "unsupported_count": detail.unsupported_count,
                    "used_word_samples": detail.used_word_samples,
                }
                for detail in result.details
            ]
        )
        st.dataframe(
            details_df,
            use_container_width=True,
            hide_index=True,
        )

        preview_options = [f"{index}: {detail.text} [{detail.path_name}]" for index, detail in enumerate(result.details)]
        selected = st.selectbox("Preview a reconstruction pair", options=preview_options)
        selected_index = int(selected.split(":", maxsplit=1)[0])
        chosen_detail = result.details[selected_index]
        preview_image = side_by_side(chosen_detail.reference_image, chosen_detail.rendered_image)
        st.image(
            preview_image,
            caption="Left: reference crop | Right: synthesized reconstruction",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
