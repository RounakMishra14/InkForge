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
    build_spacing_profile,
    build_style_profile,
    build_word_bank,
    evaluate_dataset_reconstruction,
)
from handwriter.preview import stack_horizontal  # noqa: E402
from handwriter.quality import build_quality_report  # noqa: E402
from handwriter.word_inspector import build_word_contact_sheet, export_word_samples  # noqa: E402


@st.cache_resource
def bootstrap_pipeline():
    """Create the reusable dataset, style profile, and renderer once per session."""

    paths = DatasetPaths(root=PROJECT_ROOT)
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
    return dataset, style_profile, spacing_profile, word_bank, context_renderer, flat_word_bank_renderer, glyph_renderer, quality_report


def main() -> None:
    st.set_page_config(page_title="Handwriter Prototype", layout="wide")

    dataset, style_profile, spacing_profile, word_bank, context_renderer, flat_word_bank_renderer, glyph_renderer, quality_report = bootstrap_pipeline()

    st.title("Handwriter Prototype")
    st.caption(
        "Current batch: context-aware spacing built from segmented words, compared against flat-spacing and glyph-only renderers."
    )

    left, right = st.columns([1.1, 0.9])

    with left:
        input_text = st.text_area(
            "Typed input",
            value="The quick brown fox jumps over the lazy dog",
            height=120,
        )
        seed = st.number_input("Render seed", min_value=0, max_value=9999, value=7, step=1)
        render_mode = st.radio(
            "Render mode",
            options=("Context-aware", "Flat word-bank", "Glyph only", "Compare all"),
            horizontal=True,
        )

        context_rendered = context_renderer.render_text(input_text, seed=int(seed))
        flat_word_bank_rendered = flat_word_bank_renderer.render_text(input_text, seed=int(seed))
        glyph_rendered = glyph_renderer.render_text(input_text, seed=int(seed))
        available_words, missing_words = word_bank.coverage(input_text)

        if render_mode == "Context-aware":
            st.image(context_rendered.image, caption="Context-aware handwriting", use_container_width=True)
        elif render_mode == "Flat word-bank":
            st.image(flat_word_bank_rendered.image, caption="Flat-spacing word-bank handwriting", use_container_width=True)
        elif render_mode == "Glyph only":
            st.image(glyph_rendered.image, caption="Glyph-only handwriting", use_container_width=True)
        else:
            comparison = stack_horizontal(
                [context_rendered.image, flat_word_bank_rendered.image, glyph_rendered.image],
                gap=28,
            )
            st.image(
                comparison,
                caption="Left: context-aware | Middle: flat word-bank | Right: glyph only",
                use_container_width=True,
            )

        combined_unsupported = sorted(
            set(
                context_rendered.unsupported_labels
                + flat_word_bank_rendered.unsupported_labels
                + glyph_rendered.unsupported_labels
            )
        )
        if combined_unsupported:
            st.warning(f"Unsupported labels skipped: {combined_unsupported}")
        else:
            st.success("All characters in the input are currently supported by the dataset.")

        st.caption(
            f"Word-bank coverage: {len(available_words)} matched words, {len(missing_words)} fallback words."
        )
        if context_rendered.used_word_samples:
            st.info(f"Used exact handwritten words: {context_rendered.used_word_samples}")

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

        st.subheader("Spacing Profile")
        st.json(
            {
                "fallback_gap": round(spacing_profile.fallback_gap, 2),
                "pair_override_count": len(spacing_profile.pair_gap_overrides),
                "class_override_count": len(spacing_profile.class_gap_overrides),
                "token_baseline_jitter": spacing_profile.token_baseline_jitter,
                "token_height_jitter": spacing_profile.token_height_jitter,
            }
        )
        st.dataframe(pd.DataFrame(spacing_profile.top_pairs()), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Word Bank Inspector")
    st.caption(
        "Inspect the extracted handwritten words directly so segmentation issues are visible before they affect synthesis quality."
    )
    inventory_df = pd.DataFrame(word_bank.inventory_rows())
    st.dataframe(inventory_df, use_container_width=True, hide_index=True)

    selected_word = st.selectbox("Inspect extracted word", options=word_bank.available_words())
    selected_samples = word_bank.samples_for(selected_word)
    sheet = build_word_contact_sheet(selected_samples)
    st.image(sheet, caption=f"Extracted samples for '{selected_word}'", use_container_width=True)
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "source_sentence": sample.source_sentence,
                    "source_file": sample.source_file,
                    "segmentation_mode": sample.segmentation_mode,
                    "height": sample.image.shape[0],
                    "width": sample.image.shape[1],
                }
                for sample in selected_samples
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    if st.button("Export Selected Word Sheet"):
        output_path = export_word_samples(
            word=selected_word,
            samples=selected_samples,
            output_dir=PROJECT_ROOT / "artifacts" / "word_bank_inspector",
        )
        st.success(f"Saved word preview sheet to {output_path}")

    st.divider()
    st.subheader("Renderer Evaluation")
    st.caption(
        "This is still an accuracy proxy, not OCR accuracy. The app now compares context-aware spacing, flat word-bank spacing, and glyph-only rendering against the same held-out sentence crops."
    )

    if st.button("Run Reconstruction Evaluation"):
        context_result = evaluate_dataset_reconstruction(
            dataset=dataset,
            renderer=context_renderer,
            max_samples_per_sentence=2,
            include_copies=False,
        )
        flat_word_bank_result = evaluate_dataset_reconstruction(
            dataset=dataset,
            renderer=flat_word_bank_renderer,
            max_samples_per_sentence=2,
            include_copies=False,
        )
        glyph_only_result = evaluate_dataset_reconstruction(
            dataset=dataset,
            renderer=glyph_renderer,
            max_samples_per_sentence=2,
            include_copies=False,
        )

        comparison_df = pd.DataFrame(
            [
                {
                    "renderer": "context_aware",
                    "samples": context_result.sample_count,
                    "mean_iou": round(context_result.mean_iou, 4),
                    "mean_mae": round(context_result.mean_mae, 4),
                    "mean_word_reuse": round(
                        sum(detail.used_word_samples for detail in context_result.details)
                        / max(1, context_result.sample_count),
                        2,
                    ),
                },
                {
                    "renderer": "flat_word_bank",
                    "samples": flat_word_bank_result.sample_count,
                    "mean_iou": round(flat_word_bank_result.mean_iou, 4),
                    "mean_mae": round(flat_word_bank_result.mean_mae, 4),
                    "mean_word_reuse": round(
                        sum(detail.used_word_samples for detail in flat_word_bank_result.details)
                        / max(1, flat_word_bank_result.sample_count),
                        2,
                    ),
                },
                {
                    "renderer": "glyph_only",
                    "samples": glyph_only_result.sample_count,
                    "mean_iou": round(glyph_only_result.mean_iou, 4),
                    "mean_mae": round(glyph_only_result.mean_mae, 4),
                    "mean_word_reuse": round(
                        sum(detail.used_word_samples for detail in glyph_only_result.details)
                        / max(1, glyph_only_result.sample_count),
                        2,
                    ),
                },
            ]
        )
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        details_df = pd.DataFrame(
            [
                {
                    "renderer": "context_aware",
                    "text": detail.text,
                    "file": detail.path_name,
                    "iou": round(detail.iou, 4),
                    "mae": round(detail.mae, 4),
                    "unsupported_count": detail.unsupported_count,
                    "used_word_samples": detail.used_word_samples,
                }
                for detail in context_result.details
            ]
            + [
                {
                    "renderer": "flat_word_bank",
                    "text": detail.text,
                    "file": detail.path_name,
                    "iou": round(detail.iou, 4),
                    "mae": round(detail.mae, 4),
                    "unsupported_count": detail.unsupported_count,
                    "used_word_samples": detail.used_word_samples,
                }
                for detail in flat_word_bank_result.details
            ]
            + [
                {
                    "renderer": "glyph_only",
                    "text": detail.text,
                    "file": detail.path_name,
                    "iou": round(detail.iou, 4),
                    "mae": round(detail.mae, 4),
                    "unsupported_count": detail.unsupported_count,
                    "used_word_samples": detail.used_word_samples,
                }
                for detail in glyph_only_result.details
            ]
        )
        st.dataframe(details_df, use_container_width=True, hide_index=True)

        preview_options = [
            f"{index}: {detail.text} [{detail.path_name}]"
            for index, detail in enumerate(context_result.details)
        ]
        selected = st.selectbox("Preview a reconstruction triplet", options=preview_options)
        selected_index = int(selected.split(":", maxsplit=1)[0])
        chosen_context_detail = context_result.details[selected_index]
        chosen_flat_detail = flat_word_bank_result.details[selected_index]
        chosen_glyph_detail = glyph_only_result.details[selected_index]
        preview_image = stack_horizontal(
            [
                chosen_context_detail.reference_image,
                chosen_context_detail.rendered_image,
                chosen_flat_detail.rendered_image,
                chosen_glyph_detail.rendered_image,
            ],
            gap=22,
        )
        st.image(
            preview_image,
            caption="Left: reference crop | Then: context-aware | flat word-bank | glyph only",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
