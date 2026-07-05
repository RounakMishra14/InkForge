"""Optional debug and evaluation panels for future dataset improvement work."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from .app_support import PipelineBundle
from .evaluation import EvaluationResult, evaluate_dataset_reconstruction
from .preview import stack_horizontal
from .word_inspector import build_word_contact_sheet, export_word_samples


def render_debug_toggle() -> bool:
    """Expose the advanced tools behind an off-by-default switch."""

    return st.sidebar.toggle("Show debug and evaluation tools", value=False)


def render_debug_panels(pipeline: PipelineBundle, project_root: Path) -> None:
    """Render optional inspection and evaluation views."""

    st.divider()
    st.subheader("Debug and Evaluation")
    st.caption(
        "These tools stay hidden during normal use and can be turned on later when you want to inspect segmentation quality or compare renderers."
    )

    overview_tab, word_tab, eval_tab = st.tabs(
        ["Dataset Overview", "Word Bank Inspector", "Renderer Evaluation"]
    )
    with overview_tab:
        _render_dataset_overview(pipeline)
    with word_tab:
        _render_word_bank_inspector(pipeline=pipeline, project_root=project_root)
    with eval_tab:
        _render_renderer_evaluation(pipeline)


def _render_dataset_overview(pipeline: PipelineBundle) -> None:
    st.subheader("Dataset Summary")
    st.json(
        {
            "total_sentence_samples": pipeline.quality_report.total_sentence_samples,
            "clean_sentence_samples": pipeline.quality_report.clean_sentence_samples,
            "unique_sentence_prompts": pipeline.quality_report.unique_sentence_prompts,
            "explicit_copy_files": pipeline.quality_report.explicit_copy_files,
            "duplicate_glyph_files": pipeline.quality_report.duplicate_glyph_files,
            "duplicate_sentence_files": pipeline.quality_report.duplicate_sentence_files,
            "word_bank_vocabulary": len(pipeline.word_bank.available_words()),
        }
    )

    st.subheader("Style Profile")
    st.json(
        {
            "average_sentence_height": pipeline.style_profile.average_sentence_height,
            "median_baseline_ratio": round(pipeline.style_profile.median_baseline_ratio, 4),
            "average_char_gap": round(pipeline.style_profile.average_char_gap, 2),
            "average_word_gap": round(pipeline.style_profile.average_word_gap, 2),
            "average_ink_density": round(pipeline.style_profile.average_ink_density, 4),
            "supported_sentence_count": pipeline.style_profile.supported_sentence_count,
        }
    )

    st.subheader("Spacing Profile")
    st.json(
        {
            "fallback_gap": round(pipeline.spacing_profile.fallback_gap, 2),
            "pair_override_count": len(pipeline.spacing_profile.pair_gap_overrides),
            "class_override_count": len(pipeline.spacing_profile.class_gap_overrides),
            "token_baseline_jitter": pipeline.spacing_profile.token_baseline_jitter,
            "token_height_jitter": pipeline.spacing_profile.token_height_jitter,
        }
    )
    st.dataframe(
        pd.DataFrame(pipeline.spacing_profile.top_pairs()),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Character Coverage")
    coverage_rows = [
        {"label": label, "samples": count}
        for label, count in pipeline.dataset.coverage_report().items()
    ]
    st.dataframe(pd.DataFrame(coverage_rows), use_container_width=True, hide_index=True)


def _render_word_bank_inspector(pipeline: PipelineBundle, project_root: Path) -> None:
    inventory_df = pd.DataFrame(pipeline.word_bank.inventory_rows())
    st.dataframe(inventory_df, use_container_width=True, hide_index=True)

    available_words = pipeline.word_bank.available_words()
    if not available_words:
        st.info("No reusable word-bank entries are currently available.")
        return

    selected_word = st.selectbox("Inspect extracted word", options=available_words)
    selected_samples = pipeline.word_bank.samples_for(selected_word)
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
            output_dir=project_root / "artifacts" / "word_bank_inspector",
        )
        st.success(f"Saved word preview sheet to {output_path}")


def _render_renderer_evaluation(pipeline: PipelineBundle) -> None:
    st.caption(
        "This remains a development-only quality proxy for comparing context-aware, flat word-bank, and glyph-only rendering on known sentence crops."
    )

    if not st.button("Run Reconstruction Evaluation"):
        return

    context_result = evaluate_dataset_reconstruction(
        dataset=pipeline.dataset,
        renderer=pipeline.context_renderer,
        max_samples_per_sentence=2,
        include_copies=False,
    )
    flat_word_bank_result = evaluate_dataset_reconstruction(
        dataset=pipeline.dataset,
        renderer=pipeline.flat_word_bank_renderer,
        max_samples_per_sentence=2,
        include_copies=False,
    )
    glyph_only_result = evaluate_dataset_reconstruction(
        dataset=pipeline.dataset,
        renderer=pipeline.glyph_renderer,
        max_samples_per_sentence=2,
        include_copies=False,
    )

    st.dataframe(
        pd.DataFrame(
            [
                _summary_row("context_aware", context_result),
                _summary_row("flat_word_bank", flat_word_bank_result),
                _summary_row("glyph_only", glyph_only_result),
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.dataframe(
        pd.DataFrame(
            _detail_rows("context_aware", context_result)
            + _detail_rows("flat_word_bank", flat_word_bank_result)
            + _detail_rows("glyph_only", glyph_only_result)
        ),
        use_container_width=True,
        hide_index=True,
    )

    preview_options = [
        f"{index}: {detail.text} [{detail.path_name}]"
        for index, detail in enumerate(context_result.details)
    ]
    if not preview_options:
        st.info("No evaluation samples were available to preview.")
        return

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


def _summary_row(renderer_name: str, result: EvaluationResult) -> dict[str, object]:
    return {
        "renderer": renderer_name,
        "samples": result.sample_count,
        "mean_iou": round(result.mean_iou, 4),
        "mean_mae": round(result.mean_mae, 4),
        "mean_word_reuse": round(
            sum(detail.used_word_samples for detail in result.details) / max(1, result.sample_count),
            2,
        ),
    }


def _detail_rows(renderer_name: str, result: EvaluationResult) -> list[dict[str, object]]:
    return [
        {
            "renderer": renderer_name,
            "text": detail.text,
            "file": detail.path_name,
            "iou": round(detail.iou, 4),
            "mae": round(detail.mae, 4),
            "unsupported_count": detail.unsupported_count,
            "used_word_samples": detail.used_word_samples,
        }
        for detail in result.details
    ]
