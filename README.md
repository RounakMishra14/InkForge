# Handwriter v1

This repository is the working prototype for a handwriting synthesis system built around a **single writer's handwriting style**. The immediate goal is straightforward: take typed text as input and render it so the output feels recognizably written by the person represented in the dataset.

Right now I am building the project in measured stages instead of jumping directly to a large generative model. The dataset is relatively small, so the current direction is a modular pipeline that learns spacing, baseline behavior, and visual variation from the writer's samples, then uses those signals to synthesize new text in a controlled way.

## Current Focus

The project has now moved into the third implementation batch. The active focus is:

- comparing word-bank-assisted rendering against glyph-only rendering directly
- inspecting extracted word samples before trusting them in synthesis
- keeping unreliable fallback word crops out of the active word bank
- exporting word-preview sheets so segmentation issues are easier to review outside the app
- tightening the feedback loop between segmentation quality and rendering quality

## Dataset Summary

The working dataset currently contains:

- isolated uppercase and lowercase characters
- isolated digits
- isolated punctuation and symbols
- sentence-level handwriting crops
- full-page source images used to create the sentence samples

The sentence samples are especially important in this project because they expose the writer's line rhythm, word spacing, baseline drift, and stroke consistency. Even though the first renderer is glyph-based, the sentence folder is the main source for style guidance.

## What Exists Today

The repository currently includes:

- `src/handwriter/dataset.py`
  - dataset indexing and label normalization
- `src/handwriter/style.py`
  - sentence-level style analysis and profile aggregation
- `src/handwriter/words.py`
  - transcript-aware word segmentation and a reliable-only word-bank construction path
- `src/handwriter/quality.py`
  - duplicate and copy detection helpers for dataset hygiene
- `src/handwriter/renderer.py`
  - a renderer that can combine exact handwritten word reuse with glyph fallback
- `src/handwriter/evaluation.py`
  - reconstruction metrics plus preview-ready evaluation payloads
- `src/handwriter/word_inspector.py`
  - word contact-sheet generation and artifact export helpers
- `app.py`
  - a Streamlit interface to inspect style stats, dataset quality, render text, inspect extracted words, and compare renderer outputs

This is still a measured baseline rather than the final architecture. The main difference now is that the system can reuse full handwritten words from the sentence set when they are available, which should move the output closer to the writer's real line rhythm before introducing more complex modeling.

## Why This Direction

The current dataset is useful, but it is not large enough for me to assume that a heavy generative model will generalize well to arbitrary text. Because of that, the project starts with a retrieval-and-composition style system:

- reuse the real writer's glyphs
- reuse full handwritten words when the dataset already contains them
- learn spacing and layout behavior from the sentence crops
- measure how close the generated result is to known samples
- improve incrementally from a stable baseline

That gives the project a clear feedback loop from the start instead of relying on visual guesswork.

## Running the Prototype

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start the app:

```bash
streamlit run app.py
```

## How I Am Evaluating Progress

For now, "accuracy" in this repo means **how well the current renderer can reconstruct the overall look of known sentence samples** when given the same text transcript.

The current evaluation reports:

- IoU on binarized handwriting masks
- normalized mean absolute pixel error
- unsupported character counts
- word-bank reuse counts
- reference versus reconstruction previews in the app
- renderer-to-renderer comparison against the same held-out samples

These metrics are still only a proxy, but they are already useful for checking whether the renderer is improving in the right direction, whether word-level reuse is actually helping, and whether the extracted word bank is reliable enough to trust.

## Current Capabilities

At this stage the renderer can:

- synthesize arbitrary text using isolated character, digit, and symbol glyphs
- reuse exact handwritten word samples when the input overlaps with the sentence dataset
- fall back to glyph composition for unseen words
- compare word-bank-assisted output against glyph-only output in the same interface
- inspect per-word extracted samples before reusing them
- expose basic dataset quality indicators so duplicate-heavy subsets are easier to spot
- export word preview sheets to `artifacts/` for manual inspection
- show visual reconstruction comparisons directly in the Streamlit app

## Near-Term Roadmap

The next iterations I plan to build here are:

1. improve the word segmentation heuristics so the reliable word bank grows beyond the current filtered subset
2. add character-neighbor-aware spacing instead of a single average gap model
3. support multi-line paragraph rendering and export-friendly layouts
4. build better duplicate filtering so repeated samples do not over-influence style statistics
5. add side-by-side error analysis for the worst-performing evaluation samples

## Project Status

This repository is actively under construction. The current code now includes a measurable baseline, a filtered word-retrieval layer, and inspection tools to verify whether extracted word samples are trustworthy before they influence the final handwriting output.
