# Handwriter v1

This repository is the working prototype for a handwriting synthesis system built around a **single writer's handwriting style**. The immediate goal is straightforward: take typed text as input and render it so the output feels recognizably written by the person represented in the dataset.

Right now I am building the project in measured stages instead of jumping directly to a large generative model. The dataset is relatively small, so the current direction is a modular pipeline that learns spacing, baseline behavior, and visual variation from the writer's samples, then uses those signals to synthesize new text in a controlled way.

## Current Focus

The project has now moved into the second implementation batch. The active focus is:

- keeping the glyph-based baseline intact as a controlled fallback
- extracting reusable handwritten word crops from the sentence dataset
- using exact word retrieval when the typed input overlaps with known words
- reporting duplicate pressure and dataset quality issues more explicitly
- improving evaluation so reconstruction results can be inspected visually, not only numerically

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
  - transcript-aware word segmentation and word-bank construction
- `src/handwriter/quality.py`
  - duplicate and copy detection helpers for dataset hygiene
- `src/handwriter/renderer.py`
  - a renderer that can combine exact handwritten word reuse with glyph fallback
- `src/handwriter/evaluation.py`
  - reconstruction metrics plus preview-ready evaluation payloads
- `app.py`
  - a Streamlit interface to inspect style stats, dataset quality, render text, and preview evaluation pairs

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
- side-by-side reference versus reconstruction previews in the app

These metrics are still only a proxy, but they are already useful for checking whether the renderer is improving in the right direction and whether word-level reuse is actually helping rather than just sounding good in theory.

## Current Capabilities

At this stage the renderer can:

- synthesize arbitrary text using isolated character, digit, and symbol glyphs
- reuse exact handwritten word samples when the input overlaps with the sentence dataset
- fall back to glyph composition for unseen words
- expose basic dataset quality indicators so duplicate-heavy subsets are easier to spot
- show visual reconstruction comparisons directly in the Streamlit app

## Near-Term Roadmap

The next iterations I plan to build here are:

1. improve the word segmentation heuristics and reduce fallback cases
2. add character-neighbor-aware spacing instead of a single average gap model
3. support multi-line paragraph rendering and export-friendly layouts
4. build better duplicate filtering so repeated samples do not over-influence style statistics
5. compare word-reuse rendering against pure glyph rendering in the same evaluation view

## Project Status

This repository is actively under construction. The current code now includes both the first measurable synthesis baseline and a second-stage word retrieval layer, which gives the project a stronger foundation for producing this specific writer's style more faithfully.
