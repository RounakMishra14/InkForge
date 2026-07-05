# Handwriter v1

This repository is the working prototype for a handwriting synthesis system built around a **single writer's handwriting style**. The immediate goal is straightforward: take typed text as input and render it so the output feels recognizably written by the person represented in the dataset.

Right now I am building the project in measured stages instead of jumping directly to a large generative model. The dataset is relatively small, so the current direction is a modular pipeline that learns spacing, baseline behavior, and visual variation from the writer's samples, then uses those signals to synthesize new text in a controlled way.

## Current Focus

The first implementation batch in this repo is centered on:

- organizing the dataset behind reusable loader utilities
- extracting a baseline style profile from the sentence crops
- rendering typed text through glyph composition as a measurable baseline
- adding reconstruction-style evaluation so each improvement can be checked against held-out sentence samples
- keeping the codebase modular enough to support later additions like word-level retrieval, better spacing models, and stronger style transfer

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
- `src/handwriter/renderer.py`
  - a first-pass glyph composition renderer
- `src/handwriter/evaluation.py`
  - reconstruction metrics for baseline quality checks
- `app.py`
  - a Streamlit interface to inspect style stats, render text, and run evaluation

This is intentionally a baseline, not the final architecture. The renderer is simple enough to debug and measure, which makes it a better starting point for this dataset than an opaque model that overfits early.

## Why This Direction

The current dataset is useful, but it is not large enough for me to assume that a heavy generative model will generalize well to arbitrary text. Because of that, the project starts with a retrieval-and-composition style system:

- reuse the real writer's glyphs
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

The baseline evaluation currently reports:

- IoU on binarized handwriting masks
- normalized mean absolute pixel error
- unsupported character counts

These metrics are only a first proxy, but they are useful for tracking whether spacing, scaling, and placement changes are moving the synthesis in the right direction.

## Near-Term Roadmap

The next iterations I plan to build here are:

1. better duplicate handling and dataset quality checks
2. word-level extraction from sentence images for more natural synthesis
3. stronger spacing models conditioned on character neighbors
4. multi-line layout and export improvements in the Streamlit app
5. more reliable evaluation and visual side-by-side comparisons

## Project Status

This repository is actively under construction. The current code establishes the first measurable synthesis baseline and sets up the modular pieces needed for later refinement.
