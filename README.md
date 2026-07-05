# InkForge

InkForge is a personalized handwriting note-making app built from a **single writer's dataset**. The goal is not just to render pretty handwriting once, but to turn typed notes into reusable notebook-style pages that still feel recognizably written by the person in the dataset.

This repository did not begin as a polished product. It started as a careful baseline experiment: can a small handwriting dataset be turned into a controllable synthesis system without immediately relying on a large generative model? The answer so far is yes, but only by building in stages, measuring often, and tolerating the limitations of the dataset instead of pretending they do not exist.

## Why InkForge Exists

The dataset available for this project is limited. It contains useful handwriting evidence, but not enough coverage to confidently train a large end-to-end model that can write arbitrary clean notes on demand. Because of that, InkForge takes a modular path:

- reuse real handwritten glyphs where possible
- reuse exact handwritten words when the dataset already contains them
- learn spacing and layout signals from sentence crops
- synthesize unsupported or weakly represented symbols in a controlled fallback path
- package everything into a notebook-style app instead of a raw debugging demo

That strategy gives the project something extremely important early on: a feedback loop. Every step can be inspected, improved, and measured.

## How The Project Started

The first stage was a straightforward glyph-composition baseline. At that point, the system could:

- index isolated letters, digits, and symbols
- select random glyph samples from the dataset
- place them on a shared baseline
- render short handwritten-looking lines from typed input

That baseline was useful because it proved the end-to-end path worked, but it also exposed the first major weakness immediately: handwriting is not just a bag of isolated characters. Even when the right characters were present, the output still felt mechanical because spacing, rhythm, and word shape matter a lot.

## How The System Evolved

InkForge has grown in several implementation batches:

1. **Modular baseline**
   A dataset loader, style analysis layer, and basic line renderer were built first so the core system was inspectable instead of monolithic.
2. **Word-bank retrieval**
   The sentence dataset was segmented into reusable handwritten word crops so exact words could be reused instead of always composing them letter by letter.
3. **Evaluation and inspection tooling**
   Reconstruction scoring, renderer comparisons, and word-bank inspection views were added so progress could be measured rather than guessed from screenshots.
4. **Context-aware spacing**
   Spacing estimation became more data-driven by reconciling segmented words with glyph widths and learning pair-aware spacing overrides.
5. **Paragraph and page rendering**
   The renderer moved from single-line previews to multi-line output, then to notebook-style pages with padding, line spacing, export, and multi-page layout.
6. **Notebook productization**
   The Streamlit UI was split into modules, debug tools were hidden behind a toggle, and the visible app was reshaped into a note-writing workflow instead of a development dashboard.
7. **Symbol fallback hardening**
   Weak or missing characters such as bullets, `=`, slashes, brackets, and other note-taking symbols gained synthetic fallback rendering so general note input would not break visually.

## Where We Are Right Now

InkForge is currently in the **working prototype / product-shaping stage**.

That means:

- the core rendering pipeline already works
- the app is now usable for note-style generation
- exported output can span multiple notebook pages
- debug and evaluation tooling still exists for future improvement
- quality is good enough to be useful, but not yet polished enough to be called finished

The project is no longer just a rendering experiment. It is now actively being shaped into a personalized notebook app.

## What InkForge Can Do Today

At the current stage, InkForge can:

- take typed note content and render it in a personalized handwriting style
- reuse real handwritten words from the dataset when they are available
- fall back to glyph composition for unseen words
- apply context-aware spacing rather than only flat character gaps
- wrap long content across multiple lines and multiple pages
- generate notebook-style pages with margins and paper styling
- support blank, ruled, and grid-like page layouts
- change paper color and ink color
- add title highlighting and bolder handwritten output
- download output page-by-page as PNG images
- export the full note set as a PDF
- keep dataset inspection and evaluation tools available behind a debug mode

## What Data InkForge Uses

The current dataset includes:

- isolated uppercase and lowercase characters
- isolated digits
- isolated punctuation and symbols
- sentence-level handwriting crops
- full-page source images used to create sentence samples

The sentence-level samples are especially important because they carry much more than text identity. They reveal:

- line rhythm
- word spacing
- baseline drift
- stroke darkness
- typical vertical footprint

Those signals are what make the output feel closer to the real writer instead of looking like clean font assembly.

## Difficulties We Faced

This project has been shaped more by constraints than by convenience. The main difficulties so far have been:

### 1. Small dataset size

There is not enough data to safely assume a large generative model would generalize well. We tolerated that by using a retrieval-and-composition pipeline first, which makes better use of sparse data.

### 2. Weak symbol coverage

Some characters either do not exist in the dataset or exist only as poor samples. This became especially visible in note-taking use cases where bullets, operators, and structural symbols matter. We tolerated that by adding synthetic symbol fallbacks for weak or missing characters.

### 3. Word segmentation noise

Extracting reusable handwritten words from sentence crops is useful, but imperfect. When segmentation misses boundaries, it can poison the word bank. We tolerated that by filtering aggressively, keeping fallback behavior deterministic, and exposing debug inspection tools.

### 4. Spacing realism

Early glyph-only output looked too flat and too mechanical. We tolerated that by learning spacing from sentence-derived signals and then improving it with pair-aware overrides from segmented words.

### 5. Product versus prototype tension

As soon as the renderer became capable enough, the app risked becoming cluttered with debug statistics instead of feeling usable. We tolerated that by modularizing the UI and moving diagnostic tools into a separate debug mode.

### 6. Unsupported general writing patterns

Notebook writing includes bullets, checkboxes, operators, ratios, brackets, and path-like text. The original baseline was not ready for that. We tolerated it by normalizing note prefixes and teaching the renderer to synthesize common note-taking symbols when the dataset cannot support them directly.

## How We Have Tolerated The Constraints So Far

The project has remained practical by following a few rules:

- prefer controlled synthesis over pretending unsupported cases do not exist
- keep fallbacks explicit instead of silently failing
- measure renderer quality against known sentence crops
- preserve debug tooling even while simplifying the normal app experience
- improve the product in layers rather than rewriting the whole approach each time

That is the reason InkForge is still moving forward despite dataset limits. The project is not trying to solve everything with one leap.

## Current Architecture

The repository currently centers around these modules:

- `src/handwriter/dataset.py`
  Dataset indexing and label normalization.
- `src/handwriter/style.py`
  Sentence-level style analysis and writer-profile aggregation.
- `src/handwriter/words.py`
  Transcript-aware word extraction and filtered word-bank construction.
- `src/handwriter/spacing.py`
  Context-aware spacing estimation from segmented words and glyph-width reconciliation.
- `src/handwriter/renderer.py`
  Core handwriting line renderer using exact word reuse, glyph fallback, and synthetic symbol fallback.
- `src/handwriter/synthetic_symbols.py`
  Synthetic symbol generation for note-taking and weak dataset cases.
- `src/handwriter/paragraph.py`
  Multi-line and multi-page notebook layout logic.
- `src/handwriter/page_design.py`
  Paper styling, ink compositing, and PNG/PDF export helpers.
- `src/handwriter/ui_main.py`
  Main note-taking UI for everyday use.
- `src/handwriter/ui_debug.py`
  Debug-only inspection and evaluation panels.
- `src/handwriter/app_support.py`
  Shared pipeline/bootstrap helpers used by the Streamlit app.
- `app.py`
  Thin Streamlit entrypoint for InkForge.

## Running InkForge

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the app:

```bash
streamlit run app.py
```

## How Progress Is Evaluated

InkForge still uses proxy metrics internally because there is no perfect “handwriting realism score.” The current evaluation workflow can report:

- IoU on binarized handwriting masks
- normalized mean absolute pixel error
- unsupported character counts
- word-bank reuse counts
- side-by-side renderer comparison on held-out samples
- word-bank inspection views
- spacing-profile summaries

These tools are still important even though the main UI now feels more product-like. They are how future improvements will be judged.

## What Stage The App Is In

The best description today is:

**InkForge is a functional personalized note-taking prototype with active quality-improvement work still in progress.**

It is already useful for:

- drafting personal study notes
- generating notebook-style handwritten pages
- exporting assignments or revision pages
- experimenting with personalized handwriting synthesis

It still needs more refinement in:

- symbol polish for edge cases
- even better word segmentation
- stronger writer-specific layout learning from source pages
- better handling of some math-heavy or structured text patterns

## Near-Term Direction

The most useful next steps are:

1. refine synthetic symbol shapes so operator-heavy text looks more natural
2. improve word segmentation so the reusable word bank grows safely
3. learn more page-layout behavior directly from the writer's source pages
4. improve duplicate filtering and sample hygiene further
5. strengthen support for structured note patterns like checklists and formula-heavy pages

## Project Status

InkForge is under active construction, but it has already moved past the “toy demo” stage. It now has a measurable rendering pipeline, reusable handwritten word retrieval, context-aware spacing, notebook-style page generation, export support, debug tooling, and fallback handling for real note-taking text.

The project started as a baseline renderer. It is currently becoming a personalized handwriting notebook app.
