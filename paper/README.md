# Paper Draft: dScale (MLSys 2027 template)

LaTeX skeleton for the paper described in [`../Report3.md`](../Report3.md). It uses the official
MLSys 2025 style, which the MLSys 2027 call for papers reuses (2 columns, 10 pages excluding
references; appendix uploaded separately). Submission deadline: **Oct 30, 2026, 20:00 UTC** on
OpenReview.

## Build

```bash
cd paper
latexmk -pdf main.tex        # produces main.pdf
latexmk -C                   # clean build files
```

On TinyTeX, install missing packages with
`tlmgr install microtype subfigure multirow eso-pic forloop xspace`.

## Layout

| File | Content |
|------|---------|
| `main.tex` | Preamble, macros, section includes |
| `sections/00_abstract.tex` | Abstract (numbers are TODO) |
| `sections/01_introduction.tex` | Motivation, key observation, contributions |
| `sections/02_background.tex` | dLLMs, dLLM acceleration, long-context systems, MI355X / AINIC / MoRI |
| `sections/03_characterization.tex` | Per-step cost model, KV footprint table, F1/F2, SGLang dLLM gaps |
| `sections/04_design.tex` | Denoise-step CP, shard-aligned KV streaming, step-aware EP, UMBP tiering, TPF scheduling |
| `sections/05_implementation.tex` | SGLang + MoRI changes, multi-node profiling |
| `sections/06_evaluation.tex` | Q1–Q5, setup tables, placeholder figures F3–F12, quality table |
| `sections/07_related.tex` | Related work |
| `sections/08_conclusion.tex` | Conclusion |
| `refs.bib` | References (arXiv author lists pulled from the arXiv API) |
| `mlsys2025.sty`, `mlsys2025.bst`, … | Official MLSys style files |

## Draft markers

`main.tex` defines markers that print in color while `\drafttrue` is set:

- `\todo{...}` (red): missing text or results.
- `\est{...}` (orange, superscript "est"): analytical estimates that must be replaced by measurements.
- `\hyp{...}` (blue): hypotheses the experiments must confirm or refute.
- `\placeholderfig{label}{caption}{what to plot}`: boxed figure placeholders.

Switch to `\draftfalse` before submission; all markers then disappear, so make sure none remain.
The system name is the `\sys` macro (currently "dScale"); check for name collisions before submitting.

## Track and anonymity

MLSys 2027 has a double-blind **research track** (novelty required; anonymize affiliations and avoid
self-identifying phrasing such as "our MoRI library") and an **industrial track** (author names hidden,
but company and product names allowed; judged on lessons and benchmarks at scale). The track cannot be
changed after the deadline.

## Porting to IPDPS

IPDPS uses the IEEE conference format (`IEEEtran`, 10 pages + references). The sections are
format-independent; swap the preamble in `main.tex` to `\documentclass[conference]{IEEEtran}`, drop
the `mlsys*` commands, and use `IEEEtran.bst`.
