# Project summary report (LaTeX)

A self-contained summary of the `shearWaveProcessing` project: data sources, pipeline,
GUI, scoring experiments, the phantom/Caenen successes, and the in-vivo acquisition limit.

## Files
- `main.tex` — the project summary through 2026-08-10 (single file, `thebibliography` inline,
  no external `.bib`).
- `acquisition_sweep_log.tex` — companion **documentation log** for the 2026-08-17/18 acquisition
  sweeps and their analysis (2026-08-19). Covers the four questions asked of the new phantom,
  probe-comparison and in-vivo datasets, the buffer-2 receive-layout bug that invalidated the first
  round of processing, and the revised parameter recommendation. Standalone — compiles on its own.
  Its figures are the `sweep_*.png` files in `figures/`.
- `passive_methods/passive_methods.pdf` (+ `.tex`, `figures/`) — atlas of the **passive** SWE
  processing methods explored (quantity, band-pass, directional filter, smoothing, M-line
  averaging, motion/clutter handling, literature recipes, production views) as space-time plots on
  seven in-vivo windows, no automatic speeds (2026-09-24). Figures from
  `study/analysis/passive_methods_atlas.py`; compiled with `tectonic` (in the zea env:
  `tectonic passive_methods.tex`).
- `figures/` — all figures referenced by both documents (copied from `docs/figures/` and the
  per-dataset `analysis/` outputs; kept here so the reports are self-contained).

> **Note:** `main.tex` §9 recommends 61 push elements. `acquisition_sweep_log.tex` §7 supersedes
> that: the recommendation holds at matched *transmit voltage* but inverts at matched *acoustic
> output*, which is the binding constraint. Read the log before acting on the main report's
> acquisition advice.

## Build
No LaTeX toolchain is installed on the acquisition machine. Options:

- **Overleaf (easiest):** upload the whole `report/` folder (drag the zip in) and compile with
  pdfLaTeX. No package installation needed — everything used is in a standard TeX Live.
- **Local, if you install MiKTeX / TeX Live:**
  ```bash
  cd report
  latexmk -pdf main.tex                 # or: pdflatex main.tex (twice, for references)
  latexmk -pdf acquisition_sweep_log.tex
  ```

Packages used (all standard): `graphicx, booktabs, amsmath, siunitx, xcolor, caption,
subcaption, enumitem, microtype, hyperref, geometry, lmodern, longtable, multirow`.
