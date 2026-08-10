# Project summary report (LaTeX)

A self-contained summary of the `shearWaveProcessing` project: data sources, pipeline,
GUI, scoring experiments, the phantom/Caenen successes, and the in-vivo acquisition limit.

## Files
- `main.tex` — the report (single file, `thebibliography` inline, no external `.bib`).
- `figures/` — all figures referenced by `main.tex` (copied from `docs/figures/` and the
  data-folder analysis outputs; kept here so the report is self-contained).

## Build
No LaTeX toolchain is installed on the acquisition machine. Options:

- **Overleaf (easiest):** upload the whole `report/` folder (drag the zip in) and compile with
  pdfLaTeX. No package installation needed — everything used is in a standard TeX Live.
- **Local, if you install MiKTeX / TeX Live:**
  ```bash
  cd report
  latexmk -pdf main.tex        # or: pdflatex main.tex  (run twice for references)
  ```

Packages used (all standard): `graphicx, booktabs, amsmath, siunitx, xcolor, caption,
subcaption, enumitem, microtype, hyperref, geometry, lmodern`.
