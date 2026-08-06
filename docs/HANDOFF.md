# shearWaveProcessing — handoff

Session-to-session context for continuing this repo. **Read this first**, then `docs/passive_search.md`
for the full passive-SWE investigation record. Last updated 2026-08-04.

## 1. What this repo is

Unified cardiac shear-wave elastography (SWE) pipeline combining the acquisition half (ported from
`SWI/Zea`) and the visualization half (ported from `iq2sws`). Three modular stages driven by `run.py`:

1. `convert`  — Verasonics `.mat` → zea `.hdf5` (converted RF).
2. `beamform` — RF → IQ + B-mode GIFs + per-measurement shear-wave IQ, saved **with the ARF push
   location** (`custom/push_focus_x/z`) and all scan parameters.
3. `viz` (active) / `passive` — IQ → shear-wave space-time plots (+ speed).

Environment: `D:\Luuk van Knippenberg\envs\zea_latest\python.exe`, `KERAS_BACKEND=torch`. `zea` must be
importable for stages 1–2 (numpy/scipy/matplotlib/h5py suffice for stage 3).

## 2. Data & assumptions

- Test dataset: `D:\Luuk van Knippenberg\Claude\invivo_sw` (in-vivo PLAX septum; `CombinedData.mat` +
  `RF_data_*.bin`; `output_old/` holds earlier reference results; `output/mlines/` holds the saved
  **anatomical septal M-lines** `passive_win{i}_mline.npz` + `passive_general_mline.npz`).
- 6-buffer S5-1 SWI Widebeam sequence. **Buffer 2** = active ARF (reference + push + tracking);
  **Buffer 4** = ultrafast diverging-wave stream = **passive** source (~**925.9 Hz**, 4 transmits =
  2 angles × pulse inversion = tracking PRF 3704 ÷ 4); **Buffer 5** = one co-registered B-mode frame
  per push (active M-line drawing).
- **Active SWE**: pre-push reference → `relative_to_reference` displacement; ARF push is a vertical line
  at **x=0** (stored, z=49.28 mm) → **symmetric** response → outward directional filter (`outward`).
- **Passive SWE**: **no reference** → `frame_to_frame` displacement. Natural (valve-closure) shear
  waves; in PLAX they travel **right → left (basal → apical)**. **The k-ω directional filter is now OFF
  for passive** (see §4) — it biased the apparent speed and added reverberation banding.

## 3. Window labelling (invivo_sw) — settled

Burst detection finds 4 windows. Cardiac cycle = 920 − 52 = **868 ms → 69 bpm** (matches ECG). By
timing + same-recipe comparison (`scripts/passive_compare_windows.py`):

| window | peak | event | notes |
|---|---|---|---|
| win0 | 52 ms | **MVC** (early systole) | near-identical to win3 |
| win1 | 377 ms | **AVC** (+325 ms, end systole) | the clean AVC wave, ~2–2.7 m/s (disp) |
| win2 | 550 ms | **diastolic (E-wave) / noise** | +498 ms ≈ mid-diastole, NOT a valve closure — the messy outlier |
| win3 | 920 ms | **MVC** (next cycle) | twin of win0 (one full cycle later) |

Earlier confusion (tuning to win2) came from win2 not being a valve-closure wave. **Do not tune to win2.**

## 4. Key passive findings (what the searches taught us)

Two exhaustive searches (`scripts/search_passive.py` v1, `scripts/search_passive2.py` v2) + a MATLAB
cross-check (`processIQ.m`, `ProcessDW=true`). Full detail in `docs/passive_search.md`.

1. **Directional filter OFF.** The k-ω filter injected horizontal reverberation banding and biased the
   slope toward vertical / too-high speed. Confirmed empirically (`passive_directional_test.py`): AVC
   velocity gives **c ≈ 4 m/s with no filter** (matches Vos: pig 4.2, human 3.5) vs **6 m/s with the
   leftward filter**. MATLAB's passive pipeline uses no directional filter either.
2. **Remove bulk wall motion with a band-pass whose low corner is ≥ ~10 Hz** (5 Hz lets the bulk band
   through). SVD-clutter and polynomial detrend do NOT substitute. A **250 Hz IQ low-pass hurts** (it
   washes the wavefront out, esp. acceleration).
3. **Quantity:** displacement = **cleanest wavefront for visualisation**; velocity noisier; acceleration
   sharpest front but noisiest (a clarity metric based on semblance over-rewards it). For a least-biased
   *speed* number, acceleration is preferred (Petrescu 2022); disp < vel < acc systematically (mild
   Lamb-wave dispersion).
4. **Spatial:** light Gaussian ~0.6 mm (median ≈, slightly worse). **Temporal:** light mean/median or
   none (second-order). **M-lines:** few lines; N=1 best for the sharp acceleration front, more lines
   help smoother quantities; mean ≈ median.
5. **Metric caveat (important):** the envelope `metrics.passive_coherence` AND naive semblance are
   maximised by the flat bulk band → unreliable speed. Use **`metrics.slant_stack_speed`** — a *signed*
   tau-p slant-stack fitting the **wave CENTRE** (not the onset; matches Vos/Petrescu Radon practice).
   For *reporting* call it with `remove_flat=False` (the band-pass already removed the bulk band; the
   `remove_flat=True` default over-corrects and pins at `cmin`). The built-in `ttp_ransac` speed is also
   unreliable in vivo — read montages by eye.
6. **Speeds** are physiological (AVC disp ~2–2.7, MVC disp ~3–5 m/s) but **absolute values remain
   sensitive to M-line obliquity** relative to the true propagation path (MVC vs AVC ordering not yet
   consistent with literature). The *visual* conclusions are robust; the absolute numbers are not.

## 5. Current config + "always 3 space-time plots"

Both paths now **always emit 3 space-time plots per measurement**, via shared
**`runconfig.build_views(cfg, acq)`** → `run.views` (three full, independent recipes). `run.py`
`_process_measurement` (active) and `swp.passive` both use it; falls back to the legacy `run.bands`
grid if `run.views` is absent. Montage orientation via a non-breaking `spacetime_montage(...,
transpose=)` flag: **active** keeps native (r-x, t-y; symmetric V from r0); **passive** uses the
**M-mode orientation** (x=time, y=along-line) so propagation reads as a diagonal.

- **`configs/passive.yaml`** main recipe = displacement · **no directional** · bp10-150 · gauss0.6/1.2 ·
  mean3 · 5 M-lines · no IQ pre-filter. `run.views`:
  - A `disp / bp10-150 / gauss0.6 / mean3 / 5-line`
  - B `disp / bp5-150 / median1.0 / **no temporal** / 9-line`
  - C `velocity / bp15-90 / gauss1.0 / mean5 / 9-line`
- **`configs/active.yaml`** `run.views` = iq2sws consensus top-3 (all displacement, outward directional):
  - A `disp / bp120-700 / gauss / mean3` (oc 0.753)
  - B `disp / bp80-500 / gauss / **no temporal**` (oc ~0.735)
  - C `disp / poly3 / gauss / mean3` (oc 0.728)

Verified: passive on invivo_sw (wave clear in all 3 views for win0/win1/win3); active on a phantom
measurement (3 clean symmetric-V panels, oc 0.89–0.98, SWS ~2.3–2.4 m/s).

## 6. Known issues / caveats

- **Absolute passive speed is M-line-alignment-limited** (§4.6) — the biggest open accuracy issue.
- **`ttp_ransac` + `origin_coherence` are active-oriented** (symmetric, both sides of r0); meaningless
  for the one-sided passive wave. Use `slant_stack_speed` for passive. The passive montage still overlays
  the `ttp_ransac` line (imperfect); title speed is the signed-Radon value.
- **Burst-window peak times shift slightly** when the processing/smoothing config changes (detection
  runs the config's smoothing on the overview). M-lines are index-keyed so still load; but the cropped
  100 ms window edges move a little. `detect.band` is fixed to keep this small.
- **Median spatial full-field filtering is the search runtime bottleneck** (~12 s per op) — decimate the
  field or drop median options for fast sweeps.
- Buffer 4 in `output/` was retrofitted with scan params, not freshly beamformed (`run.py beamform`
  regenerates, ~25 min). Cosmetic: git reports CRLF conversions on commit (Windows).

## 7. Next steps

1. **ECG-anchored window labelling.** Overlay `AcquisitionParametersAndECG.mat` R-peaks to confirm
   MVC/AVC/diastole directly (not by timing arithmetic) and auto-label/gate windows; drop the diastolic
   win2 or mark it. (`data.ecg` hook already exists.)
2. **Proper passive speed estimator.** Promote a normalized-Radon / signed slant-stack (wave **centre**,
   physiological range, both quantities) into `SPEED_METHODS` and overlay *its* line on the montage
   (replace the `ttp_ransac` overlay for passive). Report disp/vel/acc speeds per window (Petrescu).
3. **M-line alignment / geometry.** Ensure the same septal geometry per window (and add **vertical
   M-line** support — see the generalise-direction note below); the absolute-speed discrepancy is
   alignment-driven. A curvature/obliquity check would help.
4. **Metric hardening.** Fold the per-time spatial-mean removal into the ranking metric (or use the
   signed-Radon clarity score) so the exhaustive search stops rewarding the flat band; re-run if the
   ranking shifts.
5. **k-ω / dispersion analysis** (Lamb mode for the thin septum) — the physically correct model; would
   separate phase velocity vs a single group speed. Vos found only mild dispersion, so a group speed is
   defensible meanwhile.
6. **Validate on cleaner / more data.** AVC is the stronger wave; a higher-SNR or ECG-gated acquisition
   would let the speed estimators be validated, and enable a cross-subject consensus (as active had).
7. **(Lower priority) generalise directional naming.** `leftward`/`rightward` are horizontal-M-line
   names; for vertical M-lines rename to origin-relative (`from_start`/`from_end`). Only matters if the
   directional filter is re-enabled for some case — it is OFF for passive now.

## 8. Tooling index

- `scripts/search_passive.py` — v1 exhaustive search (6 720/window, `passive_coherence`).
- `scripts/search_passive2.py` — v2 (11 520/window, **all 7 axes**: directional on/off, N-line mean/median,
  IQ pre-filter, band/SVD/poly, gauss/median spatial, mean/median temporal, disp/vel/acc), signed-Radon
  clarity score → top-16 + **per-axis marginal montages** (`--window i --label AVC/MVC`).
- `scripts/passive_best_montage.py`, `passive_filter_variety.py`, `passive_slantstack.py`,
  `passive_directional_test.py`, `passive_compare_windows.py` — targeted montages (see `passive_search.md`).
- Metrics (`src/swp/viz/metrics.py`): `slant_stack_speed` (signed tau-p, **use for passive speed**),
  `passive_coherence` (envelope; do NOT trust its speed), `origin_coherence`/`wavefront_coherence` (active).
- Shared: `runconfig.build_views`; `spacetime_montage(transpose=)`.
- Outputs (git-ignored): `<folder>/output/swp_passive/`, `.../swp_passive/search/`, `.../search2/`,
  `.../swp_active/`; M-lines in `<folder>/output/mlines/`.

## 9. Quick commands

```
python run.py beamform "<folder>"                               # stage 1+2 (writes output/, push location)
python run.py viz      "<folder>" --config configs/active.yaml [--meas N]   # active -> 3 views/meas
python run.py viz      "<folder>" --config configs/active.yaml --phantom    # phantom (horizontal M-line)
python run.py passive  "<folder>"                               # passive -> 3 views x windows montage
python run.py passive  "<folder>" --redraw                      # re-draw all passive M-lines
# re-run a passive exhaustive search:
python scripts/search_passive2.py --window 1 --label AVC
```
