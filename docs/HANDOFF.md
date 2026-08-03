# shearWaveProcessing — handoff

Session-to-session context for continuing this repo. Read this first.

## 1. What this repo is

Unified cardiac shear-wave elastography (SWE) pipeline combining the acquisition half (ported
from `SWI/Zea`) and the visualization half (ported from `iq2sws`). Three modular stages driven by
`run.py`:

1. `convert`  — Verasonics `.mat` → zea `.hdf5` (converted RF).
2. `beamform` — RF → IQ + B-mode GIFs + per-measurement shear-wave IQ, saved **with the ARF push
   location** (`custom/push_focus_x/z`) and all scan parameters.
3. `viz` / `passive` — IQ → shear-wave space-time plots (+ speed).

Environment: `D:\Luuk van Knippenberg\envs\zea_latest\python.exe`, `KERAS_BACKEND=torch`. `zea`
must be importable for stages 1–2.

## 2. Data & assumptions

- Test dataset: `D:\Luuk van Knippenberg\Claude\invivo_sw` (in-vivo PLAX septum; `CombinedData.mat`
  + `RF_data_*.bin`; `output_old/` holds earlier reference results).
- 6-buffer S5-1 SWI Widebeam sequence. **Buffer 2** = active ARF (reference + push + tracking);
  **Buffer 4** = ultrafast diverging-wave stream = **passive** source; **Buffer 5** = one co-registered
  B-mode frame per push (for active M-line drawing).
- **Active SWE**: has a pre-push reference → `relative_to_reference` displacement; ARF push is a
  **vertical line at x=0** (confirmed from the stored push location, x=0.00 mm, z=49.28 mm), so the
  response is **symmetric** and the directional filter keeps waves travelling **outward** from r0
  (`directional_mode: outward`).
- **Passive SWE**: **no reference** → `frame_to_frame` displacement. Sampling rate of buffer 4 is
  **~925.9 Hz** (4 transmits = 2 angles × pulse inversion; = tracking PRF 3704 Hz ÷ 4; confirmed vs
  `meta` ActualFPS and the stored timestamps). The shear wave originates at the **valve closure**,
  which in this data sits at the **right (high-r) end of the M-line**, and travels **right → left**.
  So directional filtering is applied to the **whole M-line in one direction** (`directional_mode:
  leftward`, i.e. keep −r-travelling) — **not** symmetric; its only purpose is to remove reflections.
  Flip to `rightward` if a future M-line is drawn with the valve at the left end.

## 3. What is done

- **Stages 1–2** (`src/swp/acquisition/`): faithful port of `beamform_swi.py` / `swi_sequence.py` /
  `make_gifs.py` / `swi_txsettings.py`. Added: `push_focus_x/z` custom elements in the active saver;
  `scanparams.append_scan_params_to_iq` retrofits demod/tx/probe freq, sound_speed, wavelength, prf,
  dz, dx onto the active (buffer 2) and passive (buffer 4) IQ. `convert_folder` for stage 1 alone.
- **zea env workarounds** (also break the *original* `beamform_swi.py` in this env, not our port):
  `_ensure_cpu_t_peak` (zea's `Parameters.t_peak` does `np.asarray` on a CUDA tensor); dropped the
  `warn_missing_optional_fields`+`ignore_warnings` combo that `File.create` now rejects.
- **M-line** (`src/swp/mline/select.py`): ported interactive selector; `.npz` (points + n_samples)
  reused automatically. Active M-lines keyed by measurement; passive by **window index**.
- **Active viz** (`run.py viz --config configs/active.yaml`): the settled iq2sws in-vivo recipe
  (Loupas, rel-to-reference displacement, drop frame 0, band-pass 120–700, Gaussian smooth
  0.6/1.2 mm, temporal mean 3, outward-directional), r0 anchored on the **stored** push location
  (`focus.mode: stored`), multi-band montage. Validated: meas14 origin_coherence 0.94.
- **Passive viz** (`run.py passive <folder>`, `src/swp/passive.py`): general M-line → along-line
  displacement over the whole recording → **burst detection** on a **fixed band** (`detect.band`,
  independent of the processing band, so windows are stable) → ~100 ms window per valve closure →
  **fresh M-line per window** (`output/mlines/passive_win{i}_mline.npz`, reused on re-run; `--redraw`
  to force) → **montage** + a **full-window space-time plot** (`passive_full_spacetime.png`). Config:
  `frame_to_frame`, band-pass 5–150 Hz, `directional_mode: leftward`. On invivo_sw it finds 4 windows
  (~52, 377, 549, 920 ms).

## 4. Known issues / NOT done (the important part)

- **The passive metric and auto-speed are wrong for passive.** `metrics.origin_coherence` and the
  `ttp_ransac` speed are designed for the **symmetric active** case (energy on both sides of r0). With
  the passive unidirectional filter and r0 at the M-line end, one side is empty → `origin_coherence`
  reads ~0 and the speed fit fails (blank/q=0). The space-time *images* are correct; only the reported
  scores are meaningless. **A passive-specific metric + one-sided speed fit is the first task below.**
- **The passive recipe is untuned.** Band 5–150 Hz, Gaussian 0.6/1.2, temporal mean 3, leftward — all
  first guesses. No systematic search has been run (that is the next big step).
- **Passive origin (r0) is a heuristic** (M-line end). A per-window, physically anchored valve-plane
  origin would be better.
- Buffer 4 in `output/` was **reused from `output_old/` and retrofitted** with scan params this
  session, not freshly beamformed; `run.py beamform` regenerates it (~25 min; heavy).
- Active passive share the iq2sws speed methods, which the memory notes are unreliable in vivo —
  read montages manually.
- Cosmetic: git reports CRLF conversions on commit (Windows).

## 5. Next step — exhaustive method/filter/parameter search for passive SWE (plan)

Mirror what was done for active in `iq2sws/archive/` (`search.py` ≈ 6 k combos/measurement, cached
Loupas + precomputed bilinear M-line sampler + vectorised metric, ranked by `origin_coherence`; then
`analyze_search.py` re-ranks and montages, `summarize_search.py` aggregates). Adapt for passive:

**5.1 Define a passive metric (do this first).** Add `metrics.passive_coherence(st, r0, side)` — a
**one-sided** origin coherence: slant-stack the Hilbert envelope along the outward-from-the-valve
moveout `t = t0 + |r − r0| / c` on the single valid side (leftward), best `c ∈ [0.5, 10] m/s` over a
near-origin window `|r − r0| ∈ [2, 16] mm`, requiring an aligned origin time `t0` near the event
start (rejects late reverberation and rightward/standing energy). This is `origin_coherence` reduced
to one side. Validate on a synthetic leftward wave (add to `tests/`). Also add a one-sided speed fit
(TOF/TTP restricted to r < r0).

**5.2 Search axes** (per valve-closure window; the window set comes from the fixed-band burst
detection so it is stable across the sweep):
- quantity: displacement / velocity / acceleration
- detrend / motion: none / polynomial_drift(order) / temporal_bandpass(f_lo, f_hi)
- band grid: f_lo ∈ {2,5,10,20}, f_hi ∈ {80,120,150,200,300} Hz (respect the ~925 Hz Nyquist ≈ 463 Hz)
- spatial smooth: none / gaussian(σ_z, σ_x) / median
- temporal smooth: none / moving_mean(window) / moving_median(window)
- directional: leftward / off (confirm leftward really helps)
- M-line offsets: n ∈ {1,5,7}, spacing ∈ {0.5,0.8,1.2} mm
Estimate ~few thousand combos/window — keep it tractable with the caching below.

**5.3 Efficiency** (reuse the iq2sws pattern): cache the `frame_to_frame` Loupas displacement per
window once (it is the same for all field-filter combos on that window); precompute the bilinear
M-line sampler for the fixed per-window M-line; vectorise `passive_coherence`. Full-res per 100 ms
window is cheap (~130 frames); the whole sweep should be minutes per window.

**5.4 Deliverables / scripts** (put under `scripts/`, outputs under `results/passive/`, git-ignored):
- `scripts/search_passive.py --folder <f> [--window i]` → per-window leaderboard (ranked by
  `passive_coherence`) + a top-k montage.
- `scripts/analyze_passive.py` → consensus recipe across windows (and across subjects once more data
  exists), + summary figure of score vs window/cardiac-phase.
- Write the consensus back into `configs/passive.yaml` (as the active recipe was).

**5.5 Acceptance:** a single fixed passive recipe that reaches most of the per-window best
`passive_coherence` across windows, with visually clean leftward wavefronts in the montage, and a
one-sided speed that is stable where the wave is clear.

## 6. Quick commands

```
python run.py beamform "<folder>"                              # stage 1+2 (writes output/, push location)
python run.py viz      "<folder>" --config configs/active.yaml [--meas N]
python run.py passive  "<folder>"                              # general + per-window M-lines -> montage
python run.py passive  "<folder>" --redraw                     # re-draw all passive M-lines
```

Outputs: `<folder>/output/swp_active/` and `<folder>/output/swp_passive/`
(`passive_bursts.png`, `passive_full_spacetime.png`, `passive_windows_montage.png`); M-lines in
`<folder>/output/mlines/`.
