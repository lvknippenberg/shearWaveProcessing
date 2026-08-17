# shearWaveProcessing — handoff

Session-to-session context for continuing this repo. **Read this first**, then `docs/passive_search.md`
for the full passive-SWE investigation record. Last updated 2026-08-14.

## 0. LATEST (2026-08-14): S5-1 hydrophone safety — direct push & imaging measurements

Direct hydrophone characterisation of the S5-1 shear-wave sequences (NI PCI-5112 + Scope-SFP `.hws`),
with the 2026-08-10 calibration correction applied. Code + data in
`D:\Luuk van Knippenberg\Claude\MI estimation\`; results also written to the acquisition repo
(`SWI/Mechanical index/README.md`, 2026-08-14 section) and `docs/mechanical_index_safety.md` still holds
the narrative. Main outcomes:

- **L11-5 validation vs Verasonics:** after fixing an **elevation misalignment** and saturation,
  corrected MI ≈ **1.23× Verasonics** (within hydrophone-MI uncertainty); **50 Ω = 1 MΩ÷2** confirms the
  ÷2 impedance fix; the old uncorrected 1 MΩ reading was ~2× high.
- **S5-1 pushes are I_sppa.3-limited** (not MI, not I_spta): max **~20 V (79 el)**, **~24 V (61 el)**;
  MI only ~1.7 there. Depth for derating taken from `Home` (transducer centre) = 37.5 mm at the peak.
- **I_spta.3 non-binding** thanks to the real burst duty (**20 Hz for 1.2 s, then ≥30 s off →
  0.77 Hz effective**); a continuous-20 Hz assumption would have wrongly bound it at ~10 V. Still owe a
  transient-TI check for the 1.2 s burst.
- **Pulse length scales I_spta only** (MI, I_sppa unchanged) — confirmed from 1900 vs 1500-cycle data.
- **Focal gain ~1/F# (pressure ~ N^0.6–0.8):** fewer elements → higher allowed voltage but weaker/wider
  push. **41 el is estimated, not measured** (MI ~1.5–1.8).
- Tools built: `readHWS.m` (NI HWS reader), `scanPlan.m`, `measurementPlan.m`, `GenerateSafetyFigures.m`,
  `SafetyTable.m`, `ConfirmPulseLength.m`.

### Open TODO (this session)

- **Measure old vs new push settings** at the S5-1 peak location to validate the extrapolations:
  **(a) 25 V, 61 elements, 1900 cycles** (recommended config) and **(b) 30 V, 41 elements, 1900 cycles**
  (alternative). Confirm I_sppa ≈ 190 / MI, and whether 41 el @ 30 V breaches I_sppa (estimate 144–199).
- **Transient thermal (TI)** for the 1.2 s push burst — the burst-averaged I_spta doesn't capture it.
- L12-3 safety table still to do; S5-1 + L11-5 now done.

## 0. LATEST (2026-08-10): LaTeX report, manual-speed tool, cardiac-cycle speeds, MI/safety, GUI fixes

Session focused on a written summary of the whole project, a usable manual-speed workflow, and two
physics/safety questions. Main outcomes:

- **LaTeX project summary written: `report/`** (`main.tex` + `report/figures/`, standalone, inline
  bibliography — no LaTeX toolchain on this machine, compile on Overleaf / local TeX). Covers data
  sources, M-line selection + variance, pipeline + methods, the GUI, the scoring/metric-validation
  campaign, the optimal phantom recipe + voltage sweep, the acquisition recommendation (61 el + ~800 µs),
  the Caenen success, the in-vivo failure, and next steps. **New figures generated this session** (in
  `report/figures/`, reproducible via scratchpad scripts): `invivo40_push_nopush_montage.png` (optimal
  bp120-700 recipe on ALL 24 in-vivo 40 V pushes, PUSH vs its own NO-PUSH control — the quiet-diastasis
  pushes m5/m6/m19/m23 are the only ones with positive push−nopush coherence contrast, none a clean V);
  `caenen_push_nopush_montage.png` (Caenen-tuned velocity/bp120-350 on ALL 52 pushes, push>nopush in
  ~85% — the positive control the in-vivo lacks); and an **appendix** showing disp+velo, push+nopush for
  all three datasets (`appendix_phantom.png`, `appendix_invivo40.png`, `appendix_caenen_a/b.png`).
- **No-push control data-leakage note (documented in the report §11):** for reference-*trained* filters
  the control uses a **held-out split reference** (train on the 1st half of the pre-push frames, test on
  the disjoint 2nd half) — this is what exposed reference-subspace/SVD *fabricating* a false V.
- **GUI default bug FIXED:** the directional-filter selectbox defaulted to **none** (`DIRECTIONAL_METHODS`
  had `NONE` first) so the GUI default did NOT match the settled optimum. Reordered so **outward is the
  default** (`swp_gui/registry.py`); verified push-3 oc 0.816→0.931 (matches the report montage). Restart
  streamlit / Reset-ALL to pick it up.
- **Manual-speed tool rewritten (`swp_gui/speedline/` custom Plotly component + `swp_gui/render.py`
  `spacetime_png_datauri`/`fig_spacetime_with_lines`).** Was a laggy 2-click Plotly-select (3 s/click).
  Now: a **matplotlib backdrop** (same RdBu_r/clim as the grid, fixed r0 baked into the image so it can't
  be dragged), starts **empty**, click-drag to draw a wavefront, **＋ add line** for multiple lines, drag
  an endpoint (one point) or the line body (both); all in-browser, commits on release — no per-click
  rerun. Every line's speed is listed in Streamlit (never clipped). **Fixed height** (dynamic
  `scrollHeight` sizing caused an unresponsive resize loop — reverted). Clear-all forces a full re-plot
  (no stray active-shape rectangle). The heavy 3×3 quantity grid is now **PNG-cached** so unrelated
  reruns are instant. **💾 save** writes `speed_meas{m}.png` (annotated) + `speed_meas{m}.json`
  (endpoints, speed, and the fit `t = a·r + b`, a in ms/mm → speed = 1/|a|) into
  `<dataset folder>/speed_measurements/` (Caenen → its `SWE_results/`). One file per push, overwrites.
- **Manual-speed RESULTS + cardiac-cycle plot (`scripts/plot_speed_cardiac.py`).** User measured 9
  in-vivo **40 V** pushes (m0,2,3,5,6,7,9,11,14; velocity mostly, one acc). Speeds **1.4–3.7 m/s**;
  `speed_over_cardiac_cycle.png` (in the folder's `speed_measurements/`) shows a **systolic-stiffening
  shape**: rises to a **peak ~3.7 m/s at ~150–250 ms** (elapsed), then settles ~2.3–2.7 m/s.
  **CAVEATS (do not over-read):** (1) the acquisition is **free-running** (`WaitForRpeak=0`), so the
  x-axis is *elapsed time from the first push*, NOT R-peak-locked cardiac phase — the systolic alignment
  is inferred, not gated; (2) the 40 V push is **underpowered** (established: cardiac-motion-dominated),
  so these may be **natural/cardiac waves**, not the ARF push — cross-check each push against its no-push
  control before attributing; (3) absolute speed is **M-line-obliquity-limited**. The *shape* (stiffening
  through systole) is physiologically plausible; the absolute numbers and phase alignment are soft.
- **MI / acoustic-safety literature summary (full doc: `docs/mechanical_index_safety.md`).** No universal voltage→MI table
  exists (papers report MI & I_spta directly — the regulated quantities — because voltage→MI is probe/
  geometry-specific). Anchors: **P4-2, 2 MHz, full 64-el push → MI_0.3 = 1.60/1.11/1.00/0.90 at
  45/60/65/70 mm** (all <1.9, PWM cut to 27% shallower); Caenen P4-2 **MI 2.2 at 50–60 V**; linear-array
  ARFI p_r 1.5–2.65 MPa. **Scaling: MI ∝ V·N/√f (focused), strongly F#-dependent; unfocused ~independent
  of N; pulse length is free on MI (raises thermal/I_spta instead).** **Estimate for our sequence
  (S5-1, 2.25 MHz, 41 el, F/≈4.3, ~45 mm) at 30 V: MI ≈ 0.4–0.8 (central ~0.5–0.6)** — well under 1.9 and
  *low*, consistent with the underpowered-push finding. **Headroom:** 61–79 el at 40–50 V → MI ~1.1–1.8,
  i.e. 2–3× stronger ARF while still legal (thermal/I_spta likely binds first for the 667–800 µs push).
  Verasonics note: set voltage is the bipolar amplitude, so 30 V = 60 V peak-to-peak. No S5-1-specific
  published curve — a one-point **hydrophone calibration** + the scaling gives absolute MI.
- **Hydrophone MEASUREMENTS + calibration fix (2026-08-10) → code in the acquisition repo
  `SWI/Mechanical index/` (patched `CalculateSafety.m` + `README.md`); narrative in
  `docs/mechanical_index_safety.md` §8–10.** We measured our sequences (Onda HGL-0400 + AH-2010-025) and
  first over-read MI by **~2×** (intensities **~4×**). Root cause: the preamp output was read on a
  **high-Z (1 MΩ) scope**, but its 20 dB gain / the Onda combined sensitivity are referenced to **50 Ω**
  (50 Ω source into 1 MΩ ≈ ×2). Fingerprint: preamp clips at 4 Vpp into 50 Ω yet we saw ~4.8 V peak → the
  "saturation" is the **preamp**, and we were high-Z. Also confirmed the **open-circuit** cert requires the
  `C_H/(C_H+C_A)≈0.667` divider → **`ParallelCircuit` method is correct**, `Simple` under-reads 1.5×.
  `CalculateSafety.m` patched: new `ScopeImpedance_Ohm` arg (÷2 @1 MΩ, default 1e6 + warning),
  `MI_method` defaults to `ParallelCircuit`. **Corrected S5-1 push @30 V ≈ 0.69 MI** (past-focus point) —
  matches the §5 literature estimate 0.4–0.8, so the estimate is now measurement-validated; L11-5
  validation gap cut from ~2.5–3× to ~1.5× vs Verasonics. **Rescale old numbers:** Parallel MI ÷2,
  Simple ×0.75, intensities ÷4 (unsaturated points only). `HydrophoneTables.mat` is `*.mat`-gitignored →
  `git add -f`; the Onda `.txt` certs are versioned under `safety/calibration/`.
- **Voltage question (imaging + passive SWE): likely YES for SNR, but MI isn't the gate.** MI headroom is
  huge (imaging/tracking ≈0.5–0.7 vs 1.9), so raising the tracking/imaging voltage is a plausible
  displacement-SNR lever for active tracking *and* passive. BUT the binding limit for continuous high-PRF
  imaging is **I_spta.3 (430 mW/cm² cardiac, ∝ V²·PRF) and probe heating**, not MI — recompute those with
  the corrected code before raising V. Mind preamp saturation when re-measuring (attenuator / low-V
  extrapolation). Detail: `docs/mechanical_index_safety.md` §9.
- **Physics note — symmetry is NOT a cardiac-vs-ARF discriminator.** Cardiac motion CAN look symmetric:
  the r0-anchored **outward directional filter imposes symmetry** on non-propagating/standing/bulk energy
  (splits k_r≈0 energy to both lobes) → a symmetric "V" with no source at r0 (e.g. no-push m8), while a
  **one-directional** no-push (e.g. m11) is a genuine *propagating* physiological wave (valve/wall-motion
  wave) where the filter keeps only the outward lobe. Diagnostic: turn the outward filter OFF on the
  no-push to see the true (un-symmetrised) motion; use push-vs-no-push contrast, not symmetry.

### Expected pushes to capture MVC & AVC at 20 pushes/s (prospective, R-peak-triggered)

Assumes push **#0 at the R-peak**, 20 pushes/s (**50 ms** spacing); MVC ≈ R-peak/S1, AVC ≈ end-systole
(Weissler QS2 ≈ 546 − 2.1·HR ms from Q, ≈ −40 ms to R-peak). **Only valid for R-peak-triggered
acquisition** — the current in-vivo data is free-running, so this is guidance for a *future* gated scan.

| HR (bpm) | cycle (ms) | pushes / cycle | MVC ≈ push | AVC ≈ push |
|---|---|---|---|---|
| 50 | 1200 | 24 | 0–1 | 8 |
| 60 | 1000 | 20 | 0–1 | 8 |
| 70 |  857 | 17 | 0–1 | 7 |
| 80 |  750 | 15 | 0–1 | 7 |
| 90 |  667 | 13 | 0–1 | 6 |

So at a typical ~70 bpm: **MVC ≈ push 0–1, AVC ≈ push 7** (t ≈ 350 ms). Coarse: 50 ms spacing gives
~±25 ms quantisation on AVC — offset the trigger or raise the push rate to bracket the valve events.

### Open TODO (this session)

- **Safety follow-ups (`docs/mechanical_index_safety.md` §10):** (1) confirm the ~2.0 impedance factor
  with a 1 MΩ-vs-50 Ω terminator measurement; (2) finish the corrected MI/I_spta table — needs the
  **L12-3** and **L11-5 widebeam** transmit frequencies; (3) re-anchor the push element/voltage headroom
  (§6) to the corrected *focal* MI from the 30 V map; (4) compute I_spta.3 + probe heating before raising
  the imaging/passive voltage.
- **Push apodization A/B test (phantom).** For the recommended 41→61 element push, compare **uniform vs
  a light Tukey (α ≈ 0.15)** apodization (Verasonics `TX.Apod`). Default is **uniform** (maximise force —
  we are underpowered, not MI-limited); only adopt the light taper if it visibly cleans off-axis/edge
  artifacts or improves r0 localisation. Score with the existing sweep harness: focal displacement +
  wavefront ROI-contrast + mirror-symmetry + r0 localisation; never a full Hann for the push. Rationale +
  numbers: `docs/phantom_parameter_sweep.md` (Recommendation) and `docs/mechanical_index_safety.md`.

## 0b. LATEST (2026-08-07→09): in-vivo diagnosis, Caenen validation, acquisition recommendation, GUI

**The whole arc:** the 40 V in-vivo human data does **not** contain a recoverable ARF shear wave — it is
**acquisition-limited (push-strength-limited)** — while the Caenen pig data does, and the fix is a
**stronger push (larger aperture)**. Key results and where they live:

- **In-vivo 40 V speed-scan sweep** (`scripts/sweep_invivo40.py`, per-lobe: each lobe its own best-fit
  tilt 1–5 m/s since the oblique septum is not mirror-symmetric — user's point). All 24 pushes, 700
  recipes. **No credible wave:** best-fit speeds pile up at the **fast/boundary end (median 4.5 m/s, 32 %
  pinned at 5)** = near-simultaneous bulk wall motion, not propagation; high ROI comes with low symmetry.
  Quietest (diastasis-proxy) pushes: m22,m5,m6,m23. Analyzer: `invivo40_analyze.py`.
- **Caenen pig ARF-SWE, full 52-push sweep DONE** (`scripts/sweep_caenen.py` + `caenen_analyze.py` +
  `caenen_methods.py`; ~40 min/push, ~24 h; supports **resume**). **A clean symmetric V is recovered at
  every push** (52/52 ROI > 0.25; ROI∧symmetry cluster high together) — a real wave, the clean opposite
  of the in-vivo bulk-motion pile-up. **CAVEAT (user):** SWS varies with cardiac phase (52 pushes span
  ~1.8 cycles), so the per-push best-fit speed spread (~1.5–5 m/s, mostly clustered 2–3) is
  SYSTOLIC/DIASTOLIC modulation — **do NOT pool it into one mean SWS**; a systolic/diastolic fit needs
  the ECG timing (Caenen paper's piecewise model). **Method-combination trend (poolable = extraction
  quality) FOLLOWS the phantom/in-vivo family:** band **low-corner 120 Hz decisively wins** (ROI 0.46 vs
  0.28 at 10–40 Hz — the dominant lever, as in passive SWE), narrow **120–350 Hz** band best, **median≈
  gaussian smoothing** (NLM worst), and — for this clear wave — **acceleration** is the best quantity
  (0.37 vs vel 0.30 vs disp 0.18), exactly the SNR-dependent quantity shift (high SNR → derivatives).
  SVD-2 IQ clutter helps mildly. Figures `metric_experiment/caenen_{speed_overview,montage,methods}.png`.
  Data via a cropped Cartesian bridge (`swp_gui/caenen.py`).
- **Acquisition comparison** `docs/caenen_vs_invivo_acquisition.md` (read from both Verasonics structs +
  Caenen Sci Rep 2023;13:17660): in-vivo push is **F/≈4.3 @ 44.6 mm, 40 V** vs Caenen **F/1 @ 25 mm,
  50–60 V, MI 2.2**; tracking 3.7 vs ≥5.6–8.8 kHz; free-running vs R-peak-triggered. We already use
  Loupas (paper's suggested upgrade over Kasai). **Finer beamforming grid does NOT help** (tested) and
  **REFoCUS is inapplicable** (needs multi-transmit; SWE tracking is single diverging-wave na=1).
- **Phantom acquisition parameter sweep** `docs/phantom_parameter_sweep.md` (`scripts/sweep_params.py`):
  11 configs × {20,30} V × 10 pushes. **Aperture (F#) is the dominant lever** (ROI 21→79 el 0.19→0.44 @
  30 V); pulse modest; PRF negligible on the static phantom; combos super-additive at low voltage. MI is
  ~linear in element count (**61 el +48 %, 79 el +90 %**; pulse length is free on MI). **RECOMMENDATION:
  61 push elements + ~800 µs pulse.** Pipeline fixes needed: `make_combined_data.m` no longer hardcodes
  Nframes=2 and auto-selects a per-config **v7.3** base (`PhantomSweep/BaseConfig_10frames_*`).
- **Cardiac-motion-removal harness** (`scripts/motion_removal.py`; no-push control via split reference):
  **reference-subspace projection AND SVD-clutter FABRICATE a false V from pure cardiac motion** (reject
  them); **optical flow (new `optical_flow_compensation`) is the best-behaved remover** (quiets no-push
  without fabricating); IQ-vs-displacement is technique-dependent (SVD→IQ, bulk→displacement). But on the
  acquisition-limited in-vivo data nothing recovers a wave — revisit once the stronger push lands.
- **GUI (`swp_gui/`)** major additions: 3-column layout (data/pipeline/results, scrollable pipeline);
  displacement+velocity+acceleration shown together as a before/current/previous 3-row grid with
  per-column or per-plot colour scaling + colorbars; **Caenen data set** (via `swp_gui/caenen.py`);
  **drag-a-line manual speed** (custom `swp_gui/speedline/` Plotly component over a matplotlib backdrop
  = same RdBu_r colours/clim as the grid with the fixed r0 line baked into the image so it can't be
  dragged; starts **empty** → click-drag to draw a wavefront, **＋ add line** for more (multiple lines),
  drag an endpoint to move one point or the line body to move both; all in-browser, commits on release —
  no per-click rerun; every line's speed listed in Streamlit below the plot (flows, never clipped) + a
  live in-plot readout; fixed iframe height; the heavy quantity grid is now PNG-cached so unrelated
  reruns are instant); **SVD
  singular-value spectrum** plot when tuning svd_clutter cutoffs (limits raised 20→256); per-step
  **enable/disable toggle + ▲▼ reorder**; 'none' steps are true no-ops (don't recompute/relabel);
  do_run only recomputes when the recipe/data actually change (clicks were re-running the pipeline).
  origin_coherence is the ★ metric. Needs `plotly` (added to `[gui]` extras); RESTART the server after
  module edits.

## 0a. (2026-08-06): low-SNR extraction sweep — see `docs/low_snr_extraction_sweep.md`

A 700-recipe × 8-voltage (50→15 V) phantom sweep, scored by two speed-free V-detectors (ROI-contrast on
a locked 50 V template + mirror-symmetry). **The shear-wave V is detectable at every voltage down to
15 V** (ROI-contrast 0.52→0.19, above the 0.05 no-wave floor throughout). **Optimal levers shift with
SNR:** high SNR → median smoothing (~1 mm) + wide band (120–800 Hz) + velocity; low SNR → large-σ
Gaussian (~1.4 mm) + narrow low-freq band (~100–350 Hz) + displacement + SVD clutter. Robust all-SNR
recipe = **id438** (bp 80–500 → spatial median 0.9×2.6 mm → temporal median w5 → 9 offsets → outward).
**RF-NCC underperformed Loupas at every voltage (worse as SNR drops) — not worth expanding.** This
supersedes NEXT-STEP 1 below.

## 0b. method-exploration GUI, metric validation, and the settled recipe

**The in-vivo finding that started this:** the active in-vivo space-time plots are dominated by **cardiac
wall motion, not the ARF shear wave** (peak focal displacement ~42 µm, voltage-independent, ~10× the
phantom push; and the pipeline produces a "clean wave" even from the *pre-push reference* where no push
fired). See the reality-check / no-push-control diagnostics. Higher push voltage does **not** help in
vivo (unlike the phantom, where clarity scales cleanly with voltage).

**Interactive GUI** (`swp_gui/`, Streamlit): `KERAS_BACKEND=torch <zea-py> -m streamlit run swp_gui/app.py`.
Tune every stage IQ/RF→space-time (IQ pre-filters, estimator incl. fine-grid RF-NCC, cardiac-motion
removal, spatial/temporal, M-line averaging, directional, speed), each with a code viewer, a B-mode +
**spline** M-line + offset-line overlay, read-only acquisition constants, add/remove method chains,
per-step/global reset, a **no-push control** (same recipe on the reference — the key in-vivo diagnostic),
comparison history, and an "animate all pushes" GIF. Boots in ~4 s (zea/torch imported lazily only for
the fine grid). New reusable algorithms in `src/swp/viz/`: `filters/experimental.py`
(iq low/high-pass, SVD-on-displacement, reference-subspace projection, phase-unwrap, bulk-motion comp,
quality-mask, savgol, bilateral/NLM/anisotropic/coherence diffusion), `estimators/rf_ncc.py`,
`acquisition/finegrid.py` (fine-grid RF re-beamform around the M-line). Loupas gained kernel_x +
adaptive-kernel (RF-coherence-sized axial window). Directional filter was **fixed** (Tukey-windowed FFT +
smooth angular mask): +R/−R now null the opposite lobe (~80%); outward keeps both.

**Metric validation (human-in-the-loop, blind).** A big experiment (`scripts/metric_experiment_*.py`,
`swp_gui/score_app.py` absolute 1-5, `swp_gui/pairwise_app.py` 2-AFC, per-dataset scoring) graded ~600
randomized/focused recipes across phantom 15/25/30/50 V + in-vivo 30/40 V. Findings:
- **`push_specificity` was DEBUNKED** (worse than the metric it replaced; false pos/neg). **Adopt
  `origin_coherence` (mean over quantities) as THE metric** — it's the best-validated (Spearman ρ≈0.45
  across datasets, up to 0.60 on clean data), generalizes to in-vivo for *ranking recipes*, and has no
  blow-ups. Now the ★ headline metric in the GUI. Hand-crafted smoothness/symmetry additions did not
  robustly beat it. Ceiling ~0.5 is honest (discrete/near-tie human scores).
- **`origin_coherence` did NOT break in-vivo** for ranking (ρ≈0.5) — the cardiac-fooling is specific to
  the *no-push reference* window, not to ranking recipes on the tracking window.
- **All GUI options verified to affect the output** (`scripts/check_options.py`); no no-ops.

**The settled recipe — and it is SNR-ADAPTIVE (validated on phantom 30 V vs 50 V pairwise):**
- **Universal:** Loupas · `relative_to_reference` · temporal band-pass · **OUTWARD directional (decisive:
  ΔBT +4 to +9; never left/right for active)** · moving-mean temporal. Frame-to-frame, xcorr, IQ SVD /
  slow-time-highpass, bilateral, axial-strain, reference-subspace all HURT.
- **Smoothing scales inversely with SNR (clean sign-flip):** high SNR (50 V) → **light** smoothing
  (median or light Gaussian; Gaussian-σ↔score ρ=−0.18; NLM worst) + **more offsets (9)**. Low SNR (30 V)
  → **more** smoothing (larger-σ Gaussian, NLM acceptable; Gaussian-σ↔score ρ=+0.34; median worse);
  offsets less important. → recommend **SNR-adaptive σ** (ties into the Loupas adaptive-kernel), not a
  single fixed value. Figure: `metric_experiment/snr_conclusion.png`.
- **The readable quantity also shifts with SNR** (derivatives sharpen at high SNR, amplify noise at low):
  high SNR → **acceleration/velocity** (oc↔ranking: acc 0.56, vel 0.38, disp 0.10); low SNR →
  **displacement/velocity** (disp 0.56, vel 0.59, acc 0.40). **Velocity is the single most SNR-robust
  quantity.** Implication: the metric's quantity-aggregation should be SNR-aware (or default to mean/
  velocity); a fixed "best column" is wrong.

**NEXT STEPS (priority order):**
1. **Extract the shear wave in low-SNR data (phantom 15 V).** This is the honest hard case: at 15 V the
   wave is barely above noise. Use the SNR-adaptive recipe (heavy smoothing, read displacement/velocity),
   and explore the fine-grid RF-NCC estimator and stronger clutter/denoise. Success criterion: a
   confident sloped wavefront + a stable speed. Data: `2026_08_04 voltage sweep/Phantom/...13-30-35`.
2. **Cardiac-motion removal for in-vivo (currently NOT working).** The band-pass-after-displacement
   approach fails (large motion wraps the phase; cardiac energy leaks into the band). Try, at/before the
   estimator: bulk-motion compensation (global + affine/optical-flow), reference-subspace projection
   *trained on the 40 pre-push frames*, and SVD clutter — validated with the **no-push control** (a good
   filter makes the reference window go quiet while the tracking window keeps the wave). **User caveat
   (important): cardiac motion and shear-wave propagation are hard to distinguish in a lone space-time
   image without further context** — so removal (physics-based, using the reference) is the real lever,
   not a better image metric, and validation needs the no-push contrast (and ideally ECG phase / a
   simulated ground truth).
   **STATUS (2026-08-07): harness built + first evaluation done (`scripts/motion_removal.py`).** No-push
   control = SPLIT reference (train filter on 1st half of the reference, apply to 2nd half = no push);
   scored by the per-lobe speed-scan + symmetry; pushes ranked by pre-push wall-motion RMS (data-driven
   diastasis proxy — ECG in the .mat is not usable, WaitForRpeak=0). **KEY FINDINGS:** (a) **reference-
   subspace projection AND SVD-clutter FABRICATE a false shear-wave "V" from pure cardiac motion** (high
   no-push ROI 0.13-0.16 on the quietest push m22) — the no-push control catches what push_specificity
   missed; REJECT these for cardiac SWE. (b) **No method beats plain band-pass** (baseline track ROI 0.232
   / no-push 0.067 / contrast 0.164); the only honest motion-remover (`iq_slowtime_highpass`: no-push
   0.067->0.032) also suppresses the tracking signal -> the tracking "signal" IS cardiac motion, not a
   hidden push wave. (c) Confirms acquisition-limited: processing cannot manufacture the wave (and forcing
   it fabricates artifacts). The harness is the trustworthy tool; revisit filters (light honest remover +
   physiological-speed gate) once the stronger-push acquisition (item 3) clears the cardiac floor. Quietest
   pushes: m22, m5, m6, m23. Figure: `metric_experiment/motion_removal_montage.png`.
   **IQ-domain vs DISPLACEMENT-domain filtering (2026-08-07, `motion_removal.py` + 2 new registered methods
   `optical_flow_compensation` (IQ, non-rigid) + `bulk_displacement_removal` (disp)).** Numbers over the 4
   quietest pushes (want LOW no-push ROI = not fabricating): it is TECHNIQUE-dependent, not a blanket rule.
   **SVD clutter -> IQ domain** (no-push 0.097 vs disp 0.140 which FABRICATES a false V; matches Demene/lit).
   **Bulk motion -> displacement domain** (contrast 0.148 vs IQ 0.090; the IQ Fourier-shift distorts speed to
   1.2 m/s). **Slow-time high-pass -> IQ removes the most clutter (no-push 0.004) but takes the wave with it**
   (track 0.053) = wave & cardiac share the band. **Optical flow (IQ, NEW) = best-behaved aggressive remover:
   quiets the no-push (0.064) WITHOUT fabricating (unlike SVD-disp/refsub-disp) + keeps physiological 2.2 m/s
   -> carry this forward.** Still, nothing beats plain band-pass for contrast here, and every honest remover
   suppresses the tracking signal too -> acquisition-limited confirmed. Figure:
   `metric_experiment/motion_removal_iqvsdisp.png`.
3. **DONE (2026-08-07): Acquisition-optimization phantom sweep → `docs/phantom_parameter_sweep.md`.**
   11 configs × {20,30} V × 10 pushes, OAT over pulse/aperture/PRF. **Aperture (F#) is the dominant
   lever** (ROI 21→79 el: 0.19→0.44 @ 30 V, monotonic); pulse modest; **PRF negligible on the static
   phantom**; combos super-additive at low voltage (79 el+1900 c ~doubles 20 V base). Focal-displacement
   is an unreliable push proxy (strong-push decorrelation → underestimate). MI (Rayleigh coherent-sum) is
   ~linear in element count: **61 el = +48 %, 79 el = +90 %** vs 41; **pulse length is free on MI**.
   **RECOMMENDATION: 61 push elements + ~800 µs pulse** (most clarity-per-MI; 79 el buys +0.02 ROI for
   nearly double the MI). Pipeline fixes: `make_combined_data.m` no longer hardcodes Nframes=2 and
   auto-selects a per-config **v7.3** base config (PhantomSweep/BaseConfig_10frames_<cyc>c_<el>el_<pri>PRI).
   Analysis: `scripts/sweep_params.py`.
3b. **(superseded original next-step) Acquisition-optimization phantom measurements.** Motivated by the
   Caenen-vs-our-in-vivo acquisition comparison (`docs/caenen_vs_invivo_acquisition.md`): the in-vivo gap
   is acquisition-limited (weak/loose-F# push, low PRF) more than processing-limited. Re-acquire phantom
   data varying two levers and quantify the effect the same way as the voltage sweep (peak focal
   displacement + wavefront ROI-contrast/symmetry/speed vs the setting):
   - **(a) Push aperture: increasing number of push transmit elements (e.g. 40 → 80).** Lowers the push
     F-number (~4.3 → ~2.2 at the septal depth) → tighter focus, stronger ARF. Watch MI (rises with focal
     pressure) and off-axis steering quality. Success: measurably larger focal displacement / clearer wave.
   - **(b) Increased tracking PRF.** Currently 3704 Hz (chosen so the 15 cm B-mode range is unambiguous);
     a dedicated shallow high-PRF *tracking* sub-sequence (≥5.6 kHz, wall-only depth) samples the wave far
     better — the risk to check on the phantom is deep-echo range-wrap folding onto the wall.
   - Note (settled here, do NOT re-try): **finer beamforming grid does NOT help** (phantom test: ROI flat
     0.30→0.29 from dz 400→99 µm; displacement is slow-time phase, not axial-resolution limited), and
     **zea REFoCUS is inapplicable** (needs multi-transmit encoding; SWE tracking is single diverging-wave,
     na=1). Within the MI cap, **longer push duration** (impulse ∝ duration at fixed peak pressure) is a
     cheaper strength lever than voltage.
4. Optional: finer recipe sweeps; real speed/stiffness extraction with the settled recipe; generalize to
   more subjects/voltages.

Experiment artifacts live in `2026_08_04 voltage sweep/metric_experiment/<dataset>/` (manifest.json,
scores.csv, pairs.csv, plots/, per-round figures). Scripts: `metric_experiment_{generate,handpicked,
analyze,patterns,crossdataset}.py`, `metric_build.py`, `pairwise_analyze.py`, `check_options.py`.

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
