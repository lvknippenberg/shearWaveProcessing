# Review follow-up, 2026-09-24

The literature review of 2026-09-24 raised six points about the processing. This records what
was done about each and what the data said. Everything ran on existing data; branch
`review-followup`. Open actions for Luuk are in [TODO.md](TODO.md).

**Short version.** None of the six changes rescues the in-vivo ARF wave. Every check tested the
recipe against a known answer, and the verdict "acquisition-limited" now holds up against the
standard literature recipe, a positive control, 1552 study pushes and a timing fix. Along the way
four real problems turned up and are now fixed or documented:
- the reference timestamps ignored the push;
- reference and tracking frames decorrelate across the push in vivo;
- the automatic active speed is biased by the directional filter and the envelope slant stack;
- the repo had no tests and no provenance on its outputs.

## 1. The literature recipe against the no-push control

`scripts/invivo_recipe_contrast.py` -> `study/logs/invivo_recipe_contrast.{csv,log}`,
`study/montages/invivo_recipe_contrast{,_montage}.png`.

Every earlier in-vivo evaluation used displacement **relative to the averaged reference**. The
cardiac literature uses frame-to-frame particle velocity. The reference-relative phase wraps at
+/-98.6 um, and in-vivo wall motion reaches 40-90 um, so the review asked whether that choice was
hiding the wave.

Each push was scored against its own no-push control (the split pre-push reference). The two are
**equal in length and cut before any temporal filtering**. Four recipes were compared:
- `current`: the settled recipe;
- `f2f_current`: the same filters on frame-to-frame velocity;
- `caenen`: Caenen et al. 2023 exactly - lag-1 velocity, Gaussian 1.9 x 2.0 mm on the
  autocorrelation before the angle (new `kernel_shape="gaussian"` in Loupas), 6th-order
  75-750 Hz band-pass;
- `caenen_dir`: `caenen` plus the outward directional filter.

| dataset | current: push/control RMS | caenen_dir: push/control RMS | caenen_dir: pushes above control |
|---|---|---|---|
| Caenen pig (positive control) | 1.47x | **11.7x** | **52/52** (p = 3e-10) |
| CIRS phantom (positive control) | 2.43x | 1.44x | 75 % |
| in vivo 30 V, 40 V (2026-08-04) | 0.61x, 0.54x | 0.91x, 0.90x | 38 %, 46 % |
| in vivo 41 el, 61 el (2026-08-18, R-peak) | 0.67x, 0.78x | 0.86x, 1.01x | 29 %, 50 % |

What this shows:
- The literature recipe finds a real wave far better than the current recipe (8x on Caenen's data).
- No recipe finds one in any of our four acquisitions.
- No cardiac phase stands out in the R-peak-gated sets. The one high push (61 el, 50 ms after R,
  3.1x) sits at mitral valve closure, so it is most likely the natural wave.

## 2. Reference timing, continuous record, Giannantonio filter

### Timestamps

`sequence.assemble_tracking_frames` placed the last reference frame one frame before tracking, as
if no push happened. From `SetUp_SWI_Widebeam.m`, the real interval from the last reference
transmit to the first tracking transmit is PRI plus the push burst rounded up to 100 us
(SeqControl 11 + 10). That is **0.97 ms at 1500 cycles and 1.17 ms at 1900 cycles**, against the
0.27 ms that was stored. The old sliding-pair timing was also one frame off.

What changed:
- `SWGeometry.push_gap_s()` computes the interval.
- New beamforms store the corrected `t_reference` plus `custom/push_gap_s`.
- `scripts/retrofit_push_gap.py` corrects old files in place and keeps the original as
  `custom/t_reference_v0`. It was applied to the four in-vivo folders under D:, not yet to the
  Z: study (TODO 13).
- A phantom re-beamform gives bit-identical IQ.

### A new finding: in vivo, the blocks decorrelate across the push

`study/analysis/push_gap_check.py` -> `study/logs/push_gap_check.csv`.

I tried to confirm the interval from the data: wall displacement across the push should equal
wall velocity times elapsed time. The same fit *inside* the reference block returns 1500-1590 us
for a true 1620 us, so the method works. **Across the push it does not**: the fitted time does
not grow with frame number (slope ~0 instead of 270 us per frame). Speckle correlation shows why:

| | within a block, 6 frames (1.6 ms) | last reference -> tracking frame 2 |
|---|---|---|
| phantom 50 V / 15 V | 1.000 / 0.984 | 1.000 / 0.984 |
| in vivo 40 V / 61 el | 0.957 / 0.927 | 0.805 / 0.814 |

The drop is in vivo only. It gets *worse* away from the push (0.85 -> 0.68 at 15-25 mm), and the
data behave as if ~3-4 ms had passed where the sequence programs ~1 ms. The cause is open
(TODO 2: a zero-amplitude-push acquisition decides it). Consequences:
- The **corrected timing rests on the sequence definition**; the data could not confirm it.
- Anything that compares tracking frames with reference frames in vivo starts from 0.8
  correlation, not 0.95. That includes reference-relative displacement and every reference-based
  motion model. Frame-to-frame estimation does not cross the push and is unaffected.

### Continuous record and Giannantonio filter

New, off by default:
- `swp.viz.slowtime.continuous_record` joins reference and tracking on one uniform grid, so
  temporal filters see ~27 ms instead of 16 ms (`PipelineConfig.continuous_record`).
- The `giannantonio_motion_filter` field filter is the published motion filter: fitted on
  pre-push **and** late post-wave samples. That is interpolation; the old `reference_motion_comp`
  only extrapolates.

Tested with the same harness, recipes `caenen_cont` and `caenen_cont_gian`:
- A first run showed a push effect in vivo (41 el: 1.28x, p = 0.001). It was an **artifact**. The
  displacement step that spans the push is decorrelated (the finding above), and band-passed it
  leaks a spatially broad transient into the first 4 ms. A control whose join spans an equally
  decorrelated 3.2 ms removes it (1.00x / 0.89x). The phantom's real effect survives (2.08x). The
  harness now uses this join-matched control.
- With the fair control: in vivo 0.81-1.23x, nothing significant. On Caenen's data it *hurts*
  detection (1.14x, against 11.7x without it). The phantom improves (2.08x vs 1.44x).
- **Verdict:** implemented and tested, but not recommended for in-vivo detection.

## 3. The 44-subject study, screened

`scripts/study_active_screen.py`, `study/analysis/study_active_screen_summary.py`,
`study/analysis/study_active_candidates.py` -> `study/logs/study_active_screen.csv`,
`study/montages/study_active_screen.png`, `study/montages/study_active_candidates.png`.

**Setup.** All 71 processed acquisitions: 1552 pushes, 44 subjects, all at 41 el / 1500 cyc /
30 V. Recipe `caenen_dir` against the no-push control. The septal M-line is proposed
automatically (`auto_mline.propose`, kept in memory, nothing written to Z:). A wrong line lowers
push and control alike, so the screen can miss a wave but not invent one.

**Result.**
- Push/control median **0.89x**; above 1 in only 34 % of pushes. Every acquisition's median lies
  between 0.7x and 1.1x. Caenen's pushes have a median of 11.7x, and their weakest 5 % reach 3.3x.
- No cardiac phase lifts the median above 1.
- **7 pushes (0.5 %, in 7 subjects)** exceed the weakest 5 % of Caenen's real waves. 6 of them
  fall 600-950 ms after R (late diastole). By eye, two show an outward tilted pattern from r0
  (C000000026 m18, C000000024 m14); the rest look like artifacts (TODO 6).

## 4. Passive speed: normalised Radon and speed by quantity

### Normalised Radon

New: `metrics.normalized_radon_speed` (Vos 2017 / Keijzer: mean signal along the line) and
`metrics.line_tracking`. Benchmarked on the 30 hand-drawn panels (`study/analysis/radon_benchmark.py`):

| on the 13 "clear" panels | within 25 % of hand | railed at a bound |
|---|---|---|
| slant stack (reported) | **62 %** | 8 % |
| normalised Radon, 1-20 m/s | 38 % | 15 % |
| normalised Radon, 1-8 m/s | 38 % | 38 % |

- The normalised-Radon line does sit on stronger signal than the hand line (tracking score 2.40
  vs 2.24).
- But the strongest line is often the near-synchronous bulk band, which reads as a fast, railed
  speed.
- **The amplitude-along-the-line objective, recommended in `passive_speed_estimation.md`, does
  not beat the slant stack.** It stays in the library as a documented negative result; the slant
  stack remains the reported estimator.

### Speed by quantity

- Every (re)processed passive folder now writes `passive_speeds_by_quantity.json`: view A re-run
  as displacement, velocity and acceleration, same slant stack, stamped.
- The literature reports velocity (Keijzer) or acceleration (Petrescu, Santos, Espeland), never
  cumulative displacement.
- On the 15 labelled windows (`study/analysis/quantity_speeds.py`): see
  `study/logs/quantity_speeds.log` (numbers below).

On the 15 labelled windows, 12 have no railed fit in any quantity:

| | displacement | velocity | acceleration |
|---|---|---|---|
| automatic (view A recipe, quantity swapped), median | 3.80 | 3.96 | 2.91 m/s |
| hand-drawn, median over 15 windows | 2.60 | 3.92 | - |
| automatic / hand, median; share within 25 % | 1.32; 42 % | **1.04; 50 %** | - |

- **Automatic velocity agrees best with the hand values.** The hand velocity panels come from view
  C (15-90 Hz) and the automatic ones from view A's recipe (10-150 Hz), so the two are not
  identical recipes.
- By hand, velocity comes out faster than displacement in 87 % of windows.
- The automatic fits do **not** reproduce the documented ordering displacement < velocity <
  acceleration: it holds in only 33 % of windows, and acceleration is the *slowest* by median. At
  ~925 Hz frame rate and these line lengths the acceleration panels are too noisy for the slant
  stack. For comparison with the literature, report **velocity** (Keijzer's convention); do not
  quote an automatic acceleration speed.

## 5. Clutter filter wave imaging (CFWI)

New estimator `estimator: cfwi` (`estimators/cfwi.py`, Salles 2019 / Espeland 2024): slow-time
IQ high-pass at 2 cm/s, then envelope, then temporal derivative. Run on the 15 labelled windows
(`study/analysis/cfwi_benchmark.py`, `study/montages/cfwi_benchmark.png`).

- The filtered record has an end transient; 10 frames are dropped at each window end.
- On the 7 clear panels, CFWI lands within 25 % of the hand displacement speed in 4 (57 %),
  against 62 % for the slant stack on displacement.
- But the **direction agrees only 50-57 %** - chance level.
- It shows the valve events but no sharper than velocity. Not added as a default view.

## 6. Blind panel scoring

`study/analysis/score_panels.py` is now blind by default:
- no speeds, hand line, subject or event label; seeded random order;
- scores go to `panel_confidence_blind.csv`;
- `--compare` reports agreement with the earlier non-blind scores and re-checks the
  "estimators work where a wave is visible" trend.

Scoring is TODO 5.

## Caenen's own speeds on the pig data

`study/analysis/caenen_speed_comparison.py` -> `study/logs/caenen_reported_speeds.csv` (digitised
from the plot Caenen sent), `study/logs/caenen_speed_comparison.csv`,
`study/montages/caenen_speed_comparison.png`.

- Caenen could measure 11 of 52 pushes; their per-push medians are 1.2-2.1 m/s, plus push 5 at
  5.1 m/s.
- Our stored per-side speeds (120-700 Hz displacement, one-sided slant stack) match in level:
  **median ratio 0.93 (left) and 0.97 (right), within 25 % for 80 % / 70 % of pushes**.
- Push 5's right side gives 4.5 m/s against their 5.1 m/s.
- Across pushes the correlation is weak (r = 0.15), but their range is narrow.
- Running their own recipe through our automatic slant stack **fails**: it rails at whatever
  lower bound is set (0.5-0.8 m/s). The same failure made them measure by hand. A like-for-like
  check needs our hand slopes on the same pushes (TODO 7).

## Tests, provenance, paths

- `tests/` (new; `python -m pytest`, ~30 s, synthetic): 40 pass, 4 **strict xfails** that record
  known biases:
  - the outward directional filter biases speed high (+5 to +37 %);
  - `radon` is +20 to +94 % high;
  - a 75-750 Hz band-pass on a 16 ms record makes automatic speeds erratic;
  - the directional filter's Tukey taper removes ~45 % of a clean outward wave's energy.
  - Estimators alone are exact (`tof_xcorr` 1.00x at 2-4 m/s).
- `swp.provenance`: every HDF5 the pipeline writes (converted RF, beamformed IQ incl. after the
  scan-parameter rebuild, viz space-time) carries `/provenance`, and new CSVs/JSONs a header or
  key. Contents: repo commit / branch / dirty list + diff hash, zea version + commit (read
  without importing zea), config hash + JSON, host, versions, command. Verified that zea still
  reads stamped files and that re-beamformed IQ is unchanged.
- `swp.paths`: data locations with environment overrides (`SWP_DATA_ROOT` etc.).
  - 28 scripts hard-coded `D:/...`, and the voltage-sweep path they used no longer existed.
  - 42 finished campaign scripts moved to `scripts/archive/` (see its README); every script
    import-tested after the move.

## 7. Passive default adopted, buffer-3 line mapping, study rerun (evening)

**Default.** `configs/passive.yaml` now uses the report part-2 recipe: velocity, 15-150 Hz,
Gaussian 0.6 x 1.2 mm, mean 3, 5 lines x 0.5 mm; no directional filter, SVD or CFWI. The three views
differ only in the spatial filter (default / unsmoothed / median 1.0 x 2.0 mm). `manual_slope.py`
shows them side by side with the line mirrored. The old config is frozen as `passive_v1.yaml`.
Temporal smoothing and M-line count do act (`smoothing_effect_check.py`); they change little
because the signal is slow (f50 29 Hz, f90 66 Hz).

**Buffer-3 -> buffer-4 line mapping** (`swp.mline.transfer`, `map_mlines_b3_to_b4.py`):
- The line is moved by the local translation between the frame it was drawn on and buffer 4 at
  the event. Rotation is not identifiable in vivo, so it is not modelled.
- A mapping is trusted only if the ensemble agrees and known shifts are recovered.
- Phantom: <= 0.3 mm error. In vivo: 7/15 mappings reliable.
- Mapped lines beat the unmapped buffer-3 lines (6/7) but not the buffer-1 lines (2/7).
  Keep the buffer-1 lines.

**Study rerun** (`passive_study.py reprocess`, 36 folders, 116 windows, same windows and lines;
v1 outputs in `swp_passive/v1_displacement/`; `passive_v2_rerun_summary.py` ->
`study/logs/passive_v2_rerun_windows.csv`, `study/montages/passive_v2_rerun_summary.png`):
- **Population medians unchanged.** Median automatic |c| per event, v2 default vs v1 view A: MVC 4.4 vs
  4.3, AVC 4.0 vs 4.0, AK 1.8 vs 1.8, other 2.1 vs 2.1 m/s. Railing at the 1/20 m/s bounds drops
  from 16 % to 11 %.
- **Per window the automatic speed is recipe-dependent.** Where neither run rails, v2 and v1 A agree
  within 25 % in only 32 % of windows (same direction in 86 %). The automatic number is not a
  per-window measurement; hand slopes stay the reference.
- **The Gaussian 0.6 x 1.2 mm does steepen some fronts.** Against the unsmoothed view it gives the
  same speed in 45 % of windows, >10 % faster in 36 % (median +30 % there), slower in 4 %. The
  median 1.0 x 2.0 mm filter is nearly neutral (IQR of the ratio 1.00-1.05), and its semblance
  (0.48) is between unsmoothed (0.34) and Gaussian (0.58). This supports the concern raised on
  report part 2. When reading a speed by hand, check it against the median view.
- View agreement rises from 5/116 (v1) to 91/116 (v2), but that is by design: the v2 views
  differ only in the spatial filter. It is not a quality gain.
- Only 4 windows have full-line hand slopes, too few for an accuracy comparison.

**Re-detection accident, repaired.**
- The study windows were detected in energy mode before `detect_mode` entered the cache key.
  `process_single_line` therefore re-detected (in phase mode) C000000001-4 during the first rerun
  attempt, and C000000023 during a single-folder reprocess that morning. In those folders the
  hand-drawn per-event lines were archived.
- `scripts/restore_redetected_windows.py` restored all five folders. The original windows were
  reproduced to 0.1 ms and checked against the 09-18 split table or the v1 record. The re-detected
  state is kept as `*.redetected_20260924`. C000000023's v1 outputs were regenerated on the
  restored lines.
- A scan of all 44 folders shows no other re-detected window files.
- Prevention:
  - `process_single_line` raises `StaleWindowsError` when hand-drawn lines depend on the cache;
  - `reprocess` no longer detects;
  - the detection overview quantity is pinned (`detect.quantity: displacement`).
