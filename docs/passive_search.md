# Passive SWE — exhaustive parameter search

**Status (2026-08-03):** two passes done — (1) an automatic exhaustive filter/parameter search scored
by `passive_coherence`, then (2) a clinical-domain-review correction that reinterpreted the results
(valve labelling, fixed propagation direction, physiological speed). **Read the "Domain-review
follow-up" and "Proposed next steps" sections for the current understanding — they supersede parts of
the automatic findings below (flagged inline as ⚠︎REVISED).** The automatic search remains valid for
the *filter/parameter* conclusions (band, quantity, smoothing); its *direction* and *speed* readouts
were artefacts of the metric and are corrected in pass 2.

Mirror of the active-SWE optimisation (iq2sws `archive/search.py`), for the natural
(valve-closure) shear waves in the buffer-4 ultrafast stream. Tooling:

- `scripts/search_passive.py` — sweeps **6 720 combos/window** and scores each by
  `metrics.passive_coherence` (one-sided origin coherence in [0, 1]; higher = a clearer
  propagating wavefront). Also reports the best-fit shear-wave speed `c`.
- `scripts/passive_best_montage.py` — renders the two summary montages from the saved CSVs.

Run on `D:\Luuk van Knippenberg\Claude\invivo_sw` (in-vivo PLAX, 1112 frames @ ~925.9 Hz), which
has 4 detected valve-closure windows (@ 52, 377, 549, 920 ms), a fresh M-line drawn per window.

## Swept axes (per window, per burst window)

| axis | values |
|---|---|
| quantity | displacement, velocity  *(acceleration dropped — never competitive)* |
| motion / band | none · hp{10,20,30} · poly{1,2,3} · bp{5-100,5-150,10-150,20-200,30-250,40-300} · svd{1,2} |
| spatial | none · gauss{0.3,0.6,1.0,1.5} mm · median{0.6,1.2} mm |
| temporal | none · mean3 · mean5 · median3 |
| direction | leftward, rightward *(also fixes the origin end r0)* |
| M-line sampling | offsets{5,9} × step{0.5,1.0} mm |

Efficiency: the `frame_to_frame` Loupas estimate is cached per SVD rank; the (linear, separable)
temporal motion/smoothing filters and the directional filter are applied on the **sampled
space-time** (exact for the Gaussian-spatial branch, which commutes with linear temporal filters
and the linear M-line interpolation; median-spatial is a close approximation). `passive_coherence`
is vectorised over candidate speeds. ~10 min/window (dominated by the full-field median-spatial
options; drop those two for ~2 min/window).

## Headline findings

1. **Removing the bulk myocardial wall motion is the decisive lever.** A temporal high-pass whose
   **low corner sits at ≥ 10–20 Hz** is required. The old starting recipe's 5 Hz corner
   (`bp5_150`) passed the low-frequency wall motion, which dominates the space-time as a flat,
   non-propagating band → **`passive_coherence ≈ 0.00` on every window**. Raising the corner to
   10–20 Hz (`hp20`, `bp10_150`, `bp20_200`) reveals the propagating shear wave →
   **pc up to 0.94**. This single change is a 0.00 → 0.94 jump.
   - **SVD clutter and polynomial detrend do *not* substitute** for the high-pass (median pc ≈ 0
     for `svd1/2`, `poly1/2/3`, `none` across all combos). Only a ≥10 Hz temporal high-/band-pass works.

2. ⚠︎REVISED (artefact). *The search reported the propagation direction as per-window (win0/win3
   rightward, win1/win2 leftward).* The **domain review corrected this**: in PLAX both MVC and AVC
   waves travel **right→left (basal→apical)**, so the direction is **fixed (leftward)**. The apparent
   per-window flip was the envelope `passive_coherence` locking onto the spatially-uniform bulk-motion
   band from whichever end maximised the score — not a real change of wave direction.

3. **Quantity:** *displacement* + `hp20` + `mean5` gives the **highest, cleanest** single-window
   scores (pc 0.94 on win0 & win3) but collapses on the weaker windows; *velocity* + `bp10_150` +
   `mean3` is the **most robust across windows** (direction-agnostic mean pc 0.61, no collapse).
   → config default = velocity; displacement/hp20 documented as the high-contrast alternative.

4. **Secondary optima (all windows agree):** light **Gaussian** spatial smoothing (0.6–1.0 mm;
   median never wins), light **temporal** smoothing (mean3–mean5), and **tight M-line offset
   averaging** (5 lines × 0.5 mm; more/wider offsets slightly worse).

5. **win1 @ 377 ms is the weak window** (best pc 0.59): its space-time is a near-horizontal band,
   not a clean valve-closure shear wave — likely low SNR / M-line placement / not a true burst.
   win0 (52 ms), win2 (549 ms), win3 (920 ms) all show clear propagation.

6. ⚠︎REVISED (metric artefact). *The search's best-fit `c` was 5.8–8.0 m/s, pinned near `cmax`.*
   This was **not a real wave speed** — the envelope metric was slant-stacking the near-flat,
   spatially-uniform **bulk-motion band** (apparent speed → ∞). The domain review + the corrected
   `metrics.slant_stack_speed` (which removes the uniform band first) put the real wavefront at
   **~1–2 m/s** (see the follow-up section). **A high `passive_coherence` does not certify a
   tissue-like speed.**

## Best recipe per window (own optimum)

| window | pc | c (m/s) | recipe |
|---|---|---|---|
| win0 @ 52 ms  | **0.941** | 6.5 | disp / hp20 / gauss0.6 / mean5 / **rightward** / off5·0.5 mm |
| win1 @ 377 ms | 0.591 | 8.0 | velo / poly1 / gauss0.3 / mean3 / leftward / off5·0.5 mm |
| win2 @ 549 ms | 0.702 | 5.8 | velo / bp10_150 / gauss1.0 / mean3 / **leftward** / off5·0.5 mm |
| win3 @ 920 ms | **0.940** | 6.3 | disp / hp20 / gauss0.6 / mean5 / **rightward** / off5·0.5 mm |

## Consensus (single recipe, direction chosen per window)

`velocity / bp10_150 / gauss1.0 / mean3 / off5·0.5 mm` → per-window pc **0.86, 0.00, 0.70, 0.89**
(recovers 3/4 windows; win1 is intrinsically weak). This is `configs/passive.yaml`.

## Outputs (`<folder>/output/swp_passive/search/`)

`results_win{i}.csv` (all 6 720 combos), `marginals_win{i}.png`, `top_win{i}.png` (top-8 montage),
`per_window_best.png`, `consensus_perdir.png`, `baseline_montage.png` (the old 5 Hz recipe → flat
bands, pc≈0), `leaderboard.md` (full per-window + consensus tables).

## Domain-review follow-up (MVC vs AVC, fixed direction, physiological speed)

Per clinical review: **win1 = noise (ignore)**; **win0/win3 = mitral-valve-closure (MVC)**, **win2 =
aortic-valve-closure (AVC)**. In PLAX both waves travel **right→left (basal→apical)** so the
directional filter is **fixed (leftward)** — the earlier per-window direction flip was the envelope
metric chasing the flat band. Expected SWS ~2–3 m/s; the pc~0.9 / c~6–8 m/s "fits" were the metric
locking onto the **spatially-uniform bulk-motion band**, not a real wave.

Tools added:
- `scripts/passive_filter_variety.py` — quantity×band grid for one window, direction fixed leftward,
  with 2 & 3 m/s reference slopes and pc restricted to a physiological band.
- `metrics.slant_stack_speed` — **signed tau-p slant-stack** with **per-time spatial-mean removal**
  (kills the uniform bulk band, the decisive fix) → semblance-based speed + travel direction.
- `scripts/passive_slantstack.py` — refined **low-band** grid (disp/velo × bp{5-80..20-120}) in the
  **reference M-mode orientation** (x=time, y=along-line), overlaying the band slant-stack fit, an
  optional leading-edge (Theil–Sen) fit, and 2 & 3 m/s references. Outputs `slantstack_win{0,2}_*.png`,
  `filtervariety_win{0,2}_*.png`.

Findings:
- **A coherent propagating wavefront is visible in BOTH AVC (win2) and MVC (win0)**, clearest in
  **velocity** at a **low band (5–150 / 10–150 Hz)** (displacement is smoother; ≥20 Hz low-corner or
  high-cut <100 Hz degrade it). Direction is consistent basal→apical, as expected.
- **Speed:** after removing the uniform bulk band, the whole-band slant-stack gives **~1.0–1.2 m/s**
  along this M-line (AVC and MVC similar); the few stable leading-edge fits gave ~1.2–1.9 m/s. This is
  **slower than the 2–3 m/s expected / the MATLAB reference's 2.12 m/s (displacement)**. The gap is
  because (a) the exact number depends strongly on *which* feature is fitted (leading edge ≫ band
  centre) — the user's own MATLAB line fits are admittedly rough — and (b) apparent speed depends on
  M-line alignment with the true propagation path. **Automated leading-edge speed is unstable on this
  data; a manual line fit on the reference-oriented plot (guided by the 2 & 3 m/s references) remains
  the reliable read**, matching the MATLAB workflow.
- **Metric lesson:** neither the envelope `passive_coherence` nor a naive signed semblance is safe for
  passive speed — both are maximised by the flat bulk band. Removing the per-time spatial mean before
  any moveout fit is essential.

## Current best understanding (summary)

- **Windows:** win1 = noise (ignore); **win0/win3 = MVC**, **win2 = AVC**.
- **Direction:** fixed **right→left (basal→apical)** = `directional_mode: leftward`.
- **Recipe that reveals the wave:** **velocity**, **low band-pass 5–150 / 10–150 Hz**, light Gaussian
  (0.6–1.0 mm), light temporal smoothing, tight M-line averaging (5 × 0.5 mm). Displacement is
  smoother/cleaner but lower-contrast; a ≥20 Hz low corner or a <100 Hz high cut degrades the wave.
- **Speed:** a coherent wavefront is visible in **both AVC and MVC**; measured **~1.0–1.2 m/s**
  (whole-band slant-stack) to **~1.2–1.9 m/s** (leading edge) along this M-line — below the 2–3 m/s
  expectation / MATLAB reference (2.12 m/s, displacement). The number is dominated by *which feature*
  is fitted and by *M-line alignment*; a manual leading-edge line fit remains the reliable read.
- **Metric lesson:** remove the **per-time spatial mean** before any passive speed/coherence fit —
  otherwise the flat bulk-motion band dominates and inflates the apparent speed.

## Tooling (all under `scripts/`, run with `envs\zea_latest` python, `KERAS_BACKEND=torch`)

| script / function | purpose |
|---|---|
| `search_passive.py` | exhaustive 6 720-combo/window sweep → CSVs, marginals, top/consensus/baseline montages, `leaderboard.md` |
| `passive_best_montage.py` | per-window-best + direction-per-window consensus montages from the CSVs |
| `passive_filter_variety.py` | quantity×band grid for one window, fixed leftward, 2 & 3 m/s references, physiological-band pc |
| `passive_slantstack.py` | refined low-band grid in the **MATLAB M-mode orientation** with band + leading-edge fits and speed refs |
| `metrics.slant_stack_speed` | signed tau-p slant-stack (per-time spatial-mean removed) → speed **and** travel direction |
| `metrics.passive_coherence` | envelope origin-coherence (kept, but **do not** trust its speed — see finding 6) |

## Proposed next steps  *(reprioritised after the MATLAB comparison)*

1. **Drop the directional (k-ω) filter for passive** and fit speed with a **normalized Radon transform
   on the signed M-mode** (as MATLAB does) — confirmed to give physiological AVC ≈ 4 m/s vs the
   directional filter's biased 6 m/s. Fit the **peak ridge (centre), not the onset** (matches the
   literature). Retire the envelope `passive_coherence` and the leading-edge tracker for *speed*.
2. **Report displacement, velocity AND acceleration speeds** per window (Radon on each), as MATLAB
   does — velocity as the primary, acceleration as the least-biased (Petrescu 2022), displacement as
   the smoothest. Expect disp < vel < acc.
3. **Use an anatomical (septal) M-line**, and average several (~11) parallel lines with a median (as
   MATLAB) — check/curved-fit the win2 line along the basal→apical septum; oblique sampling inflates
   the apparent speed.
4. **Port the small MATLAB pre-steps**: low-pass the IQ at ~250 Hz in slow-time before the estimator;
   keep the 5–150 Hz band; `medfilt3` + moving-mean spatial/temporal smoothing.
5. **k–ω dispersion check** per window (phase velocity vs frequency) — the physically correct model for
   the thin septum; Vos found only mild dispersion, so a single group speed is defensible.
6. **Validate on a cleaner/gated AVC case** (stronger wave) once the Radon pipeline is in.

## Exhaustive search v2 (`scripts/search_passive2.py`) — all 7 requested axes

11 520 combos/window over: quantity (disp/vel/**acc**) · IQ pre-filter (none / low-pass 250 Hz) ·
SVD rank (0/1) · motion (none / bp 5-150,10-150,10-80 / poly1) · spatial (none / gauss 0.6,1.0 /
median 1.0) · temporal (none / mean 3,5 / median 3) · **directional (none / leftward)** · **N M-lines
(1/5/11) × mean|median**. Scored by a signed-Radon **propagation-clarity** metric =
(best tilted-slope semblance − flat semblance) over both directions, c∈[1.5,6] m/s; the Radon speed is
reported. Deliverables are the **plots**: `top_win{i}.png` (top-16) + `marg_{axis}_win{i}.png`
(vary one axis, hold the rest at the best) + `leaderboard_win{i}.md`, in the reference M-mode orientation.

**Verdict — the wave IS clearly propagating in BOTH windows once the directional filter is off.**
Marginal-montage findings (consistent across win0/MVC and win2/AVC unless noted):

1. **Directional filter: OFF.** Decisive. `leftward` injects horizontal reverberation banding and
   biases the slope toward vertical (score collapses ~0.15→~0). Matches the MATLAB pipeline.
2. **IQ pre-filter: none.** The 250 Hz slow-time low-pass **washes the wavefront out** (score 0.15→−0.01)
   — it removes the high-frequency content the wave (esp. acceleration) lives on. (MATLAB uses it with
   velocity, not acceleration.)
3. **Motion: a LOW band-pass** (`bp10-80` or `bp10-150`) is best; `none`/`poly1` noisier, `bp5-150` lets
   more low bulk motion through.
4. **Spatial: light Gaussian (~0.6 mm).** median ≈ gauss but slightly worse; `none` noisy.
5. **Quantity:** the clarity metric favours **acceleration** (sharpest front → highest semblance) but it
   is visually the **noisiest**; **displacement gives the cleanest wavefront**, velocity is intermediate.
   → use displacement/velocity for *visualisation*, acceleration for the *least-biased speed* (matches
   Petrescu 2022). All three give physiological speeds.
6. **N M-lines / aggregation:** for the sharp *acceleration* front **N = 1** is best (averaging blurs it);
   smoother quantities tolerate/benefit from more lines. mean ≈ median (second-order).
7. **SVD rank & temporal smoothing:** second-order (0 vs 1, and none…mean5 all ≈ equal).

**Speeds (signed Radon):** AVC (win2) ≈ **1.7 m/s (disp) / 2.0 m/s (acc)**; MVC (win0) ≈ **4.8 m/s (disp)**.
These are physiological (cf. Vos: pig MVC 2.2 / AVC 4.2; human 3.2 / 3.5) but the **MVC > AVC ordering is
inverted vs literature and the magnitudes disagree between windows** → the two windows use *separately
drawn* M-lines, so **absolute speed is still limited by M-line alignment** (each line samples the wall at
a different obliquity). Fix = anatomical septal M-lines (next steps). The *visual* conclusion —
clear propagation, no directional filter, displacement/velocity + low band-pass + light Gaussian — is robust.

### win1 is the clean AVC wave (not win2) — re-run 2026-08-04

The user re-labelled: **win1 (@ ~348 ms) = AVC, win2 = noise**. The v2 search on win1 confirms it
visually: **displacement** gives a **strong, clean, coherent wavefront at ~2.7 m/s** (no directional
filter, bp5-150/bp10-150, light median/gauss spatial, ~5 M-lines) — far clearer than win2. Velocity and
acceleration are noisier on win1 and their Radon fit degrades to ~6 m/s (near-vertical). The clarity
*metric* actually scored win1 (0.078) **below** win2 (0.154) **only because win2's winner was noisy
*acceleration*** (sharp speckle inflates semblance) — a clear demonstration that the semblance metric is
a poor proxy for *visual* clarity, and that **displacement is the quantity to visualise** (velocity/acc
for speed only). M-lines used = the saved anatomical septal lines `passive_win{i}_mline.npz`; win1 and
win2 lines are nearly identical geometry, so the difference is the cardiac *timing/wave content*, not the
line. **Updated best AVC recipe: displacement · no directional · bp5-150 (or 10-150) · light spatial
(median 1 mm ≈ gauss 0.6) · ~5 M-lines · c ≈ 2.7 m/s.** MVC (win0) displacement ≈ 4.8 m/s still reads
high vs AVC 2.7 — plausibly a phase/alignment effect; revisit once win labelling is final.

### Window labelling by same-recipe comparison + cardiac timing (2026-08-04)

`scripts/passive_compare_windows.py` runs the **config-default recipe** (disp / no-dir / bp10-150 /
gauss0.6 / mean3 / 5-line) identically on all four windows (`search2/compare_windows.png`):

| window | t_peak | Δ from win0 | signed-Radon c | appearance |
|---|---|---|---|---|
| win0 | 52 ms | 0 | 4.8 m/s | strong compact near-vertical red band |
| win1 | 377 ms | +325 ms | 6.0 m/s | near-vertical band |
| win2 | 550 ms | +498 ms | 1.5 m/s | strongly **tilted** (the outlier) |
| win3 | 920 ms | +868 ms | 3.7 m/s | strong compact near-vertical red band — **twin of win0** |

Cycle length = 920−52 = **868 ms → 69 bpm** (matches the ECG HR). So by timing: **win0 & win3 are one
full cycle apart → the same event = MVC** (early systole); **win1 (+325 ms ≈ end-systole) = AVC**;
**win2 (+498 ms ≈ mid-diastole) is NOT a valve closure** — a diastolic (E-wave/filling) event or noise.
Visually win0 and win3 are near-identical; win2 is the distinct tilted one. **Conclusion: the MVC pair is
win0 ≈ win3 (not win0 ≈ win2); win1 = AVC; win2 = diastolic outlier.** (The absolute speeds remain
recipe/alignment-sensitive; win1's own optimum bp5-150/median gave a cleaner ~2.7 m/s than bp10-150 here.)

### Three-view confirmation montage (config `run.views`, 2026-08-04)

Like the active side's multi-band view, `configs/passive.yaml` now defines **`run.views`** — three
full, deliberately-different high-performing recipes — and `run.py passive <folder>` renders **one
space-time plot per view per window** (rows = windows, columns = views; `swp.passive._build_views`).
The montage is drawn in the **M-mode orientation** (x = time, y = along-line; `spacetime_montage(...,
transpose=True)`), so a propagating wave reads as a clear diagonal. Speed per panel = signed-Radon
`slant_stack_speed(remove_flat=False)` (wave centre).

The three views (all no-directional; view B has **no temporal filter**):
- **A** `disp / bp10-150 / gauss0.6 / mean3 / 5-line`
- **B** `disp / bp5-150 / median1.0 / (no temporal) / 9-line`
- **C** `velocity / bp15-90 / gauss1.0 / mean5 / 9-line`

On invivo_sw the wavefront is **clearly visible in all three views for the valve-closure windows**
(win0/MVC, win1/AVC, win3/MVC); win2 (diastolic/noise) is the messy one, especially view C. Radon
speeds are physiological (disp views ~2.5-4.8 m/s; the velocity view reads faster, pinning ~6). A real
wave shows a consistent slope across the independent recipes; noise does not — that is the point of the
three-view design.

## Comparison with the MATLAB reference pipeline (`processIQ.m`, `ProcessDW = true`)

The user's MATLAB passive pipeline shows clearly sloped wavefronts where ours looked near-vertical.
Major differences (passive branch):

| step | MATLAB (ProcessDW) | our `swp` passive | impact |
|---|---|---|---|
| **directional (k-ω) filter** | **none** — Radon on the raw band-passed M-mode | leftward k-ω filter | **large — the prime culprit** |
| **speed / line fit** | **normalized Radon** on the *signed* M-mode, separate +/− wavefronts, on **disp, vel AND accel** | envelope `passive_coherence` / signed slant-stack | large (envelope metric is fooled) |
| slow-time band | Butterworth **5–150 Hz**, filtfilt (order 3) on disp/vel/acc | bandpass (now 10–150) | small |
| IQ pre-filter | low-pass IQ at **250 Hz** in slow-time before Kasai | none | small (SNR) |
| velocity estimator | Kasai 1-lag autocorr (3×3 spatial avg of R) | Loupas 2-D autocorr | negligible |
| spatial filter | `medfilt3` [3×3] + moving-mean [1 1 3] | Gaussian + moving-mean | small |
| M-line | **anatomical spline along the septum**, median of **11** parallel lines | manual line, mean of 5 offsets | medium (alignment) |
| bulk motion | band-pass only (no spatial-mean removal) | band-pass (+ spatial-mean removal in the metric) | — |

**Empirical confirmation** (`scripts/passive_directional_test.py`, `directional_test_win2_AVC.png`):
on win2 (AVC) velocity bp10–150, a **signed Radon fit with NO directional filter gives c ≈ 4.0 m/s**
— matching the literature (Vos pig AVC 4.2 m/s, human 3.5 m/s). The **leftward directional filter
steepens the band to 6.0 m/s** (toward vertical); rightward gives 3.2 m/s. **The k-ω directional
filter biases the apparent speed and should be dropped for passive** (MATLAB does not use it); a
**signed Radon/slant-stack on the raw band-passed M-mode** is the right fit.

## How the sloped line is fit — brief literature review

Cardiac *natural* SWE (valve-closure waves on the septum) — key methods and choices:

- **Quantity (displacement / velocity / acceleration).** Velocity is the classic choice (tissue-Doppler
  / 1-lag autocorrelation: Kanai; Vos 2017; Keijzer 2019; Santos 2019). Petrescu et al. 2022 (*Ultrasound
  Med Biol*) compared all three and found **acceleration-based time-domain speeds ≈10 % higher than
  velocity**, and *recommend acceleration* to minimise underestimation of the true speed. Ordering is
  systematic: **displacement < velocity < acceleration** — differentiation weights higher frequencies,
  which travel faster in the (weakly) dispersive Lamb-wave regime of the wall (Vos: speed rises *mildly*
  with frequency). Trade-off: displacement = smoothest/highest-SNR but most bulk-motion leakage + most
  under-estimation; acceleration = sharpest front but noisiest. This explains the MATLAB figure
  (disp 2.12 < vel 4.89 < acc 5.06 m/s).
- **Line-fit method.** (a) **Radon transform** on the space-time (Vos 2017; the MATLAB code) — the
  dominant standard for natural cardiac SWE; integrates the whole ridge to find the max-energy slope.
  (b) **Time-to-peak / time-of-flight** regression (per-position peak time vs distance; ARFI standard,
  Palmeri). (c) **Cross-correlation** lag between positions. (d) **Frequency-domain / k-ω** phase-velocity
  & dispersion (needs high SNR; often too noisy in vivo).
- **Centre or onset?** The standard methods fit the **peak / centre** of the wave, **not the onset**:
  Radon locks to the dominant ridge, TTP tracks the peak, cross-correlation aligns the dominant feature.
  These give the **group velocity**. Onset/leading-edge tracking is uncommon and noise-sensitive (and
  biases toward the fastest/highest-frequency component). → Our earlier leading-edge idea was
  non-standard; **fit the peak ridge with a (normalized) Radon transform**, as MATLAB does.

Refs: Vos et al. 2017 *Ultrasound Med Biol* (porcine, Radon on tissue velocity; MVC 2.2, AVC 4.2 m/s);
Petrescu et al. 2022 *Ultrasound Med Biol* (disp/vel/acc comparison; acceleration recommended);
Keijzer et al. 2019; Santos et al. 2019; Kanai 2005; Rouze et al. 2010 (Radon for SWS). See the chat
message for the linked sources.

## Reproduce

```
# exhaustive search + summary montages
python scripts/search_passive.py "D:\Luuk van Knippenberg\Claude\invivo_sw" --config configs/passive.yaml
python scripts/passive_best_montage.py

# domain-review montages (fixed direction, reference orientation, speed fits)
python scripts/passive_filter_variety.py --window 2 --label AVC
python scripts/passive_filter_variety.py --window 0 --label MVC
python scripts/passive_slantstack.py     --window 2 --label AVC
python scripts/passive_slantstack.py     --window 0 --label MVC
```

All outputs land in `<folder>/output/swp_passive/search/`.
