# Replacing the M-line with a 2-D field estimator

Plan for moving passive shear-wave speed estimation off a hand-drawn one-dimensional M-line and
onto the beamformed 2-D field. Written 2026-09-21, after the 34-panel hand-labelling exercise
(`docs/passive_speed_estimation.md`, report v5).

**Status (2026-09-21).**

| stage | outcome |
|---|---|
| 1 - synthetic validation of the **structure tensor** | speed/direction **PASS** (0.7 % / 0.2 deg p90, holds to 0 dB SNR); coherence gate **FAILS** its negative control |
| 2 - ROI | implemented (`field/roi.py`) |
| 3 - **structure tensor** on real data | **FAILED** - it locks onto axial speckle, not the wave; no spatial scale rescues it |
| 1b - speckle added to the synthetic harness | **reproduces the real failure exactly**: tensor 3.01 -> 0.32 m/s, theta 0 -> -72 deg |
| 1c - synthetic validation of the **phase-gradient** estimator | **PASSES under full speckle** at 6 dB SNR (4.8 % / 2.4 deg p90); fails under moving axial clutter |
| 3b - phase gradient on real data | plausible speeds and a wall-aligned direction at AVC; reproduces the independently measured dispersion |
| 4 - phase gradient vs the 34 hand-drawn panels | **FAILED** - V2 and V3 both miss by a wide margin |
| **verdict** | **the field approach does not work on this data**; see section 4e |

Read [Stage 1 results](#stage-1-results-2026-09-21) and Stage 3 before the method sections: the
structure tensor described in section 3 is **superseded** for this data, and section 3b says why
the replacement should work where it did not.

---

## 1. Why the M-line has to go

Not because the estimator on it is badly implemented, but because the measurement it defines is
ill-posed. Four independent results say so, all measured rather than argued:

**A single line cannot separate speed from direction.** The apparent speed along a line is
`c / cos(theta)`, where `theta` is the angle between the propagation direction and the line. Every
1-D estimate is therefore biased *high*, with an unknown, per-window factor. That is exactly what
the labelled set shows: the automatic fit reads higher than the hand in 21 of 24 comparable panels,
median +24 %. It also explains why the apparent speed varies *along* a line - on the C000000023 MVC
velocity panel it falls from ~2.5 to ~1.1 m/s between r = 0 and r = 13 mm, which no real wave does.

**The aperture cannot be fixed by making it longer.** Doubling the line from ~21 to ~43 mm made
every measure worse: the cross-correlation lag residual went from 0.14-0.49 ms to 1.9-30.3 ms, i.e.
the arrival time stopped being a linear function of position at all. The wave is coherent over
roughly 20 mm and not over 43 mm. So the half is coherent but too short to constrain a slope, and
the full line is long enough but not coherent. **There is no good choice available in 1-D.**

**The line carries almost no independent information.** A 250-sample M-line panel holds about 4
independent along-line positions and 9-18 independent time points (1/e correlation lengths of
4.5-5.1 mm and 8-14 ms). Fitting a slope to ~4 points is the root of the +/-25 % hand precision.

**No 1-D quality gate works.** Five reference-free statistics were scored as classifiers over 30
hand-drawn panels. Semblance was the best at AUC 0.80/0.87, and the best combination of gates keeps
11 of 30 panels while being right on 7. There is currently no way to tell a good 1-D fit from a bad
one reliably enough to automate.

What the field approach changes: **direction stops being an input and becomes an output.**

---

## 2. What the user still has to provide

This is the question to be honest about, because the answer is "less, but not nothing".

| | today (M-line) | proposed (field) |
|---|---|---|
| geometry input | an **oriented line**: 2 endpoints, per cardiac event | a **region**: the septal wall, once per acquisition |
| what it fixes | where to look **and which direction to measure** | where to look only |
| propagation direction | implicitly assumed = along the line | **estimated per pixel** |
| wave origin `r0` | needed by the ARF-derived code paths | not needed |
| per acquisition | ~4 lines (one per event) | 1 ROI |
| automatable later? | no - direction cannot be guessed | yes - this is a segmentation problem |

So yes, user input is still required, and it is still a point/region of interest where the wave
propagates. But it no longer determines the answer, which is the whole point: an ROI that is drawn
10 % too large changes the averaging, whereas an M-line drawn 10 degrees off changes the speed by
`1/cos(10 deg)`.

The ROI can start as a polygon drawn on the same B-mode frame the M-line is drawn on now, reusing
`swp.mline.select`. A seed point plus a wall-thickness parameter is an acceptable minimum.

---

## 3. Method

### 3.1 The core idea

For a locally plane wave `u(x, z, t) = f(k . x - omega t)`, the 3-D gradient

```
grad_3 u = (du/dx, du/dz, du/dt)  is parallel to  (kx, kz, -omega)
```

so the local orientation of the space-time volume *is* the wave vector. Form the structure tensor
over a local window `W`:

```
J = < grad_3 u   grad_3 u^T >_W        (3x3, symmetric, positive semi-definite)
```

Its dominant eigenvector `v = (vx, vz, vt)` estimates the direction of `(kx, kz, -omega)`, and

```
direction  theta = atan2(vz, vx)
speed      c     = |vt| / sqrt(vx^2 + vz^2)        (with vx, vz, vt in physical units)
```

Gradients must be scaled by `dx`, `dz`, `dt` before the tensor is formed, or the speed is wrong by
the pixel aspect ratio - this is the single most likely silent bug and is what Stage 1 exists to
catch.

### 3.2 Why this is not subject to the 1-D geometric objection

The slant stack fails because it needs the wave to move measurably *across the aperture*: over
21 mm a 170-290 mm wave moves ~5 ms out of a ~60 ms period. The structure tensor does not measure
displacement across an aperture; it measures the **local orientation of the gradient**, which is
well defined even where the wave is nearly spatially uniform, provided the temporal gradient is
resolved - and at 926 Hz frame rate against a 15-25 Hz wave, it is, generously.

This is a claim, not a fact, until Stage 1 tests it. It is the central assumption of the whole
plan and Stage 1's synthetic sweep is designed to find where it breaks.

### 3.3 Confidence, for free

The eigenvalues `lambda1 >= lambda2 >= lambda3` give a local planarity measure, e.g.

```
coherence = (lambda1 - lambda2) / (lambda1 + lambda2 + eps)
```

which is near 1 for a single plane wave and near 0 for noise, interference of two waves, or bulk
motion. **This is the quality gate the 1-D path never had.** It must be validated as such
(Stage 5), not assumed.

### 3.4 Fallback

If the structure tensor proves too noise-sensitive at the achievable SNR, the fallback is a local
3-D plane-wave fit (Radon / normalised cross-correlation of neighbouring traces) over small
patches: slower, more robust, same outputs. Decide at Stage 3, not before.

---

## 3b. The phase-gradient estimator (the replacement)

Stage 3 showed the problem precisely: the structure tensor assumes the **gradient field is
dominated by the wave**, and in a real axial displacement field it is dominated by **speckle
amplitude structure** instead. The fix is to work on a quantity speckle does not corrupt.

Take the temporal Fourier component at the wave's dominant frequency, giving a complex field

```
U(x, z) = A(x, z) exp( i phi(x, z) ),      phi = -k . r
```

Speckle and attenuation act on the *amplitude* `A`, which is real and slowly varying. The wave
vector is in the *phase*, and

```
grad U / U = grad(ln A) + i grad(phi)        so       k = -Im( grad U / U )
```

**The real amplitude term falls entirely in the real part**, so taking the imaginary part removes
speckle to first order. Then

```
c = omega / |k|,      theta = atan2(kz, kx)
```

Three further properties make this the right tool here:

* **no phase unwrapping** - the gradient of the complex field is used directly, so 2-pi jumps
  never arise;
* **amplitude becomes a weight, not a contaminant** - `|U|^2` is the natural confidence weight,
  and it is exactly where the wave is strong;
* **it is frequency-resolved by construction** - which the dispersion result
  (1.4 m/s at 7 Hz to 9.3 m/s at 28 Hz on one real window) says is necessary, and which the
  structure tensor could not provide.

This is local frequency estimation as used routinely in MR elastography, where recovering `k`
from a noisy displacement field is the standard problem. That is a reason to expect it to work,
not evidence that it does - the speckle-augmented synthetic test is what would turn it into
evidence, and it must be run before real data.

## 4. Implementation stages

Each stage has an exit criterion. **Do not start a stage before the previous one passes.**

### Stage 1 - synthetic validation harness (no real data)

Build a generator for a known field and verify the estimator recovers it.

* plane wave at known `c` and `theta`, sampled on the real grid (`dx`, `dz` ~0.39 mm, `dt` ~1.08 ms)
  over a realistic ROI (~10 mm wall x ~40 mm length) and window (~100 ms);
* sweep `c` over 1-8 m/s, `theta` over 0-80 degrees, frequency over 10-40 Hz;
* add the pipeline's own processing: spatial Gaussian (0.6/1.2 mm), temporal moving mean,
  band-pass, and offset averaging;
* add noise at measured SNR, plus a bulk-motion component (spatially uniform, low frequency),
  because that is what actually contaminates these panels;
* additional cases: a **curved** wavefront (radiating from a point off the ROI), a **dispersive**
  packet, and **two crossing waves**.

**Exit criterion:** `c` recovered within 10 % and `theta` within 10 degrees for the plane-wave
sweep at realistic SNR; coherence must drop on the two-crossing-waves case. If the estimator cannot
do this on data it was *given*, it will not do it on real data.

### Stage 2 - ROI definition

Polygon or band drawn on the B-mode frame already used for M-line drawing, stored per acquisition
alongside the existing `mlines/` artefacts. Reuse `swp.mline.select`. Keep the stored M-lines - they
are needed for Stage 4 validation.

**Exit criterion:** an ROI round-trips to disk and masks the field correctly on 3 acquisitions.

### Stage 3 - estimator on real data

Run over the event windows. Outputs per window: coherence-weighted speed, direction field, coherence
map, and the speed projected onto any stored M-line for that window.

**Exit criterion:** displacement and velocity fields agree (Stage 5, V2) on a majority of windows.
Differentiation multiplies by `i*omega`, which cannot change a spatial phase gradient - so a
disagreement here is a bug or a noise floor, not physiology. This is a much stronger internal check
than anything available in 1-D, and it is the reason to prefer this design.

### Stage 4 - validation against the labelled set

The 34 hand-drawn panels are 1-D *apparent* speeds. The correct comparison is therefore a
**projection**, not a direct one:

```
predicted apparent speed along the drawn line = c_field / cos(theta_field - theta_line)
```

This tests `c` and `theta` jointly, which is strictly stronger than comparing speeds. A 2-D
estimate that gets `c` right and `theta` wrong will fail it.

### Stage 5 - the pre-registered criteria

Fixed before any real-data run, so that they cannot be adjusted to fit the outcome.

| id | test | criterion |
|---|---|---|
| V1 | synthetic plane-wave sweep (Stage 1) | `c` within 10 %, `theta` within 10 deg |
| V2 | displacement field vs velocity field | agree within 25 % on > 70 % of windows |
| V3 | projection onto the 34 hand labels (Stage 4) | median absolute error < 25 % |
| V4 | tune on half the subjects, report on the other half | nothing reported is tuned on |
| V5 | negative control: `other` / noise windows | coherence separates real from null, **AUC > 0.87** |

**V5's bar is set at semblance's measured AUC of 0.87.** If the coherence gate cannot beat the
statistic already available, the added complexity has not earned its place and the result is a
negative one worth writing down.

**V4 exists because of a specific failure in this project.** Three conclusions drawn from
C000000023 alone did not survive the labelled set: a ridge-tracking estimator that looked
convincing there reads systematically high on dispersive packets; the cross-correlation lag
residual looked like a decisive quality gate and is worse than semblance; and the AVC-faster-than-
MVC ordering does not generalise. The one conclusion that did survive - the automatic fit's upward
bias - was the only one that had been tested against an independent reference rather than against
itself.

---

## 4b. Stage 1 results (2026-09-21)
<a name="stage-1-results-2026-09-21"></a>

Code: `src/swp/field/{structure,synthetic}.py`, `study/analysis/field_validate.py`. Everything is
sampled on the real grid (dz = dx = 0.3945 mm, dt = 1.08 ms) in a 12 x 40 mm ROI over 110 ms and
passed through the pipeline's own smoothing and band-pass.

### What passed

| test | result | criterion |
|---|---|---|
| noiseless plane wave, c = 1.5-8 m/s, theta = 0-75 deg | speed p90 **7.6 %**, direction p90 **2.3 deg** | 10 % / 10 deg - **pass** |
| with noise | still within criterion at **0 dB SNR** | - |
| curved wavefront (point source 25 mm outside the ROI) | speed within **1.2 %** | - |
| bulk motion at 1x signal RMS | p90 **9.8 %** - pass; at 3x it breaks (29 %) | - |

**The central assumption holds.** The structure tensor does not need the wave to cross the
aperture, so the geometry that defeats the slant stack does not apply: at 3 m/s and 16 Hz,
`|k| ~ 33 rad/m` against `omega ~ 100 rad/s` is an ordinary, well-conditioned orientation.

### Two implementation faults the synthetic stage caught

* **Edge contamination.** The averaging window uses `mode="nearest"`, so within ~2 sigma of a face
  the tensor is built partly from replicated samples, which biases the speed **high**: +14.6 % at
  theta = 60 deg over the whole volume against +8.0 % over the interior. Now excluded
  automatically by `FieldEstimate.interior()`.
* **The default window was far too large.** At 3 mm, edge exclusion left **2.3 %** of a 12 mm wall
  usable; at 1.2 mm it leaves **23 %** at no cost in accuracy. Defaults are now
  `window_space_mm=1.2`, `window_time_ms=8.0`.

### A limitation to design around

Oblique propagation in a *thin* wall is biased: +5.8 % at 60 deg and +7.3 % at 75 deg in a 12 mm
wall, against <= 0.5 % at every angle in a 24 mm ROI. A septal shear wave travels *along* the wall
(theta near 0), which is the favourable case - but the method wants the longest wall segment
available, not the thinnest.

### What failed: the coherence gate

**Coherence does not detect two co-propagating crossing waves.** A single wave scores 0.995 and
two waves 60 deg apart score 0.992, while the reported speed is corrupted from 3.0 to 3.5-4.4 m/s
with full confidence. This holds at every window from 0.8 to 5 mm and 4 to 16 ms, and no
alternative tried (speed IQR, direction spread) separates them either.

The reason is structural rather than a bug: **a superposition of two plane waves is still
*locally* a well-oriented space-time structure.** The tensor measures local orientation, which is
perfectly well defined - it simply is not either wave vector. So this is a genuine blind spot of
the method.

Coherence does work for the other failure modes - pure noise 0.38, counter-propagating waves 0.94
(plus speed IQR 0.79). Only the co-propagating case is invisible.

### Decision taken (agreed 2026-09-21)

1. **Adopt `interference_indicator` as a candidate, not a gate.** Interference leaves standing
   spatial structure in the *amplitude* even where the local orientation stays clean. The
   envelope's coefficient of variation separates the cases 3-14x: single 0.008, two crossing at
   60 deg 0.025-0.035, counter-propagating 0.115, noise 0.168. **It must be validated on real
   data before being used**, because in vivo the envelope also varies with attenuation, speckle,
   coupling and wall curvature, any of which could swamp a 3x effect. Adopting it on synthetic
   evidence alone would repeat a mistake this project has now made three times.
2. **Document the blind spot and rely on V2 to catch it indirectly.** The estimator assumes a
   single dominant wave. Displacement and velocity are the same field up to a factor `i*omega`,
   so a disagreement between them is evidence that the assumption is broken.
3. **Do not pursue narrowband decomposition yet.** Two waves at the same frequency stay merged
   under it anyway; whether a second coherent wave is present often enough in vivo to matter is
   unanswered and cheap to probe once Stage 3 runs on real windows.

### Consequence for the pre-registered criteria

V5 as written - "coherence separates real from null, AUC > 0.87" - is now known to be testing
something coherence cannot do in the co-propagating case. It stands for the noise / null-window
comparison it was written for, and the interference question moves to a new criterion:

| id | test | criterion |
|---|---|---|
| V6 | `interference_indicator` on real windows vs hand-labelled single-wavefront panels | separates by > 2x, or it is dropped |

## 4c. Stage 3 first contact with real data (2026-09-21): FAILED

Run on C000000023, both valve closures, ROI derived as a 12 mm band around the stored per-event
M-line.

**The projections looked excellent and are spurious.** AVC projected 3.52 m/s against a hand value
of 3.56 (-1.1 %), MVC 2.75 against 2.89 (-4.7 %). But the underlying estimate was `c = 0.71 m/s`
at `theta = -87 deg`, and the projection divides by `cos(78 deg) = 0.2`, close to the 1/8 cap.
A tiny speed divided by a near-zero cosine can land anywhere; the agreement carries no
information. **Do not quote those numbers.**

### The diagnosis

Direction histogram over the ROI piles up at **+/- 90 degrees - purely axial** - and the *higher*
the coherence, the *lower* the speed:

| coherence percentile | direction | speed |
|---|---|---|
| top 50 % | -83 deg | 0.68 m/s |
| top 20 % | -80 deg | 0.37 m/s |
| top 5 %  | -79 deg | 0.22 m/s |

against coherence 0.65 (synthetic single wave: 0.998), direction spread 37-46 deg (synthetic:
0.6 deg), and the interference indicator at **1.3-1.4 where pure synthetic noise scores 0.168**.

The tensor is locking onto **axial speckle and strain structure, not the shear wave**. The cause
is a modelling gap, not a coding bug: Loupas estimates only the *axial* component of motion, so a
real displacement field carries strong z-structure from speckle and from the estimator itself,
while the synthetic generator produced a clean scalar wave with no axial speckle at all. Stage 1
passing at 0 dB of *white* noise said nothing about *structured* axial noise.

### A coarser scale does not rescue it

Sweeping the spatial derivative scale from 0.8 to 8 mm on the real AVC window:

| sigma_space | c | theta | spread | kept |
|---|---|---|---|---|
| 0.8 mm | 0.71 | -87 deg | 37 deg | 0.34 % |
| 2.0 mm | 2.67 | -81 deg | 48 deg | 0.47 % |
| 4.0 mm | 4.71 | -83 deg | 52 deg | 0.51 % |
| 6.0 mm | 6.55 | +40 deg | 51 deg | 0.65 % |
| 8.0 mm | 7.42 | +43 deg | 29 deg | 0.82 % |

**The speed varies by 10x across the sweep with no plateau**, and the direction never settles on
the wall. A genuine wave would give a stable `c` over a range of scales. MVC behaves the same way
(c wanders 1.30 -> 0.59 -> 2.75 -> 3.98 -> 4.56, theta 78 -> 40 -> 13 -> 29 -> 40 deg). The
`sigma = 4 mm` MVC point happens to give theta within 0.4 deg of the M-line and c = 2.75 against a
hand value of 2.89 - that is one point in a wandering sequence, not a plateau, and quoting it
would repeat the single-dataset error this project has already made three times.

### Conclusion

**The structure tensor on the raw axial displacement field does not work on this data.** The
mathematics is sound (Stage 1 is unambiguous) but the measurement model is wrong: amplitude
structure from speckle dominates the gradient field that the method assumes is dominated by the
wave.

## 4d. The phase-gradient estimator: results (2026-09-21)

### Speckle first - the harness now contains the confound

`synthetic.speckle_amplitude` adds a static, spatially correlated amplitude field at the real
speckle correlation length, and `axial_clutter` adds slowly moving axial banding. With speckle at
contrast 1.0 the **structure tensor reproduces the real-data failure exactly**:

| | c (truth 3.00) | theta (truth 0) | coherence |
|---|---|---|---|
| clean | 3.01 | +0.0 deg | 0.998 |
| + speckle 1.0 | **0.32** | **-71.6 deg** | 0.591 |
| real C000000023 AVC | 0.71 | -87 deg | 0.65 |

That match is what makes the harness a gate rather than a demonstration.

### The phase estimator passes it

Swept over c = 1.5-6 m/s, theta = 0-60 deg, f = 12-22 Hz (75 cases per row):

| condition | median / p90 speed error | median / p90 direction error |
|---|---|---|
| no speckle, SNR 15 dB | 0.7 % / 1.0 % | 0.1 / 0.3 deg |
| speckle 1.0, SNR 15 dB | 3.8 % / **4.6 %** | 0.5 / **0.9 deg** |
| speckle 1.0, SNR 6 dB | 3.4 % / **4.8 %** | 0.9 / **2.4 deg** |
| speckle 1.0 + clutter x0.5 | 28.8 % / 59.5 % | 19.3 / 46.8 deg |
| speckle 1.0 + clutter x1.0 | 48.7 % / 75.7 % | 27.4 / 61.4 deg |

**Speckle is solved; moving axial clutter is not.** On the exact case that broke the tensor the
side-by-side is 0.32 m/s at -72 deg against **3.17 m/s at +0.3 deg**.

A targeted clutter filter was tried and **rejected**: subtracting the per-(t, z) lateral mean
removes laterally-uniform clutter perfectly, but it destroys the wave with it (c -> 0.90 m/s,
theta -> +52 deg, identical at every clutter level). The reason is the one already documented for
`remove_flat` in the 1-D path - over a 40 mm ROI a 187 mm wave *is* nearly laterally uniform. The
window is also too short to separate the two spectrally: 110 ms gives ~9 Hz resolution, so 8 Hz
clutter leaks into a 16 Hz projection.

### First real-data behaviour

C000000023, both valve closures, per-frequency:

| f [Hz] | AVC c | AVC theta | MVC c | MVC theta |
|---|---|---|---|---|
| 10 | 1.06 | -42 deg | 1.97 | +88 deg |
| 16 | 2.59 | -18 deg | 3.58 | +59 deg |
| 20 | 3.87 | +35 deg | 3.51 | +58 deg |
| 25 | 4.40 | +50 deg | 3.08 | +63 deg |

Three things changed for the better against the structure tensor:

1. **The speeds are physically plausible** (1-4.4 m/s) rather than 0.2-0.7 m/s.
2. **The AVC direction histogram now peaks along the wall** (0 to +30 deg, against an M-line at
   0 deg), where the tensor piled up at +/-90 deg. A secondary lobe near +/-90 deg remains.
3. **The frequency dependence reproduces the dispersion measured independently** from the 1-D
   cross-spectral analysis (1.4 m/s at 7 Hz to 9.3 at 28 Hz). Two unrelated methods agreeing on
   the dispersion is the first genuine cross-validation this approach has produced.

**Not yet a result.** Projected onto the hand-drawn lines at 16 Hz, AVC gives 2.73 against a hand
value of 3.56 (-23 %) and MVC 5.17 against 2.89 (+79 %) - one of two within the V3 tolerance, on
n = 2. MVC's direction (+59 deg) is not wall-aligned and the direction spread is 22-50 deg
throughout. Stage 4 over the 34 hand-drawn panels is what decides this, and it has not run.

## 4e. Stage 4 verdict (2026-09-22): the field approach fails, and why

15 windows x 2 views x 4 frequencies, `study/logs/field_stage4.csv`. Nothing was tuned on it -
every parameter was fixed on synthetic data beforehand.

### V2 fails, and it is the decisive one

Displacement and velocity are the same field times `i*omega`, so `grad U / U` is **mathematically
required to be identical** at a fixed frequency. Over 43 window/frequency pairs:

| | median | p90 | within tolerance |
|---|---|---|---|
| speed ratio | **1.25x** | 2.29x | 53 % within 25 % |
| direction difference | **12.4 deg** | 32.4 deg | 42 % within 10 deg |

Two quantities that must agree exactly disagree by 25 % typically. That is the estimator's noise
floor on real data, and it is far too large for the measurement. (Part of it is the two views'
different bands and smoothing, but not a 2.3x tail.)

### V3 fails at every frequency

Criterion: median projected error < 25 %.

| f | n | median | p90 | within 25 % |
|---|---|---|---|---|
| 13 Hz | 15 | 39 % | 163 % | 33 % |
| 16 Hz | 15 | 59 % | 394 % | 20 % |
| 20 Hz | 14 | 98 % | 623 % | 21 % |

### The direction is not along the wall

| | median &#124;dtheta&#124; from the M-line | within 30 deg |
|---|---|---|
| AVC | 38.9 deg | 29 % |
| MVC | 72.6 deg | 14 % |

Median speeds are 1.29-1.76 m/s against hand picks of 2-5 and a literature range of 1.6-4.8.
C000000023's wall-aligned AVC direction, reported as encouraging, was not representative - the
**fourth** conclusion from that dataset alone not to survive the labelled set. It was selected as
the best case, so it is systematically unrepresentative by construction.

### It is not the ROI

Sweeping the band width from 20 mm down to 2 mm on four windows leaves the estimated speed low
(0.6-2.0 m/s) and the direction 30-75 deg off the wall at every width. The apparent improvement at
2 mm comes from the `1/cos(dtheta)` factor, not from a better `c`.

### What actually explains it

The synthetic sweep **predicted this failure**. The one contaminant the phase estimator cannot
handle is moving axial clutter, and its signature there - speed collapsing, direction swinging
towards +/-90 deg - is exactly what the real data shows:

| | speed error | direction error |
|---|---|---|
| synthetic, speckle + clutter x0.5 | 29 % median | 19 deg median |
| synthetic, speckle + clutter x1.0 | 49 % median | 27 deg median |
| **real data** | **39-98 % median** | **39-73 deg median** |

So the diagnosis is not "unknown": these fields contain substantial moving axial clutter, and
**there is no filter for it that preserves the wave**. The lateral-demean filter removes it
perfectly and destroys the wave with it, because over a 40 mm ROI a 187 mm wave is itself nearly
laterally uniform.

### The unifying fact

That last point is the whole project in one sentence. **The shear wavelength is 5-10x the
imageable region** (150-290 mm against 19-43 mm of M-line or ~40 mm of ROI). That single geometric
fact defeats, for the same reason each time:

* the **slant stack** - a wave nearly uniform across the aperture has a poorly constrained slope
  (+24 % bias, 20 % of fits unusable);
* `remove_flat` and every other **bulk-motion filter** - what is uniform in space is the wave too;
* the **structure tensor** - the spatial gradient it needs is smaller than the speckle gradient;
* the **phase gradient** - the phase advance across the ROI is a small fraction of a cycle, so
  clutter of comparable magnitude dominates it;
* the **lateral-demean clutter filter** - same argument as `remove_flat`.

It is not an algorithmic problem and no estimator fixes it. What would: a longer imaging aperture
(a wider sector or a stitched view), a higher-frequency wave where the wavelength is shorter, or
an acquisition that suppresses the clutter at source.

### Recommendation

**Stop the field-estimator work here.** The code, the synthetic harness and both negative results
are kept - `structure.py` and `phase.py` are correct implementations that pass their synthetic
gates, and the harness is now realistic enough to predict real failures, which is worth having.
What should not happen is a fourth estimator on the same geometry.

The defensible measurement for this data remains a hand-drawn slope with a stated +/-25 %
uncertainty, cross-checked between displacement and velocity, on windows whose R-peak record
passes `swp.acquisition.rrcheck`.

## 5. What this does *not* fix

State these in any write-up, because a 2-D estimator will otherwise be over-sold.

* **Out-of-plane propagation.** A 2-D image of a 3-D wave still yields `c_apparent >= c_true` for
  the component leaving the imaging plane. This reduces the obliquity bias, it does not eliminate
  it. Salles et al. went to 4-D acquisition for exactly this reason.
* **Dispersion.** Phase speed on one AVC window rises from 1.4 m/s at 7 Hz to 9.3 m/s at 28 Hz.
  Any single number is band-dependent; the band must be fixed and reported. Group delay over the
  band is the agreed measurand; full dispersion analysis is a later goal.
* **The trigger, for 20 of 44 folders.** Assessed on the trigger record itself (never on the
  logged waveform - see `docs/ecg_timing.md`), 24 of 44 acquisitions have an R-peak record whose
  cardiac phases can be trusted; 15 warn and 5 are unusable fixed-rate pulse trains
  (`swp.acquisition.rrcheck`, `study/logs/ecg_check.csv`). The gating itself is exact - the
  passive block starts within 6 us of a recorded R-peak in every folder - so this is a labelling
  problem, not a timing one. No estimator fixes a window labelled MVC that was really an atrial
  kick, so the warned and unusable folders must be excluded from any MVC/AVC comparison.
* **Wall thickness.** ~10 mm limits the spatial window of the tensor; the coherence gate has to
  carry that load.

---

## 6. Proposed layout

```
src/swp/field/__init__.py
src/swp/field/roi.py           # draw / store / mask a septal ROI            [done]
src/swp/field/structure.py     # 3-D structure tensor                        [done, SUPERSEDED]
src/swp/field/phase.py         # phase-gradient / local frequency estimator  [next]
src/swp/field/synthetic.py     # generators: plane, curved, dispersive, crossing, SPECKLE
src/swp/field/estimate.py      # per-window driver; projection onto a stored M-line [done]
study/analysis/field_validate.py   # synthetic sweep + verdict against the criteria
```

`structure.py` is kept rather than deleted: Stage 1 proves the implementation is correct, so it
stays as the reference against which the phase estimator is compared, and as the record of why
the approach was changed.

The existing `focus` / `r0` / directional-filter machinery is ARF-derived (a push radiating from a
focus) and should **not** be reused here; a passive wave has no origin on the line.

`run_pipeline` already returns the filtered 2-D field as `res.field` with `res.times`, so no change
to the processing front end is needed - this is a change of estimator, not of data.

---

## 7. Effort

| stage | estimated | actual |
|---|---|---|
| 1 - synthetic harness + sweep | ~1 day | done |
| 2 - ROI UI and storage | ~0.5 day | done |
| 3 - estimator on real data | ~0.5 day | done - **failed** |
| 4/5 - validation, held-out split, write-up | ~1 day | not reached |

**The original "about 3 days" was too confident, and the reason is worth recording.** The estimate
priced the mathematics, which was never the risk: Stage 1 passed at 0.7 % error. What it did not
price was the *measurement model* - that Loupas returns only the axial component, and that a real
axial displacement field is dominated by speckle structure rather than by the wave. A synthetic
harness that omits the dominant confound cannot gate anything, and Stage 1 passing gave false
confidence for a day.

The rule that follows: **the synthetic model must contain the thing most likely to break the
method before its result counts as a gate.** For the phase estimator that means speckle first,
real data second.
