# Replacing the M-line with a 2-D field estimator

Plan for moving passive shear-wave speed estimation off a hand-drawn one-dimensional M-line and
onto the beamformed 2-D field. Written 2026-09-21, after the 34-panel hand-labelling exercise
(`docs/passive_speed_estimation.md`, report v5).

**Status: not started.** Stage 1 (synthetic validation) is the agreed entry point.

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
src/swp/field/roi.py           # draw / store / mask a septal ROI
src/swp/field/structure.py     # 3-D structure tensor -> speed, direction, coherence
src/swp/field/synthetic.py     # Stage 1 generators: plane, curved, dispersive, crossing
src/swp/field/estimate.py      # per-window driver; projection onto a stored M-line
study/analysis/field_validate.py   # Stage 1 sweep + Stage 4/5 report
```

The existing `focus` / `r0` / directional-filter machinery is ARF-derived (a push radiating from a
focus) and should **not** be reused here; a passive wave has no origin on the line.

`run_pipeline` already returns the filtered 2-D field as `res.field` with `res.times`, so no change
to the processing front end is needed - this is a change of estimator, not of data.

---

## 7. Effort

| stage | estimate |
|---|---|
| 1 - synthetic harness + sweep | ~1 day |
| 2 - ROI UI and storage | ~0.5 day |
| 3 - estimator on real data | ~0.5 day |
| 4/5 - validation, held-out split, write-up | ~1 day |

About 3 days, with a real possibility that Stage 1 says no. That outcome is worth the day it costs.
