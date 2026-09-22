# Reading a speed off a passive space-time plot

How the passive shear-wave speed is estimated, why the automatic estimator is not trustworthy on
its own, and how it compares against wavefronts drawn by hand. Established 2026-09-18 on
C000000023. Code: `swp.viz.metrics.slant_stack_speed`, `study/analysis/manual_slope.py`.

## A passive wave has no symmetric origin

ARF-push SWE has a focus: the push radiates outward from it, so the space-time plot shows a
symmetric V and the natural estimator fits both lobes about `r0`. **A valve-closure wave does not
work that way** - it enters the M-line at one end and crosses it in a single direction.

Two consequences, both settled on 2026-09-18:

* The **reported** speed always came from `metrics.slant_stack_speed`, a signed tau-p slant stack
  over the whole line whose sign encodes direction. Its docstring is explicit that `r0` is unused.
  Nothing in the reported numbers was ever symmetric.
* The **overlay** on the passive montage did not follow. It came from `speed.tof.ttp_ransac_speed`,
  which splits the line at `r0` (`_sides()`) and fits each side separately, over only 2-14 mm
  either side of the origin - the ARF picture, wrong here. That is what produced the two short
  black segments straddling a dashed `r0` line.

The passive path now overlays the slant stack's own moveout as **one continuous wavefront**
(`swp.passive._single_wave_speed`) and drops the origin marker
(`spacetime_montage(..., show_r0=False)`). Panels are labelled `radon-signed: <c>/-- m/s`. The
active path is untouched: there the symmetric V is real.

## Automatic versus manual: the automatic fit does not follow the wave

`study/analysis/manual_slope.py` draws the wavefront by hand on the M-mode panel (two clicks; speed
is then `dr/dt`, mm/ms = m/s) and stores the picks in `output/swp_passive/manual_slopes.json`. That
gives a reference for judging the automatic estimator.

Two things have to be separated: whether the **slope** is right, and whether the **line is on the
wave**. The second is measured with a *tracking score* - the mean `|signal|` sampled along the
fitted line divided by the panel RMS. A line on the crest scores well above 1, a line through noise
scores about 1, a line in a trough scores below 1.

C000000023, left part of the M-line, four hand-drawn panels:

| window | quantity | manual | automatic | bias | tracks (manual) | tracks (auto) |
|---|---|---|---|---|---|---|
| MVC | displacement | 2.67 | 3.27 | +22 % | 1.40 | 0.94 |
| MVC | velocity | 0.94 | 1.65 | +76 % | 1.23 | 0.44 |
| AVC | displacement | 3.58 | 3.96 | +11 % | 2.58 | 2.47 |
| AVC | velocity | 3.80 | 4.32 | +14 % | 2.60 | **0.23** |
| | mean | | | **+31 %** | **1.95** | **1.02** |

Findings:

1. **The automatic estimate is biased high in every case**, by 11-76 %. Never low.
2. **Its mean tracking score is 1.02** - averaged over these panels the fitted line is
   statistically indistinguishable from a line drawn through noise, while the manual lines score
   1.95.
3. **A right slope does not mean a right line.** In the AVC velocity panel the automatic slope is
   within 14 % of the manual one, but the line sits about 5 ms early - in a trough (0.23) rather
   than on the crest (2.60). In the MVC panels it strays further still.

**Why.** The slant stack maximises a *global* semblance over the whole panel and then places the
line at the peak of the coherent stack. Nothing in that objective requires the line to pass through
the wave the operator cares about: a band of bulk motion, or the wavefront's leading edge rather
than its centre, satisfies it equally well. For a measurement that should track peak displacement
or peak velocity, this is the wrong objective.

**The fix to make.** An estimator whose objective is the amplitude sampled along the fitted line -
essentially maximising the tracking score above - would be matched to the task by construction.
This is the single highest-value change to the passive speed path and has not been made yet.

## Until then: read the speed by hand

```
python study/analysis/manual_slope.py --folder "<folder>" --window 1 --part left
python study/analysis/manual_slope.py --folder "<folder>" --window 1 --part left --figure out.png
```

Click two points on the wavefront; the line, its speed and the automatic fit for comparison are
drawn live. `r` clears, ENTER accepts, closing the window skips that panel. Picks are keyed by
`(window, part, view)` and reused on a later run unless `--redraw` is given, so a figure can be
re-rendered without redrawing.

Use the automatic fit only to **rank and triage** windows, never as the reported number. Cross-check
displacement against velocity: where they agree the result is credible (AVC: 3.56 vs 3.82 m/s,
7 %), where they disagree by a factor of two or more the window should not be reported (MVC: 2.89
vs 1.28 m/s, 2.3x).

**Add a plausibility band to that rule.** Over 15 hand-drawn windows the cross-check passes 9, but
two of those passes are spurious - C000000008 agrees with itself to 1.03x on a wave travelling
*backwards* at 0.45 m/s, and C000000022 at 0.91/0.93 m/s. Two views of the same field can agree
perfectly on the same artefact. Rejecting outside roughly 1-8 m/s removes both and takes the real
pass rate to 7/15.

**The hand pick itself is good to about +/-25 %.** Redrawn blind, the AVC pair reproduced to 0.6 %
and MVC displacement to 8 %, while an independent ridge-tracking fit landed 25 % away with an
*identical* on-wave score. Two clicks over a ~5 ms moveout is not a precise instrument: a 1 ms slip
is a 25 % error. Quote speeds accordingly, not at the two decimals the tool prints.

## Speeds differ between cardiac events - but the ORDERING does not generalise

C000000023 gives 2.89 m/s at MVC against 3.56 m/s at AVC in displacement, a ratio of about 1.2.
**That is a single-subject observation and it does not survive a larger sample.** Across 15
windows in 15 subjects the ordering reverses depending on which defensible subset is taken:

| subset | MVC | AVC |
|---|---|---|
| all displacement panels | 2.29 (n=8) | 2.80 (n=7) |
| all velocity panels | 3.93 (n=8) | 3.52 (n=7) |
| passing the disp/vel cross-check | 2.73 (n=5) | 2.10 (n=4) |

The interquartile ranges overlap almost completely, so **this dataset cannot resolve a difference
between the two events**. It is not a labelling problem: 10 of the 15 windows come from
acquisitions whose R-peak record passes `swp.acquisition.rrcheck`, and restricting to those does
not separate the events either.

The physiological reasoning also needs care. It is tempting to say MVC catches relaxed
end-diastolic myocardium and AVC contracted end-systolic myocardium, so AVC must be faster - but
the MVC wave propagates during **isovolumic contraction**, when active tension is rising fastest.
The published ordering is correspondingly inconsistent: Keijzer et al. (pigs) and Salles et al.
(4D, human) both find AVC faster, while Strachinaru et al. find MVC faster in every subject.

Do not pool events. A single "shear-wave speed" for a subject is not a meaningful quantity without
stating the cardiac phase it was measured at.

## The wave is dispersive, so "the" speed needs a definition

Phase speed estimated per frequency bin from the cross-spectral phase gradient along the line, on
one AVC window: **1.4 m/s at 7 Hz, 3.1 at 14 Hz, 5.8 at 21 Hz, 9.3 at 28 Hz**. That is a flexural,
Lamb-type *guided* mode in the wall, not a bulk shear wave - expected physics for a ~10 mm wall,
and consistent with the 0.17 m wavelength Salles et al. report.

So five defensible estimators on the same window give 3.56 (hand), 3.93 (cross-correlation group
delay), 3.96 (slant stack), 4.64 (ridge peak tracking) and 3.12 m/s (phase speed at the dominant
14 Hz bin). These are not estimators disagreeing; they are group speed, phase speed and peak speed
on a dispersive packet. **The agreed measurand is group delay over the band**, which is what the
comparable literature measures.

## The geometry behind the poor automatic yield

At 3 m/s and ~20 Hz the shear wavelength is of order 150 mm, while the M-lines are 19-43 mm. The
fit therefore works on a small fraction of a wavelength, where a genuine wave is nearly spatially
uniform and a slope is poorly constrained. The same fact explains three other observations:

* subtracting the per-time spatial mean (the standard bulk-motion remedy) removes the signal along
  with the motion - hence `remove_flat=False` in the passive path;
* short segments give **high semblance with meaningless slopes** (see the half-line results in
  `docs/passive_mlines.md`), so semblance must never be used alone to pick a window;
* about 20 % of all automatic fits rail at the 1.0 / 20.0 m/s search bounds, which means "no front
  found", not a measurement.


## Why a 2-D field estimator does not rescue this (2026-09-22)

Two direction-resolved estimators were built and tested against the 34 hand-drawn panels, on the
reasoning that a 1-D line cannot separate speed from direction (apparent speed is `c/cos(theta)`,
so every 1-D estimate is biased high). **Both failed on real data.** Full record:
`docs/field_estimator_plan.md`.

* the **structure tensor** recovers a synthetic plane wave to 0.7 % and then locks onto axial
  speckle on real data, reporting 0.2-0.7 m/s travelling into depth;
* the **phase-gradient** estimator survives speckle (4.8 % at p90, to 6 dB SNR) but fails its own
  internal check on real data - displacement and velocity are the same field times `i*omega`, so
  they must give identical `k`, and they differ by a median 1.25x in speed and 12.4 deg in
  direction.

**The unifying reason is geometric, and it is the same one that limits the 1-D path.** The shear
wavelength is 5-10x the imageable region (150-290 mm against 19-43 mm of M-line or ~40 mm of ROI).
That single fact defeats the slant stack, `remove_flat`, the structure tensor, the phase gradient
and every clutter filter tried, each for the same reason: what is nearly uniform across the
aperture is *the wave*. No estimator fixes it. What would: a longer aperture, a higher-frequency
wave, or clutter suppression at acquisition.

**Practical consequence: the hand-drawn slope with a stated +/-25 % uncertainty remains the
defensible measurement for this data.**
