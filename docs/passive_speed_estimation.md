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
displacement against velocity: where they agree the result is credible (AVC above: 3.58 vs
3.80 m/s, 6 %), where they disagree by a factor of two or more the window should not be reported
(MVC above: 2.67 vs 0.94 m/s).

## Speeds differ between cardiac events - that is expected

MVC occurs at end-diastole with the myocardium relaxed; AVC at end-systole with it contracted and
therefore stiffer. **A higher speed at AVC than at MVC is the expected ordering**, and that is what
C000000023 gives: 2.67 m/s at MVC against 3.58 m/s at AVC in displacement, a ratio of about 1.3.

Do not pool events. A single "shear-wave speed" for a subject is not a meaningful quantity without
stating the cardiac phase it was measured at.

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
