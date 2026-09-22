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

**The common factor is geometric.** The shear wavelength is 5-10x the imageable region
(150-290 mm against 19-43 mm of M-line or ~40 mm of ROI), so what is nearly uniform across the
aperture is *the wave itself*. That is why the slant stack, `remove_flat`, the structure tensor,
the phase gradient and every clutter filter tried all fail in the same way.

**Do not over-read that as "the acquisition cannot work".** Keijzer et al. obtained usable medians
with a significant AVC > MVC difference from M-mode Radon on a comparable geometry, and Strachinaru
and Salles likewise report speeds from similar apertures. The honest statement is that this
geometry makes the **per-window** uncertainty large - which is consistent both with the +/-25 %
measured here and with the literature's approach of reporting medians over many sequences, on
open-chest pigs or selected volunteers with far better SNR than this cohort. What is not supported
is a per-window automatic number.

**Practical consequence: the hand-drawn slope with a stated uncertainty remains the defensible
measurement for this data**, and group statistics over many events are more likely to be
meaningful than any single window.


## The manual measurements have their own uncertainty

Everything above benchmarks automatic estimators *against the hand-drawn slopes*, which makes it
easy to read the hand values as ground truth. They are not. Two separate sources of error:

**Drawing precision, ~+/-25 %.** Measured, not assumed: on a blind redraw the AVC pair reproduced
to 0.6 % and MVC displacement to 8 %, while an independent ridge-tracking fit of the same panels
landed 25 % away *with an identical on-wave score*. Two clicks over a ~5 ms moveout is not a
precise instrument.

**Whether there was a wavefront to draw at all.** This is the larger and less tractable one. The
propagation is often unclear and sometimes not visible, and the operator still has to put a line
somewhere. Concretely, from the 15-window labelled set:

* of the 66 labelled valve windows in the study, **not one** has all three processing views
  agreeing on a speed - the set contains no easy cases because there are none;
* 6 of 15 windows fail the displacement-vs-velocity cross-check, and two that pass it do so on
  physically impossible values (0.45 m/s travelling backwards, 0.91 m/s);
* on the MVC velocity panel of C000000023 the tracked ridge propagates only over the first
  ~14 mm and is stationary beyond - the panel does not contain a single wavefront to draw.

**C000000023 is the best case for the manual measurement too, not just for the automatic one.** It
was selected for image quality and consistency, and four separate conclusions drawn from it alone
did not survive the labelled set. Its hand values are the most reliable in the cohort and should
not be taken as typical.

**Consequence for how these numbers are used.** A per-window hand value carries at least the
+/-25 % drawing precision, plus a term for panel quality that can be much larger where the
wavefront is faint. That term has now been measured.

### Panel confidence: the estimators work where a wave is visible

All 30 hand-drawn panels were scored for whether a wavefront was actually visible
(`study/analysis/score_panels.py` -> `study/logs/panel_confidence.csv`): **13 clear, 10 plausible,
4 guess, 3 none**. Estimator error tracks that score monotonically.

| panel confidence | n | automatic unusable | automatic median bias | field-estimator median error |
|---|---|---|---|---|
| clear     | 13 |  8 % | **+14 %** |  32 % |
| plausible | 10 | 20 % | +35 % |  33 % |
| guess     |  4 | 25 % | +59 % |  69 % |
| none      |  3 | 67 % | **+355 %** | 142 % |

**This reframes the headline numbers.** The study-wide "+24 % median bias, 20 % of fits unusable"
is dominated by panels where there was nothing to measure. Where a wavefront is clearly visible
the automatic fit is within **14 %** and fails outright in 1 case of 13. The catastrophic errors
(+389 %, direction flips, railing at a search bound) are concentrated almost entirely in the
`guess` and `none` panels.

So the yield problem is substantially about **event quality, not algorithms** - which is also
consistent with the literature obtaining usable results from comparable acquisitions by reporting
medians over many sequences on better-SNR data. The practical route is to screen panels for a
visible wavefront first and report only those, rather than to keep rebuilding the estimator.

**One check does NOT track visibility**: displacement-vs-velocity agreement is 1.34x on clear
panels but 2.72x / 1.52x / 1.69x on the rest, with only 2-6 windows per cell. On this evidence it
is a useful rejection rule but not a proxy for whether a wave was visible.

**Caveat on how the scores were collected.** The scoring tool displayed the hand and automatic
speeds in each panel title, so the scoring was **not blind** and anchoring cannot be excluded. The
effect is large and monotonic across four levels, which is hard to produce by anchoring alone, but
a blinded re-scoring (hide the numbers) would make this result solid rather than suggestive. That
is a one-line change to `score_panels.py` and an open action.

## Two ways to draw the line

`study/analysis/manual_slope.py --mode clicks` (default) is the original: click two points on the
wavefront, speed is `dr/dt` between them.

`--mode slider` anchors the line with **one** click and sets the slope with a slider. It separates
the two judgements the operator is actually making - *where* the wavefront is, and *how steep* it
is - so a slip in one does not corrupt the other, and it makes the sensitivity visible: if a wide
range of speeds looks equally good on a panel, that is information about the panel rather than a
failure to click accurately. Arrow keys nudge the slope in 0.05 m/s steps.

Both store the same format (two points on the line plus the speed, now with a `method` field), so
everything downstream reads them identically and the two-click method remains available
unchanged.
