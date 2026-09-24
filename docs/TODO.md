# Open actions for Luuk

Things that need a person, an acquisition or a decision - code-only work is tracked in the docs
where it was found. Added 2026-09-24 (branch `review-followup`); findings behind each item are in
[review_followup_2026-09-24.md](review_followup_2026-09-24.md).

## Measurements

### 1. Calibrated phantom: speed accuracy (the pipeline has never been checked against a known speed)

Every speed so far is compared with hand picks or with other estimators, never with a known
value; the +/-25 % we quote is *repeatability*, not accuracy. Measure a phantom with certified
stiffness.

**Orientation: parallel to the tube (long axis) as the main set, transverse as a secondary set.**

* Parallel reproduces the septal geometry in PLAX: the stiff structure runs along the image, the
  push enters perpendicular to it, and the wave travels *along* a finite-thickness guide - exactly
  what our M-line samples. It also gives a long propagation path (the whole field of view) instead
  of one tube diameter, which is what the slope fit needs: our in-vivo lines are 19-43 mm and the
  short ones are the unreliable ones.
* Transverse gives a circular cross-section: the path inside the tube is at most one diameter and
  the wave leaves into the background within a few mm, so it tests boundary/guided-wave effects,
  not speed accuracy. Worth a small set to see how much the tube wall changes the apparent speed.
* In both, also measure the **homogeneous background** (away from the tube): that is the cleanest
  ground truth for a bulk shear wave, and the difference tube-vs-background is the guided-wave bias
  we expect in the septum (Lamb-type dispersion, see `docs/passive_speed_estimation.md`).

Protocol suggestion: the in-vivo push (41 el / 1500 cyc / 30 V) and the recommended one
(61 el / 1900 cyc), 10 pushes each, push focus on the tube axis (parallel) or tube centre
(transverse), at >= 2 certified stiffnesses. Process with `configs/active.yaml --phantom` and
with the literature recipe (`scripts/invivo_recipe_contrast.py` recipes); read speeds by hand
**and** automatically. Deliverable: bias and spread per estimator against the certified value -
this also settles the automatic-estimator biases found in the synthetic tests (item 9).

### 2. In vivo: a zero-amplitude-push acquisition

The reference and tracking blocks decorrelate across the push in vivo (speckle correlation ~0.8
vs ~0.95 within a block; none in the phantom) and we could not tell whether the push itself or
the sequence causes it. Bouchard et al. (2009) repeat the sequence with the push amplitude set to
zero. One such acquisition (same settings, push TX voltage/elements off) answers it: decorrelation
still there -> sequence/timing; gone -> the push disturbs the tissue/probe. Analyse with
`study/analysis/push_gap_check.py`.

### 3. Safety before any stronger push

The safety notes treat I_sppa.3 as binding (~24 V at 61 elements) while the in-vivo data were
taken at 30-40 V with 41 elements. Confirm which limit the protocol applies (FDA Track 3 allows
MI <= 1.9 *or* I_sppa.3 <= 190 W/cm^2) and the outstanding transient-TI check before planning a
stronger push. (Unchanged from the review; listed so it is not lost.)

## Observers and scoring

### 4. Second observer

Everything hand-drawn so far is one observer. Caenen et al. used 2 observers x 5 M-lines per
acquisition; Keijzer et al. report intra/inter-observer and test-retest variability. Needed:
someone to redraw, blind to the first set,
* the passive M-lines + hand slopes on the 15 labelled windows (`study/analysis/manual_slope.py`,
  `--mode slider`), and
* the active M-lines on one in-vivo acquisition (`scripts/draw_invivo_mlines.py`).
Deliverable: inter-observer spread of the speed (to set against the +/-25 % intra-observer figure).

### 5. Blind re-scoring of the panel confidence

`python study/analysis/score_panels.py` is now blind (no speeds, no hand line, no subject/label,
seeded random order) and writes `study/logs/panel_confidence_blind.csv`. After scoring, run
`python study/analysis/score_panels.py --compare` - it reports agreement with the earlier
non-blind scores and whether the "estimators work where a wave is visible" trend survives.

### 6. Look at the 7 candidate ARF pushes from the study screen

`study/montages/study_active_candidates.png`. My read: C000000026 m18 and C000000024 m14 are the
only ones with an outward, tilted pattern starting at r0; the others are vertical stripes, a blob
or off-origin. If you agree that some are real, those pushes (all late diastole) are the best
evidence the push can work in vivo - redraw their M-lines by hand before reading a speed.

### 7. Caenen comparison, like for like

Our stored speeds match Caenen's per-push medians (ratio 0.93-0.97, 70-80 % within 25 %) but
theirs are hand-drawn and ours automatic. Draw hand slopes on the same 11 pushes (1, 5, 15, 18,
20-25, 27) in the GUI speed tool, and ask Caenen (a) whether their push number is the
`..._ARF_<n>.mat` index, (b) what the five colours are (observers or M-lines).

### 8. Explore active-SWE recipes with manual M-lines and manual speed fitting

The active recipe was tuned on metrics that later proved unreliable (`origin_coherence` latches
onto cardiac motion; `push_specificity` was retired), and the automatic speed is biased
(item 9). Re-do the active recipe choice the way the passive work ended up: **hand-drawn
M-lines and hand-fitted speeds** as the reference, judged by eye.

* **Recipes to compare** (all exist in `scripts/invivo_recipe_contrast.py` / the GUI): the current
  view A (reference-relative displacement, 120-700 Hz), frame-to-frame velocity with the same
  filters, the Caenen recipe (velocity, Gaussian 1.9 x 2.0 mm on the autocorrelation,
  75-750 Hz), with and without the outward directional filter; displacement vs velocity vs
  acceleration; band corners 50/75/120 Hz.
* **Datasets:** Caenen pig (positive control, 11 pushes with their reported speeds - compare
  directly), CIRS phantom (voltage sweep, and the calibrated phantom of item 1 once measured),
  and several in-vivo sets (2026-08-04 30/40 V, 2026-08-18 41/61 el, the 7 study candidates of
  item 6 plus a few study subjects at diastasis).
* **Per push:** draw the M-line (`scripts/draw_invivo_mlines.py`, Caenen: `SWE_results/batch_draw.py`),
  render every recipe side by side, fit the wavefront by hand on each (GUI speed tool or
  `study/analysis/manual_slope.py --mode slider`), and note whether a V is visible at all.
* **Deliverable:** per recipe, visibility score and hand speed vs Caenen's values / the certified
  phantom value; pick the active view A from that, not from an automatic metric. A second
  observer (item 4) on a subset makes the comparison defensible.

## Decisions (code is ready, defaults were deliberately not changed)

9. **Automatic active speed.** On a synthetic wave with known speed (`tests/test_pipeline.py`):
   the estimators are exact, but the outward directional filter biases speed +5 to +37 %, the
   `radon` slant stack +20 to +94 %, and the 75-750 Hz band-pass makes it erratic (0.73-2.09x).
   `configs/active.yaml` uses `ttp_ransac` + directional. Options: switch the *speed* to
   `tof_xcorr` on the non-directional space-time (exact in the tests), and/or fix the directional
   filter's Tukey taper (it is never undone, and removes ~45 % of a clean wave's energy - at the
   record start, where the wave leaves r0). Either changes outputs, so it is your call.
10. **Recipe for active SWE.** The literature recipe (frame-to-frame velocity, 75-750 Hz) separates
   a real push from its control ~8x better than the current one on Caenen's data (11.7x vs 1.47x).
   If the calibrated phantom (item 1) and item 8 agree, adopt it as view A in `configs/active.yaml`.

## Runs (commands ready; not run because they rewrite study outputs or take hours)

11. Correct the reference timestamps of the study's buffer-2 files (only the timing arrays change;
    the originals are kept as `custom/t_reference_v0`):
    `python scripts/retrofit_push_gap.py --root "Z:/raw_data"` (dry run), then `--apply`.
12. Per-quantity passive speeds for the whole study (~4.5 min/folder, ~3 h):
    `python scripts/passive_study.py reprocess --root "Z:/raw_data"` - writes
    `output/swp_passive/passive_speeds_by_quantity.json` next to each montage.
