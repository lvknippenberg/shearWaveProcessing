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

### 2. In vivo: a zero-amplitude-push acquisition (a true no-push control)

**What.** The active sequence exactly as used - same R-peak gating, 20 pushes/s, 39 reference
frames, the ~1 ms push interval, 59 tracking frames, same TX/receive and TPC settings - but with
the push transmitting nothing. On the Verasonics: set the push `TX.Apod` to zeros, so the push
*event* and its `timeToNextAcq` stay in the sequence (check it is not skipped when the apodisation
is all zero). This is the control Bouchard et al. (2009) used.

**Why it gives more than the pre-push frames** (the split-reference control used so far):

1. *It crosses the push boundary.* In vivo the reference and tracking blocks decorrelate across
   the push (speckle correlation ~0.8 vs ~0.95 within a block; none in the phantom), the echo is
   4-10 % weaker after it, and tracking frame 0 is corrupted. Frames inside the reference block
   never see that boundary, so they cannot say whether the push (tissue/probe motion) or the
   sequence (timing, supply sag, hardware settling) causes it. Zero push: decorrelation still
   there -> sequence; gone -> the push. This decides whether reference-relative methods can work
   in vivo at all, and it is what created the false 1.28x "push effect" in the continuous-record
   test, which needed an improvised gap-matched control to remove.
2. *It is the real window.* The split reference is ~8 ms after an 8-frame reference, so the push
   window has to be cut and processed to match. A zero-push window is the full ~16 ms tracking
   window after the full reference at the same cardiac phase: the unchanged production recipe
   runs on it, giving a genuine null distribution for every measure - amplitude, coherence, the
   V-detector, and hand reading (push and zero-push panels can be mixed in a blind scoring).
3. *Limitation:* it is another heartbeat. Pair it phase-matched (R-peak gated, push k vs push k),
   or better, alternate push on / off within one acquisition if the sequence allows.

**Analysis.** `study/analysis/push_gap_check.py` (cross-push decorrelation) and
`scripts/invivo_recipe_contrast.py` with the zero-push acquisition as the control dataset.

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

## Decisions (code is ready; defaults change only when you decide)

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

11. **Default passive pipeline - DECIDED 2026-09-24, done.** `configs/passive.yaml` now uses the
   part-2 default: velocity, 15-150 Hz, Gaussian 0.6 x 1.2 mm, moving mean 3, 5 M-lines x 0.5 mm,
   no directional filter, no SVD, no CFWI; further tuning did not improve the panels. The "medium"
   setting (Gauss 1.0 x 2.0, mean 5, 9 lines) over-smooths. Because 0.6 x 1.2 mm may already
   steepen the front, the three views are now default / unsmoothed / median 1.0 x 2.0 mm (same
   recipe otherwise) - use them side by side for manual speeds. The small effect of temporal
   smoothing and M-line count was checked (`study/analysis/smoothing_effect_check.py`): both are
   applied and scale with strength (mean 9: 27 % RMS change, 15 lines x 0.8 mm: 50 %), but the
   signal is slow (f50 29 Hz, f90 66 Hz; mean 3 passes 94 % at f90) and varies little across the
   line. `tests/test_passive_default.py` pins the recipe. The old config is frozen as
   `configs/passive_v1.yaml` (the scored panels and manual slopes are keyed by its view names;
   `score_panels.py` / `field_stage4.py` use it). Reprocess the study with the new default (item 14).

12. **Which image to draw passive M-lines on.** Part 3 (`report/passive_methods/passive_methods_v3.pdf`):
   buffer-3 frames are 2-3 beats before buffer 4 and the heart has moved a median 3.1 mm (buffer 1:
   0.8 mm); the buffers themselves agree to < 35 um (phantom). The redrawn buffer-3 lines are not
   better. Options: keep buffer-1 lines; or shift each buffer-3 line by its measured buffer-3 ->
   buffer-4 anatomy offset (`study/analysis/mline_difference_check.py` computes it) and re-test.
   **Done (mapping):** `swp.mline.transfer.transfer_line` moves a line by the local translation
   between the frame it was drawn on and buffer 4 at the event, with two checks (ensemble agreement
   and recovery of known shifts). Phantom: <= 0.3 mm error. In vivo
   (`study/analysis/map_mlines_b3_to_b4.py`): 7/15 mappings reliable (motion 0.4-6 mm); rotation
   is not identifiable in vivo and is left out. On the reliable 7, the mapped line scores higher
   than the unmapped buffer-3 line in 6/7, but higher than the old buffer-1 line in only 2/7.
   Conclusion: mapping corrects the motion, but a buffer-3 line is still not better than a
   buffer-1 line drawn 1-2 beats from the event. Keep buffer-1 lines for the existing data.
   **For new acquisitions:** acquire the focused buffer directly before/after buffer 4 (same or
   adjacent beat) so the M-line image and the passive data share the heart position.

## Runs (commands ready; not run because they rewrite study outputs or take hours)

13. Correct the reference timestamps of the study's buffer-2 files (only the timing arrays change;
    the originals are kept as `custom/t_reference_v0`):
    `python scripts/retrofit_push_gap.py --root "Z:/raw_data"` (dry run), then `--apply`.
14. Reprocess the study with the new passive default (~4 min/folder, ~2.5 h) - RUNNING 2026-09-24:
    `python scripts/passive_study.py reprocess --root "Z:/raw_data"` - rewrites each montage,
    `passive_speeds.json` and `passive_speeds_by_quantity.json` on the EXISTING windows and lines
    (no detection). The v1 (displacement) outputs are kept in `swp_passive/v1_displacement/`.
    **Note:** the stored windows were detected in energy mode, before `detect_mode` entered the
    cache key, so any re-detection (`process`, `reprocess --redetect`) now stops with
    StaleWindowsError in folders with hand-drawn lines rather than archiving them. The first run
    of this reprocess (before that guard) re-detected C000000001-4 and archived their lines;
    they were restored with `scripts/restore_redetected_windows.py` (all 14 windows reproduced to
    0.1 ms and checked against the v1 record). Moving the study to phase-mode windows still needs
    the per-event lines redrawn.
