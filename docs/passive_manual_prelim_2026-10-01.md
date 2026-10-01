# Manual passive study, re-read with the window review: preliminary results (2026-10-01)

This covers the 27 folders first read with the energy detector (2026-09-25 to 09-29), re-read with
the valves detector and the window review ([passive_manual.md](passive_manual.md)), plus three new
folders. It also covers the "two slopes" pattern and the 2D wave maps.

The state counted here is the export of 2026-10-01:
- 30 folders;
- 86 scored windows: 51 MVC, 31 AVC, 4 AK.

## 1. How the reading is doing

Script: `study/analysis/passive_valves_prelim.py`. Output: `study/logs/passive_manual_valves_prelim/`
and `study/montages/passive_manual_valves_prelim/summary.png`.

**Window choice was the main problem, and the review fixes it for AVC.** Same folders, old vs
new:

| | usable (confidence ≥ 2) | clear (3) | scored 0 |
|---|---|---|---|
| AVC, valves + review | **84 %** (31) | 58 % | 1 |
| AVC, energy detector | 44 % (34) | 29 % | 16 |
| MVC, valves + review | 80 % (51) | 37 % | 0 |
| MVC, energy detector | 85 % (39) | 59 % | 3 |

- **MVC is now scored more strictly.** Of the events found in both readings (48), 31 kept their
  confidence, 13 went down (mostly MVC 3 → 2) and 4 went up. Where both readings are usable (40),
  the speeds agree: new/old median 1.00, |difference| median 0.47 m/s.
- **Window review.** Of the 77 proposed windows (your ROIs, as 120 ms windows), 74 were kept, 1
  moved and 2 deleted; 12 were added.
- **The fully automatic windows would have done almost as well.** In the folders with an ECG
  they cover the accepted MVC / AVC window by ≥ 80 % in 71 of 73. The 10 they miss are all in the
  four folders without a usable ECG (C05, C07, C12, C14), where the events were placed by hand.

**Speeds (confidence ≥ 2):**
- MVC: median 3.4 m/s (IQR 2.7-5.6, n = 41).
- AVC: median 4.9 m/s (IQR 3.8-6.3, n = 26).
- AVC (end-systole) is faster than MVC (end-diastole), the expected direction.
- 9 AVC and 6 MVC are ≥ 6 m/s. There a hand slope is only a lower bound.
- MVC beat to beat (two usable MVCs in the same folder, 14 folders): |difference| / mean median
  17 %. It was 27 % on 2026-09-29.

**Automatic vs hand speed:** median ratio 1.00 (velocity gauss), 1.09 (Keijzer), 0.94
(displacement). **Caveat: anchoring.** 39 of 85 slopes were accepted at exactly the automatic
starting tilt of the slider, including 26 of the 39 scored 3. On these the agreement holds by
construction. Section 3 gives an independent check for the clear (3) events.

## 2. Two slopes: an in-phase basal segment

Found on C000000011 event 1 (MVC, R+22 ms):
- the basal ~13 mm of the line show a vertical band;
- the rest a slope of ~2 m/s.

Cross-correlation against the line's first 5 mm (`st_win0.npz`) shows what happens:
- **r 0-12 mm:** delay 0 ms, correlation 0.93-1.00. This part moves **in phase**.
- **r ≈ 14 mm:** the delay jumps to 13-17 ms (velocity views).
- **Further on:** the delay grows to ~25 ms at r = 47 mm, a travelling wave of 1.7-2.4 m/s.
- **The far end moves against the base.** At zero lag it is anti-correlated with the base (−0.4
  to −0.75).

**The line's geometry is excluded:** it bends only ~17°, which changes an apparent speed by
< 10 %. The 2D arrival map (section 3) shows that the in-phase region is a **2D block**. It covers
the whole basal region, ~22 mm along the line and the tissue below it, so it is rigid motion of
the base at valve closure, with the shear wave running on from where it ends. A partly standing
wave (reflection at the annulus) cannot be excluded for the anti-correlated far end.

**How common.** An in-phase segment ≥ 10 mm from r = 0 (velocity gauss, zero-lag correlation
≥ 0.9, |lag| ≤ 1.5 ms) occurs in 26 of 86 windows: 13 of 51 MVC, 12 of 31 AVC. It does not
change the median automatic/hand ratio (1.00 either way), but anchoring confounds that comparison.

**For reading:** fit the sloped (distal) part. The vertical part carries no speed. The automatic
straight-line fit over the whole line is pulled up by it.

## 3. 2D wave maps

Script: `study/analysis/passive_wave_map.py` (`--batch`, `--no-video`), summary
`passive_wave_map_summary.py`. Output: `study/montages/passive_wave_map/` and
`study/logs/passive_wave_map/`.

**What it does.** The same processing as the "velocity gauss" space-time is run on a box
(± 12 mm) around the event's M-line. The full 2D axial velocity field is kept. From it come:
- **a slow-motion video** (×~60): the velocity overlaid on the buffer-4 B-mode, with the M-line, a
  marker where the hand slope puts the wavefront, and the space-time with a time cursor;
- **an arrival-time map:** for every pixel, the lag of maximum cross-correlation with the line's
  first 5 mm, shown as isochrones;
- **a plane fit:** the 2D speed and direction;
- **the origin:** the earliest 2 % of the reliable pixels.

Only the demo video is committed: `C000000027_MVC_w2.mp4` and its still. The other videos stay
local, ~12 MB per event.

**Run on** all 39 events scored 3, plus the C11 two-slope event. 40 maps, none failed.

- **The 2D map confirms the hand slopes independently of the slider.** The arrival-time slope
  along the same line, from cross-correlation, against the hand slope: median ratio **1.03** (IQR
  0.96-1.21, n = 40). The four outliers have a near-flat arrival (in-phase dominated). So for the
  clear events the hand speeds are not an anchoring artefact.
- **Every wave travels from the base towards the apex** (none towards r = 0).
- **Every wave first appears at or beyond the basal end of the line**, a median 4-6 mm off it.
  That points to a source at the base / annulus, not on the drawn septal line. The box is
  ± 12 mm, so the true source can lie further out.
- **2D speed** (plane fit, R² ≥ 0.5 in 34 of 40, median R² 0.85):
  - MVC median 3.2 m/s (hand 3.2);
  - AVC median 4.5 m/s (hand 4.3);
  - 2D / hand median 0.97.
- **The fitted direction is not a reliable obliquity measure.** It lies a median 33° from the line,
  but the speeds do not follow the c / cos θ relation such an angle implies. A single plane wave
  over a ±6 mm band around the line is too crude: wall thickness, curved fronts near the source,
  and the in-phase block. **Do not correct along-line speeds for angle on this basis.**
- **C11 (two slopes).** The arrival is ~0 ms over the whole basal region, then rises along the
  distal septum (on the line: 2.15 m/s overall, hand 2.09 m/s). The origin is inside the block,
  6 mm off the line.

## 4. Changes made with these results

- **M-line editor (2026-10-01).** All lines are now drawn on buffer 4, with buffers 1 and 3 only
  to read the anatomy. A line drawn on buffer 4 is accepted with **one ENTER**: no registration, no
  review, because it only moved the display lines on buffers 1 / 3. Lines drawn on buffer 1 or 3
  still go through the registration review.

## 5. Open points for the next session

1. **Anchoring.** Start the slope slider at a random tilt for some windows (blinded), or hide it,
   to measure the anchoring effect on the confidence-2 / 1 slopes. Section 3 covers only the 3s.
2. **Fast speeds.** Flag hand slopes ≥ 6 m/s (line crossing in < 5 frames) as lower bounds in the
   export.
3. **In-phase basal block.** Decide whether to report its length per window, and whether the
   automatic speed should skip it, for example by fitting only beyond the in-phase segment.
4. **Wave maps for confidence 2.** Would the arrival-time slope also back up the plausible (2)
   events?
