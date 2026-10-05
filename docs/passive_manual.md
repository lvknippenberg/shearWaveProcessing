# Manual passive reading of the whole study

`scripts/passive_manual.py` walks through every SW_data folder under `Z:/raw_data`: M-lines drawn
with buffers 1, 3 and 4 side by side, and hand slopes drawn once and mirrored on five processing
views. It is built for a long, interrupted effort (724 folders, ~4 events each). Every answer is on
disk the moment it is given, and a new session resumes where the last one stopped. Started
2026-09-25. It supersedes `passive_study.py draw / draw-events` and `study/analysis/manual_slope.py`
for this study, which keep working on their own outputs.

## Running it

```
python scripts/passive_manual.py session          # the interactive work (default root Z:/raw_data)
python scripts/passive_manual.py status [-v]      # where every folder stands
python scripts/passive_manual.py export           # -> study/logs/passive_manual_slopes.csv
python scripts/passive_manual.py archive --tag <why>   # move all manual outputs aside, start over
python scripts/passive_manual.py redetect         # move folders of the old detector onto the valves detector
```

Use the `envs\zea_latest` python. `session` starts two background workers, logged to
`study/logs/passive_manual_worker*.log`, and stops them on exit. They run the window detection
once a general line exists, and the five space-times once an event line exists. With
`--workers 0`, run `passive_manual.py worker --watch` yourself instead, for example on another
machine.

- **Stop:** `q`, or close the window. Everything accepted is on disk.
- **Go back:** `b` re-opens the previous prompt with your answer pre-loaded.
- **Choose what to work on:** `--task lines | windows | events | slopes | lines+events` limits the
  session to one kind of prompt. The default `auto` finishes folders first: slopes, then event
  lines, then window reviews, then general lines, each in queue order.
- **Revisit later:**
  - one folder: `--folder <f> --redo general`, `--redo review`, or `--redo event --window i` /
    `--redo slope --window i`;
  - everything you skipped: `--retry-skipped`. Measurements excluded with `v` (not PLAX) are
    not offered again; `--folder <f> --redo general` reopens one. Folders the view review labels
    PSAX / Apical: `--all-views`;
  - MVC lines reused without a prompt (step 4): `--redo event --window i`, or `--no-reuse` to
    switch reuse off;
  - windows the detector screened out (below the semblance threshold, see step 2):
    `--include-screened`.
- **Folders still being beamformed:** they are picked up automatically. A folder counts as ready
  once the batch has written its buffer-4 GIF.

**Queue order.** The 44 folders that already have September lines come first; their old lines are
pre-loaded, so ENTER accepts them. After that come the remaining folders, sorted. `Strain_data`
folders are not listed (no passive buffer).

**Preliminary results of the re-reading, the "two slopes" pattern and the 2D wave maps:**
[passive_manual_prelim_2026-10-01.md](passive_manual_prelim_2026-10-01.md).

**2026-10-06: evaluation, reader decisions, and changes to the session.**

[study/analysis/passive_manual_eval/REPORT_2026-10-05.md](../study/analysis/passive_manual_eval/REPORT_2026-10-05.md)
evaluates the reading so far (126 folders, 363 slopes). It covers reproducibility, M-lines, septal
thickness, event placement, slope fitting and the quality metric. `run_all.py` there repeats it on a
new snapshot when the reading is done.

Decisions taken on it:
- **AK is not part of the main reading.** It was not in the study plan.
  - AK windows need not be added in the window review.
  - The 32 already added stay as they are.
  - At the end of the study a separate AK pass will measure in how many subjects an AK wave is
    visible / quantifiable, and correlate it with the MVC/AVC speeds, BMI and condition.
  - Timing for that pass: the AK energy peak sits at 0.94 RR (~47 ms before the next R-peak), and
    the wave ~25 ms before that peak.
- **Scoring rule**, written down because the score drifted between days: on 10-01, comparable
  windows got ~0.25 point less than later. See "Scoring" under "Drawing the slope". In short:
  - high = clear propagation in all panels, ideally clean enough for an (automatic) slope fit on
    the acceleration panel;
  - strong but near-vertical bands show no propagation and score low;
  - slow waves (e.g. AK) are scored by the same rule.
- **Line pre-loads** (only proposals: every line is still shown and accepted with ENTER):
  - event lines start from the general line *registered* onto the event frame (AVC with rotation);
  - general lines of a subject's later acquisitions start from the previous acquisition's line,
    registered onto the new R-peak frame.

  Steps 1 and 4 below give the details.
- **Slope prompt:**
  - the slider starts at the velocity-median automatic speed (before: velocity gauss);
  - the first click sets the tilt automatically through the clicked point (`t` repeats it);
  - the status line gives the number of frames the line needs to cross the M-line.

Each change has a switch in `configs/passive_manual.yaml`, and the old behaviour is one line away:
- `events.preload.registered: false`;
- `general.preload.previous_acquisition: false`;
- `slope.init_view: "velocity gauss"`, `slope.auto_tilt: false`.

Each was also committed separately, so it can be reverted.

**2026-10-05: session crash fixed; reversed-looking waves.**
- **Crash.** A session reading quickly (general lines every ~8 s) died with
  `RuntimeError: main thread is not in main loop`. The Tk window then hung. Nothing was lost:
  every answer up to the crash was on disk. The cause and fix are under Implementation notes
  ("Tk and the prefetch thread"). The same `session` command resumes.
- **State after the crash:** done 65, need-review 9, need-general 227, not-plax 421; 182 slopes.
- **Reversed-looking waves.** Some windows show a dominant band that reaches the apical end
  first. See "Waves that seem to run backwards" under "Drawing the slope". Open decision: how to record them.

**2026-10-02: view filter, MVC line reuse and one exclusion.**
- **View filter.** Only folders the manual view review labels PLAX (or Unclear) are read. That
  leaves 303 of 724 folders; the other 421 have the stage `not-plax`.
- **MVC line reuse.** MVC lines near the R-peak reuse the general line (step 4).
- **Folders flagged as possibly PSAX.** Three already-read folders were flagged by EchoPrime and
  checked on the general-line prompt:
  - C000000001 12-18-31 is PSAX and was excluded with `v`. Its readings (3 slopes, all confidence 0)
    are in `archive_20261002_155832_general_excluded/`.
  - C000000033 08-43-33 and C000000035 10-54-57 are PLAX and kept their readings. The manual
    review agrees with all three.

**Re-read from 2026-10-01 with the window review.** The 27 folders read with the energy
detector (C000000001-31) were moved onto the valves detector with `redetect`. Their general lines
and ROIs stay. Windows, event lines, space-times and slopes were archived in each folder
(`archive_<time>_redetect_valves/`). The review proposes the ROIs marked by eye. Reference copy of
the old results: `study/logs/passive_manual_reference_2026-10-01_energy/` (README there).

**Restarted 2026-09-29 on unwrap version 2** (all 724 SW folders beamformed, buffer 3 unwrapped:
650 unwrapped + 21 chronological = timed; 53 ambiguous = chosen by anatomy). Nothing had been
drawn since the 09-25 archive. Quitting is safe at any point, including a hard kill: lines and
records are written atomically, the record last, and a worker lock left by a dead process on the
same machine is released immediately (another host's lock after 3 h).

## Per folder

1. **General M-line.** Buffers 1 | 3 | 4 at the R-peak, each at the frame nearest a logged R-peak
   (buffer 4: frame 0). This line is used to detect the valve events.
   - **Only PLAX is read (since 2026-10-02).** The SW protocol records about 6 PLAX and then
     about 9 PSAX acquisitions per subject, and only PLAX is used for passive SWE. The view of
     every SW loop was reviewed by hand (`study/logs/view_classification/sw_views_manual.csv`,
     label PLAX / PSAX / Apical / Unclear). Folders labelled PSAX or Apical get the stage
     `not-plax`. No prompt is asked for them, the worker leaves them alone, and nothing on disk
     changes. At the time of the review that was 421 of 724 folders. PLAX and Unclear folders are
     read, and the title shows the label (`view: PLAX (manual review)`). For a folder not in the
     review, the title shows the EchoPrime call from `all_sw_views.csv` as a hint. To read the
     filtered folders anyway: `session --all-views` (also passed to the workers it starts).
   - **Pre-load (since 2026-10-06).** Used when the folder has no September line. The proposal is
     the general line of the same subject's nearest acquisition that has one (earlier preferred).
     It is registered from that acquisition's buffer-4 R-peak frame onto this one
     (`swp.mline.transfer`; box margins 12/20/30 mm, shifts up to 25 mm, because the probe moved).
     - The title says which acquisition it came from, how far it moved, and "CHECK it" when the
       registration is not trusted. The registered line is proposed either way.
     - Evaluation: it lands 1.09 mm from the line drawn by hand, as close as the reader's own
       redraw (1.06 mm).
     - The first acquisition of a subject is drawn from scratch: an automatic septum finder was
       not reliable.
     - Off: `general.preload.previous_acquisition: false`.
   - **Not a PLAX view after all (e.g. Unclear)? Press `v`** to exclude the whole measurement.
   - `general.json` then records `skipped: true, excluded: "not PLAX"` and the view call. The
     folder's stage is `excluded`, not `skipped`, so `--retry-skipped` leaves it alone. If the
     folder already had windows, lines or slopes, they are archived
     (`archive_<time>_general_excluded/`).
2. *(worker)* **Detection, since 2026-10-01** (`detect.picker: valves`): the automatic MVC / AVC
   windows of `swp.passive_valves` (fixed 120 ms; see "Automatic MVC / AVC windows" below), and
   the whole-recording "velocity gauss" space-time along the general line (`general_st.npz`).
   ~25 s per folder. Atrial kicks are not detected (not analysed).
3. **Window review** (since 2026-10-01). The whole-recording space-time of the general line, with
   the R-peaks, the expected MVC / AVC search windows and the proposed 120 ms event windows. Move,
   add or delete them; ENTER accepts. These reviewed windows are the events of steps 4-6.
   - **Proposed**, first available: the current review (`--redo review`); your ROIs from
     `passive_roi.py` on the same general line, as 120 ms windows centred on them; otherwise the
     automatic windows above the semblance screen.
   - Screened automatic windows are shown dashed in their own lane, as hints. They are not
     proposed.
   - **No usable ECG:** no automatic windows and no expected windows. Click the events in by hand.
   - Each window's event time (`t_peak`, used to pick the buffer-1 / 3 frames) is the peak of the
     20 ms energy inside it.
   - Moving a window makes its event line (and with it the space-time and slope) stale.

   | mouse / key | |
   |---|---|
   | click / drag | place a 120 ms window centred there |
   | drag a window | move it |
   | right-click | delete the window under the mouse |
   | scroll, `0` | zoom time around the mouse, reset |
   | ↑ ↓ | contrast |
   | `l` | cycle the label (MVC, AVC, AK, other) |
   | `c` | clear all |
   | ENTER / `n` | accept the windows / accept "no event in this recording" |
   | `x` / `b` / `q` | skip / back / quit |

   *Before 2026-10-01* (`detect.picker: energy`, still the windows of folders detected then;
   `redetect` moves them over): up to 4 windows of 100 ms, labelled MVC / AVC / AK / other, used
   as detected, without review.
   - **Search windows** from the R-peaks, as in `configs/passive.yaml`: MVC R+0-150 ms, AVC
     QS2 ± 120 ms (Weissler QS2 from the heart rate).
   - **Time picked by displacement energy**, as before.
   - **Since 2026-09-29** (`swp.passive_screen`, `configs/passive_manual.yaml` `detect:`):
     - A search window less than half inside the recording is dropped (`min_inside`). The ~1 s
       recording holds one AVC, and the second one used to be "found" on the masked, zero-energy
       end.
     - Every picked window is scored by the slant-stack semblance of the default velocity view
       along the general line (`screen`). Below 0.3 (`screen_min`) it is kept but **screened**:
       no event line is asked for it (`--include-screened` asks anyway).
     - The whole score track is stored in `windows.json` (`screen_track`).
     - Energy top-ups to 4 windows stay on (`top_up`).
     - Picking the time by the semblance itself (`picker: semblance`) was tried and **rejected**:
       it chases near-vertical bands. The reasons and both dry runs are in
       [passive_manual_prelim_2026-09-29.md](passive_manual_prelim_2026-09-29.md).
   - **Folders detected before that** (the 27 read on 2026-09-29) keep their windows. Their
     `windows.json` key has no `min_inside`.
4. **Event lines** (one per reviewed window). Buffers 1 | 3 | 4 at the event's cardiac phase: buffer 4 at the event itself,
   buffers 1 and 3 at the frame with the same time since the preceding R-peak. The panel titles
   give the phase and the offset from the event. Pre-loaded, in order of preference:
   - the September line of that event, when an old window lies within 40 ms;
   - otherwise (since 2026-10-06) the general line **registered** onto the event: the buffer-4
     frame at the general line is registered onto the frame at the event, and the line moved with it.
     - AVC windows also search a rotation of ±20°: in systole the septum turns ~7° and the
       line's basal end moves ~8 mm along it.
     - The title says how far it moved, and "CHECK it" when the registration is not trusted.
     - Evaluation: the proposals land 0.8-1.0 mm (MVC / AK) and ~1.8 mm (AVC) from the line drawn
       by hand. The unchanged general line was 1.0-1.3 / 3.7 mm away.
     - Off (`events.preload.registered: false`): the general line as drawn, as before.

   The general line is also shown as a dashed magenta reference on buffer 4.

   **MVC near the R-peak: no prompt (since 2026-10-02).** An MVC window gets the general line
   automatically, with no prompt, when three conditions hold
   (`events.reuse_general` in `configs/passive_manual.yaml`; evidence in
   [passive_mvc_line_reuse.md](passive_mvc_line_reuse.md)):
   - it is at most R+50 ms (`max_phase_ms`);
   - its buffer-4 anatomy moved at most 1 mm across the general line (`max_perp_mm`) and at most
     3 mm in total (`max_shift_mm`) since the general line's frame. This is measured with the
     line-transfer registration, which must also pass its own checks;
   - the window has no answer yet.

   Otherwise the prompt appears as before, for example when the probe or breathing moved the
   septum between the beats. The terminal prints `event i: general line, reused without a
   prompt (MVC R+31 ms, anatomy moved 0.3 mm across the line, ...)`. `events.json` stores the
   registration as `auto_reuse`, and the export has `line_reused_general`. A prompted MVC keeps
   the check that failed as `reuse_check`. To look at a reused line, run `--redo event --window i`
   (that always opens the editor). `session --no-reuse` draws every event line.
5. *(worker)* **Five space-times** per event, over the window ± 20 ms.
6. **Slope** on the five views plus the buffer-4 B-mode.

### Drawing an M-line

- Draw on whichever panel shows the septum best: buffer 4 when readable (it is the processed data),
  otherwise buffer 1, otherwise buffer 3.
- Points can be clicked in any order; the first click is r = 0 (the yellow star).
- While drawing, the other panels show the same coordinates dashed.
- **Drawn on buffer 4** (the usual case since 2026-10-01: all lines are drawn on buffer 4, with
  buffers 1 and 3 only to read the anatomy): **one ENTER accepts.** The saved line is exactly the
  one drawn, so there is nothing to register and no review step.
- **Drawn on buffer 1 or 3:** the first ENTER registers the line onto buffer 4
  (`swp.mline.transfer`: anatomy-scale phase correlation in a box around the line, an ensemble of
  box sizes, and a known-shift check). This shows:
  - the moved lines on the other buffers (orange);
  - the line that will be saved on buffer 4 (green);
  - whether the registration is **trusted**: ensemble agreement ≥ 0.6 and known-shift error
    ≤ 1 mm.
- If it is not trusted, nudge the green line with the arrow keys (0.25 mm; shift: 1 mm), or press
  `i` to use the uncorrected coordinates. ENTER accepts.

| key | |
|---|---|
| click / drag / right-click | add / move / delete a point |
| ENTER | buffer 4: accept. Buffer 1 / 3: map and review; ENTER again accepts |
| `c` | clear the line (to draw on another buffer) |
| arrows, shift+arrows | nudge the green buffer-4 line by 0.25 / 1 mm (review) |
| `i` | motion correction on / off (review) |
| `z` | zoom all panels on the line ± 25 mm / full sector |
| `[` `]` | step the frame of the panel under the mouse (buffer 4: 5 frames) |
| `a` | buffer-4 averaging (9 frames ≈ 10 ms) on / off |
| `x` | skip: no usable septum. For a general line this skips the whole folder (`--retry-skipped` offers it again) |
| `v` | general line only: exclude the measurement, not a PLAX view (never offered again) |
| `b` / `q` | back / quit |

**How the buffer 1 / 3 frames are chosen (changed 2026-09-25, after the first folders).**

- **Buffer 3 had to be unwrapped first.** It is a circular live loop, and its stored frames were
  rotated at an unknown point, so every buffer-3 time and phase used before 2026-09-25 was wrong
  (see **[buffer3_unwrap.md](buffer3_unwrap.md)**). It showed up while drawing: on C000000001 and
  C000000005 the "R-peak" buffer-3 frame showed another cardiac phase.
  - **Unwrapped folders** (`custom/unwrap_status` `unwrapped` / `chronological` in the buffer-3 IQ):
    buffer 3 is chosen by timing, from the stored chronological frame-centre times.
  - **Ambiguous or not-yet-unwrapped folders:** buffer 3 is chosen by anatomy, as the frame that
    looks most like the buffer-1 frame. The panel title says which applies.
  - All buffers are now matched at their frame **centres**: the trigger fires on the first
    transmit, and the centre is ~5.7 ms (buffer 1), ~19.7 ms (buffer 3) or ~0.5 ms (buffer 4) later.
  - The lines drawn earlier on 2026-09-25 (C000000001-5) were archived
    (`passive_manual.py archive --tag buffer3_unwrap`) and are drawn again.
- **A phase buffer 3 does not hold (known limitation, left as is).** A 26-frame buffer 3 spans
  about 1.0 s. When the RR interval is longer than that, part of the cycle has no buffer-3 frame,
  and the nearest frame can be far off. Example: C000000002 10-13-12, logged RR about 1.3 s
  (44-52 bpm), buffer 3 covers R+333 to R+1319 only. For an MVC at R+75 the panel showed R+333
  ("+258 vs event"). The offset is in the panel title. Draw on buffer 4 or 1 then. The phase
  match does not wrap to the next R-peak. A fix was not made, because such slow logged rates are
  exceptional or a triggering error (2026-10-02).
- **No valid ECG.** When `swp.acquisition.rrcheck` rejects the R-peak record (for example
  C000000005: "unusable: sparse"), every "R-peak" / "R+x ms" is meaningless. The titles then say
  NO VALID ECG, and buffer 1 is chosen by anatomy match to buffer 4.
- **Every mirrored line carries a verdict** (lines drawn on buffer 1 / 3). In review, each
  non-source panel shows the registered line (orange = trusted, red dotted = not trusted, verdict
  in the title) and the uncorrected coordinates (dashed white). Lines drawn on buffer 4 are saved
  as drawn. Since 2026-10-01 they are accepted without this review, because it only moved the
  display lines on buffers 1 and 3.

**Why the registration.** Buffers 1 and 3 are recorded in other heartbeats than buffer 4. At the
same cardiac phase the heart still sits a median 0.8 mm (buffer 1) to 3.1 mm (buffer 3) away, and
up to ~12 mm (`report/passive_methods/passive_methods_v3.pdf`). On the phantom the buffers agree
to < 35 µm, so the offset is anatomy, not reconstruction. Rotation is not modelled because it is
not identifiable in vivo. On C000000023 the September buffer-1 R-peak line moves 5.6 mm and is
*not* trusted (agreement 0.33), while the per-event lines move 1.5-3.6 mm and are trusted. The
review step is therefore needed, not a formality.

### Drawing the slope

Top row:
- displacement, Gaussian (`passive_v1` view A: 10-150 Hz, Gauss 0.6 × 1.2 mm, mean 3);
- velocity, median (15-150 Hz, median 1 × 2 mm, mean 3);
- velocity, Gaussian (the default: 15-150 Hz, Gauss 0.6 × 1.2 mm, mean 3).

Bottom row:
- **Keijzer velocity**: IQ slow-time low-pass 250 Hz, Gaussian 4 × 3 mm autocorrelation kernel,
  15-100 Hz;
- **acceleration** (Petrescu / Santos): mean 3, 10-150 Hz, Gauss 0.6 × 1.2 mm, then the
  derivative;
- the buffer-4 B-mode of the box around the line, with the line and r = 0.

All views use 5 lines × 0.5 mm and no directional filter. The two literature recipes use
different filters from the top row, so a slope that also fits them is not an artefact of our own
filter choices. Definitions: `configs/passive_manual.yaml`, the same as the "literature" family of
`study/analysis/passive_methods_atlas.py`.

How to draw:
- Click one point on the wavefront in any panel, then tilt the line with the slider or the arrow
  keys. The line is the same on all five panels.
- Axes: x = time, y = distance along the line. Speed = dr/dt in m/s; positive means travelling
  away from r = 0.
- The slider starts at the automatic slant-stack speed of the **velocity median** view
  (`slope.init_view`; until 2026-10-06 velocity gauss, whose fit is ~18 % too fast on the slopes
  the reader corrected, against ~10 % for velocity median). It is 3 m/s if that fit rails. The
  number itself is not shown, so it does not anchor the reading.
- **Automatic tilt (since 2026-10-06, `slope.auto_tilt`).** The first click also sets the tilt:
  the straight line through the clicked point that stays inside one band of the clicked panel best
  (largest |mean signal| along the line, ±0.5 to ±12 m/s).
  - Click on the band you want to follow; on the acceleration panel that is the ridge, which is
    also the zero crossing of the velocity bands.
  - Further clicks only move the anchor and keep your tilt. `t` sets the automatic tilt again
    through the current anchor.
  - The status line says "(auto tilt)" while the tilt is untouched.
  - Evaluation: through the reader's own anchors, this tilt was the automatic fit closest to the
    hand slopes on clear windows (median 13 % from the hand speed vs 21 % for the old start).
- The status line also gives the **number of frames the line takes to cross the M-line**. Below 5 it
  adds "near-vertical, no resolved propagation": see "Scoring".
- `u` unlinks the displacement panel so it can get its own tilt. Displacement and velocity weight
  different frequencies of a dispersive wave; by hand, velocity came out faster in 87 % of windows.
  Linked, both are stored as the same speed.
- Accept by scoring how clearly a wavefront is visible:
  - `3` clear, `2` plausible, `1` guess, `0` none (no line needed);
  - `x` skips the window undecided.

The score is part of the measurement: automatic bias was +14 % on *clear* panels against +355 % on
*none* (`docs/passive_speed_estimation.md`).

#### Scoring (rule written down 2026-10-06)

The score is about **visible propagation**, not about how strong the band is:

| score | when |
|---|---|
| `3` clear | a propagating wavefront is clearly visible in all panels; ideally clean enough that a slope could be fitted automatically on the acceleration panel |
| `2` plausible | propagation visible, but not in every panel or not cleanly (noisy, two slopes, partly in-phase) |
| `1` guess | a slope can be guessed, but propagation is doubtful. Also: a strong band that is near-vertical (crosses the M-line in < 5 frames): no resolved propagation |
| `0` none | no wavefront; also a purely vertical (in-phase) band |

- **Strong but vertical bands show no shear-wave propagation and are scored low** (1 or 0),
  however strong they look. The crossing-frame count in the status line helps.
- **Slow waves** (e.g. AK at 1-1.5 m/s) are scored by the same rule: high when the propagation
  is clear.
- **Why written down:** the evaluation of 2026-10-05 found that the score drifted between reading
  days. On 10-01, comparable windows got ~0.25 point less than on 10-02 / 10-05, also within the
  same subjects. Some strong vertical bands were scored 2, others 1. Windows worth re-checking
  against this rule:
  `study/analysis/passive_manual_eval/results/20261005_1530/07_quality/outliers.csv` (contact sheets
  in `outlier_sheets/`). Re-score one with `--folder <f> --redo slope --window i`.

| key | |
|---|---|
| click | anchor the line (on the displacement panel: its own anchor when unlinked) |
| slider, ← → | ± 0.05 m/s |
| ↑ ↓ | ± 0.5 m/s |
| `t` | automatic tilt through the current anchor (the first click does this by itself) |
| `f` | flip the direction |
| `u` | unlink / re-link displacement |
| `r` | clear the anchor |
| `3` `2` `1` `0` | accept with confidence |
| `x` / `b` / `q` | skip / back / quit |

#### Waves that seem to run backwards (noted 2026-10-05)

**Direction is consistent.** r = 0 is the right-hand (basal) end of the line in all 79 general
lines checked on 2026-10-05. So a positive speed is base → apex in every folder, and a reversal
is not a line drawn the other way round.

**Example.** C000000008 10-37-16, "AK" at 712 ms. RR is 760 ms, so the event is ~63 ms before
the next R-peak. The reader's line (+2.98 m/s, base → apex) follows a faint early front. The
dominant displacement band reaches r = 20-45 mm at ~690-700 ms but r = 0-15 mm only at
~705-715 ms, so it looks apex → base. The velocity bands are almost vertical.

**Likely causes, most likely first.** These are hypotheses; none has been tested.
1. **The beam angle changes along the line.** Only the axial (along-beam) component is measured.
   On the example line the beam direction turns by ~42° (−16° at the apical end, +26° at the
   basal end). The beam picks up:
   - motion along the septum: ~0.13 of it at the apical end, ~0.76 at the basal end;
   - wall-normal motion (thinning / thickening): ~0.99 at the apical end, ~0.65 at the basal end.

   When these motions have different time courses (atrial contraction pulls the base toward the
   atrium while the wall stretches), the timing of the measured signal drifts along the line.
   Nothing propagates, yet an apparent slope of either sign appears. The in-phase basal block
   (rigid base motion) belongs to the same family.
2. **Hydraulic loading.** Atrial kick and valve closure raise LV pressure through the blood
   almost instantly (~1500 m/s). The whole septum is loaded together: vertical bands, with small
   timing differences from local thickness or stiffness.
3. **Ventricular activation.** Mechanical activation runs roughly apex → base. It matters little
   for late-diastolic AK windows, more for MVC windows (~R−17 to R+103 ms).
4. **Reflections or guided waves in the thin septum.** At 3 m/s a wave crosses a 49 mm line in
   ~16 ms. Forward and reflected waves overlap and give bands whose tilt is ambiguous.
5. **Near-vertical bands have no meaningful sign.** ±2 ms over 49 mm is already ±25 m/s. The
   ≥ 6 m/s lower-bound caveat applies to the sign as well.
6. **A wave arriving at an angle or from outside the image plane.** A front that reaches the
   apical end first gives a genuine negative slope. Probably rarer than the causes above.

**Reading so far (2026-10-05).** Of 190 accepted slopes, 2 are negative (both MVC, confidence
1). When the dominant band tilts apex → base, the reader has fitted a weaker base → apex front.
That is a selection bias, to be decided on purpose. **Decided 2026-10-06 for the vertical case:**
a strong band without resolved propagation is scored low (see "Scoring"). For a genuinely
reversed slope, these options remain:
- record reversed patterns as drawn (`f`) with a low score;
- add a separate "reversed / vertical" flag so these events can be counted.

A check of cause 1 is still open: does the reversed tilt follow the change in beam angle along
each line?

### Marking ROIs by eye (added 2026-10-01)

The detector's window choice is a major source of error: the energy pick and the semblance
screen often miss the wave or land on a near-vertical band. `scripts/passive_roi.py` lets you
mark the time windows worth investigating yourself, on the whole recording.

```
python scripts/passive_roi.py session                  # resumes; --all re-opens every folder
python scripts/passive_roi.py session --folder "<f>"   # one folder, its ROIs pre-loaded
python scripts/passive_roi.py status -v [--root Z:/raw_data]
python scripts/passive_roi.py export                   # -> study/logs/passive_rois.csv
```

- **Shown:** the R-peaks, the expected search windows (MVC R+0-150 ms, AVC QS2 ± 120 ms), and the
  "velocity gauss" space-time along the general line over the whole buffer-4 recording, on one
  colour scale (the top half of the `study/montages/passive_general/` figures). The 100 ms
  moving-RMS view is left out on purpose: it mostly amplifies noise.
- **Input:** the general-line cache `study/analysis/general_screen_cache/`. A folder is shown only
  when the cache's general-line hash equals the current `general.json`. To add folders with a
  general line that are not cached yet, or whose line was redrawn, run
  `study/analysis/passive_general_screen.py --any-general` (~50 s per folder).
- **Drawing:**
  - Drag left-right on the space-time: the horizontal line's time span is one ROI. Draw as many as needed.
  - Each ROI gets a suggested label: the expected window that covers most of it, otherwise
    `other`. Press `l` to change it.
  - The height at which you draw is kept as `r_mm`, for reference only.

| mouse / key | |
|---|---|
| drag | draw a ROI (≥ 5 ms) |
| drag an end / the middle | move that end / the whole ROI (on the space-time, also its height) |
| right-click | delete the ROI under the mouse |
| scroll, `0` | zoom time around the mouse, reset |
| ↑ ↓ | contrast: colour-scale percentile 90 … 99.95 (default 99.5) |
| `l` | cycle the label of the ROI under the mouse: MVC, AVC, AK, other |
| `c` | clear all ROIs |
| ENTER | accept (at least one ROI) |
| `n` | accept "nothing worth investigating" |
| `x` / `b` / `q` | skip undecided / back / quit |

**Saved:** `rois.json` in the manual dir. It holds:
- `status`: `done`, `none` or `skipped`;
- per ROI: `t0`, `t1` and `t_mid` in s on the buffer-4 clock, `duration_ms`, `label`,
  `suggested_label`, the expected window and its overlap, and the phase since the preceding
  R-peak;
- the colour scale, the cache file and provenance.

It is keyed to the general line's hash, so a redrawn general line asks again (and `--redo general`
archives it). `rois.png` is the snapshot, and every answer is appended to `log.jsonl`. These ROIs
do not feed the event-line / slope steps yet; they are a separate first pass.

### Automatic MVC / AVC windows (added 2026-10-01)

`swp.passive_valves.valve_windows` reproduces the hand-marked ROIs. Atrial kicks are left out on
purpose: they are not always visible and are not analysed.

1. **Search windows** come from the R-peaks: MVC R+0-150 ms, AVC QS2 ± 120 ms. A window is kept
   when at least half of it lies inside the recording.
2. **Burst:** the peak of the 20 ms smoothed along-line velocity energy, weighted by a Gaussian
   prior (σ 45 ms) around the typical timing: MVC at R + 43 ms, AVC 28 ms before the QS2 centre.
3. **Window:** a fixed **120 ms**, centred on the burst (MVC +6 ms, AVC −2 ms).
4. **Presence:** below a semblance of 0.3 the window is `screened`.

**Fit.** The method was fitted on the 27 folders marked on 2026-10-01 (66 MVC/AVC ROIs in the 23
folders with an ECG) and checked leave-one-subject-out
(`study/analysis/passive_roi_auto_features.py`, then `passive_roi_auto.py`):
- the window fully contains 24/24 AVC and 41/42 MVC ROIs;
- the centre error is a median of 3 ms (AVC);
- the 3 AVC search windows left empty are all screened, and no marked window is.

**Why the short-time energy.** Scores averaged over the whole window (100-120 ms energy or
semblance) did worse: only 58-63 % of AVC ROIs were fully inside. A fixed delay after the R-peak
also missed 16 % of the later-beat MVCs.

**Exceptions:**
- **C000000021:** an MVC ROI 187 ms after the R-peak, at the end of the search window, is missed.
- **C000000023:** the second of two adjacent MVC ROIs is not covered.
- **No usable ECG** (4 of 27 folders): no automatic windows. Mark these by hand; with
  `--click-ms 120` a click places a 120 ms window centred on it.

```
python scripts/passive_roi.py auto              # -> study/logs/passive_auto_windows.csv (+ agreement)
python scripts/passive_roi.py session --auto    # open with the automatic windows pre-drawn
python scripts/passive_roi.py session --click-ms 120   # semi-automatic: click = 120 ms window
```

With `--auto`, a folder without ROIs opens with the unscreened automatic windows already drawn.
ENTER accepts them; adjust, delete or add as usual. The proposals are stored in `rois.json`
(`auto_proposals`), so the changes made by hand can be measured later.
Figure: `study/montages/passive_roi_prelim/auto_vs_rois.png`.

## What is saved

Everything is saved in `<folder>/output/swp_passive_manual/`. The September outputs (`mlines/`,
`swp_passive/`) are only read. Each file has one writer, so the session and any number of workers
can run together:

| file | writer | content |
|---|---|---|
| `general.json`, `general_mline.npz`, `general.png` | session | line in buffer-4 coordinates; the buffer and points drawn, frames, registration, nudge, what was pre-loaded (since 2026-10-06 also the proposed points `preload_points4_mm` and their registration `preload_registration`) |
| `windows.json`, `passive_bursts.png`, `passive_full_spacetime.png` | worker | windows + labels + phases, ECG R-peak check, detection key (general-line hash), provenance; `needs_review` for the valves detector (no PNGs then) |
| `general_st.npz` | worker | the whole-recording "velocity gauss" space-time along the general line + R-peaks (valves detector) |
| `review.json`, `review.png` | session | the reviewed event windows + phases, what was proposed, keyed to the detected windows' hash |
| `events.json`, `event<i>_mline.npz`, `event<i>.png` | session | per-event line records, keyed to the (reviewed) windows' hash; since 2026-10-06 with `preload_points4_mm` / `preload_registration` (the proposal, so corrections can be measured) |
| `processed.json`, `st_win<i>.npz` | worker | the five space-times + automatic speeds, keyed to the line's hash |
| `slopes.json`, `slope<i>.png` | session | the slope(s), confidence, anchor view, automatic speeds for comparison, keyed to the space-times' hash; since 2026-10-06 `slider_init` (view, speed), and per line `auto_tilt` (speed, view of the last automatic tilt) and `crossing_frames` |
| `rois.json`, `rois.png` | `passive_roi.py` | time windows marked by eye on the whole-recording general-line space-time, keyed to the general line's hash |
| `log.jsonl` | session | every accept / skip, append-only |
| `worker_errors.json` | worker | a failed detection / processing (reported by `status`, not retried until the input changes) |

The PNGs are snapshots of each accepted prompt, kept for review.

**Staleness follows the hashes.** When a line is redrawn, everything that depends on it stops
counting and is redone:
- a redrawn **general line** archives windows, the window review, event lines, space-times and
  slopes into `archive_<time>_general_redrawn/`;
- a changed **window review** (a window moved, added or deleted) makes every event line stale;
  the next event line archives them (`archive_<time>_stale_events/`);
- a redrawn **event line** makes that window's space-time and slope stale.

Nothing is deleted.

`export` writes one row per window:
- the slope, the displacement slope, confidence, label, phase;
- ECG trustworthiness, line source, registration verdict;
- the automatic speeds of every view;
- the detector (`energy` / `semblance`) and the window's screen score and burst ratio.

Only rows whose slope matches the current line are written. Screened windows are exported too,
with `screened=True` and no slope.

## Implementation notes

- **Cropped reads.** The worker reads only a box around the line(s) ± 10 mm from the ~1 GB
  buffer-4 file (`load_acquisition(roi=...)`). Detection on the crop reproduces the full-frame
  windows on C000000023 to within one frame (≤ 1.1 ms at 2 of 4 events). The residual comes from
  the stride-2 overview grid starting on a different pixel. Detection takes ~50 s per folder and
  the five views ~20 s for four windows.
- **Buffer 4 as displayed.** One diverging-wave frame rarely shows the septum, so buffer 4 is
  shown, and registered, as the mean envelope of 9 frames (~10 ms) around the target.
- **Fast start.** The editors avoid importing zea/torch, which costs up to 60 s here; the trigger
  log and line transfer are loaded straight from their files (`swp.manual._light`). A session
  starts in a few seconds, and the next prompt is loaded in the background while one is open.
- **Tk and the prefetch thread (2026-10-05).** A closed editor leaves Tk `Variable` / `Image`
  objects in reference cycles. If Python's automatic garbage collection runs in the prefetch
  thread, it frees them there. Tk then raises `main thread is not in main loop` and may hang.
  A small script on this machine reproduced it: 14 errors and a hang in 15 open / close cycles.
  `Session.run` now switches automatic collection off for the session and calls `gc.collect()` on
  the main thread after every prompt. The same script then finishes cleanly. Keep Tk work on the
  main thread when changing the session.
- **Screen cost.** Detection also runs the default view along the general line over the whole
  recording and scores a 100 ms window every 5 ms. That adds ~30 s per folder in the background
  worker (31 s measured on C000000001, on a loaded server).
- **Tests.**
  - Run with `KERAS_BACKEND=torch` set: without it the 2 tests that import zea fail.
  - `tests/test_manual.py`: the state machine, staleness, locking, archiving, the cropped loader,
    the five views and the display constants.
  - `tests/test_passive_screen.py`: the picker (phase-window peak, dropped out-of-record window,
    screening, top-ups), the score on a synthetic wave, and that screened windows do not hold a
    folder.
