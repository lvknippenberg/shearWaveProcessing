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
  - everything you skipped: `--retry-skipped`;
  - windows the detector screened out (below the semblance threshold, see step 2):
    `--include-screened`.
- **Folders still being beamformed:** they are picked up automatically. A folder counts as ready
  once the batch has written its buffer-4 GIF.

**Queue order.** The 44 folders that already have September lines come first; their old lines are
pre-loaded, so ENTER accepts them. After that come the remaining folders, sorted. `Strain_data`
folders are not listed (no passive buffer).

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
   - otherwise the general line as drawn.

   The general line is also shown as a dashed magenta reference on buffer 4.
5. *(worker)* **Five space-times** per event, over the window ± 20 ms.
6. **Slope** on the five views plus the buffer-4 B-mode.

### Drawing an M-line

- Draw on whichever panel shows the septum best: buffer 4 when readable (it is the processed data),
  otherwise buffer 1, otherwise buffer 3.
- Points can be clicked in any order; the first click is r = 0 (the yellow star).
- While drawing, the other panels show the same coordinates dashed.
- ENTER registers the drawn buffer onto the others (`swp.mline.transfer`: anatomy-scale phase
  correlation in a box around the line, an ensemble of box sizes, and a known-shift check). This
  shows:
  - the moved lines on the other buffers (orange);
  - the line that will be saved on buffer 4 (green);
  - whether the registration is **trusted**: ensemble agreement ≥ 0.6 and known-shift error
    ≤ 1 mm.
- If it is not trusted, nudge the green line with the arrow keys (0.25 mm; shift: 1 mm), or press
  `i` to use the uncorrected coordinates. ENTER accepts.

| key | |
|---|---|
| click / drag / right-click | add / move / delete a point |
| ENTER | map and review; ENTER again accepts |
| `c` | clear the line (to draw on another buffer) |
| arrows, shift+arrows | nudge the green buffer-4 line by 0.25 / 1 mm (review) |
| `i` | motion correction on / off (review) |
| `z` | zoom all panels on the line ± 25 mm / full sector |
| `[` `]` | step the frame of the panel under the mouse (buffer 4: 5 frames) |
| `a` | buffer-4 averaging (9 frames ≈ 10 ms) on / off |
| `x` | skip: no usable septum. For a general line this skips the whole folder |
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
- **No valid ECG.** When `swp.acquisition.rrcheck` rejects the R-peak record (for example
  C000000005: "unusable: sparse"), every "R-peak" / "R+x ms" is meaningless. The titles then say
  NO VALID ECG, and buffer 1 is chosen by anatomy match to buffer 4.
- **Every mirrored line carries a verdict.** In review, each non-source panel shows the registered
  line (orange = trusted, red dotted = not trusted, verdict in the title) and the uncorrected
  coordinates (dashed white). When you draw on buffer 4, the saved line is exactly what you drew;
  the registration only moves the display lines on buffers 1 and 3.

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
- The slider starts at the automatic slant-stack speed of the default view (3 m/s if that fit
  rails). The number itself is not shown, so it does not anchor the reading.
- `u` unlinks the displacement panel so it can get its own tilt. Displacement and velocity weight
  different frequencies of a dispersive wave; by hand, velocity came out faster in 87 % of windows.
  Linked, both are stored as the same speed.
- Accept by scoring how clearly a wavefront is visible:
  - `3` clear, `2` plausible, `1` guess, `0` none (no line needed);
  - `x` skips the window undecided.

The score is part of the measurement: automatic bias was +14 % on *clear* panels against +355 % on
*none* (`docs/passive_speed_estimation.md`).

| key | |
|---|---|
| click | anchor the line (on the displacement panel: its own anchor when unlinked) |
| slider, ← → | ± 0.05 m/s |
| ↑ ↓ | ± 0.5 m/s |
| `f` | flip the direction |
| `u` | unlink / re-link displacement |
| `r` | clear the anchor |
| `3` `2` `1` `0` | accept with confidence |
| `x` / `b` / `q` | skip / back / quit |

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
| `general.json`, `general_mline.npz`, `general.png` | session | line in buffer-4 coordinates; the buffer and points drawn, frames, registration, nudge, what was pre-loaded |
| `windows.json`, `passive_bursts.png`, `passive_full_spacetime.png` | worker | windows + labels + phases, ECG R-peak check, detection key (general-line hash), provenance; `needs_review` for the valves detector (no PNGs then) |
| `general_st.npz` | worker | the whole-recording "velocity gauss" space-time along the general line + R-peaks (valves detector) |
| `review.json`, `review.png` | session | the reviewed event windows + phases, what was proposed, keyed to the detected windows' hash |
| `events.json`, `event<i>_mline.npz`, `event<i>.png` | session | per-event line records, keyed to the (reviewed) windows' hash |
| `processed.json`, `st_win<i>.npz` | worker | the five space-times + automatic speeds, keyed to the line's hash |
| `slopes.json`, `slope<i>.png` | session | the slope(s), confidence, anchor view, automatic speeds for comparison, keyed to the space-times' hash |
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
- **Screen cost.** Detection also runs the default view along the general line over the whole
  recording and scores a 100 ms window every 5 ms. That adds ~30 s per folder in the background
  worker (31 s measured on C000000001, on a loaded server).
- **Tests.**
  - `tests/test_manual.py`: the state machine, staleness, locking, archiving, the cropped loader,
    the five views and the display constants.
  - `tests/test_passive_screen.py`: the picker (phase-window peak, dropped out-of-record window,
    screening, top-ups), the score on a synthetic wave, and that screened windows do not hold a
    folder.
