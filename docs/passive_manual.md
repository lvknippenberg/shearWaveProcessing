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
```

Use the `envs\zea_latest` python. `session` starts two background workers, logged to
`study/logs/passive_manual_worker*.log`, and stops them on exit. They run the burst detection
once a general line exists, and the five space-times once an event line exists. With
`--workers 0`, run `passive_manual.py worker --watch` yourself instead, for example on another
machine.

- **Stop:** `q`, or close the window. Everything accepted is on disk.
- **Go back:** `b` re-opens the previous prompt with your answer pre-loaded.
- **Choose what to work on:** `--task lines | events | slopes | lines+events` limits the session
  to one kind of prompt. The default `auto` finishes folders first: slopes, then event lines, then
  general lines, each in queue order.
- **Revisit later:**
  - one folder: `--folder <f> --redo general`, or `--redo event --window i` / `--redo slope --window i`;
  - everything you skipped: `--retry-skipped`.
- **Folders still being beamformed:** they are picked up automatically. A folder counts as ready
  once the batch has written its buffer-4 GIF.

**Queue order.** The 44 folders that already have September lines come first; their old lines are
pre-loaded, so ENTER accepts them. After that come the remaining folders, sorted.

## Per folder

1. **General M-line.** Buffers 1 | 3 | 4 at the R-peak, each at the frame nearest a logged R-peak
   (buffer 4: frame 0). This line is used to detect the valve events.
2. *(worker)* **Detection.** Phase-aware, identical to `configs/passive.yaml` (5-150 Hz
   displacement, MVC / AVC windows from the R-peaks). Up to 4 windows of 100 ms, labelled
   MVC / AVC / AK / other.
3. **Event lines.** Buffers 1 | 3 | 4 at the event's cardiac phase: buffer 4 at the event itself,
   buffers 1 and 3 at the frame with the same time since the preceding R-peak. The panel titles
   give the phase and the offset from the event. Pre-loaded, in order of preference:
   - the September line of that event, when an old window lies within 40 ms;
   - otherwise the general line as drawn.

   The general line is also shown as a dashed magenta reference on buffer 4.
4. *(worker)* **Five space-times** per event, over the window ± 20 ms.
5. **Slope** on the five views plus the buffer-4 B-mode.

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

## What is saved

Everything is saved in `<folder>/output/swp_passive_manual/`. The September outputs (`mlines/`,
`swp_passive/`) are only read. Each file has one writer, so the session and any number of workers
can run together:

| file | writer | content |
|---|---|---|
| `general.json`, `general_mline.npz`, `general.png` | session | line in buffer-4 coordinates; the buffer and points drawn, frames, registration, nudge, what was pre-loaded |
| `windows.json`, `passive_bursts.png`, `passive_full_spacetime.png` | worker | windows + labels + phases, ECG R-peak check, detection key (general-line hash), provenance |
| `events.json`, `event<i>_mline.npz`, `event<i>.png` | session | per-event line records, keyed to the windows' hash |
| `processed.json`, `st_win<i>.npz` | worker | the five space-times + automatic speeds, keyed to the line's hash |
| `slopes.json`, `slope<i>.png` | session | the slope(s), confidence, anchor view, automatic speeds for comparison, keyed to the space-times' hash |
| `log.jsonl` | session | every accept / skip, append-only |
| `worker_errors.json` | worker | a failed detection / processing (reported by `status`, not retried until the input changes) |

The PNGs are snapshots of each accepted prompt, kept for review.

**Staleness follows the hashes.** When a line is redrawn, everything that depends on it stops
counting and is redone:
- a redrawn **general line** archives windows, event lines, space-times and slopes into
  `archive_<time>_general_redrawn/`;
- a redrawn **event line** makes that window's space-time and slope stale.

Nothing is deleted.

`export` writes one row per window: the slope, the displacement slope, confidence, label, phase,
ECG trustworthiness, line source, registration verdict and the automatic speeds of every view.
Only rows whose slope matches the current line are written.

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
- **Tests.** `tests/test_manual.py` covers the state machine, staleness, locking, archiving, the
  cropped loader, the five views and the display constants.
