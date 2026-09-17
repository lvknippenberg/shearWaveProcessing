# Passive M-lines: where to draw them, and the study workflow

The passive workflow needs manually drawn M-lines along the septum. This describes the settled way to
draw them (2026-09-17), what was tried and rejected, the commands, what is saved where, and the
C000000001 results that motivated each step. Cardiac timing background: `docs/ecg_timing.md`.

## Where to draw: buffer 1 at the matching cardiac phase

| tried | verdict | why |
|---|---|---|
| buffer 4 as a looping cine (`draw_passive_mlines.py`, 2026-09-14) | rejected | the septum moves too much over the cycle for one line to fit the loop |
| buffer 4, still frame | rejected | in most folders the septum is not clearly visible in the diverging-wave image |
| buffer 3 (focused), frame 0 | rejected | buffer 3 is **not** R-peak gated; frame 0 is at an arbitrary phase (C000000001 ~+200 ms), so the anatomy does not match buffer-4 frame 0 |
| **buffer 1 (widebeam), frame nearest an R-peak** | **single/general line** | good image; buffer 4 frame 0 is on an R-peak in 44/44 folders and every folder has a buffer-1 frame within ±5 ms of an R-peak |
| **buffer 1, phase-matched frame per event** | **per-event lines** | each detected window gets the buffer-1 frame at the same time-since-R-peak as the event (matched within ~4 ms on C000000001) |

The line is stored in metres and every IQ file carries per-pixel coordinates, so a line drawn on
buffer 1 maps onto the buffer-4 grid unchanged.

## Workflow (`scripts/passive_study.py`)

```
python scripts/passive_study.py draw    --root "Z:/raw_data"             # 1 prompt per folder: buffer 1 R-peak frame
python scripts/passive_study.py process --root "Z:/raw_data" [--watch]   # detect bursts along it, process all windows
python scripts/passive_study.py label   --root "Z:/raw_data"             # MVC / AVC / AK per window from the trigger log
python scripts/passive_study.py draw-events --folder "<folder>"          # 1 prompt per detected event, then reprocess
python scripts/passive_study.py status  --root "Z:/raw_data"
```

- `draw`: still frame (GIF display: adaptive levels + gamma2). Click points in any order, drag, right-click
  to delete, **Enter** to accept; **closing the window skips the folder**. `--buffer` / `--frame`
  override the source (`--frame rpeak` is the default; e.g. `--buffer 3 --frame 0` reproduces the old choice).
  Only one frame is read, so prompts come back to back. Resumable; `--redraw`, `--retry-skipped`.
- `process` (unattended, ~2.5 min/folder): loads buffer 4, detects bursts along the line (cached in
  `passive_windows.json`), uses the single line for every window unless per-event lines exist, runs
  the 3 passive views. `--watch` keeps polling so it can run next to `draw`.
- `draw-events`: for each detected window, a still of the phase-matched buffer-1 frame with the general
  line overlaid dashed. **Enter without clicking reuses the general line; closing skips that window.**
  Then reprocesses with the per-event lines (they are not overwritten by later `process` runs).
- `label`: see `docs/ecg_timing.md`; windows whose ECG log is unusable get `?`.

`run.py passive <folder>` still runs the older all-in-one path (lines drawn on buffer 4; `cine=True`
available) and `scripts/draw_passive_mlines.py` the buffer-4 cine general line; both are superseded
for the study.

## What is saved

```
<folder>/output/mlines/passive_general_mline.npz/.png   single line (+ record of what was drawn)
<folder>/output/mlines/passive_general_mline.json       source: buffer, frame, R-peak offset
<folder>/output/mlines/passive_win<i>_mline.npz/.png    line used for window i
<folder>/output/swp_passive/passive_windows.json        detection key, windows (+ label), window_mlines
                                                        (buffer/frame/phase per event), window_phases
<folder>/output/swp_passive/passive_windows_montage.png rows = windows; col 0 = B-mode frame + line
                                                        (yellow dot r = 0, + = r0); cols = 3 views
<folder>/output/swp_passive/passive_speeds.json         per window x view: label, speed, semblance, line length
<folder>/output/swp_passive/passive_bursts.png, passive_full_spacetime.png
```

A new detection (different general line or settings) archives stale `passive_win*` lines into
`mlines/archive_<timestamp>/` rather than reusing them for the wrong event. Earlier C000000001 runs
are kept in `swp_passive/compare_line*` folders.

## C000000001 results (2026-09-17)

Windows (buffer-4 time; R-peaks at 0 and 695 ms): win0 50 ms **MVC**, win1 441 ms **AVC**, win2 610 ms
**AK**, win3 782 ms **MVC**. Speeds in m/s for the three views (disp bp10-150 / disp bp5-150 median /
velocity bp15-90); semblance in brackets where it matters.

| line | win0 MVC | win1 AVC | win2 AK | win3 MVC |
|---|---|---|---|---|
| buffer 3 frame 0, 59 mm (first study line) | 1.4* | 3.4* | 1.0* | 5.3* |
| buffer 3 frame 0, 21 mm (redrawn) | 2.7 / 1.8 / 4.0 | 3.3 / 10.8 / 3.3 | 2.0 / 1.0 / 2.0 (at 651 ms) | - |
| buffer 1 R-peak frame, 34 mm, single | 2.9 / 2.0 / 3.2 | 1.7 / 20.0 / 3.2 | 1.1 / 1.4 / 1.1 | 4.0 / 3.6 / 6.8 |
| **buffer 1 per-event, 48-57 mm** | 3.6 / 2.1 / 5.0 | **-3.6 / -2.2 / -2.6** | 1.1 / 1.0 / 1.2 | 8.0 / 3.2 / 20.0 |
| per-event, **left half** (24-28 mm) | **3.5 / 2.6 / 3.4 (0.81-0.95)** | 10.8 / -5.6 / -2.2 | 1.0 / 1.0 / 1.0 | 6.4 / 8.8 / 6.4 |
| per-event, right half (24-29 mm) | 16.5 / 2.6 / -16.5 | -1.0 / -1.0 / -1.6 | 12.2 / -6.0 / 3.1 | 6.4 / 1.8 / -9.7 |

\* median of the three views. 1.0 and 20.0 are the search bounds (no front found).

Reading:
- **MVC (win0) propagates in the left half of the septum**: 2.6-3.5 m/s, all views agree, highest
  semblance so far. The right half shows no front.
- **AVC (win1) needs the full per-event line**: only there do all three views agree (2.2-3.6 m/s,
  opposite direction to MVC). Its energy sits in the upper ~25 mm of that line; each half only sees a
  fragment.
- **win2 is atrial contraction, not noise**: burst energy 3.6e-4, as large as the MVC in win3 (3.4e-4),
  timed 85 ms before the next R-peak. But no segment gives a consistent front (views disagree or rail
  at the bounds) - label AK and exclude from speed estimates.
- **win3 (second MVC) is unresolved**: a near-vertical band on every segment, speeds 1.8-20 m/s.
- **Line length cuts both ways**: at 3 m/s a wave crosses 25 mm in ~8 ms against 20-30 ms wide bands, so
  short segments hardly resolve a tilt, while long lines run into regions without the wave. The
  result depends strongly on segment choice; only the left-half MVC is robust across views.

Split analysis (full / left / right of each per-event line, own montages + table + figure):

```
python study/analysis/passive_mline_split.py --folder "<folder>"
```

Figures: `study/montages/c1_passive_per_event_montage.png`, `study/montages/c1_passive_split_lines.png`.

## Study status (2026-09-17)

- 36/44 folders processed with the **buffer 3 frame 0** lines (8 skipped at drawing: C000000017, 24,
  26, 31, 36, 39, 40, 43). These lines are at the wrong cardiac phase - redraw with
  `draw --root Z:/raw_data --redraw` (buffer 1 R-peak frame) and reprocess.
- Only C000000001 has buffer-1 and per-event lines.
- Automatic speeds across the 36 (399 fits): median |c| 3.05 m/s, 50 % in 1.5-6 m/s, 20 % at the search
  bounds - a screening number, not a measurement; read the montages.
- Next: automate M-line selection (the drawn lines are the reference set), possibly several lines over
  the cycle.
