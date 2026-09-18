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
python scripts/passive_study.py draw-events --root "Z:/raw_data" [--defer-process]   # 1 prompt per detected event
python scripts/passive_study.py reprocess   --root "Z:/raw_data" [--only-event-lines] # after --defer-process
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
  Over a whole study use **`--defer-process`**: reprocessing a folder takes ~2.5 min and otherwise
  stalls the prompts between folders (134 events over 37 folders were drawn in one sitting this way).
- `reprocess`: forces processing regardless of status - needed after `--defer-process`, because
  `status()` compares the montage against the *general* line and cannot see that the per-window lines
  changed. `--only-event-lines` restricts it to folders that have per-event lines.
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

## Part-wise analysis over the whole study

`passive_mline_split.py` does one folder. `passive_split_study.py` drives it over a tree and gathers
every fit of every part into one table:

```
python study/analysis/passive_split_study.py --root "Z:/raw_data" --jobs 4 [--watch]
python study/analysis/passive_split_study.py --root "Z:/raw_data" --collect-only   # rebuild the table
```

- Each folder runs as its own subprocess (the split script monkey-patches `swp.passive._paths` while
  it works) and writes `output/swp_passive/split_run.log`; one failure never stops the study.
- `--jobs N` splits N folders at once. The work is single-threaded numpy and a worker holds ~1 GB of
  buffer-4 IQ, so on the 40-core machine `--jobs 4` runs ~4x faster than serial (~5 min/folder wall
  clock instead of ~1.3 min/folder of CPU).
- `--watch` polls for folders that `passive_study.py process` has not finished yet, so it can run
  next to `process` the way `process` runs next to `draw`. It waits only on folders that have a line
  drawn — the hundreds of never-beamformed folders in the tree do not hold it open.
- Resumable: a folder is skipped when its `split_speeds.txt` is newer than its montage (the marker
  that `process` finished). Staleness is deliberately **not** judged against `passive_windows.json`,
  which the `label` pass rewrites without changing any window.

Output: `study/logs/passive_split_speeds.csv` — one row per folder x window x view x part, with the
signed speed, semblance, part length, cardiac label, whether the fit sits on a search bound, and
whether the folder used per-event or general lines. The script also prints medians per part, per
part x cardiac event, and how often the three views agree on a part (a real front gives the same
speed and sign in all three recipes; the agreement counts are the useful column).

## Study status (2026-09-18): redrawn on buffer 1, per-event lines, all parts analysed

The buffer-3 frame-0 lines are gone. Every measurement folder under `Z:/raw_data` that carries an
`output/` folder (44; C000000046-49 are not beamformed) was offered for redrawing on the **buffer-1
R-peak frame**, processed, then redrawn **per detected event** and reprocessed, then split into
full / left / right.

- **37 folders have a general line** (36 drawn in one 26-minute pass, plus C000000001's). 18-65 mm,
  median 34 mm. **7 skipped at drawing** - no usable septum on the buffer-1 R-peak frame:
  C000000009, 17, 24, 34, 38, 41, 43.
- **Per-event lines: all 134 detected events prompted** on their phase-matched buffer-1 frame -
  **79 drawn fresh**, 37 kept the general line, **18 skipped**. C000000036 had all 4 events skipped,
  so nothing was reprocessed for it and it is excluded (its general-line split is archived under
  `swp_passive/archive_general_line_20260918/`).
- Final table: **36 folders x 116 windows x 3 views x 3 parts = 1044 fits** in
  `study/logs/passive_split_speeds.csv`.

| part | n | median \|c\| | IQR | at bound | med semblance | median length |
|---|---|---|---|---|---|---|
| full | 348 | 3.38 | 1.9-6.3 | 18 % | 0.74 | 38 mm |
| left | 348 | 3.38 | 2.0-6.4 | 25 % | 0.87 | 19 mm |
| right | 348 | 3.72 | 2.1-6.1 | 23 % | 0.88 | 19 mm |

| part | all 3 views off the bounds | same direction | spread < 25 % | **usable** |
|---|---|---|---|---|
| full | 69 | 44 | 3 | **2** |
| left | 56 | 41 | 2 | **2** |
| right | 59 | 39 | 3 | **1** |

Reading:

- **Splitting the line still does not help.** Same verdict as the general-line round, and the halves
  still carry the higher semblance (0.87/0.88 vs 0.74) purely because they are ~19 mm against ~38 mm:
  a short segment spans a small fraction of a wavelength, so almost any slope fits it. Semblance
  must never be used on its own to pick a window.
- **Per-event lines did not raise the automatic agreement** - strictly usable windows went from 6 to
  2 on the full line. This is **not** a like-for-like comparison: 18 events were skipped and one
  folder dropped out, so the per-event set is 116 windows against 134. Per window the effect is
  mixed: C000000014 win2 improved sharply (spread 28 % -> 7 %, and off the 1.0 m/s floor: 1.1-1.5 ->
  2.1-2.2 m/s on a line that grew 34 -> 46 mm), C000000023 win3 was unchanged, C000000033 win3 and
  C000000021 win1 got worse.
- The point of per-event lines is **anatomical correctness** at each event's cardiac phase, not a
  higher score. A general line that does not follow the septum at that phase can still produce a
  tidy-looking fit; some of the general-line agreement was of that kind.
- ~20 % of fits still rail at the 1.0/20.0 m/s bounds. The table screens; the montages measure.

Usable windows now: full C000000014 w2 (2.15 m/s, 7 % spread) and C000000023 w3 AK (2.78, 17 %);
left C000000003 w3 AK (1.52) and C000000012 w0 (10.8, non-physical); right C000000025 w0 MVC (3.05,
a 13 mm segment - vertical bands, the length artefact).

**Note.** `C000000044` win3, the cleanest wavefront of the general-line round (MVC, 2.8 m/s in all
three views, zero spread), had its event **skipped** during per-event drawing and is no longer in the
table. Redraw it with `draw-events --folder <that folder>` if it should be kept.

### Reading the speed by hand

The automatic slope fit is biased high and frequently does not sit on the wave at all;
`study/analysis/manual_slope.py` draws the wavefront by hand on the space-time panel instead.
The benchmark against the automatic estimator, the tracking score it is judged with, and the
recommended change to the estimator are in **[docs/passive_speed_estimation.md](passive_speed_estimation.md)**.

```
python study/analysis/manual_slope.py --folder "<folder>" --window 1 --part left
```

### Where the origin marker went

Passive panels no longer draw the dashed `r0` line or the two-sided fit. `r0` is where an ARF push
radiates outward from; a valve-closure wave enters the line at one end and crosses in one direction.
The old overlay came from `ttp_ransac_speed`, which splits the line at `r0` and fits each side
separately over only 2-14 mm either side. The reported speed was never affected - it comes from
`metrics.slant_stack_speed`, a signed slant stack over the whole line whose docstring already says
`r0` is unused and the direction is inferred from the data. The panel now overlays that estimator's
own moveout as **one continuous wavefront** (`swp.passive._single_wave_speed`, montage `show_r0=False`)
and labels it `radon-signed: <c>/-- m/s`. The active path is unchanged, where the symmetric V is real.

### Scrolling through the results

```
python study/analysis/collect_passive_montages.py --root "Z:/raw_data" --out "<folder>"
```

Copies every folder's montages into one flat tree - `main/`, `split_full/`, `split_left/`,
`split_right/`, `split_lines/`, `bursts/` - named by subject, so each set pages through in order in
an image viewer. Folders whose montage is stale (skipped at drawing, so the montage predates the
current line) are left out, and the `main/` file name records what the montage was built from
(`per-event`, `per-event-3of4`, `general-line`).

**Concurrency note.** Running `process` and `passive_split_study.py --watch` together is supported,
but on the network share one folder (C000000018) hit `PermissionError` on the `os.replace` of
`passive_windows.json` while the split was copying that same file. It was re-run on its own and is
correct in the table. If it recurs, run the split without `--watch` after `process` finishes.
