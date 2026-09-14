# Passive M-lines: drawing them once, resuming any time

The passive workflow needs **manually drawn M-lines**, and across a 44-folder study that is the
only part a person has to sit through. This describes how the drawing is split so it can be
stopped and resumed, and exactly what is saved where.

## How many M-lines, and why they can't all be front-loaded

Per measurement folder the passive workflow (`run.py passive`) needs:

| M-line | when it is drawn | depends on |
|---|---|---|
| **1 general** | first, before anything else | nothing — just a B-mode frame |
| **up to 4 per-window** | one per detected burst | burst detection, which needs the general M-line **and** a ~2 min displacement overview |

So ~5 per folder, ~220 for the study. Only the **general** one can be front-loaded; the others
are drawn on the B-mode frame *at their own burst time*, which is not known until the general
line has been processed.

## Stage 1 — the general M-lines (`scripts/draw_passive_mlines.py`)

```
python scripts/draw_passive_mlines.py --root "Z:\raw_data"            # draw / resume
python scripts/draw_passive_mlines.py --root "Z:\raw_data" --dry-run  # what is left
```

Each folder opens a window showing the buffer-4 B-mode **as a looping cine**, because cardiac
anatomy is far easier to identify in motion than in a still — a single frame of a low-SNR
diverging-wave acquisition often does not show the wall at all.

Playback defaults span the **whole buffer in a 5 s loop**: `--duration 5 --fps 25` keeps
125 frames (every 7th of ~926) → **5x slow motion** over the full ~1 s acquisition.

| flag | meaning |
|---|---|
| `--duration` | seconds to play the whole buffer in (default 5) |
| `--fps` | playback rate (default 25); with `--duration` this sets how many frames are kept |
| `--start` / `--end` | restrict to part of the cycle. `--start 0` is the **R-peak** — the acquisition is R-peak gated, so early frames sit at end-diastole where the heart is near-stationary and the image is sharpest |
| `--still` | fall back to a single frame |
| `--redraw` | redraw folders that already have a line |

Cost note: striding across the whole 1.3 GB buffer takes **~34 s per folder** to load. Narrowing
with `--end` (and a shorter `--duration`) cuts that if the wait outweighs the benefit.

### Resuming

**The run is resumable and safe to interrupt.** Each M-line is written as soon as it is drawn:

```
<folder>/output/mlines/passive_general_mline.npz    (+ .png record of what was drawn)
```

On the next run, folders that already have that file are **skipped** — the header reports
`N to draw, M already saved`. Ctrl-C stops cleanly and keeps everything drawn so far; closing a
window without clicking skips just that folder. So the 43 folders can be done over several
sittings with no bookkeeping.

To redo specific folders, delete their `passive_general_mline.npz` (or pass `--redraw`, which
redoes all of them).

## Stage 2 — the per-window M-lines (during `run.py passive`)

```
python run.py passive "<folder>" --config configs/passive.yaml
```

This reuses the saved general M-line (`[M-line] reuse passive_general_mline.npz`), runs burst
detection, then prompts for one M-line per detected window. Those are saved index-keyed as

```
<folder>/output/mlines/passive_win<i>_mline.npz
```

and are likewise reused on re-runs, so re-processing with different filter settings costs no
extra drawing. `--redraw` forces all of them to be re-drawn.

> Keyed by window **index**, not by time span, so a recipe tweak that jitters the detected window
> edges still reuses the right line. If the *number or order* of detected windows changes, re-draw
> (`--redraw`) — otherwise a saved line may be reused for a different event.

## Checking progress across the study

```
python -c "import glob; \
  print(len(glob.glob(r'Z:\raw_data\C*\*\output\mlines\passive_general_mline.npz')), 'general M-lines')"
```

or `scripts/draw_passive_mlines.py --root "Z:\raw_data" --dry-run`, which lists exactly what is
still outstanding.
