"""Draw the GENERAL passive M-line for many measurement folders in one sitting.

The passive workflow (`run.py passive`) needs a general M-line before it can detect bursts, and
each detected burst window then needs its own M-line. Doing that folder-by-folder means waiting
~2 min for the burst overview between every drawing - painful across 44 folders.

This script front-loads only the **general** M-line, which needs a single B-mode frame (a lazy
one-frame read, not the 1.3 GB stack), so you can draw all of them back to back. It saves to
exactly the path the passive workflow looks for::

    <folder>/output/mlines/passive_general_mline.npz

so a later `run.py passive` picks each one up automatically (``[M-line] reuse ...``) and goes
straight to burst detection.

Resumable: folders that already have a general M-line are skipped unless ``--redraw``, so you
can stop at any point and continue later.

.. note::

   The per-window M-lines (up to ``--max-events`` per folder) still have to be drawn during the
   passive run itself - they depend on burst detection, which depends on this line. This pass
   covers 1 of the ~5 M-lines per folder.

Usage:
    python scripts/draw_passive_mlines.py --root "Z:\\raw_data"
    python scripts/draw_passive_mlines.py --root "Z:\\raw_data" --dry-run
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

import numpy as np

PASSIVE_IQ = "CombinedData_buffer4_iq.hdf5"
MLINE_NAME = "passive_general_mline.npz"
PASSIVE_PRF = 925.93        # buffer-4 frame rate (Hz), for the slow-motion factor


def frame_count(iq_path):
    """Number of frames in a beamformed IQ file, read from the shape only (no data)."""
    import h5py
    with h5py.File(str(iq_path), "r") as f:
        return f["tracks/track_0/data/beamformed_data/values"].shape[0]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=None, help=r'study root, e.g. "Z:\raw_data"')
    p.add_argument("--folder", action="append", default=[])
    p.add_argument("--n-samples", type=int, default=250, help="samples along the M-line")
    p.add_argument("--still", action="store_true",
                   help="draw on a single still frame instead of a looping cine")
    p.add_argument("--duration", type=float, default=5.0,
                   help="seconds to play the WHOLE buffer in, then loop (default 5). Frames are "
                        "skipped evenly to hit this, so the full ~1 s acquisition is shown as a "
                        "5 s loop = ~5x slow motion.")
    p.add_argument("--fps", type=float, default=25.0,
                   help="cine playback rate (default 25); with --duration this sets how many "
                        "frames are kept (duration x fps)")
    p.add_argument("--start", type=int, default=0,
                   help="first frame (default 0 = the R-peak; the acquisition is R-peak gated, "
                        "so early frames sit at end-diastole where the heart is near-stationary "
                        "and the image is sharpest)")
    p.add_argument("--end", type=int, default=None,
                   help="last frame to include (default: the end of the buffer)")
    p.add_argument("--redraw", action="store_true", help="redraw even if one is saved")
    p.add_argument("--dry-run", action="store_true", help="list what would be drawn and exit")
    a = p.parse_args()
    if not a.root and not a.folder:
        p.error("give --root and/or --folder")

    from process_raw_data import find_measurement_folders
    folders = find_measurement_folders(a.root, a.folder, None)

    todo, skipped, missing = [], [], []
    for f in folders:
        iq = f / "output" / PASSIVE_IQ
        npz = f / "output" / "mlines" / MLINE_NAME
        if not iq.is_file():
            missing.append(f)
        elif npz.is_file() and not a.redraw:
            skipped.append(f)
        else:
            todo.append((f, iq, npz))

    print(f"=== general passive M-lines: {len(todo)} to draw, {len(skipped)} already saved, "
          f"{len(missing)} without {PASSIVE_IQ} ===")
    for f in missing:
        print(f"  MISSING passive IQ: {f}")
    if a.dry_run:
        for f, _, _ in todo:
            print(f"  would draw: {f}")
        return 0
    if not todo:
        print("nothing to draw.")
        return 0

    print("\nFor each folder: a B-mode window opens. Left-click points along the anatomy the")
    print("shear wave travels through, then press ENTER (with the figure focused) to accept.")
    print("Close the window without clicking to SKIP that folder. Ctrl-C stops the run;")
    print("everything drawn so far is saved and the script resumes where you left off.\n")

    from swp.mline.select import (load_bmode_cine, load_bmode_frame, save_mline,
                                  select_mline, select_mline_cine, draw_mline_on_bmode)

    done, t0 = 0, time.perf_counter()
    for i, (folder, iq, npz) in enumerate(todo, 1):
        name = f"{folder.parent.name}/{folder.name}"
        try:
            if a.still:
                bmode, coords, n = load_bmode_frame(iq, a.start)
                title = (f"[{i}/{len(todo)}] {name}\n"
                         f"GENERAL passive M-line (buffer 4, still frame {a.start}/{n})")
                ml = select_mline(bmode, coords, n_samples=a.n_samples, title=title)
            else:
                # Span the whole buffer in `--duration` seconds of playback: keep
                # duration*fps frames, evenly skipped across the acquisition.
                n_total = frame_count(iq)
                end = min(a.end or n_total, n_total)
                n_show = max(2, int(round(a.duration * a.fps)))
                stride = max(1, int(round((end - a.start) / n_show)))
                n_show = min(n_show, (end - a.start + stride - 1) // stride)
                cine, coords, n = load_bmode_cine(iq, a.start, n_show, stride)
                slowdown = a.duration / ((end - a.start) / PASSIVE_PRF)
                title = (f"[{i}/{len(todo)}] {name}\n"
                         f"GENERAL passive M-line - whole buffer ({a.start}-{end}/{n}) as a "
                         f"{a.duration:.0f} s loop = {slowdown:.1f}x slow motion "
                         f"({len(cine)} frames, every {stride}th)")
                ml = select_mline_cine(cine, coords, n_samples=a.n_samples,
                                       title=title, fps=a.fps)
                bmode = cine[0]
        except KeyboardInterrupt:
            print(f"\ninterrupted - {done} saved this run; rerun to continue.")
            break
        except Exception as exc:                          # noqa: BLE001
            print(f"  [{i}/{len(todo)}] {name}: SKIPPED ({type(exc).__name__}: {exc})")
            continue
        npz.parent.mkdir(parents=True, exist_ok=True)
        save_mline(npz, ml)
        draw_mline_on_bmode(bmode, coords, ml, str(npz).replace(".npz", ".png"), title=title)
        done += 1
        print(f"  [{i}/{len(todo)}] {name}: saved ({ml.length * 1e3:.1f} mm)")

    mins = (time.perf_counter() - t0) / 60
    print(f"\n{done} M-line(s) drawn in {mins:.1f} min. "
          f"{len(skipped) + done}/{len(folders)} folders now have a general M-line.")
    print("Next: run the passive workflow per folder; it will reuse these and go straight to "
          "burst detection (each window then needs its own M-line).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
