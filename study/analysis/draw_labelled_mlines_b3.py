"""Redraw the passive M-line of each labelled window on the focused-beam B-mode (buffer 3).

The 15 hand-labelled passive windows (``study/logs/labelled_panels.json``) were analysed on
M-lines drawn on the widebeam buffer 1. Buffer 3 (focused beams, REFoCUS-adjoint reconstruction)
shows the septum more sharply, so a line drawn there may follow the wall more accurately.

For each window the buffer-3 frame at the **same cardiac phase as the event** in the diverging-
wave buffer 4 is shown - the frame whose time after its preceding R-peak is closest to the
event's (``swp.passive.event_bmode_frames``, trigger log). Buffer 3 has only 26-32 frames per
acquisition, so that frame can be up to ~20 ms from the event phase; the offset is in the title.

Shown for reference: the old buffer-1 line (dashed) and the segment of it that was analysed
(``part``: full / left / right half). Draw the line along the septum over the stretch where the
valve wave travels.

    ENTER without clicking  keep the old analysed segment for this window
    close the window        skip the window (no line saved)

Writes, next to the old lines (which are NOT touched):
    <folder>/output/mlines/passive_win<i>_mline_b3.npz (+ .png record)
    <folder>/output/mlines/passive_mlines_b3.json     frame, phases, kept/skipped
and a summary ``study/logs/passive_mlines_b3.csv``. Resumable: windows with a line are skipped
unless ``--redraw``. Order is by subject; the confidence scores are not shown.

    python study/analysis/draw_labelled_mlines_b3.py            (needs a desktop: MPLBACKEND=TkAgg)
    python study/analysis/draw_labelled_mlines_b3.py --only C000000019 --redraw
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "study" / "analysis"))
os.environ.setdefault("KERAS_BACKEND", "torch")

from swp import paths as P                                     # noqa: E402

BUFFER = 3
CONFIG = str(_REPO / "configs" / "passive.yaml")


def b3_npz(mlines_dir, i):
    return os.path.join(mlines_dir, f"passive_win{i}_mline_b3.npz")


def main():
    import swp.passive as SP
    from swp.mline.select import load_bmode_frame
    from swp.viz.mline import mline_from_points
    from passive_mline_split import split_line

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--redraw", action="store_true", help="redraw windows that already have a line")
    ap.add_argument("--only", nargs="*", default=None, help="subjects to (re)draw, e.g. C000000019")
    a = ap.parse_args()

    panels = sorted(json.load(open(_REPO / "study/logs/labelled_panels.json")),
                    key=lambda c: (c["subject"], c["window"]))
    if a.only:
        panels = [c for c in panels if c["subject"] in a.only]
    summary = []
    for n_done, c in enumerate(panels, 1):
        folder = f"{P.RAW_DATA}/{c['folder']}"
        cfg, p = SP._paths(folder, CONFIG)
        n = cfg["mline"].get("n_samples", 250)
        st, ws = SP.read_windows(p["windows_json"])
        i = int(c["window"])
        w = ws[i]
        out = b3_npz(p["mlines"], i)
        rec_path = os.path.join(p["mlines"], "passive_mlines_b3.json")
        recs = json.load(open(rec_path)) if os.path.exists(rec_path) else {}
        if os.path.exists(out) and not a.redraw:
            print(f"[{n_done}/{len(panels)}] {c['subject']} win{i}: already drawn - skipped")
            summary.append(dict(subject=c["subject"], window=i, event=c["label"], **recs.get(str(i), {})))
            continue
        fr = SP.event_bmode_frames(folder, [w], buffer=BUFFER)[0]
        img, coords, n_frames = load_bmode_frame(os.path.join(p["output"], SP.bmode_file(BUFFER)), fr["frame"])
        old = SP._load_line(SP._window_npz(p["mlines"], i), n)
        seg_pts = old.points if c["part"] == "full" else split_line(old, n)[c["part"]]
        seg = mline_from_points(np.asarray(seg_pts, float), n)
        off = fr["frame_phase_ms"] - fr["event_phase_ms"]
        title = (f"[{n_done}/{len(panels)}] {c['subject']} window {i} ({c['label']}): "
                 f"buffer 3 frame {fr['frame']}/{n_frames}, R+{fr['frame_phase_ms']:.0f} ms "
                 f"(event R+{fr['event_phase_ms']:.0f} ms, offset {off:+.0f} ms)\n"
                 f"draw along the septum where the valve wave travels  |  ENTER without clicking = "
                 f"keep old segment  |  close = skip")
        ref = [("old buffer-1 line", old.x, old.z), (f"old analysed segment ({c['part']})", seg.x, seg.z)]
        print(f"[{n_done}/{len(panels)}] {c['subject']} win{i}: drawing on buffer 3 frame {fr['frame']}", flush=True)
        try:
            _, kept = SP._draw_line(out, img[None], coords, title, n, 1.0, reference=ref,
                                    labels=[f"buffer 3, frame {fr['frame']}"], fallback_points=seg_pts)
            recs[str(i)] = dict(buffer=BUFFER, **fr, kept_old_segment=bool(kept), part_before=c["part"])
        except SP.SkipLine:
            if os.path.exists(out):
                os.remove(out)
            recs[str(i)] = dict(buffer=BUFFER, **fr, skipped=True, part_before=c["part"])
            print("   skipped")
        with open(rec_path, "w") as fh:
            json.dump(recs, fh, indent=1)
        summary.append(dict(subject=c["subject"], window=i, event=c["label"], **recs[str(i)]))

    keys = ["subject", "window", "event", "buffer", "frame", "event_phase_ms", "frame_phase_ms",
            "kept_old_segment", "skipped", "part_before"]
    with open(_REPO / "study/logs/passive_mlines_b3.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(summary)
    drawn = sum(1 for s in summary if not s.get("skipped") and "frame" in s)
    print(f"\n{drawn} of {len(summary)} windows have a buffer-3 line -> study/logs/passive_mlines_b3.csv")


if __name__ == "__main__":
    main()
