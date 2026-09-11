"""Hand-draw the septal M-line for every in-vivo push, one push at a time.

Opens the buffer-5 B-mode for each push with the ARF push focus marked, and lets you click
points along the septum (the repo's own `swp.mline.select_mline`). Each line is saved to
`<folder>/output/mlines/active_meas{m}_mline.npz` as soon as it is drawn, so the run is
**resumable** -- close the window / kill the script and re-run, and it continues where it
stopped. Anything already drawn is skipped unless `--redraw` is given.

The automatic proposal from `scripts/auto_mline.py` is drawn as a faint dashed guide, so you
can accept its shape by clicking along it or ignore it entirely. It is overwritten by yours.

Controls (from the repo selector): left-click to add a point in any order, drag a point to move
it, right-click to delete one, ENTER (with the figure focused) to finish that push.

    python scripts/draw_invivo_mlines.py D:/swp_iv                # both folders, all pushes
    python scripts/draw_invivo_mlines.py D:/swp_iv --meas 0-5     # a subset
    python scripts/draw_invivo_mlines.py D:/swp_iv --redraw       # start over
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in ("src", "swp_gui", "scripts"):
    if os.path.join(_ROOT, _p) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _p))
os.environ.setdefault("KERAS_BACKEND", "torch")

import h5py                                                      # noqa: E402
import swe_lib as L                                              # noqa: E402
from auto_mline import propose                                   # noqa: E402


def push_focus_mm(folder):
    with h5py.File(os.path.join(folder, "output",
                                "CombinedData_buffer2_meas0_iq.hdf5"), "r") as f:
        return (float(np.array(f["custom/push_focus_x"])) * 1e3,
                float(np.array(f["custom/push_focus_z"])) * 1e3)


def draw_one(folder, meas, px, pz, guide=True, n_samples=250):
    """Show the buffer-5 frame for this push and return the clicked M-line points (m)."""
    from swp.mline import load_bmode_frame, select_mline
    import matplotlib.pyplot as plt

    bmode = os.path.join(folder, "output", "CombinedData_buffer5_iq.hdf5")
    img, coords, _ = load_bmode_frame(bmode, meas)

    # Overlay the push focus and the automatic proposal onto the live axes after it is built.
    orig_subplots = plt.subplots

    def _patched(*a, **kw):
        fig, ax = orig_subplots(*a, **kw)
        try:
            ax.plot(px, pz, "r+", ms=16, mew=2.5, zorder=5)
            if guide:
                g = propose(folder, meas, px, pz, x_span_mm=34, order=1) * 1e3
                ax.plot(g[:, 0], g[:, 1], "--", color="lime", lw=1.2, alpha=0.6, zorder=4)
            ax.set_xlim(-45, 45)
            ax.set_ylim(80, 5)
        except Exception:                                        # noqa: BLE001
            pass
        return fig, ax

    plt.subplots = _patched
    try:
        ml = select_mline(img, coords, n_samples=n_samples,
                          title=f"{os.path.basename(folder)[:38]}   push {meas}   "
                                f"(red + = ARF push focus, dashed = automatic proposal)")
    finally:
        plt.subplots = orig_subplots
    return ml


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="folder holding the in-vivo measurement folders")
    ap.add_argument("--meas", default="all", help="e.g. 'all', '7', '0-5'")
    ap.add_argument("--redraw", action="store_true", help="redraw lines that already exist")
    ap.add_argument("--no-guide", action="store_true", help="hide the automatic proposal")
    a = ap.parse_args()

    from swp.mline import save_mline

    folders = sorted(os.path.join(a.root, d) for d in os.listdir(a.root)
                     if os.path.isdir(os.path.join(a.root, d))
                     and L.n_pushes(os.path.join(a.root, d)) > 0)
    if not folders:
        raise SystemExit(f"no beamformed measurement folders under {a.root}")

    if a.meas == "all":
        sel = None
    elif "-" in a.meas:
        lo, hi = (int(v) for v in a.meas.split("-"))
        sel = list(range(lo, hi + 1))
    else:
        sel = [int(a.meas)]

    total = done = 0
    for folder in folders:
        px, pz = push_focus_mm(folder)
        n = L.n_pushes(folder)
        todo = [m for m in (sel if sel is not None else range(n)) if m < n]
        print(f"\n=== {os.path.basename(folder)}")
        print(f"    push focus ({px:.1f}, {pz:.1f}) mm   |   {len(todo)} push(es) to consider")
        for m in todo:
            total += 1
            out = os.path.join(folder, "output", "mlines", f"active_meas{m}_mline.npz")
            if os.path.isfile(out) and not a.redraw:
                # Distinguish an automatic proposal from a hand-drawn line: auto lines are
                # written with exactly 7 evenly spaced anchors by scripts/auto_mline.py.
                pts = np.load(out)["points"]
                tag = "auto" if len(pts) == 7 else "hand-drawn"
                if tag == "hand-drawn":
                    print(f"    push {m:2d}: already hand-drawn - skipping")
                    done += 1
                    continue
            ml = draw_one(folder, m, px, pz, guide=not a.no_guide)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            save_mline(out, ml)
            done += 1
            print(f"    push {m:2d}: saved {len(ml.points)} points -> {os.path.basename(out)}")

    print(f"\ndrew/kept {done} of {total} M-lines.")
    print("Next: python scripts/task4_invivo_compare.py --root <root> --outdir <root>/analysis")


if __name__ == "__main__":
    main()
