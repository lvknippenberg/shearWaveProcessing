"""Launch the interactive M-line picker for one measurement, save the .npz (used by the GUI's
'Draw/redraw M-line' button). Draws on the co-registered buffer-5 B-mode frame for the push.

    python swp_gui/draw_mline.py "<measurement folder>" <meas>
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("MPLBACKEND", "TkAgg")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import numpy as np


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: draw_mline.py <folder> <meas>")
    folder, meas = sys.argv[1], int(sys.argv[2])
    bmode = os.path.join(folder, "output", "CombinedData_buffer5_iq.hdf5")
    if not os.path.exists(bmode):
        raise SystemExit(f"no buffer-5 B-mode ({bmode}); this looks like a phantom - use the "
                         "horizontal-push M-line instead (no drawing needed).")
    from swp.mline import load_bmode_frame, select_mline, save_mline, draw_mline_on_bmode
    npz = os.path.join(folder, "output", "mlines", f"active_meas{meas}_mline.npz")
    os.makedirs(os.path.dirname(npz), exist_ok=True)
    u8, coords, n = load_bmode_frame(bmode, meas)
    ml = select_mline(u8, coords, n_samples=250, title=f"M-line for meas{meas}")
    save_mline(npz, ml)
    draw_mline_on_bmode(u8, coords, ml, npz.replace(".npz", ".png"), title=f"meas{meas} M-line")
    print(f"saved {npz} ({ml.length*1e3:.1f} mm)")


if __name__ == "__main__":
    main()
