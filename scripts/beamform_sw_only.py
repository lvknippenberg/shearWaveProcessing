"""Beamform only the buffers the shear-wave analysis needs (2 = active SW, 5 = co-registered B-mode).

The in-vivo acquisitions also carry a 106-frame orientation B-mode (buffer 1), a 32-frame focused
B-mode (3), an 1112-frame ultrafast diverging-wave stream (4) and a 318-frame strain buffer (6).
Beamforming those costs far more time than buffers 2+5 and none of it is used by the active-SWE
space-time analysis, so this driver restricts `swp.acquisition.beamform.run` to buffers 2 and 5.

    python scripts/beamform_sw_only.py <folder> [<folder> ...] [--buffers 2,5] [--overwrite]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
os.environ.setdefault("KERAS_BACKEND", "torch")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--buffers", default="2,5")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    bufs = [int(b) for b in a.buffers.split(",")]

    from swp.acquisition.beamform import run, find_mat, init_device
    init_device(verbose=True)
    for folder in a.folders:
        t0 = time.time()
        print(f"=== {folder} buffers={bufs}", flush=True)
        run(find_mat(folder), output_dir=os.path.join(folder, "output"), buffers_matlab=bufs,
            save_converted=False, overwrite=a.overwrite)
        print(f"=== done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
