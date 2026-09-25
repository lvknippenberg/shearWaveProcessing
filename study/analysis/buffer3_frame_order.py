"""Are the stored buffer-3 frames in chronological order? (circular-buffer rotation test)

Buffer 3 is the focused LIVE loop. SetUp_SWI_Widebeam.m writes frame i of every pass into slot i
(i = 1..Nframes), then jumps back to slot 1, and returns control to MATLAB only after frames
1, 6, 11, 16, 21, 26 (, 31) (``mod(i-1, 5) == 0``). If the loop stops after slot s < Nframes, the
buffer holds slots s+1..N from the previous pass and 1..s from the last one, i.e. the stored order
is ROTATED, while ``swp.acquisition.triggerlog`` assigns the last N frame triggers to slots 0..N-1 in
order.

Signature in the images: exactly one neighbouring pair (s, s+1) is a jump of ~N-1 frames in time
(an anatomy discontinuity), while the wrap pair (N, 1) is then continuous - and s can only be one of
the MATLAB return points. Without rotation there is no internal jump and the wrap pair is the jump.

Per folder: anatomy-scale images of every frame (``swp.mline.transfer.anatomy``: log-envelope
band-passed 1-6 mm), correlation of each neighbouring pair and of the wrap pair.

    python study/analysis/buffer3_frame_order.py [--root Z:/raw_data] [--limit N]
-> study/logs/buffer3_frame_order.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from swp.manual import store as S                     # noqa: E402
from swp.manual._light import transfer                # noqa: E402

import h5py                                           # noqa: E402
import hdf5plugin                                     # noqa: E402,F401

_BF = "tracks/track_0/data/beamformed_data"


def frame_anatomy(path, step=2):
    with h5py.File(path, "r") as f:
        v = np.asarray(f[f"{_BF}/values"][:, ::step, ::step, :], np.float32)
        c = np.asarray(f[f"{_BF}/coordinates"])
    env = np.sqrt(v[..., 0] ** 2 + v[..., 1] ** 2)
    pix = abs(c[0, 1, 0] - c[0, 0, 0]) * 1e3 * step
    # central part of the sector, 20-100 mm deep: where the heart is and the sector is full
    z = c[::step, 0, 2] * 1e3
    x = c[0, ::step, 0] * 1e3
    rz, rx = (z > 20) & (z < 100), np.abs(x) < 25
    A = np.stack([transfer().anatomy(e, pix)[rz][:, rx] for e in env])
    return A


def corr(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def return_points(n):
    """1-based slots after which MATLAB gets control: mod(i-1, 5) == 0."""
    return [i for i in range(1, n + 1) if (i - 1) % 5 == 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="Z:/raw_data")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    folders = [f for f in S.find_folders(a.root) if S.ready(S.Paths(f))
               and Path(S.Paths(f).bmode(3)).exists()]
    if a.limit:
        folders = folders[:a.limit]
    rows = []
    for k, f in enumerate(folders):
        A = frame_anatomy(S.Paths(f).bmode(3))
        n = len(A)
        c = np.array([corr(A[i], A[i + 1]) for i in range(n - 1)])
        wrap = corr(A[-1], A[0])
        med = float(np.median(c))
        j = int(np.argmin(c))                      # 0-based pair (j, j+1) -> after 1-based slot j+1
        c_sorted = np.sort(c)
        row = dict(folder=f"{Path(f).parent.name}/{Path(f).name}", n_frames=n,
                   median_pair=round(med, 3), min_pair=round(float(c[j]), 3),
                   second_min=round(float(c_sorted[1]), 3), jump_after_slot=j + 1,
                   at_return_point=(j + 1) in return_points(n), wrap_pair=round(wrap, 3),
                   pairs=" ".join(f"{x:.2f}" for x in c))
        rows.append(row)
        print(f"[{k + 1}/{len(folders)}] {row['folder'][:40]:40s} n={n} median {med:.2f} "
              f"min {c[j]:.2f} after slot {j + 1:2d} (return pt {row['at_return_point']!s:5s}) "
              f"wrap {wrap:.2f}", flush=True)
    out = _REPO / "study" / "logs" / "buffer3_frame_order.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
