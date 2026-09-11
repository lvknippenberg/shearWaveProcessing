"""Automatic septal M-line proposal for the in-vivo acquisitions.

The settled workflow draws the septal M-line by hand on the buffer-5 B-mode (one frame per push)
and stores it as `output/mlines/active_meas{m}_mline.npz`. This module proposes that line
automatically so a batch can run unattended: it follows the bright septal ridge through the ARF
push focus and writes the same .npz format, which `swp.viz` / `swp_gui` then load unchanged.

Method (deliberately simple and inspectable):
  1. log-envelope buffer-5 frame, smoothed;
  2. a search band of +/- `half_mm` in depth around the push focus, over the lateral span
     `x_span_mm` around the push x;
  3. per lateral column, the depth of the brightest smoothed sample inside the band -> a ridge;
  4. columns whose ridge brightness is below the median are dropped (drop-out / shadow), and the
     surviving ridge is fitted with a low-order polynomial z(x);
  5. the polynomial is re-anchored so it passes exactly through the push focus depth at the push
     x (the wave origin r0 must sit on the line), and sampled at `n_points` anchors.

The proposal must be eyeballed against the B-mode (`--figure`) before the space-times built on it
are trusted; it is a starting point, not a replacement for the manual line.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in ("src", "swp_gui"):
    if os.path.join(_ROOT, _p) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _p))

import core                                                     # noqa: E402


def propose(folder, meas, push_x_mm, push_z_mm, half_mm=9.0, x_span_mm=26.0,
            order=2, n_points=7, smooth_mm=1.2):
    """Return the proposed M-line anchor points (k,2) = (x,z) in metres."""
    bmode = os.path.join(folder, "output", "CombinedData_buffer5_iq.hdf5")
    img, ext = core._bmode_h5py(bmode, meas)            # dB image, extent in mm (x0,x1,z1,z0)
    x = np.linspace(ext[0], ext[1], img.shape[1])
    z = np.linspace(ext[3], ext[2], img.shape[0])
    dx, dz = abs(x[1] - x[0]), abs(z[1] - z[0])
    sm = gaussian_filter(img, (smooth_mm / dz, smooth_mm / dx))

    cols = np.where(np.abs(x - push_x_mm) <= x_span_mm / 2)[0]
    rows = np.where(np.abs(z - push_z_mm) <= half_mm)[0]
    band = sm[np.ix_(rows, cols)]
    ridge_i = np.argmax(band, axis=0)
    ridge_z = z[rows][ridge_i]
    ridge_v = band[ridge_i, np.arange(band.shape[1])]

    keep = ridge_v >= np.median(ridge_v)
    if keep.sum() < order + 2:
        keep = np.ones_like(keep, dtype=bool)
    p = np.polyfit(x[cols][keep], ridge_z[keep], order)
    xs = np.linspace(x[cols][0], x[cols][-1], n_points)
    zs = np.polyval(p, xs)
    zs = zs + (push_z_mm - np.polyval(p, push_x_mm))     # anchor on the push focus depth
    return np.stack([xs, zs], axis=1) * 1e-3


def save_npz(folder, meas, points_xz_m, n_samples=250):
    out = os.path.join(folder, "output", "mlines")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"active_meas{meas}_mline.npz")
    np.savez(path, points=np.asarray(points_xz_m, dtype=float), n_samples=int(n_samples))
    return path


def _cli():
    import argparse
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scipy.io as sio

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--meas", default="all")
    ap.add_argument("--half-mm", type=float, default=9.0)
    ap.add_argument("--x-span-mm", type=float, default=26.0)
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--figure", default=None, help="write a check figure here")
    ap.add_argument("--save", action="store_true", help="write the .npz files")
    a = ap.parse_args()

    m = sio.loadmat(os.path.join(a.folder, "AcquisitionParametersAndECG.mat"),
                    squeeze_me=True, struct_as_record=False)["SW"] \
        if os.path.isfile(os.path.join(a.folder, "AcquisitionParametersAndECG.mat")) else None
    if m is not None:
        px, pz = float(m.FocusX_cm) * 10, float(m.FocusZ_cm) * 10
    else:                                                       # fall back to the stored IQ
        import h5py
        with h5py.File(os.path.join(a.folder, "output",
                                    "CombinedData_buffer2_meas0_iq.hdf5"), "r") as f:
            g = f
            px = float(np.array(g["custom/push_focus_x"])) * 1e3
            pz = float(np.array(g["custom/push_focus_z"])) * 1e3
    print(f"push focus = ({px:.2f}, {pz:.2f}) mm")

    import glob
    n = len(glob.glob(os.path.join(a.folder, "output", "CombinedData_buffer2_meas*_iq.hdf5")))
    meas = range(n) if a.meas == "all" else [int(a.meas)]

    pts = {}
    for k in meas:
        pts[k] = propose(a.folder, k, px, pz, half_mm=a.half_mm, x_span_mm=a.x_span_mm,
                         order=a.order)
        if a.save:
            save_npz(a.folder, k, pts[k])
    print(f"proposed {len(pts)} M-lines" + (" (saved)" if a.save else ""))

    if a.figure:
        show = list(pts)[:8]
        fig, axes = plt.subplots(2, 4, figsize=(16, 9))
        for k, ax in zip(show, axes.ravel()):
            img, ext = core._bmode_h5py(os.path.join(a.folder, "output",
                                                     "CombinedData_buffer5_iq.hdf5"), k)
            ax.imshow(img, extent=ext, cmap="gray", aspect="equal")
            P = pts[k] * 1e3
            ax.plot(P[:, 0], P[:, 1], "-o", color="lime", ms=3, lw=1.5)
            ax.plot(px, pz, "r+", ms=14, mew=2)
            ax.set_xlim(-45, 45)
            ax.set_ylim(80, 5)
            ax.set_title(f"push {k}", fontsize=9)
        fig.suptitle(f"Automatic septal M-line proposal - {os.path.basename(a.folder)[:40]}")
        fig.tight_layout()
        fig.savefig(a.figure, dpi=100)
        print("wrote", a.figure)


if __name__ == "__main__":
    _cli()
