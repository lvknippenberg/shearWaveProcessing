"""Caenen pig ARF-SWE data adapter for the GUI.

The Caenen data is not in the zea buffer-2 layout: it is the MATLAB-exported polar/Cartesian IQ in
``_h5_tmp/push_<p>.h5`` (see docs/caenen_vs_invivo_acquisition.md, scripts/sweep_caenen.py). This module
loads the **Cartesian** grid via swp_bridge and crops it to the drawn M-line ROI (so the GUI runs at
interactive speed, like sweep_caenen). The push is on-axis (x=0), so r0 is the M-line crossing of x=0.
"""
from __future__ import annotations

import dataclasses
import os
import sys

import numpy as np

H5DIR = r"D:/Luuk van Knippenberg/Claude/Data Caenen/SWE_results/_h5_tmp"
SWE = r"D:/Luuk van Knippenberg/Claude/Data Caenen/SWE_results"
if SWE not in sys.path:
    sys.path.insert(0, SWE)                              # for swp_bridge (load_push/build_acq)


def available():
    return os.path.isdir(H5DIR) and any(
        f.startswith("push_") and f.endswith(".h5") for f in os.listdir(H5DIR))


def pushes():
    ids = []
    for f in os.listdir(H5DIR):
        if f.startswith("push_") and f.endswith(".h5"):
            try:
                ids.append(int(f[5:-3]))
            except ValueError:
                pass
    return sorted(ids)


def _mline(push, acq):
    from swp.viz.mline.mline import mline_from_points, horizontal_mline
    p = os.path.join(SWE, f"push_{push}", "mline.npz")
    if os.path.exists(p):
        pts = np.load(p)["points"]
        return mline_from_points(np.asarray(pts, float), n_samples=250)
    return horizontal_mline(acq.x, float(acq.push_z), 250)   # fallback: horizontal at push depth


def _crop(acq, ml, margin=10e-3):
    """Crop the full-sector Cartesian volume to a box around the M-line (+ margin for offsets/filters)."""
    ix = np.where((acq.x >= ml.x.min() - margin) & (acq.x <= ml.x.max() + margin))[0]
    iz = np.where((acq.z >= ml.z.min() - margin) & (acq.z <= ml.z.max() + margin))[0]
    return dataclasses.replace(acq, x=acq.x[ix], z=acq.z[iz],
                               iq=acq.iq[:, iz][:, :, ix], ref_iq=acq.ref_iq[:, iz][:, :, ix])


def load(push):
    """Return (cropped Cartesian Acquisition, MLine) for a Caenen push."""
    from swp_bridge import load_push, build_acq
    Fp, Fc, zp, xp, zc, xc, taxis, params = load_push(os.path.join(H5DIR, f"push_{push}.h5"))
    acq = build_acq(Fc, zc, xc, taxis, params, "cart")
    ml = _mline(push, acq)
    return _crop(acq, ml), ml


def bmode(acq):
    """(u8 image, extent_mm) B-mode from a mid-reference frame envelope, for the GUI backdrop."""
    fr = acq.ref_iq[len(acq.ref_iq) // 2]
    env = np.abs(fr); env = env / (env.max() + 1e-12)
    u8 = (np.clip((20 * np.log10(env + 1e-6) + 60) / 60, 0, 1) * 255).astype(np.uint8)
    return u8, (acq.x[0] * 1e3, acq.x[-1] * 1e3, acq.z[-1] * 1e3, acq.z[0] * 1e3)
