"""Round 4: does it generalise?  Three subjects, every frame, anatomy-free metrics.

Rounds 1-3 measured one frame of one subject against hand-drawn ROIs.  This round drops the
ROIs entirely and runs over every cached frame of three subjects plus the phantom.

Anatomy-free metrics used here:

``dark``  Depth-normalised dark-region contrast.  Each depth row is divided by its own median
          (removing TGC and the depth trend), then ``20 log10(p50 / p10)`` over the in-sector
          pixels.  It asks: how far below the typical tissue level does the DARK end of the
          image sit?  Clutter filling echo-free regions pushes p10 up and this number down.
          It needs no ROI, no segmentation, and no assumption about the anatomy.
``offb``  Off-beam energy: |compound of the transmits whose geometric cone EXCLUDES the pixel|
          over |compound of the ones that include it|, in dB, 70-100 mm.  A direct measurement
          of what the all-21 compound is adding.
``dr``    In-sector dynamic range, 99.9th/1st percentile, dB.
"""
from __future__ import annotations

import numpy as np

import b1_lib as L
import b1_masks as M
import b1_metrics as Q

TAGS = ("invivo", "invivo2", "invivo3", "phantom")


def dark_contrast(env, coords, sector, z_min=25.0):
    """20 log10(p50/p10) of the depth-detrended in-sector envelope, dB."""
    x, z = Q.axes_mm(coords)
    rows = env.copy()
    med = np.array([np.median(env[i][sector[i]]) if sector[i].any() else np.nan
                    for i in range(env.shape[0])])
    ok = np.isfinite(med) & (med > 0) & (z > z_min)
    rows = rows[ok] / med[ok, None]
    v = rows[sector[ok]]
    v = v[v > 0]
    return float(20 * np.log10(np.percentile(v, 50) / max(np.percentile(v, 10), 1e-12)))


def off_beam_db(stack, geom, coords, scale=1.0, band=(70, 100)):
    w_on = M.w_cone(geom, "rect", scale)
    on = np.abs(M.compose(stack, w_on)).mean(axis=0)
    off = np.abs(M.compose(stack, 1.0 - w_on)).mean(axis=0)
    x, z = Q.axes_mm(coords)
    X, Z = np.meshgrid(x, z)
    ang = np.abs(np.degrees(np.arctan2(X, Z)))
    m = (Z >= band[0]) & (Z < band[1]) & (ang < 35) & (on > 0)
    return float(20 * np.log10(off[m].mean() / on[m].mean()))


def main():
    print(f"{'dataset':9s} {'method':20s} {'dr/dB':>12s} {'dark/dB':>12s}")
    print("-" * 58)
    summary = {}
    for tag in TAGS:
        stack, meta = L.build_stack(tag)
        coords, geom = meta["coords"], L.geometry_from_meta(meta)
        sector = Q.sector_mask(coords)
        offb = off_beam_db(stack, geom, coords)
        sets = [("all-21 (pipeline)", M.w_all(geom)),
                ("cone rect x1", M.w_cone(geom, "rect", 1.0)),
                ("cone rect x1.5", M.w_cone(geom, "rect", 1.5)),
                ("cone tukey50 x1.5", M.w_cone(geom, "tukey50", 1.5)),
                ("cone rect x2", M.w_cone(geom, "rect", 2.0))]
        base = None
        for lab, w in sets:
            env = np.abs(M.compose(stack, w))
            dr = np.array([Q.dynamic_range_db(env[k], sector) for k in range(env.shape[0])])
            dk = np.array([dark_contrast(env[k], coords, sector) for k in range(env.shape[0])])
            if base is None:
                base = (dr.mean(), dk.mean())
            print(f"{tag:9s} {lab:20s} {dr.mean():7.1f}+-{dr.std():3.1f} "
                  f"{dk.mean():7.2f}+-{dk.std():4.2f}"
                  + ("" if lab.startswith("all-21")
                     else f"   ({dr.mean() - base[0]:+.1f} dB, {dk.mean() - base[1]:+.2f} dB)"))
            summary[(tag, lab)] = (dr.mean(), dk.mean())
        print(f"{tag:9s} {'off-beam 70-100mm':20s} {offb:7.1f} dB relative to on-beam\n")


if __name__ == "__main__":
    main()
