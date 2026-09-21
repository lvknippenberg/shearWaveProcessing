"""Re-check the dynamic-range claim on a COMMON SUPPORT mask.

A bug found in round 6: ``sector_mask`` (|angle| <= 40 deg) contains a large number of pixels
that are exactly zero - outside the reconstructed sector, which is narrower than +/-40 deg at
shallow depth, and at the rim where the f-number mask removes every element. ``dynamic_range_db``
drops zeros before taking percentiles, so it never returned inf, but **the set of surviving
pixels differs between reconstructions**: a transmit window creates more zeros, and the ones it
creates are the darkest pixels, so the low percentile is taken over a different population.
That is a confound in the headline "+4 to +6 dB" number and it has to be removed before the
number can be trusted.

Fix: evaluate every reconstruction on ONE fixed mask, the intersection of
  * a conservative sector (|angle| <= 32 deg, 20 <= z <= 145 mm), and
  * pixels where EVERY candidate reconstruction is strictly positive,
and read the low end at the 5th percentile rather than the 1st, so a thin rim cannot drive it.
"""
from __future__ import annotations

import numpy as np

import b1_lib as L
import b1_masks as M
import b1_metrics as Q

TAGS = ("invivo", "invivo2", "invivo3", "phantom")


def candidates(geom):
    return [("all-21 (pipeline)", M.w_all(geom)),
            ("cone rect x1", M.w_cone(geom, "rect", 1.0)),
            ("cone rect x1.5", M.w_cone(geom, "rect", 1.5)),
            ("cone rect x2", M.w_cone(geom, "rect", 2.0)),
            ("cone tukey50 x1.5", M.w_cone(geom, "tukey50", 1.5))]


def common_mask(stack, geom, coords, half_deg=32.0, z=(20.0, 145.0)):
    m = Q.sector_mask(coords, half_deg=half_deg, z_min_mm=z[0])
    zz = Q.axes_mm(coords)[1]
    m &= (zz <= z[1])[:, None]
    for _, w in candidates(geom):
        env = np.abs(M.compose(stack, w)).mean(axis=0)
        m &= env > 0
    return m


def dr_p5(env, mask):
    v = env[mask]
    return float(20 * np.log10(np.percentile(v, 99.9) / np.percentile(v, 5)))


def dark_p10(frame, mask):
    """Depth-normalised p50/p10 of ONE frame on a fixed mask (rows / their own median)."""
    med = np.array([np.median(frame[i][mask[i]]) if mask[i].any() else np.nan
                    for i in range(frame.shape[0])])
    ok = np.isfinite(med) & (med > 0)
    v = (frame[ok] / med[ok, None])[mask[ok]]
    return float(20 * np.log10(np.percentile(v, 50) / np.percentile(v, 10)))


def main():
    for tag in TAGS:
        stack, meta = L.build_stack(tag)
        coords, geom = meta["coords"], L.geometry_from_meta(meta)
        mask = common_mask(stack, geom, coords)
        frac = mask.sum() / Q.sector_mask(coords).sum()
        print(f"\n[{tag}] common-support mask: {mask.sum()} px "
              f"({frac * 100:.0f}% of the |angle|<40 sector mask used before)")
        base = None
        print(f"  {'method':20s} {'dr p99.9/p5':>14s} {'dark p50/p10':>14s}")
        for lab, w in candidates(geom):
            env = np.abs(M.compose(stack, w))
            d = np.array([dr_p5(env[j], mask) for j in range(env.shape[0])])
            k = np.array([dark_p10(env[j], mask) for j in range(env.shape[0])])
            if base is None:
                base = (d.mean(), k.mean())
            print(f"  {lab:20s} {d.mean():8.1f}+-{d.std():3.1f} {k.mean():9.2f}+-{k.std():4.2f}"
                  + ("" if lab.startswith("all-21")
                     else f"   ({d.mean() - base[0]:+.1f} dB, {k.mean() - base[1]:+.2f} dB)"))


if __name__ == "__main__":
    main()
