"""Round 7: two independent confirmations of where the chamber clutter comes from.

Both answer the same question - the structure that appears to run continuously across
z ~ 68-88 mm in the standard reconstruction, which the transmit cone breaks - and neither
depends on a display choice or on calling any region "anatomy" by eye.

**(a) Static: per-transmit origin.** For a region, how much amplitude does each of the 21
transmits deposit, and does its geometric cone even cover that region? A real structure should
be dominated by the transmits aimed at it. The chamber is not.

**(b) Time-resolved: what the clutter tracks.** The cached frames span a cardiac cycle. If the
chamber haze is radiated from the specular arc, it must rise and fall WITH the arc; if it were
chest-wall reverberation it would track the near field instead. The control - arc vs near field
are anti-correlated over the beat - rules out a global gain fluctuation producing the signs.
"""
from __future__ import annotations

import numpy as np

import b1_lib as L
import b1_masks as M
import b1_metrics as Q

TAG = "invivo"
ROIS = {
    "arc (x 5..45, z 68..88)": ((5, 45), (68, 88)),
    "chamber (x -50..-25, z 72..92)": ((-50, -25), (72, 92)),
    "near-field (x -18..10, z 26..40)": ((-18, 10), (26, 40)),
    "deep (x -20..20, z 135..148)": ((-20, 20), (135, 148)),
}


def per_transmit_origin(stack, geom, coords, out="invivo_pertx_roi.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    deg = np.degrees(geom.theta)
    w = M.w_cone(geom, "rect", 1.5)
    keys = list(ROIS)[:3]
    fig, axes = plt.subplots(1, len(keys), figsize=(5.7 * len(keys), 4.6))
    for ax, nm in zip(axes, keys):
        xl, zl = ROIS[nm]
        mk = Q.roi_mask(coords, xl, zl)
        amp = np.array([np.abs(np.asarray(stack[i])[:, mk]).mean() for i in range(geom.n_tx)])
        inside = np.array([w[i][mk].mean() for i in range(geom.n_tx)])
        ax.bar(deg, 20 * np.log10(amp / amp.max()), width=3,
               color=["tab:green" if v > 0.5 else "tab:red" for v in inside])
        ax.set_title(nm, fontsize=9)
        ax.set_xlabel("transmit steering angle (deg)")
        ax.set_ylabel("per-transmit amplitude in ROI (dB re max)")
        ax.set_ylim(-25, 2)
        ax.grid(alpha=0.3)
    fig.suptitle("green = this transmit's cone covers the ROI, red = it does not")
    fig.tight_layout()
    fig.savefig(L.FIGS / out, dpi=110)
    plt.close(fig)


def temporal_origin(stack, geom, coords, scale=1.5):
    A = np.abs(M.compose(stack, M.w_all(geom)))
    B = np.abs(M.compose(stack, M.w_cone(geom, "rect", scale)))
    B = B * (np.median(A[A > 0]) / np.median(B[B > 0]))
    removed = np.maximum(A - B, 0)
    mk = {nm: Q.roi_mask(coords, *ROIS[nm]) for nm in ROIS}
    arc = A[:, mk["arc (x 5..45, z 68..88)"]].mean(1)
    near = A[:, mk["near-field (x -18..10, z 26..40)"]].mean(1)
    ch = mk["chamber (x -50..-25, z 72..92)"]

    def r(a, b):
        a = (a - a.mean()) / a.std()
        b = (b - b.mean()) / b.std()
        return float((a * b).mean())

    print(f"n = {A.shape[0]} frames spanning the cardiac cycle "
          f"(|r| > 0.50 is p < 0.05 for n = 16)\n")
    print(f"{'series':34s} {'vs arc':>8s} {'vs near-field':>14s}")
    for lab, v in [("chamber, baseline (all-21)", A[:, ch].mean(1)),
                   ("the part the cone removes", removed[:, ch].mean(1)),
                   ("chamber, what the cone keeps", B[:, ch].mean(1))]:
        print(f"{lab:34s} {r(v, arc):+8.3f} {r(v, near):+14.3f}")
    print(f"{'control: arc vs near-field':34s} {'':8s} {r(arc, near):+14.3f}")


if __name__ == "__main__":
    stack, meta = L.build_stack(TAG)
    coords, geom = meta["coords"], L.geometry_from_meta(meta)
    per_transmit_origin(stack, geom, coords)
    temporal_origin(stack, geom, coords)
