"""Round 6: clutter, specifically.

The user read the montage and spotted something no metric in rounds 1-5 was looking for: in the
standard reconstruction a structure appears to run continuously left-to-right across z ~ 68-88 mm,
and under the transmit cone it does not - so the left half of it is not anatomy. That is off-axis
clutter from the specular arc on the right, and it is the failure mode worth attacking.

Two levers beyond the transmit cone, both "which data contributes to this pixel":

* **Receive aperture.** The pipeline runs f/1 with zea's default rectangular f-number mask. At
  100 mm depth the f/1 acceptance cone is +/-26.6 deg while the 20 mm array only subtends
  +/-5.7 deg, so no element is ever masked and the receive aperture is effectively RECTANGULAR -
  -13 dB sidelobes, straight into the dark regions. Tapering it (fixed, or dynamic at a realistic
  f-number) is the receive-side twin of the transmit cone.
* **Moderated coherence weighting.** Round 3 rejected the transmit-domain coherence factor at
  full strength (gCNR 0.40 -> 0.24). CF^gamma with gamma << 1 is the standard way to keep the
  clutter suppression without destroying speckle; that was never tried.

Metrics here are clutter-specific:

``arc``    contrast between the specular arc and the chamber beside it at the same depth, dB.
           Peak over z in the band for the arc; 25th percentile over the chamber box. This is
           the number that moves when a false lateral extension is removed.
``dark``   depth-normalised p50/p10 over the sector (anatomy-free, as in round 4).
``spk``    speckle SNR in the near-field tissue ROI - the guard against "improved by smoothing".
"""
from __future__ import annotations

import numpy as np

import b1_lib as L
import b1_masks as M
import b1_metrics as Q
import b1_render as R
import b1_rois as RO
import b1_recheck as RC

TAG = "invivo"
BAND = (68.0, 88.0)              # depth band of the flagged structure
ARC_X = (5.0, 45.0)              # the real specular arc
CHAMBER_X = (-50.0, -25.0)       # where the standard reconstruction extends it
CHAMBER_Z = (72.0, 92.0)       # kept well inside the sector: |x| < z*tan(40deg) everywhere


def arc_contrast(env, coords):
    """20 log10(arc peak / chamber floor), dB."""
    x, z = Q.axes_mm(coords)
    band = (z >= BAND[0]) & (z <= BAND[1])
    arc_cols = (x >= ARC_X[0]) & (x <= ARC_X[1])
    arc = env[band][:, arc_cols].max(axis=0).mean()
    ch = Q.roi_mask(coords, CHAMBER_X, CHAMBER_Z) & (env > 0)
    floor = np.percentile(env[ch], 25)
    return float(20 * np.log10(arc / max(floor, 1e-20)))


def cf_weighted(stack, w, gamma, chunk=4, eps=1e-20):
    """Coherent compound scaled by CF^gamma (transmit-domain coherence factor)."""
    n_tx, n_fr, nz, nx = stack.shape
    coh = np.zeros((n_fr, nz, nx), np.complex64)
    inc = np.zeros((n_fr, nz, nx), np.float32)
    for s in range(0, n_tx, chunk):
        blk = np.asarray(stack[s:s + chunk])
        ww = w[s:s + chunk]
        coh += np.einsum("tzx,tfzx->fzx", ww, blk)
        inc += np.einsum("tzx,tfzx->fzx", ww, np.abs(blk) ** 2)
    neff = np.maximum(M.coverage(w), 1)[None]
    cf = np.clip(np.abs(coh) ** 2 / (neff * inc + eps), 0.0, 1.0)
    return coh * (cf ** gamma)


def main():
    _, meta = L.build_stack(TAG)
    coords, geom = meta["coords"], L.geometry_from_meta(meta)
    stack0, _ = L.build_stack(TAG)
    mask = RC.common_mask(stack0, geom, coords)   # fixed common support (see b1_recheck)
    rmask = {nm: Q.roi_mask(coords, xl, zl) for nm, xl, zl, _ in RO.INVIVO}
    pairs = [("dark_mid", "tissue_deep"), ("dark_upper", "tissue_near")]
    fr = RO.INVIVO_FRAME

    w_all = M.w_all(geom)
    w_cone = M.w_cone(geom, "rect", 1.5)
    w_cone1 = M.w_cone(geom, "rect", 1.0)

    # (label, rx setting, transmit weight, post)
    cases = [
        ("1. all-21, f/1 rect  (pipeline)", None, w_all, None),
        ("2. cone x1.5", None, w_cone, None),
        ("3. cone x1.0", None, w_cone1, None),
        ("4. rx hann (fixed)", dict(apod="hann"), w_all, None),
        ("5. cone x1.5 + rx hann", dict(apod="hann"), w_cone, None),
        ("6. cone x1.5 + rx tukey50", dict(apod="tukey50"), w_cone, None),
        ("7. cone x1.5 + rx hann dyn f/2.5", dict(apod="hann", mode="dynamic", f=2.5),
         w_cone, None),
        ("8. cone x1.5 + rx hann dyn f/1", dict(apod="hann", mode="dynamic", f=1.0),
         w_cone, None),
        ("9. cone x1.5, f/2", dict(f=2.0), w_cone, None),
        ("10. cone x1.5, f/3", dict(f=3.0), w_cone, None),
        ("11. cone x1.5 + CF^0.15", None, w_cone, 0.15),
        ("12. cone x1.5 + CF^0.30", None, w_cone, 0.30),
        ("13. cone x1.5 + CF^0.50", None, w_cone, 0.50),
        ("14. cone+rx hann + CF^0.30", dict(apod="hann"), w_cone, 0.30),
    ]

    print(f"{'reconstruction':34s} {'arc/dB':>7s} {'dark':>6s} {'dr':>6s} {'gCNR':>6s} "
          f"{'spk':>5s}")
    print("-" * 72)
    panels, base_img = [], None
    for lab, rx, w, gam in cases:
        stack, _ = L.build_stack(TAG, rx=rx)
        env = np.abs(cf_weighted(stack, w, gam)) if gam else np.abs(M.compose(stack, w))
        img = env[fr]
        if base_img is None:
            base_img = img
        g = np.mean([Q.gcnr(img[rmask[a]], img[rmask[b]]) for a, b in pairs])
        print(f"{lab:34s} {arc_contrast(img, coords):7.2f} "
              f"{RC.dark_p10(img, mask):6.2f} "
              f"{RC.dr_p5(img, mask):6.1f} {g:6.3f} "
              f"{Q.speckle_snr(img[rmask['speckle']]):5.2f}")
        panels.append((lab, img))

    R.montage(panels, coords, L.FIGS / "invivo_round6_matched.png", ncols=5,
              level_mode="matched", baseline_env=base_img,
              title="in vivo buffer 1 - receive aperture and coherence weighting on top of "
                    "the transmit cone")
    band_zoom(panels, coords)


def band_zoom(panels, coords, out="invivo_round6_band.png"):
    """Zoom on the flagged band for every case, plus the arc peak profile."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x, z = Q.axes_mm(coords)
    band = (z >= 64) & (z <= 96)
    ext = [x[0], x[-1], 96, 64]
    base = panels[0][1]
    bmed = np.median(base[base > 0])
    ref, dr = R.levels(base)
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(11, 1.7 * n), squeeze=False)
    for ax, (lab, img) in zip(axes[:, 0], panels):
        im = img * (bmed / np.median(img[img > 0]))
        ax.imshow(R.to_8bit(im[band], ref, dr), cmap="gray", vmin=0, vmax=255,
                  extent=ext, aspect="auto")
        ax.set_ylabel(lab, fontsize=6.5, rotation=0, ha="right", va="center")
        ax.set_xlim(-75, 70)
        ax.set_yticks([])
        ax.set_xticks(np.arange(-70, 71, 10))
        ax.tick_params(labelsize=6)
    fig.suptitle("z 64-96 mm, medians matched, shared levels: the structure that looks "
                 "continuous left-to-right in row 1", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(L.FIGS / out, dpi=125)
    plt.close(fig)


if __name__ == "__main__":
    main()
