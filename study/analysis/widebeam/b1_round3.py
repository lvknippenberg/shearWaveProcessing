"""Round 3: how much of the baseline image is off-beam energy, and can anything beat a cone?

Two parts.

**(a) The anti-mask measurement.**  Split the 21 transmits per pixel into the ones whose
geometric cone contains it and the ones whose cone does not, and reconstruct each set
separately.  The off-beam image contains no legitimate echo from that pixel by construction,
so ``20 log10(|off-beam| / |on-beam|)`` is an anatomy-free, ROI-free measurement of what the
current all-21 compound is adding.  Nothing here depends on a hand-drawn region or on a
display choice.

**(b) Beyond a geometric window.**  Three further rules on the same cached stack:
transmit-domain coherence factor, the cone multiplied by the simulated transmit field, and
incoherent (envelope) compounding of the cone-masked transmits.
"""
from __future__ import annotations

import numpy as np

import b1_lib as L
import b1_masks as M
import b1_metrics as Q
import b1_pfield as P
import b1_render as R
import b1_rois as RO


def anti_mask_report(tag, scale=1.0):
    stack, meta = L.build_stack(tag)
    coords, geom = meta["coords"], L.geometry_from_meta(meta)
    w_on = M.w_cone(geom, "rect", scale)
    w_off = 1.0 - w_on
    on = np.abs(M.compose(stack, w_on)).mean(axis=0)
    off = np.abs(M.compose(stack, w_off)).mean(axis=0)
    allx = np.abs(M.compose(stack, M.w_all(geom))).mean(axis=0)

    x, z = Q.axes_mm(coords)
    X, Z = np.meshgrid(x, z)
    ang = np.abs(np.degrees(np.arctan2(X, Z)))
    print(f"\n[{tag}] off-beam energy (transmits whose cone does NOT contain the pixel)")
    print(f"{'depth band':>14s} {'off/on dB':>10s} {'all/on dB':>10s} {'n_on':>6s}")
    for z0, z1 in ((10, 40), (40, 70), (70, 100), (100, 130), (130, 150)):
        m = (Z >= z0) & (Z < z1) & (ang < 35) & (on > 0)
        print(f"{z0:5d}-{z1:3d} mm  {20 * np.log10(off[m].mean() / on[m].mean()):10.2f} "
              f"{20 * np.log10(allx[m].mean() / on[m].mean()):10.2f} "
              f"{np.median(w_on.sum(axis=0)[m]):6.1f}")
    return coords, on, off, allx


def coherence_factor(stack, w, chunk=4, eps=1e-20):
    """Transmit-domain coherence factor: |sum w s|^2 / (N_eff sum w |s|^2), applied as a
    weight to the coherent compound.  Suppresses pixels where the contributing transmits
    disagree in phase - i.e. clutter - and leaves coherent echoes alone."""
    n_tx, n_fr, nz, nx = stack.shape
    coh = np.zeros((n_fr, nz, nx), np.complex64)
    inc = np.zeros((n_fr, nz, nx), np.float32)
    for s in range(0, n_tx, chunk):
        blk = np.asarray(stack[s:s + chunk])
        ww = w[s:s + chunk]
        coh += np.einsum("tzx,tfzx->fzx", ww, blk)
        inc += np.einsum("tzx,tfzx->fzx", ww, np.abs(blk) ** 2)
    neff = M.coverage(w)[None]
    cf = np.abs(coh) ** 2 / (np.maximum(neff, 1) * inc + eps)
    return coh * cf


def main():
    for tag in ("phantom", "invivo"):
        anti_mask_report(tag)

    tag = "invivo"
    stack, meta = L.build_stack(tag)
    coords, geom = meta["coords"], L.geometry_from_meta(meta)
    pfr = P.pfield_map(tag, norm=False)
    w1 = M.w_cone(geom, "rect", 1.0)
    w15 = M.w_cone(geom, "rect", 1.5)
    wt = M.w_cone(geom, "tukey50", 1.5)

    sets = [
        ("all-21 (pipeline)", lambda: np.abs(M.compose(stack, M.w_all(geom)))),
        ("cone rect x1", lambda: np.abs(M.compose(stack, w1))),
        ("cone rect x1.5", lambda: np.abs(M.compose(stack, w15))),
        ("cone tukey50 x1.5", lambda: np.abs(M.compose(stack, wt))),
        ("cone x1.5 * pfield", lambda: np.abs(M.compose(stack, w15 * M.w_pfield(geom, pfr)))),
        ("cone x1.5 + tx-CF", lambda: np.abs(coherence_factor(stack, w15))),
        ("all-21 + tx-CF", lambda: np.abs(coherence_factor(stack, M.w_all(geom)))),
        ("cone x1.5 incoherent", lambda: M.compose_incoherent(stack, w15)),
    ]

    rmask = {nm: Q.roi_mask(coords, xl, zl) for nm, xl, zl, _ in RO.INVIVO}
    pairs = [("dark_mid", "tissue_deep"), ("dark_upper", "tissue_near")]
    sector = Q.sector_mask(coords)
    fr = RO.INVIVO_FRAME
    panels = []
    print(f"\n{'method':24s} {'dr':>6s} {'gCNR':>6s} {'C/dB':>6s} {'spk':>5s}")
    print("-" * 52)
    for lab, fn in sets:
        env = fn()
        img = env[fr]
        g = np.mean([Q.gcnr(img[rmask[a]], img[rmask[b]]) for a, b in pairs])
        c = np.mean([abs(Q.contrast_db(img[rmask[b]], img[rmask[a]])) for a, b in pairs])
        print(f"{lab:24s} {Q.dynamic_range_db(img, sector):6.1f} {g:6.3f} {c:6.2f} "
              f"{Q.speckle_snr(img[rmask['speckle']]):5.2f}")
        panels.append((lab, img))
    R.montage(panels, coords, L.FIGS / "invivo_round3_matched.png", ncols=4,
              level_mode="matched", baseline_env=panels[0][1],
              title="in vivo buffer 1 - beyond a geometric window")


if __name__ == "__main__":
    main()
