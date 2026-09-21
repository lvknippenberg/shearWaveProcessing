"""Round 2: normalisation, the simulated transmit field, and an honest field-of-view check.

Round 1 (b1_eval.py) showed every geometric cone mask raising dynamic range by 6-8 dB while
the ``fov`` metric FELL by ~10 points.  Those two cannot both be read at face value: ``fov``
counts pixels above -50 dB of the clip's own bright end, so lowering the noise floor by 8 dB
drops pixels out of the count without removing any tissue.  This script separates them by
measuring the lateral extent of actual signal, depth by depth.
"""
from __future__ import annotations

import numpy as np

import b1_lib as L
import b1_masks as M
import b1_metrics as Q
import b1_pfield as P
import b1_render as R

TAG = "invivo"


def methods(geom, tag):
    pf = P.pfield_map(tag, norm=True)
    pfr = P.pfield_map(tag, norm=False)
    m = [("all-21 (pipeline)", M.w_all(geom), "none")]
    for norm in ("none", "sum", "rms"):
        m.append((f"cone rect x1", M.w_cone(geom, "rect", 1.0), norm))
    for norm in ("none", "rms"):
        m.append((f"cone hann x1.5", M.w_cone(geom, "hann", 1.5), norm))
        m.append((f"cone tukey50 x1", M.w_cone(geom, "tukey50", 1.0), norm))
        m.append(("pfield (norm)", M.w_pfield(geom, pf), norm))
        m.append(("pfield (raw)", M.w_pfield(geom, pfr), norm))
    m.append(("pfield mask -6dB", M.w_pfield_mask(geom, pfr, -6.0), "none"))
    m.append(("pfield mask -20dB", M.w_pfield_mask(geom, pfr, -20.0), "none"))
    return m


def lateral_profiles(stack, geom, coords, sets, depths=(40, 70, 100, 130), band_mm=6.0):
    """Mean envelope vs x at several depths, each curve scaled to its own sector median.

    This is the field-of-view question asked directly: does a mask make the sector narrower,
    or does it only lower the floor outside the signal?
    """
    x, z = Q.axes_mm(coords)
    out = {}
    for lab, w, norm in sets:
        env = np.abs(M.compose(stack, w, norm=norm)).mean(axis=0)
        scale = np.median(env[env > 0])
        curves = []
        for d in depths:
            sel = np.abs(z - d) <= band_mm / 2
            curves.append(env[sel].mean(axis=0) / scale)
        out[lab] = curves
    return x, np.asarray(depths), out


def main():
    stack, meta = L.build_stack(TAG)
    coords, geom = meta["coords"], L.geometry_from_meta(meta)
    sets = methods(geom, TAG)
    rois, pairs, spk = ({}, [], "")
    import b1_rois as RO
    rois = RO.INVIVO
    rmask = {nm: Q.roi_mask(coords, xl, zl) for nm, xl, zl, _ in rois}
    pairs = [("dark_mid", "tissue_deep"), ("dark_upper", "tissue_near")]
    sector = Q.sector_mask(coords)
    fr = RO.INVIVO_FRAME

    print(f"{'method':28s} {'norm':5s} {'cov':>5s} {'dr':>6s} {'gCNR':>6s} {'C/dB':>6s} "
          f"{'spk':>5s} {'fov%':>6s}")
    print("-" * 74)
    panels = []
    for lab, w, norm in sets:
        env = np.abs(M.compose(stack, w, norm=norm))
        img = env[fr]
        g = np.mean([Q.gcnr(img[rmask[a]], img[rmask[b]]) for a, b in pairs])
        c = np.mean([abs(Q.contrast_db(img[rmask[b]], img[rmask[a]])) for a, b in pairs])
        print(f"{lab:28s} {norm:5s} {np.median(M.coverage(w)[sector]):5.2f} "
              f"{Q.dynamic_range_db(img, sector):6.1f} {g:6.3f} {c:6.2f} "
              f"{Q.speckle_snr(img[rmask['speckle']]):5.2f} "
              f"{Q.sector_coverage(img, coords):6.1f}")
        panels.append((f"{lab} /{norm}" if norm != "none" else lab, img))

    R.montage(panels, coords, L.FIGS / f"{TAG}_round2_matched.png", ncols=5,
              level_mode="matched", baseline_env=panels[0][1],
              title="in vivo buffer 1 - normalisation and transmit-field weighting")
    R.montage(panels, coords, L.FIGS / f"{TAG}_round2_own.png", ncols=5, level_mode="own",
              title="in vivo buffer 1 - auto-gained")

    # --- field-of-view check -------------------------------------------------
    sel = [s for s in sets if s[0] in ("all-21 (pipeline)", "cone rect x1", "pfield (norm)")
           and s[2] in ("none", "rms")]
    sel = [sets[0], (("cone rect x1"), M.w_cone(geom, "rect", 1.0), "none"),
           ("cone rect x1 /rms", M.w_cone(geom, "rect", 1.0), "rms"),
           ("pfield (norm)", M.w_pfield(geom, P.pfield_map(TAG, norm=True)), "none")]
    x, depths, curves = lateral_profiles(stack, geom, coords, sel)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(depths), figsize=(4.1 * len(depths), 3.6), squeeze=False)
    for k, d in enumerate(depths):
        ax = axes[0][k]
        for lab in curves:
            ax.plot(x, 20 * np.log10(curves[lab][k] + 1e-6), lw=1.1, label=lab)
        ax.set_title(f"z = {d} mm")
        ax.set_xlabel("x (mm)")
        ax.set_ylim(-40, 25)
        ax.axvline(-d * np.tan(np.radians(40)), color="k", ls=":", lw=0.8)
        ax.axvline(+d * np.tan(np.radians(40)), color="k", ls=":", lw=0.8)
        ax.grid(alpha=0.3)
        if k == 0:
            ax.set_ylabel("dB re. sector median")
            ax.legend(fontsize=7)
    fig.suptitle("Lateral profile per depth, each curve scaled to its own sector median "
                 "(dotted = +/-40 deg sector edge)")
    fig.tight_layout()
    fig.savefig(L.FIGS / f"{TAG}_lateral_profiles.png", dpi=115)
    print(f"\nwrote {L.FIGS / (TAG + '_lateral_profiles.png')}")


if __name__ == "__main__":
    main()
