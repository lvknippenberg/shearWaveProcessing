"""Figures for the buffer-1 widebeam reconstruction write-up."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import b1_lib as L
import b1_masks as M
import b1_metrics as Q
import b1_render as R


def fig_geometry():
    """What one widebeam covers, and how many transmits reach each pixel."""
    _, meta = L.build_stack("invivo")
    coords, geom = meta["coords"], L.geometry_from_meta(meta)
    x, z = Q.axes_mm(coords)
    ext = [x[0], x[-1], z[-1], z[0]]
    w = M.w_cone(geom, "rect", 1.0)
    cov = w.sum(axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    axes[0].imshow(w[10], extent=ext, aspect="equal", cmap="magma", vmin=0, vmax=1)
    axes[0].set_title("one transmit's geometric cone\n(tx 11 of 21, steered 0 deg)")
    axes[1].imshow(w[[4, 10, 16]].sum(axis=0), extent=ext, aspect="equal", cmap="magma")
    axes[1].set_title("three transmits (-16, 0, +16 deg)\nshowing the overlap")
    im = axes[2].imshow(cov, extent=ext, aspect="equal", cmap="viridis", vmin=0, vmax=21)
    axes[2].set_title("transmits that insonify each pixel\n(of 21; the pipeline uses all 21)")
    plt.colorbar(im, ax=axes[2], fraction=0.04)
    for a in axes:
        a.set_xlabel("x (mm)")
        a.set_ylabel("z (mm)")
    fig.suptitle("Buffer 1: 21 widebeams, 4 deg apart, virtual source 123 mm behind the array")
    fig.tight_layout()
    fig.savefig(L.FIGS / "fig_geometry.png", dpi=115)
    plt.close(fig)


def fig_offbeam():
    """The image the OFF-beam transmits alone produce - phantom vs in vivo."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 11))
    for row, tag in enumerate(("phantom", "invivo")):
        stack, meta = L.build_stack(tag)
        coords, geom = meta["coords"], L.geometry_from_meta(meta)
        w = M.w_cone(geom, "rect", 1.0)
        fr = 0
        on = np.abs(M.compose(stack, w))[fr]
        off = np.abs(M.compose(stack, 1.0 - w))[fr]
        allx = np.abs(M.compose(stack, M.w_all(geom)))[fr]
        x, z = Q.axes_mm(coords)
        ext = [x[0], x[-1], z[-1], z[0]]
        ref, dr = R.levels(allx)
        for ax, (lab, img) in zip(axes[row], [("all 21 transmits (pipeline)", allx),
                                              ("on-beam transmits only", on),
                                              ("OFF-beam transmits only", off)]):
            ax.imshow(R.to_8bit(img, ref, dr), cmap="gray", vmin=0, vmax=255,
                      extent=ext, aspect="equal")
            ax.set_title(f"{tag}: {lab}", fontsize=10)
            ax.set_xlim(-90, 90)
            ax.axis("off")
    fig.suptitle("Same display levels within each row. The off-beam panel contains no echo "
                 "that transmit could have produced:\non the phantom it is 18 dB down, in vivo "
                 "it is 5 dB down.", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(L.FIGS / "fig_offbeam.png", dpi=105)
    plt.close(fig)


def fig_headline():
    """Baseline vs cone mask, three subjects, auto-gained as they would ship."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    for col, tag in enumerate(("invivo", "invivo2", "invivo3")):
        stack, meta = L.build_stack(tag)
        coords, geom = meta["coords"], L.geometry_from_meta(meta)
        x, z = Q.axes_mm(coords)
        ext = [x[0], x[-1], z[-1], z[0]]
        fr = 0
        for row, (lab, w) in enumerate([("all-21 (pipeline)", M.w_all(geom)),
                                        ("cone rect x1.5", M.w_cone(geom, "rect", 1.5))]):
            img = np.abs(M.compose(stack, w))[fr]
            ref, dr = R.levels(img)
            ax = axes[row][col]
            ax.imshow(R.to_8bit(img, ref, dr), cmap="gray", vmin=0, vmax=255,
                      extent=ext, aspect="equal")
            ax.set_title(f"{tag} - {lab}", fontsize=10)
            ax.set_xlim(-90, 90)
            ax.axis("off")
    fig.suptitle("Buffer 1, three subjects: all-21 compound (top) vs transmit-cone mask "
                 "(bottom), each auto-gained by the pipeline's own rule")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(L.FIGS / "fig_headline.png", dpi=105)
    plt.close(fig)


def fig_tradeoff():
    """Dynamic-range gain vs lateral cost, across mask widths and the rank-based rules."""
    stack, meta = L.build_stack("phantom")
    coords, geom = meta["coords"], L.geometry_from_meta(meta)
    sector = Q.sector_mask(coords)
    base = np.abs(M.compose(stack, M.w_all(geom))).mean(axis=0)
    targets = Q.find_targets(base, coords)
    x, z = Q.axes_mm(coords)
    dx, dz = abs(x[1] - x[0]), abs(z[1] - z[0])

    def paired(env):
        d = []
        for iz, ix in targets:
            a, b = Q.fwhm(base[iz], dx, ix), Q.fwhm(env[iz], dx, ix)
            if a and b:
                d.append(b / a - 1)
        return np.median(d) * 100

    base_dr = Q.dynamic_range_db(base, sector)
    pts = []
    for s in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0):
        env = np.abs(M.compose(stack, M.w_cone(geom, "rect", s))).mean(axis=0)
        pts.append((f"cone x{s:g}", paired(env), Q.dynamic_range_db(env, sector) - base_dr))
    kpts = []
    for k in (1, 2, 3, 5, 7, 11, 15):
        env = np.abs(M.compose(stack, M.w_nearest_k(geom, k))).mean(axis=0)
        kpts.append((f"k={k}", paired(env), Q.dynamic_range_db(env, sector) - base_dr))

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for pp, c, mk, nm in ((pts, "tab:blue", "o", "geometric cone, width x scale"),
                          (kpts, "tab:orange", "s", "nearest-k transmits")):
        ax.plot([p[1] for p in pp], [p[2] for p in pp], mk + "-", color=c, label=nm)
        for lab, a, b in pp:
            ax.annotate(lab, (a, b), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(0, color="k", lw=0.8)
    ax.plot(0, 0, "k*", ms=12, label="all-21 (pipeline)")
    ax.set_xlabel("lateral -6 dB width, paired change vs the pipeline (%)")
    ax.set_ylabel("in-sector dynamic range, change vs the pipeline (dB)")
    ax.set_title("Resolution phantom, 66 wire targets: what each pixel-inclusion rule buys")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(L.FIGS / "fig_tradeoff.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    fig_geometry(); print("geometry")
    fig_offbeam(); print("offbeam")
    fig_headline(); print("headline")
    fig_tradeoff(); print("tradeoff")
