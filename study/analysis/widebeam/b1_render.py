"""Rendering helpers: B-mode frames that look like the production GIFs, and montages.

Two level modes, because they answer different questions:

* ``own``    - each panel gets the pipeline's own per-clip adaptive levels
               (``swp.acquisition.gifs.iq_to_bmode``).  This is what the user would
               actually see, and it hides brightness differences (auto-gain).
* ``shared`` - all panels share the reference level derived from the BASELINE clip, so a
               genuine change in signal level, noise floor or dynamic range is visible.
"""
from __future__ import annotations

import numpy as np

import b1_lib as L  # noqa: F401  (sets sys.path for swp)
from swp.acquisition.gifs import DR_LIMITS, HI_PCT, LO_PCT
from swp.viz.tonecurves import apply_curve


def levels(env, gain_db=0.0):
    """(ref, dr) white point and dynamic range, exactly as the GIF renderer derives them."""
    inside = env[env > 0]
    ref = float(np.percentile(inside, HI_PCT)) / (10.0 ** (gain_db / 20.0))
    floor = float(np.percentile(inside, LO_PCT))
    ref = max(ref, 1e-12)
    dr = 20.0 * np.log10(ref / floor) if floor > 0 else DR_LIMITS[1]
    return ref, float(np.clip(dr, *DR_LIMITS))


def to_8bit(env, ref, dr, curve="gamma2"):
    with np.errstate(divide="ignore"):
        db = 20.0 * np.log10(env / ref + 1e-12)
    norm = np.clip((db + dr) / dr, 0.0, 1.0)
    return (apply_curve(norm, curve) * 255).astype(np.uint8)


def montage(panels, coords, out_path, ncols=4, title=None, level_mode="shared",
            baseline_env=None, rois=None, figsize_scale=3.1, dpi=110):
    """panels: list of (label, env2d).  Writes a PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = coords[0, :, 0] * 1e3
    z = coords[:, 0, -1] * 1e3
    extent = [x[0], x[-1], z[-1], z[0]]

    base = baseline_env if baseline_env is not None else panels[0][1]
    if level_mode in ("shared", "matched"):
        ref, dr = levels(base)
    # "matched": rescale every panel so its in-sector median equals the baseline's before
    # applying the shared levels.  A B-mode is auto-gained anyway, so absolute coherent gain
    # is not the question; what a shared display should expose is the ratio of tissue signal
    # to noise/clutter, and matching the medians is what isolates that.
    if level_mode == "matched":
        inside = base > 0
        bmed = np.median(base[inside])
        panels = [(lab, env * (bmed / (np.median(env[env > 0]) + 1e-20)))
                  for lab, env in panels]

    n = len(panels)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_scale * ncols,
                                                    figsize_scale * 1.35 * nrows),
                             squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for k, (label, env) in enumerate(panels):
        ax = axes[k // ncols][k % ncols]
        r, d = (ref, dr) if level_mode in ("shared", "matched") else levels(env)
        ax.imshow(to_8bit(env, r, d), cmap="gray", vmin=0, vmax=255,
                  extent=extent, aspect="equal")
        ax.set_title(label, fontsize=8)
        ax.axis("off")
        if rois:
            for (nm, xl, zl, col) in rois:
                ax.add_patch(plt.Rectangle((xl[0], zl[0]), xl[1] - xl[0], zl[1] - zl[0],
                                           fill=False, ec=col, lw=0.8))
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97 if title else 1))
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path
