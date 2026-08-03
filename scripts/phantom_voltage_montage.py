"""Voltage-sweep montage for the phantom ARF shear-wave measurements.

Auto-discovers the measurement subfolders under ``--root`` (acquisition order), reads the
per-folder space-time maps written by `run.py viz --phantom`
(`output/swp_active/swp_meas{m}.hdf5`), and lays them out side by side so the shear wave can be
compared across push voltage. Displacement (top row) and velocity (bottom row) for the recipe
band (default bp120-700). Each column is labelled with the *delivered* push voltage read from
the data (so a clamped sweep is shown honestly, not by the intended label).

Usage:
  python scripts/phantom_voltage_montage.py --root "<Phantom parent folder>" [--out montage.png]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
from swp.viz.core.geometry import robust_clim          # noqa: E402
from swp.acquisition import discover_measurements       # noqa: E402


def _load_panel(h5_path, quantity, band_tag):
    """Return (data (n_t,n_r), r_mm, t_ms, r0_mm, origin_coherence) for one panel."""
    with h5py.File(h5_path, "r") as f:
        r = np.asarray(f["r_m"]) * 1e3
        r0 = float(f.attrs["r0_m"]) * 1e3
        g = f[quantity]
        t = np.asarray(g["t_s"]) * 1e3
        d = g[band_tag]
        data = np.asarray(d)
        oc = float(d.attrs.get("origin_coherence", np.nan))
    return data, r, t, r0, oc


def _panel_clim(data, r_mm, unit):
    rc = (r_mm > 0.1 * r_mm[-1]) & (r_mm < 0.9 * r_mm[-1])
    return robust_clim(data, rc, pct=97) * unit


def _draw(ax, data, r_mm, t_ms, r0_mm, quantity, title, clim):
    unit = 1e3 if quantity == "velocity" else 1e6      # mm/s or um
    img = data * unit
    extent = (r_mm[0], r_mm[-1], t_ms[-1], t_ms[0])
    ax.imshow(img, extent=extent, cmap="RdBu_r", vmin=-clim, vmax=clim,
              aspect="auto", origin="upper")
    ax.axvline(r0_mm, color="0.2", ls="--", lw=0.8, alpha=0.6)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="Phantom folder containing the measurement subfolders")
    p.add_argument("--meas", type=int, default=0, help="measurement index within each folder (default 0)")
    p.add_argument("--band", default="bp120-700", help="band tag stored in the hdf5 (default bp120-700)")
    p.add_argument("--out", default=None, help="output PNG (default <root>/phantom_voltage_montage.png)")
    p.add_argument("--per-panel-clim", action="store_true",
                   help="auto-scale each panel independently (default: one shared absolute scale per "
                        "row, so the amplitude growth with voltage is visible)")
    a = p.parse_args()

    out = a.out or os.path.join(a.root, "phantom_voltage_montage.png")
    cols = []
    for folder, pv in discover_measurements(a.root):
        h5 = os.path.join(folder, "output", "swp_active", f"swp_meas{a.meas}.hdf5")
        if not os.path.exists(h5):
            print(f"  missing (skipped): {os.path.basename(folder)} -> {h5}")
            continue
        # Label with the DELIVERED push voltage; flag any that sat at the profile ceiling.
        label = pv.label()
        cols.append((label, h5))
        print(f"  {os.path.basename(folder)}: push {pv.hv} V (limit {pv.high_voltage_limit})"
              + ("  <-- AT PROFILE MAX" if pv.at_profile_max else ""))
    if not cols:
        raise SystemExit("no per-folder swp_meas hdf5 files found; run the phantom viz first")

    quantities = ("displacement", "velocity")
    # Load every panel once; compute a shared absolute colour scale per row so a stronger
    # (higher-voltage) wave reads as stronger contrast instead of being re-normalised away.
    panels = {q: [] for q in quantities}
    for label, h5 in cols:
        for q in quantities:
            data, r, t, r0, oc = _load_panel(h5, q, a.band)
            unit = 1e3 if q == "velocity" else 1e6
            panels[q].append((label, data, r, t, r0, oc, _panel_clim(data, r, unit)))
    # Shared scale = 75th percentile of the per-panel robust clims (a mid-strong panel sets it,
    # so weak low-voltage panels stay faint and strong high-voltage panels saturate a little).
    shared = {q: float(np.percentile([p[6] for p in panels[q]], 75)) for q in quantities}

    ncols = len(cols)
    fig, axs = plt.subplots(2, ncols, figsize=(2.5 * ncols, 6.2), squeeze=False)
    for i, q in enumerate(quantities):
        clim = None if a.per_panel_clim else shared[q]
        for j, (label, data, r, t, r0, oc, pclim) in enumerate(panels[q]):
            title = (f"{label}\n{q[:4]}  oc={oc:.2f}") if i == 0 else f"{q[:4]}  oc={oc:.2f}"
            _draw(axs[i][j], data, r, t, r0, q, title, clim if clim is not None else pclim)
            if j == 0:
                axs[i][j].set_ylabel("t [ms]", fontsize=8)
            if i == 1:
                axs[i][j].set_xlabel("r [mm]", fontsize=8)
    scale_note = "per-panel auto-scale" if a.per_panel_clim else "shared colour scale per row"
    fig.suptitle(f"Phantom ARF shear wave vs push voltage  (band {a.band}, meas{a.meas}; "
                 f"dashed = push origin r0; {scale_note})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
