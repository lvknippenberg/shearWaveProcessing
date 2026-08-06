"""Compare active-SWE space-time results between the 30 V and 40 V in-vivo folders.

Both folders were processed with the current 3-view active config (displacement only):
  A = "disp bp120-700 gauss mean3"  (top recipe)
  B = "disp bp80-500 gauss NO-temporal"
  C = "disp poly3 gauss mean3"
Reads every swp_meas{m}.hdf5, pulls origin_coherence per measurement per view, prints a table +
summary, and builds (1) an origin-coherence comparison figure and (2) a stacked space-time montage
(30 V row over 40 V row) for view A across all 24 pushes.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py

_REPO = r"D:/Luuk van Knippenberg/Github/shearWaveProcessing"
sys.path.insert(0, os.path.join(_REPO, "src"))
from swp.viz.core.geometry import robust_clim  # noqa: E402

ROOT = r"D:/Luuk van Knippenberg/Claude/2026_08_04 voltage sweep/Invivo"
FOLDERS = [
    ("30 V", os.path.join(ROOT, "Luuk30V_SW_data_04-August-2026_13-35-06")),
    ("40 V", os.path.join(ROOT, "Luuk40V_SW_data_04-August-2026_13-36-07")),
]
VIEWS = ["disp bp120-700 gauss mean3", "disp bp80-500 gauss NO-temporal", "disp poly3 gauss mean3"]
VSHORT = {"disp bp120-700 gauss mean3": "A bp120-700/mean3",
          "disp bp80-500 gauss NO-temporal": "B bp80-500/no-temp",
          "disp poly3 gauss mean3": "C poly3/mean3"}
PRIMARY = VIEWS[0]
NMEAS = 24


def load_oc(folder):
    out = {v: np.full(NMEAS, np.nan) for v in VIEWS}
    for m in range(NMEAS):
        h5 = os.path.join(folder, "output", "swp_active", f"swp_meas{m}.hdf5")
        if not os.path.exists(h5):
            continue
        with h5py.File(h5, "r") as f:
            g = f["displacement"]
            for v in VIEWS:
                if v in g:
                    out[v][m] = g[v].attrs.get("origin_coherence", np.nan)
    return out


def load_panel(h5_path, view):
    with h5py.File(h5_path, "r") as f:
        r = np.asarray(f["r_m"]) * 1e3
        r0 = float(f.attrs["r0_m"]) * 1e3
        g = f["displacement"]
        t = np.asarray(g["t_s"]) * 1e3
        d = g[view]
        data = np.asarray(d)
        oc = float(d.attrs.get("origin_coherence", np.nan))
    return data, r, t, r0, oc


def main():
    data = {lbl: load_oc(folder) for lbl, folder in FOLDERS}

    print("origin_coherence (displacement, view A = bp120-700/gauss/mean3), per push:")
    print(f"{'meas':>4}{'30 V':>9}{'40 V':>9}")
    for m in range(NMEAS):
        print(f"{m:>4}{data['30 V'][PRIMARY][m]:>9.3f}{data['40 V'][PRIMARY][m]:>9.3f}")

    print("\nSummary origin_coherence (mean / median / n>=0.7 of 24):")
    print(f"{'view':>22}" + "".join(f"{lbl+' mean':>12}{lbl+' med':>10}{lbl+' n>=.7':>10}"
                                    for lbl, _ in FOLDERS))
    for v in VIEWS:
        line = f"{VSHORT[v]:>22}"
        for lbl, _ in FOLDERS:
            oc = data[lbl][v]
            ocf = oc[np.isfinite(oc)]
            line += f"{np.nanmean(oc):>12.3f}{np.nanmedian(oc):>10.3f}{int((ocf>=0.7).sum()):>10}"
        print(line)

    # ---- Figure 1: origin_coherence comparison ----
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    colors = {"30 V": "tab:blue", "40 V": "tab:red"}
    ax = axs[0]
    for lbl, _ in FOLDERS:
        oc = data[lbl][PRIMARY]
        ax.plot(np.arange(NMEAS), oc, "-o", ms=4, color=colors[lbl],
                label=f"{lbl} (median {np.nanmedian(oc):.2f})")
    ax.set_xlabel("measurement index (push, ~cardiac phase order)")
    ax.set_ylabel("origin_coherence  (view A)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Per-push wave cleanliness (view A: bp120-700/gauss/mean3)")
    ax.grid(alpha=0.3); ax.legend()

    ax = axs[1]
    positions, tick_labels = [], []
    for j, v in enumerate(VIEWS):
        for k, (lbl, _) in enumerate(FOLDERS):
            pos = j * 3 + k
            vals = data[lbl][v]
            vals = vals[np.isfinite(vals)]
            bp = ax.boxplot([vals], positions=[pos], widths=0.7, patch_artist=True,
                            showmeans=True, meanprops=dict(marker="D", markerfacecolor="k",
                                                           markeredgecolor="k", markersize=5))
            for box in bp["boxes"]:
                box.set(facecolor=colors[lbl], alpha=0.55)
            x = np.random.normal(pos, 0.06, size=vals.size)
            ax.plot(x, vals, ".", color="0.2", ms=3, alpha=0.6)
            positions.append(pos); tick_labels.append(f"{lbl}\n{VSHORT[v].split()[0]}")
    ax.set_xticks(positions); ax.set_xticklabels(tick_labels, fontsize=7)
    ax.set_ylabel("origin_coherence (disp)"); ax.set_ylim(0, 1.02)
    ax.set_title("Distribution over 24 pushes (per view)")
    ax.grid(alpha=0.3, axis="y")
    fig.suptitle("In-vivo active SWE: shear-wave cleanliness vs push voltage (30 V vs 40 V)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out1 = os.path.join(ROOT, "invivo_voltage_oc_comparison.png")
    fig.savefig(out1, dpi=150); plt.close(fig)
    print(f"\nwrote {out1}")

    # ---- Figure 2: stacked space-time montage, view A ----
    clims = []
    for lbl, folder in FOLDERS:
        for m in range(NMEAS):
            h5 = os.path.join(folder, "output", "swp_active", f"swp_meas{m}.hdf5")
            if not os.path.exists(h5):
                continue
            d, r, t, r0, oc = load_panel(h5, PRIMARY)
            rc = (r > 0.1 * r[-1]) & (r < 0.9 * r[-1])
            clims.append(robust_clim(d, rc, pct=97) * 1e6)
    shared = float(np.percentile(clims, 75))

    fig, axs = plt.subplots(2, NMEAS, figsize=(1.35 * NMEAS, 5.2), squeeze=False)
    for i, (lbl, folder) in enumerate(FOLDERS):
        for m in range(NMEAS):
            ax = axs[i][m]
            h5 = os.path.join(folder, "output", "swp_active", f"swp_meas{m}.hdf5")
            if not os.path.exists(h5):
                ax.axis("off"); continue
            d, r, t, r0, oc = load_panel(h5, PRIMARY)
            extent = (r[0], r[-1], t[-1], t[0])
            ax.imshow(d * 1e6, extent=extent, cmap="RdBu_r", vmin=-shared, vmax=shared,
                      aspect="auto", origin="upper")
            ax.axvline(r0, color="0.2", ls="--", lw=0.6, alpha=0.6)
            ax.set_title(f"m{m} oc={oc:.2f}", fontsize=6)
            ax.tick_params(labelsize=5)
            if m == 0:
                ax.set_ylabel(f"{lbl}\nt [ms]", fontsize=8)
            else:
                ax.set_yticklabels([])
            if i == 1:
                ax.set_xlabel("r", fontsize=6)
            else:
                ax.set_xticklabels([])
    fig.suptitle("In-vivo active SWE space-time (view A: disp bp120-700/gauss/mean3; dashed=r0). "
                 "Top: 30 V, bottom: 40 V; one column per push; shared colour scale",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out2 = os.path.join(ROOT, "invivo_voltage_spacetime_montage.png")
    fig.savefig(out2, dpi=140); plt.close(fig)
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
