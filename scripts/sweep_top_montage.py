"""Render the sweep's TOP recipe per voltage as a space-time montage, with the locked V-template
overlaid, so the extracted shear-wave "V" and its degradation with SNR (transmit voltage) is visible.

Reads sweep_results.csv, picks the #1 recipe by ROI-contrast (band1) at each voltage, rebuilds its
best-quantity space-time with the exact sweep processing, and lays them out high->low voltage with the
V-template ridge drawn on top and both detector scores in each title.

    <zea-python> scripts/sweep_top_montage.py [--metric roi1_best] [--csv sweep_results.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "swp_gui"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import sweep_extract as sw                                # noqa: E402
from swp.viz.core.geometry import robust_clim            # noqa: E402

VORDER = ["50V", "45V", "40V", "35V", "30V", "25V", "20V", "15V"]
QFULL = {"dis": "displacement", "vel": "velocity", "acc": "acceleration"}
UNIT = {"displacement": 1e6, "velocity": 1e3, "acceleration": 1.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="roi1_best")
    ap.add_argument("--csv", default=os.path.join(sw.BASE, "sweep_results.csv"))
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
    recipes = json.load(open(a.csv.replace(".csv", "_recipes.json"), encoding="utf-8"))
    tmpl = json.load(open(os.path.join(sw.BASE, "v_roi_template.json"), encoding="utf-8"))

    fig, axs = plt.subplots(2, 4, figsize=(17, 8))
    for ax, v in zip(axs.ravel(), VORDER):
        rr = [r for r in rows if r["voltage"] == v]
        best = max(rr, key=lambda r: float(r[a.metric]))
        rid = best["recipe_id"]; quant = QFULL[best[a.metric.replace("_best", "_q")]]
        rec = {**recipes[rid], "sm": best["sm"], "tm": best["tm"], "f_lo": 0, "f_hi": 0}
        acq, ml, r0 = sw.load_voltage(v)
        est = sw.estimator_for_iq(acq, rec["iq"])
        st = sw.spacetime_for(est, acq, ml, r0, rec, quant)
        u = UNIT[quant]
        rc = (st.r > 0.1 * st.r[-1]) & (st.r < 0.9 * st.r[-1])
        cl = (robust_clim(st.data, rc, 97) * u) or 1.0
        r = st.r * 1e3; t = st.t * 1e3
        ax.imshow(st.data * u, extent=(r[0], r[-1], t[-1], t[0]), cmap="RdBu_r",
                  vmin=-cl, vmax=cl, aspect="auto", origin="upper")
        line = (tmpl["t0_s"] + np.abs(st.r - r0) / tmpl["c_mps"]) * 1e3
        for b in (0, tmpl["band_s"] * 1e3, -tmpl["band_s"] * 1e3):
            ax.plot(st.r * 1e3, line + b, "k", ls="-" if b == 0 else ":", lw=1.0)
        ax.axvline(r0 * 1e3, color="0.3", ls="--", lw=0.8)
        ax.set_title(f"{v}  id{rid} ({quant[:3]}, {rec['sm']})\n"
                     f"roi1={float(best['roi1_best']):.2f} roi2={float(best['roi2_best']):.2f} "
                     f"sym={float(best['sym_best']):.2f}", fontsize=9)
        ax.set_xlabel("r [mm]", fontsize=8); ax.tick_params(labelsize=7)
    for ax in axs[:, 0]:
        ax.set_ylabel("t [ms]", fontsize=9)
    fig.suptitle("Top extraction recipe per transmit voltage (SNR), with locked V-template overlaid",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(sw.BASE, "sweep_top_montage.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
