"""Summary of the 2D wave maps (passive_wave_map.py --batch): how the wave really travels.

Reads study/montages/passive_wave_map/*_metrics.json. Per event: the hand slope along the M-line,
the arrival-time slope along the same line (c_line_tau, from the 2D arrival map), the plane-fit 2D
speed and direction (c2d, theta to the line), the along-line speed that direction predicts
(c_line_pred = c2d / cos theta) and where the wave starts (origin: off the line / along it).

-> study/logs/passive_wave_map/{metrics.csv, summary.txt}, study/montages/passive_wave_map/summary.png
"""
from __future__ import annotations

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402
import numpy as np                                              # noqa: E402
import pandas as pd                                             # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(REPO, "study", "montages", "passive_wave_map")
LOGS = os.path.join(REPO, "study", "logs", "passive_wave_map")
COL = {"MVC": "#2a78d6", "AVC": "#eb6834", "AK": "#1f9e6e"}
INK2, SURF = "#52514e", "#fcfcfb"


def main():
    os.makedirs(LOGS, exist_ok=True)
    M = pd.DataFrame([json.load(open(f)) for f in sorted(glob.glob(os.path.join(SRC, "*_metrics.json")))])
    M["hand"] = M.hand_m_s.abs()
    M.to_csv(os.path.join(LOGS, "metrics.csv"), index=False)
    ok = M[M.c2d.notna() & (M.r2 >= 0.5)]
    L = [f"{len(M)} events ({', '.join(f'{k} {v}' for k, v in M.label.value_counts().items())}); "
         f"plane fit R2 >= 0.5 in {len(ok)} (median R2 {M.r2.median():.2f}, coherent fraction of the septum "
         f"band median {M.frac_coherent.median():.2f})"]

    def ratio(a, b):
        r = (a / b).replace([np.inf, -np.inf], np.nan).dropna()
        return f"median {r.median():.2f} (IQR {r.quantile(.25):.2f}-{r.quantile(.75):.2f}, n={r.size})"
    L.append(f"arrival slope along the line / hand slope: {ratio(M.c_line_tau.abs(), M.hand)}")
    L.append(f"direction-predicted along-line speed / hand: {ratio(ok.c_line_pred.abs(), ok.hand)}")
    L.append(f"2D speed / hand: {ratio(ok.c2d, ok.hand)}")
    for lab, g in ok.groupby("label"):
        L.append(f"  {lab} (n={len(g)}): 2D speed median {g.c2d.median():.2f} m/s, hand {g.hand.median():.2f} m/s, "
                 f"angle to the line median {g.theta_deg.median():.0f} deg (IQR {g.theta_deg.quantile(.25):.0f}-"
                 f"{g.theta_deg.quantile(.75):.0f}), origin {g.origin_off_line_mm.median():.0f} mm off the line, "
                 f"{g.origin_r_mm.median():.0f} mm along it")
    L.append(f"angle > 30 deg (along-line speed >= 15 % above the 2D speed): {(ok.theta_deg > 30).sum()}/{len(ok)}; "
             f"wave travelling towards r = 0 (angle > 90 deg): {(ok.theta_deg > 90).sum()}")
    c11 = M[(M.subject == "C000000011") & (M.window == 0)]
    for r in c11.itertuples():
        L.append(f"C000000011 two-slope MVC: hand {r.hand:.2f}, arrival slope on the line {abs(r.c_line_tau):.2f}, 2D "
                 f"{r.c2d:.2f} m/s at {r.theta_deg:.0f} deg (R2 {r.r2:.2f}), origin {r.origin_off_line_mm:.0f} mm off the "
                 f"line at r = {r.origin_r_mm:.0f} mm")
    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(LOGS, "summary.txt"), "w") as fh:
        fh.write(txt + "\n")

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), facecolor=SURF)
    for lab, g in M.groupby("label"):
        ax[0].scatter(g.hand, g.c_line_tau.abs().clip(upper=15), color=COL.get(lab, INK2), s=26, label=lab)
    ax[0].plot([0, 10], [0, 10], color=INK2, lw=0.8, ls="--")
    ax[0].set_xlabel("hand slope [m/s]")
    ax[0].set_ylabel("arrival-time slope along the line [m/s]")
    ax[0].set_title("The 2D arrival map reproduces the hand slope", fontsize=9, loc="left")
    ax[0].legend(frameon=False, fontsize=8)
    for lab, g in ok.groupby("label"):
        ax[1].scatter(g.theta_deg, g.c2d / g.hand, color=COL.get(lab, INK2), s=26, label=lab)
    th = np.linspace(0, 80, 100)
    ax[1].plot(th, np.cos(np.radians(th)), color=INK2, lw=1, ls="--", label="cos(angle)")
    ax[1].set_xlabel("angle between propagation and the M-line [deg]")
    ax[1].set_ylabel("2D speed / hand slope")
    ax[1].set_title("Oblique propagation: along-line speed > true speed", fontsize=9, loc="left")
    ax[1].legend(frameon=False, fontsize=8)
    for lab, g in ok.groupby("label"):
        ax[2].scatter(g.origin_r_mm, g.origin_off_line_mm, color=COL.get(lab, INK2), s=26, label=lab)
    ax[2].axvline(0, color=INK2, lw=0.8, ls=":")
    ax[2].set_xlabel("origin along the line from r = 0 [mm]")
    ax[2].set_ylabel("origin distance from the line [mm]")
    ax[2].set_title("Where the wave first appears (earliest 2 % of pixels)", fontsize=9, loc="left")
    for a in ax:
        a.set_facecolor(SURF)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(SRC, "summary.png"), dpi=100, facecolor=SURF)
    print("->", LOGS, os.path.join(SRC, "summary.png"))


if __name__ == "__main__":
    main()
