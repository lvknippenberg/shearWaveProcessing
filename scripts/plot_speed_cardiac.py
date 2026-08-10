"""Plot manually-measured shear-wave speed over the cardiac cycle.

Reads the per-push JSON records written by the GUI's manual-speed **save** button
(``<dataset folder>/speed_measurements/speed_meas*.json``) and plots speed vs cardiac time,
assuming the pushes are acquired at a fixed rate (default 20 pushes/s, so push m is at m/fps seconds
after the first push / R-peak).

    python scripts/plot_speed_cardiac.py --folder "<dataset folder>" [--fps 20] [--out speed_cardiac.png]

If ``--folder`` is omitted it defaults to the 40 V in-vivo folder.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_FOLDER = (r"D:/Luuk van Knippenberg/Claude/2026_08_04 voltage sweep/Invivo/"
                  r"Luuk40V_SW_data_04-August-2026_13-36-07")


def load_records(folder):
    """Return sorted list of per-push dicts: {meas, fps, quantity, speeds:[...], sides:[...]}."""
    recs = []
    for p in sorted(glob.glob(os.path.join(folder, "speed_measurements", "speed_meas*.json"))):
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        speeds = [ln["speed_mps"] for ln in d.get("lines", []) if ln.get("speed_mps") is not None]
        sides = [ln.get("side", "?") for ln in d.get("lines", []) if ln.get("speed_mps") is not None]
        if speeds:
            recs.append(dict(meas=int(d["meas"]), fps=float(d.get("fps", 20)),
                             quantity=d.get("quantity", "?"), speeds=speeds, sides=sides))
    return sorted(recs, key=lambda r: r["meas"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=DEFAULT_FOLDER)
    ap.add_argument("--fps", type=float, default=None, help="pushes/s (default: from the JSON, else 20)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    recs = load_records(a.folder)
    if not recs:
        raise SystemExit(f"no speed_measurements/speed_meas*.json found under {a.folder}")
    fps = a.fps or recs[0]["fps"] or 20.0
    out = a.out or os.path.join(a.folder, "speed_measurements", "speed_over_cardiac_cycle.png")

    fig, ax = plt.subplots(figsize=(9, 4.6))
    all_m, all_mean = [], []
    for r in recs:
        t_ms = r["meas"] / fps * 1e3                      # cardiac time of this push
        for s, side in zip(r["speeds"], r["sides"]):
            ax.plot(t_ms, s, "o", ms=6, color=("#1f77b4" if side == "R" else "#d62728"),
                    alpha=0.85, zorder=3)
        m = float(np.mean(r["speeds"]))
        all_m.append(t_ms); all_mean.append(m)
        if len(r["speeds"]) > 1:                           # error bar = spread of lines on this push
            ax.plot([t_ms, t_ms], [min(r["speeds"]), max(r["speeds"])], color="0.6", lw=1, zorder=2)
    ax.plot(all_m, all_mean, "-", color="0.25", lw=1.6, zorder=1, label="per-push mean")

    # annotate push index on a secondary axis
    ax.set_xlabel(f"cardiac time [ms]  (push index at {fps:.0f} pushes/s)")
    ax.set_ylabel("shear-wave speed [m/s]")
    secx = ax.secondary_xaxis("top", functions=(lambda t: t * fps / 1e3, lambda m: m / fps * 1e3))
    secx.set_xlabel("push index (m)")
    ax.grid(True, alpha=0.3)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color="#1f77b4", label="right of r0"),
                       Line2D([], [], marker="o", ls="", color="#d62728", label="left of r0"),
                       Line2D([], [], color="0.25", label="per-push mean")], fontsize=9, loc="best")
    q = ", ".join(sorted({r["quantity"] for r in recs}))
    ax.set_title(f"Manually measured shear-wave speed over the cardiac cycle\n"
                 f"{os.path.basename(a.folder)}  ·  {len(recs)} pushes  ·  quantity: {q}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    print(f"  {len(recs)} pushes, fps={fps:.0f}; speed range "
          f"{min(all_mean):.2f}–{max(all_mean):.2f} m/s (per-push means)")


if __name__ == "__main__":
    main()
