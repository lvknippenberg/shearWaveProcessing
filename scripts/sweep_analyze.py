"""Analyze the extraction sweep: per-voltage (SNR) top contenders + how the optimal levers shift with SNR.

Reads sweep_results.csv (from sweep_extract.py). For each voltage it:
  - ranks recipes by the ROI-contrast detector (band1, the drawn V-template) and reports each contender's
    band2 (2x wide) and mirror-symmetry scores, flagging recipes that are top on BOTH detectors;
  - summarizes which levers the top contenders favor (spatial method + sigma, band, quantity, offsets).
Then it plots those winning-lever trends against voltage, so the SNR-dependence is visible at a glance.

    <zea-python> scripts/sweep_analyze.py [--k 12] [--csv sweep_results.csv]
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics as st
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = r"D:/Luuk van Knippenberg/Claude/2026_08_04 voltage sweep/metric_experiment"
VORDER = ["50V", "45V", "40V", "35V", "30V", "25V", "20V", "15V"]
FLOOR = {"roi1_best": 0.05, "roi2_best": 0.05, "sym_best": 0.10}   # empirical no-wave floors


def load(csv_path):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    for r in rows:
        for k in ("roi1_best", "roi2_best", "sym_best", "roi1_disp", "roi1_vel", "roi1_acc"):
            r[k] = float(r[k])
        for k in ("sz_um", "f_lo", "f_hi", "offsets", "recipe_id"):
            r[k] = float(r[k])
    by_v = {v: [r for r in rows if r["voltage"] == v] for v in VORDER}
    return {v: rr for v, rr in by_v.items() if rr}


def mode(xs):
    return Counter(xs).most_common(1)[0][0] if xs else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--csv", default=os.path.join(BASE, "sweep_results.csv"))
    a = ap.parse_args()
    by_v = load(a.csv)
    volts = [v for v in VORDER if v in by_v]

    lines, trend = [], {}
    for v in volts:
        rr = by_v[v]
        top1 = sorted(rr, key=lambda r: r["roi1_best"], reverse=True)[:a.k]
        top_sym = set(r["recipe_id"] for r in sorted(rr, key=lambda r: r["sym_best"], reverse=True)[:a.k])
        lines.append(f"\n=== {v}  (n={len(rr)})   ROI-contrast leaderboard (top {a.k}) ===")
        lines.append(f"  {'rid':>4} {'roi1':>6} {'roi2':>6} {'sym':>6}  {'q':>4} {'spatial':>8} "
                     f"{'sig_um':>7} {'band':>10} {'off':>4}  both?")
        for r in top1:
            both = "**" if r["recipe_id"] in top_sym else ""
            lines.append(f"  {int(r['recipe_id']):>4} {r['roi1_best']:>6.3f} {r['roi2_best']:>6.3f} "
                         f"{r['sym_best']:>6.3f}  {r['roi1_q']:>4} {r['sm']:>8} {int(r['sz_um']):>7} "
                         f"{int(r['f_lo'])}-{int(r['f_hi']):<6} {int(r['offsets']):>4}  {both}")
        sigs = [r["sz_um"] for r in top1 if r["sz_um"] > 0]
        trend[v] = {
            "roi1_top": top1[0]["roi1_best"], "sym_top": max(r["sym_best"] for r in rr),
            "above_floor": top1[0]["roi1_best"] > FLOOR["roi1_best"],
            "sm": mode([r["sm"] for r in top1]), "sig": st.median(sigs) if sigs else 0.0,
            "f_lo": st.median([r["f_lo"] for r in top1]), "f_hi": st.median([r["f_hi"] for r in top1]),
            "q": mode([r["roi1_q"] for r in top1]), "off": st.median([r["offsets"] for r in top1]),
            "iq": mode([r["iq"] for r in top1]),
        }

    lines.append("\n\n=== HOW THE OPTIMAL LEVERS SHIFT WITH SNR (top-%d medians) ===" % a.k)
    lines.append(f"  {'V':>4} {'roi1*':>6} {'sym*':>6} {'>flr':>4}  {'spatial':>8} {'sig_um':>7} "
                 f"{'band':>10} {'quant':>6} {'off':>4} {'iq':>6}")
    for v in volts:
        t = trend[v]
        lines.append(f"  {v:>4} {t['roi1_top']:>6.3f} {t['sym_top']:>6.3f} {'yes' if t['above_floor'] else 'NO':>4}"
                     f"  {t['sm']:>8} {int(t['sig']):>7} {int(t['f_lo'])}-{int(t['f_hi']):<6} "
                     f"{t['q']:>6} {int(t['off']):>4} {t['iq']:>6}")
    report = "\n".join(lines)
    print(report)
    open(os.path.join(BASE, "sweep_leaderboards.txt"), "w", encoding="utf-8").write(report)

    # ---- trend figure ----
    xs = list(range(len(volts)))
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].plot(xs, [trend[v]["roi1_top"] for v in volts], "o-", label="ROI-contrast (band1)")
    ax[0, 0].plot(xs, [trend[v]["sym_top"] for v in volts], "s-", label="mirror-symmetry")
    ax[0, 0].axhline(FLOOR["roi1_best"], color="r", ls=":", lw=1, label="ROI no-wave floor")
    ax[0, 0].set_title("best detector score vs voltage"); ax[0, 0].legend(fontsize=8)
    ax[0, 0].set_ylabel("score")
    ax[0, 1].plot(xs, [trend[v]["sig"] for v in volts], "o-", color="tab:green")
    ax[0, 1].set_title("median spatial sigma of top contenders [um]"); ax[0, 1].set_ylabel("um")
    ax[1, 0].plot(xs, [trend[v]["f_lo"] for v in volts], "o-", label="f_lo")
    ax[1, 0].plot(xs, [trend[v]["f_hi"] for v in volts], "s-", label="f_hi")
    ax[1, 0].set_title("median band corners of top contenders [Hz]"); ax[1, 0].legend(fontsize=8)
    ax[1, 0].set_ylabel("Hz")
    qmap = {"dis": 0, "vel": 1, "acc": 2}
    ax[1, 1].plot(xs, [qmap.get(trend[v]["q"], 0) for v in volts], "o-", color="tab:purple")
    ax[1, 1].set_yticks([0, 1, 2]); ax[1, 1].set_yticklabels(["displacement", "velocity", "acceleration"])
    ax[1, 1].set_title("dominant best quantity of top contenders")
    for axi in ax.ravel():
        axi.set_xticks(xs); axi.set_xticklabels(volts); axi.grid(alpha=0.3)
    fig.suptitle("Extraction sweep: optimal levers vs transmit voltage (SNR)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(BASE, "sweep_snr_trends.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"\nwrote {out}\nwrote {os.path.join(BASE, 'sweep_leaderboards.txt')}")


if __name__ == "__main__":
    main()
