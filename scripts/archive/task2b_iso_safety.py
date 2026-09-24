"""Wave quality re-plotted against MEASURED ACOUSTIC OUTPUT instead of transmit voltage.

The 2026-08-07 parameter sweep compared apertures **at matched TX voltage**, where a bigger
aperture is unambiguously better. But the S5-1 push is `I_sppa.3`-limited, and a bigger aperture
reaches that limit at a *lower* voltage. With all three apertures now measured directly
(`SWI/Mechanical index/hydrophone_analysis/SafetyTableAll.m`, 2026-08-17 no-preamp sweep) the
limiting transmit voltages are **41 el 32.0 V, 61 el 23.1 V, 79 el 20.1 V**, all I_sppa.3-limited
(MI only reaches 1.5-1.6 there).

The operational question is therefore not "which aperture is best at 30 V" but **"which aperture
is best at the same, legal, acoustic output"**. This script answers it by plotting the same sweep
against MI and I_sppa.3 rather than against volts, and by interpolating each aperture's quality to
exactly I_sppa.3 = 190 W/cm^2.

    python scripts/archive/task2b_iso_safety.py --csv <scores_velo_phantom.csv> --out <fig.png>
"""
from __future__ import annotations
import sys
import os
_HERE = os.path.dirname(os.path.abspath(__file__))           # archived: siblings + scripts/
for _p in (_HERE, os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LIM_MI, LIM_ISPPA = 1.9, 190.0
MAXV = {41: 32.0, 61: 23.1, 79: 20.1}          # measured I_sppa.3-limited maxima
COLORS = {41: "tab:blue", 61: "tab:orange", 79: "tab:green"}
STYLE = {1500: ("--", "o"), 1900: ("-", "s")}


def read(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({k: (int(float(v)) if k in ("elements", "cycles") else float(v))
                         for k, v in r.items()})
    return rows


def at_limit(rows, cyc, el, metric, xkey="Isppa3_W_cm2", target=LIM_ISPPA):
    """Interpolate `metric` to the operating point where `xkey` equals `target`."""
    sel = sorted((r for r in rows if r["cycles"] == cyc and r["elements"] == el),
                 key=lambda r: r[xkey])
    x = np.array([r[xkey] for r in sel])
    y = np.array([r[metric] for r in sel])
    keep = np.concatenate([[True], np.diff(x) > 0])     # the 79-el curve folds back above 40 V
    return float(np.interp(target, x[keep], y[keep]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--metric", default="sym_median")
    a = ap.parse_args()
    rows = read(a.csv)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for xkey, xlabel, lim, ax in (("V", "commanded push TX voltage [V]", None, axes[0]),
                                  ("MI", "MI (derated, measured)", LIM_MI, axes[1]),
                                  ("Isppa3_W_cm2", "I_sppa.3 [W/cm2] (measured)", LIM_ISPPA,
                                   axes[2])):
        for cyc in (1500, 1900):
            for el in (41, 61, 79):
                sel = sorted((r for r in rows if r["cycles"] == cyc and r["elements"] == el),
                             key=lambda r: r[xkey])
                if not sel:
                    continue
                ls, mk = STYLE[cyc]
                ax.plot([r[xkey] for r in sel], [r[a.metric] for r in sel], ls, marker=mk,
                        color=COLORS[el], ms=5, lw=1.6, label=f"{el} el, {cyc} cyc")
        if lim is not None:
            ax.axvline(lim, color="firebrick", lw=1.5, ls=":")
            ax.text(lim, 0.02, "  FDA limit", color="firebrick", fontsize=8, rotation=90,
                    va="bottom")
        else:
            for el, v in MAXV.items():
                ax.axvline(v, color=COLORS[el], lw=1.1, ls=":", alpha=0.8)
                ax.text(v, 1.0, f" {v:.0f} V", color=COLORS[el], fontsize=7.5, rotation=90,
                        va="top")
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("mirror symmetry (median of 10 pushes)")
    axes[0].set_title("vs transmit voltage\n(what the 2026-08-07 sweep compared;\ndotted = each "
                      "aperture's measured legal maximum)", fontsize=9.5)
    axes[1].set_title("vs MI\nMI is NOT the binding limit\n(only 1.5-1.6 at each legal maximum)",
                      fontsize=9.5)
    axes[2].set_title("vs I_sppa.3 -- the binding limit\niso-safety: at equal output,\nFEWER "
                      "elements is better", fontsize=9.5)
    axes[2].legend(fontsize=8, loc="lower right")
    fig.suptitle("Phantom shear-wave quality vs transmit voltage and vs delivered acoustic output "
                 "(2026-08-17 sweep)\nAt equal voltage a bigger aperture wins; at equal (legal) "
                 "acoustic output a SMALLER aperture wins.\n"
                 "MI and I_sppa.3 are MEASURED for all three apertures (2026-08-17 direct "
                 "no-preamp hydrophone sweep, SafetyTableAll.m).", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(a.out, dpi=130)
    print("wrote", a.out)

    print(f"\nQuality interpolated to exactly I_sppa.3 = {LIM_ISPPA:.0f} W/cm2 "
          f"(each aperture at its own legal maximum voltage):")
    print(f"  {'cyc':>5} {'el':>4} {'V_max':>6} {'MI there':>9} {'sym':>6} {'oc':>6}")
    for cyc in (1500, 1900):
        for el in (41, 61, 79):
            sym = at_limit(rows, cyc, el, "sym_median")
            oc = at_limit(rows, cyc, el, "oc_median")
            mi = at_limit(rows, cyc, el, "MI")
            print(f"  {cyc:5d} {el:4d} {MAXV[el]:6.1f} {mi:9.2f} {sym:6.2f} {oc:6.2f}")


if __name__ == "__main__":
    main()
