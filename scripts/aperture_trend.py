"""Does "fewer push elements is better at equal acoustic output" continue below 41 elements?

The 2026-08-17 sweep covers 41 / 61 / 79 elements, all with *measured* acoustic output
(SafetyTableAll, 2026-08-17 no-preamp captures). The 2026-08-07 one-at-a-time sweep adds
**21 elements** (20 and 30 V, 1500 cycles) on the same phantom and geometry, so the trend can be
extended downwards without new data.

Acoustic output for 21 elements is not measured. It is scaled from the *measured* focal-gain
exponent: fitting p_r ~ N^q to the 41/61/79-element MI at a low, unsaturated transmit voltage
gives q ~ 0.84, so I_sppa ~ N^1.68, and
    I_sppa(N,V) = I_sppa(41,V) * (N/41)^(2q).
The same kind of scaling was used for 41 elements before it was measured, and the direct
measurement later reproduced it to ~8 %, so it is good enough to place a point on the axis.

Two constraints bound the useful aperture from both sides:
  * from above -- I_sppa.3 <= 190 W/cm^2 forces the transmit voltage down as N grows;
  * from below -- the transmit hardware stops at 50 V, so a small enough aperture can no longer
    reach the limit at all. The crossover (the smallest aperture that still reaches
    I_sppa.3 = 190 at 50 V) is computed and printed.

    python scripts/aperture_trend.py --outdir <dir>
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import swe_lib as L                                              # noqa: E402

AUG07 = r"D:/swp_v04/Phantom parameter sweep"
V_MEAS = np.array([15, 20, 25, 30, 35, 40, 45, 50], float)
MI_MEAS = {41: [0.71, 0.93, 1.15, 1.37, 1.64, 1.94, 2.25, 2.25],
           61: [1.03, 1.33, 1.56, 2.15, 2.66, 2.86, 3.19, 3.59],
           79: [1.25, 1.61, 2.08, 2.65, 2.86, 2.86, 2.25, 2.59]}
IS_MEAS = {41: [38.3, 68.0, 107.9, 163.0, 229.8, 290.6, 347.5, 388.6],
           61: [78.9, 142.9, 219.5, 326.6, 470.6, 586.3, 530.8, 690.2],
           79: [109.6, 188.7, 298.9, 436.0, 480.9, 488.7, 175.6, 21.5]}
MAXV = {41: 32.0, 61: 23.1, 79: 20.1}          # I_sppa.3-limited, measured
LIM_ISPPA = 190.0
V_HARDWARE = 50.0
COL = {21: "tab:red", 41: "tab:blue", 61: "tab:orange", 79: "tab:green"}


def focal_gain_exponent(v_ref=20.0):
    i = int(np.where(V_MEAS == v_ref)[0][0])
    n = np.array([41, 61, 79], float)
    mi = np.array([MI_MEAS[int(k)][i] for k in n])
    q, _ = np.polyfit(np.log(n), np.log(mi), 1)
    return float(q)


def isppa_of(n_el, v, q):
    """Measured where available, otherwise scaled from the measured 41-element curve."""
    if n_el in IS_MEAS:
        return float(np.interp(v, V_MEAS, IS_MEAS[n_el]))
    return float(np.interp(v, V_MEAS, IS_MEAS[41]) * (n_el / 41.0) ** (2 * q))


def legal_max_voltage(n_el, q):
    """Highest transmit voltage that keeps I_sppa.3 <= 190, capped by the 50 V hardware ceiling.

    The *first* upward crossing is used, not the largest voltage that happens to sit under the
    limit: the measured 79-element curve is non-monotonic above 40 V (the capture collapses --
    alignment or supply), and taking the maximum would read that artefact as headroom.
    """
    v = np.arange(10, V_HARDWARE + 0.01, 0.05)
    for vv in v:
        if isppa_of(n_el, vv, q) > LIM_ISPPA:
            return float(vv - 0.05), False
    return V_HARDWARE, True


def score_folder_set(root, cycles, pri=270, quantity="velocity"):
    out = {}
    for folder in L.measurement_folders(root):
        p = L.acq_params(folder)
        if p["cyc"] != cycles or int(p.get("PRI_us", 270)) != pri:
            continue
        sc = []
        for k in range(L.n_pushes(folder)):
            try:
                st, r0 = L.st_for(folder, k, L.REC_PHANTOM, quantity=quantity, phantom=True)
            except Exception as exc:                    # noqa: BLE001
                print(f"  {p['el']}el {p['V']:.0f}V push{k}: {exc}")
                continue
            sc.append(L.scores(st, r0))
            L._ACQ_CACHE.pop((folder, k, True), None)
        if sc:
            sc = np.array(sc)
            out[(p["el"], int(round(p["V"])))] = (float(np.median(sc[:, 0])),
                                                  float(np.median(sc[:, 1])))
            print(f"  {p['el']:2d} el  {p['V']:.0f} V  oc={np.median(sc[:,0]):.2f} "
                  f"sym={np.median(sc[:,1]):.2f}  (n={len(sc)})", flush=True)
    return out


def read_aug17(csv_path, cycles=1500):
    out = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if int(r["cycles"]) == cycles:
                out[(int(r["elements"]), int(float(r["V"])))] = (float(r["oc_median"]),
                                                                 float(r["sym_median"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--aug17-csv", default=r"D:/swp_ph17/analysis/scores_velo_phantom.csv")
    ap.add_argument("--quantity", default="velocity")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    q = focal_gain_exponent()
    n_opt = 41.0 * (LIM_ISPPA / isppa_of(41, V_HARDWARE, q)) ** (1 / (2 * q))
    print(f"measured focal-gain exponent: p_r ~ N^{q:.3f}   =>   I_sppa ~ N^{2*q:.2f}")
    print(f"smallest aperture that still reaches I_sppa.3 = {LIM_ISPPA:.0f} at the "
          f"{V_HARDWARE:.0f} V hardware ceiling: N = {n_opt:.0f} elements\n")

    print("2026-08-07 sweep (21/41/61/79 el, 1500 cycles, 270 us PRI):")
    aug07 = score_folder_set(AUG07, 1500, quantity=a.quantity)
    aug17 = read_aug17(a.aug17_csv, cycles=1500) if os.path.isfile(a.aug17_csv) else {}
    print(f"2026-08-17 sweep: {len(aug17)} cells read from {os.path.basename(a.aug17_csv)}")

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))

    # ---- panel 1: quality vs aperture at matched voltage (2026-08-07, includes 21 el)
    ax = axes[0]
    els07 = sorted({k[0] for k in aug07})
    for v, style in ((20, "--o"), (30, "-s")):
        x = [e for e in els07 if (e, v) in aug07]
        ax.plot(x, [aug07[(e, v)][1] for e in x], style, color="0.25", ms=7,
                label=f"{v} V", mfc="w" if v == 20 else "0.25")
    ax.set_xlabel("push elements")
    ax.set_ylabel("mirror symmetry (median of 10 pushes)")
    ax.set_title("matched TRANSMIT VOLTAGE (2026-08-07)\nmore elements is better, "
                 "monotonically", fontsize=10)
    ax.set_xticks(els07)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.02)

    # ---- panel 2: everything against acoustic output
    ax = axes[1]
    for src, mk, ls, lbl in ((aug17, "o", "-", "2026-08-17"), (aug07, "^", "--", "2026-08-07")):
        for e in sorted({k[0] for k in src}):
            pts = sorted((isppa_of(e, v, q), src[(e, v)][1]) for (ee, v) in src if ee == e)
            if not pts:
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], mk, ls=ls, color=COL.get(e, "0.5"),
                    ms=6, lw=1.5, alpha=0.95 if src is aug17 else 0.55,
                    label=f"{e} el ({lbl})")
    ax.axvline(LIM_ISPPA, color="firebrick", ls=":", lw=1.6)
    ax.text(LIM_ISPPA * 1.03, 0.04, "FDA I_sppa.3 limit", color="firebrick", fontsize=8,
            rotation=90)
    for e, vmax in MAXV.items():
        ax.plot(isppa_of(e, vmax, q), np.nan, "")            # keep colours aligned
    ax.set_xscale("log")
    ax.set_xlabel("I_sppa.3 [W/cm2]  (measured for 41/61/79, scaled for 21)")
    ax.set_title("matched ACOUSTIC OUTPUT\nthe curves do NOT collapse: at equal output,\n"
                 "41 el beats 61 and 79", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.set_ylim(0, 1.02)

    # ---- panel 3: what each aperture can legally reach
    ax = axes[2]
    reach = []
    for e in (21, 31, 41, 51, 61, 79):
        vmax, hw_capped = legal_max_voltage(e, q)
        out_at_max = isppa_of(e, vmax, q)
        src = aug17 if e in {41, 61, 79} else aug07
        pts = sorted((isppa_of(e, v, q), src[(e, v)][1]) for (ee, v) in src if ee == e)
        if len(pts) >= 2:
            xs, ys = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
            sym = float(np.interp(out_at_max, xs, ys)) if out_at_max <= xs[-1] else \
                float(ys[-1] + (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]) * (out_at_max - xs[-1]))
            sym = float(np.clip(sym, 0, 1))
        else:
            sym = np.nan
        reach.append((e, vmax, out_at_max, sym, hw_capped))
        print(f"  {e:2d} el: legal max {vmax:4.0f} V"
              f"{' (HARDWARE-capped)' if hw_capped else ' (I_sppa-capped)'}"
              f" -> I_sppa.3 {out_at_max:5.0f} W/cm2, mirror symmetry {sym:.2f}"
              + ("  [EXTRAPOLATED beyond the measured range]"
                 if (not np.isnan(sym) and len(pts) >= 2 and out_at_max > pts[-1][0]) else ""))
    have = [r for r in reach if not np.isnan(r[3])]
    ax.plot([r[0] for r in have], [r[3] for r in have], "-D", color="tab:purple", ms=7, lw=1.6)
    for e, vmax, o, s, hw in have:
        ax.annotate(f"{vmax:.0f} V\n{o:.0f} W/cm$^2$", (e, s), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=7.5,
                    color="firebrick" if hw else "0.25")
    ax.axvline(n_opt, color="tab:purple", ls=":", lw=1.4)
    ax.text(n_opt + 1, 0.06, f"$N\\approx${n_opt:.0f}: smallest aperture that\nstill reaches the "
            "limit at 50 V", fontsize=7.5, color="tab:purple")
    ax.set_xlabel("push elements")
    ax.set_title("best each aperture can LEGALLY reach\n(red labels: capped by the 50 V "
                 "transmit ceiling,\nnot by safety)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.02)

    fig.suptitle("Does 'fewer elements is better' continue below 41 elements? "
                 f"Phantom, velocity, 1500-cycle push; I_sppa scaled as N^{2*q:.2f} where not "
                 "measured", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(a.outdir, "aperture_trend.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)

    with open(os.path.join(a.outdir, "aperture_trend.csv"), "w", encoding="utf-8",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["campaign", "elements", "V", "oc_median", "sym_median", "Isppa3",
                    "Isppa3_source"])
        for lbl, src in (("2026-08-07", aug07), ("2026-08-17", aug17)):
            for (e, v), (oc, sym) in sorted(src.items()):
                w.writerow([lbl, e, v, f"{oc:.3f}", f"{sym:.3f}", f"{isppa_of(e, v, q):.0f}",
                            "measured" if e in IS_MEAS else f"scaled N^{2*q:.2f}"])
    print("wrote aperture_trend.csv")


if __name__ == "__main__":
    main()
