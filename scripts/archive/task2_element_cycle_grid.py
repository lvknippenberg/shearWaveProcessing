"""Task 2 -- 5x3 space-time grids for the 2026-08-17 phantom sweep (TX voltage x elements),
one figure per push pulse length (1500 / 1900 cycles).

rows = commanded push TX voltage 20/25/30/35/40 V,  cols = 41/61/79 push elements.
Each cell shows the space-time of the *median-quality* push of that measurement's 10 pushes,
with the median (IQR) origin-coherence and mirror-symmetry over all 10 in the title, plus the
acoustic-output indices for that (elements, voltage, cycles).

    python scripts/archive/task2_element_cycle_grid.py --root D:/swp_ph17 --outdir <dir>
                                               [--quantity velocity|displacement]
"""
from __future__ import annotations

import argparse
import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))           # archived: siblings + scripts/
for _p in (_HERE, os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import swe_lib as L                                              # noqa: E402

TX = [20, 25, 30, 35, 40]
EL = [41, 61, 79]
CYC = [1500, 1900]

# --- acoustic output --------------------------------------------------------------------
# MEASURED derated acoustic output of the S5-1 ARF push, per aperture and transmit voltage
# (2026-08-17 direct no-preamp hydrophone captures, re-analysed by
# `SWI/Mechanical index/hydrophone_analysis/SafetyTableAll.m`). These replace the earlier
# extrapolation from the 61/79-element fit -- the 41-element push has now been measured directly.
# I_spta.3 = I_sppa.3 * PD * PRF_eff, with PD = 0.854 ms at 1900 cycles (scaling with cycles) and
# PRF_eff = 24 pushes / (1.2 s burst + 30 s idle) = 0.769 Hz.
V_MEAS = [15, 20, 25, 30, 35, 40, 45, 50]
MI_MEAS = {41: [0.71, 0.93, 1.15, 1.37, 1.64, 1.94, 2.25, 2.25],
           61: [1.03, 1.33, 1.56, 2.15, 2.66, 2.86, 3.19, 3.59],
           79: [1.25, 1.61, 2.08, 2.65, 2.86, 2.86, 2.25, 2.59]}
ISPPA_MEAS = {41: [38.3, 68.0, 107.9, 163.0, 229.8, 290.6, 347.5, 388.6],
              61: [78.9, 142.9, 219.5, 326.6, 470.6, 586.3, 530.8, 690.2],
              79: [109.6, 188.7, 298.9, 436.0, 480.9, 488.7, 175.6, 21.5]}
# I_sppa.3-limited maximum transmit voltage per aperture (SafetyTableAll)
MAXV = {41: 32.0, 61: 23.1, 79: 20.1}
LIM_MI, LIM_ISPPA, LIM_ISPTA = 1.9, 190.0, 720.0
PD_1900_S = 0.854e-3          # measured push duration at 1900 cycles
PRF_EFF = 24 / (1.2 + 30.0)   # 20 Hz for 1.2 s then >=30 s off -> 0.769 Hz effective


def safety(el, V, cyc):
    mi = float(np.interp(V, V_MEAS, MI_MEAS[el]))
    isppa = float(np.interp(V, V_MEAS, ISPPA_MEAS[el]))
    ispta = isppa * PD_1900_S * (cyc / 1900.0) * PRF_EFF * 1e3     # mW/cm^2
    return mi, isppa, ispta


def build_index(root):
    idx = {}
    for folder in L.measurement_folders(root):
        p = L.acq_params(folder)
        idx[(p["el"], p["cyc"], int(round(p["V"])))] = folder
    return idx


def compute(root, quantity, recipe_name):
    """Per (el, cyc, V): scores for every push + the space-time of the median-quality push."""
    rec = L.REC_PHANTOM if recipe_name == "phantom" else L.REC_INVIVO
    idx = build_index(root)
    out = {}
    for cyc in CYC:
        for el in EL:
            for V in TX:
                folder = idx.get((el, cyc, V))
                if folder is None:
                    print(f"  MISSING {el}el {cyc}c {V}V")
                    continue
                sts, sc = [], []
                for p in range(L.n_pushes(folder)):
                    try:
                        st, r0 = L.st_for(folder, p, rec, quantity=quantity, phantom=True)
                    except Exception as exc:                       # noqa: BLE001
                        print(f"  {el}el {cyc}c {V}V push{p} failed: {exc}")
                        continue
                    oc, sym = L.scores(st, r0)
                    sts.append((st.data, st.r, st.t, r0))
                    sc.append((oc, sym))
                    L._ACQ_CACHE.pop((folder, p, True), None)      # keep memory bounded
                if not sts:
                    continue
                sc = np.array(sc)
                pick = int(np.argsort(sc[:, 1])[len(sc) // 2])     # median mirror-symmetry push
                out[(cyc, el, V)] = dict(scores=sc, pick=pick, panel=sts[pick], n=len(sts))
                print(f"  {el:2d}el {cyc}c {V:2d}V  oc={np.median(sc[:,0]):.2f} "
                      f"sym={np.median(sc[:,1]):.2f}  (n={len(sts)})", flush=True)
    return out


def draw_grid(res, cyc, quantity, outpath, recipe_name):
    fig, axes = plt.subplots(len(TX), len(EL), figsize=(12.0, 16.0), squeeze=False)
    unit = 1e3 if quantity == "velocity" else 1e6
    lims = [np.percentile(np.abs(res[(cyc, el, V)]["panel"][0]), 99)
            for el in EL for V in TX if (cyc, el, V) in res]
    clim = float(np.median(lims)) * unit
    for i, V in enumerate(TX):
        for j, el in enumerate(EL):
            ax = axes[i][j]
            key = (cyc, el, V)
            mi, isppa, ispta = safety(el, V, cyc)
            if key not in res:
                ax.text(0.5, 0.5, "(no data)", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            r = res[key]
            data, rr, tt, r0 = r["panel"]
            ax.imshow(data * unit, extent=(rr[0] * 1e3, rr[-1] * 1e3, tt[-1] * 1e3, tt[0] * 1e3),
                      cmap="RdBu_r", vmin=-clim, vmax=clim, aspect="auto", origin="upper")
            ax.axvline(r0 * 1e3, color="0.2", ls="--", lw=0.8, alpha=0.7)
            ax.tick_params(labelsize=7)
            oc, sym = np.median(r["scores"], axis=0)
            oc_i = np.percentile(r["scores"][:, 0], [25, 75])
            sym_i = np.percentile(r["scores"][:, 1], [25, 75])
            over = mi > LIM_MI or isppa > LIM_ISPPA
            ax.set_title(f"{el} el   {V} V   (push {r['pick']}/{r['n']})\n"
                         f"oc {oc:.2f} [{oc_i[0]:.2f}-{oc_i[1]:.2f}]   "
                         f"sym {sym:.2f} [{sym_i[0]:.2f}-{sym_i[1]:.2f}]\n"
                         f"MI {mi:.2f}  Isppa.3 {isppa:.0f}  Ispta.3 {ispta:.0f}",
                         fontsize=8, color="firebrick" if over else "black")
            if over:
                for s in ax.spines.values():
                    s.set_color("firebrick")
                    s.set_linewidth(2.0)
            if j == 0:
                ax.set_ylabel(f"{V} V\nt [ms]", fontsize=9)
            if i == len(TX) - 1:
                ax.set_xlabel("r [mm]", fontsize=9)
    rec = L.REC_PHANTOM if recipe_name == "phantom" else L.REC_INVIVO
    fig.suptitle(
        f"Phantom ARF shear wave - {cyc} push cycles ({cyc / 2.25:.0f} us)   |   "
        f"rows = TX voltage, cols = push elements\n"
        f"{quantity}, {recipe_name} recipe ({rec['tag']}), outward-directional; "
        f"dashed = push origin r0; shared colour scale\n"
        f"oc = origin coherence, sym = mirror symmetry: median [IQR] over the 10 pushes\n"
        f"MI / I_sppa.3 [W/cm2] / I_spta.3 [mW/cm2] MEASURED (2026-08-17 hydrophone sweep, all "
        f"three apertures);\n"
        f"red = above an FDA limit (MI 1.9, I_sppa.3 190 W/cm2)", fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.925])
    fig.savefig(outpath, dpi=110)
    plt.close(fig)
    print("wrote", outpath)


def draw_summary(res, quantity, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    styles = {1500: "--o", 1900: "-s"}
    colors = {41: "tab:blue", 61: "tab:orange", 79: "tab:green"}
    for ax, metric, mi in zip(axes, ["origin coherence", "mirror symmetry"], [0, 1]):
        for cyc in CYC:
            for el in EL:
                v = [V for V in TX if (cyc, el, V) in res]
                if not v:
                    continue
                y = [np.median(res[(cyc, el, V)]["scores"][:, mi]) for V in v]
                lo = [np.percentile(res[(cyc, el, V)]["scores"][:, mi], 25) for V in v]
                hi = [np.percentile(res[(cyc, el, V)]["scores"][:, mi], 75) for V in v]
                ax.plot(v, y, styles[cyc], color=colors[el], label=f"{el} el, {cyc} cyc", ms=4)
                ax.fill_between(v, lo, hi, color=colors[el], alpha=0.12)
        ax.set_xlabel("commanded push TX voltage [V]")
        ax.set_title(metric)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("score (median, IQR over 10 pushes)")
    axes[1].legend(fontsize=8, ncol=2)
    fig.suptitle("2026-08-17 phantom sweep - wave quality vs voltage, aperture and pulse "
                 f"length ({quantity})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    print("wrote", outpath)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--quantity", default="velocity")
    ap.add_argument("--recipe", default="phantom", choices=["phantom", "invivo"])
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    res = compute(a.root, a.quantity, a.recipe)
    tag = f"{a.quantity[:4]}_{a.recipe}"
    for cyc in CYC:
        if any(k[0] == cyc for k in res):
            draw_grid(res, cyc, a.quantity,
                      os.path.join(a.outdir, f"grid_{cyc}cycles_{tag}.png"), a.recipe)
    draw_summary(res, a.quantity, os.path.join(a.outdir, f"summary_scores_{tag}.png"))

    with open(os.path.join(a.outdir, f"scores_{tag}.csv"), "w", encoding="utf-8") as f:
        f.write("cycles,elements,V,n_pushes,oc_median,oc_q25,oc_q75,sym_median,sym_q25,sym_q75,"
                "MI,Isppa3_W_cm2,Ispta3_mW_cm2\n")
        for (cyc, el, V), r in sorted(res.items()):
            s = r["scores"]
            mi, isppa, ispta = safety(el, V, cyc)
            f.write(f"{cyc},{el},{V},{r['n']},"
                    f"{np.median(s[:,0]):.3f},{np.percentile(s[:,0],25):.3f},"
                    f"{np.percentile(s[:,0],75):.3f},"
                    f"{np.median(s[:,1]):.3f},{np.percentile(s[:,1],25):.3f},"
                    f"{np.percentile(s[:,1],75):.3f},"
                    f"{mi:.2f},{isppa:.0f},{ispta:.0f}\n")
    print("wrote scores CSV")


if __name__ == "__main__":
    main()
