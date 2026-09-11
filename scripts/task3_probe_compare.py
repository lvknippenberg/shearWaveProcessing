"""Task 3 -- TU/e vs MUMC S5-1 probe comparison on the same phantom (2026-08-18).

Each probe was swept over the same 3 apertures x 2 pulse lengths x 5 TX voltages as the
2026-08-17 sweep, on the same CIRS phantom, plus one B-mode acquisition on a resolution phantom.
This script scores every (elements, cycles, voltage) cell for both probes over all 10 pushes and
reports:

  * `probe_compare_summary.png`  quality vs voltage, both probes, per aperture / pulse length;
  * `probe_compare_<cyc>c.png`   paired space-time grids (TU/e above MUMC) per configuration;
  * `probe_compare_bmode.png`    the resolution-phantom B-mode of each probe side by side;
  * `probe_compare.csv`          the numbers.

Duplicate cells: each probe folder contains the resolution-phantom acquisition (which shares the
41 el / 1500 cyc / 20 V settings) plus a few deliberate repeats. The resolution acquisition is
excluded by name and the *first* sweep occurrence of each cell is used.

    python scripts/task3_probe_compare.py --tue D:/swp_tue --mumc D:/swp_mumc --outdir <dir>
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import swe_lib as L                                              # noqa: E402
from task2_element_cycle_grid import TX, EL, CYC, safety         # noqa: E402

# The resolution-phantom B-mode acquisition of each probe (first folder of each; readme order).
BMODE_FOLDERS = {"TU/e": "17-53-11", "MUMC": "17-50-03"}


def index_probe(root, bmode_tag):
    """(el, cyc, V) -> folder, skipping the resolution-phantom acquisition, first sweep hit wins."""
    idx, bmode = {}, None
    for folder in L.measurement_folders(root):
        if folder.endswith(bmode_tag):
            bmode = folder
            continue
        p = L.acq_params(folder)
        idx.setdefault((p["el"], p["cyc"], int(round(p["V"]))), folder)
    return idx, bmode


def score_probe(idx, recipe, quantity):
    out = {}
    for cyc in CYC:
        for el in EL:
            for V in TX:
                folder = idx.get((el, cyc, V))
                if folder is None or L.n_pushes(folder) == 0:
                    print(f"    MISSING {el}el {cyc}c {V}V")
                    continue
                sts, sc = [], []
                for p in range(L.n_pushes(folder)):
                    try:
                        st, r0 = L.st_for(folder, p, recipe, quantity=quantity, phantom=True)
                    except Exception as exc:                      # noqa: BLE001
                        print(f"    {el}el {cyc}c {V}V push{p}: {exc}")
                        continue
                    sc.append(L.scores(st, r0))
                    sts.append((st.data, st.r, st.t, r0))
                    L._ACQ_CACHE.pop((folder, p, True), None)
                if not sts:
                    continue
                sc = np.array(sc)
                pick = int(np.argsort(sc[:, 1])[len(sc) // 2])
                out[(cyc, el, V)] = dict(scores=sc, panel=sts[pick], pick=pick, n=len(sts))
                print(f"    {el:2d}el {cyc}c {V:2d}V  oc={np.median(sc[:,0]):.2f} "
                      f"sym={np.median(sc[:,1]):.2f}", flush=True)
    return out


def summary(res, outpath, quantity):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True, sharex=True)
    probe_style = {"TU/e": ("-o", "tab:blue"), "MUMC": ("--s", "tab:red")}
    for i, cyc in enumerate(CYC):
        for j, el in enumerate(EL):
            ax = axes[i][j]
            for probe, r in res.items():
                st, c = probe_style[probe]
                v = [V for V in TX if (cyc, el, V) in r]
                if not v:
                    continue
                y = [np.median(r[(cyc, el, V)]["scores"][:, 1]) for V in v]
                lo = [np.percentile(r[(cyc, el, V)]["scores"][:, 1], 25) for V in v]
                hi = [np.percentile(r[(cyc, el, V)]["scores"][:, 1], 75) for V in v]
                ax.plot(v, y, st, color=c, ms=5, label=probe)
                ax.fill_between(v, lo, hi, color=c, alpha=0.12)
            ax.set_title(f"{el} el, {cyc} cyc", fontsize=10)
            ax.grid(alpha=0.3)
            ax.set_ylim(0, 1.02)
            if i == 1:
                ax.set_xlabel("push TX voltage [V]")
            if j == 0:
                ax.set_ylabel("mirror symmetry\n(median, IQR over 10 pushes)")
    axes[0][0].legend(fontsize=9)
    fig.suptitle(f"TU/e vs MUMC S5-1 on the same phantom - shear-wave mirror symmetry "
                 f"({quantity}, phantom recipe)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    print("wrote", outpath)


def paired_grid(res, cyc, outpath, quantity):
    """Rows alternate TU/e and MUMC for each voltage; columns are the three apertures."""
    probes = list(res)
    nrow = len(TX) * len(probes)
    fig, axes = plt.subplots(nrow, len(EL), figsize=(12, 2.9 * nrow), squeeze=False)
    unit = 1e3 if quantity == "velocity" else 1e6
    lims = [np.percentile(np.abs(r[(cyc, el, V)]["panel"][0]), 99)
            for r in res.values() for el in EL for V in TX if (cyc, el, V) in r]
    clim = float(np.median(lims)) * unit
    for i, V in enumerate(TX):
        for pi, probe in enumerate(probes):
            for j, el in enumerate(EL):
                ax = axes[i * len(probes) + pi][j]
                key = (cyc, el, V)
                r = res[probe].get(key)
                if r is None:
                    ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                            transform=ax.transAxes)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue
                data, rr, tt, r0 = r["panel"]
                ax.imshow(data * unit,
                          extent=(rr[0] * 1e3, rr[-1] * 1e3, tt[-1] * 1e3, tt[0] * 1e3),
                          cmap="RdBu_r", vmin=-clim, vmax=clim, aspect="auto", origin="upper")
                ax.axvline(r0 * 1e3, color="0.2", ls="--", lw=0.8, alpha=0.7)
                ax.tick_params(labelsize=6)
                oc, sym = np.median(r["scores"], axis=0)
                ax.set_title(f"{probe}   {el} el  {V} V    oc {oc:.2f}  sym {sym:.2f}", fontsize=8)
                if j == 0:
                    ax.set_ylabel(f"{V} V - {probe}\nt [ms]", fontsize=8)
                if i == len(TX) - 1 and pi == len(probes) - 1:
                    ax.set_xlabel("r [mm]", fontsize=9)
    fig.suptitle(f"TU/e vs MUMC S5-1, same phantom - {cyc} push cycles ({cyc / 2.25:.0f} us)\n"
                 f"{quantity}, phantom recipe, shared colour scale; each voltage shows TU/e above "
                 f"MUMC", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(outpath, dpi=100)
    plt.close(fig)
    print("wrote", outpath)


def bmode_compare(bmodes, outpath):
    import core
    fig, axes = plt.subplots(1, len(bmodes), figsize=(6.2 * len(bmodes), 7))
    for ax, (probe, folder) in zip(np.atleast_1d(axes), bmodes.items()):
        if folder is None:
            ax.axis("off")
            continue
        img, ext = core._bmode_h5py(os.path.join(folder, "output",
                                                 "CombinedData_buffer3_iq.hdf5"), 0)
        ax.imshow(img, extent=ext, cmap="gray", aspect="equal")
        ax.set_xlim(-45, 45)
        ax.set_ylim(120, 0)
        ax.set_title(f"{probe} S5-1 - focused B-mode, resolution phantom", fontsize=11)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("z [mm]")
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    print("wrote", outpath)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tue", required=True)
    ap.add_argument("--mumc", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--quantity", default="velocity")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    rec = L.REC_PHANTOM

    res, bmodes = {}, {}
    for probe, root in (("TU/e", a.tue), ("MUMC", a.mumc)):
        print(f"=== {probe}", flush=True)
        idx, bm = index_probe(root, BMODE_FOLDERS[probe])
        bmodes[probe] = bm
        res[probe] = score_probe(idx, rec, a.quantity)

    summary(res, os.path.join(a.outdir, "probe_compare_summary.png"), a.quantity)
    for cyc in CYC:
        if any(k[0] == cyc for r in res.values() for k in r):
            paired_grid(res, cyc, os.path.join(a.outdir, f"probe_compare_{cyc}c.png"), a.quantity)
    try:
        bmode_compare(bmodes, os.path.join(a.outdir, "probe_compare_bmode.png"))
    except Exception as exc:                                      # noqa: BLE001
        print(f"B-mode comparison skipped: {exc}")

    with open(os.path.join(a.outdir, "probe_compare.csv"), "w", encoding="utf-8") as f:
        f.write("probe,cycles,elements,V,n_pushes,oc_median,sym_median,MI,Isppa3,Ispta3\n")
        for probe, r in res.items():
            for (cyc, el, V), d in sorted(r.items()):
                mi, isppa, ispta = safety(el, V, cyc)
                f.write(f"{probe},{cyc},{el},{V},{d['n']},{np.median(d['scores'][:,0]):.3f},"
                        f"{np.median(d['scores'][:,1]):.3f},{mi:.2f},{isppa:.0f},{ispta:.0f}\n")
    print("wrote probe_compare.csv")


if __name__ == "__main__":
    main()
