"""Task 1 - reproduce the report's high-quality phantom space-time montage (clear symmetric V).

Rebuilds report/figures/phantom_voltage_montage.png straight from the beamformed buffer-2 IQ
(the stored swp_active HDF5s only carry displacement views, so we recompute), using the report
recipe REC_INVIVO (bp120-700 / gauss / mean3 / 7 offsets / outward), displacement (top row) and
velocity (bottom row), one column per delivered push voltage.

    python scripts/archive/task1_repro_voltage.py --root "<2026_08_04 voltage sweep/Phantom>" --out fig.png
"""
from __future__ import annotations
import sys

import argparse
import os
_HERE = os.path.dirname(os.path.abspath(__file__))           # archived: siblings + scripts/
for _p in (_HERE, os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import swe_lib as L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--meas", type=int, default=0)
    ap.add_argument("--recipe", default="invivo", choices=["invivo", "phantom"])
    a = ap.parse_args()

    rec = L.REC_INVIVO if a.recipe == "invivo" else L.REC_PHANTOM
    folders = L.measurement_folders(a.root)
    info = [L.acq_params(f) for f in folders]
    print(f"{len(folders)} measurements")

    quants = ["displacement", "velocity"]
    panels = {}
    for f, inf in zip(folders, info):
        for q in quants:
            st, r0 = L.st_for(f, a.meas, rec, quantity=q, phantom=True)
            oc, sym = L.scores(st, r0)
            panels[(f, q)] = (st, r0, oc, sym)
            print(f"  {inf['V']:.0f} V {q:12s} oc={oc:.2f} sym={sym:.2f}")

    fig, axes = plt.subplots(len(quants), len(folders),
                             figsize=(2.3 * len(folders), 3.4 * len(quants)), squeeze=False)
    for qi, q in enumerate(quants):
        clim = np.median([L.draw(plt.figure().add_subplot(), *panels[(f, q)][:2]) for f in folders])
        plt.close("all")
        fig2 = fig
        for fi, (f, inf) in enumerate(zip(folders, info)):
            ax = axes[qi][fi]
            st, r0, oc, sym = panels[(f, q)]
            L.draw(ax, st, r0, clim=clim)
            ax.set_title(f"{inf['V']:.0f} V\n{q[:4]}  oc={oc:.2f} sym={sym:.2f}", fontsize=9)
            if fi == 0:
                ax.set_ylabel("t [ms]")
            if qi == len(quants) - 1:
                ax.set_xlabel("r [mm]")
    fig.suptitle(f"Phantom ARF shear wave vs push voltage - recipe {rec['tag']}, push {a.meas} "
                 f"(dashed = push origin r0; shared colour scale per row)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=110)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
